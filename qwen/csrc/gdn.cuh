// Gated DeltaNet decode task: one (value head h, batch row b) per task.
//
// Mirrors Qwen3_5GatedDeltaNet's single-token path in transformers:
//   causal conv1d update (+SiLU) on this head's q/k/v channels,
//   l2-normalised q/k, beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias),
//   S = S*exp(g); delta = (v - S^T k) * beta; S += k delta^T; o = S^T q,
//   gated RMSNorm: bf16(w * bf16(rmsnorm(o))) * silu(z).
// The three value heads sharing a key head each recompute the q/k conv (it is
// 256 channels x 4 taps); only h % 3 == 0 writes that conv state back.
// Conv state is double-buffered by step parity so concurrent readers of the
// old taps never race the writer.
#pragma once

#include "abi.h"
#include "ptx.cuh"

namespace qmk {

struct GdnScratch {
    float q[QMK_LIN_DK];
    float k[QMK_LIN_DK];
    float v[QMK_LIN_DV];
    float red[4][64];          // per-row-quarter partial sums for one 64-column pass
    float o[QMK_LIN_DV];
    float wsum[4];
    float beta, decay, q_scale, k_scale;
};

__device__ void gdn_task(const QmkStepParams& P, const QmkGdnArgs& a, int idx, GdnScratch& sc) {
    // idx = h * bs + b: head-major, so the first half of the heads (out_proj's
    // first K-split) completes early.
    constexpr int NV = QMK_LIN_NV, DK = QMK_LIN_DK, DV = QMK_LIN_DV;
    constexpr int CH = QMK_LIN_CONV_CH, TAPS = QMK_LIN_CONV_TAPS;
    constexpr int GROUP = QMK_LIN_NV / QMK_LIN_NK;
    const int h = idx / P.bs, b = idx % P.bs, kh = h / GROUP;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int slot = P.slot[b], par = P.conv_par[b];
    const bf16* row = a.qkvz + static_cast<size_t>(b) * QMK_LIN_QKVZ;
    float* S = a.state + (static_cast<size_t>(slot) * NV + h) * DK * DV;
    // The 64 KiB recurrent state is read in two passes below; start pulling it
    // into L2 now so the conv / l2norm phases hide the DRAM latency.
    if (tid == 0) bulk_prefetch_l2(S, DK * DV * sizeof(float));

    // ── causal conv update on this head's 384 channels ──
    const bf16* st_in = a.conv_state + static_cast<size_t>(slot * 2 + par) * (TAPS - 1) * CH;
    bf16* st_out = a.conv_state + static_cast<size_t>(slot * 2 + (par ^ 1)) * (TAPS - 1) * CH;
    const bf16* grp = row + kh * QMK_LIN_GROUP;  // this key head's slice of the grouped QKVZ output
    const int hv = h % GROUP;
    for (int t = tid; t < 3 * 128; t += 256) {
        const int set = t >> 7, i = t & 127;
        // c: channel in the checkpoint's conv layout [q | k | v]; x: same channel in the grouped QKVZ output.
        const int c = set == 0 ? kh * DK + i : set == 1 ? QMK_LIN_NK * DK + kh * DK + i : 2 * QMK_LIN_NK * DK + h * DV + i;
        const int x = set == 0 ? i : set == 1 ? QMK_LIN_GROUP_K + i : QMK_LIN_GROUP_V + hv * DV + i;
        const float xt = __bfloat162float(__ldcg(grp + x));
        float taps[TAPS - 1];
#pragma unroll
        for (int j = 0; j < TAPS - 1; ++j) taps[j] = __bfloat162float(__ldcg(st_in + j * CH + c));
        const bf16* w = a.conv_w + c * TAPS;
        float y = __bfloat162float(w[TAPS - 1]) * xt;
#pragma unroll
        for (int j = 0; j < TAPS - 1; ++j) y = fmaf(__bfloat162float(w[j]), taps[j], y);
        y = round_bf16(y);
        y = round_bf16(y / (1.0f + expf(-y)));
        (set == 0 ? sc.q : set == 1 ? sc.k : sc.v)[i] = y;
        if (set == 2 || h % GROUP == 0) {
#pragma unroll
            for (int j = 0; j < TAPS - 2; ++j) st_out[j * CH + c] = __float2bfloat16(taps[j + 1]);
            st_out[(TAPS - 2) * CH + c] = __float2bfloat16(xt);
        }
    }
    consumer_sync();

    // ── l2norm(q), l2norm(k), beta, decay ──
    if (warp < 2) {
        const float* src = warp == 0 ? sc.q : sc.k;
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) ss += src[lane * 4 + j] * src[lane * 4 + j];
        ss = warp_sum(ss);
        if (lane == 0) {
            const float inv = rsqrtf(ss + 1e-6f);
            if (warp == 0) sc.q_scale = inv; else sc.k_scale = inv;
        }
    } else if (warp == 2 && lane == 0) {
        const bf16* ab = a.ab + static_cast<size_t>(b) * QMK_LIN_AB;
        const float braw = __bfloat162float(__ldcg(ab + h));
        sc.beta = round_bf16(1.0f / (1.0f + expf(-braw)));
        const float x = __bfloat162float(__ldcg(ab + NV + h)) + a.dt_bias[h];
        const float sp = x > 20.f ? x : log1pf(expf(x));
        sc.decay = expf(-expf(a.a_log[h]) * sp);
    }
    consumer_sync();

    // ── recurrent update ──
    // Two passes over 64 columns; thread owns column pass*64 + (tid & 63) and
    // rows [quarter*32, quarter*32 + 32), so only 32 state values live in
    // registers (the block is capped at 168 registers per thread).
    const int jj = tid & 63, quarter = tid >> 6;
    const float qs = sc.q_scale / sqrtf(static_cast<float>(DK)), ks = sc.k_scale;
    const float beta = sc.beta, decay = sc.decay;
    for (int pass = 0; pass < 2; ++pass) {
        const int j = pass * 64 + jj;
        float s[DK / 4];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < DK / 4; ++i) {
            const int kk = quarter * (DK / 4) + i;
            s[i] = __ldcg(S + kk * DV + j) * decay;
            kv = fmaf(s[i], sc.k[kk] * ks, kv);
        }
        sc.red[quarter][jj] = kv;
        consumer_sync();
        const float delta = (sc.v[j] - (sc.red[0][jj] + sc.red[1][jj] + sc.red[2][jj] + sc.red[3][jj])) * beta;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < DK / 4; ++i) {
            const int kk = quarter * (DK / 4) + i;
            s[i] = fmaf(sc.k[kk] * ks, delta, s[i]);
            o = fmaf(s[i], sc.q[kk] * qs, o);
            __stcg(S + kk * DV + j, s[i]);
        }
        consumer_sync();  // everyone has read red[][] for kv
        sc.red[quarter][jj] = o;
        consumer_sync();
        if (quarter == 0) sc.o[j] = sc.red[0][jj] + sc.red[1][jj] + sc.red[2][jj] + sc.red[3][jj];
        consumer_sync();
    }

    // ── gated RMSNorm over the head (threads 0..127) ──
    const int j = tid & (DV - 1), half = tid >> 7;
    float ob = 0.f;
    if (half == 0) {
        ob = round_bf16(sc.o[j]);
        const float ss = warp_sum(ob * ob);
        if (lane == 0) sc.wsum[warp] = ss;
    }
    consumer_sync();
    if (half == 0) {
        const float var = (sc.wsum[0] + sc.wsum[1] + sc.wsum[2] + sc.wsum[3]) / DV;
        float hs = round_bf16(ob * rsqrtf(var + QMK_RMS_EPS));
        hs = round_bf16(__bfloat162float(a.norm_w[j]) * hs);
        const float z = __bfloat162float(__ldcg(grp + QMK_LIN_GROUP_Z + hv * DV + j));
        a.out[static_cast<size_t>(b) * QMK_MIX_OUT + h * DV + j] = __float2bfloat16(hs * (z / (1.0f + expf(-z))));
    }
}

}  // namespace qmk
