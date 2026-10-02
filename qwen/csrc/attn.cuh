// Gated full-attention decode task: one (kv head g, batch row b, KV split s).
//
// Mirrors Qwen3_5Attention for one new token:
//   q/k: zero-centred RMSNorm per head, then RoPE on the first 64 dims
//        (bf16 arithmetic, cos/sin tables precomputed exactly as HF does);
//   scores in fp32, online softmax across the split, split partials merged by
//   whichever task finishes its (b, g) group last;
//   out = bf16(attn) * bf16(sigmoid(gate)).
// The split containing the new position also writes that position's K/V
// into the cache; no other split reads it.
#pragma once

#include "abi.h"
#include "ptx.cuh"

namespace qmk {

constexpr int ATT_GROUP = QMK_ATT_NQ / QMK_ATT_NKV;  // 6 query heads per kv head

struct AttnScratch {
    float q[ATT_GROUP][QMK_ATT_HD];
    float kbuf[QMK_ATT_HD];
    float tree_o[4][ATT_GROUP][QMK_ATT_HD];
    float tree_ml[4][ATT_GROUP][2];
    int last;
};

// RMSNorm (zero-centred) of one head, 8 dims per lane; returns bf16-rounded values.
__device__ __forceinline__ void head_rmsnorm(const bf16* src, const bf16* w, int lane, float (&y)[8]) {
    float x[8], wf[8];
    unpack_bf16x8(ldcg_u4(src + lane * 8), x);
    unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(w + lane * 8)), wf);
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
    const float inv = rsqrtf(warp_sum(ss) / QMK_ATT_HD + QMK_RMS_EPS);
#pragma unroll
    for (int e = 0; e < 8; ++e) y[e] = round_bf16((x[e] * inv) * (1.0f + wf[e]));
}

// Partial RoPE in place on buf[0..ROT) (HF rotate_half, all ops rounded to bf16).
// `y` holds this lane's 8 dims; buf must already contain the whole head.
__device__ __forceinline__ void rope_head(float* buf, float (&y)[8], const bf16* cos, const bf16* sin, int lane) {
    constexpr int HALF = QMK_ATT_ROT / 2;
    const int d0 = lane * 8;
    if (d0 < QMK_ATT_ROT) {
        float rot[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int d = d0 + e;
            rot[e] = d < HALF ? -buf[d + HALF] : buf[d - HALF];
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float c = __bfloat162float(cos[d0 + e]), s = __bfloat162float(sin[d0 + e]);
            y[e] = round_bf16(round_bf16(y[e] * c) + round_bf16(rot[e] * s));
        }
    }
    __syncwarp();
#pragma unroll
    for (int e = 0; e < 8; ++e) buf[d0 + e] = y[e];
    __syncwarp();
}

__device__ __forceinline__ void merge_softmax(float& m, float& l, float* o, float m2, float l2, const float* o2) {
    const float M = fmaxf(m, m2);
    if (M == -INFINITY) return;
    const float c1 = expf(m - M), c2 = expf(m2 - M);
    l = l * c1 + l2 * c2;
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = o[e] * c1 + o2[e] * c2;
    m = M;
}

// idx = (g * bs + b) * nsplit + s (kv-group-major, so o_proj's K-split g can
// start once group g is done). Returns true if this task merged the
// (b, g) group, i.e. the op's output for it is complete.
__device__ bool attn_task(const QmkStepParams& P, const QmkAttnArgs& a, int idx, int* counters, AttnScratch& sc) {
    constexpr int HD = QMK_ATT_HD, NKV = QMK_ATT_NKV;
    const int nsplit = a.nsplit;
    const int s = idx % nsplit, g = idx / nsplit / P.bs, b = idx / nsplit % P.bs;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = P.pos[b], slot = P.slot[b], ctx = pos + 1;
    const int chunk = (ctx + nsplit - 1) / nsplit;
    const int p0 = s * chunk, p1 = min(ctx, p0 + chunk);
    const bool owns_new = p0 <= pos && pos < p1;
    const bf16* row = a.qkv + static_cast<size_t>(b) * QMK_ATT_QKV + g * QMK_ATT_GROUP;  // grouped per kv head
    const size_t kv_off = static_cast<size_t>(slot * NKV + g) * P.max_ctx * HD;
    bf16* Kc = a.kcache + kv_off;
    bf16* Vc = a.vcache + kv_off;
    const bf16* cos = a.rope_cos + static_cast<size_t>(pos) * QMK_ATT_ROT;
    const bf16* sin = a.rope_sin + static_cast<size_t>(pos) * QMK_ATT_ROT;

    // ── q (warps 0-5), new k (warp 6), new v (warp 7) ──
    if (warp < ATT_GROUP) {
        float y[8];
        head_rmsnorm(row + warp * 2 * HD, a.q_norm_w, lane, y);
#pragma unroll
        for (int e = 0; e < 8; ++e) sc.q[warp][lane * 8 + e] = y[e];
        __syncwarp();
        rope_head(sc.q[warp], y, cos, sin, lane);
    } else if (warp == 6 && owns_new) {
        float y[8];
        head_rmsnorm(row + QMK_ATT_GROUP_K, a.k_norm_w, lane, y);
#pragma unroll
        for (int e = 0; e < 8; ++e) sc.kbuf[lane * 8 + e] = y[e];
        __syncwarp();
        rope_head(sc.kbuf, y, cos, sin, lane);
        *reinterpret_cast<uint4*>(Kc + static_cast<size_t>(pos) * HD + lane * 8) = pack_bf16x8(y);
    } else if (warp == 7 && owns_new) {
        const uint4 v = ldcg_u4(row + QMK_ATT_GROUP_V + lane * 8);
        *reinterpret_cast<uint4*>(Vc + static_cast<size_t>(pos) * HD + lane * 8) = v;
    }
    __threadfence();
    consumer_sync();

    // ── scores + online softmax; warp w takes positions p0+w, p0+w+8, ... ──
    const float scaling = rsqrtf(static_cast<float>(HD));
    float qr[ATT_GROUP][8];
#pragma unroll
    for (int i = 0; i < ATT_GROUP; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) qr[i][e] = sc.q[i][lane * 8 + e] * scaling;
    float m[ATT_GROUP], l[ATT_GROUP], o[ATT_GROUP][8];
#pragma unroll
    for (int i = 0; i < ATT_GROUP; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) o[i][e] = 0.f;
    }
    for (int p = p0 + warp; p < p1; p += QMK_CONSUMER_WARPS) {
        float kf[8], vf[8];
        unpack_bf16x8(ldcg_u4(Kc + static_cast<size_t>(p) * HD + lane * 8), kf);
        unpack_bf16x8(ldcg_u4(Vc + static_cast<size_t>(p) * HD + lane * 8), vf);
        float sc_[ATT_GROUP];
#pragma unroll
        for (int i = 0; i < ATT_GROUP; ++i) {
            float d = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) d = fmaf(qr[i][e], kf[e], d);
            sc_[i] = d;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
#pragma unroll
            for (int i = 0; i < ATT_GROUP; ++i) sc_[i] += __shfl_xor_sync(0xffffffffu, sc_[i], off);
#pragma unroll
        for (int i = 0; i < ATT_GROUP; ++i) {
            const float mn = fmaxf(m[i], sc_[i]);
            const float corr = expf(m[i] - mn), pe = expf(sc_[i] - mn);
            l[i] = l[i] * corr + pe;
#pragma unroll
            for (int e = 0; e < 8; ++e) o[i][e] = fmaf(o[i][e], corr, pe * vf[e]);
            m[i] = mn;
        }
    }

    // ── merge the 8 warps: 4→0..3, 2→0..1, 1→0 ──
    for (int half = 4; half >= 1; half >>= 1) {
        if (warp >= half && warp < 2 * half) {
            const int t = warp - half;
#pragma unroll
            for (int i = 0; i < ATT_GROUP; ++i) {
#pragma unroll
                for (int e = 0; e < 8; ++e) sc.tree_o[t][i][lane * 8 + e] = o[i][e];
                if (lane == 0) {
                    sc.tree_ml[t][i][0] = m[i];
                    sc.tree_ml[t][i][1] = l[i];
                }
            }
        }
        consumer_sync();
        if (warp < half) {
#pragma unroll
            for (int i = 0; i < ATT_GROUP; ++i)
                merge_softmax(m[i], l[i], o[i], sc.tree_ml[warp][i][0], sc.tree_ml[warp][i][1],
                              &sc.tree_o[warp][i][lane * 8]);
        }
        consumer_sync();
    }

    // ── publish this split, then let the last split of (b, g) combine ──
    const int group = b * NKV + g;
    const size_t pi = static_cast<size_t>(group) * nsplit + s;
    if (warp == 0) {
#pragma unroll
        for (int i = 0; i < ATT_GROUP; ++i) {
            float* dst = a.part_acc + (pi * ATT_GROUP + i) * HD + lane * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) __stcg(dst + e, o[i][e]);
            if (lane == 0) {
                __stcg(a.part_ml + (pi * ATT_GROUP + i) * 2, m[i]);
                __stcg(a.part_ml + (pi * ATT_GROUP + i) * 2 + 1, l[i]);
            }
        }
    }
    __threadfence();
    consumer_sync();
    if (threadIdx.x == 0) sc.last = atomicAdd(counters + a.comb_ctr + group, 1) == nsplit - 1;
    consumer_sync();
    if (!sc.last) return false;
    __threadfence();

    const size_t gbase = static_cast<size_t>(group) * nsplit;
    for (int e = threadIdx.x; e < ATT_GROUP * HD; e += 256) {
        const int i = e / HD, d = e % HD;
        float M = -INFINITY;
        for (int k = 0; k < nsplit; ++k) M = fmaxf(M, __ldcg(a.part_ml + ((gbase + k) * ATT_GROUP + i) * 2));
        float L = 0.f, O = 0.f;
        for (int k = 0; k < nsplit; ++k) {
            const size_t pk = (gbase + k) * ATT_GROUP + i;
            const float w = expf(__ldcg(a.part_ml + pk * 2) - M);
            L = fmaf(__ldcg(a.part_ml + pk * 2 + 1), w, L);
            O = fmaf(__ldcg(a.part_acc + pk * HD + d), w, O);
        }
        const int hq = g * ATT_GROUP + i;
        const float attn = round_bf16(O / L);
        const float gate = __bfloat162float(__ldcg(row + i * 2 * HD + HD + d));
        const float sg = round_bf16(1.0f / (1.0f + expf(-gate)));
        a.out[static_cast<size_t>(b) * QMK_MIX_OUT + hq * HD + d] = __float2bfloat16(attn * sg);
    }
    return true;
}

}  // namespace qmk
