// Weight-streaming GEMM task: out[b, n0:n0+TN] = x[b, :] · W[n0:n0+TN, :]^T, b < 8.
//
// Weights are pre-tiled on the host (qmk/layout.py) so that each 16 KiB ring
// chunk is contiguous and every consumer warp owns a 2 KiB, 16-row slice of
// it laid out in mma fragment order. Warps are arranged as R row groups x S
// K-slices (R*S = 8); within a chunk each warp covers 16 rows x 128 k (FP8)
// or 16 rows x 64 k (BF16). The batch maps onto the mma N dimension (n = 8).
//
// K is permuted inside every 32-wide step: thread (g, tig) holds physical
// k = 8*tig .. 8*tig+7 for both A and B, so one 16-byte load feeds two
// m16n8k16 MMAs. The permutation is applied identically to A and B, so the
// dot product is unchanged.
#pragma once

#include "abi.h"
#include "ptx.cuh"

namespace qmk {

struct Ring {
    uint8_t* stages;  // [QMK_STAGES][QMK_CHUNK_BYTES]
    uint64_t* full;
    uint64_t* empty;
    int stage = 0;
    uint32_t phase = 0;

    __device__ __forceinline__ uint8_t* data() const { return stages + stage * QMK_CHUNK_BYTES; }
    __device__ __forceinline__ void advance() {
        if (++stage == QMK_STAGES) {
            stage = 0;
            phase ^= 1;
        }
    }
};

struct GemmScratch {
    float red[QMK_CONSUMER_WARPS][16][8];  // per-warp 16x8 accumulators
    float val[128][8];                     // tile result [row][batch] (fp32)
    float sq[128][8];                      // RESID epilogue: squared outputs
    int last;
};

// Chunk range [c0, c1) of K-split `split`.
__device__ __forceinline__ void split_chunks(const QmkGemmArgs& a, int split, int& c0, int& c1) {
    c0 = split * a.nchunks / a.ksplit;
    c1 = (split + 1) * a.nchunks / a.ksplit;
}

// Producer side: stream this task's chunks into the ring.
// (Also pulling later chunks into L2 early — always, or only while the task
// waits on its inputs — measured no faster, so the ring is the only lookahead.)
__device__ __forceinline__ void gemm_issue(const QmkGemmArgs& a, int idx, Ring& ring, uint64_t policy) {
    const int tile = idx % a.ntiles;
    int c0, c1;
    split_chunks(a, idx / a.ntiles, c0, c1);
    const uint8_t* src =
        static_cast<const uint8_t*>(a.w) + (static_cast<size_t>(tile) * a.nchunks + c0) * QMK_CHUNK_BYTES;
    const int n = c1 - c0;
    for (int c = 0; c < n; ++c) {
        mbar_wait(&ring.empty[ring.stage], ring.phase ^ 1);
        mbar_arrive_expect_tx(&ring.full[ring.stage], QMK_CHUNK_BYTES);
        bulk_copy_g2s(ring.data(), src + static_cast<size_t>(c) * QMK_CHUNK_BYTES, QMK_CHUNK_BYTES,
                      &ring.full[ring.stage], policy);
        ring.advance();
    }
}

// Apply the zero-centred RMSNorm to 8 raw x values
// (HF: bf16((x.float() * inv_rms) * (1 + w.float()))).
__device__ __forceinline__ uint4 norm_x8(uint4 v, uint4 w, float inv) {
    float xf[8], wf[8];
    unpack_bf16x8(v, xf);
    unpack_bf16x8(w, wf);
#pragma unroll
    for (int i = 0; i < 8; ++i) xf[i] = (xf[i] * inv) * (1.0f + wf[i]);
    return pack_bf16x8(xf);
}

// Returns the tile index once the tile's output is final (always, unless
// split-K and another split of this tile is still running), else -1.
template <bool FP8>
__device__ int gemm_task(const QmkStepParams& P, const QmkGemmArgs& a, int idx, Ring& ring, GemmScratch& sc) {
    constexpr int SUBK = FP8 ? 128 : 64;  // k covered by one warp per chunk
    constexpr int STEPS = SUBK / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, tig = lane & 3;
    const int R = a.tile_n >> 4, S = QMK_CONSUMER_WARPS / R;
    const int s = warp % S;
    const int KC = S * SUBK;
    const bool row_ok = g < P.bs;
    const int tile = idx % a.ntiles, split = idx / a.ntiles;
    int c_begin, c_end;
    split_chunks(a, split, c_begin, c_end);

    const float inv = a.norm_w != nullptr && row_ok ? __ldcg(a.inv_in + g) : 0.f;

    const bf16* xrow = a.x + static_cast<size_t>(row_ok ? g : 0) * a.ldx;
    const float* scales = FP8 ? a.wscale + static_cast<size_t>(tile) * a.nchunks * QMK_CONSUMER_WARPS + warp : nullptr;
    const bool norm = a.norm_w != nullptr;

    // Activations come from L2; fetch chunk c+1's x (and norm weights, scale)
    // while chunk c computes so the L2 latency is off the per-chunk path.
    uint4 xr[STEPS], wr[STEPS];
    float next_scale = 1.f;
    auto fetch = [&](int c) {
        const int kb = c * KC + s * SUBK + tig * 8;
#pragma unroll
        for (int st = 0; st < STEPS; ++st) {
            xr[st] = row_ok ? ldcg_u4(xrow + kb + st * 32) : make_uint4(0, 0, 0, 0);
            if (norm) wr[st] = __ldg(reinterpret_cast<const uint4*>(a.norm_w + kb + st * 32));
        }
        if (FP8) next_scale = __ldg(scales + c * QMK_CONSUMER_WARPS);
    };
    fetch(c_begin);

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int c = c_begin; c < c_end; ++c) {
        uint4 X[STEPS];
#pragma unroll
        for (int st = 0; st < STEPS; ++st) X[st] = norm && row_ok ? norm_x8(xr[st], wr[st], inv) : xr[st];
        const float scale = next_scale;
        if (c + 1 < c_end) fetch(c + 1);

        mbar_wait(&ring.full[ring.stage], ring.phase);
        const uint8_t* wb = ring.data() + warp * 2048;
        // Two independent accumulator chains (even / odd k16 halves) for MMA ILP.
        float pa[4] = {0.f, 0.f, 0.f, 0.f}, pb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int st = 0; st < STEPS; ++st) {
            if constexpr (FP8) {
                // 16 B per lane: row g (k 0..7), then row g+8 (k 0..7).
                const uint4 A = lds_u4(wb + st * 512 + lane * 16);
                uint32_t g01, g23, g45, g67, h01, h23, h45, h67;
                e4m3x4_to_bf16x4(A.x, g01, g23);
                e4m3x4_to_bf16x4(A.y, g45, g67);
                e4m3x4_to_bf16x4(A.z, h01, h23);
                e4m3x4_to_bf16x4(A.w, h45, h67);
                mma_bf16_16816(pa, g01, h01, g23, h23, X[st].x, X[st].y);
                mma_bf16_16816(pb, g45, h45, g67, h67, X[st].z, X[st].w);
            } else {
                // 32 B per lane, split into a row-g half and a row-g+8 half so
                // each LDS.128 wavefront stays bank-conflict free.
                const uint4 A0 = lds_u4(wb + st * 1024 + lane * 16);
                const uint4 A1 = lds_u4(wb + st * 1024 + 512 + lane * 16);
                mma_bf16_16816(pa, A0.x, A1.x, A0.y, A1.y, X[st].x, X[st].y);
                mma_bf16_16816(pb, A0.z, A1.z, A0.w, A1.w, X[st].z, X[st].w);
            }
        }
        // Release the stage: every lane's generic-proxy reads must be ordered
        // before the producer's next bulk copy (async proxy) into it.
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&ring.empty[ring.stage]);
        ring.advance();
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = fmaf(pa[i] + pb[i], scale, acc[i]);
    }

    // D fragment: (row g, col 2tig), (g, 2tig+1), (g+8, 2tig), (g+8, 2tig+1).
    sc.red[warp][g][2 * tig] = acc[0];
    sc.red[warp][g][2 * tig + 1] = acc[1];
    sc.red[warp][g + 8][2 * tig] = acc[2];
    sc.red[warp][g + 8][2 * tig + 1] = acc[3];
    consumer_sync();

    // Sum the S K-slices of each (row, batch).
    const int TN = a.tile_n;
    for (int e = threadIdx.x; e < TN * 8; e += 256) {
        const int i = e >> 3, b = e & 7, rg = i >> 4, row = i & 15;
        float v = 0.f;
        for (int k = 0; k < S; ++k) v += sc.red[rg * S + k][row][b];
        sc.val[i][b] = v;
    }

    if (a.ksplit > 1) {
        // Publish this split's partial; the last split of the tile reduces all
        // of them in split order (deterministic) and runs the epilogue.
        for (int e = threadIdx.x; e < TN * 8; e += 256) {
            const int i = e >> 3, b = e & 7;
            if (b < P.bs) __stcg(a.partial + (static_cast<size_t>(split) * 8 + b) * a.ntiles * TN + tile * TN + i,
                                 sc.val[i][b]);
        }
        __threadfence();
        consumer_sync();
        if (threadIdx.x == 0) sc.last = atomicAdd(P.counters + a.tile_ctr + tile, 1) == a.ksplit - 1;
        consumer_sync();
        if (!sc.last) return -1;
        __threadfence();
        for (int e = threadIdx.x; e < TN * 8; e += 256) {
            const int i = e >> 3, b = e & 7;
            float v = 0.f;
            if (b < P.bs)
                for (int k = 0; k < a.ksplit; ++k)
                    v += __ldcg(a.partial + (static_cast<size_t>(k) * 8 + b) * a.ntiles * TN + tile * TN + i);
            sc.val[i][b] = v;
        }
    }
    consumer_sync();
    auto value = [&](int i, int b) { return sc.val[i][b]; };

    if (a.epi == QMK_EPI_STORE) {
        for (int e = threadIdx.x; e < TN * 8; e += 256) {
            const int i = e >> 3, b = e & 7;
            if (b < P.bs) a.out[static_cast<size_t>(b) * a.ldo + tile * TN + i] = __float2bfloat16(value(i, b));
        }
    } else if (a.epi == QMK_EPI_SILU_MUL) {
        const int half = TN >> 1;
        for (int e = threadIdx.x; e < half * 8; e += 256) {
            const int i = e >> 3, b = e & 7;
            if (b >= P.bs) continue;
            const float gate = round_bf16(value(i, b));
            const float up = round_bf16(value(i + half, b));
            const float act = round_bf16(gate / (1.0f + expf(-gate)));
            a.out[static_cast<size_t>(b) * a.ldo + tile * half + i] = __float2bfloat16(act * up);
        }
    } else {  // QMK_EPI_RESID
        for (int e = threadIdx.x; e < TN * 8; e += 256) {
            const int i = e >> 3, b = e & 7;
            float sq = 0.f;
            if (b < P.bs) {
                bf16* p = a.resid + static_cast<size_t>(b) * a.ldo + tile * TN + i;
                const float upd = round_bf16(value(i, b));
                const float nv = round_bf16(__bfloat162float(__ldcg(p)) + upd);
                *p = __float2bfloat16(nv);
                sq = nv * nv;
            }
            sc.sq[i][b] = sq;
        }
        consumer_sync();
        if (threadIdx.x < 8) {
            float sum = 0.f;
            for (int i = 0; i < TN; ++i) sum += sc.sq[i][threadIdx.x];
            a.ss_out[tile * 8 + threadIdx.x] = sum;
        }
        // The last tile to finish turns the per-tile sums into each row's
        // 1/rms for the next RMSNorm, then signals the whole op as done (so
        // consumers read 8 floats instead of summing every tile's partial).
        __threadfence();
        consumer_sync();
        if (threadIdx.x == 0) sc.last = atomicAdd(P.counters + a.norm_ctr, 1) == a.ntiles - 1;
        consumer_sync();
        if (!sc.last) return -1;
        __threadfence();
        if (warp < P.bs) {
            float sum = 0.f;
            for (int t = lane; t < a.ntiles; t += 32) sum += __ldcg(a.ss_out + t * 8 + warp);
            sum = warp_sum(sum);
            if (lane == 0) a.inv_out[warp] = rsqrtf(sum / QMK_HIDDEN + QMK_RMS_EPS);
        }
    }
    return tile;
}

}  // namespace qmk
