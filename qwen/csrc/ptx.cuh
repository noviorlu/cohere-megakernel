// Thin PTX wrappers used by the Qwen3.8 decode megakernel (sm_120).
//
// Everything here is available on consumer Blackwell (sm_120): mbarrier,
// 1-D bulk async copies (cp.async.bulk), and Ampere-style mma.sync. There is
// no WGMMA / tcgen05 on this architecture, which is why the kernel is built on
// mma.sync rather than ThunderKittens' Hopper path.
#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace qmk {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ── mbarrier ────────────────────────────────────────────────────────────────

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(count));
}

// Make barrier initialisation visible to the async (bulk-copy) proxy.
__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_addr(bar)) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(bar)),
                 "r"(bytes)
                 : "memory");
}

// True once the phase with the given parity has completed. A fresh barrier
// reports parity 1 as complete, which is what lets producers start on an
// "empty" ring without a priming round.
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t parity) {
    uint32_t done;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(done)
        : "r"(smem_addr(bar)), "r"(parity)
        : "memory");
    return done != 0;
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    while (!mbar_try_wait(bar, parity)) {
    }
}

// ── bulk async copy (global → shared), completion via mbarrier tx-count ─────

__device__ __forceinline__ uint64_t policy_evict_first() {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}

__device__ __forceinline__ void bulk_copy_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar,
                                              uint64_t policy) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], "
        "%4;" ::"r"(smem_addr(dst)),
        "l"(src), "r"(bytes), "r"(smem_addr(bar)), "l"(policy)
        : "memory");
}

// Hint: start pulling `bytes` (multiple of 16) from DRAM into L2; no smem, no completion.
__device__ __forceinline__ void bulk_prefetch_l2(const void* src, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

// ── cross-block synchronisation through global counters ─────────────────────

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Named barrier over the 8 consumer warps only (the producer warp never joins).
__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, 256;" ::: "memory"); }

// ── loads that bypass L1 ────────────────────────────────────────────────────
// Activations, recurrent state and KV are rewritten by other SMs during the
// same launch; L1 is not coherent, so every such read goes to L2.

__device__ __forceinline__ uint4 ldcg_u4(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }

__device__ __forceinline__ uint4 lds_u4(const void* p) { return *reinterpret_cast<const uint4*>(p); }

// ── tensor core MMA ─────────────────────────────────────────────────────────

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ── FP8 (e4m3) → BF16 ───────────────────────────────────────────────────────
// Moving the 7 exponent+mantissa bits of an e4m3 byte into bf16 position
// yields the right value times 2^-120 for normals *and* subnormals (both
// formats keep the same relative offset), so one exact bf16 multiply by 2^120
// finishes the conversion. Two bytes in, one bf16x2 out.

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t spread) {
    // `spread` holds the two fp8 bytes in the high byte of each 16-bit lane.
    uint32_t bits = (spread & 0x80008000u) | ((spread >> 4) & 0x07f007f0u);
    const __nv_bfloat162 k2p120 = __halves2bfloat162(__ushort_as_bfloat16(0x7B80), __ushort_as_bfloat16(0x7B80));
    __nv_bfloat162 v = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&bits), k2p120);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Bytes {b0,b1,b2,b3} of `w` → lo = bf16x2(b0,b1), hi = bf16x2(b2,b3).
__device__ __forceinline__ void e4m3x4_to_bf16x4(uint32_t w, uint32_t& lo, uint32_t& hi) {
    lo = e4m3x2_to_bf16x2(__byte_perm(w, 0, 0x1404));
    hi = e4m3x2_to_bf16x2(__byte_perm(w, 0, 0x3424));
}

// ── misc ────────────────────────────────────────────────────────────────────

__device__ __forceinline__ int64_t globaltimer() {
    int64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ int smid() {
    int id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ void unpack_bf16x8(const uint4& v, float (&f)[8]) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ uint4 pack_bf16x8(const float (&f)[8]) {
    uint4 v;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return v;
}

}  // namespace qmk
