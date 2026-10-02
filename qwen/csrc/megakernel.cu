// Qwen3.8-27B decode megakernel for sm_120 (RTX 5090).
//
// One persistent block per SM runs one whole decode step. Each block has a
// producer warp and 8 consumer warps:
//   producer  claims tasks from a global ticket (QMK_CTR_TICKET), hands them
//             to the consumers through a small smem queue, and streams GEMM
//             weight chunks into a QMK_STAGES-deep ring with bulk async copies
//             — including for tasks whose inputs are not ready yet;
//   consumers wait on the task's dependency counters, compute, then bump the
//             op's signal counter.
// Tasks are listed in topological order and claimed in that order, so every
// dependency of a claimed task is held by a block that is already running:
// the schedule cannot deadlock however the ticket interleaves.
#include <cstdio>

#include "abi.h"
#include "attn.cuh"
#include "gdn.cuh"
#include "gemm.cuh"
#include "ptx.cuh"

namespace qmk {

struct __align__(128) Smem {
    uint8_t ring[QMK_STAGES][QMK_CHUNK_BYTES];
    union {
        GemmScratch gemm;
        GdnScratch gdn;
        AttnScratch attn;
    } u;
    uint64_t full[QMK_STAGES];
    uint64_t empty[QMK_STAGES];
    uint64_t task_full[QMK_TASK_QUEUE];
    uint64_t task_empty[QMK_TASK_QUEUE];
    int task_id[QMK_TASK_QUEUE];
    int finished;  // tasks the consumers have completed (written by consumer thread 0)
};

__device__ __forceinline__ bool is_gemm(int type) { return type == QMK_OP_GEMM_FP8 || type == QMK_OP_GEMM_BF16; }

__device__ void producer(const QmkStepParams& P, Smem& sm) {
    if ((threadIdx.x & 31) != 0) return;
    Ring ring{&sm.ring[0][0], sm.full, sm.empty};
    const uint64_t policy = policy_evict_first();
    int q = 0, claimed = 0;
    uint32_t qphase = 0;
    while (true) {
        mbar_wait(&sm.task_empty[q], qphase ^ 1);
        // Claiming ahead only pays off for GEMMs (their weights start streaming).
        // A GDN/attention task claimed by a busy block would sit behind that
        // block's current task while other blocks idle, so leave those to
        // blocks whose consumers have nothing left to do.
        while (true) {
            const int next = *reinterpret_cast<volatile int*>(P.counters + QMK_CTR_TICKET);
            if (next >= P.ntasks || is_gemm(P.ops[P.tasks[next].op].type)) break;
            if (*reinterpret_cast<volatile int*>(&sm.finished) == claimed) break;
            __nanosleep(32);
        }
        const int t = atomicAdd(P.counters + QMK_CTR_TICKET, 1);
        ++claimed;
        const int id = t < P.ntasks ? t : -1;
        sm.task_id[q] = id;
        mbar_arrive(&sm.task_full[q]);
        if (++q == QMK_TASK_QUEUE) {
            q = 0;
            qphase ^= 1;
        }
        if (id < 0) return;
        const QmkTask task = P.tasks[id];
        const QmkOpDesc& op = P.ops[task.op];
        if (is_gemm(op.type)) gemm_issue(op.gemm, task.idx, ring, policy, P.l2_prefetch_chunks);
    }
}

__device__ __forceinline__ int dep_key(const QmkKey& k, int i) {
    int v = i / k.div1;
    if (k.mod > 0) v %= k.mod;
    return min(v / k.div2, k.kmax);
}

__device__ void wait_deps(const QmkStepParams& P, const QmkOpDesc& op, int idx) {
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const QmkDep& d = op.wait[i];
            if (d.ctr < 0) continue;
            const int c = d.ctr + dep_key(d.key, idx);
            const int target = P.targets[c];
            while (ld_acquire(P.counters + c) < target) __nanosleep(64);
        }
    }
    consumer_sync();
}

// `unit`: task index (GDN, attention) or tile index (GEMM) the signal key is computed from.
__device__ void signal(const QmkStepParams& P, const QmkOpDesc& op, int unit) {
    consumer_sync();
    if (threadIdx.x == 0 && op.signal.ctr >= 0) {
        __threadfence();
        atomicAdd(P.counters + op.signal.ctr + dep_key(op.signal.key, unit), 1);
    }
}

__device__ void consumer(const QmkStepParams& P, Smem& sm) {
    Ring ring{&sm.ring[0][0], sm.full, sm.empty};
    const int lane = threadIdx.x & 31;
    int q = 0;
    uint32_t qphase = 0;
    while (true) {
        mbar_wait(&sm.task_full[q], qphase);
        const int id = sm.task_id[q];
        __syncwarp();
        if (lane == 0) mbar_arrive(&sm.task_empty[q]);
        if (++q == QMK_TASK_QUEUE) {
            q = 0;
            qphase ^= 1;
        }
        if (id < 0) return;
        const QmkTask task = P.tasks[id];
        const QmkOpDesc& op = P.ops[task.op];
        int64_t* prof = P.prof != nullptr && threadIdx.x == 0 ? P.prof + 4 * static_cast<size_t>(id) : nullptr;
        if (prof) prof[0] = globaltimer();
        wait_deps(P, op, task.idx);
        if (prof) prof[1] = globaltimer();
        int unit = task.idx;  // < 0: output not complete yet (another split will signal)
        switch (op.type) {
            case QMK_OP_GEMM_FP8: unit = gemm_task<true>(P, op.gemm, task.idx, ring, sm.u.gemm); break;
            case QMK_OP_GEMM_BF16: unit = gemm_task<false>(P, op.gemm, task.idx, ring, sm.u.gemm); break;
            case QMK_OP_GDN: gdn_task(P, op.gdn, task.idx, sm.u.gdn); break;
            case QMK_OP_ATTN:
                if (!attn_task(P, op.attn, task.idx, P.counters, sm.u.attn)) unit = -1;
                break;
        }
        if (unit >= 0) signal(P, op, unit);
        else consumer_sync();
        if (prof) {
            prof[2] = globaltimer();
            prof[3] = smid();
        }
        if (threadIdx.x == 0) *reinterpret_cast<volatile int*>(&sm.finished) += 1;
    }
}

__global__ void __launch_bounds__(QMK_THREADS, 1) decode_kernel(const __grid_constant__ QmkStepParams P) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    if (threadIdx.x == 0) {
        for (int s = 0; s < QMK_STAGES; ++s) {
            mbar_init(&sm.full[s], 1);
            mbar_init(&sm.empty[s], QMK_CONSUMER_WARPS);
        }
        for (int s = 0; s < QMK_TASK_QUEUE; ++s) {
            mbar_init(&sm.task_full[s], 1);
            mbar_init(&sm.task_empty[s], QMK_CONSUMER_WARPS);
        }
        fence_barrier_init();
        sm.finished = 0;
    }
    __syncthreads();
    if (threadIdx.x >= QMK_CONSUMER_WARPS * 32) producer(P, sm);
    else consumer(P, sm);
}

// Test helper: e4m3 byte → bf16 through the same path the GEMM uses.
__global__ void e4m3_to_bf16_kernel(const uint8_t* in, bf16* out, int n) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    uint32_t w = 0;
    for (int k = 0; k < 4 && i + k < n; ++k) w |= static_cast<uint32_t>(in[i + k]) << (8 * k);
    uint32_t lo, hi;
    e4m3x4_to_bf16x4(w, lo, hi);
    const bf16* l = reinterpret_cast<const bf16*>(&lo);
    const bf16* h = reinterpret_cast<const bf16*>(&hi);
    const bf16 vals[4] = {l[0], l[1], h[0], h[1]};
    for (int k = 0; k < 4 && i + k < n; ++k) out[i + k] = vals[k];
}

}  // namespace qmk

#define QMK_FIELD(T, f) static_cast<int64_t>(offsetof(T, f))

extern "C" {

// Sizes/offsets the ctypes mirror must reproduce.
int qmk_abi_layout(int64_t* out, int n) {
    const int64_t v[] = {
        sizeof(QmkOpDesc),          QMK_FIELD(QmkOpDesc, gemm),     sizeof(QmkGemmArgs),
        sizeof(QmkGdnArgs),         sizeof(QmkAttnArgs),            sizeof(QmkStepParams),
        QMK_FIELD(QmkGemmArgs, ldx), QMK_FIELD(QmkAttnArgs, nsplit), QMK_FIELD(QmkStepParams, pos),
        QMK_FIELD(QmkOpDesc, signal), QMK_FIELD(QmkGemmArgs, ksplit), QMK_FIELD(QmkGemmArgs, norm_ctr),
        static_cast<int64_t>(sizeof(qmk::Smem)),
    };
    const int count = static_cast<int>(sizeof(v) / sizeof(v[0]));
    for (int i = 0; i < n && i < count; ++i) out[i] = v[i];
    return count;
}

int qmk_smem_bytes() { return static_cast<int>(sizeof(qmk::Smem)); }

// Launch one decode step. counters must be zeroed beforehand (stream-ordered).
int qmk_launch(const QmkStepParams* params, int num_blocks, void* stream) {
    const int smem = static_cast<int>(sizeof(qmk::Smem));
    static bool configured = false;
    if (!configured) {
        cudaError_t e = cudaFuncSetAttribute(qmk::decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return static_cast<int>(e);
        configured = true;
    }
    qmk::decode_kernel<<<num_blocks, QMK_THREADS, smem, static_cast<cudaStream_t>(stream)>>>(*params);
    return static_cast<int>(cudaGetLastError());
}

int qmk_e4m3_to_bf16(const void* in, void* out, int n, void* stream) {
    const int threads = 256, per = threads * 4;
    qmk::e4m3_to_bf16_kernel<<<(n + per - 1) / per, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(in), static_cast<qmk_bf16*>(out), n);
    return static_cast<int>(cudaGetLastError());
}

const char* qmk_error_string(int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); }

}  // extern "C"
