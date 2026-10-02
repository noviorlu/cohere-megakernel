// Host ↔ device ABI for the Qwen3.8 decode megakernel.
//
// Python mirrors these structs with ctypes (qmk/native.py) and checks the
// layout at load time through qmk_abi_layout(); keep both sides in sync.
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_bf16.h>
typedef __nv_bfloat16 qmk_bf16;
#else
typedef uint16_t qmk_bf16;
#endif

// ── model geometry (Qwen3.8-27B text decoder) ───────────────────────────────
#define QMK_HIDDEN 5120
#define QMK_RMS_EPS 1e-6f
#define QMK_MAX_BS 8

// Gated DeltaNet (linear attention) layers
#define QMK_LIN_NK 16      // key heads
#define QMK_LIN_NV 48      // value heads
#define QMK_LIN_DK 128
#define QMK_LIN_DV 128
#define QMK_LIN_CONV_CH 10240  // q(2048) + k(2048) + v(6144)
#define QMK_LIN_CONV_TAPS 4    // causal conv kernel; state keeps TAPS-1 inputs
#define QMK_LIN_QKVZ 16384     // fused in_proj_qkv (10240) + in_proj_z (6144)
#define QMK_LIN_AB 96          // fused in_proj_b (48) + in_proj_a (48)

// Gated full-attention layers
#define QMK_ATT_NQ 24
#define QMK_ATT_NKV 4
#define QMK_ATT_HD 256
#define QMK_ATT_ROT 64      // partial rotary: first 64 dims of each head
#define QMK_ATT_QKV 14336   // fused q_proj+gate (12288) + k (1024) + v (1024)
#define QMK_MIX_OUT 6144    // input width of out_proj / o_proj

// ── kernel structure ────────────────────────────────────────────────────────
#define QMK_CONSUMER_WARPS 8
#define QMK_THREADS ((QMK_CONSUMER_WARPS + 1) * 32)
#define QMK_CHUNK_BYTES 16384  // one weight ring stage
#define QMK_STAGES 4
#define QMK_TASK_QUEUE 3       // how many tasks a block may claim ahead

enum QmkOpType : int32_t {
    QMK_OP_GEMM_FP8 = 0,
    QMK_OP_GEMM_BF16 = 1,
    QMK_OP_GDN = 2,
    QMK_OP_ATTN = 3,
};

enum QmkEpilogue : int32_t {
    QMK_EPI_STORE = 0,     // out[b, n] = bf16(acc)
    QMK_EPI_SILU_MUL = 1,  // tile rows are [gate(TN/2); up(TN/2)] → out = silu(gate) * up
    QMK_EPI_RESID = 2,     // resid[b, n] += bf16(acc); write per-tile sum of squares
};

// Counter 0 is the global task ticket.
#define QMK_CTR_TICKET 0

typedef struct {
    int32_t ctr;  // -1: no dependency
    int32_t val;  // wait until counters[ctr] >= val
} QmkDep;

typedef struct {
    const void* w;          // pre-tiled weights, [tile][chunk][16 KiB]
    const float* wscale;    // FP8 only: [tile][chunk][8 warps]
    const qmk_bf16* x;      // input activations [bs][ldx]
    const qmk_bf16* norm_w; // zero-centred RMSNorm weight over K, or NULL
    const float* ss_in;     // sum-of-squares partials [n_ss][8] for the norm
    qmk_bf16* out;          // STORE / SILU_MUL output [bs][ldo]
    qmk_bf16* resid;        // RESID: residual stream [bs][ldo]
    float* ss_out;          // RESID: per-tile sum of squares [ntiles][8]
    int32_t ldx;
    int32_t n_ss;
    int32_t tile_n;         // TN: rows per task (16, 32, 64 or 128)
    int32_t nchunks;        // 16 KiB chunks per task
    int32_t epi;
    int32_t ldo;
} QmkGemmArgs;

typedef struct {
    const qmk_bf16* qkvz;     // [bs][QKVZ]
    const qmk_bf16* ab;       // [bs][AB]
    const qmk_bf16* conv_w;   // [CONV_CH][TAPS]
    qmk_bf16* conv_state;     // [slots][2 parities][TAPS-1][CONV_CH] for this layer
    float* state;             // [slots][NV][DK][DV] for this layer
    const float* a_log;       // [NV]
    const float* dt_bias;     // [NV]
    const qmk_bf16* norm_w;   // [DV] gated RMSNorm weight
    qmk_bf16* out;            // [bs][MIX_OUT]
} QmkGdnArgs;

typedef struct {
    const qmk_bf16* qkv;      // [bs][ATT_QKV]
    const qmk_bf16* q_norm_w; // [HD]
    const qmk_bf16* k_norm_w; // [HD]
    const qmk_bf16* rope_cos; // [max_ctx][ROT]
    const qmk_bf16* rope_sin; // [max_ctx][ROT]
    qmk_bf16* kcache;         // [slots][NKV][max_ctx][HD] for this layer
    qmk_bf16* vcache;
    float* part_ml;           // [bs][NKV][nsplit][NQ/NKV][2]
    float* part_acc;          // [bs][NKV][nsplit][NQ/NKV][HD]
    qmk_bf16* out;            // [bs][MIX_OUT]
    int32_t nsplit;
    int32_t comb_ctr;         // first of bs*NKV combine counters
} QmkAttnArgs;

typedef struct {
    int32_t type;
    int32_t ntasks;
    QmkDep wait[2];
    int32_t signal;  // counter bumped once per finished task (attention: per combined head group)
    int32_t pad_;
    union {
        QmkGemmArgs gemm;
        QmkGdnArgs gdn;
        QmkAttnArgs attn;
    };
} QmkOpDesc;

typedef struct {
    int32_t op;
    int32_t idx;
} QmkTask;

typedef struct {
    const QmkOpDesc* ops;
    const QmkTask* tasks;
    int32_t* counters;
    int64_t* prof;                 // optional [ntasks][4]: claim, deps-ready, done (globaltimer ns), smid
    int32_t ntasks;
    int32_t bs;
    int32_t max_ctx;
    int32_t l2_prefetch_chunks;    // per claimed GEMM task, chunks beyond the ring pulled into L2 early
    int32_t pos[QMK_MAX_BS];       // position of the token being decoded (= tokens already cached)
    int32_t slot[QMK_MAX_BS];      // cache slot of each batch row
    int32_t conv_par[QMK_MAX_BS];  // which conv-state buffer holds the current inputs
} QmkStepParams;
