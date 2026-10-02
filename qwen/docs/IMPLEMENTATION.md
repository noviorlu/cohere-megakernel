# How the Qwen3.8 megakernel is implemented

This document explains how `qwen/` runs one decode step of
**Qwen3.8-27B-FP8** as a single persistent CUDA kernel on an **RTX 5090
(sm_120)**, why it is built the way it is, what went wrong on the way, and
which ideas were measured and dropped. Usage and headline numbers are in
[`../README.md`](../README.md).

Contents

1. [Starting point and constraints](#1-starting-point-and-constraints)
2. [The model as the kernel sees it](#2-the-model-as-the-kernel-sees-it)
3. [Execution model: one block per SM, a task list, counters](#3-execution-model)
4. [The weight-streaming GEMM](#4-the-weight-streaming-gemm)
5. [Fusions: RMSNorm, SiLU·up, residual, split-K](#5-fusions)
6. [Gated DeltaNet decode task](#6-gated-deltanet-decode-task)
7. [Gated attention decode task](#7-gated-attention-decode-task)
8. [The schedule: ops, fine-grained dependencies, deadlock freedom](#8-the-schedule)
9. [Host side: weights, prefill, caches, engine](#9-host-side)
10. [Correctness strategy](#10-correctness-strategy)
11. [sm_120 traps and bugs found](#11-sm_120-traps-and-bugs-found)
12. [Performance: what moved the needle, what did not](#12-performance)
13. [Benchmarking against vLLM (and the baseline mistake)](#13-benchmarking-against-vllm)
14. [Decisions: whose call was what](#14-decisions)
15. [File map](#15-file-map)

---

## 1. Starting point and constraints

The repository is Cohere's decode megakernel for North Mini Code (a 30B
MoE) on H100. It is built on ThunderKittens' Hopper path: **WGMMA** tensor
cores, **TMA** tensor-map copies, `setmaxnreg`, 227 KB of shared memory per
block. The goal was to get a megakernel running on a single RTX 5090.

Two facts changed the plan:

* **The architecture.** sm_120 (consumer Blackwell) has neither WGMMA nor
  tcgen05. It has Ampere-style `mma.sync`, `mbarrier`, 1-D bulk async copies
  (`cp.async.bulk`), 99 KB of dynamic shared memory per block, 170 SMs and a
  96 MB L2. The Hopper kernel cannot be "ported" by flags; the compute core has
  to be rewritten.
* **The model.** North Mini Code is 61 GB in BF16 and does not fit in 32 GB.
  After discussing options (NVFP4 North Mini Code vs. a different model), the
  user chose **Qwen3.8-27B-FP8**: 24.7 B parameters in FP8 with 128×128 block
  scales plus 3.1 B in BF16. Text-only, that is ~27 GB of VRAM, leaving ~4 GB
  for caches.

So the result is a new subproject next to the H100 code. It reuses Cohere's
*design* (persistent blocks, a task list, dependency counters in global memory,
weight prefetch while waiting on activations) but none of its code.

**Why a megakernel can only win a little here.** Decode of a dense 27B model
is bound by reading weights: 26.9 GB per step (24.3 GB FP8 linears + 2.5 GB
BF16 `lm_head`). At the ~1.65 TB/s the 5090 sustains, that is ~16.3 ms per
step no matter what. A megakernel removes launch gaps and lets work overlap
across op boundaries; with CUDA graphs and fusion a conventional engine
already removes most launch cost. The realistic target was "close to the
bandwidth floor", not a multiple of vLLM.

## 2. The model as the kernel sees it

Qwen3.8-27B (`model_type: qwen3_5`) has 64 decoder layers, hidden size 5120,
MLP intermediate 17408, vocab 248320:

* **48 Gated DeltaNet layers** (linear attention). Per token:
  `in_proj_qkv` (5120→10240: 16 key heads × 128 for q and k, 48 value heads ×
  128 for v), `in_proj_z` (→6144, gate), `in_proj_b`/`in_proj_a` (→48 each,
  BF16), a depthwise causal conv1d (kernel 4) + SiLU over the 10240 q/k/v
  channels, l2-normalised q/k, the gated delta rule on a 128×128 fp32 state per
  value head, a gated RMSNorm (`norm(o) · silu(z)`), and `out_proj` (6144→5120).
* **16 gated full-attention layers** (every 4th): `q_proj` produces query
  *and* an output gate per head (24 heads × 256 × 2), `k_proj`/`v_proj` 4 KV
  heads × 256, zero-centred RMSNorm on q and k per head, RoPE on the first 64
  of 256 dims (θ = 1e7; the multimodal RoPE collapses to plain RoPE for text),
  GQA attention (6 query heads per KV head), `out · sigmoid(gate)`, `o_proj`.
* **Every layer:** zero-centred RMSNorm (`x · rsqrt(mean x² + ε) · (1 + w)`),
  residual add, MLP `down(silu(gate(x)) · up(x))`.

The exact semantics (where HF rounds to bf16, fp32 state, `1 + w` vs `w`
norms) were read from transformers' `modeling_qwen3_5.py`, implemented in
PyTorch first (`qmk/torch_model.py`) and checked against transformers itself
before any CUDA was written for them (§10).

The CUDA side hard-codes this geometry in `csrc/abi.h`; `Config.load`
refuses a checkpoint that differs.

## 3. Execution model

One kernel launch = one decode step for the whole batch (≤ 8 rows).

```
grid = 170 blocks (one per SM), 288 threads each
  warp 8      producer: claims tasks, streams GEMM weights into smem
  warps 0-7   consumers: wait on dependencies, compute, signal
shared memory (97.7 KB): 4 × 16 KB weight ring | 31 KB task scratch (union) | barriers
```

**Tasks.** The host builds a list of ~190 K tasks (`QmkTask{op, idx, flags}`)
in topological order and an array of op descriptors (`QmkOpDesc`). A task is
one unit of one op: a GEMM tile (or one K-split of a tile), one DeltaNet head
for one batch row, or one KV-split of one attention head group.

**Claiming.** The producer lane takes tickets from a global atomic counter
(`counters[0]`) and passes task ids to the consumers through a 3-deep smem
queue guarded by `mbarrier`s. Because the ticket order is the task-list order,
tasks are claimed in topological order across the whole GPU (dynamic load
balancing, no static assignment).

**Weights ahead of data.** For a GEMM task the producer immediately starts
streaming the task's weight chunks into the ring with `cp.async.bulk`, even if
the task's inputs are not ready yet. Weights never depend on activations, so
the memory system works while consumers wait.

**Dependencies.** Each op bumps counters when units of its output are done;
each task waits until specific counters reach host-computed targets
(`QmkDep{ctr, key}`; §8). Consumer thread 0 spins with `ld.acquire.gpu` +
`__nanosleep`, then a named barrier (`bar.sync 1, 256`, consumers only)
releases the other consumer threads.

**Coherence.** L1 is not coherent across SMs, and activations are rewritten by
other SMs during the same launch, so every activation/state/KV read uses
`ld.global.cg` (L2). Writers fence (`__threadfence`) before bumping a counter.

**Why it cannot deadlock.** A task only waits on counters bumped by tasks
earlier in the list. Every earlier task has already been claimed by a running
block, and each block processes its claimed tasks in order, so the
lowest-numbered unfinished task always has all of its inputs and makes
progress. `schedule.check_order` verifies the topological property by
simulating the list sequentially (tests turn it on).

**Claim-when-idle tasks.** DeltaNet and attention tasks carry a
`QMK_TASK_WAIT_IDLE` flag: a producer only claims one when its own consumers
have finished everything they hold. Otherwise a busy block would grab a
DeltaNet head and sit on it behind a 17 µs GEMM tile while other blocks idle.
GEMM tasks are claimed eagerly (that is what starts their weight streams).

## 4. The weight-streaming GEMM

Every linear layer is `out[b, n] = x[b, :] · W[n, :]`, b < 8. With so few
rows this is a matrix-vector product per row, and the only goal is to read W
at full bandwidth while doing just enough compute.

**MMA shape.** `mma.sync.m16n8k16` (BF16 in, FP32 accumulate): the **16 rows
are weight rows**, the **8 columns are the batch**. Batch sizes below 8 waste
MMA columns, which costs nothing: compute is far from the bottleneck.

**W8A16.** Weights stay FP8 (e4m3) in memory with their 128×128 block scales;
activations stay BF16. Consumers convert FP8→BF16 in registers:

```
e4m3 byte s eeee mmm  →  bf16 bits s 0000eeee mmm0000  (shift the 7 low bits left by 4)
value = true value · 2^-120   (for normals *and* subnormals: both formats are offset the same)
one exact bf16 multiply by 2^120 finishes the conversion
```

Two bytes become one `bf16x2` with a `prmt`, a shift/mask (`lop3`) and an
`hmul2`. `test_gemm.py` checks all 254 finite e4m3 codes bit-exactly. The
block scale is applied once per 16 KB chunk to the FP32 partial sum (a warp's
16 rows × 128 k always lie inside one 128×128 scale block).

**Tiling and the 16 KB chunk.** A matrix is cut into tiles of `TN` rows
(16/32/64/128). A tile streams in 16 KB chunks; inside a chunk the 8 consumer
warps form `R = TN/16` row groups × `S = 8/R` K-slices, and each warp owns a
contiguous 2 KB block: 16 rows × 128 k (FP8) or 16 rows × 64 k (BF16). The
host pre-tiles every matrix into exactly this order (`qmk/layout.py`), so

* each chunk is one contiguous `cp.async.bulk` of 16 KB, and
* each lane reads its MMA fragment with one bank-conflict-free 16-byte `LDS`.

**K permutation.** In each 32-wide k step, lane `(g, tig)` holds physical
`k = 8·tig … 8·tig+7` for both A (weights) and B (activations). One 16-byte
load of x then feeds two MMAs, and one 16-byte load of weights covers rows g
and g+8. The same permutation is applied to A and B, so the dot product is
unchanged. The layout is a pure reshape+permute, so `untile_weight` inverts it
exactly (used by prefill and tests).

**The ring protocol.** Four 16 KB stages, each with a `full` barrier (producer
`arrive.expect_tx`, completed by the bulk copy's byte count) and an `empty`
barrier (8 arrivals, one per consumer warp). Consumers release a stage with
`fence.proxy.async` → `__syncwarp` → `mbarrier.arrive` (the fence matters,
§11).

**Consumer loop details that mattered:**
* x (and the norm weights, and the scale) for chunk c+1 are fetched while chunk
  c computes, so the L2 latency of the activations is off the per-chunk path.
* Two independent accumulator chains (even/odd k16 halves) give the MMAs some
  ILP.
* One block can consume ~31 GB/s of FP8 weights (~55 GB/s BF16): dequant + MMA
  latency with only 8 consumer warps. At full occupancy each SM needs only
  ~9.6 GB/s, so this only shows in op tails.

Measured in isolation with a working set far larger than L2, every GEMM shape
streams at **1.63–1.70 TB/s** (`tests/bench_stream.py`), above what a plain
torch reduction reaches (1.54 TB/s).

**Tile sizes** (`model.py`): QKVZ 32, AB 32 (BF16), out/o_proj 32, gate_up 32,
down 16, QKV 32, lm_head 64 (BF16). Smaller tiles mean more tasks and shorter
op tails; halving them from the first version saved ~2.5 ms/step.

## 5. Fusions

No op writes an intermediate that only the next op reads if it can be folded
into a GEMM's prologue or epilogue.

* **RMSNorm into the consumer GEMM.** The norm is applied while loading x:
  `bf16((x · inv_rms) · (1 + w))`, exactly HF's rounding. `inv_rms` per row is
  produced once (below), so a GEMM task reads 8 floats instead of a norm pass.
* **SiLU·up.** `gate_proj` and `up_proj` are fused into one matrix whose rows
  are interleaved per tile (16 gate rows, then the matching 16 up rows); the
  epilogue writes `bf16(silu(bf16(gate))) · bf16(up)` directly.
* **Residual add + next norm's statistics.** out/o_proj and down use the RESID
  epilogue: `resid = bf16(resid + bf16(acc))`, then the tile's per-row sum of
  squares. An arrival counter finds the last tile of the op; that tile sums all
  tiles' partials and writes each row's `1/rms` for the next layer's norm, and
  only then signals the op as complete.
* **Split-K with deterministic reduction.** out_proj (4 splits over head
  groups), o_proj (4 splits = 4 KV-head groups) and down_proj (2 splits over
  the hidden state) write FP32 partials; a per-tile arrival counter picks the
  last split, which sums the partials **in split order** (bit-reproducible) and
  runs the epilogue.

## 6. Gated DeltaNet decode task

One task = one value head × one batch row (48 × bs tasks per layer), 256
threads, ~7 µs. Order is head-major (`idx = h·bs + b`) so the first group of
heads finishes first.

1. **Conv update.** The task needs its key head's q and k channels (shared by
   3 value heads) and its own v channels: 384 channels, 4 taps,
   `silu(bf16(conv))`. The conv state is **double-buffered by step parity**:
   tasks read taps from buffer `p` and write the shifted taps to `p^1`, so the
   three heads sharing q/k never race; only `h % 3 == 0` writes the q/k taps.
2. **Gates.** l2-normalise q and k, `beta = bf16(sigmoid(b))`,
   `g = -exp(A_log) · softplus(a + dt_bias)`.
3. **State update.** The 128×128 FP32 state (64 KB) is processed in two passes
   of 64 columns; each thread owns one column and 32 rows, so only 32 floats
   live in registers. The 9-warp block is capped at 168 registers per thread
   (one SM sub-partition holds 3 warps), and the first version, with 64 state
   values per thread, spilled. Per column:
   `S *= exp(g); delta = (v − Sᵀk)·beta; S += k·deltaᵀ; o = Sᵀq`.
4. **Gated RMSNorm** over the head's 128 outputs and `· silu(z)`.

The task issues `cp.async.bulk.prefetch.L2` for its 64 KB state at the start,
so the conv/gate phases hide the DRAM latency of the state passes.

## 7. Gated attention decode task

One task = one (KV head g, batch row b, KV split s). The number of splits is
chosen per step (power of two, ~one task per SM, ≥ 32 positions per split).

1. Warps 0-5 normalise and rotate the 6 query heads (zero-centred RMSNorm,
   partial RoPE with bf16 rounding at each op as HF does; cos/sin tables built
   exactly like HF's rotary module). The split that contains the new position
   also normalises/rotates k and writes the new K/V into the cache; no other
   split reads that position.
2. Each warp walks positions `p0 + warp, p0 + warp + 8, …` with an online
   softmax in FP32 (lanes hold 8 of the 256 dims; scores reduced with
   shuffles). Attention compute is small next to KV bandwidth, so CUDA cores
   suffice.
3. The 8 warps are merged with a 3-round tree in shared memory, the split's
   (m, l, acc) is written to global, and the **last split to finish** (atomic
   arrival counter per (b, g)) combines all splits, applies `· sigmoid(gate)`
   and writes the output. Only that task signals.

## 8. The schedule

`qmk/schedule.py` builds, for a given (batch size, split count), the op
descriptors, the task list and every counter's completion target:

```
DeltaNet layer:   AB, QKVZ  →  GDN  →  out_proj(+resid)  →  gate_up  →  down(+resid)
attention layer:  QKV  →  ATTN  →  o_proj(+resid)  →  gate_up  →  down(+resid)
then lm_head
```

**Keyed counters.** A dependency is `{ctr, key}`; the counter a task waits on
is `ctr + key(task index)` with
`key(i) = min(((i / div1) % mod) / div2, kmax)`, and it waits until that
counter reaches `targets[...]` (computed by the host by enumerating which
units bump it). The same mechanism decides which counter an op bumps. This
expresses every dependency in the model without per-task tables:

| consumer | waits on |
|---|---|
| GDN head h | the 32 QKVZ tiles of its key head (rows are regrouped per key head: q, k, 3 v, 3 z) + all of AB |
| out_proj K-split j | GDN heads 12j … 12j+11 (all batch rows) |
| attention group g | the 112 QKV tiles of KV head g (rows regrouped per KV head) |
| o_proj K-split g | attention group g |
| down K-split j | the gate_up tiles producing hidden features of half j |
| gate_up, next layer's QKVZ/AB/QKV, lm_head | the whole previous residual op (true barrier: the norm needs the full row) |

The row regrouping of the fused QKVZ/QKV matrices is a permutation stored with
the weights (`TiledLinear.inv_perm`), undone transparently for prefill. Only
the two RMSNorm points per layer remain full barriers.

## 9. Host side

* **Weights** (`qmk/model.py`): read from safetensors to the GPU, fused
  (QKVZ, AB, QKV, gate/up), row-permuted, tiled, with per-16-row-group scales
  in kernel order. The BF16 embedding table stays **on the CPU**
  (a per-token row lookup), which is what frees the memory for 8 slots of
  DeltaNet state. VRAM after loading: 27.0 GB, ~4 GB free.
* **Caches** (`qmk/cache.py`): K/V `[full layer, slot, kv head, pos, 256]`
  BF16, DeltaNet state `[lin layer, slot, 48, 128, 128]` FP32 (150 MB per
  slot), conv state `[lin layer, slot, parity, 3 taps, 10240]`.
* **Prefill** (`qmk/torch_model.py`): plain PyTorch, chunked, dequantizing
  weights block by block; DeltaNet via flash-linear-attention's chunked
  kernel, attention via SDPA. It fills the caches the kernel continues from.
* **Engine** (`qmk/engine.py`): per step, upload the batch's embeddings and
  their `1/rms`, zero the counters, launch, return logits. Schedules are cached
  per (bs, splits). `generate.py` adds chat templating, sampling and streaming.
* **Build** (`qmk/native.py`): nvcc from `/mnt/data/envs/cuda13` with the
  system g++ (`-arch=sm_120a`), cached by source hash; the ctypes mirror of
  `abi.h` is checked against `offsetof`/`sizeof` values exported by the
  library.

## 10. Correctness strategy

Layered, each layer checked against the one below:

1. `test_torch_vs_hf.py`: our PyTorch model vs transformers'
   `Qwen3_5ForCausalLM` with the same dequantized weights on 4 real layers
   (prefill, chunked prefill, cached decode step): ~2e-3 relative, same top-5.
2. `test_gemm.py`: every GEMM path (FP8/BF16, all tile shapes, norm, SiLU,
   RESID with `1/rms`, split-K 2/3/4 incl. uneven chunk splits) vs FP32 torch;
   FP8 conversion exhaustive.
3. `test_decode.py`: the megakernel step vs an FP32-linear PyTorch reference on
   cloned caches, logits and all caches, batch 1-8, multiple steps, contexts up
   to ~570: ~1e-3 on 4-8 layers. Slot 0 is left unused to catch slot-indexing
   bugs; schedule order validation is on.
4. **Full model, noise floor.** Over 64 layers bf16 rounding differences
   compound to 2-6 % relative logit differences, so a fixed tolerance is
   meaningless. `--noise-floor` runs a second correct reference (BF16 linears)
   and requires the kernel to be within 1.5× of the reference-vs-reference
   distance; it is ~1.0×, with the same argmax at every step.

## 11. sm_120 traps and bugs found

* **`cp.async.bulk.shared::cluster` is a driver call on sm_120.** It compiles
  to `CALL.ABS` into a routine that needs ~14 KB of stack per thread, so the
  first launch raised the stack limit and reserved **3.6 GB** of VRAM for local
  memory, which made the full model OOM at launch. `.shared::cta` compiles to
  a native `UBLKCP`. (Found by bisecting kernel features, then `cuobjdump`.)
* **Missing `fence.proxy.async` in the ring.** Consumers read a stage with
  generic-proxy `LDS`, then the producer's next bulk copy (async proxy) writes
  it. Without a proxy fence between them, some tiles occasionally used
  half-overwritten weights: results were nondeterministic, visible only as a
  small accuracy loss at larger batch. Found by running the same launch 20×
  and diffing, then bisecting op combinations.
* **Register cap.** 9 warps per block → 3 warps on one sub-partition → 168
  registers per thread; the DeltaNet task had to be restructured (§6).
* **Toolchain.** The conda env's GCC 15 headers break nvcc; use the system
  g++. Triton (used by fla in prefill) needs Python headers that the system
  Python lacks; they are unpacked in `/mnt/data/envs/py312-dev-headers`.
* **Shared GPU.** The 5090 also drives the desktop; graphics work preempts the
  kernel for 0.3-0.5 ms a few times per step. Compare variants back to back;
  `bench.py` reports the minimum as well as the median.

## 12. Performance

bs=1, ctx 512, in-kernel time per step (each line is a measured change):

| version | ms/step | what changed |
|---|---:|---|
| first working full model | 24.8 | coarse op-level barriers, larger tiles |
| halve tile sizes | 22.3 | shorter op tails |
| DeltaNet/attn claimed only by idle blocks; ~1 attention task per SM | 21.2 | no stranded GDN tasks |
| keyed fine-grained deps, split-K, regrouped QKVZ/QKV | 20.8 | next op starts on finished parts |
| ring proxy fence (correctness), last tile computes `1/rms`, GDN state L2 prefetch, out_proj 4-way split | ~20.1 | norm prologue was 320 partial loads per task |

Tools that found these: in-kernel per-task timestamps
(`decode_step(profile=True)`), `bench.py --profile`, `tests/timeline.py`
(per-op ready/done spans), `tests/bwtrace.py` (DRAM bandwidth over time, from
task timestamps), `tests/bench_chain.py` (cost of a barrier in isolation:
2-14 µs).

**Where the remaining ~4 ms goes** (floor ≈ 16.3 ms): per layer, the hand-off
from QKVZ/QKV to out_proj while DeltaNet/attention runs (~10-15 µs, DRAM nearly
idle), the two RMSNorm barriers (~5 µs each), and ramp-up at the start of
QKVZ/QKV.

**Tried and dropped (measured slower):**

* *Interleaving DeltaNet/attention tasks (and out_proj splits) into the QKVZ
  stream* so they run before QKVZ ends: 21.8-22.9 ms vs ~20.5. Tasks claimed a
  wave after their inputs often still waited on them while holding a block, and
  DeltaNet's state loads crawl under full DRAM load.
* *L2 prefetch of GEMM weights* (`cp.async.bulk.prefetch.L2` beyond the ring),
  always or only for tasks whose inputs are not ready: no gain or slower in
  every variant.

**Ideas not yet tried:** applying the RMSNorm scale after the matmul so the
norm-consuming GEMMs can split K over residual columns and start before the
residual is complete (removes the two barriers); a lower-latency DeltaNet task
(two half-head tasks); a faster FP8 consumer (Marlin-style dequant, or FP8 MMA
with quantized activations, which would change numerics to W8A8).

## 13. Benchmarking against vLLM

`baseline/vllm_decode.py` measures vLLM 0.30.0 on the same checkpoint and
shapes, decode only: TPOT = (t(160 new tokens) − t(32)) / 128, so prefill and
the chunked-prefill overlap cancel; GPU time per step is taken from an Nsight
Systems trace.

**The baseline mistake.** The first baseline ran vLLM *without* torch.compile,
because Inductor's compile-time autotuning ran out of memory next to the
weights. That left ~1,470 unfused elementwise kernels per step and made the
megakernel look 1.13× faster at bs=1 (and 2.3× at bs=8, which was really vLLM
running out of memory). A reviewer flagged the number as wrong; it was. The
fix keeps torch.compile on and avoids the OOM with lazy autotuning
(`triton.autotune_at_compile_time=False`) and no combo-kernel benchmarking.

**Corrected:** compiled vLLM decodes in 21.1 ms of GPU time per step at bs=1
(17.4 ms of it FP8 GEMMs); the megakernel takes 20.1 ms → **~1.05×**; ~1.11×
at bs=2. Batch > 2 cannot be compared fairly on this machine yet: with the
desktop holding ~1.9 GB, compiled vLLM fits ~2 sequences (its embedding table
is on the GPU and each sequence needs ~150 MB of FP32 DeltaNet state).

Other vLLM settings needed here: `CUDA_HOME` pointing at a toolkit (it
JIT-compiles kernels), `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer's sampler
fails to link), `limit_mm_per_prompt` 0 (text only).

## 14. Decisions

User's decisions: target the RTX 5090; switch from North Mini Code to
Qwen3.8-27B-FP8; implementation before the vLLM baseline; decode is what to
benchmark; publish on the fork `noviorlu/cohere-megakernel`.

Mine (reasonable defaults, open to change): W8A16 numerics (BF16
activations) instead of vLLM's W8A8; a new `qwen/` subproject instead of
modifying the H100 code; no ThunderKittens on sm_120; embedding table on the
CPU; text-only (no vision tower, no MTP head); batch ≤ 8; PyTorch prefill;
the specific tile sizes and split counts (all measured).

## 15. File map

| path | role |
|---|---|
| `csrc/abi.h` | geometry constants, op/task/step structs shared with Python |
| `csrc/ptx.cuh` | PTX wrappers: mbarrier, bulk copy, MMA, FP8→BF16, loads |
| `csrc/gemm.cuh` | weight streaming (producer) and the GEMM task with all epilogues |
| `csrc/gdn.cuh` | Gated DeltaNet decode task |
| `csrc/attn.cuh` | gated attention decode task |
| `csrc/megakernel.cu` | the persistent kernel: claiming, dependencies, dispatch, C API |
| `qmk/layout.py` | weight tiling / untiling, scale layout, dequant |
| `qmk/model.py` | config, weight loading, fusion and row permutations, tile sizes |
| `qmk/schedule.py` | ops, keyed counters, targets, task order, order checker |
| `qmk/engine.py` | caches + buffers + schedules, one launch per decode step |
| `qmk/torch_model.py` | PyTorch prefill and the reference decode step |
| `qmk/cache.py`, `qmk/sampling.py`, `qmk/native.py` | caches, sampling, build + ctypes ABI |
| `generate.py`, `bench.py` | chat CLI, decode benchmark / profiler |
| `baseline/vllm_decode.py` | vLLM decode baseline |
| `tests/` | correctness tests, microbenchmarks, timeline / bandwidth traces |
