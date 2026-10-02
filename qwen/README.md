# Qwen3.8-27B-FP8 decode megakernel for RTX 5090 (sm_120)

A single persistent CUDA kernel runs one whole decode step of
[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (text
decoder: 48 Gated DeltaNet + 16 gated full-attention layers, dense MLP,
248K vocab) on a 32 GB RTX 5090. Prefill runs in PyTorch.

The design follows the Cohere megakernel in this repository (one block per SM,
a task list, dependency counters in global memory), but the code is new: the
H100 version is built on WGMMA/TMA and ThunderKittens' Hopper path, none of
which exist on consumer Blackwell.

## Results (RTX 5090, ctx ≈ 512, greedy decode)

Decode only (prefill is done beforehand and not timed), measured back to back
on the same machine. Megakernel: median / min GPU time of one decode step for
the whole batch (CUDA events). vLLM 0.30.0 with its default torch.compile +
CUDA graphs: GPU time per step from an Nsight Systems trace, and wall-clock
TPOT = (t(160 tokens) − t(32 tokens)) / 128 from `baseline/vllm_decode.py`.

| batch | megakernel GPU step | vLLM GPU step (nsys) | vLLM TPOT (wall) | speedup |
|---:|---:|---:|---:|---:|
| 1 | 20.1 ms (min 19.1) → 49.7 tok/s | 21.1 ms (min 19.9) | 21.0 ms → 47.6 tok/s | ~1.05× |
| 2 | 20.3 ms (min 19.0) → 98.4 tok/s | — | 22.6 ms → 88.5 tok/s | ~1.11× |

vLLM's step is GPU-bound: 17.4 ms of its 20.9 ms of kernel time is the FP8
block-scaled GEMMs (CUTLASS, W8A8 with per-token-group activation quant).

**Correction:** an earlier version of this table claimed 1.13× / 1.21× /
2.3× at bs 1 / 4 / 8. That vLLM baseline was wrong: it ran *without*
torch.compile (Inductor's compile-time autotuning ran out of memory next to the
weights), which leaves ~1,470 unfused elementwise kernels per step, and its
bs=8 number was dominated by memory pressure. `vllm_decode.py` now keeps
torch.compile on and avoids the OOM with lazy autotuning and no combo-kernel
benchmarking (`--no-compile` reproduces the old run).

**Batch > 2 is not measured yet.** With the desktop holding ~1.9 GB of VRAM,
compiled vLLM fits only ~2.25 concurrent sequences (27.6 GiB of weights incl.
the 2.4 GiB embedding table, plus ~150 MB of fp32 DeltaNet state per
sequence). The megakernel fits 8 (embedding table on the CPU) and runs bs=8 at
22.2 ms/step (360 tok/s), but there is no fair vLLM number to compare yet;
that needs the desktop's VRAM freed.

The GPU also drives the desktop, which preempts compute now and then (+~1
ms/step on some runs), hence the minimums.

Floor: the step streams 26.9 GB of weights; at the ~1.65 TB/s the kernel
reaches in steady state that is ~16.3 ms, so bs=1 runs at ~80% of it. Other
data points: 8K context costs ~+0.9 ms/step at bs=1.

Accuracy: the full 64-layer model matches an fp32-linear PyTorch reference as
closely as two correct references (fp32 vs bf16 linears) match each other,
with the same top token at every step (`tests/test_decode.py --noise-floor`);
4-8 layer runs match to ~1e-3. The PyTorch implementation itself matches
transformers' `Qwen3_5ForCausalLM` (`tests/test_torch_vs_hf.py`).

## Usage

```bash
# environment: repo-root .venv (torch cu130, transformers>=5.8, flash-linear-attention);
# nvcc from /mnt/data/envs/cuda13 with the system g++ (qmk/native.py builds on first use)
.venv/bin/python qwen/generate.py -p "Write a haiku about GPUs."             # streams
.venv/bin/python qwen/generate.py -p "Q1" -p "Q2" --temperature 1.0         # batch ≤ 8
.venv/bin/python qwen/bench.py --bs 1 --ctx 512 --profile                   # decode timing
```

Weights: `models/Qwen3.8-27B-FP8` (symlink to /mnt/data/models). Only the text
decoder is loaded; the embedding table stays on the CPU. After loading, ~4 GB
of VRAM is left for KV cache (64 KB/token) and DeltaNet state (150 MB/sequence).

## How it works

The full write-up — design, the sm_120 traps, bugs found, optimization
history and dropped ideas — is in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).
Short version:

* **Block:** 1 producer warp + 8 consumer warps per SM, 170 blocks.
  The producer claims tasks from a global ticket and streams GEMM weights into
  a 4 × 16 KiB shared-memory ring with `cp.async.bulk`; consumers wait on the
  task's dependency counters, compute, then bump the op's counter.
* **GEMM (W8A16):** weights stay FP8 with their 128×128 block scales, tiled on
  the host so each 16 KiB chunk is contiguous and in `mma.m16n8k16` fragment
  order (`qmk/layout.py`). FP8→BF16 is a bit-shift plus one exact `hmul2`; the
  batch (≤ 8) is the MMA's N dimension. RMSNorm, SiLU·up, residual add and
  split-K reduction are fused into the GEMM tasks.
* **Dependencies are as fine as the data allows:** fused QKVZ rows are grouped
  per key head so a DeltaNet head waits only for its own tiles; out/o_proj are
  K-split by head group; down_proj is K-split by halves of the hidden state.
  Only the two RMSNorms per layer are full barriers; the last residual tile
  computes each row's 1/rms for the next layer.
* **Tasks:** `csrc/gemm.cuh`, `csrc/gdn.cuh` (conv + gated delta rule + gated
  norm, one head × row per task), `csrc/attn.cuh` (split-KV attention with
  partial RoPE and output gate), scheduled by `qmk/schedule.py`, which also
  checks the task order is topological (that makes the kernel deadlock-free).

## Tests

```bash
.venv/bin/python qwen/tests/test_gemm.py                    # GEMM paths, split-K, FP8 conversion
.venv/bin/python qwen/tests/test_torch_vs_hf.py --layers 4  # torch model vs transformers
.venv/bin/python qwen/tests/test_decode.py --layers 8 --bs 8
.venv/bin/python qwen/tests/test_decode.py --layers 64 --bs 2 --max-ctx 512 --noise-floor
```

`tests/bwtrace.py` and `tests/timeline.py` print in-kernel bandwidth and op
timelines for one layer.

## Known limitations / next steps

* Decode is ~20% above the bandwidth floor and only ~5% faster than compiled
  vLLM at bs=1. Remaining holes per layer: the
  hand-off from QKVZ/QKV to out_proj while DeltaNet / attention runs
  (~10–15 µs) and the two RMSNorm barriers (~5 µs each). Tried and dropped
  (slower): interleaving DeltaNet/attention tasks into the QKVZ stream, and L2
  prefetch of GEMM weights.
* No server, no prefix caching, no continuous batching; slots are fixed.
* Prefill is plain PyTorch with per-call FP8 dequantization (~1.6 s for a short
  prompt).
* The kernel hard-codes the Qwen3.8-27B geometry (`csrc/abi.h`).
