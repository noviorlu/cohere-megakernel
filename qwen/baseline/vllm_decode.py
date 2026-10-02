"""vLLM baseline: decode TPOT for Qwen3.8-27B-FP8 on the same shapes as qwen/bench.py.

  /mnt/data/envs/vllm/bin/python qwen/baseline/vllm_decode.py --bs 1 --ctx 512

TPOT = (latency of N new tokens - latency of N0 new tokens) / (N - N0), with
random-token prompts of length ctx, greedy, ignore_eos. N0 is past the point
where every request has finished its (chunked) prefill, so the difference is
steady-state batched decode only.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

CKPT = Path(__file__).resolve().parents[2] / "models" / "Qwen3.8-27B-FP8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[1])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--new", type=int, default=160)
    ap.add_argument("--new0", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    args = ap.parse_args()
    import random

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(str(CKPT), max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=max(args.bs),
              enable_prefix_caching=False, max_num_batched_tokens=512, enable_chunked_prefill=True,
              # torch.compile's autotuner clones a 2.4 GiB input, which does not fit next to
              # the 27.6 GiB of weights; run uncompiled but with full CUDA graphs for decode.
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    rng = random.Random(0)

    def run(bs: int, n: int) -> float:
        prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 200000) for _ in range(args.ctx)])
                   for _ in range(bs)]
        sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
        t = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t

    for bs in args.bs:
        run(bs, 8)  # warm-up
        t0 = min(run(bs, args.new0) for _ in range(3))
        tn = min(run(bs, args.new) for _ in range(3))
        tpot = (tn - t0) / (args.new - args.new0)
        print(f"vLLM bs={bs} ctx≈{args.ctx}: {tpot * 1e3:.2f} ms/step → {bs / tpot:.1f} tok/s")


if __name__ == "__main__":
    main()
