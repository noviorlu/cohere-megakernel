"""Megakernel decode step vs the fp32-linear torch reference, on the real checkpoint.

Prefills several slots with different prompts, then runs N decode steps with
both the megakernel and TorchModel(precise=True) on a cloned cache, comparing
logits each step and the cache contents at the end.
Run: .venv/bin/python qwen/tests/test_decode.py [--layers 4] [--bs 3] [--steps 4]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmk.engine import Engine  # noqa: E402
from qmk.model import Weights  # noqa: E402
from qmk.torch_model import TorchModel  # noqa: E402

CKPT = Path(__file__).resolve().parents[2] / "models" / "Qwen3.8-27B-FP8"
DEV = torch.device("cuda")
PROMPTS = [
    "Write a haiku about GPUs and memory bandwidth.",
    "What is 17 * 23? Answer briefly.",
    "List three prime numbers greater than 100.",
    "Explain what a persistent CUDA kernel is in one sentence.",
    "Translate 'good morning' into French, German and Japanese.",
    "Name the planets of the solar system.",
    "Why is the sky blue?",
    "Give me a Python one-liner that reverses a string.",
]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--bs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--noise-floor", action="store_true", help="compare against a bf16-linear reference too")
    ap.add_argument("--repeat-prompt", type=int, default=1, help="repeat prompts to get longer contexts")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    w = Weights.load(CKPT, DEV, layers=args.layers, log=None)
    eng = Engine(w, slots=args.bs + 1, max_ctx=args.max_ctx)
    slots = list(range(1, args.bs + 1))  # leave slot 0 unused to catch slot-indexing bugs
    tokens = []
    for b, s in enumerate(slots):
        msg = " ".join([PROMPTS[b % len(PROMPTS)]] * args.repeat_prompt)
        text = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False)
        ids = tok(text).input_ids
        logits = eng.prefill(ids, s)
        tokens.append(int(logits.argmax()))
    print(f"layers={args.layers} bs={args.bs} ctx={[eng.cache.length[s] for s in slots]}")

    ref_cache = eng.cache.clone()
    ref = TorchModel(w, ref_cache, precise=True)
    # A second correct implementation (bf16 linears) gives the noise floor: bf16
    # rounding differences compound over many layers, so a fixed tolerance only
    # works for shallow models.
    alt = TorchModel(w, ref_cache.clone(), precise=False) if args.noise_floor else None
    worst = 0.0
    worst_ratio = 0.0
    for step in range(args.steps):
        nsplit = eng.nsplit_for(args.bs, max(eng.cache.length[s] for s in slots) + 1)
        torch.cuda.synchronize()
        t0 = time.time()
        got = eng.decode_step(tokens, slots).float()
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1e3
        want = ref.decode_step(tokens, slots)
        r = rel(got, want)
        worst = max(worst, r)
        agree = (got.argmax(-1) == want.argmax(-1)).tolist()
        floor = ""
        if alt is not None:
            nf = rel(alt.decode_step(tokens, slots), want)
            worst_ratio = max(worst_ratio, r / nf)
            floor = f"  (noise floor {nf:.2e})"
        print(f"step {step}: nsplit={nsplit} {dt:7.2f} ms  logits rel L2 {r:.2e}{floor}  argmax agree {agree}")
        tokens = want.argmax(-1).tolist()

    c, rc = eng.cache, ref_cache
    for name in ("state", "conv"):
        print(f"cache.{name}: rel L2 {rel(getattr(c, name), getattr(rc, name)):.2e}")
    for name in ("k", "v"):
        print(f"cache.{name}: rel L2 {rel(getattr(c, name), getattr(rc, name)):.2e}")
    if alt is not None:
        assert worst_ratio < 1.5, f"kernel error {worst_ratio:.2f}x the noise floor"
    else:
        assert worst < 3e-2, f"logits mismatch (worst rel {worst:.3g})"
    print("OK")


if __name__ == "__main__":
    main()
