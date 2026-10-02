"""Chat with Qwen3.8-27B-FP8 through the decode megakernel.

  .venv/bin/python qwen/generate.py -p "Write a haiku about GPUs."
  .venv/bin/python qwen/generate.py -p "Q1" -p "Q2" -p "Q3" --temperature 1.0   # batched
  .venv/bin/python qwen/generate.py -p "..." --no-think                         # skip the thinking block

Prefill runs in PyTorch, decode is one megakernel launch per token for the
whole batch. A single prompt is streamed; a batch is printed when done.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qmk.engine import Engine  # noqa: E402
from qmk.model import Weights  # noqa: E402
from qmk.sampling import sample  # noqa: E402

CKPT = Path(__file__).resolve().parents[1] / "models" / "Qwen3.8-27B-FP8"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prompt", action="append", required=True, help="user message (repeat for a batch, <= 8)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-think", action="store_true", help="chat template with enable_thinking=False")
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    args = ap.parse_args()
    if len(args.prompt) > 8:
        ap.error("at most 8 prompts")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    t0 = time.time()
    w = Weights.load(args.ckpt, torch.device("cuda"), log=None)
    eng = Engine(w, slots=len(args.prompt), max_ctx=args.max_ctx)
    print(f"[loaded in {time.time() - t0:.0f} s]", file=sys.stderr)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    eos = set(w.cfg.eos_token_ids)

    def pick(logits: torch.Tensor) -> list[int]:
        return sample(logits, args.temperature, args.top_k, args.top_p, gen)

    # ── prefill ──
    tmpl = {"enable_thinking": False} if args.no_think else {}
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       tokenize=False, **tmpl)).input_ids for p in args.prompt]
    torch.cuda.synchronize()
    t0 = time.time()
    first = [pick(eng.prefill(x, slot)[None])[0] for slot, x in enumerate(ids)]
    torch.cuda.synchronize()
    t_prefill = time.time() - t0

    # ── decode ──
    out: list[list[int]] = [[t] for t in first]
    active = [s for s in range(len(ids)) if first[s] not in eos]
    stream = len(ids) == 1
    shown = ""
    if stream:
        shown = tok.decode(out[0], skip_special_tokens=False)
        print(shown, end="", flush=True)
    steps = 0
    t0 = time.time()
    while active and max(len(o) for o in out) < args.max_new_tokens:
        if max(eng.cache.length[s] for s in active) >= args.max_ctx:
            break
        nxt = pick(eng.decode_step([out[s][-1] for s in active], active))
        steps += 1
        for s, t in zip(active, nxt):
            out[s].append(t)
        active = [s for s in active if out[s][-1] not in eos and len(out[s]) < args.max_new_tokens]
        if stream:
            text = tok.decode(out[0], skip_special_tokens=False)
            print(text[len(shown):], end="", flush=True)
            shown = text
    torch.cuda.synchronize()
    t_dec = time.time() - t0

    if not stream:
        for s, o in enumerate(out):
            print(f"\n===== [{s}] {args.prompt[s]}\n{tok.decode(o, skip_special_tokens=False)}")
    n_new = sum(len(o) - 1 for o in out)
    print(f"\n[prefill {sum(map(len, ids))} tok in {t_prefill:.2f} s | decode {n_new} tok in {steps} steps, "
          f"{t_dec:.2f} s → {n_new / max(t_dec, 1e-9):.1f} tok/s, {t_dec / max(steps, 1) * 1e3:.1f} ms/step]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
