"""Decode throughput of the megakernel on Qwen3.8-27B-FP8.

  .venv/bin/python qwen/bench.py --bs 1 --ctx 1024 --steps 32 [--profile]

Prefills `bs` slots with random tokens to `ctx`, then times decode steps:
kernel-only (CUDA events around the launch) and end-to-end per step
(embedding upload, launch, logits). --profile prints a per-op breakdown from
the in-kernel task timestamps.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qmk.engine import Engine  # noqa: E402
from qmk.model import Weights  # noqa: E402

CKPT = Path(__file__).resolve().parents[1] / "models" / "Qwen3.8-27B-FP8"


def op_kind(name: str) -> str:
    return name.split(".")[-1]


def profile_report(eng: Engine, bw_gbs: float = 1650.0) -> None:
    """Per op kind: median per-layer span vs. the time its weight bytes need at `bw_gbs`."""
    prof = eng.last_profile.cpu()
    sched = eng.last_schedule
    op_of_task = sched.tasks[:, 0].cpu()
    total = (prof[:, 2].max() - prof[:, 0].min()).item() / 1e3
    weights = {}
    for i, lw in enumerate(eng.w.layers):
        for name, t in vars(lw).items():
            if hasattr(t, "tiling"):
                weights[f"L{i}.{name}"] = t.w.numel() * t.w.element_size()
    weights["lm_head"] = eng.w.lm_head.w.numel() * 2
    kinds = defaultdict(list)  # kind -> [(span_us, ideal_us)]
    for op, name in enumerate(sched.op_names):
        sel = prof[op_of_task == op]
        span = (sel[:, 2].max() - sel[:, 1].min()).item() / 1e3
        ideal = weights.get(name, 0) / bw_gbs / 1e3
        kinds[op_kind(name)].append((span, ideal))
    print(f"\nin-kernel step {total:.0f} us; per op kind (median over layers; ideal = weight bytes at {bw_gbs:.0f} GB/s)")
    print(f"{'op':10s} {'n':>4s} {'span':>8s} {'ideal':>8s} {'loss':>8s} {'loss x n':>9s} {'max span':>9s}")
    tot_loss = 0.0
    for name, rows in sorted(kinds.items(), key=lambda kv: -sum(r[0] for r in kv[1])):
        spans = torch.tensor([r[0] for r in rows])
        ideal = torch.tensor([r[1] for r in rows]).median().item()
        med = spans.median().item()
        loss = (med - ideal) * len(rows)
        tot_loss += loss
        print(f"{name:10s} {len(rows):4d} {med:8.1f} {ideal:8.1f} {med - ideal:8.1f} {loss:9.0f} {spans.max().item():9.1f}")
    print(f"sum of median losses {tot_loss:.0f} us (ab overlaps qkvz; gdn/attn move no weights)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    w = Weights.load(CKPT, torch.device("cuda"), log=None)
    eng = Engine(w, slots=args.bs, max_ctx=args.max_ctx)
    gen = torch.Generator().manual_seed(0)
    slots = list(range(args.bs))
    for s in slots:
        eng.prefill(torch.randint(0, 200000, (args.ctx,), generator=gen).tolist(), s)
    tokens = [1000 + s for s in slots]

    for _ in range(3):  # warm-up (schedule build)
        eng.decode_step(tokens, slots)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    kern, wall = [], []
    for _ in range(args.steps):
        t = time.perf_counter()
        e0.record()
        logits = eng.decode_step(tokens, slots)
        e1.record()
        tokens = logits.argmax(-1).tolist()  # syncs, like real sampling
        wall.append((time.perf_counter() - t) * 1e3)
        kern.append(e0.elapsed_time(e1))
    k = sorted(kern)[len(kern) // 2]
    wl = sorted(wall)[len(wall) // 2]
    # The GPU also drives the desktop; graphics preemption only ever adds time,
    # so the minimum is the cleanest view of the kernel itself.
    print(f"bs={args.bs} ctx≈{args.ctx}: kernel {k:.2f} ms/step (min {min(kern):.2f}), end-to-end {wl:.2f} ms/step "
          f"→ {args.bs * 1e3 / wl:.1f} tok/s (kernel-only {args.bs * 1e3 / k:.1f} tok/s)")
    weight_bytes = sum(t.w.numel() * t.w.element_size() for lw in w.layers
                       for t in vars(lw).values() if hasattr(t, "tiling")) + w.lm_head.w.numel() * 2
    print(f"weights streamed per step {weight_bytes / 1e9:.2f} GB → {weight_bytes / k / 1e6:.0f} GB/s effective")
    if args.profile:
        eng.decode_step(tokens, slots, profile=True)
        torch.cuda.synchronize()
        profile_report(eng)


if __name__ == "__main__":
    main()
