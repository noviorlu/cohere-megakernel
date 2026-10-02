"""Per-op timeline of one decode step (in-kernel task timestamps)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmk.engine import Engine
from qmk.model import Weights
CKPT = Path(__file__).resolve().parents[2] / "models" / "Qwen3.8-27B-FP8"

def main(bs=1, ctx=512, layers=(8, 9, 10, 11)):
    w = Weights.load(CKPT, torch.device("cuda"), log=None)
    eng = Engine(w, slots=bs, max_ctx=2048)
    for s in range(bs):
        eng.prefill(list(range(1000, 1000 + ctx)), s)
    toks = [5] * bs
    for _ in range(3): eng.decode_step(toks, list(range(bs)))
    eng.decode_step(toks, list(range(bs)), profile=True); torch.cuda.synchronize()
    prof, sched = eng.last_profile.cpu(), eng.last_schedule
    opi = sched.tasks[:, 0].cpu()
    t0 = prof[:, 0].min().item()
    us = lambda v: (v - t0) / 1e3
    print(f"{'op':14s} {'tasks':>5s} {'claim0':>8s} {'ready0':>8s} {'ready50':>8s} {'readyMax':>8s} {'done0':>8s} {'doneMax':>8s} {'task us med':>11s} {'SMs':>4s}")
    for op, name in enumerate(sched.op_names):
        if not any(name.startswith(f"L{l}.") for l in layers): continue
        sel = prof[opi == op]
        ready = sel[:, 1].sort().values
        d = ((sel[:, 2] - sel[:, 1]).float() / 1e3)
        dur = d.median().item()
        q = d.quantile(torch.tensor([0.1, 0.9])).tolist() if d.numel() > 1 else [dur, dur]
        print(f"{name:14s} {sel.shape[0]:5d} {us(sel[:,0].min().item()):8.1f} {us(ready[0].item()):8.1f} {us(ready[len(ready)//2].item()):8.1f} "
              f"{us(ready[-1].item()):8.1f} {us(sel[:,2].min().item()):8.1f} {us(sel[:,2].max().item()):8.1f} {dur:11.1f} {sel[:,3].unique().numel():4d}  p10/p90 {q[0]:.1f}/{q[1]:.1f}")

if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:3]])
