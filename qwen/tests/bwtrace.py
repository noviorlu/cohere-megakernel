"""DRAM-bandwidth timeline of one decode step from in-kernel task timestamps.

Each GEMM task's weight bytes are spread uniformly over [deps ready, done];
summing over tasks in 5 us bins shows where the memory system idles.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmk.engine import Engine
from qmk.model import Weights
from qmk import native
CKPT = Path(__file__).resolve().parents[2] / "models" / "Qwen3.8-27B-FP8"

def trace(eng, layers, bin_us=5.0):
    prof, sched = eng.last_profile.cpu().double(), eng.last_schedule
    opi = sched.tasks[:, 0].cpu()
    raw = sched.ops.cpu().numpy().tobytes()
    ops = (native.OpDesc * (len(raw) // native.C.sizeof(native.OpDesc))).from_buffer_copy(raw)
    bytes_per_task = torch.zeros(len(opi), dtype=torch.float64)
    for k, op in enumerate(ops):
        if op.type in (native.OP_GEMM_FP8, native.OP_GEMM_BF16):
            g = op.gemm
            bytes_per_task[opi == k] = g.nchunks * 16384 / g.ksplit
    sel_ops = [k for k, n in enumerate(sched.op_names) if any(n.startswith(f"L{l}.") for l in layers)]
    mask = torch.isin(opi, torch.tensor(sel_ops))
    t0 = prof[mask, 1].min(); t1 = prof[mask, 2].max()
    nb = int((t1 - t0) / 1e3 / bin_us) + 1
    bw = torch.zeros(nb, dtype=torch.float64)
    active = [set() for _ in range(nb)]
    for i in torch.nonzero(mask).flatten().tolist():
        a, b = (prof[i, 1] - t0) / 1e3, (prof[i, 2] - t0) / 1e3
        if b <= a: continue
        rate = bytes_per_task[i] / (b - a)  # bytes per us
        for j in range(int(a // bin_us), int(b // bin_us) + 1):
            lo, hi = max(a, j * bin_us), min(b, (j + 1) * bin_us)
            if hi > lo and j < nb:
                bw[j] += rate * (hi - lo)
                active[j].add(sched.op_names[opi[i]].split('.')[-1])
    for j in range(nb):
        gbs = bw[j] / bin_us / 1e3
        bar = '#' * int(gbs / 50)
        print(f"{j * bin_us:6.0f} us {gbs:6.0f} GB/s {bar:34s} {','.join(sorted(active[j]))}")

if __name__ == "__main__":
    w = Weights.load(CKPT, torch.device("cuda"), log=None)
    eng = Engine(w, slots=1, max_ctx=2048)
    eng.prefill(list(range(1000, 1512)), 0)
    for _ in range(3): eng.decode_step([5], [0])
    eng.decode_step([5], [0], profile=True); torch.cuda.synchronize()
    trace(eng, layers=[int(a) for a in sys.argv[1:]] or [10, 11])
