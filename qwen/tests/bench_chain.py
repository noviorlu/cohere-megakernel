"""Cost of op-to-op dependency barriers: a chain of GEMM ops with/without waits (working set >> L2)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmk import layout, native
from qmk.native import EPI_STORE, OP_GEMM_FP8

DEV = torch.device("cuda")

def chain(lib, n, k, tn, nops, deps, bs=1, label=""):
    t = layout.Tiling(n=n, k=k, tile_n=tn, fp8=True)
    nbytes = n * k
    big = torch.randint(0, 0x7e, (nops * nbytes,), dtype=torch.uint8, device=DEV)
    scales = torch.ones(t.ntiles * t.nchunks * 8, device=DEV)
    x = torch.randn(bs, k, device=DEV).to(torch.bfloat16)
    out = torch.zeros(bs, n, dtype=torch.bfloat16, device=DEV)
    ops = []
    for i in range(nops):
        op = native.OpDesc(); op.type = OP_GEMM_FP8; op.ntasks = t.ntiles
        op.wait[0] = native.Dep(i, t.ntiles) if (deps and i > 0) else native.Dep(-1, 0)
        op.wait[1] = native.Dep(-1, 0); op.signal = i + 1
        g = op.gemm
        g.w = big.data_ptr() + i * nbytes; g.wscale = scales.data_ptr(); g.x = x.data_ptr(); g.ldx = k
        g.tile_n = tn; g.nchunks = t.nchunks; g.epi = EPI_STORE; g.out = out.data_ptr(); g.ldo = n
        ops.append(op)
    ops_d = native.ops_to_device(ops, DEV)
    tasks = torch.stack([torch.arange(nops, dtype=torch.int32).repeat_interleave(t.ntiles),
                         torch.arange(t.ntiles, dtype=torch.int32).repeat(nops)], 1)
    tasks = native.tasks_to_device(tasks, DEV)
    counters = torch.zeros(nops + 2, dtype=torch.int32, device=DEV)
    p = native.StepParams(); p.ops, p.tasks, p.counters = ops_d.data_ptr(), tasks.data_ptr(), counters.data_ptr()
    p.ntasks, p.bs, p.max_ctx = tasks.shape[0], bs, 1
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for i in range(6):
        counters.zero_(); e0.record(); lib.launch(p, 170); e1.record(); torch.cuda.synchronize()
        if i: ts.append(e0.elapsed_time(e1))
    ms = sorted(ts)[len(ts) // 2]
    per_op_us = ms * 1e3 / nops
    print(f"{label:32s} {t.ntiles:5d} tasks/op  {per_op_us:7.1f} us/op  {nops * nbytes / ms / 1e6:6.0f} GB/s")
    return per_op_us

if __name__ == "__main__":
    lib = native.Lib()
    for (n, k, tn, nm) in [(34816, 5120, 32, "gate_up"), (5120, 17408, 16, "down"), (5120, 6144, 16, "out")]:
        nops = max(8, int(4e9 // (n * k)))
        a = chain(lib, n, k, tn, nops, False, label=f"{nm} TN{tn} no deps")
        b = chain(lib, n, k, tn, nops, True, label=f"{nm} TN{tn} chained")
        print(f"   → barrier cost {b - a:.1f} us/op")
