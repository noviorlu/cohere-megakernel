"""DRAM streaming rate of GEMM ops with a working set >> L2 (many independent copies in one launch)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmk import layout, native
from qmk.native import EPI_STORE, EPI_SILU_MUL, OP_GEMM_FP8, OP_GEMM_BF16

DEV = torch.device("cuda")

def bench(lib, n, k, tn, bs, fp8, norm, copies, epi=EPI_STORE, label=""):
    t = layout.Tiling(n=n, k=k, tile_n=tn, fp8=fp8)
    nbytes = n * k * (1 if fp8 else 2)
    big = torch.randint(0, 0x7e, (copies * nbytes,), dtype=torch.uint8, device=DEV)  # contents irrelevant
    scales = torch.ones(t.ntiles * t.nchunks * 8, device=DEV)
    x = torch.randn(bs, k, device=DEV).to(torch.bfloat16)
    nw = torch.zeros(k, device=DEV, dtype=torch.bfloat16)
    ss = torch.ones(160, 8, device=DEV)
    out = torch.zeros(bs, n, dtype=torch.bfloat16, device=DEV)
    ops = []
    for c in range(copies):
        op = native.OpDesc(); op.type = OP_GEMM_FP8 if fp8 else OP_GEMM_BF16; op.ntasks = t.ntiles
        op.wait[0] = native.Dep(-1, 0); op.wait[1] = native.Dep(-1, 0); op.signal = 1
        g = op.gemm
        g.w = big.data_ptr() + c * nbytes; g.wscale = scales.data_ptr(); g.x = x.data_ptr(); g.ldx = k
        g.norm_w = nw.data_ptr() if norm else None; g.ss_in = ss.data_ptr(); g.n_ss = 160
        g.tile_n = tn; g.nchunks = t.nchunks; g.epi = epi; g.out = out.data_ptr(); g.ldo = n
        ops.append(op)
    ops_d = native.ops_to_device(ops, DEV)
    tasks = torch.stack([torch.arange(copies, dtype=torch.int32).repeat_interleave(t.ntiles),
                         torch.arange(t.ntiles, dtype=torch.int32).repeat(copies)], 1)
    tasks = native.tasks_to_device(tasks, DEV)
    counters = torch.zeros(4, dtype=torch.int32, device=DEV)
    p = native.StepParams(); p.ops, p.tasks, p.counters = ops_d.data_ptr(), tasks.data_ptr(), counters.data_ptr()
    p.ntasks, p.bs, p.max_ctx = tasks.shape[0], bs, 1
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for i in range(6):
        counters.zero_(); e0.record(); lib.launch(p, 170); e1.record(); torch.cuda.synchronize()
        if i >= 1: ts.append(e0.elapsed_time(e1))
    ms = sorted(ts)[len(ts) // 2]
    print(f"{label:28s} {copies * nbytes / 1e9:5.2f} GB  {ms:7.2f} ms  {copies * nbytes / ms / 1e6:6.0f} GB/s")

lib = native.Lib()
for bs in (1, 8):
    print(f"-- bs={bs}")
    bench(lib, 34816, 5120, 64, bs, True, True, 16, epi=EPI_SILU_MUL, label="fp8 gate_up norm silu")
    bench(lib, 34816, 5120, 64, bs, True, False, 16, label="fp8 34816x5120 TN64")
    bench(lib, 5120, 17408, 16, bs, True, False, 32, label="fp8 down TN16")
    bench(lib, 34816, 5120, 64, bs, False, False, 8, label="bf16 34816x5120 TN64")
    bench(lib, 34816, 5120, 64, bs, False, True, 8, label="bf16 34816x5120 TN64 norm")
