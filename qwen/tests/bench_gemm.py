"""Bandwidth of a single GEMM op through the megakernel (random weights)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmk import layout, native
from qmk.native import EPI_STORE, EPI_RESID, EPI_SILU_MUL, OP_GEMM_FP8, OP_GEMM_BF16

DEV = torch.device("cuda")

def bench(lib, n, k, tn, bs, fp8=True, epi=EPI_STORE, iters=20):
    t = layout.Tiling(n=n, k=k, tile_n=tn, fp8=fp8)
    if fp8:
        w = torch.randint(0, 0x7e, (n, k), dtype=torch.uint8, device=DEV)
        tiled = layout.tile_weight(w, t); del w
        scales = layout.tile_scales(torch.ones(n // 16, k // 128, device=DEV), t)
    else:
        tiled = layout.tile_weight(torch.randn(n, k, device=DEV).to(torch.bfloat16), t)
        scales = None
    x = torch.randn(bs, k, device=DEV).to(torch.bfloat16)
    out_cols = n // 2 if epi == EPI_SILU_MUL else n
    out = torch.zeros(bs, out_cols, dtype=torch.bfloat16, device=DEV)
    ss = torch.zeros(t.ntiles, 8, device=DEV)
    op = native.OpDesc(); op.type = OP_GEMM_FP8 if fp8 else OP_GEMM_BF16; op.ntasks = t.ntiles
    op.wait[0] = native.Dep(-1, 0); op.wait[1] = native.Dep(-1, 0); op.signal = 1
    g = op.gemm
    g.w = tiled.data_ptr(); g.wscale = scales.data_ptr() if fp8 else None; g.x = x.data_ptr(); g.ldx = k
    g.tile_n = tn; g.nchunks = t.nchunks; g.epi = epi; g.out = out.data_ptr(); g.ldo = out_cols
    g.resid = out.data_ptr(); g.ss_out = ss.data_ptr()
    ops = native.ops_to_device([op], DEV)
    tasks = native.tasks_to_device(torch.stack([torch.zeros(t.ntiles, dtype=torch.int32), torch.arange(t.ntiles, dtype=torch.int32)], 1), DEV)
    counters = torch.zeros(16, dtype=torch.int32, device=DEV)
    p = native.StepParams(); p.ops, p.tasks, p.counters = ops.data_ptr(), tasks.data_ptr(), counters.data_ptr()
    p.ntasks, p.bs, p.max_ctx = t.ntiles, bs, 1
    for _ in range(3):
        counters.zero_(); lib.launch(p, 170)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        counters.zero_(); e0.record(); lib.launch(p, 170); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    ms = sorted(ts)[len(ts)//2]
    nbytes = tiled.numel() * tiled.element_size()
    print(f"N={n:6d} K={k:5d} TN={tn:3d} bs={bs} {'fp8 ' if fp8 else 'bf16'} {nbytes/1e6:7.1f} MB  {ms*1e3:8.1f} us  {nbytes/ms/1e6:7.0f} GB/s  tasks={t.ntiles}")

lib = native.Lib()
for bs in (1, 8):
    bench(lib, 34816, 5120, 64, bs, epi=EPI_SILU_MUL)
    bench(lib, 5120, 17408, 16, bs, epi=EPI_RESID)
    bench(lib, 5120, 6144, 32, bs, epi=EPI_RESID)
    bench(lib, 16384, 5120, 64, bs)
    bench(lib, 248320, 5120, 64, bs, fp8=False)
