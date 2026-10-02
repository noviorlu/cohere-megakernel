"""Helpers for driving single ops through the megakernel in tests and microbenchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmk import layout, native  # noqa: E402
from qmk.native import NO_DEP, OP_GEMM_BF16, OP_GEMM_FP8, WHOLE, Dep  # noqa: E402

DEV = torch.device("cuda")


def gemm_op(t: layout.Tiling, *, w: int, scales: int | None, x: torch.Tensor, epi: int, out: torch.Tensor | None = None,
            ldo: int = 0, norm_w=None, inv_in=None, resid=None, ss_out=None, inv_out=None,
            norm_ctr: int = -1, ksplit=1, partial=None,
            tile_ctr=0, wait: Dep = NO_DEP, signal_ctr: int = 1) -> native.OpDesc:
    op = native.OpDesc()
    op.type = OP_GEMM_FP8 if t.fp8 else OP_GEMM_BF16
    op.ntasks = t.ntiles * ksplit
    op.wait[0], op.wait[1] = wait, NO_DEP
    op.signal = Dep(signal_ctr, WHOLE)
    g = op.gemm
    g.w, g.wscale = w, scales
    g.x, g.ldx = x.data_ptr(), x.shape[1]
    g.norm_w = None if norm_w is None else norm_w.data_ptr()
    g.inv_in = None if inv_in is None else inv_in.data_ptr()
    g.out, g.ldo = (None if out is None else out.data_ptr()), ldo
    g.resid = None if resid is None else resid.data_ptr()
    g.ss_out = None if ss_out is None else ss_out.data_ptr()
    g.inv_out = None if inv_out is None else inv_out.data_ptr()
    g.norm_ctr = norm_ctr
    g.tile_n, g.nchunks, g.epi, g.ntiles, g.ksplit = t.tile_n, t.nchunks, epi, t.ntiles, ksplit
    g.partial = None if partial is None else partial.data_ptr()
    g.tile_ctr = tile_ctr
    return op


class Launch:
    """Ops + their tasks (op-major, in order) ready to launch repeatedly."""

    def __init__(self, lib: native.Lib, ops: list[native.OpDesc], bs: int, n_counters: int = 64,
                 targets: list[int] | None = None):
        self.lib = lib
        self.ops = native.ops_to_device(ops, DEV)
        nt = [op.ntasks for op in ops]
        task_op = torch.repeat_interleave(torch.arange(len(ops), dtype=torch.int32), torch.tensor(nt))
        task_idx = torch.cat([torch.arange(n, dtype=torch.int32) for n in nt])
        self.tasks = native.tasks_to_device(torch.stack([task_op, task_idx], 1), DEV)
        self.counters = torch.zeros(n_counters, dtype=torch.int32, device=DEV)
        tg = targets or [0] * n_counters
        self.targets = torch.tensor(tg + [0] * (n_counters - len(tg)), dtype=torch.int32, device=DEV)
        p = native.StepParams()
        p.ops, p.tasks, p.counters, p.targets = (self.ops.data_ptr(), self.tasks.data_ptr(),
                                                 self.counters.data_ptr(), self.targets.data_ptr())
        p.ntasks, p.bs, p.max_ctx = self.tasks.shape[0], bs, 1
        self.params = p

    def __call__(self, blocks: int = 170) -> None:
        self.counters.zero_()
        self.lib.launch(self.params, blocks)

    def time_ms(self, iters: int = 5, blocks: int = 170) -> float:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self(blocks)
        ts = []
        for _ in range(iters):
            self.counters.zero_()
            e0.record()
            self.lib.launch(self.params, blocks)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return sorted(ts)[len(ts) // 2]
