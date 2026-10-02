"""Cost of op-to-op dependency barriers: a chain of GEMM ops with/without waits (working set >> L2)."""
from __future__ import annotations

import torch
from util import DEV, Launch, gemm_op

from qmk import layout, native
from qmk.native import EPI_STORE, Dep, WHOLE


def chain(lib, n, k, tn, nops, deps, bs=1, label=""):
    t = layout.Tiling(n=n, k=k, tile_n=tn, fp8=True)
    nbytes = n * k
    big = torch.randint(0, 0x7E, (nops * nbytes,), dtype=torch.uint8, device=DEV)
    scales = torch.ones(t.ntiles * t.nchunks * 8, device=DEV)
    x = torch.randn(bs, k, device=DEV).to(torch.bfloat16)
    out = torch.zeros(bs, n, dtype=torch.bfloat16, device=DEV)
    ops = [gemm_op(t, w=big.data_ptr() + i * nbytes, scales=scales.data_ptr(), x=x, epi=EPI_STORE, out=out, ldo=n,
                   wait=Dep(i, WHOLE) if (deps and i > 0) else native.NO_DEP, signal_ctr=i + 1)
           for i in range(nops)]
    targets = [0] + [t.ntiles] * nops
    ms = Launch(lib, ops, bs, n_counters=nops + 2, targets=targets).time_ms()
    per_op_us = ms * 1e3 / nops
    print(f"{label:32s} {t.ntiles:5d} tasks/op  {per_op_us:7.1f} us/op  {nops * nbytes / ms / 1e6:6.0f} GB/s")
    return per_op_us


if __name__ == "__main__":
    lib = native.Lib()
    for (n, k, tn, nm) in [(34816, 5120, 32, "gate_up"), (5120, 17408, 16, "down"), (5120, 6144, 32, "out")]:
        nops = max(8, int(4e9 // (n * k)))
        a = chain(lib, n, k, tn, nops, False, label=f"{nm} TN{tn} no deps")
        b = chain(lib, n, k, tn, nops, True, label=f"{nm} TN{tn} chained")
        print(f"   → barrier cost {b - a:.1f} us/op")
