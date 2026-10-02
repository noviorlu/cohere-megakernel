"""DRAM streaming rate of GEMM ops with a working set >> L2 (many independent copies in one launch)."""
from __future__ import annotations

import torch
from util import DEV, Launch, gemm_op

from qmk import layout, native
from qmk.native import EPI_SILU_MUL, EPI_STORE


def bench(lib, n, k, tn, bs, fp8, norm, copies, epi=EPI_STORE, label=""):
    t = layout.Tiling(n=n, k=k, tile_n=tn, fp8=fp8)
    nbytes = n * k * (1 if fp8 else 2)
    big = torch.randint(0, 0x7E, (copies * nbytes,), dtype=torch.uint8, device=DEV)  # contents irrelevant
    scales = torch.ones(t.ntiles * t.nchunks * 8, device=DEV)
    x = torch.randn(bs, k, device=DEV).to(torch.bfloat16)
    nw = torch.zeros(k, device=DEV, dtype=torch.bfloat16)
    inv = torch.ones(8, device=DEV)
    out = torch.zeros(bs, n, dtype=torch.bfloat16, device=DEV)
    ops = [gemm_op(t, w=big.data_ptr() + c * nbytes, scales=scales.data_ptr(), x=x, epi=epi, out=out, ldo=n,
                   norm_w=nw if norm else None, inv_in=inv if norm else None) for c in range(copies)]
    ms = Launch(lib, ops, bs).time_ms()
    print(f"{label:28s} {copies * nbytes / 1e9:5.2f} GB  {ms:7.2f} ms  {copies * nbytes / ms / 1e6:6.0f} GB/s")


if __name__ == "__main__":
    lib = native.Lib()
    for bs in (1, 8):
        print(f"-- bs={bs}")
        bench(lib, 34816, 5120, 32, bs, True, True, 16, epi=EPI_SILU_MUL, label="fp8 gate_up norm silu TN32")
        bench(lib, 5120, 17408, 16, bs, True, False, 32, label="fp8 down TN16")
        bench(lib, 34816, 5120, 64, bs, False, True, 8, label="bf16 TN64 norm")
