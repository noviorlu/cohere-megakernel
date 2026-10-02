"""GEMM task correctness through the real megakernel (random weights).

Run: .venv/bin/python qwen/tests/test_gemm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmk import layout, native  # noqa: E402
from qmk.native import EPI_RESID, EPI_SILU_MUL, EPI_STORE, OP_GEMM_BF16, OP_GEMM_FP8  # noqa: E402

DEV = torch.device("cuda")
HIDDEN = 5120
EPS = 1e-6


def random_fp8(n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    w = (torch.randn(n, k, generator=gen, device=DEV) * 64).clamp(-448, 448)
    return w.to(torch.float8_e4m3fn).view(torch.uint8)


def test_e4m3_exhaustive(lib: native.Lib) -> None:
    codes = torch.arange(256, dtype=torch.uint8, device=DEV)
    ok = (codes != 0x7F) & (codes != 0xFF)  # NaN encodings
    got = lib.e4m3_to_bf16(codes)
    want = codes.view(torch.float8_e4m3fn).to(torch.bfloat16)
    assert torch.equal(got[ok].view(torch.int16), want[ok].view(torch.int16)), "e4m3→bf16 mismatch"
    print("e4m3→bf16: all 254 finite codes exact")


def run_gemm(lib, *, fp8, n, k, tn, bs, epi, norm, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    t = layout.Tiling(n=n, k=k, tile_n=tn, fp8=fp8)
    if fp8:
        w = random_fp8(n, k, gen)
        rg = torch.rand(n // 16, k // 128, generator=gen, device=DEV) * 1e-3 + 1e-4
        w_dense = layout.dequant(w, rg, torch.float32)
        tiled = layout.tile_weight(w, t)
        scales = layout.tile_scales(rg, t)
    else:
        w_dense = (torch.randn(n, k, generator=gen, device=DEV) * 0.02).to(torch.bfloat16)
        tiled = layout.tile_weight(w_dense, t)
        w_dense = w_dense.float()
        scales = None
    x = (torch.randn(bs, k, generator=gen, device=DEV) * 2).to(torch.bfloat16)
    norm_w = (torch.randn(k, generator=gen, device=DEV) * 0.1).to(torch.bfloat16) if norm else None
    ss_in = torch.zeros(1, 8, device=DEV)
    if norm:
        ss_in[0, :bs] = x.float().pow(2).sum(-1)
        xin = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)) * (1 + norm_w.float())
        xin = xin.to(torch.bfloat16).float()
    else:
        xin = x.float()
    acc = xin @ w_dense.T  # [bs, n] fp32

    out_cols = n // 2 if epi == EPI_SILU_MUL else n
    out = torch.zeros(bs, out_cols, dtype=torch.bfloat16, device=DEV)
    resid = (torch.randn(bs, n, generator=gen, device=DEV)).to(torch.bfloat16)
    resid0 = resid.clone()
    ss_out = torch.zeros(t.ntiles, 8, device=DEV)

    op = native.OpDesc()
    op.type = OP_GEMM_FP8 if fp8 else OP_GEMM_BF16
    op.ntasks = t.ntiles
    op.wait[0] = native.Dep(-1, 0)
    op.wait[1] = native.Dep(-1, 0)
    op.signal = 1
    g = op.gemm
    g.w = tiled.data_ptr()
    g.wscale = scales.data_ptr() if fp8 else None
    g.x = x.data_ptr()
    g.ldx = k
    g.norm_w = norm_w.data_ptr() if norm else None
    g.ss_in = ss_in.data_ptr()
    g.n_ss = 1
    g.tile_n = tn
    g.nchunks = t.nchunks
    g.epi = epi
    g.out = out.data_ptr()
    g.ldo = out_cols if epi != EPI_RESID else n
    g.resid = resid.data_ptr()
    g.ss_out = ss_out.data_ptr()
    ops = native.ops_to_device([op], DEV)
    tasks = torch.stack([torch.zeros(t.ntiles, dtype=torch.int32), torch.arange(t.ntiles, dtype=torch.int32)], 1)
    tasks = native.tasks_to_device(tasks, DEV)
    counters = torch.zeros(16, dtype=torch.int32, device=DEV)

    p = native.StepParams()
    p.ops, p.tasks, p.counters = ops.data_ptr(), tasks.data_ptr(), counters.data_ptr()
    p.ntasks, p.bs, p.max_ctx = t.ntiles, bs, 1
    lib.launch(p, 170)
    torch.cuda.synchronize()
    assert counters[1].item() == t.ntiles, counters[:2]

    if epi == EPI_STORE:
        want = acc.to(torch.bfloat16)
        got = out
    elif epi == EPI_SILU_MUL:
        half = tn // 2
        a3 = acc.view(bs, t.ntiles, 2, half)
        gate, up = a3[:, :, 0].reshape(bs, -1).to(torch.bfloat16), a3[:, :, 1].reshape(bs, -1).to(torch.bfloat16)
        want = torch.nn.functional.silu(gate) * up
        got = out
    else:
        want = resid0 + acc.to(torch.bfloat16)
        got = resid
        ss_want = want.float().pow(2).view(bs, t.ntiles, tn).sum(-1).T
        torch.testing.assert_close(ss_out[:, :bs], ss_want, rtol=1e-4, atol=1e-3)
    err = (got.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item()
    rel = err / max(scale, 1e-6)
    # bf16 output rounding (2^-8) plus fp32 summation-order differences.
    assert rel < 1e-2, f"rel err {rel:.3g} (abs {err:.3g}, scale {scale:.3g})"
    return rel


def main() -> None:
    lib = native.Lib()
    print(f"lib {lib.path.name}, smem {lib.smem_bytes} B")
    test_e4m3_exhaustive(lib)
    cases = [
        dict(fp8=True, n=512, k=5120, tn=128, bs=1, epi=EPI_STORE, norm=False),
        dict(fp8=True, n=512, k=5120, tn=64, bs=3, epi=EPI_STORE, norm=True),
        dict(fp8=True, n=1024, k=6144, tn=32, bs=8, epi=EPI_RESID, norm=False),
        dict(fp8=True, n=640, k=17408, tn=16, bs=5, epi=EPI_RESID, norm=False),
        dict(fp8=True, n=1024, k=5120, tn=64, bs=8, epi=EPI_SILU_MUL, norm=True),
        dict(fp8=False, n=1024, k=5120, tn=64, bs=2, epi=EPI_STORE, norm=True),
        dict(fp8=False, n=96, k=5120, tn=32, bs=8, epi=EPI_STORE, norm=True),
    ]
    for c in cases:
        rel = run_gemm(lib, **c)
        print(f"ok  rel={rel:.2e}  {c}")


if __name__ == "__main__":
    main()
