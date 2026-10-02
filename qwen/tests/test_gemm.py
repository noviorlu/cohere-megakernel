"""GEMM task correctness through the real megakernel (random weights).

Run: .venv/bin/python qwen/tests/test_gemm.py
"""

from __future__ import annotations

import torch
from util import DEV, Launch, gemm_op

from qmk import layout, native
from qmk.native import EPI_RESID, EPI_SILU_MUL, EPI_STORE

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


def run_gemm(lib, *, fp8, n, k, tn, bs, epi, norm, ksplit=1, seed=0):
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
    inv_in = torch.zeros(8, device=DEV)
    if norm:
        assert k == 5120, "kernel normalises over QMK_HIDDEN"
        inv_in[:bs] = torch.rsqrt(x.float().pow(2).mean(-1) + EPS)
        xin = (x.float() * inv_in[:bs].unsqueeze(-1)) * (1 + norm_w.float())
        xin = xin.to(torch.bfloat16).float()
    else:
        xin = x.float()
    acc = xin @ w_dense.T  # [bs, n] fp32

    out_cols = n // 2 if epi == EPI_SILU_MUL else n
    out = torch.zeros(bs, out_cols, dtype=torch.bfloat16, device=DEV)
    resid = (torch.randn(bs, n, generator=gen, device=DEV)).to(torch.bfloat16)
    resid0 = resid.clone()
    ss_out = torch.zeros(t.ntiles, 8, device=DEV)
    inv_out = torch.zeros(8, device=DEV)
    partial = torch.zeros(ksplit, 8, n, device=DEV)
    # For RESID epilogue: norm_ctr is the counter index used by the last-tile
    # arrival logic; signal goes to counter 1 once when all tiles are done.
    norm_ctr_idx = 2 + t.ntiles if epi == EPI_RESID else -1
    op = gemm_op(t, w=tiled.data_ptr(), scales=scales.data_ptr() if fp8 else None, x=x, epi=epi, out=out,
                 ldo=out_cols if epi != EPI_RESID else n, norm_w=norm_w, inv_in=inv_in if norm else None,
                 resid=resid, ss_out=ss_out, inv_out=inv_out if epi == EPI_RESID else None,
                 norm_ctr=norm_ctr_idx,
                 ksplit=ksplit, partial=partial, tile_ctr=2)
    n_counters = 2 + t.ntiles + (1 if epi == EPI_RESID else 0)
    run = Launch(lib, [op], bs, n_counters=n_counters)
    run()
    torch.cuda.synchronize()
    # RESID epilogue: only the last tile to finish signals (once).
    # Other epilogues: every tile signals.
    expected_signals = 1 if epi == EPI_RESID else t.ntiles
    assert run.counters[1].item() == expected_signals, f"counters[1]={run.counters[1].item()} expected {expected_signals}"

    if epi == EPI_STORE:
        want, got = acc.to(torch.bfloat16), out
    elif epi == EPI_SILU_MUL:
        a3 = acc.view(bs, t.ntiles, 2, tn // 2)
        gate, up = a3[:, :, 0].reshape(bs, -1).to(torch.bfloat16), a3[:, :, 1].reshape(bs, -1).to(torch.bfloat16)
        want, got = torch.nn.functional.silu(gate) * up, out
    else:
        want, got = resid0 + acc.to(torch.bfloat16), resid
        ss_want = want.float().pow(2).view(bs, t.ntiles, tn).sum(-1).T
        torch.testing.assert_close(ss_out[:, :bs], ss_want, rtol=1e-4, atol=1e-3)
        # Check that inv_out contains correct 1/rms of the updated residual.
        # The kernel divides by QMK_HIDDEN (5120), so this only makes sense
        # when n covers the full hidden dim.
        if n == 5120:
            inv_want = torch.rsqrt(want.float().pow(2).mean(-1) + EPS)
            torch.testing.assert_close(inv_out[:bs], inv_want, rtol=1e-4, atol=1e-3)
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
        # split-K: uneven chunk split (17 chunks / 2) and 4-way
        dict(fp8=True, n=640, k=17408, tn=16, bs=3, epi=EPI_RESID, norm=False, ksplit=2),
        dict(fp8=True, n=512, k=6144, tn=32, bs=8, epi=EPI_RESID, norm=False, ksplit=4),
        dict(fp8=True, n=512, k=5120, tn=32, bs=2, epi=EPI_STORE, norm=True, ksplit=3),
        # RESID with full hidden dim (n=5120) to verify inv_out (1/rms)
        dict(fp8=True, n=5120, k=17408, tn=16, bs=4, epi=EPI_RESID, norm=False, ksplit=2),
    ]
    for c in cases:
        rel = run_gemm(lib, **c)
        print(f"ok  rel={rel:.2e}  {c}")


if __name__ == "__main__":
    main()
