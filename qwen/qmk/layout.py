"""Weight tiling for the decode megakernel (see csrc/gemm.cuh for the consumer side).

A weight matrix W[N, K] is cut into tasks of TN rows. Each task streams
16 KiB chunks; inside a chunk the 8 consumer warps are arranged as R = TN/16
row groups x S = 8/R K-slices and each warp owns 2 KiB:

  FP8  : 16 rows x 128 k, as [step(4)][lane(32)][row g | row g+8][8 k]
  BF16 : 16 rows x  64 k, as [step(2)][row g | row g+8][lane(32)][8 k]

lane = 4*g + tig holds physical k = 8*tig .. 8*tig+7 of each 32-wide step.
Every layout below is a pure reshape+permute, so it inverts exactly
(`untile_*`), which is how prefill gets plain matrices back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

CHUNK_BYTES = 16384
WARPS = 8
FP8_BLOCK = 128  # checkpoint block-scale granularity (rows and columns)


@dataclass(frozen=True)
class Tiling:
    n: int
    k: int
    tile_n: int
    fp8: bool

    @property
    def r(self) -> int:
        return self.tile_n // 16

    @property
    def s(self) -> int:
        return WARPS // self.r

    @property
    def subk(self) -> int:
        return 128 if self.fp8 else 64

    @property
    def kc(self) -> int:  # k per chunk
        return self.s * self.subk

    @property
    def ntiles(self) -> int:
        return self.n // self.tile_n

    @property
    def nchunks(self) -> int:
        return self.k // self.kc

    def validate(self) -> None:
        if self.tile_n not in (16, 32, 64, 128):
            raise ValueError(f"tile_n must be 16/32/64/128, got {self.tile_n}")
        if self.n % self.tile_n or self.k % self.kc:
            raise ValueError(f"[{self.n}x{self.k}] not divisible by tile {self.tile_n}x{self.kc}")


# View of W as (T, R, hh, g, C, S, step, tig, e) and the permutation into chunk order.
def _view_dims(t: Tiling) -> tuple[int, ...]:
    steps = t.subk // 32
    return (t.ntiles, t.r, 2, 8, t.nchunks, t.s, steps, 4, 8)


_PERM_FP8 = (0, 4, 1, 5, 6, 3, 7, 2, 8)   # → T, C, R, S, step, g, tig, hh, e
_PERM_BF16 = (0, 4, 1, 5, 6, 2, 3, 7, 8)  # → T, C, R, S, step, hh, g, tig, e


def _perm(t: Tiling) -> tuple[int, ...]:
    return _PERM_FP8 if t.fp8 else _PERM_BF16


def _inverse(perm: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def tile_weight(w: torch.Tensor, t: Tiling) -> torch.Tensor:
    """W[N, K] (uint8 FP8 bytes or bf16) → flat tiled tensor of the same dtype."""
    t.validate()
    assert w.shape == (t.n, t.k), (w.shape, t)
    assert (w.dtype == torch.uint8) == t.fp8
    return w.reshape(_view_dims(t)).permute(_perm(t)).contiguous().reshape(-1)


def untile_weight(flat: torch.Tensor, t: Tiling) -> torch.Tensor:
    dims = _view_dims(t)
    perm = _perm(t)
    permuted_dims = tuple(dims[p] for p in perm)
    return flat.reshape(permuted_dims).permute(_inverse(perm)).reshape(t.n, t.k)


def tile_scales(rg_scale: torch.Tensor, t: Tiling) -> torch.Tensor:
    """Per-16-row-group scales [N/16, K/128] → kernel order [tile][chunk][warp]."""
    assert t.fp8 and rg_scale.shape == (t.n // 16, t.k // FP8_BLOCK)
    return (
        rg_scale.float()
        .reshape(t.ntiles, t.r, t.nchunks, t.s)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )


def row_group_scales(scale_inv: torch.Tensor, n: int) -> torch.Tensor:
    """Checkpoint block scales [ceil(n/128), K/128] → one row per 16-row group."""
    return scale_inv.repeat_interleave(FP8_BLOCK // 16, dim=0)[: n // 16]


def dequant(w_u8: torch.Tensor, rg_scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """FP8 bytes [N, K] + row-group scales → dense weight (fp32 math, then `dtype`)."""
    n, k = w_u8.shape
    w = w_u8.view(torch.float8_e4m3fn).float().view(n // 16, 16, k // FP8_BLOCK, FP8_BLOCK)
    w.mul_(rg_scale.float()[:, None, :, None])
    return w.view(n, k).to(dtype)
