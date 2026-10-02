"""Qwen3.8-27B-FP8 text decoder: config and weights in megakernel layout.

Only the text decoder is loaded (no vision tower, no MTP head). FP8 linears
are stored tiled (layout.py) together with their row-group scales, which is
enough both for the decode kernel and for exact untiling in prefill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from . import layout
from .layout import Tiling

PREFIX = "model.language_model."


@dataclass(frozen=True)
class Config:
    hidden: int
    layers: int
    layer_types: tuple[str, ...]
    inter: int
    vocab: int
    eps: float
    # full attention
    nq: int
    nkv: int
    head_dim: int
    rot_dim: int
    rope_theta: float
    # gated deltanet
    lin_nk: int
    lin_nv: int
    lin_dk: int
    lin_dv: int
    conv_taps: int
    eos_token_ids: tuple[int, ...] = ()

    @staticmethod
    def load(ckpt: Path) -> "Config":
        raw = json.loads((ckpt / "config.json").read_text())
        t = raw.get("text_config", raw)
        rope = t["rope_parameters"]
        gen = ckpt / "generation_config.json"
        eos = t.get("eos_token_id")
        if gen.exists():
            eos = json.loads(gen.read_text()).get("eos_token_id", eos)
        eos = tuple(eos) if isinstance(eos, list) else (eos,)
        cfg = Config(
            hidden=t["hidden_size"],
            layers=t["num_hidden_layers"],
            layer_types=tuple(t["layer_types"]),
            inter=t["intermediate_size"],
            vocab=t["vocab_size"],
            eps=t["rms_norm_eps"],
            nq=t["num_attention_heads"],
            nkv=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            rot_dim=int(t["head_dim"] * rope.get("partial_rotary_factor", 1.0)),
            rope_theta=rope["rope_theta"],
            lin_nk=t["linear_num_key_heads"],
            lin_nv=t["linear_num_value_heads"],
            lin_dk=t["linear_key_head_dim"],
            lin_dv=t["linear_value_head_dim"],
            conv_taps=t["linear_conv_kernel_dim"],
            eos_token_ids=eos,
        )
        cfg.check_kernel_geometry()
        return cfg

    def check_kernel_geometry(self) -> None:
        """The CUDA side hard-codes these (csrc/abi.h)."""
        want = dict(hidden=5120, inter=17408, nq=24, nkv=4, head_dim=256, rot_dim=64,
                    lin_nk=16, lin_nv=48, lin_dk=128, lin_dv=128, conv_taps=4)
        bad = {k: (getattr(self, k), v) for k, v in want.items() if getattr(self, k) != v}
        if bad:
            raise ValueError(f"checkpoint geometry differs from the compiled kernel: {bad}")

    def is_linear(self, layer: int) -> bool:
        return self.layer_types[layer] == "linear_attention"

    @property
    def lin_layers(self) -> list[int]:
        return [i for i in range(self.layers) if self.is_linear(i)]

    @property
    def full_layers(self) -> list[int]:
        return [i for i in range(self.layers) if not self.is_linear(i)]

    @property
    def conv_ch(self) -> int:
        return 2 * self.lin_nk * self.lin_dk + self.lin_nv * self.lin_dv


@dataclass
class TiledLinear:
    tiling: Tiling
    w: torch.Tensor                  # flat tiled bytes (uint8) or bf16
    scales: torch.Tensor | None      # kernel-order scales (FP8)
    rg_scale: torch.Tensor | None    # [N/16, K/128] (FP8), for untiling
    parts: tuple[tuple[str, int], ...]  # (name, rows) of fused sources, concatenated = "source order"
    inv_perm: torch.Tensor | None = None  # source row j sits at kernel row inv_perm[j] (None: identity)

    def matmul(self, x: torch.Tensor, precise: bool = False, block_rows: int = 2048) -> torch.Tensor:
        """y = x @ W^T (bf16 out), dequantizing one block of tiles at a time to bound memory.

        precise: fp32 weights and matmul (the kernel's numerics); else bf16 cuBLAS.
        """
        t = self.tiling
        block_rows -= block_rows % t.tile_n
        dt = torch.float32 if precise else torch.bfloat16
        xin = x.float() if precise else x
        outs = []
        for r0 in range(0, t.n, block_rows):
            rows = min(block_rows, t.n - r0)
            bt = Tiling(n=rows, k=t.k, tile_n=t.tile_n, fp8=t.fp8)
            flat = self.w[r0 * t.k: (r0 + rows) * t.k]
            wb = layout.untile_weight(flat, bt)
            if t.fp8:
                wb = layout.dequant(wb, self.rg_scale[r0 // 16: (r0 + rows) // 16], dt)
            else:
                wb = wb.to(dt)
            outs.append(xin @ wb.T)
            del wb
        y = torch.cat(outs, dim=-1)
        if self.inv_perm is not None:
            y = y[..., self.inv_perm]
        return y.to(torch.bfloat16)

    def dense(self, dtype=torch.bfloat16) -> torch.Tensor:
        """Back to a plain [N, K] matrix in source row order (for prefill / reference)."""
        w = layout.untile_weight(self.w, self.tiling)
        if self.tiling.fp8:
            w = layout.dequant(w, self.rg_scale, dtype)
        else:
            w = w.to(dtype)
        if self.inv_perm is not None:
            w = w[self.inv_perm]
        return w

    def split_dense(self, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
        w = self.dense(dtype)
        out, off = {}, 0
        for name, rows in self.parts:
            out[name] = w[off: off + rows]
            off += rows
        return out


@dataclass
class LinearLayerWeights:
    ln1: torch.Tensor
    ln2: torch.Tensor
    qkvz: TiledLinear
    ab: TiledLinear
    out: TiledLinear
    gate_up: TiledLinear
    down: TiledLinear
    conv_w: torch.Tensor   # [CH, TAPS] bf16
    a_log: torch.Tensor    # [NV] f32
    dt_bias: torch.Tensor  # [NV] f32
    norm_w: torch.Tensor   # [DV] bf16


@dataclass
class FullLayerWeights:
    ln1: torch.Tensor
    ln2: torch.Tensor
    qkv: TiledLinear
    out: TiledLinear
    gate_up: TiledLinear
    down: TiledLinear
    q_norm: torch.Tensor
    k_norm: torch.Tensor


# Rows per tile for each matrix; picked so every op has >= ~160 tasks on 170 SMs.
TILE_QKVZ = 32
TILE_AB = 32
TILE_OUT = 32  # o_proj K-splits must land on 1536-wide kv-head groups: KC = 512
TILE_GATE_UP = 32
TILE_DOWN = 16
TILE_QKV = 32
TILE_LM_HEAD = 64


# K-splits of the GEMMs that consume per-head outputs / the MLP hidden state,
# so they can start on the part of their input that is already done.
KSPLIT_OUT = 4      # GDN out_proj: 12 heads (4 key heads) per split
KSPLIT_O = 4        # attention o_proj: one split per kv-head group
KSPLIT_DOWN = 2     # down_proj: hidden 0..8191 | 8192..17407


def qkvz_order(nk: int = 16, dk: int = 128, nv: int = 48, dv: int = 128) -> torch.Tensor:
    """Source rows [q | k | v | z] regrouped per key head: q[kh] k[kh] v[3kh:3kh+3] z[3kh:3kh+3]."""
    g = nv // nk
    q0, k0, v0, z0 = 0, nk * dk, 2 * nk * dk, 2 * nk * dk + nv * dv
    rows = []
    for kh in range(nk):
        rows += [torch.arange(q0 + kh * dk, q0 + (kh + 1) * dk), torch.arange(k0 + kh * dk, k0 + (kh + 1) * dk),
                 torch.arange(v0 + kh * g * dv, v0 + (kh + 1) * g * dv),
                 torch.arange(z0 + kh * g * dv, z0 + (kh + 1) * g * dv)]
    return torch.cat(rows)


def qkv_order(nq: int = 24, nkv: int = 4, hd: int = 256) -> torch.Tensor:
    """Source rows [q_proj (query|gate per head) | k | v] regrouped per kv head."""
    g = nq // nkv
    k0, v0 = nq * 2 * hd, nq * 2 * hd + nkv * hd
    rows = []
    for kv in range(nkv):
        rows += [torch.arange(kv * g * 2 * hd, (kv + 1) * g * 2 * hd),
                 torch.arange(k0 + kv * hd, k0 + (kv + 1) * hd), torch.arange(v0 + kv * hd, v0 + (kv + 1) * hd)]
    return torch.cat(rows)


def gate_up_order(inter: int, tile_n: int) -> torch.Tensor:
    """Source rows [gate | up]: each tile holds tile_n/2 gate rows then the matching up rows."""
    half = tile_n // 2
    t = torch.arange(inter // half)
    gate = (t[:, None] * half + torch.arange(half)[None, :])
    return torch.stack([gate, gate + inter], 1).reshape(-1)


@dataclass
class Weights:
    cfg: Config
    layers: list[LinearLayerWeights | FullLayerWeights]
    final_norm: torch.Tensor
    lm_head: TiledLinear
    embed: torch.Tensor  # [vocab, hidden] bf16, kept on CPU
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    @staticmethod
    def load(ckpt: Path, device: torch.device, layers: int | None = None, log=print) -> "Weights":
        cfg = Config.load(ckpt)
        if layers is not None:
            cfg = _truncate(cfg, layers)
        loader = _Loader(ckpt, device, cfg.inter)
        out: list[LinearLayerWeights | FullLayerWeights] = []
        for i in range(cfg.layers):
            out.append(loader.linear_layer(i) if cfg.is_linear(i) else loader.full_layer(i))
            if log and (i % 8 == 7 or i == cfg.layers - 1):
                log(f"  loaded layer {i + 1}/{cfg.layers}  ({torch.cuda.memory_allocated(device) / 1e9:.1f} GB)")
        final_norm = loader.bf16(PREFIX + "norm.weight")
        lm_head = loader.bf16_linear([("lm_head", "lm_head.weight")], TILE_LM_HEAD)
        embed = loader.get(PREFIX + "embed_tokens.weight", device="cpu")
        del loader
        torch.cuda.empty_cache()  # return loading transients; only ~4 GB is left for everything else
        return Weights(cfg=cfg, layers=out, final_norm=final_norm, lm_head=lm_head, embed=embed, device=device)


def _truncate(cfg: Config, layers: int) -> Config:
    d = dict(cfg.__dict__)
    d["layers"] = layers
    d["layer_types"] = cfg.layer_types[:layers]
    return Config(**d)


class _Loader:
    def __init__(self, ckpt: Path, device: torch.device, inter: int):
        self.inter = inter
        self.ckpt = ckpt
        self.device = device
        self.index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
        self._files: dict[str, object] = {}

    def get(self, name: str, device: torch.device | str | None = None) -> torch.Tensor:
        fname = self.index[name]
        dev = str(device or self.device)
        key = f"{fname}@{dev}"
        if key not in self._files:
            self._files[key] = safe_open(str(self.ckpt / fname), framework="pt", device=dev)
        return self._files[key].get_tensor(name)

    def bf16(self, name: str) -> torch.Tensor:
        return self.get(name).to(torch.bfloat16).contiguous()

    def f32(self, name: str) -> torch.Tensor:
        return self.get(name).float().contiguous()

    def _fp8(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.get(name + ".weight")
        assert w.dtype == torch.float8_e4m3fn, (name, w.dtype)
        s = self.get(name + ".weight_scale_inv").float()
        return w.view(torch.uint8), layout.row_group_scales(s, w.shape[0])

    def fp8_linear(self, parts: list[tuple[str, str]], tile_n: int, order: torch.Tensor | None = None) -> TiledLinear:
        """Fuse FP8 matrices by rows (in `parts` order), then reorder rows by `order` (kernel row → source row)."""
        ws, rgs, meta = [], [], []
        for short, name in parts:
            w, rg = self._fp8(name)
            ws.append(w)
            rgs.append(rg)
            meta.append((short, w.shape[0]))
        w = torch.cat(ws) if len(ws) > 1 else ws[0]
        rg = torch.cat(rgs) if len(rgs) > 1 else rgs[0]
        del ws, rgs
        inv = None
        if order is not None:
            order = order.to(w.device)
            assert order.numel() == w.shape[0] and bool((order.view(-1, 16)[:, 0] % 16 == 0).all())
            w = w[order]
            rg = rg[order.view(-1, 16)[:, 0] // 16]
            inv = torch.argsort(order)
        tl = _tile_fp8(w, rg, tile_n, tuple(meta))
        tl.inv_perm = inv
        return tl

    def gate_up(self, prefix: str) -> TiledLinear:
        return self.fp8_linear([("gate", prefix + ".gate_proj"), ("up", prefix + ".up_proj")], TILE_GATE_UP,
                               gate_up_order(self.inter, TILE_GATE_UP))

    def bf16_linear(self, parts: list[tuple[str, str]], tile_n: int) -> TiledLinear:
        ws = [self.bf16(name) for _, name in parts]
        w = torch.cat(ws) if len(ws) > 1 else ws[0]
        t = Tiling(n=w.shape[0], k=w.shape[1], tile_n=tile_n, fp8=False)
        tiled = layout.tile_weight(w, t)
        meta = tuple((short, x.shape[0]) for (short, _), x in zip(parts, ws))
        return TiledLinear(t, tiled, None, None, meta)

    def linear_layer(self, i: int) -> LinearLayerWeights:
        p = f"{PREFIX}layers.{i}."
        la = p + "linear_attn."
        return LinearLayerWeights(
            ln1=self.bf16(p + "input_layernorm.weight"),
            ln2=self.bf16(p + "post_attention_layernorm.weight"),
            qkvz=self.fp8_linear([("qkv", la + "in_proj_qkv"), ("z", la + "in_proj_z")], TILE_QKVZ, qkvz_order()),
            ab=self.bf16_linear([("b", la + "in_proj_b.weight"), ("a", la + "in_proj_a.weight")], TILE_AB),
            out=self.fp8_linear([("out", la + "out_proj")], TILE_OUT),
            gate_up=self.gate_up(p + "mlp"),
            down=self.fp8_linear([("down", p + "mlp.down_proj")], TILE_DOWN),
            conv_w=self.bf16(la + "conv1d.weight").flatten(1).contiguous(),  # [CH, 1, TAPS] → [CH, TAPS]
            a_log=self.f32(la + "A_log"),
            dt_bias=self.f32(la + "dt_bias"),
            norm_w=self.bf16(la + "norm.weight"),
        )

    def full_layer(self, i: int) -> FullLayerWeights:
        p = f"{PREFIX}layers.{i}."
        sa = p + "self_attn."
        return FullLayerWeights(
            ln1=self.bf16(p + "input_layernorm.weight"),
            ln2=self.bf16(p + "post_attention_layernorm.weight"),
            qkv=self.fp8_linear([("q", sa + "q_proj"), ("k", sa + "k_proj"), ("v", sa + "v_proj")], TILE_QKV,
                                qkv_order()),
            out=self.fp8_linear([("o", sa + "o_proj")], TILE_OUT),
            gate_up=self.gate_up(p + "mlp"),
            down=self.fp8_linear([("down", p + "mlp.down_proj")], TILE_DOWN),
            q_norm=self.bf16(sa + "q_norm.weight"),
            k_norm=self.bf16(sa + "k_norm.weight"),
        )


def _tile_fp8(w: torch.Tensor, rg: torch.Tensor, tile_n: int, parts) -> TiledLinear:
    t = Tiling(n=w.shape[0], k=w.shape[1], tile_n=tile_n, fp8=True)
    tiled = layout.tile_weight(w, t)
    return TiledLinear(t, tiled, layout.tile_scales(rg, t), rg.contiguous(), parts)
