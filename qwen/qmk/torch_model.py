"""PyTorch implementation of the Qwen3.8 text decoder on the tiled weights.

Two jobs:
  * prefill (chunked) — fills the Cache that the decode megakernel continues from;
  * a reference decode step for testing the megakernel.
Numerics follow transformers' Qwen3_5 modeling code (bf16 roundings in the
same places). Linear layers dequantize the FP8 weights on the fly; the
reference step does that in fp32, which is also what the kernel computes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cache import Cache
from .model import FullLayerWeights, LinearLayerWeights, TiledLinear, Weights

bf16 = torch.bfloat16


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNorm (zero-centred weight)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + w.float())).to(x.dtype)


def gated_rmsnorm(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNormGated."""
    xf = x.float()
    h = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    h = w * h
    return (h * F.silu(gate.float())).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., T, H, D]; cos/sin [T, rot] (bf16). Partial rotary on the first `rot` dims."""
    rot = cos.shape[-1]
    cos, sin = cos[:, None, :], sin[:, None, :]
    xr, xp = x[..., :rot], x[..., rot:]
    return torch.cat([(xr * cos) + (rotate_half(xr) * sin), xp], dim=-1)


class TorchModel:
    def __init__(self, weights: Weights, cache: Cache, precise: bool = False):
        self.w = weights
        self.cfg = weights.cfg
        self.cache = cache
        self.dev = weights.device
        # precise: fp32 dequant + fp32 matmul (the reference); else bf16 cuBLAS (prefill).
        self.precise = precise

    # ── building blocks ──────────────────────────────────────────────────────
    def linear(self, x: torch.Tensor, tl: TiledLinear) -> torch.Tensor:
        return tl.matmul(x, precise=self.precise)

    def mlp(self, h: torch.Tensor, layer) -> torch.Tensor:
        gu = self.linear(h, layer.gate_up)
        gate, up = gu.split(self.cfg.inter, dim=-1)
        return self.linear(F.silu(gate) * up, layer.down)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.w.embed[tokens.cpu()].to(self.dev)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(rmsnorm(x, self.w.final_norm, self.cfg.eps), self.w.lm_head)

    def _gdn_gates(self, ab: torch.Tensor, lw: LinearLayerWeights) -> tuple[torch.Tensor, torch.Tensor]:
        b, a = ab.split(self.cfg.lin_nv, dim=-1)
        beta = b.sigmoid()
        g = -lw.a_log.exp() * F.softplus(a.float() + lw.dt_bias)
        return beta, g

    def _conv(self, taps_and_x: torch.Tensor, lw: LinearLayerWeights, n_out: int) -> torch.Tensor:
        """Depthwise causal conv over [L, CH] (L = TAPS-1 + n_out) → SiLU, [n_out, CH] bf16."""
        y = F.conv1d(taps_and_x.T[None], lw.conv_w[:, None, :], groups=self.cfg.conv_ch)[0, :, -n_out:]
        return F.silu(y).T

    def _split_qkv_lin(self, mixed: torch.Tensor):
        c = self.cfg
        kd = c.lin_nk * c.lin_dk
        q, k, v = mixed.split([kd, kd, c.lin_nv * c.lin_dv], dim=-1)
        T = mixed.shape[0]
        rep = c.lin_nv // c.lin_nk
        q = q.reshape(T, c.lin_nk, c.lin_dk).repeat_interleave(rep, dim=1)
        k = k.reshape(T, c.lin_nk, c.lin_dk).repeat_interleave(rep, dim=1)
        return q, k, v.reshape(T, c.lin_nv, c.lin_dv)

    def _split_qkv_full(self, qkv: torch.Tensor, lw: FullLayerWeights):
        c = self.cfg
        T = qkv.shape[0]
        qg, k, v = qkv.split([c.nq * 2 * c.head_dim, c.nkv * c.head_dim, c.nkv * c.head_dim], dim=-1)
        q, gate = qg.view(T, c.nq, 2 * c.head_dim).chunk(2, dim=-1)
        q = rmsnorm(q, lw.q_norm, c.eps)
        k = rmsnorm(k.view(T, c.nkv, c.head_dim), lw.k_norm, c.eps)
        return q, k, v.view(T, c.nkv, c.head_dim), gate.reshape(T, -1)

    # ── prefill ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def prefill(self, tokens: list[int], slot: int, chunk: int = 2048) -> torch.Tensor:
        """Append `tokens` to `slot` (fresh or continuing) and return logits of the last token [vocab]."""
        cache = self.cache
        if cache.length[slot] + len(tokens) > cache.max_ctx:
            raise ValueError("prompt exceeds max_ctx")
        x_last = None
        for s in range(0, len(tokens), chunk):
            x_last = self._prefill_chunk(torch.tensor(tokens[s: s + chunk]), slot)
        return self.logits(x_last)[0].float()

    def _prefill_chunk(self, tokens: torch.Tensor, slot: int) -> torch.Tensor:
        c, cache = self.cfg, self.cache
        start, T = cache.length[slot], tokens.numel()
        x = self.embed(tokens)
        for i, lw in enumerate(self.w.layers):
            h = rmsnorm(x, lw.ln1, c.eps)
            if isinstance(lw, LinearLayerWeights):
                mix = self._gdn_prefill(h, lw, cache.kind_index[i], slot)
            else:
                mix = self._attn_prefill(h, lw, cache.kind_index[i], slot, start)
            x = x + mix
            x = x + self.mlp(rmsnorm(x, lw.ln2, c.eps), lw)
        cache.length[slot] = start + T
        return x[-1:]

    def _gdn_prefill(self, h, lw: LinearLayerWeights, li: int, slot: int) -> torch.Tensor:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        c, cache = self.cfg, self.cache
        T = h.shape[0]
        qkvz = self.linear(h, lw.qkvz)
        mixed, z = qkvz.split([c.conv_ch, c.lin_nv * c.lin_dv], dim=-1)
        beta, g = self._gdn_gates(self.linear(h, lw.ab), lw)
        par = cache.parity[slot]
        seq = torch.cat([cache.conv[li, slot, par], mixed])
        cache.conv[li, slot, par] = seq[-(c.conv_taps - 1):]
        q, k, v = self._split_qkv_lin(self._conv(seq, lw, T))
        o, state = chunk_gated_delta_rule(
            q[None], k[None], v[None], g=g[None], beta=beta[None],
            initial_state=cache.state[li, slot][None].clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        cache.state[li, slot] = state[0]
        o = gated_rmsnorm(o[0].reshape(-1, c.lin_dv), z.reshape(-1, c.lin_dv), lw.norm_w, c.eps)
        return self.linear(o.reshape(T, -1), lw.out)

    def _attn_prefill(self, h, lw: FullLayerWeights, fi: int, slot: int, start: int) -> torch.Tensor:
        c, cache = self.cfg, self.cache
        T = h.shape[0]
        q, k, v, gate = self._split_qkv_full(self.linear(h, lw.qkv), lw)
        cos, sin = cache.cos[start: start + T], cache.sin[start: start + T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        cache.k[fi, slot, :, start: start + T] = k.transpose(0, 1)
        cache.v[fi, slot, :, start: start + T] = v.transpose(0, 1)
        L = start + T
        keys, vals = cache.k[fi, slot, :, :L][None], cache.v[fi, slot, :, :L][None]
        mask = torch.arange(L, device=self.dev)[None, :] <= (start + torch.arange(T, device=self.dev))[:, None]
        attn = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], keys, vals, attn_mask=mask, scale=c.head_dim ** -0.5, enable_gqa=True
        )
        attn = attn[0].transpose(0, 1).reshape(T, -1)
        return self.linear(attn * torch.sigmoid(gate), lw.out)

    # ── reference decode step ───────────────────────────────────────────────
    @torch.no_grad()
    def decode_step(self, tokens: list[int], slots: list[int]) -> torch.Tensor:
        """One token per row; mirrors the megakernel step (and its cache updates). Returns logits [bs, vocab]."""
        c, cache = self.cfg, self.cache
        bs = len(tokens)
        pos = [cache.length[s] for s in slots]
        par = [cache.parity[s] for s in slots]
        x = self.embed(torch.tensor(tokens))
        for i, lw in enumerate(self.w.layers):
            h = rmsnorm(x, lw.ln1, c.eps)
            if isinstance(lw, LinearLayerWeights):
                li = cache.kind_index[i]
                qkvz = self.linear(h, lw.qkvz)
                mixed, z = qkvz.split([c.conv_ch, c.lin_nv * c.lin_dv], dim=-1)
                beta, g = self._gdn_gates(self.linear(h, lw.ab), lw)
                outs = []
                for b, s in enumerate(slots):
                    seq = torch.cat([cache.conv[li, s, par[b]], mixed[b: b + 1]])
                    cache.conv[li, s, par[b] ^ 1] = seq[1:]
                    q, k, v = self._split_qkv_lin(self._conv(seq, lw, 1))
                    q = q.float()
                    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / (c.lin_dk ** 0.5)
                    kf = k.float()
                    kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
                    S = cache.state[li, s] * g[b].exp()[:, None, None]
                    kv_mem = (S * kf[0][:, :, None]).sum(1)
                    delta = (v[0].float() - kv_mem) * beta[b].float()[:, None]
                    S = S + kf[0][:, :, None] * delta[:, None, :]
                    cache.state[li, s] = S
                    outs.append((S * q[0][:, :, None]).sum(1).to(bf16))
                o = torch.stack(outs).reshape(-1, c.lin_dv)
                o = gated_rmsnorm(o, z.reshape(-1, c.lin_dv), lw.norm_w, c.eps)
                mix = self.linear(o.reshape(bs, -1), lw.out)
            else:
                fi = cache.kind_index[i]
                q, k, v, gate = self._split_qkv_full(self.linear(h, lw.qkv), lw)
                rows = []
                for b, s in enumerate(slots):
                    p = pos[b]
                    cos, sin = cache.cos[p: p + 1], cache.sin[p: p + 1]
                    qb, kb = apply_rope(q[b: b + 1], cos, sin), apply_rope(k[b: b + 1], cos, sin)
                    cache.k[fi, s, :, p] = kb[0]
                    cache.v[fi, s, :, p] = v[b]
                    keys = cache.k[fi, s, :, : p + 1].float().repeat_interleave(c.nq // c.nkv, 0)
                    vals = cache.v[fi, s, :, : p + 1].float().repeat_interleave(c.nq // c.nkv, 0)
                    sc = (qb[0].float()[:, None, :] @ keys.transpose(1, 2))[:, 0] * c.head_dim ** -0.5
                    rows.append((sc.softmax(-1)[:, None, :] @ vals)[:, 0].to(bf16).reshape(-1))
                attn = torch.stack(rows)
                mix = self.linear(attn * torch.sigmoid(gate), lw.out)
            x = x + mix
            x = x + self.mlp(rmsnorm(x, lw.ln2, c.eps), lw)
        for b, s in enumerate(slots):
            cache.length[s] += 1
            cache.parity[s] ^= 1
        return self.logits(x).float()
