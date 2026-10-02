"""Per-slot decode state: KV cache (full layers), conv + recurrent state (GDN layers), RoPE tables."""

from __future__ import annotations

import torch

from .model import Config


class Cache:
    def __init__(self, cfg: Config, slots: int, max_ctx: int, device: torch.device):
        self.cfg = cfg
        self.slots = slots
        self.max_ctx = max_ctx
        self.device = device
        n_lin, n_full = len(cfg.lin_layers), len(cfg.full_layers)
        # Index of each layer within its kind.
        self.kind_index = {layer: i for i, layer in enumerate(cfg.lin_layers)}
        self.kind_index.update({layer: i for i, layer in enumerate(cfg.full_layers)})

        bf = torch.bfloat16
        # conv[l, slot, parity, tap, ch]: the TAPS-1 most recent conv inputs, oldest first.
        self.conv = torch.zeros(n_lin, slots, 2, cfg.conv_taps - 1, cfg.conv_ch, dtype=bf, device=device)
        self.state = torch.zeros(n_lin, slots, cfg.lin_nv, cfg.lin_dk, cfg.lin_dv, dtype=torch.float32, device=device)
        self.k = torch.zeros(n_full, slots, cfg.nkv, max_ctx, cfg.head_dim, dtype=bf, device=device)
        self.v = torch.zeros_like(self.k)
        self.cos, self.sin = rope_tables(cfg, max_ctx, device)
        self.length = [0] * slots      # tokens cached per slot
        self.parity = [0] * slots      # conv buffer holding the current taps

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.conv, self.state, self.k, self.v))

    def reset_slot(self, slot: int) -> None:
        self.conv[:, slot].zero_()
        self.state[:, slot].zero_()
        self.length[slot] = 0
        self.parity[slot] = 0

    def clone(self) -> "Cache":
        c = Cache.__new__(Cache)
        c.__dict__.update(self.__dict__)
        for name in ("conv", "state", "k", "v"):
            setattr(c, name, getattr(self, name).clone())
        c.length = list(self.length)
        c.parity = list(self.parity)
        return c


def rope_tables(cfg: Config, max_ctx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [max_ctx, rot_dim] in bf16, computed like Qwen3_5TextRotaryEmbedding for text positions."""
    dim = cfg.rot_dim
    # HF builds inv_freq on CPU at init and the cos/sin on the activation's device.
    inv_freq = (1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))).to(device)
    pos = torch.arange(max_ctx, device=device).float()
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(torch.bfloat16).contiguous(), emb.sin().to(torch.bfloat16).contiguous()
