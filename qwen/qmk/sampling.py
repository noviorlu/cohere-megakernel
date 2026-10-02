"""Token sampling from logits (greedy, or temperature with top-k / top-p)."""

from __future__ import annotations

import torch


def sample(logits: torch.Tensor, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
           generator: torch.Generator | None = None) -> list[int]:
    """logits [bs, vocab] → one token id per row."""
    if temperature <= 0:
        return logits.argmax(-1).tolist()
    x = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(x, min(top_k, x.shape[-1]), dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if top_p < 1.0:
        sorted_x, idx = torch.sort(x, descending=True, dim=-1)
        probs = sorted_x.softmax(-1)
        # drop tokens once the mass *before* them already exceeds top_p
        drop = probs.cumsum(-1) - probs > top_p
        x = x.scatter(-1, idx, sorted_x.masked_fill(drop, float("-inf")))
    return torch.multinomial(x.softmax(-1), 1, generator=generator).squeeze(-1).tolist()
