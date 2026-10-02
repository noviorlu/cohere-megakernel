"""Decode engine: torch prefill + one megakernel launch per decode step."""

from __future__ import annotations

import math

import torch

from . import native, schedule
from .cache import Cache
from .model import Weights
from .torch_model import TorchModel

MIN_SPLIT_TOKENS = 32  # fewest KV positions worth a separate attention task


class Engine:
    def __init__(self, weights: Weights, slots: int, max_ctx: int, lib: native.Lib | None = None):
        self.w = weights
        self.dev = weights.device
        self.lib = lib or native.Lib()
        self.cache = Cache(weights.cfg, slots, max_ctx, self.dev)
        self.buf = schedule.Buffers(weights, self.dev)
        self.torch = TorchModel(weights, self.cache)
        self.num_blocks = torch.cuda.get_device_properties(self.dev).multi_processor_count
        self._schedules: dict[tuple[int, int], schedule.Schedule] = {}
        self._counters = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._warmup()

    def _warmup(self) -> None:
        """Empty launch: makes the driver reserve the kernel's local memory before torch grabs the rest."""
        p = native.StepParams()
        p.counters, p.ntasks, p.bs, p.max_ctx = self._counters.data_ptr(), 0, 1, self.cache.max_ctx
        self.lib.launch(p, self.num_blocks)
        torch.cuda.synchronize(self.dev)

    # ── scheduling ──────────────────────────────────────────────────────────
    def nsplit_for(self, bs: int, ctx: int) -> int:
        """Attention splits per (row, kv head): about one task per SM, power of two (bounded schedule count)."""
        groups = self.w.cfg.nkv * bs
        cap = min(schedule.MAX_NSPLIT, max(1, self.num_blocks // groups), max(1, ctx // MIN_SPLIT_TOKENS))
        n = 1
        while n * 2 <= cap:
            n *= 2
        return n

    def schedule(self, bs: int, nsplit: int) -> schedule.Schedule:
        key = (bs, nsplit)
        if key not in self._schedules:
            self._schedules[key] = schedule.build(self.w, self.cache, self.buf, bs, nsplit)
            n = self._schedules[key].n_counters
            if self._counters.numel() < n:
                self._counters = torch.zeros(n, dtype=torch.int32, device=self.dev)
        return self._schedules[key]

    # ── stepping ────────────────────────────────────────────────────────────
    def prefill(self, tokens: list[int], slot: int) -> torch.Tensor:
        return self.torch.prefill(tokens, slot)

    @torch.no_grad()
    def decode_step(self, tokens: list[int], slots: list[int], profile: bool = False) -> torch.Tensor:
        """Feed one token per row; returns logits [bs, vocab] (bf16) for the next position.

        profile: record per-task timestamps into self.last_profile ([ntasks, 4] int64:
        claim, deps ready, done (ns), smid) together with self.last_schedule.
        """
        bs = len(tokens)
        cache = self.cache
        pos = [cache.length[s] for s in slots]
        if max(pos) >= cache.max_ctx:
            raise ValueError("context full")
        x = self.w.embed[torch.tensor(tokens)].to(self.dev, non_blocking=True)
        self.buf.x[:bs] = x
        self.buf.inv_embed[:bs] = torch.rsqrt(x.float().pow(2).mean(-1) + self.w.cfg.eps)
        sched = self.schedule(bs, self.nsplit_for(bs, max(pos) + 1))
        self._counters.zero_()
        p = native.StepParams()
        p.ops, p.tasks, p.counters = sched.ops.data_ptr(), sched.tasks.data_ptr(), self._counters.data_ptr()
        p.targets = sched.targets.data_ptr()
        p.ntasks, p.bs, p.max_ctx = sched.ntasks, bs, cache.max_ctx
        if profile:
            self.last_profile = torch.zeros(sched.ntasks, 4, dtype=torch.int64, device=self.dev)
            self.last_schedule = sched
            p.prof = self.last_profile.data_ptr()
        for b, s in enumerate(slots):
            p.pos[b], p.slot[b], p.conv_par[b] = pos[b], s, cache.parity[s]
        self.lib.launch(p, self.num_blocks)
        for s in slots:
            cache.length[s] += 1
            cache.parity[s] ^= 1
        return self.buf.logits[:bs]
