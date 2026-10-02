"""Decode-step task graph: op descriptors, topologically ordered task list, counters.

Per GDN layer:   AB, QKVZ → GDN → OUT(+resid) → GATE_UP → DOWN(+resid)
Per attn layer:  QKV → ATTN → O(+resid) → GATE_UP → DOWN(+resid)
then LM_HEAD. Every op waits on the counters of the op(s) it reads; the
residual RMSNorms are folded into the consumer GEMMs (sum-of-squares partials
written by the RESID epilogues).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import native
from .cache import Cache
from .model import LinearLayerWeights, TiledLinear, Weights
from .native import (EPI_RESID, EPI_SILU_MUL, EPI_STORE, OP_ATTN, OP_GDN, OP_GEMM_BF16, OP_GEMM_FP8, MAX_BS)

MAX_NSPLIT = 64
ATT_GROUP = 6  # query heads per kv head


class Buffers:
    """Activation scratch shared by all decode steps."""

    def __init__(self, w: Weights, device: torch.device):
        c = w.cfg
        bf, f32 = torch.bfloat16, torch.float32
        z = lambda *shape, dtype=bf: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        self.x = z(MAX_BS, c.hidden)
        self.qkvz = z(MAX_BS, 16384)
        self.ab = z(MAX_BS, 96)
        self.mix = z(MAX_BS, 6144)
        self.h = z(MAX_BS, c.inter)
        self.qkv = z(MAX_BS, 14336)
        self.logits = z(MAX_BS, c.vocab)
        out_tiles = c.hidden // w.layers[0].out.tiling.tile_n
        down_tiles = c.hidden // w.layers[0].down.tiling.tile_n
        self.ss_embed = z(1, 8, dtype=f32)
        self.ss_attn = z(out_tiles, 8, dtype=f32)
        self.ss_mlp = z(down_tiles, 8, dtype=f32)
        self.part_ml = z(MAX_BS * c.nkv * MAX_NSPLIT * ATT_GROUP * 2, dtype=f32)
        self.part_acc = z(MAX_BS * c.nkv * MAX_NSPLIT * ATT_GROUP * c.head_dim, dtype=f32)


@dataclass
class Schedule:
    bs: int
    nsplit: int
    ops: torch.Tensor        # device bytes of QmkOpDesc[]
    tasks: torch.Tensor      # device int32 [n, 2]
    n_counters: int
    op_names: list[str] = field(default_factory=list)

    @property
    def ntasks(self) -> int:
        return self.tasks.shape[0]


def _ptr(t: torch.Tensor | None) -> int | None:
    return None if t is None else t.data_ptr()


class _Builder:
    def __init__(self, bs: int, nsplit: int):
        self.bs, self.nsplit = bs, nsplit
        self.ops: list[native.OpDesc] = []
        self.names: list[str] = []
        self.ntasks: list[int] = []
        self.next_ctr = native.CTR_TICKET + 1

    def _alloc(self, n: int = 1) -> int:
        c = self.next_ctr
        self.next_ctr += n
        return c

    def _add(self, name: str, op: native.OpDesc, ntasks: int, waits) -> tuple[int, int]:
        waits = [w for w in waits if w is not None]
        assert len(waits) <= 2
        for i in range(2):
            op.wait[i] = native.Dep(*waits[i]) if i < len(waits) else native.Dep(-1, 0)
        op.signal = self._alloc()
        op.ntasks = ntasks
        self.ops.append(op)
        self.names.append(name)
        self.ntasks.append(ntasks)
        return op.signal

    def gemm(self, name, tl: TiledLinear, *, x, ldx, epi, waits, norm_w=None, ss=None, out=None, ldo=0,
             resid=None, ss_out=None) -> tuple[int, int]:
        t = tl.tiling
        op = native.OpDesc()
        op.type = OP_GEMM_FP8 if t.fp8 else OP_GEMM_BF16
        g = op.gemm
        g.w, g.wscale = _ptr(tl.w), _ptr(tl.scales)
        g.x, g.ldx = _ptr(x), ldx
        g.norm_w = _ptr(norm_w)
        if ss is not None:
            g.ss_in, g.n_ss = _ptr(ss[0]), ss[1]
        g.out, g.ldo = _ptr(out), ldo
        g.resid, g.ss_out = _ptr(resid), _ptr(ss_out)
        g.tile_n, g.nchunks, g.epi = t.tile_n, t.nchunks, epi
        ctr = self._add(name, op, t.ntiles, waits)
        return ctr, t.ntiles

    def gdn(self, name, args: dict, waits) -> tuple[int, int]:
        op = native.OpDesc()
        op.type = OP_GDN
        for k, v in args.items():
            setattr(op.gdn, k, v)
        n = 48 * self.bs
        return self._add(name, op, n, waits), n

    def attn(self, name, args: dict, waits) -> tuple[int, int]:
        op = native.OpDesc()
        op.type = OP_ATTN
        for k, v in args.items():
            setattr(op.attn, k, v)
        op.attn.nsplit = self.nsplit
        op.attn.comb_ctr = self._alloc(4 * self.bs)
        groups = 4 * self.bs
        return self._add(name, op, groups * self.nsplit, waits), groups


def build(w: Weights, cache: Cache, buf: Buffers, bs: int, nsplit: int) -> Schedule:
    assert 1 <= bs <= MAX_BS and 1 <= nsplit <= MAX_NSPLIT
    c = w.cfg
    B = _Builder(bs, nsplit)
    H = c.hidden
    prev = None                      # (counter, value) the next layer's input waits on
    ss_prev = (buf.ss_embed, 1)
    n_out_tiles, n_down_tiles = buf.ss_attn.shape[0], buf.ss_mlp.shape[0]
    for i, lw in enumerate(w.layers):
        ki = cache.kind_index[i]
        if isinstance(lw, LinearLayerWeights):
            ab, n_ab = B.gemm(f"L{i}.ab", lw.ab, x=buf.x, ldx=H, norm_w=lw.ln1, ss=ss_prev, epi=EPI_STORE,
                              out=buf.ab, ldo=96, waits=[prev])
            qkvz, n_qkvz = B.gemm(f"L{i}.qkvz", lw.qkvz, x=buf.x, ldx=H, norm_w=lw.ln1, ss=ss_prev, epi=EPI_STORE,
                                  out=buf.qkvz, ldo=16384, waits=[prev])
            mix = B.gdn(f"L{i}.gdn", dict(
                qkvz=_ptr(buf.qkvz), ab=_ptr(buf.ab), conv_w=_ptr(lw.conv_w), conv_state=_ptr(cache.conv[ki]),
                state=_ptr(cache.state[ki]), a_log=_ptr(lw.a_log), dt_bias=_ptr(lw.dt_bias),
                norm_w=_ptr(lw.norm_w), out=_ptr(buf.mix)), waits=[(qkvz, n_qkvz), (ab, n_ab)])
        else:
            qkv, n_qkv = B.gemm(f"L{i}.qkv", lw.qkv, x=buf.x, ldx=H, norm_w=lw.ln1, ss=ss_prev, epi=EPI_STORE,
                                out=buf.qkv, ldo=14336, waits=[prev])
            mix = B.attn(f"L{i}.attn", dict(
                qkv=_ptr(buf.qkv), q_norm_w=_ptr(lw.q_norm), k_norm_w=_ptr(lw.k_norm), rope_cos=_ptr(cache.cos),
                rope_sin=_ptr(cache.sin), kcache=_ptr(cache.k[ki]), vcache=_ptr(cache.v[ki]),
                part_ml=_ptr(buf.part_ml), part_acc=_ptr(buf.part_acc), out=_ptr(buf.mix)), waits=[(qkv, n_qkv)])
        out, n_out = B.gemm(f"L{i}.out", lw.out, x=buf.mix, ldx=6144, epi=EPI_RESID, resid=buf.x, ldo=H,
                            ss_out=buf.ss_attn, waits=[mix])
        assert n_out == n_out_tiles
        gu, n_gu = B.gemm(f"L{i}.gate_up", lw.gate_up, x=buf.x, ldx=H, norm_w=lw.ln2, ss=(buf.ss_attn, n_out),
                          epi=EPI_SILU_MUL, out=buf.h, ldo=c.inter, waits=[(out, n_out)])
        down, n_down = B.gemm(f"L{i}.down", lw.down, x=buf.h, ldx=c.inter, epi=EPI_RESID, resid=buf.x, ldo=H,
                              ss_out=buf.ss_mlp, waits=[(gu, n_gu)])
        assert n_down == n_down_tiles
        prev, ss_prev = (down, n_down), (buf.ss_mlp, n_down)
    B.gemm("lm_head", w.lm_head, x=buf.x, ldx=H, norm_w=w.final_norm, ss=ss_prev, epi=EPI_STORE,
           out=buf.logits, ldo=c.vocab, waits=[prev])

    task_op = torch.repeat_interleave(torch.arange(len(B.ops), dtype=torch.int32), torch.tensor(B.ntasks))
    task_idx = torch.cat([torch.arange(n, dtype=torch.int32) for n in B.ntasks])
    dev = w.device
    return Schedule(
        bs=bs, nsplit=nsplit,
        ops=native.ops_to_device(B.ops, dev),
        tasks=native.tasks_to_device(torch.stack([task_op, task_idx], 1), dev),
        n_counters=B.next_ctr,
        op_names=B.names,
    )
