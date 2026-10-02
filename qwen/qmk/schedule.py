"""Decode-step task graph: op descriptors, topologically ordered task list, counters.

Per GDN layer:   AB, QKVZ → GDN → OUT(+resid) → GATE_UP → DOWN(+resid)
Per attn layer:  QKV → ATTN → O(+resid) → GATE_UP → DOWN(+resid)
then LM_HEAD.

Dependencies are as fine as the data allows, so an op starts on the part of
its input that is already done instead of waiting for the whole previous op:
  QKVZ (rows grouped per key head) → GDN head h        : that head's 32 tiles
  GDN heads 24j..24j+23            → OUT K-split j
  QKV (rows grouped per kv head)   → attention group g : that group's 112 tiles
  attention group g                → O K-split g
  GATE_UP tiles of hidden half j   → DOWN K-split j
Only the two RMSNorms (before GATE_UP and before the next layer) need the
whole residual row, so those stay full barriers. The norms are folded into
the consumer GEMMs: RESID epilogues write per-tile sums of squares and the
last tile turns them into each row's 1/rms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import native
from .cache import Cache
from .model import KSPLIT_DOWN, KSPLIT_O, KSPLIT_OUT, LinearLayerWeights, TiledLinear, Weights
from .native import (EPI_RESID, EPI_SILU_MUL, EPI_STORE, MAX_BS, NO_DEP, OP_ATTN, OP_GDN, OP_GEMM_BF16,
                     OP_GEMM_FP8, WHOLE, Dep, key)

MAX_NSPLIT = 64
ATT_GROUP = 6           # query heads per kv head
LIN_GROUP_ROWS = 1024   # fused QKVZ rows per key head (abi.h QMK_LIN_GROUP)
ATT_GROUP_ROWS = 3584   # fused QKV rows per kv head (abi.h QMK_ATT_GROUP)
MAX_KSPLIT = max(KSPLIT_OUT, KSPLIT_O, KSPLIT_DOWN)


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
        self.ss_attn = z(out_tiles, 8, dtype=f32)   # per-tile sums of squares
        self.ss_mlp = z(down_tiles, 8, dtype=f32)
        self.inv_embed = z(8, dtype=f32)            # 1/rms of each row: embeddings,
        self.inv_attn = z(8, dtype=f32)             # after the mixer residual add,
        self.inv_mlp = z(8, dtype=f32)              # after the MLP residual add
        self.part_mix = z(MAX_KSPLIT, 8, c.hidden, dtype=f32)   # OUT / O split-K partials
        self.part_down = z(MAX_KSPLIT, 8, c.hidden, dtype=f32)
        self.part_ml = z(MAX_BS * c.nkv * MAX_NSPLIT * ATT_GROUP * 2, dtype=f32)
        self.part_acc = z(MAX_BS * c.nkv * MAX_NSPLIT * ATT_GROUP * c.head_dim, dtype=f32)


@dataclass
class Schedule:
    bs: int
    nsplit: int
    ops: torch.Tensor        # device bytes of QmkOpDesc[]
    tasks: torch.Tensor      # device int32 [n, 2]
    targets: torch.Tensor    # device int32 [n_counters]
    op_names: list[str] = field(default_factory=list)

    @property
    def ntasks(self) -> int:
        return self.tasks.shape[0]

    @property
    def n_counters(self) -> int:
        return self.targets.numel()


def _ptr(t: torch.Tensor | None) -> int | None:
    return None if t is None else t.data_ptr()


@dataclass(frozen=True)
class Signal:
    """Counters an op bumps: base index and the key mapping units onto them."""
    ctr: int
    key: native.Key

    def on(self, key: native.Key | None = None) -> Dep:
        """Dependency on these counters; `key` maps the *waiting* task's index to the counter."""
        return Dep(self.ctr, key if key is not None else WHOLE)


class _Builder:
    def __init__(self, bs: int, nsplit: int):
        self.bs, self.nsplit = bs, nsplit
        self.ops: list[native.OpDesc] = []
        self.names: list[str] = []
        self.ntasks: list[int] = []
        self.targets: list[int] = [0]  # counter 0: ticket (no target)

    def _alloc(self, n: int) -> int:
        base = len(self.targets)
        self.targets += [0] * n
        return base

    def _add(self, name: str, op: native.OpDesc, ntasks: int, waits: list[Dep], sig_key: native.Key,
             units: range) -> Signal:
        """Register op; it signals once per unit in `units` on counter base + sig_key(unit)."""
        assert len(waits) <= 2
        for i in range(2):
            op.wait[i] = waits[i] if i < len(waits) else NO_DEP
        nkeys = max(sig_key(u) for u in units) + 1
        base = self._alloc(nkeys)
        for u in units:
            self.targets[base + sig_key(u)] += 1
        op.signal = Dep(base, sig_key)
        op.ntasks = ntasks
        self.ops.append(op)
        self.names.append(name)
        self.ntasks.append(ntasks)
        return Signal(base, sig_key)

    def task_order(self) -> torch.Tensor:
        """[n, 3] int32 (op, idx, flags), op-major.

        (Pulling GDN/attention and out-proj tasks forward into the producing
        GEMM's stream was measured slower: they wait on unfinished inputs while
        holding a block, and GDN state loads crawl under full DRAM load.)
        """
        task_op = torch.repeat_interleave(torch.arange(len(self.ops), dtype=torch.int32), torch.tensor(self.ntasks))
        task_idx = torch.cat([torch.arange(n, dtype=torch.int32) for n in self.ntasks])
        gemm = torch.tensor([op.type in (OP_GEMM_FP8, OP_GEMM_BF16) for op in self.ops])
        flags = (~gemm[task_op.long()]).to(torch.int32) * native.TASK_WAIT_IDLE
        return torch.stack([task_op, task_idx, flags], 1)

    def gemm(self, name, tl: TiledLinear, *, x, ldx, epi, waits, sig_key=WHOLE, ksplit=1, partial=None,
             norm_w=None, inv_in=None, out=None, ldo=0, resid=None, ss_out=None, inv_out=None) -> Signal:
        t = tl.tiling
        op = native.OpDesc()
        op.type = OP_GEMM_FP8 if t.fp8 else OP_GEMM_BF16
        g = op.gemm
        g.w, g.wscale = _ptr(tl.w), _ptr(tl.scales)
        g.x, g.ldx = _ptr(x), ldx
        assert (norm_w is None) == (inv_in is None)
        g.norm_w, g.inv_in = _ptr(norm_w), _ptr(inv_in)
        g.out, g.ldo = _ptr(out), ldo
        g.resid, g.ss_out = _ptr(resid), _ptr(ss_out)
        units = range(t.ntiles)
        if epi == EPI_RESID:
            # signals once, from the last tile, after it has written inv_out
            assert inv_out is not None and sig_key is WHOLE
            g.inv_out = _ptr(inv_out)
            g.norm_ctr = self._alloc(1)
            units = range(1)
        g.tile_n, g.nchunks, g.epi, g.ntiles, g.ksplit = t.tile_n, t.nchunks, epi, t.ntiles, ksplit
        assert t.nchunks >= ksplit
        if ksplit > 1:
            g.partial = _ptr(partial)
            g.tile_ctr = self._alloc(t.ntiles)  # arrival counters, no target
        return self._add(name, op, t.ntiles * ksplit, waits, sig_key, units)

    def gdn(self, name, args: dict, waits, sig_key) -> Signal:
        op = native.OpDesc()
        op.type = OP_GDN
        for k, v in args.items():
            setattr(op.gdn, k, v)
        n = 48 * self.bs
        return self._add(name, op, n, waits, sig_key, range(n))

    def attn(self, name, args: dict, waits, sig_key) -> Signal:
        op = native.OpDesc()
        op.type = OP_ATTN
        for k, v in args.items():
            setattr(op.attn, k, v)
        op.attn.nsplit = self.nsplit
        groups = 4 * self.bs
        op.attn.comb_ctr = self._alloc(groups)
        # the combining task of group gr may be any of its splits; the key ignores the split
        return self._add(name, op, groups * self.nsplit, waits, sig_key, range(0, groups * self.nsplit, self.nsplit))


def build(w: Weights, cache: Cache, buf: Buffers, bs: int, nsplit: int) -> Schedule:
    assert 1 <= bs <= MAX_BS and 1 <= nsplit <= MAX_NSPLIT
    c = w.cfg
    B = _Builder(bs, nsplit)
    H = c.hidden
    nv = c.lin_nv
    prev: Signal | None = None         # last DOWN of the previous layer
    inv_prev = buf.inv_embed
    for i, lw in enumerate(w.layers):
        ki = cache.kind_index[i]
        start = [prev.on()] if prev else []
        if isinstance(lw, LinearLayerWeights):
            ab = B.gemm(f"L{i}.ab", lw.ab, x=buf.x, ldx=H, norm_w=lw.ln1, inv_in=inv_prev, epi=EPI_STORE,
                        out=buf.ab, ldo=96, waits=start)
            tiles_per_head_group = LIN_GROUP_ROWS // lw.qkvz.tiling.tile_n
            qkvz = B.gemm(f"L{i}.qkvz", lw.qkvz, x=buf.x, ldx=H, norm_w=lw.ln1, inv_in=inv_prev, epi=EPI_STORE,
                          out=buf.qkvz, ldo=16384, waits=start, sig_key=key(div1=tiles_per_head_group))
            heads_per_split = nv // KSPLIT_OUT
            gdn = B.gdn(f"L{i}.gdn", dict(
                qkvz=_ptr(buf.qkvz), ab=_ptr(buf.ab), conv_w=_ptr(lw.conv_w), conv_state=_ptr(cache.conv[ki]),
                state=_ptr(cache.state[ki]), a_log=_ptr(lw.a_log), dt_bias=_ptr(lw.dt_bias),
                norm_w=_ptr(lw.norm_w), out=_ptr(buf.mix)),
                # task h*bs+b waits on its key head's QKVZ tiles
                waits=[qkvz.on(key(div1=bs, div2=nv // c.lin_nk)), ab.on()],
                sig_key=key(div1=bs, div2=heads_per_split))
            ot = lw.out.tiling
            assert all((j * ot.nchunks // KSPLIT_OUT) * ot.kc == j * heads_per_split * c.lin_dv
                       for j in range(KSPLIT_OUT + 1))
            ntiles_out = ot.ntiles
            out = B.gemm(f"L{i}.out", lw.out, x=buf.mix, ldx=6144, epi=EPI_RESID, resid=buf.x, ldo=H,
                         ss_out=buf.ss_attn, inv_out=buf.inv_attn, ksplit=KSPLIT_OUT, partial=buf.part_mix,
                         waits=[gdn.on(key(div1=ntiles_out))])
        else:
            tiles_per_group = ATT_GROUP_ROWS // lw.qkv.tiling.tile_n
            qkv = B.gemm(f"L{i}.qkv", lw.qkv, x=buf.x, ldx=H, norm_w=lw.ln1, inv_in=inv_prev, epi=EPI_STORE,
                         out=buf.qkv, ldo=14336, waits=start, sig_key=key(div1=tiles_per_group))
            group_key = key(div1=nsplit * bs)  # task (g*bs+b)*nsplit+s → g
            attn = B.attn(f"L{i}.attn", dict(
                qkv=_ptr(buf.qkv), q_norm_w=_ptr(lw.q_norm), k_norm_w=_ptr(lw.k_norm), rope_cos=_ptr(cache.cos),
                rope_sin=_ptr(cache.sin), kcache=_ptr(cache.k[ki]), vcache=_ptr(cache.v[ki]),
                part_ml=_ptr(buf.part_ml), part_acc=_ptr(buf.part_acc), out=_ptr(buf.mix)),
                waits=[qkv.on(group_key)], sig_key=group_key)
            ot = lw.out.tiling
            assert KSPLIT_O == c.nkv and all(
                (j * ot.nchunks // KSPLIT_O) * ot.kc == j * ATT_GROUP * c.head_dim for j in range(KSPLIT_O + 1))
            ntiles_out = ot.ntiles
            out = B.gemm(f"L{i}.out", lw.out, x=buf.mix, ldx=6144, epi=EPI_RESID, resid=buf.x, ldo=H,
                         ss_out=buf.ss_attn, inv_out=buf.inv_attn, ksplit=KSPLIT_O, partial=buf.part_mix,
                         waits=[attn.on(key(div1=ntiles_out))])
        n_out = lw.out.tiling.ntiles
        assert n_out == buf.ss_attn.shape[0]
        # gate_up tile t produces hidden features [t*TN/2, (t+1)*TN/2); key = the DOWN K-split reading them
        feats_per_tile = lw.gate_up.tiling.tile_n // 2
        down_t = lw.down.tiling
        split_feats = (1 * down_t.nchunks // KSPLIT_DOWN) * down_t.kc  # first K-split boundary (KSPLIT_DOWN == 2)
        assert KSPLIT_DOWN == 2
        gu = B.gemm(f"L{i}.gate_up", lw.gate_up, x=buf.x, ldx=H, norm_w=lw.ln2, inv_in=buf.inv_attn,
                    epi=EPI_SILU_MUL, out=buf.h, ldo=c.inter, waits=[out.on()],
                    sig_key=key(div1=split_feats // feats_per_tile, kmax=KSPLIT_DOWN - 1))
        down = B.gemm(f"L{i}.down", lw.down, x=buf.h, ldx=c.inter, epi=EPI_RESID, resid=buf.x, ldo=H,
                      ss_out=buf.ss_mlp, inv_out=buf.inv_mlp, ksplit=KSPLIT_DOWN, partial=buf.part_down,
                      waits=[gu.on(key(div1=down_t.ntiles))])
        assert down_t.ntiles == buf.ss_mlp.shape[0]
        prev, inv_prev = down, buf.inv_mlp
    B.gemm("lm_head", w.lm_head, x=buf.x, ldx=H, norm_w=w.final_norm, inv_in=inv_prev, epi=EPI_STORE,
           out=buf.logits, ldo=c.vocab, waits=[prev.on()])

    dev = w.device
    sched = Schedule(
        bs=bs, nsplit=nsplit,
        ops=native.ops_to_device(B.ops, dev),
        tasks=native.tasks_to_device(B.task_order(), dev),
        targets=torch.tensor(B.targets, dtype=torch.int32, device=dev),
        op_names=B.names,
    )
    check_order(sched, B.ops)
    return sched


def check_order(sched: Schedule, ops: list[native.OpDesc]) -> None:
    """Assert the task list is topological: run tasks one at a time in list order
    and check every wait is already satisfied. (The kernel relies on this to be
    deadlock-free: a claimed task only ever waits on earlier-claimed tasks.)"""
    targets = sched.targets.cpu().tolist()
    counters = [0] * len(targets)
    pending: dict[tuple[int, int], int] = {}  # (op, unit) → splits seen
    for op_i, idx, *_ in sched.tasks.cpu().tolist():
        op = ops[op_i]
        for d in op.wait:
            if d.ctr >= 0:
                c = d.ctr + d.key(idx)
                assert counters[c] >= targets[c], f"task ({op_i},{idx}) waits on counter {c} before it can complete"
        if op.type in (native.OP_GEMM_FP8, native.OP_GEMM_BF16):
            ks, nt = op.gemm.ksplit, op.gemm.ntiles
            unit = idx % nt
            pending[(op_i, unit)] = pending.get((op_i, unit), 0) + 1
            if pending[(op_i, unit)] < ks:
                continue
            if op.gemm.epi == EPI_RESID:  # only the last finished tile signals
                pending[(op_i, -1)] = pending.get((op_i, -1), 0) + 1
                if pending[(op_i, -1)] < nt:
                    continue
                unit = 0
        elif op.type == native.OP_ATTN:
            ns = op.attn.nsplit
            unit = idx // ns * ns
            pending[(op_i, unit)] = pending.get((op_i, unit), 0) + 1
            if pending[(op_i, unit)] < ns:
                continue
            unit = idx
        else:
            unit = idx
        counters[op.signal.ctr + op.signal.key(unit)] += 1
    bad = [i for i, (c, t) in enumerate(zip(counters, targets)) if c < t]
    assert not bad, f"counters never reach their targets: {bad[:5]}"
