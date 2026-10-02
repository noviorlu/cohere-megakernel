"""Build/load libqmk.so and mirror its ABI (csrc/abi.h) with ctypes."""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import subprocess
from pathlib import Path

import torch

QWEN_DIR = Path(__file__).resolve().parent.parent
CSRC = QWEN_DIR / "csrc"
BUILD = QWEN_DIR / "build"
NVCC = os.environ.get("QMK_NVCC", "/mnt/data/envs/cuda13/bin/nvcc")
HOST_CXX = os.environ.get("QMK_HOST_CXX", "/usr/bin/g++")

# Mirrors of abi.h constants used on the Python side.
MAX_BS = 8
CONSUMER_WARPS = 8
CHUNK_BYTES = 16384

OP_GEMM_FP8, OP_GEMM_BF16, OP_GDN, OP_ATTN = 0, 1, 2, 3
EPI_STORE, EPI_SILU_MUL, EPI_RESID = 0, 1, 2
CTR_TICKET = 0


BIG = 1 << 30


class Key(C.Structure):
    """key(i) = min(((i // div1) % mod) // div2, kmax); mod == 0 skips the modulo."""
    _fields_ = [("div1", C.c_int32), ("mod", C.c_int32), ("div2", C.c_int32), ("kmax", C.c_int32)]

    def __call__(self, i: int) -> int:
        v = i // self.div1
        if self.mod:
            v %= self.mod
        return min(v // self.div2, self.kmax)


def key(div1: int = BIG, mod: int = 0, div2: int = 1, kmax: int = BIG) -> Key:
    return Key(div1, mod, div2, kmax)


WHOLE = key()  # every task maps to the same counter


class Dep(C.Structure):
    _fields_ = [("ctr", C.c_int32), ("key", Key)]


NO_DEP = Dep(-1, WHOLE)


class GemmArgs(C.Structure):
    _fields_ = [
        ("w", C.c_void_p),
        ("wscale", C.c_void_p),
        ("x", C.c_void_p),
        ("norm_w", C.c_void_p),
        ("inv_in", C.c_void_p),
        ("out", C.c_void_p),
        ("resid", C.c_void_p),
        ("ss_out", C.c_void_p),
        ("inv_out", C.c_void_p),
        ("partial", C.c_void_p),
        ("ldx", C.c_int32),
        ("norm_ctr", C.c_int32),
        ("tile_n", C.c_int32),
        ("nchunks", C.c_int32),
        ("epi", C.c_int32),
        ("ldo", C.c_int32),
        ("ntiles", C.c_int32),
        ("ksplit", C.c_int32),
        ("tile_ctr", C.c_int32),
        ("pad_", C.c_int32),
    ]


class GdnArgs(C.Structure):
    _fields_ = [
        ("qkvz", C.c_void_p),
        ("ab", C.c_void_p),
        ("conv_w", C.c_void_p),
        ("conv_state", C.c_void_p),
        ("state", C.c_void_p),
        ("a_log", C.c_void_p),
        ("dt_bias", C.c_void_p),
        ("norm_w", C.c_void_p),
        ("out", C.c_void_p),
    ]


class AttnArgs(C.Structure):
    _fields_ = [
        ("qkv", C.c_void_p),
        ("q_norm_w", C.c_void_p),
        ("k_norm_w", C.c_void_p),
        ("rope_cos", C.c_void_p),
        ("rope_sin", C.c_void_p),
        ("kcache", C.c_void_p),
        ("vcache", C.c_void_p),
        ("part_ml", C.c_void_p),
        ("part_acc", C.c_void_p),
        ("out", C.c_void_p),
        ("nsplit", C.c_int32),
        ("comb_ctr", C.c_int32),
    ]


class OpArgs(C.Union):
    _fields_ = [("gemm", GemmArgs), ("gdn", GdnArgs), ("attn", AttnArgs)]


class OpDesc(C.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", C.c_int32),
        ("ntasks", C.c_int32),
        ("wait", Dep * 2),
        ("signal", Dep),
        ("u", OpArgs),
    ]


TASK_WAIT_IDLE = 1


class Task(C.Structure):
    _fields_ = [("op", C.c_int32), ("idx", C.c_int32), ("flags", C.c_int32), ("pad_", C.c_int32)]


class StepParams(C.Structure):
    _fields_ = [
        ("ops", C.c_void_p),
        ("tasks", C.c_void_p),
        ("counters", C.c_void_p),
        ("targets", C.c_void_p),
        ("prof", C.c_void_p),
        ("ntasks", C.c_int32),
        ("bs", C.c_int32),
        ("max_ctx", C.c_int32),
        ("l2_prefetch_chunks", C.c_int32),
        ("pos", C.c_int32 * MAX_BS),
        ("slot", C.c_int32 * MAX_BS),
        ("conv_par", C.c_int32 * MAX_BS),
    ]


def _sources() -> list[Path]:
    return sorted(CSRC.glob("*.cu*")) + sorted(CSRC.glob("*.h"))


def build(force: bool = False) -> Path:
    """Compile csrc/ into build/libqmk-<hash>.so (cached by source hash)."""
    h = hashlib.sha256()
    for p in _sources():
        h.update(p.name.encode())
        h.update(p.read_bytes())
    flags = ["-std=c++17", "-O3", "-arch=sm_120a", "-lineinfo", "-shared", "-Xcompiler", "-fPIC"]
    h.update(" ".join(flags).encode())
    out = BUILD / f"libqmk-{h.hexdigest()[:16]}.so"
    if out.exists() and not force:
        return out
    BUILD.mkdir(exist_ok=True)
    cmd = [NVCC, "-ccbin", HOST_CXX, *flags, "-o", str(out), str(CSRC / "megakernel.cu")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{res.stdout}\n{res.stderr}")
    return out


class Lib:
    """Loaded libqmk with checked ABI."""

    def __init__(self, path: Path | None = None):
        self.path = path or build()
        lib = C.CDLL(str(self.path))
        lib.qmk_abi_layout.argtypes = [C.POINTER(C.c_int64), C.c_int]
        lib.qmk_launch.argtypes = [C.POINTER(StepParams), C.c_int, C.c_void_p]
        lib.qmk_e4m3_to_bf16.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
        lib.qmk_error_string.restype = C.c_char_p
        self.lib = lib
        self._check_abi()
        self.smem_bytes = lib.qmk_smem_bytes()

    def _check_abi(self) -> None:
        buf = (C.c_int64 * 16)()
        n = self.lib.qmk_abi_layout(buf, 16)
        native = list(buf[:n])
        mine = [
            C.sizeof(OpDesc), OpDesc.u.offset, C.sizeof(GemmArgs), C.sizeof(GdnArgs), C.sizeof(AttnArgs),
            C.sizeof(StepParams), GemmArgs.ldx.offset, AttnArgs.nsplit.offset, StepParams.pos.offset,
            OpDesc.signal.offset, GemmArgs.ksplit.offset, GemmArgs.norm_ctr.offset,
        ]
        if native[: len(mine)] != mine:
            raise RuntimeError(f"ABI mismatch: native={native[:len(mine)]} ctypes={mine}")

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"CUDA error {code}: {self.lib.qmk_error_string(code).decode()}")

    def launch(self, params: StepParams, num_blocks: int) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.qmk_launch(C.byref(params), num_blocks, C.c_void_p(stream)))

    def e4m3_to_bf16(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.numel(), dtype=torch.bfloat16, device=x.device)
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.qmk_e4m3_to_bf16(x.data_ptr(), out.data_ptr(), x.numel(), C.c_void_p(stream)))
        return out


def ops_to_device(ops: list[OpDesc], device: torch.device) -> torch.Tensor:
    raw = (OpDesc * len(ops))(*ops)
    host = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return host.to(device)


def tasks_to_device(tasks: torch.Tensor, device: torch.device) -> torch.Tensor:
    """tasks: int32 [n, 2] (op, idx) or [n, 3] (op, idx, flags) → QmkTask[n]."""
    assert tasks.dtype == torch.int32 and tasks.shape[1] in (2, 3)
    full = torch.zeros(tasks.shape[0], 4, dtype=torch.int32)
    full[:, : tasks.shape[1]] = tasks
    return full.to(device)
