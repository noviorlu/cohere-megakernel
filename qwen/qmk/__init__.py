"""Qwen3.8-27B-FP8 decode megakernel for RTX 5090 (sm_120)."""

import os

# Weights take ~27 of the 32 GB; avoid losing the rest to allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Triton (used by fla in prefill) compiles a small C shim against Python.h;
# this machine's system Python has no -dev package, so point at an unpacked copy.
_PY_HEADERS = "/mnt/data/envs/py312-dev-headers/root/usr/include"
if os.path.isdir(_PY_HEADERS) and "C_INCLUDE_PATH" not in os.environ:
    os.environ["C_INCLUDE_PATH"] = ":".join(
        [f"{_PY_HEADERS}/python3.12", f"{_PY_HEADERS}/x86_64-linux-gnu/python3.12", _PY_HEADERS]
    )
