"""Build pipeline: write C++ source, compile via torch.utils.cpp_extension.load()."""

from __future__ import annotations

import hashlib
import os
import pathlib
from typing import Callable

import torch
import torch.utils.cpp_extension


def detect_isa() -> str:
    """Detect the best supported SIMD ISA: 'avx512' or 'avx2'.

    Override with the STEP_ISA environment variable (e.g. STEP_ISA=avx2).
    """
    override = os.environ.get("STEP_ISA")
    if override in ("avx512", "avx2"):
        return override
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        if "avx512f" in cpuinfo:
            return "avx512"
    except OSError:
        pass
    return "avx2"


def build_extension(func_name: str, cpp_source: str, isa: str | None = None,
                    reuse_cached: bool = False) -> Callable:
    """Compile a C++ source string into a loaded PyTorch extension.

    Uses torch.utils.cpp_extension.load() with source-hash caching.
    Returns a callable that dispatches to torch.ops.step_ops.<func_name>_step.

    Args:
        func_name: Base name for the generated op (e.g., "gpt2_mlp_fused6").
        cpp_source: The C++ source string to compile.
        isa: SIMD ISA to compile for ('avx512', 'avx2', or None for no SIMD flags).
        reuse_cached: If True, skip recompilation when source hash matches.
    """
    op_name = func_name + "_step"

    # Include ISA in cache path to prevent stale .so collisions
    cache_name = f"{func_name}_{isa}" if isa else func_name
    cache_dir = pathlib.Path.home() / ".cache" / "mocha" / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_hash = hashlib.sha256(cpp_source.encode()).hexdigest()[:16]
    hash_file = cache_dir / "source_hash"
    so_files = list(cache_dir.glob("*.so"))

    if reuse_cached and hash_file.exists() and so_files:
        if hash_file.read_text().strip() == src_hash:
            torch.ops.load_library(str(so_files[0]))
            op_fn = getattr(torch.ops.step_ops, op_name)
            def wrapper(*args):
                return op_fn(*args)
            wrapper.__name__ = func_name
            wrapper.__qualname__ = func_name
            return wrapper

    cpp_path = cache_dir / f"{op_name}.cpp"
    cpp_path.write_text(cpp_source)

    extra_cflags = ["-O3", "-std=c++17"]
    extra_ldflags = []
    if isa == "avx512":
        extra_cflags += ["-march=native", "-fopenmp", "-mavx512f", "-mfma", "-ffast-math"]
        extra_ldflags += ["-fopenmp"]
    elif isa == "avx2":
        extra_cflags += ["-march=native", "-fopenmp", "-mavx2", "-mfma", "-ffast-math"]
        extra_ldflags += ["-fopenmp"]

    # Compile — is_python_module=False because we use TORCH_LIBRARY_FRAGMENT
    # which registers ops via torch.ops rather than exporting a Python module.
    torch.utils.cpp_extension.load(
        name=f"step_{func_name}",
        sources=[str(cpp_path)],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    # Save hash after successful compilation
    hash_file.write_text(src_hash)

    # Retrieve the registered op
    op_fn = getattr(torch.ops.step_ops, op_name)

    def wrapper(*args: torch.Tensor) -> torch.Tensor:
        return op_fn(*args)

    wrapper.__name__ = func_name
    wrapper.__qualname__ = func_name
    return wrapper
