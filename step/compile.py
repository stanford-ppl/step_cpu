"""Build pipeline: write C++ source, compile via torch.utils.cpp_extension.load()."""

from __future__ import annotations

import hashlib
import os
import pathlib
from typing import Callable

import torch
import torch.utils.cpp_extension


def build_extension(func_name: str, cpp_source: str) -> Callable:
    """Compile a C++ source string into a loaded PyTorch extension.

    Uses torch.utils.cpp_extension.load() with source-hash caching.
    Returns a callable that dispatches to torch.ops.step_ops.<func_name>_step.
    """
    op_name = func_name + "_step"

    # Build directory for caching
    cache_dir = pathlib.Path.home() / ".cache" / "mocha" / func_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / f"{op_name}.cpp"
    cpp_path.write_text(cpp_source)

    # Compile — is_python_module=False because we use TORCH_LIBRARY_FRAGMENT
    # which registers ops via torch.ops rather than exporting a Python module.
    torch.utils.cpp_extension.load(
        name=f"step_{func_name}",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    # Retrieve the registered op
    op_fn = getattr(torch.ops.step_ops, op_name)

    def wrapper(*args: torch.Tensor) -> torch.Tensor:
        return op_fn(*args)

    wrapper.__name__ = func_name
    wrapper.__qualname__ = func_name
    return wrapper
