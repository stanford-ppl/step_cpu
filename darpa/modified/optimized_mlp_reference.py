#!/usr/bin/env python
"""
Hand-written optimized GPT2 MLP C++ kernel — reference implementation.

This script:
  1. Defines an optimized C++ kernel inline (no transpose, no tiling, fused GELU)
  2. Compiles it via torch.utils.cpp_extension.load()
  3. Validates correctness against PyTorch native MLP
  4. Benchmarks against both PyTorch baseline and the current STeP kernel

The key insight: the STeP codegen does W.T.contiguous() then mm(a, w.T) inside
the inner loop — the two transposes cancel. The net computation is just
hidden_states @ W, the original Conv1D forward. This kernel emits that directly.

Usage:
  python3 darpa/modified/optimized_mlp_reference.py
"""

import math
import os
import sys
import time
import pathlib
import hashlib

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

# ---------------------------------------------------------------------------
#  Project setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])  # mocha/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
#  Optimized C++ kernel source
# ---------------------------------------------------------------------------
_FUSED_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>

torch::Tensor gpt2_mlp_fused_step(
    torch::Tensor hidden_states, torch::Tensor W_fc,
    torch::Tensor b_fc, torch::Tensor W_proj, torch::Tensor b_proj) {

    // c_fc: hidden_states @ W_fc + b_fc  (Conv1D layout: W is [in, out])
    auto h = torch::mm(hidden_states, W_fc);
    h.add_(b_fc);

    // GELU (tanh approximation) — single fused call
    h = at::gelu(h, "tanh");

    // c_proj: h @ W_proj + b_proj
    auto output = torch::mm(h, W_proj);
    output.add_(b_proj);

    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused_step", gpt2_mlp_fused_step);
}
"""


# ---------------------------------------------------------------------------
#  Build the fused kernel
# ---------------------------------------------------------------------------
def build_fused_kernel():
    """Compile the fused C++ kernel and return a callable."""
    cache_dir = pathlib.Path.home() / ".cache" / "mocha" / "gpt2_mlp_fused"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_mlp_fused_step.cpp"
    cpp_path.write_text(_FUSED_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_mlp_fused",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_mlp_fused_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


# ---------------------------------------------------------------------------
#  Benchmark helper
# ---------------------------------------------------------------------------
def bench(fn, warmup=5, iters=50):
    """Run *fn* with warmup, return median time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    from transformers import AutoModelForCausalLM

    print("Loading distilgpt2 weights ...")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    model.eval()
    mlp = model.transformer.h[0].mlp

    W_fc = mlp.c_fc.weight.detach()      # [768, 3072] Conv1D
    b_fc = mlp.c_fc.bias.detach()        # [3072]
    W_proj = mlp.c_proj.weight.detach()  # [3072, 768] Conv1D
    b_proj = mlp.c_proj.bias.detach()    # [768]

    print(f"  W_fc   {tuple(W_fc.shape)}   W_proj {tuple(W_proj.shape)}")

    # Build kernels
    print("\nCompiling fused C++ kernel ...")
    fused_kernel = build_fused_kernel()

    print("Compiling STeP kernel ...")
    import step
    from causal_language_modeling import _build_gpt2mlp_replacement
    step_kernel = _build_gpt2mlp_replacement()

    # -----------------------------------------------------------------------
    #  Correctness validation
    # -----------------------------------------------------------------------
    print("\n=== Correctness Validation ===")
    M_VALUES = [1, 5, 20, 128]
    for M in M_VALUES:
        x = torch.randn(M, 768)

        # PyTorch reference
        ref = torch.mm(x, W_fc) + b_fc
        ref = F.gelu(ref, approximate="tanh")
        ref = torch.mm(ref, W_proj) + b_proj

        # Fused kernel
        fused_out = fused_kernel(x, W_fc, b_fc, W_proj, b_proj)

        # STeP kernel
        step_out = step_kernel(x, W_fc, b_fc, W_proj, b_proj)

        err_fused = (fused_out - ref).abs().max().item()
        err_step = (step_out - ref).abs().max().item()

        status_fused = "PASS" if err_fused < 1e-4 else "FAIL"
        status_step = "PASS" if err_step < 1e-4 else "FAIL"
        print(f"  M={M:>3}:  fused max_err={err_fused:.2e} [{status_fused}]"
              f"   step max_err={err_step:.2e} [{status_step}]")

    # -----------------------------------------------------------------------
    #  Benchmark
    # -----------------------------------------------------------------------
    print("\n=== Performance Benchmark ===")
    print(f"  {'M':>5}  {'PyTorch (ms)':>13}  {'Fused (ms)':>11}  {'STeP (ms)':>10}"
          f"  {'Fused/Torch':>12}  {'STeP/Torch':>11}")
    print(f"  {'---':>5}  {'---':>13}  {'---':>11}  {'---':>10}"
          f"  {'---':>12}  {'---':>11}")

    for M in M_VALUES:
        x = torch.randn(M, 768)

        t_pytorch = bench(lambda: torch.mm(F.gelu(torch.mm(x, W_fc) + b_fc,
                                                   approximate="tanh"),
                                            W_proj) + b_proj)
        t_fused = bench(lambda: fused_kernel(x, W_fc, b_fc, W_proj, b_proj))
        t_step = bench(lambda: step_kernel(x, W_fc, b_fc, W_proj, b_proj))

        r_fused = t_fused / t_pytorch if t_pytorch > 0 else float("inf")
        r_step = t_step / t_pytorch if t_pytorch > 0 else float("inf")

        print(f"  {M:5d}  {t_pytorch*1000:13.3f}  {t_fused*1000:11.3f}  {t_step*1000:10.3f}"
              f"  {r_fused:11.1f}x  {r_step:10.1f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
