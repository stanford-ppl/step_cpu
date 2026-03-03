#!/usr/bin/env python
"""
Diagnostic benchmark: isolate sources of ~10x slowdown in STeP-compiled
GPT2MLP kernel vs PyTorch native MLP.

Benchmarks:
  A  PyTorch MLP forward (baseline)
  B  STeP compiled kernel forward (total overhead)
  C  Weight .t().contiguous() x2 (per-call transpose+copy)
  D  Tiled matmul (72 tiny torch.mm) vs single GEMM (2 calls)
  E  Accumulator + buffer allocations (torch.zeros / torch.empty)
  F  Tiled GELU (per-tile temps) vs single F.gelu
  G  Inner-loop .t() on tiles (redundant re-transpose)

Usage:
  conda activate py312clean
  source /home/ginasohn/research/mocha/.venv/bin/activate
  python3 darpa/modified/benchmark_mlp_slowdown.py
"""

import math
import sys
import time
import pathlib

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Project setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])  # mocha/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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
#  Load model weights once
# ---------------------------------------------------------------------------
print("Loading distilgpt2 weights …")
from transformers import AutoModelForCausalLM

_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
_model.eval()
_mlp = _model.transformer.h[0].mlp  # first MLP block

# NOTE: HuggingFace GPT-2 uses Conv1D which stores weights as (in_features, out_features),
# i.e. W_fc is [768, 3072] and W_proj is [3072, 768].
# The original MLP forward does: hidden @ W + b  (not F.linear which expects [out, in]).
W_fc = _mlp.c_fc.weight.detach()      # [768, 3072]  (Conv1D layout)
b_fc = _mlp.c_fc.bias.detach()        # [3072]
W_proj = _mlp.c_proj.weight.detach()  # [3072, 768]  (Conv1D layout)
b_proj = _mlp.c_proj.bias.detach()    # [768]

print(f"  W_fc   {tuple(W_fc.shape)}   ({W_fc.nelement()*4/1e6:.1f} MB)  [Conv1D: in×out]")
print(f"  W_proj {tuple(W_proj.shape)}  ({W_proj.nelement()*4/1e6:.1f} MB)  [Conv1D: in×out]")
print()

# ---------------------------------------------------------------------------
#  Compile STeP kernel (once)
# ---------------------------------------------------------------------------
print("Compiling STeP kernel (may take a moment on first run) …")
import step

def gpt2_mlp_step(hidden_states, W_fc_, b_fc_, W_proj_, b_proj_):
    M = hidden_states.shape[0]
    K_tile = 256
    N_tile = 256

    act_fc = step.tensor_to_stream(hidden_states, vec=[M, K_tile])
    act_fc = step.Flatten(act_fc, min_rank=0, max_rank=1)
    act_fc_buf = step.Bufferize(act_fc, rank=1)
    act_fc_rep = step.Streamify(act_fc_buf, repeat_factor=[12], rank=1)

    W_fc_T = W_fc_.T.contiguous()
    wfc_stream = step.tensor_to_stream(W_fc_T, vec=[N_tile, K_tile])

    fc_partial = step.BinaryMap(act_fc_rep, wfc_stream, lambda a, w: torch.mm(a, w.T))
    fc_accum = step.Accum(fc_partial, rank=1)

    bfc_stream = step.tensor_to_stream(b_fc_.unsqueeze(0), vec=[1, N_tile])
    bfc_stream = step.Flatten(bfc_stream, min_rank=0, max_rank=1)

    fc_biased = step.BinaryMap(fc_accum, bfc_stream, lambda a, b: a + b)

    c_sqrt = math.sqrt(2.0 / math.pi)
    c_pow = 0.044715

    x3 = step.UnaryMap(fc_biased, lambda a: torch.pow(a, 3.0))
    x_in = step.BinaryMap(fc_biased, x3, lambda a, b: a + c_pow * b)
    t_in = step.UnaryMap(x_in, lambda a: c_sqrt * a)
    t = step.UnaryMap(t_in, lambda a: torch.tanh(a))
    onep = step.UnaryMap(t, lambda a: 1.0 + a)
    gelu_out = step.BinaryMap(fc_biased, onep, lambda a, b: 0.5 * a * b)

    proj_buf = step.Bufferize(gelu_out, rank=1)
    proj_act_rep = step.Streamify(proj_buf, repeat_factor=[3], rank=1)

    W_proj_T = W_proj_.T.contiguous()
    wproj_stream = step.tensor_to_stream(W_proj_T, vec=[N_tile, K_tile])

    proj_partial = step.BinaryMap(proj_act_rep, wproj_stream, lambda a, w: torch.mm(a, w.T))
    proj_accum = step.Accum(proj_partial, rank=1)

    bproj_stream = step.tensor_to_stream(b_proj_.unsqueeze(0), vec=[1, N_tile])
    bproj_stream = step.Flatten(bproj_stream, min_rank=0, max_rank=1)

    proj_biased = step.BinaryMap(proj_accum, bproj_stream, lambda a, b: a + b)

    output = step.stream_to_tensor(proj_biased, like_tensor=hidden_states)
    return output

gpt2_mlp_step._param_ndims = [2, 2, 1, 2, 1]
compiled_kernel = step.cpu_compile(gpt2_mlp_step)
print("  STeP kernel compiled.\n")


# ---------------------------------------------------------------------------
#  Benchmark functions
# ---------------------------------------------------------------------------

def bench_A_pytorch_baseline(x):
    """A: PyTorch native MLP forward (Conv1D-style: x @ W + b, then GELU)."""
    def fn():
        h = torch.mm(x, W_fc) + b_fc       # [M,768] @ [768,3072] → [M,3072]
        h = F.gelu(h, approximate="tanh")
        h = torch.mm(h, W_proj) + b_proj    # [M,3072] @ [3072,768] → [M,768]
        return h
    return bench(fn)


def bench_B_step_kernel(x):
    """B: STeP compiled kernel forward (total)."""
    def fn():
        return compiled_kernel(x, W_fc, b_fc, W_proj, b_proj)
    return bench(fn)


def bench_C_weight_transpose(x):
    """C: Weight .t().contiguous() x2 (transpose + alloc + copy)."""
    def fn():
        _ = W_fc.t().contiguous()
        _ = W_proj.t().contiguous()
    return bench(fn)


def bench_D_tiled_vs_single_gemm(x):
    """D: 72 tiny torch.mm calls vs 2 single GEMM calls."""
    M = x.shape[0]
    # The kernel transposes Conv1D weights: [768,3072].t() → [3072,768] (contiguous copy)
    W_fc_tc = W_fc.t().contiguous()      # [3072, 768]
    W_proj_tc = W_proj.t().contiguous()  # [768, 3072]

    # D1: tiled matmul (mimics generated kernel exactly)
    # Kernel tiles W_fc_tc[3072,768] as [i0:i0+256, i1:i1+256] then does mm(act, w.t())
    def fn_tiled():
        # c_fc stage: 12 outer(3072/256) x 3 inner(768/256) = 36 calls
        for i0 in range(0, 3072, 256):
            i0_end = min(i0 + 256, 3072)
            acc = torch.zeros(M, i0_end - i0)
            for i1 in range(0, 768, 256):
                i1_end = min(i1 + 256, 768)
                act = x[:, i1:i1_end]
                w = W_fc_tc[i0:i0_end, i1:i1_end]   # [256,256]
                acc.add_(torch.mm(act, w.t()))        # mm([M,256],[256,256])
        # c_proj stage: 3 outer(768/256) x 12 inner(3072/256) = 36 calls
        h = torch.randn(M, 3072)  # stand-in for GELU output
        for i0 in range(0, 768, 256):
            i0_end = min(i0 + 256, 768)
            acc = torch.zeros(M, i0_end - i0)
            for i1 in range(0, 3072, 256):
                i1_end = min(i1 + 256, 3072)
                act = h[:, i1:i1_end]
                w = W_proj_tc[i0:i0_end, i1:i1_end]  # [256,256]
                acc.add_(torch.mm(act, w.t()))         # mm([M,256],[256,256])

    # D2: single GEMM calls (equivalent to Conv1D: x @ W)
    def fn_gemm():
        _ = torch.mm(x, W_fc)            # [M,768] @ [768,3072]
        h = torch.randn(M, 3072)
        _ = torch.mm(h, W_proj)           # [M,3072] @ [3072,768]

    t_tiled = bench(fn_tiled)
    t_gemm = bench(fn_gemm)
    return t_tiled, t_gemm


def bench_E_allocations(x):
    """E: Accumulator + buffer allocations per call."""
    M = x.shape[0]
    def fn():
        # Mirrors generated kernel: 15 torch.zeros accumulators + 2 torch.empty buffers
        # c_fc: 12 accumulators
        for _ in range(12):
            _ = torch.zeros(M, 256)
        # c_proj: 3 accumulators
        for _ in range(3):
            _ = torch.zeros(M, 256)
        # output + buf_1
        _ = torch.empty(M, 3072)
        _ = torch.empty(M, 768)
    return bench(fn)


def bench_F_tiled_vs_single_gelu(x):
    """F: Tiled GELU (per-tile temps) vs single F.gelu."""
    M = x.shape[0]
    c_sqrt = math.sqrt(2.0 / math.pi)
    c_pow = 0.044715

    # Create a fake fc output to apply GELU to
    fc_out = torch.randn(M, 3072)

    # F1: tiled GELU — 12 tiles, 7 temp tensors each
    def fn_tiled():
        for i0 in range(0, 3072, 256):
            b_1 = fc_out[:, i0:i0+256]
            u = torch.pow(b_1, 3.0)
            b_2 = b_1 + c_pow * u
            u_1 = c_sqrt * b_2
            u_2 = torch.tanh(u_1)
            u_3 = 1.0 + u_2
            _ = 0.5 * b_1 * u_3

    # F2: single fused F.gelu
    def fn_single():
        _ = F.gelu(fc_out, approximate="tanh")

    t_tiled = bench(fn_tiled)
    t_single = bench(fn_single)
    return t_tiled, t_single


def bench_G_inner_transpose(x):
    """G: Redundant inner-loop .t() on tiles — both stages."""
    W_fc_tc = W_fc.t().contiguous()      # [3072, 768]
    W_proj_tc = W_proj.t().contiguous()  # [768, 3072]
    def fn():
        # c_fc: 12×3 = 36 inner .t() calls
        for i0 in range(0, 3072, 256):
            i0_end = min(i0 + 256, 3072)
            for i1 in range(0, 768, 256):
                i1_end = min(i1 + 256, 768)
                w = W_fc_tc[i0:i0_end, i1:i1_end]
                _ = w.t()
        # c_proj: 3×12 = 36 inner .t() calls
        for i0 in range(0, 768, 256):
            i0_end = min(i0 + 256, 768)
            for i1 in range(0, 3072, 256):
                i1_end = min(i1 + 256, 3072)
                w = W_proj_tc[i0:i0_end, i1:i1_end]
                _ = w.t()
    return bench(fn)


# ---------------------------------------------------------------------------
#  Main: sweep over M values
# ---------------------------------------------------------------------------
M_VALUES = [1, 5, 20, 128]
WARMUP = 5
ITERS = 50

print("=" * 80)
print("GPT2 MLP STeP Kernel Slowdown Diagnostic Benchmark")
print(f"  Warmup={WARMUP}  Iters={ITERS}  M values={M_VALUES}")
print(f"  Tile size: 256x256  |  W_fc: {tuple(W_fc.shape)}  |  W_proj: {tuple(W_proj.shape)}")
print("=" * 80)

all_results = {}

for M in M_VALUES:
    print(f"\n{'─' * 80}")
    print(f"  M = {M}  (autoregressive token count / batch×seq flattened)")
    print(f"{'─' * 80}")

    x = torch.randn(M, 768)

    t_A = bench_A_pytorch_baseline(x)
    t_B = bench_B_step_kernel(x)

    t_C = bench_C_weight_transpose(x)

    t_D_tiled, t_D_gemm = bench_D_tiled_vs_single_gemm(x)

    t_E = bench_E_allocations(x)

    t_F_tiled, t_F_single = bench_F_tiled_vs_single_gelu(x)

    t_G = bench_G_inner_transpose(x)

    gap = t_B - t_A

    results = {
        "A_pytorch": t_A,
        "B_step": t_B,
        "C_transpose": t_C,
        "D_tiled_mm": t_D_tiled,
        "D_single_gemm": t_D_gemm,
        "E_allocs": t_E,
        "F_tiled_gelu": t_F_tiled,
        "F_single_gelu": t_F_single,
        "G_inner_t": t_G,
        "gap": gap,
    }
    all_results[M] = results

    # --- Print table ---
    print(f"\n  {'Benchmark':<45} {'Time (ms)':>10} {'vs Base':>8} {'% of Gap':>9}")
    print(f"  {'─'*45} {'─'*10} {'─'*8} {'─'*9}")

    def row(label, t, is_component=False):
        ratio = t / t_A if t_A > 0 else float("inf")
        pct = (t / gap * 100) if (gap > 0 and is_component) else None
        pct_s = f"{pct:7.1f}%" if pct is not None else "       -"
        print(f"  {label:<45} {t*1000:10.3f} {ratio:7.1f}x {pct_s}")

    row("A  PyTorch baseline (2 F.linear + GELU)", t_A)
    row("B  STeP compiled kernel", t_B)
    print(f"  {'':45} {'─'*10}")
    print(f"  {'   GAP (B − A)':<45} {gap*1000:10.3f}")
    print()
    row("C  Weight .t().contiguous() ×2", t_C, is_component=True)
    row("D  Tiled matmul (72 torch.mm)", t_D_tiled, is_component=True)
    row("D' Single GEMM (2 torch.mm)", t_D_gemm, is_component=True)
    row("   D overhead (tiled − single)", t_D_tiled - t_D_gemm, is_component=True)
    row("E  Allocations (15 zeros + 2 empty)", t_E, is_component=True)
    row("F  Tiled GELU (12 tiles × 7 temps)", t_F_tiled, is_component=True)
    row("F' Single F.gelu", t_F_single, is_component=True)
    row("   F overhead (tiled − single)", t_F_tiled - t_F_single, is_component=True)
    row("G  Inner-loop .t() (36 tile transposes)", t_G, is_component=True)

    # --- Estimated breakdown ---
    est_transpose = t_C
    est_tile_mm = t_D_tiled - t_D_gemm
    est_allocs = t_E
    est_gelu = t_F_tiled - t_F_single
    est_inner_t = t_G
    est_total = est_transpose + est_tile_mm + est_allocs + est_gelu + est_inner_t
    est_other = gap - est_total

    print(f"\n  Estimated breakdown of the {gap*1000:.2f} ms gap:")
    components = [
        ("Weight transpose+copy (C)", est_transpose),
        ("Tile dispatch overhead (D)", est_tile_mm),
        ("Tensor allocations (E)", est_allocs),
        ("Tiled GELU overhead (F)", est_gelu),
        ("Inner-loop .t() (G)", est_inner_t),
        ("Other / interaction", est_other),
    ]
    for name, t in components:
        pct = t / gap * 100 if gap > 0 else 0
        bar = "█" * max(0, int(pct / 2))
        print(f"    {name:<35} {t*1000:8.3f} ms  ({pct:5.1f}%)  {bar}")

    # --- FLOP / dispatch analysis ---
    # c_fc: M×768×3072 MACs, c_proj: M×3072×768 MACs  →  total ~4.7M×M FLOPs (MAC=2 FLOP)
    total_flops = 2 * M * 768 * 3072 * 2  # 2 matmuls, each M×K×N MACs × 2 FLOP/MAC
    flops_per_dispatch_tiled = total_flops / 72
    flops_per_dispatch_gemm = total_flops / 2
    print(f"\n  FLOP analysis (M={M}):")
    print(f"    Total FLOPs (2 matmuls):           {total_flops:>14,}")
    print(f"    FLOPs per dispatch (72 tiles):     {flops_per_dispatch_tiled:>14,.0f}")
    print(f"    FLOPs per dispatch (2 GEMMs):      {flops_per_dispatch_gemm:>14,.0f}")
    print(f"    Ratio (GEMM/tile):                 {flops_per_dispatch_gemm/flops_per_dispatch_tiled:>14,.0f}x")


# ---------------------------------------------------------------------------
#  Summary across M values
# ---------------------------------------------------------------------------
print(f"\n\n{'=' * 80}")
print("SUMMARY: Slowdown ratio (STeP / PyTorch) across M values")
print(f"{'=' * 80}")
print(f"\n  {'M':>5}  {'PyTorch (ms)':>13}  {'STeP (ms)':>10}  {'Slowdown':>9}  {'Gap (ms)':>9}")
print(f"  {'─'*5}  {'─'*13}  {'─'*10}  {'─'*9}  {'─'*9}")
for M in M_VALUES:
    r = all_results[M]
    ratio = r["B_step"] / r["A_pytorch"] if r["A_pytorch"] > 0 else float("inf")
    print(f"  {M:5d}  {r['A_pytorch']*1000:13.3f}  {r['B_step']*1000:10.3f}  {ratio:8.1f}x  {r['gap']*1000:9.3f}")

print(f"\n{'=' * 80}")
print("Key finding: at M=1 (autoregressive), per-call overhead (weight transpose,")
print("BLAS dispatch for 72 tiny matmuls, tensor allocations) dominates because")
print("the actual compute is negligible. At larger M, compute grows while overhead")
print("stays constant, so the ratio improves.")
print(f"{'=' * 80}")
