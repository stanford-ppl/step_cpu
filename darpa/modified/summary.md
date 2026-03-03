Content of `causal_language_modeling.py`

# Changes made to `causal_language_modeling.py`

1. parse_clm() (lines 336–347) — Added --mode (choices: train/infer, default: train) and --prompt
(string, optional) arguments.
2. main() (lines 388–477) — Inserted an inference branch that fires before dataset loading when
args.mode == "infer":
- Validates --prompt is present, prints a clear error and returns early if not.
- Runs 4 timed stages: create_tokenizer → load_model → tokenize_prompt → generate.
- Prints the generated text, then the same benchmark summary table (or total runtime if
--no-instrument) that the training path uses.
- Returns BenchmarkResult(perplexity=None, elapsed_time=total_elapsed).
- The training path is unchanged — it runs when args.mode == "train" (the default) by falling
through the if args.mode == "infer": block.

# Usage:
## Training (unchanged)
python3 darpa/modified/causal_language_modeling.py --cpu-only --samples 50

## Inference
python3 darpa/modified/causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only

## Inference without instrumentation
python3 darpa/modified/causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only --no-instrument

## Missing prompt — prints error and exits cleanly
python3 darpa/modified/causal_language_modeling.py --mode infer --cpu-only

# With replacement
```
python3 darpa/modified/causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only --re
place gpt2mlp

=== System resources ===
Available CPUs : 144
Running in CPU-only mode
==================================================

Loading fine‑tuned model from clm-example-model
Loading weights: 100%|█| 76/76 [00:00<00:00, 2844.79it/s, Materializing param=transf

Applying module replacements: ['gpt2mlp']
Replaced 6 GPT2MLP blocks with STeP-compiled kernel.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

=== Generated Text ===
Why is the sky blue? It's a question that has been answered for years. The answer is yes. It's a question


=== Benchmark Summary (CPU-only mode) ===

| Stage              |   Time (s) | CPU %   | Cores Used   |
|--------------------|------------|---------|--------------|
| create_tokenizer   |       0.52 | 0.6     | 0.9          |
| load_model         |       0.13 | 0.5     | 0.6          |
| apply_replacements |      32.72 | 1.2     | 1.7          |
| tokenize_prompt    |       0.01 | 0.0     | 0.0          |
| generate           |       0.53 | 35.5    | 51.1         |
| TOTAL              |      33.91 | -       | -            |

==================================================
((.venv) ) (py312clean) ginasohn@lagos:~/research/mocha$ python3 darpa/modified/causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only

=== System resources ===
Available CPUs : 144
Running in CPU-only mode
==================================================

Loading fine‑tuned model from clm-example-model
Loading weights: 100%|█| 76/76 [00:00<00:00, 2937.95it/s, Materializing param=transf
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

=== Generated Text ===
Why is the sky blue? It's a question that has been answered for years. The answer is yes. It's a question


=== Benchmark Summary (CPU-only mode) ===

| Stage            |   Time (s) | CPU %   | Cores Used   |
|------------------|------------|---------|--------------|
| create_tokenizer |       0.72 | 0.6     | 0.8          |
| load_model       |       0.13 | 0.3     | 0.5          |
| tokenize_prompt  |       0.01 | 16.7    | 24.0         |
| generate         |       0.53 | 34.2    | 49.3         |
| TOTAL            |       1.39 | -       | -            |

==================================================
```

---

# GPT2MLP STeP Kernel ~10x Slowdown Analysis

## Executive Summary

The STeP-compiled GPT2MLP kernel is **109x slower** than PyTorch at M=1 (autoregressive generation) and **48x slower** at M=128. The slowdown ranges from 48-109x depending on batch size. The dominant bottleneck is **per-call weight transposition** (~18ms of fixed overhead copying 18.8MB every forward call), followed by **tiled matmul dispatch overhead** (72 tiny BLAS calls instead of 2 optimized GEMMs). These are artifacts of the tensor-to-stream compilation pattern, not fundamental limitations.

## Root Causes (Ranked by Impact)

### 1. Per-call weight transpose + contiguous copy (~77% of gap)

Every forward call executes:
```cpp
auto W_fc_t_c_ = W_fc.t().contiguous();     // allocates + copies [768,3072] → [3072,768] = 9.4MB
auto W_proj_t_c_ = W_proj.t().contiguous();  // allocates + copies [3072,768] → [768,3072] = 9.4MB
```

This allocates and copies **18.8MB** of weight data every call. PyTorch's native Conv1D stores weights in `(in, out)` layout and just does `x @ W` directly — no transpose needed. The STeP kernel transposes the weights to match its `tensor_to_stream` tiling pattern, but this is unnecessary since the inner loop then transposes the tiles back with `w.t()`.

Measured: **17.975ms** regardless of M.

### 2. Tiled matmul dispatch overhead (~15% of gap)

The kernel makes **72 tiny `torch::mm` calls** (36 per matmul stage) instead of 2 optimized GEMM calls:
- c_fc: 12 outer (3072/256) × 3 inner (768/256) = 36 calls
- c_proj: 3 outer (768/256) × 12 inner (3072/256) = 36 calls

At M=1, each call is `torch::mm([1,256], [256,256])` — the BLAS per-call overhead (thread pool dispatch, cache warmup, function call overhead) completely dwarfs the ~131K FLOPs of actual compute. A single GEMM amortizes this to just 2 dispatch calls for ~4.7M FLOPs.

Measured: **3.453ms at M=1**.

### 3. Tiled GELU temporary tensors (~2.7% of gap)

GELU is computed tile-by-tile (12 tiles of 256 columns), with **7 temporary tensors per tile** (pow, add, mul, tanh, etc.) = **84 tensor allocations** per call. PyTorch's fused `F.gelu` operates on the full `[M,3072]` tensor in a single kernel launch.

Measured: **0.621ms at M=1**.

### 4. Inner-loop .t() redundant transposes (~1-3% of gap)

Weights are transposed at kernel start (`W_fc.t().contiguous()`), then each tile is re-transposed in the inner loop: `torch::mm(act, w.t())`. The net effect is accessing the weight in its original layout but through two transpose operations. The inner `.t()` creates a transposed view (cheap but not free — 72 view creations per call).

Measured: **0.453ms constant**.

### 5. Tensor allocations (~0.1-0.3% of gap)

15 `torch::zeros` accumulators + 2 `torch::empty` buffers per call. Individually small, but adds up to 0.057ms (57µs).

## Annotated C++ Kernel Walkthrough

```cpp
// Generated by STeP codegen (_emit_stage in step/codegen.py)
torch::Tensor gpt2_mlp_step_step(
    torch::Tensor hidden_states,  // [M, 768]
    torch::Tensor W_fc,           // [768, 3072]  Conv1D layout
    torch::Tensor b_fc,           // [3072]
    torch::Tensor W_proj,         // [3072, 768]  Conv1D layout
    torch::Tensor b_proj          // [768]
) {
    int64_t M = hidden_states.size(0);
    auto output = torch::empty_like(hidden_states);          // ALLOC: output [M,768]

    // *** BOTTLENECK #1: Per-call weight copies (18.8MB) ***
    auto W_fc_t_c_ = W_fc.t().contiguous();                 // ALLOC+COPY: [3072,768]
    auto W_proj_t_c_ = W_proj.t().contiguous();             // ALLOC+COPY: [768,3072]

    auto b_fc_us0_ = b_fc.unsqueeze(0);                     // view, cheap
    auto b_proj_us0_ = b_proj.unsqueeze(0);                 // view, cheap
    auto buf_1 = torch::empty({M, 3072}, ...);              // ALLOC: intermediate buffer

    // Stage 1: c_fc matmul + GELU (fused per-tile)
    for (int64_t i0 = 0; i0 < 3072; i0 += 256) {           // 12 outer iterations
        auto acc = torch::zeros({M, 256}, ...);              // ALLOC: accumulator
        for (int64_t i1 = 0; i1 < 768; i1 += 256) {        // 3 inner iterations
            auto act = hidden_states.slice(1, i1, i1_end);  // view
            auto w = W_fc_t_c_.slice(0, i0, ...).slice(1, i1, ...);  // view [256,256]
            acc.add_(torch::mm(act, w.t()));                 // *** BOTTLENECK #2+4: tiny mm + redundant .t() ***
        }
        // *** BOTTLENECK #3: 7 temp tensors per GELU tile ***
        auto b_1 = acc + b_fc_us0_.slice(...);               // bias add
        auto u = torch::pow(b_1, 3.0);                      // temp 1
        auto b_2 = b_1 + 0.044715 * u;                      // temp 2
        auto u_1 = 0.7978845608028654 * b_2;                // temp 3
        auto u_2 = torch::tanh(u_1);                        // temp 4
        auto u_3 = 1.0 + u_2;                               // temp 5
        auto b_3 = 0.5 * b_1 * u_3;                         // temp 6
        buf_1.slice(1, i0, i0_end).copy_(b_3);              // copy to buffer (Bufferize)
    }

    // Stage 2: c_proj matmul (same tiling pattern)
    for (int64_t i0 = 0; i0 < 768; i0 += 256) {            // 3 outer iterations
        auto acc = torch::zeros({M, 256}, ...);
        for (int64_t i1 = 0; i1 < 3072; i1 += 256) {       // 12 inner iterations
            // same tiny mm + .t() pattern
            acc.add_(torch::mm(act, w.t()));
        }
        output.slice(1, i0, i0_end).copy_(acc + bias);
    }
    return output;
}
```

## Quantified Benchmark Results

Benchmark script: `darpa/modified/benchmark_mlp_slowdown.py`

### Slowdown ratio across M values

| M | PyTorch (ms) | STeP (ms) | Slowdown | Gap (ms) |
|---|---|---|---|---|
| 1 | 0.216 | 23.558 | **109x** | 23.342 |
| 5 | 0.327 | 28.912 | **88.5x** | 28.586 |
| 20 | 0.501 | 34.112 | **68.1x** | 33.611 |
| 128 | 1.059 | 50.983 | **48.2x** | 49.924 |

### Breakdown at M=1 (autoregressive generation case)

| Cause | Time (ms) | % of Gap |
|---|---|---|
| Weight transpose+copy | 17.975 | 77.0% |
| Tile dispatch overhead (72 vs 2 GEMMs) | 3.453 | 14.8% |
| Tiled GELU overhead | 0.621 | 2.7% |
| Inner-loop .t() | 0.453 | 1.9% |
| Tensor allocations | 0.057 | 0.2% |
| Other / interaction effects | 0.783 | 3.4% |

### Breakdown at M=128 (batch inference)

| Cause | Time (ms) | % of Gap |
|---|---|---|
| Weight transpose+copy | 17.975 | 36.0% |
| Tile dispatch overhead | 3.453 | 6.9% |
| Tiled GELU overhead | 0.621 | 1.2% |
| Inner-loop .t() | 0.453 | 0.9% |
| Tensor allocations | 0.057 | 0.1% |
| Other / interaction effects | 27.365 | 54.8% |

## Does tensor-to-stream cause inefficiencies?

**Yes.** The `tensor_to_stream` → tiling abstraction is the root cause of every measured bottleneck:

1. **Tiling forces 72 small GEMMs** instead of 2 large ones. The `Streamify`/`BinaryMap` pattern with `repeat_factor=[12]` and `repeat_factor=[3]` generates the nested tile loops. Each `torch::mm` on a tiny `[M,256]×[256,256]` tile wastes most of its time in BLAS dispatch overhead.

2. **`tensor_to_stream` on weights forces transpose + contiguous copy.** The `W_fc.T.contiguous()` in the STeP function definition maps directly to `W_fc.t().contiguous()` in the generated C++. This is because `tensor_to_stream` needs a specific memory layout for its tiling pattern, but the Conv1D weights are already usable as-is for `x @ W`.

3. **`Bufferize`/`Streamify` forces an intermediate buffer** (`buf_1`) and explicit `copy_()` between stages. PyTorch just passes the GELU output tensor directly to the next matmul.

4. **Per-tile GELU** creates 84 temporary tensors instead of 1 fused `F.gelu` call. The stream abstraction processes each tile independently, preventing fusion across the full output dimension.

The overhead is **purely structural** — it comes from how the streaming/tiling pattern maps to eager PyTorch C++ API calls, not from algorithmic incorrectness. The compiled kernel produces correct results; it just does far more work to get there.