 Improve STeP codegen kernel speedup over PyTorch baseline

 Context

 Current sweep results on Intel Core Ultra 9 285K (24 cores, AVX2 only — no AVX-512):

 ┌───────────────┬────────────┬─────────────┬──────────┬─────────────────┐
 │ prompt_tokens │ gen_tokens │ PyTorch (s) │ STeP (s) │     Speedup     │
 ├───────────────┼────────────┼─────────────┼──────────┼─────────────────┤
 │ 254           │ 32         │ 1.14        │ 0.69     │ 1.64x           │
 ├───────────────┼────────────┼─────────────┼──────────┼─────────────────┤
 │ 254           │ 128        │ 1.83        │ 2.33     │ 0.79x (slower!) │
 ├───────────────┼────────────┼─────────────┼──────────┼─────────────────┤
 │ 620           │ 32         │ 1.24        │ 1.23     │ 1.01x           │
 ├───────────────┼────────────┼─────────────┼──────────┼─────────────────┤
 │ 620           │ 128        │ 2.54        │ 2.95     │ 0.86x (slower!) │
 └───────────────┴────────────┴─────────────┴──────────┴─────────────────┘

 STeP is slower than PyTorch for 3 of 4 configs. The pattern: longer generation and longer prompts hurt STeP. This points to
 per-decode-step overhead in the attention entry point and OpenMP misconfiguration.

 Root causes identified

 1. Redundant KV cache clones in attention (HIGH impact)

 File: _ATTN_ENTRY_POINT_CPP in 15_transformers/causal_language_modeling_codegen.py

 K = torch::cat({past_key, K}, 2);   // cat already returns NEW contiguous tensor
 V = torch::cat({past_value, V}, 2);
 auto present_key = has_past ? K.clone() : K;    // REDUNDANT — doubles copy cost
 auto present_value = has_past ? V.clone() : V;  // REDUNDANT — doubles copy cost

 cat already allocates a fresh tensor. K is only used read-only afterward (transpose, matmul). The .clone() copies the entire
 growing KV cache every decode step, every layer.

 Cost estimate: For 128 gen tokens × 6 layers × 2 clones, cache growing to [1,12,~382,64]: ~1.5 GB unnecessary memcpy total.
 Also trashes CPU caches, degrading subsequent GEMV performance.

 Fix: auto present_key = K; / auto present_value = V;

 2. Wasted causal mask allocation during decode (MEDIUM impact)

 Every decode step (S=1), the attention kernel:
 1. Allocates torch::zeros({1, total_len})
 2. Runs the fill loop (which does nothing when S=1, since fill_start == total_len)
 3. Unsqueezes twice, adds to scores

 That's a tensor alloc + broadcast add of all-zeros, 6 layers × 128 steps = 768 times.

 Fix: Guard with if (S > 1) { ... } to skip entirely during decode.

 3. OMP_NUM_THREADS=24 is too many for GEMV (HIGH impact)

 OMP_NUM_THREADS is unset, defaulting to all 24 cores. With AVX2 (VL=8, TILE=32):

 ┌──────────────────┬──────┬──────────────┬───────────────────────────┐
 │    Operation     │  N   │ Tiles (N/32) │ Tiles/thread (24 threads) │
 ├──────────────────┼──────┼──────────────┼───────────────────────────┤
 │ Attn output proj │ 768  │ 24           │ 1 (barely any work)       │
 ├──────────────────┼──────┼──────────────┼───────────────────────────┤
 │ MLP GEMV2        │ 768  │ 24           │ 1                         │
 ├──────────────────┼──────┼──────────────┼───────────────────────────┤
 │ QKV proj         │ 2304 │ 72           │ 3                         │
 ├──────────────────┼──────┼──────────────┼───────────────────────────┤
 │ MLP GEMV1        │ 3072 │ 96           │ 4                         │
 └──────────────────┴──────┴──────────────┴───────────────────────────┘

 For the 768-width ops, each thread gets 1 tile (32 floats) but pays the full #pragma omp parallel barrier cost. With ~3000 GEMV
  calls per generation (128 steps × 6 layers × 4 calls/layer), this overhead adds up significantly.

 No thread affinity set either — threads can migrate between cores, killing cache locality.

 Fix in sweep.sh:
 export OMP_NUM_THREADS=4    # experiment with 4, 6, 8
 export OMP_PROC_BIND=close
 export OMP_PLACES=cores

 4. Missing -ffast-math compiler flag (LOW-MEDIUM impact)

 File: step/compile.py

 Current: -O3 -std=c++17 -march=native -fopenmp -mavx2 -mfma

 Missing -ffast-math (or safer subset -fno-math-errno -ffinite-math-only) prevents the compiler from reordering FP ops for
 better scheduling and using faster reciprocal approximations. Safe for these kernels since they don't rely on NaN/Inf
 semantics.

 5. Attention core uses PyTorch fallback ops (LARGER, for later)

 Steps 4-8 in gpt2_attn_codegen_step (Q@K^T, scale, causal mask, softmax, attn@V) are all generic PyTorch ops (torch::matmul,
 at::softmax). These are NOT using the custom SIMD kernels. For longer sequences this becomes a bigger fraction of runtime. A
 full custom attention implementation would help but is a larger undertaking.

 Plan — Changes to make

 Change 1: Eliminate KV clones in _ATTN_ENTRY_POINT_CPP

 In 15_transformers/causal_language_modeling_codegen.py, edit the _ATTN_ENTRY_POINT_CPP string:

 // BEFORE:
 auto present_key = has_past ? K.clone() : K;
 auto present_value = has_past ? V.clone() : V;

 // AFTER:
 auto present_key = K;
 auto present_value = V;

 Change 2: Skip causal mask for S=1 in _ATTN_ENTRY_POINT_CPP

 // BEFORE:
 {
     auto causal = torch::zeros({S, total_len}, scores.options());
     ...
     scores.add_(causal.unsqueeze(0).unsqueeze(0));
 }

 // AFTER:
 if (S > 1) {
     auto causal = torch::zeros({S, total_len}, scores.options());
     ...
     scores.add_(causal.unsqueeze(0).unsqueeze(0));
 }

 Change 3: Set OMP environment variables in sweep.sh

 Add near the top, after mkdir -p:
 export OMP_NUM_THREADS=4
 export OMP_PROC_BIND=close
 export OMP_PLACES=cores

 Change 4: Add -ffast-math to compiler flags

 In step/compile.py, for both AVX2 and AVX512 branches, add -ffast-math to extra_cflags.

 Files to modify

 1. 15_transformers/causal_language_modeling_codegen.py — _ATTN_ENTRY_POINT_CPP string (changes 1-2)
 2. sweep.sh — OMP env vars (change 3)
 3. step/compile.py — compiler flags (change 4)

 Verification

 1. Delete cached .so files so recompilation picks up new flags: rm -rf ~/.cache/mocha/gpt2_*
 2. Run source sweep.sh && python3 analyze_sweep.py in the container
 3. Compare speedup column — expect improvement across all 4 configs, especially the 128-gen-token ones that are currently
 slower than PyTorch
 4. Experiment with OMP_NUM_THREADS=4,6,8 to find the sweet spot for this CPU