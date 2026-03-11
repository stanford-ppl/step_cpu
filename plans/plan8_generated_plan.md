 Plan: GPT2Attention Optimized C++ Replacement

 Context

 The codebase already has a module replacement system for GPT2MLP with 8 progressively optimized kernel variants (gpt2mlp through gpt2mlp_fused6). The goal
 is to add a similar replacement for GPT2Attention — a new wrapper class, optimized C++ kernels, and registry entries — so users can run --replace
 gpt2attn_flash (and combine with MLP replacements like --replace gpt2mlp_fused6 gpt2attn_flash).

 Files to Modify

 - /home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py — All new code (wrapper class, build/apply functions, registry entries)
 - /home/ginasohn/step_cpu/darpa/modified/test_clm_replace.py — New test classes for attention replacement

 Reference Files (read-only)

 - /home/ginasohn/llama.cpp/ggml/src/ggml-cpu/ops.cpp lines 8347-8633 — Tiled flash attention algorithm
 - /home/ginasohn/llama.cpp/ggml/src/ggml-cpu/vec.h lines 1187-1212 — AVX-512 fast exp (ggml_v_expf)
 - /home/ginasohn/llama.cpp/ggml/src/ggml-cpu/simd-gemm.h — SIMD GEMM micro-kernel pattern

 Step 1: GPT2AttentionStepWrapper Class

 Insert after GPT2MLPStepWrapper (line 358). This is the drop-in replacement module.

 class GPT2AttentionStepWrapper(torch.nn.Module):
     def __init__(self, attn_module, compiled_kernel):
         super().__init__()
         self.c_attn = attn_module.c_attn       # Conv1D(768, 2304) — QKV projection
         self.c_proj = attn_module.c_proj        # Conv1D(768, 768) — output projection
         self.attn_dropout = attn_module.attn_dropout
         self.resid_dropout = attn_module.resid_dropout
         self.num_heads = attn_module.num_heads  # 12
         self.head_dim = attn_module.head_dim    # 64
         self.embed_dim = attn_module.embed_dim  # 768
         self.scale_attn_weights = attn_module.scale_attn_weights                                                                          07:08:45 [127/229]
         self._compiled_kernel = compiled_kernel
         self._original_attn = attn_module       # training fallback

     def forward(self, hidden_states, layer_past=None, attention_mask=None,
                 head_mask=None, encoder_hidden_states=None,
                 encoder_attention_mask=None, use_cache=False,
                 output_attentions=False):
         if self.training:
             return self._original_attn(hidden_states, ...)  # full fallback

         # Eval path: reshape, prepare past KV, call C++ kernel
         # Returns (attn_output, present_key_value) tuple matching GPT2Attention

 Key design decisions:
 - Forward signature matches GPT2Attention exactly (GPT2Block depends on this)
 - Training mode delegates to original module (no autograd in C++ kernels)
 - layer_past handling: pass past_key/past_value as separate tensors (or empty tensors if None)
 - attention_mask from HuggingFace is additive: 0.0 for allowed, -10000.0 for masked, shape [B,1,1,total_seq_len]
 - Return format: (attn_output, present) where present = (key, value) if use_cache else None
 - Conv1D weight layout: [in_features, out_features], so projection is x @ W + b

 Step 2: Baseline Kernel — gpt2attn (torch::mm + at::softmax)

 Establishes correctness. Uses PyTorch ops internally (no SIMD).

 C++ function signature:
 std::vector<torch::Tensor> gpt2_attn_step(
     torch::Tensor x,          // [M, 768]  (M = B*S)
     torch::Tensor W_attn,     // [768, 2304]
     torch::Tensor b_attn,     // [2304]
     torch::Tensor W_proj,     // [768, 768]
     torch::Tensor b_proj,     // [768]
     torch::Tensor past_key,   // [B, 12, past_len, 64] or empty
     torch::Tensor past_value, // [B, 12, past_len, 64] or empty
     torch::Tensor attn_mask,  // [B, 1, 1, total_len] or empty
     int64_t num_heads,        // 12
     int64_t batch_size        // B
 )

 Algorithm:
 1. QKV projection: qkv = x @ W_attn + b_attn → [M, 2304]                                                                                   07:08:45 [86/229]
 2. Split Q, K, V each [M, 768], reshape to [B, 12, S, 64]
 3. Concatenate past K, V along dim=2 if provided
 4. scores = Q @ K^T * (1/sqrt(64)) → [B, 12, S, total_len]
 5. Build causal mask (upper triangular -inf) for new positions
 6. Add HuggingFace attn_mask
 7. at::softmax(scores, -1)
 8. context = scores @ V → [B, 12, S, 64]
 9. Merge heads → [M, 768]
 10. Output projection: output = context @ W_proj + b_proj
 11. Return [output, present_key, present_value]

 Step 3: Fused Kernel — gpt2attn_fused (AVX-512 projections)

 Optimization over baseline:
 - Reuse avx512_omp_gemm/avx512_omp_gemv from fused6 for QKV and output projections (the two largest matmuls: M×768→M×2304 and M×768→M×768)
 - M=1 decode path uses GEMV
 - Inner attention (Q@K^T, softmax, attn@V) still uses torch::mm/at::softmax

 Step 4: Flash Kernel — gpt2attn_flash (Full tiled flash attention)

 The fully optimized kernel. Key optimizations from llama.cpp + fused6:

 Projections (same as fused):
 - AVX-512 GEMM/GEMV with bias fusion for QKV and output projections

 Core attention (new, ported from llama.cpp pattern):
 - Tiled flash attention with Q_TILE=64, KV_TILE=64
 - Online softmax: Single-pass with running max M[i] and sum S[i], accumulator rescaling
   - For each KV tile: compute Q@K^T tile, find row maxima, rescale accumulator, compute exp(score - max), accumulate scores @ V
   - Final normalization: acc /= S
 - Fast vectorized exp: Port ggml_v_expf from llama.cpp (AVX-512 Cephes-style polynomial, ~1.5 ULP error)
 - SIMD GEMM micro-kernels for Q@K^T and scores@V: reuse gemm_ukernel<RM,RN> template from fused6
 - Causal mask skipping: Skip entire KV tiles that are fully masked (all future positions)
 - OpenMP parallelism across heads: #pragma omp parallel for over B * num_heads
 - M=1 decode fast path: Single query row per head — Q@K^T becomes dot products, no tiling needed

 Flash attention pseudocode per head:
 for each Q-tile (q rows):
     M[i] = -inf, S[i] = 0, acc[i][d] = 0
     for each KV-tile:
         if causal_fully_masked(q_tile, kv_tile): skip
         scores = simd_gemm(Q_tile, K_tile^T)  // [Q_TILE x KV_TILE]
         scores *= 1/sqrt(64)
         apply causal mask + attn_mask
         // Online softmax update
         tile_max = row_max(scores)
         M_new = max(M, tile_max)
         rescale = exp(M - M_new)  // using fast_exp
         acc *= rescale; S *= rescale
         M = M_new
         scores = exp(scores - M_new)  // fast_exp
         S += row_sum(scores)
         acc += simd_gemm(scores, V_tile)
     acc /= S  // normalize

 Step 5: Build/Apply Functions & Registry

 Following exact patterns from fused6 (lines 1343-1696):

 def _build_gpt2attn():          # baseline → ~/.cache/mocha/gpt2_attn/
 def _build_gpt2attn_fused():    # AVX projections → ~/.cache/mocha/gpt2_attn_fused/
 def _build_gpt2attn_flash():    # full flash → ~/.cache/mocha/gpt2_attn_flash/

 def _apply_gpt2attn(model):     # iterates model.transformer.h, replaces block.attn
 def _apply_gpt2attn_fused(model):
 def _apply_gpt2attn_flash(model):

 Add to _REPLACEMENT_REGISTRY:
 "gpt2attn":       _apply_gpt2attn,        # baseline: torch::mm + at::softmax
 "gpt2attn_fused": _apply_gpt2attn_fused,  # AVX-512 projections + torch softmax
 "gpt2attn_flash": _apply_gpt2attn_flash,  # full tiled flash attention

 Step 6: Tests

 Add to test_clm_replace.py:

 1. TestGPT2AttentionStepWrapper — Unit tests:
   - test_output_shape_preserved: verify output shape matches original
   - test_output_numerically_close: compare against original GPT2Attention (atol=1e-4, rtol=1e-3)
   - test_kv_cache_shapes: verify present key/value shapes are 
   - test_various_sequence_lengths: S=1, 5, 128                                                                                              07:08:45 [5/229]
   - test_training_mode_fallback
 2. TestAttentionRegistry:
   - test_registry_contains_gpt2attn: verify all 3 variants exist
   - test_gpt2attn_replaces_all_blocks: verify all blocks get replaced
 3. TestAttentionInference — End-to-end:
   - test_infer_with_gpt2attn_replace: run full model.generate() with replacement
   - test_timing_comparison: baseline vs replaced timing

 Implementation Order

 1. GPT2AttentionStepWrapper class
 2. _build_gpt2attn() baseline + _apply_gpt2attn() + registry entry
 3. Test baseline for correctness (output matches original, model.generate works)
 4. _build_gpt2attn_fused() — copy AVX-512 GEMM/GEMV from fused6, use for projections
 5. _build_gpt2attn_flash() — full flash attention with online softmax + fast exp
 6. Tests in test_clm_replace.py
 7. Benchmark all variants

 Verification

 # Run existing tests (ensure nothing breaks)
 cd /home/ginasohn/step_cpu/darpa/modified && pytest test_clm_replace.py -v

 # Test attention replacement inference
 python causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only --no-instrument --replace gpt2attn

 # Test flash attention
 python causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only --no-instrument --replace gpt2attn_flash

 # Test combined MLP + attention replacement
 python causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only --no-instrument --replace gpt2mlp_fused6 gpt2attn_flash

 # Run new tests
 pytest test_clm_replace.py -v -k "Attention"