Plan: Codegen'd GPT2 Attention Kernel (plan12)

 Context

 Plan10 built the STeP IR and AVXCodegen for MLP (gpt2_mlp_fused6). The attention kernel (_build_gpt2attn_fused in causal_language_modeling.py) uses the same GEMV/GEMM code for
 QKV and output projections. We want to code-generate those projections from STeP programs, keeping the inner attention logic (softmax, masking, Q@K^T, etc.) hardcoded as C++.

 The generated GEMV/GEMM functions from AVXCodegen also include the tile-aligned OMP partitioning fix (from plan10 step 5) that the hardcoded version lacks.

 Approach

 Use AVXCodegen to generate ONLY the kernel functions (gemv_tiled_chunk, avx512_omp_gemv, gemm_ukernel, avx512_omp_gemm), then concatenate with a hardcoded C++ entry point string
  for the attention logic.

 Steps

 1. Add generate_kernels_only() to AVXCodegen

 File: step/avx_codegen.py

 Add a new method that emits includes + fast_tanh + GEMV functions + GEMM functions, but NOT the entry point or registration. Accepts extra_includes parameter for attention's
 #include <limits>.

 def generate_kernels_only(self, extra_includes: list[str] | None = None) -> str:
     # Same setup as generate()
     # Emit includes, extra_includes, fast_tanh
     # Emit GEMV microkernel + OMP wrapper (from first gemv stage in self.decode)
     # Emit GEMM microkernel + OMP wrapper (from first gemm stage in self.prefill)
     # NO entry point, NO registration
     return "\n".join(self.lines) + "\n"

 2. Add attention STeP programs in step_kernels.py

 File: darpa/modified/step_kernels.py

 Add two builders using existing _build_gemv_stage / _build_gemm_stage:

 - build_gpt2_attn_gemv_program() — single GEMV stage, name="gpt2_attn_codegen"
 - build_gpt2_attn_gemm_program() — single GEMM stage, same name

 These only need one stage each (no GELU). The IndexVar config (RN=4, VL=16) matches MLP, producing identical GEMV/GEMM functions. The programs are needed as input to AVXCodegen.

 3. Add attention codegen in causal_language_modeling_codegen.py

 File: darpa/modified/causal_language_modeling_codegen.py

 Add three components:

 3a. GPT2AttentionStepWrapper — Copy from causal_language_modeling.py:775-861. Drop-in wrapper that in eval mode calls the compiled kernel with 10 args (x2d, W_attn, b_attn,
 W_proj, b_proj, past_key, past_value, attn_mask, num_heads, B), updates KV cache.

 3b. _build_gpt2attn_codegen() — Core integration:
 1. Build STeP programs, run codegen.generate_kernels_only(extra_includes=["#include <limits>"])
 2. Define hardcoded entry point string — gpt2_attn_codegen_step(x, W_attn, b_attn, W_proj, b_proj, past_key, past_value, attn_mask, num_heads, batch_size) -> vector<Tensor>.
 Body copied from _build_gpt2attn_fused lines 1150-1242, with function name changed to gpt2_attn_codegen_step.
 3. Concatenate: cpp_source = kernel_cpp + entry_point_cpp
 4. Compile via build_extension("gpt2_attn_codegen", cpp_source, avx512=True)

 The entry point's inner attention (steps 2-8) stays identical to _build_gpt2attn_fused:
 - Split Q,K,V + reshape/permute
 - Concat past KV
 - Q @ K^T * scale (torch::matmul)
 - Causal mask (loop-based)
 - Softmax (at::softmax)
 - Context: attn_weights @ V
 - Merge heads

 Only the QKV projection (step 1) and output projection (step 9) call the codegen'd avx512_omp_gemv/avx512_omp_gemm.

 3c. _apply_gpt2attn_codegen(model) — Replaces block.attn for all transformer blocks.

 3d. Wire into registry — Add "gpt2attn_codegen": _apply_gpt2attn_codegen to _REPLACEMENT_REGISTRY. Update --replace help text.

 Files to Modify

 ┌────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                        File                        │                                                      Change                                                      │
 ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ step/avx_codegen.py                                │ Add generate_kernels_only() (~20 lines)                                                                          │
 ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ darpa/modified/step_kernels.py                     │ Add build_gpt2_attn_gemv_program() + build_gpt2_attn_gemm_program() (~60 lines)                                  │
 ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ darpa/modified/causal_language_modeling_codegen.py │ Add GPT2AttentionStepWrapper, _build_gpt2attn_codegen(), _apply_gpt2attn_codegen(), update registry (~170 lines) │
 └────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 Key Details

 - Function name: gpt2_attn_codegen_step (not gpt2_attn_fused_step) to avoid TORCH_LIBRARY name collision
 - build_extension return type: Works as-is — torch.ops dispatch handles std::vector<torch::Tensor> returns correctly (returns Python list)
 - fast_tanh_avx512: Emitted but unused by attention (no GELU); compiler drops the dead static function
 - Codegen'd GEMV differs from hardcoded: No prefetch instructions, tile-aligned OMP partitioning (the fix from plan10). Numerically equivalent.

 Verification

 # In Docker container:
 source /home/dockeruser/mochaenv/bin/activate
 cd /home/dockeruser/step_cpu
 PYTHONPATH=/home/dockeruser/step_cpu:$PYTHONPATH

 # Attention only
 python3 darpa/modified/causal_language_modeling_codegen.py \
   --mode infer --prompt "Why is the sky blue?" --cpu-only --replace gpt2attn_codegen

 # Both MLP + attention
 python3 darpa/modified/causal_language_modeling_codegen.py \
   --mode infer --prompt "Why is the sky blue?" --cpu-only --replace gpt2mlp_fused6 gpt2attn_codegen

 Success criteria: Output matches baseline (no replacement) text identically.

 Reference: Attention entry point C++ (from causal_language_modeling.py:1150-1242)

 The hardcoded entry point string to embed in _build_gpt2attn_codegen() is the gpt2_attn_fused_step body from lines 1150-1242, with the function name changed to
 gpt2_attn_codegen_step and TORCH_LIBRARY_FRAGMENT updated accordingly.