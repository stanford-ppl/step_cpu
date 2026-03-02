
# Vectorization
Can I see the generated c++ code for Gelu? Also, is it using vectorized instructions for high performance?

● Read 1 file (ctrl+o to expand)

● Here's the generated code. To answer your question directly: no, it's not using vectorized instructions for high performance. There are two major performance issues:

  1. No raw pointer / SIMD access. Every operation (torch::pow, torch::tanh, +, *) goes through PyTorch's ATen tensor dispatch, which involves function call overhead, dtype
  dispatch, and memory allocation for each intermediate tile. For a 1x32 tile, this overhead dominates the actual compute.
  2. slice() creates tensor views with metadata overhead. Each input.slice(0, ...).slice(1, ...) allocates a TensorImpl view object — no data copy, but the bookkeeping is
  expensive relative to the tiny tile size.

  To get actual high performance, the codegen would need to emit something like:

  #include <immintrin.h>  // AVX2/AVX-512

  // Inside the tile loop:
  float* in_ptr = input.data_ptr<float>() + i0 * dim1 + i1;
  float* out_ptr = output.data_ptr<float>() + i0 * dim1 + i1;
  int64_t tile_len = i1_end - i1;

  for (int64_t k = 0; k + 8 <= tile_len; k += 8) {
      __m256 x = _mm256_loadu_ps(in_ptr + k);
      __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
      // ... fused GELU math with SIMD intrinsics ...
      _mm256_storeu_ps(out_ptr + k, result);
  }
  // scalar tail for remainder

  This would require:
  - Raw pointer arithmetic instead of slice()
  - SIMD intrinsics (AVX2 __m256 for 8 floats at a time, or AVX-512 __m512 for 16)
  - Polynomial approximations for tanh (since there's no SIMD tanh instruction — typically done via a Padé or minimax approximation)
  - No ATen calls inside the inner loop

  Would you like me to implement a vectorized codegen backend?