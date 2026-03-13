"""Unit tests for AVX512 code generator: verify generated C++ string structure."""

import unittest


def _generate_cpp():
    """Generate the C++ source from STeP programs."""
    from darpa.modified.step_kernels import (
        build_gpt2_mlp_gemv_program,
        build_gpt2_mlp_gemm_program,
    )
    from step.avx_codegen import AVXCodegen

    gemv = build_gpt2_mlp_gemv_program()
    gemm = build_gpt2_mlp_gemm_program()
    codegen = AVXCodegen(decode=gemv, prefill=gemm)
    return codegen.generate()


# Generate once, reuse across all tests
_CPP = _generate_cpp()


class TestAVXCodegenStructure(unittest.TestCase):
    """Verify top-level C++ structure: includes, helpers, registration."""

    def test_includes(self):
        self.assertIn("#include <torch/extension.h>", _CPP)
        self.assertIn("#include <immintrin.h>", _CPP)
        self.assertIn("#include <omp.h>", _CPP)

    def test_fast_tanh(self):
        self.assertIn("fast_tanh_avx512", _CPP)
        # Pade coefficients
        self.assertIn("135135.0f", _CPP)
        self.assertIn("17325.0f", _CPP)

    def test_registration(self):
        self.assertIn("TORCH_LIBRARY_FRAGMENT(step_ops, m)", _CPP)
        self.assertIn('m.def("gpt2_mlp_fused6_step"', _CPP)


class TestAVXCodegenGEMV(unittest.TestCase):
    """Verify GEMV microkernel: signature, accumulators, intrinsics."""

    def test_gemv_signature(self):
        self.assertIn("static void gemv_tiled_chunk(", _CPP)
        self.assertIn("const float* __restrict__ x", _CPP)
        self.assertIn("const float* __restrict__ W", _CPP)
        self.assertIn("const float* __restrict__ bias", _CPP)
        self.assertIn("float* __restrict__ y", _CPP)

    def test_gemv_accumulators(self):
        # 4 accumulators from register_block=4
        self.assertIn("__m512 acc0 = bias0;", _CPP)
        self.assertIn("__m512 acc1 = bias1;", _CPP)
        self.assertIn("__m512 acc2 = bias2;", _CPP)
        self.assertIn("__m512 acc3 = bias3;", _CPP)

    def test_gemv_broadcast(self):
        self.assertIn("_mm512_set1_ps(x[k])", _CPP)

    def test_gemv_vector_load(self):
        self.assertIn("_mm512_loadu_ps(W + k * N + n)", _CPP)

    def test_gemv_fmadd(self):
        self.assertIn("_mm512_fmadd_ps(xk, w0, acc0)", _CPP)
        self.assertIn("_mm512_fmadd_ps(xk, w3, acc3)", _CPP)

    def test_gemv_store(self):
        self.assertIn("_mm512_storeu_ps(y + n", _CPP)

    def test_omp_gemv_wrapper(self):
        self.assertIn("static void avx512_omp_gemv(", _CPP)
        self.assertIn("#pragma omp parallel", _CPP)
        self.assertIn("omp_get_num_threads()", _CPP)
        self.assertIn("gemv_tiled_chunk(x, W, bias, y, K, N, n_start, n_end)", _CPP)


class TestAVXCodegenGEMM(unittest.TestCase):
    """Verify GEMM microkernel: template, 2D acc, instantiations."""

    def test_gemm_template(self):
        self.assertIn("template <int RM, int RN>", _CPP)
        self.assertIn("static inline void gemm_ukernel(", _CPP)

    def test_gemm_acc_array(self):
        self.assertIn("__m512 acc[RM][RN];", _CPP)

    def test_gemm_bias_init(self):
        self.assertIn("_mm512_loadu_ps(bias + n_offset + r * VL)", _CPP)

    def test_gemm_broadcast_a(self):
        self.assertIn("_mm512_set1_ps(A[i * K + kk])", _CPP)

    def test_gemm_fmadd(self):
        self.assertIn("_mm512_fmadd_ps(a, Bv[r], acc[i][r])", _CPP)

    def test_gemm_store(self):
        self.assertIn("_mm512_storeu_ps(C + i * N + n_offset + r * VL, acc[i][r])", _CPP)

    def test_omp_gemm_wrapper(self):
        self.assertIn("static void avx512_omp_gemm(", _CPP)
        self.assertIn("constexpr int RM = 4, RN = 4, VL = 16;", _CPP)

    def test_gemm_instantiations(self):
        # Main tile + remainder tails
        self.assertIn("gemm_ukernel<RM, RN>(", _CPP)
        self.assertIn("gemm_ukernel<RM, 1>(", _CPP)
        self.assertIn("gemm_ukernel<1, RN>(", _CPP)
        self.assertIn("gemm_ukernel<1, 1>(", _CPP)


class TestAVXCodegenGELU(unittest.TestCase):
    """Verify GELU lowering: constants, fast_tanh call, parallel for."""

    def test_gelu_constants(self):
        self.assertIn("0.7978845608028654f", _CPP)
        self.assertIn("0.044715f", _CPP)

    def test_gelu_fast_tanh_call(self):
        self.assertIn("fast_tanh_avx512(targ)", _CPP)

    def test_gelu_parallel_for(self):
        # M>1 path has parallelized GELU
        self.assertIn("#pragma omp parallel for if(M > 1)", _CPP)

    def test_gelu_scalar_tail(self):
        self.assertIn("std::tanh(ta)", _CPP)


class TestAVXCodegenEntryPoint(unittest.TestCase):
    """Verify entry point: M==1 branch, dim extraction, contiguity checks."""

    def test_function_signature(self):
        self.assertIn("torch::Tensor gpt2_mlp_fused6_step(", _CPP)
        self.assertIn("torch::Tensor hidden_states", _CPP)
        self.assertIn("torch::Tensor W_fc", _CPP)

    def test_contiguity_checks(self):
        self.assertIn('TORCH_CHECK(hidden_states.is_contiguous()', _CPP)
        self.assertIn('TORCH_CHECK(W_fc.is_contiguous()', _CPP)
        self.assertIn('TORCH_CHECK(W_proj.is_contiguous()', _CPP)

    def test_m_branch(self):
        self.assertIn("if (M == 1)", _CPP)

    def test_dim_extraction(self):
        self.assertIn("const int64_t D = hidden_states.size(1)", _CPP)
        self.assertIn("const int64_t K = W_fc.size(1)", _CPP)

    def test_gemv_calls(self):
        self.assertIn("avx512_omp_gemv(x_ptr, wfc_ptr, bfc_ptr, h_ptr, D, K)", _CPP)
        self.assertIn("avx512_omp_gemv(h_ptr, wproj_ptr, bproj_ptr, out_ptr, K, D)", _CPP)

    def test_gemm_calls(self):
        self.assertIn("avx512_omp_gemm(x_ptr, wfc_ptr, bfc_ptr, h_ptr, M, D, K)", _CPP)
        self.assertIn("avx512_omp_gemm(h_ptr, wproj_ptr, bproj_ptr, out_ptr, M, K, D)", _CPP)

    def test_temp_buffer(self):
        self.assertIn("auto h = torch::empty({1, K}", _CPP)
        self.assertIn("auto h = torch::empty({M, K}", _CPP)

    def test_output_alloc(self):
        self.assertIn("auto output = torch::empty({1, D}", _CPP)
        self.assertIn("auto output = torch::empty({M, D}", _CPP)


if __name__ == "__main__":
    unittest.main()
