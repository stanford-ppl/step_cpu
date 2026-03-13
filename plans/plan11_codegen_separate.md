If you take a look at /home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py, the kernels use different code for prefill and decode (M==1), which is a very common practice for optimizing for ML. 

Therefore, modity the code generation logic so that it has the program for when M==1 (decode) and M>1 (prefill). The AvxCodegen can have some hardcoding for checking the outermost dimension of hidden_states and generating the if and else statements to generate the code.

```
class AVXCodegen:
    """Generates AVX512+OpenMP C++ from a GEMV StepProgram and a GEMM StepProgram."""

    def __init__(
        self,
        decode: StepProgram,
        prefill: StepProgram,
    ):
```

    
```

torch::Tensor gpt2_mlp_fused6_step(
    torch::Tensor hidden_states, torch::Tensor W_fc,
    torch::Tensor b_fc, torch::Tensor W_proj, torch::Tensor b_proj) {

    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(W_fc.is_contiguous(), "W_fc must be contiguous");
    TORCH_CHECK(W_proj.is_contiguous(), "W_proj must be contiguous");

    const int64_t M = hidden_states.size(0);

    if (M == 1) {
        const int64_t D = hidden_states.size(1);  // 768
        const int64_t K = W_fc.size(1);            // 3072

        const float* x_ptr = hidden_states.data_ptr<float>();
        const float* wfc_ptr = W_fc.data_ptr<float>();
        const float* bfc_ptr = b_fc.data_ptr<float>();
        const float* wproj_ptr = W_proj.data_ptr<float>();
        const float* bproj_ptr = b_proj.data_ptr<float>();

        // GEMV1: h = x * W_fc + b_fc  [1x768] x [768x3072] -> [1x3072]
        auto h = torch::empty({1, K}, hidden_states.options());
        float* h_ptr = h.data_ptr<float>();
        avx512_omp_gemv(x_ptr, wfc_ptr, bfc_ptr, h_ptr, D, K);

        // Fused Pade-tanh GELU on h (3072 floats, fits in L1)
        const __m512 sqrt2overpi = _mm512_set1_ps(0.7978845608028654f);
        const __m512 gelu_coeff  = _mm512_set1_ps(0.044715f);
        const __m512 half_v      = _mm512_set1_ps(0.5f);
        const __m512 one_v       = _mm512_set1_ps(1.0f);

        int64_t k = 0;
        for (; k + 15 < K; k += 16) {
            __m512 v    = _mm512_loadu_ps(h_ptr + k);
            __m512 v2   = _mm512_mul_ps(v, v);
            __m512 v3   = _mm512_mul_ps(v2, v);
            __m512 inner = _mm512_fmadd_ps(gelu_coeff, v3, v);
            __m512 targ = _mm512_mul_ps(sqrt2overpi, inner);
            __m512 tval = fast_tanh_avx512(targ);
            __m512 res  = _mm512_mul_ps(half_v, _mm512_mul_ps(v, _mm512_add_ps(one_v, tval)));
            _mm512_storeu_ps(h_ptr + k, res);
        }
        for (; k < K; k++) {
            float xv = h_ptr[k];
            float x3 = xv * xv * xv;
            float ta = 0.7978845608028654f * (xv + 0.044715f * x3);
            h_ptr[k] = 0.5f * xv * (1.0f + std::tanh(ta));
        }

        // GEMV2: out = h * W_proj + b_proj  [1x3072] x [3072x768] -> [1x768]
        auto output = torch::empty({1, D}, hidden_states.options());
        float* out_ptr = output.data_ptr<float>();
        avx512_omp_gemv(h_ptr, wproj_ptr, bproj_ptr, out_ptr, K, D);

        return output;
    }

    // M > 1 (prefill) — hand-written AVX512 GEMM with fused bias
    const int64_t D = hidden_states.size(1);  // 768
    const int64_t K = W_fc.size(1);           // 3072
    const float* x_ptr = hidden_states.data_ptr<float>();
    const float* wfc_ptr = W_fc.data_ptr<float>();
    const float* bfc_ptr = b_fc.data_ptr<float>();
    const float* wproj_ptr = W_proj.data_ptr<float>();
    const float* bproj_ptr = b_proj.data_ptr<float>();

    // GEMM1: h = hidden_states * W_fc + b_fc  (bias fused into GEMM)
    auto h = torch::empty({M, K}, hidden_states.options());
    float* h_ptr = h.data_ptr<float>();
    avx512_omp_gemm(x_ptr, wfc_ptr, bfc_ptr, h_ptr, M, D, K);

    // Fused Pade-tanh GELU (in-place, bias already applied)
    const __m512 sqrt2overpi = _mm512_set1_ps(0.7978845608028654f);
    const __m512 gelu_coeff  = _mm512_set1_ps(0.044715f);
    const __m512 half_v      = _mm512_set1_ps(0.5f);
    const __m512 one_v       = _mm512_set1_ps(1.0f);

    #pragma omp parallel for if(M > 1)
    for (int64_t m = 0; m < M; m++) {
        float* row = h_ptr + m * K;
        int64_t k = 0;
        for (; k + 15 < K; k += 16) {
            __m512 v     = _mm512_loadu_ps(row + k);
            __m512 v2    = _mm512_mul_ps(v, v);
            __m512 v3    = _mm512_mul_ps(v2, v);
            __m512 inner = _mm512_fmadd_ps(gelu_coeff, v3, v);
            __m512 targ  = _mm512_mul_ps(sqrt2overpi, inner);
            __m512 tval  = fast_tanh_avx512(targ);
            __m512 res   = _mm512_mul_ps(half_v, _mm512_mul_ps(v, _mm512_add_ps(one_v, tval)));
            _mm512_storeu_ps(row + k, res);
        }
        for (; k < K; k++) {
            float xv = row[k];
            float x3 = xv * xv * xv;
            float ta = 0.7978845608028654f * (xv + 0.044715f * x3);
            row[k] = 0.5f * xv * (1.0f + std::tanh(ta));
        }
    }

    // GEMM2: output = h * W_proj + b_proj  (bias fused into GEMM)
    auto output = torch::empty({M, D}, hidden_states.options());
    float* out_ptr = output.data_ptr<float>();
    avx512_omp_gemm(h_ptr, wproj_ptr, bproj_ptr, out_ptr, M, K, D);
    return output;
}

```
