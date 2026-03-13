#include <torch/extension.h>
#include <torch/library.h>
#include <immintrin.h>
#include <cmath>
#include <omp.h>

// Fast tanh approximation for AVX-512: [7,6] Pade rational form
static inline __m512 fast_tanh_avx512(__m512 x) {
    __m512 lim = _mm512_set1_ps(4.97f);
    __m512 x_clamped = _mm512_min_ps(_mm512_max_ps(x, _mm512_sub_ps(_mm512_setzero_ps(), lim)), lim);
    __m512 x2 = _mm512_mul_ps(x_clamped, x_clamped);
    __m512 x4 = _mm512_mul_ps(x2, x2);
    __m512 x6 = _mm512_mul_ps(x4, x2);

    __m512 c135135 = _mm512_set1_ps(135135.0f);
    __m512 c17325  = _mm512_set1_ps(17325.0f);
    __m512 c378    = _mm512_set1_ps(378.0f);
    __m512 c62370  = _mm512_set1_ps(62370.0f);
    __m512 c3150   = _mm512_set1_ps(3150.0f);
    __m512 c28     = _mm512_set1_ps(28.0f);

    __m512 num = _mm512_fmadd_ps(c17325, x2, c135135);
    num = _mm512_fmadd_ps(c378, x4, num);
    num = _mm512_add_ps(num, x6);

    __m512 den = _mm512_fmadd_ps(c62370, x2, c135135);
    den = _mm512_fmadd_ps(c3150, x4, den);
    den = _mm512_fmadd_ps(c28, x6, den);

    return _mm512_mul_ps(x_clamped, _mm512_div_ps(num, den));
}

// GEMV tiled micro-kernel: RN=4 AVX512 vectors (64 floats) per tile
static void gemv_tiled_chunk(const float* __restrict__ x,
                             const float* __restrict__ W,
                             const float* __restrict__ bias,
                             float* __restrict__ y,
                             int64_t K, int64_t N,
                             int64_t n_start, int64_t n_end) {
    constexpr int RN = 4;

    int64_t n = n_start;
    for (; n + RN * 16 <= n_end; n += RN * 16) {
        __m512 bias0 = _mm512_loadu_ps(bias + n);
        __m512 bias1 = _mm512_loadu_ps(bias + n + 16);
        __m512 bias2 = _mm512_loadu_ps(bias + n + 32);
        __m512 bias3 = _mm512_loadu_ps(bias + n + 48);
        __m512 acc0 = bias0;
        __m512 acc1 = bias1;
        __m512 acc2 = bias2;
        __m512 acc3 = bias3;

        for (int64_t k = 0; k < K; k++) {
            __m512 xk = _mm512_set1_ps(x[k]);
            __m512 w0 = _mm512_loadu_ps(W + k * N + n);
            __m512 w1 = _mm512_loadu_ps(W + k * N + n + 16);
            __m512 w2 = _mm512_loadu_ps(W + k * N + n + 32);
            __m512 w3 = _mm512_loadu_ps(W + k * N + n + 48);
            acc0 = _mm512_fmadd_ps(xk, w0, acc0);
            acc1 = _mm512_fmadd_ps(xk, w1, acc1);
            acc2 = _mm512_fmadd_ps(xk, w2, acc2);
            acc3 = _mm512_fmadd_ps(xk, w3, acc3);
        }

        _mm512_storeu_ps(y + n + 0, acc0);
        _mm512_storeu_ps(y + n + 16, acc1);
        _mm512_storeu_ps(y + n + 32, acc2);
        _mm512_storeu_ps(y + n + 48, acc3);
    }
}

// OpenMP parallel GEMV: y[0..N) = x[0..K) * W[K,N] + bias[0..N)
static void avx512_omp_gemv(const float* x, const float* W, const float* bias,
                            float* y, int64_t K, int64_t N) {
    constexpr int64_t TILE = 64;  // RN * VL
    int64_t ntiles = N / TILE;
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        // Distribute tiles evenly; excess threads idle
        int64_t tiles_per = ntiles / nthreads;
        int64_t extra = ntiles % nthreads;
        int64_t my_tiles = tiles_per + (tid < extra ? 1 : 0);
        int64_t n_start = (tid < extra ? tid * (tiles_per + 1)
                                       : extra * (tiles_per + 1) + (tid - extra) * tiles_per) * TILE;
        int64_t n_end = n_start + my_tiles * TILE;
        if (n_start < n_end) {
            gemv_tiled_chunk(x, W, bias, y, K, N, n_start, n_end);
        }
    }
}

// Register-blocked GEMM micro-kernel: RM rows x RN vectors
template <int RM, int RN>
static inline void gemm_ukernel(
    float* __restrict__ C, const float* __restrict__ A,
    const float* __restrict__ B, const float* __restrict__ bias,
    int64_t K, int64_t N, int64_t n_offset) {
    constexpr int VL = 16;
    __m512 acc[RM][RN];
    for (int i = 0; i < RM; i++)
        for (int r = 0; r < RN; r++)
            acc[i][r] = bias ? _mm512_loadu_ps(bias + n_offset + r * VL)
                             : _mm512_setzero_ps();
    for (int64_t kk = 0; kk < K; kk++) {
        __m512 Bv[RN];
        for (int r = 0; r < RN; r++)
            Bv[r] = _mm512_loadu_ps(B + kk * N + n_offset + r * VL);
        for (int i = 0; i < RM; i++) {
            __m512 a = _mm512_set1_ps(A[i * K + kk]);
            for (int r = 0; r < RN; r++)
                acc[i][r] = _mm512_fmadd_ps(a, Bv[r], acc[i][r]);
        }
    }
    for (int i = 0; i < RM; i++)
        for (int r = 0; r < RN; r++)
            _mm512_storeu_ps(C + i * N + n_offset + r * VL, acc[i][r]);
}

// OpenMP parallel GEMM: C[M,N] = A[M,K] * B[K,N] + bias[N]
static void avx512_omp_gemm(
    const float* A, const float* B, const float* bias,
    float* C, int64_t M, int64_t K, int64_t N) {
    constexpr int RM = 4, RN = 4, VL = 16;
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t base = (M / nthreads / RM) * RM;
        int64_t extra = M - base * nthreads;
        int64_t extra_threads = extra / RM;
        int64_t m_start, m_end;
        if (tid < extra_threads) {
            int64_t chunk = base + RM;
            m_start = tid * chunk;
            m_end = m_start + chunk;
        } else {
            m_start = extra_threads * (base + RM) + (tid - extra_threads) * base;
            m_end = m_start + base;
        }
        if (tid == nthreads - 1) m_end = M;
        int64_t ii = m_start;
        for (; ii + RM <= m_end; ii += RM) {
            for (int64_t jj = 0; jj + RN*VL <= N; jj += RN*VL)
                gemm_ukernel<RM, RN>(C + ii*N, A + ii*K, B, bias, K, N, jj);
            for (int64_t jj = (N/(RN*VL))*(RN*VL); jj + VL <= N; jj += VL)
                gemm_ukernel<RM, 1>(C + ii*N, A + ii*K, B, bias, K, N, jj);
        }
        for (; ii < m_end; ii++) {
            for (int64_t jj = 0; jj + RN*VL <= N; jj += RN*VL)
                gemm_ukernel<1, RN>(C + ii*N, A + ii*K, B, bias, K, N, jj);
            for (int64_t jj = (N/(RN*VL))*(RN*VL); jj + VL <= N; jj += VL)
                gemm_ukernel<1, 1>(C + ii*N, A + ii*K, B, bias, K, N, jj);
        }
    }
}

torch::Tensor gpt2_mlp_fused6_step(
    torch::Tensor hidden_states, torch::Tensor W_fc, torch::Tensor b_fc, torch::Tensor W_proj, torch::Tensor b_proj) {

    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(W_fc.is_contiguous(), "W_fc must be contiguous");
    TORCH_CHECK(W_proj.is_contiguous(), "W_proj must be contiguous");

    const int64_t M = hidden_states.size(0);

    if (M == 1) {
        const int64_t D = hidden_states.size(1);
        const int64_t K = W_fc.size(1);

        const float* x_ptr = hidden_states.data_ptr<float>();
        const float* wfc_ptr = W_fc.data_ptr<float>();
        const float* bfc_ptr = b_fc.data_ptr<float>();
        const float* wproj_ptr = W_proj.data_ptr<float>();
        const float* bproj_ptr = b_proj.data_ptr<float>();

        // GEMV1: h = x * W_fc + b_fc
        auto h = torch::empty({1, K}, hidden_states.options());
        float* h_ptr = h.data_ptr<float>();
        avx512_omp_gemv(x_ptr, wfc_ptr, bfc_ptr, h_ptr, D, K);

        // GELU activation (in-place on h)
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

        // GEMV2: out = h * W_proj + b_proj
        auto output = torch::empty({1, D}, hidden_states.options());
        float* out_ptr = output.data_ptr<float>();
        avx512_omp_gemv(h_ptr, wproj_ptr, bproj_ptr, out_ptr, K, D);

        return output;
    }

    // M > 1 (prefill) — AVX512 GEMM with fused bias
    const int64_t D = hidden_states.size(1);
    const int64_t K = W_fc.size(1);

    const float* x_ptr = hidden_states.data_ptr<float>();
    const float* wfc_ptr = W_fc.data_ptr<float>();
    const float* bfc_ptr = b_fc.data_ptr<float>();
    const float* wproj_ptr = W_proj.data_ptr<float>();
    const float* bproj_ptr = b_proj.data_ptr<float>();

    // GEMM1: h = hidden_states * W_fc + b_fc
    auto h = torch::empty({M, K}, hidden_states.options());
    float* h_ptr = h.data_ptr<float>();
    avx512_omp_gemm(x_ptr, wfc_ptr, bfc_ptr, h_ptr, M, D, K);

    // GELU activation (in-place, parallelized over M)
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

    // GEMM2: output = h * W_proj + b_proj
    auto output = torch::empty({M, D}, hidden_states.options());
    float* out_ptr = output.data_ptr<float>();
    avx512_omp_gemm(h_ptr, wproj_ptr, bproj_ptr, out_ptr, M, K, D);

    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused6_step", gpt2_mlp_fused6_step);
}

