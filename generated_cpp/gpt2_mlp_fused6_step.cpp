#include <torch/extension.h>
#include <torch/library.h>
#include <immintrin.h>
#include <cmath>
#include <omp.h>

// Fast tanh approximation: [7,6] Pade rational form
static inline __m256 fast_tanh_avx2(__m256 x) {
    __m256 lim = _mm256_set1_ps(4.97f);
    __m256 x_clamped = _mm256_min_ps(_mm256_max_ps(x, _mm256_sub_ps(_mm256_setzero_ps(), lim)), lim);
    __m256 x2 = _mm256_mul_ps(x_clamped, x_clamped);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    __m256 x6 = _mm256_mul_ps(x4, x2);

    __m256 c135135 = _mm256_set1_ps(135135.0f);
    __m256 c17325  = _mm256_set1_ps(17325.0f);
    __m256 c378    = _mm256_set1_ps(378.0f);
    __m256 c62370  = _mm256_set1_ps(62370.0f);
    __m256 c3150   = _mm256_set1_ps(3150.0f);
    __m256 c28     = _mm256_set1_ps(28.0f);

    __m256 num = _mm256_fmadd_ps(c17325, x2, c135135);
    num = _mm256_fmadd_ps(c378, x4, num);
    num = _mm256_add_ps(num, x6);

    __m256 den = _mm256_fmadd_ps(c62370, x2, c135135);
    den = _mm256_fmadd_ps(c3150, x4, den);
    den = _mm256_fmadd_ps(c28, x6, den);

    return _mm256_mul_ps(x_clamped, _mm256_div_ps(num, den));
}

// GEMV tiled micro-kernel: RN=4 vectors of 8 (32 floats) per tile
static void gemv_tiled_chunk(const float* __restrict__ x,
                             const float* __restrict__ W,
                             const float* __restrict__ bias,
                             float* __restrict__ y,
                             int64_t K, int64_t N,
                             int64_t n_start, int64_t n_end) {
    constexpr int RN = 4;

    int64_t n = n_start;
    for (; n + RN * 8 <= n_end; n += RN * 8) {
        __m256 bias0 = _mm256_loadu_ps(bias + n);
        __m256 bias1 = _mm256_loadu_ps(bias + n + 8);
        __m256 bias2 = _mm256_loadu_ps(bias + n + 16);
        __m256 bias3 = _mm256_loadu_ps(bias + n + 24);
        __m256 acc0 = bias0;
        __m256 acc1 = bias1;
        __m256 acc2 = bias2;
        __m256 acc3 = bias3;

        for (int64_t k = 0; k < K; k++) {
            __m256 xk = _mm256_set1_ps(x[k]);
            __m256 w0 = _mm256_loadu_ps(W + k * N + n);
            __m256 w1 = _mm256_loadu_ps(W + k * N + n + 8);
            __m256 w2 = _mm256_loadu_ps(W + k * N + n + 16);
            __m256 w3 = _mm256_loadu_ps(W + k * N + n + 24);
            acc0 = _mm256_fmadd_ps(xk, w0, acc0);
            acc1 = _mm256_fmadd_ps(xk, w1, acc1);
            acc2 = _mm256_fmadd_ps(xk, w2, acc2);
            acc3 = _mm256_fmadd_ps(xk, w3, acc3);
            if (k + 1 < K) {
                _mm_prefetch((const char*)(W + (k+1) * N + n), _MM_HINT_T0);
                _mm_prefetch((const char*)(W + (k+1) * N + n + 8), _MM_HINT_T0);
                _mm_prefetch((const char*)(W + (k+1) * N + n + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(W + (k+1) * N + n + 24), _MM_HINT_T0);
            }
        }

        _mm256_storeu_ps(y + n + 0, acc0);
        _mm256_storeu_ps(y + n + 8, acc1);
        _mm256_storeu_ps(y + n + 16, acc2);
        _mm256_storeu_ps(y + n + 24, acc3);
    }
}

// OpenMP parallel GEMV: y[0..N) = x[0..K) * W[K,N] + bias[0..N)
static void avx2_omp_gemv(const float* x, const float* W, const float* bias,
                          float* y, int64_t K, int64_t N) {
    constexpr int64_t TILE = 32;  // RN * VL
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
    constexpr int VL = 8;
    __m256 acc[RM][RN];
    for (int i = 0; i < RM; i++)
        for (int r = 0; r < RN; r++)
            acc[i][r] = bias ? _mm256_loadu_ps(bias + n_offset + r * VL)
                             : _mm256_setzero_ps();
    for (int64_t kk = 0; kk < K; kk++) {
        __m256 Bv[RN];
        for (int r = 0; r < RN; r++)
            Bv[r] = _mm256_loadu_ps(B + kk * N + n_offset + r * VL);
        if (kk + 1 < K) {
            for (int r = 0; r < RN; r++)
                _mm_prefetch((const char*)(B + (kk+1) * N + n_offset + r * VL), _MM_HINT_T0);
        }
        for (int i = 0; i < RM; i++) {
            __m256 a = _mm256_set1_ps(A[i * K + kk]);
            for (int r = 0; r < RN; r++)
                acc[i][r] = _mm256_fmadd_ps(a, Bv[r], acc[i][r]);
        }
    }
    for (int i = 0; i < RM; i++)
        for (int r = 0; r < RN; r++)
            _mm256_storeu_ps(C + i * N + n_offset + r * VL, acc[i][r]);
}

// OpenMP parallel GEMM: C[M,N] = A[M,K] * B[K,N] + bias[N]
static void avx2_omp_gemm(
    const float* A, const float* B, const float* bias,
    float* C, int64_t M, int64_t K, int64_t N) {
    constexpr int RM = 4, RN = 2, VL = 8;
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
        avx2_omp_gemv(x_ptr, wfc_ptr, bfc_ptr, h_ptr, D, K);

        // GELU activation (in-place on h)
        const __m256 sqrt2overpi = _mm256_set1_ps(0.7978845608028654f);
        const __m256 gelu_coeff  = _mm256_set1_ps(0.044715f);
        const __m256 half_v      = _mm256_set1_ps(0.5f);
        const __m256 one_v       = _mm256_set1_ps(1.0f);

        int64_t k = 0;
        for (; k + 7 < K; k += 8) {
            __m256 v    = _mm256_loadu_ps(h_ptr + k);
            __m256 v2   = _mm256_mul_ps(v, v);
            __m256 v3   = _mm256_mul_ps(v2, v);
            __m256 inner = _mm256_fmadd_ps(gelu_coeff, v3, v);
            __m256 targ = _mm256_mul_ps(sqrt2overpi, inner);
            __m256 tval = fast_tanh_avx2(targ);
            __m256 res  = _mm256_mul_ps(half_v, _mm256_mul_ps(v, _mm256_add_ps(one_v, tval)));
            _mm256_storeu_ps(h_ptr + k, res);
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
        avx2_omp_gemv(h_ptr, wproj_ptr, bproj_ptr, out_ptr, K, D);

        return output;
    }

    // M > 1 (prefill) — GEMM with fused bias
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
    avx2_omp_gemm(x_ptr, wfc_ptr, bfc_ptr, h_ptr, M, D, K);

    // GELU activation (in-place, parallelized over M)
    const __m256 sqrt2overpi = _mm256_set1_ps(0.7978845608028654f);
    const __m256 gelu_coeff  = _mm256_set1_ps(0.044715f);
    const __m256 half_v      = _mm256_set1_ps(0.5f);
    const __m256 one_v       = _mm256_set1_ps(1.0f);

    #pragma omp parallel for if(M > 1)
    for (int64_t m = 0; m < M; m++) {
        float* row = h_ptr + m * K;
        int64_t k = 0;
        for (; k + 7 < K; k += 8) {
            __m256 v     = _mm256_loadu_ps(row + k);
            __m256 v2    = _mm256_mul_ps(v, v);
            __m256 v3    = _mm256_mul_ps(v2, v);
            __m256 inner = _mm256_fmadd_ps(gelu_coeff, v3, v);
            __m256 targ  = _mm256_mul_ps(sqrt2overpi, inner);
            __m256 tval  = fast_tanh_avx2(targ);
            __m256 res   = _mm256_mul_ps(half_v, _mm256_mul_ps(v, _mm256_add_ps(one_v, tval)));
            _mm256_storeu_ps(row + k, res);
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
    avx2_omp_gemm(h_ptr, wproj_ptr, bproj_ptr, out_ptr, M, K, D);

    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused6_step", gpt2_mlp_fused6_step);
}

