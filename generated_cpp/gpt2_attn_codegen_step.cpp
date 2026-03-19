#include <torch/extension.h>
#include <torch/library.h>
#include <immintrin.h>
#include <cmath>
#include <omp.h>

#include <limits>

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


std::vector<torch::Tensor> gpt2_attn_codegen_step(
    torch::Tensor x,
    torch::Tensor W_attn,
    torch::Tensor b_attn,
    torch::Tensor W_proj,
    torch::Tensor b_proj,
    torch::Tensor past_key,
    torch::Tensor past_value,
    torch::Tensor attn_mask,
    int64_t num_heads,
    int64_t batch_size
) {
    const int64_t M = x.size(0);
    const int64_t embed_dim = x.size(1);
    const int64_t head_dim = embed_dim / num_heads;
    const int64_t S = M / batch_size;

    // 1. QKV projection with AVX-512 GEMM/GEMV
    auto qkv = torch::empty({M, 3 * embed_dim}, x.options());
    if (M == 1) {
        avx2_omp_gemv(x.data_ptr<float>(), W_attn.data_ptr<float>(),
                        b_attn.data_ptr<float>(), qkv.data_ptr<float>(),
                        embed_dim, 3 * embed_dim);
    } else {
        avx2_omp_gemm(x.data_ptr<float>(), W_attn.data_ptr<float>(),
                        b_attn.data_ptr<float>(), qkv.data_ptr<float>(),
                        M, embed_dim, 3 * embed_dim);
    }

    // 2. Split Q, K, V
    auto qkv_split = qkv.chunk(3, 1);
    auto Q = qkv_split[0].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto K = qkv_split[1].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto V = qkv_split[2].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3});

    // 3. Concatenate past KV
    bool has_past = past_key.numel() > 0;
    if (has_past) {
        K = torch::cat({past_key, K}, 2);
        V = torch::cat({past_value, V}, 2);
    }

    auto present_key = K;
    auto present_value = V;

    const int64_t total_len = K.size(2);

    // 4. Attention scores
    auto K_t = K.transpose(-2, -1).contiguous();
    auto scores = torch::matmul(Q.contiguous(), K_t);
    float scale = 1.0f / std::sqrt((float)head_dim);
    scores.mul_(scale);

    // 5. Causal mask (raw pointer, upper-triangle only) — skip for single-token decode
    if (S > 1) {
        auto causal = torch::zeros({S, total_len}, scores.options());
        float* mask_ptr = causal.data_ptr<float>();
        for (int64_t i = 0; i < S; i++) {
            int64_t fill_start = total_len - S + i + 1;
            for (int64_t j = fill_start; j < total_len; j++) {
                mask_ptr[i * total_len + j] = -1e4f;
            }
        }
        scores.add_(causal.unsqueeze(0).unsqueeze(0));
    }

    // 6. Add HF mask
    if (attn_mask.numel() > 0) scores.add_(attn_mask);

    // 7. Softmax
    auto attn_weights = at::softmax(scores, -1);

    // 8. Context
    auto context = torch::matmul(attn_weights, V.contiguous());
    auto merged = context.permute({0, 2, 1, 3}).contiguous().reshape({M, embed_dim});

    // 9. Output projection with AVX-512
    auto output = torch::empty({M, embed_dim}, x.options());
    if (M == 1) {
        avx2_omp_gemv(merged.data_ptr<float>(), W_proj.data_ptr<float>(),
                        b_proj.data_ptr<float>(), output.data_ptr<float>(),
                        embed_dim, embed_dim);
    } else {
        avx2_omp_gemm(merged.data_ptr<float>(), W_proj.data_ptr<float>(),
                        b_proj.data_ptr<float>(), output.data_ptr<float>(),
                        M, embed_dim, embed_dim);
    }

    return {output, present_key, present_value};
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_attn_codegen_step", gpt2_attn_codegen_step);
}
