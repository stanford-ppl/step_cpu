## Goal
Through plan 10, I wrote STeP programs that is equivalent to the c++ implementation in _apply_gpt2mlp_fused6 (
/home/ginasohn/step_cpu/darpa/modified/causal_language_modeling_codegen.py) and was able to codegen a similar c++ code from the STeP implementation in /home/ginasohn/step_cpu/darpa/modified/step_kernels.py.

Since the _apply_gpt2attn_fused uses similar GEMV and GEMM code generated from the codegen pass, I would like to apply them too.
You can see the code for the code generated implementation of GEMV and GEMM by:
```bash
# Enter the container
docker exec -it mocha-bg bash -c 
# In the container:
cat /root/.cache/mocha/gpt2_mlp_fused6/gpt2_mlp_fused6_step.cpp
```

Since I currently only have code for generating gemv and gemm, you can keep parts of the cpp code in _build_gpt2attn_fused (/home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py) for the other parts outside of gemv and gemm.

Implement a GPT2AttentionStepWrapper and functions to build and apply the kernel compiled from the combination of (1) code generated from gemv and gemm like in gpt2mlp_fused6 and (2) code from below (brought from _build_gpt2attn_fused in /home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py) for the other parts outside of gemv and gemm.


```cpp
#include <torch/extension.h>
#include <torch/library.h>
#include <immintrin.h>
#include <cmath>
#include <omp.h>
#include <limits>

// ============================================================
// Tiled GEMV micro-kernel (reused from fused6 MLP)
// ============================================================
static void gemv_tiled_chunk(const float* __restrict__ x,
                             const float* __restrict__ W,
                             const float* __restrict__ bias,
                             float* __restrict__ y,
                             int64_t K, int64_t N,
                             int64_t n_start, int64_t n_end) {
    constexpr int RN = 4;
    int64_t n = n_start;
    for (; n + RN * 16 <= n_end; n += RN * 16) {
        __m512 acc0 = _mm512_loadu_ps(bias + n);
        __m512 acc1 = _mm512_loadu_ps(bias + n + 16);
        __m512 acc2 = _mm512_loadu_ps(bias + n + 32);
        __m512 acc3 = _mm512_loadu_ps(bias + n + 48);
        for (int64_t k = 0; k < K; k++) {
            __m512 xk = _mm512_set1_ps(x[k]);
            const float* w_row = W + k * N + n;
            if (k + 2 < K) {
                _mm_prefetch((const char*)(W + (k+2)*N + n), _MM_HINT_T0);
                _mm_prefetch((const char*)(W + (k+2)*N + n + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(W + (k+2)*N + n + 32), _MM_HINT_T0);
                _mm_prefetch((const char*)(W + (k+2)*N + n + 48), _MM_HINT_T0);
            }
            acc0 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w_row), acc0);
            acc1 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w_row + 16), acc1);
            acc2 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w_row + 32), acc2);
            acc3 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w_row + 48), acc3);
        }
        _mm512_storeu_ps(y + n, acc0);
        _mm512_storeu_ps(y + n + 16, acc1);
        _mm512_storeu_ps(y + n + 32, acc2);
        _mm512_storeu_ps(y + n + 48, acc3);
    }
    for (; n + 16 <= n_end; n += 16) {
        __m512 acc = _mm512_loadu_ps(bias + n);
        for (int64_t k = 0; k < K; k++) {
            __m512 xk = _mm512_set1_ps(x[k]);
            acc = _mm512_fmadd_ps(xk, _mm512_loadu_ps(W + k * N + n), acc);
        }
        _mm512_storeu_ps(y + n, acc);
    }
    for (; n < n_end; n++) {
        float sum = bias[n];
        for (int64_t k = 0; k < K; k++) sum += x[k] * W[k * N + n];
        y[n] = sum;
    }
}

static void avx512_omp_gemv(const float* x, const float* W, const float* bias,
                            float* y, int64_t K, int64_t N) {
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t base_chunk = (N / nthreads) & ~15LL;
        int64_t remainder = N - base_chunk * nthreads;
        int64_t n_start, n_end;
        if (tid < remainder / 16) {
            int64_t my_chunk = base_chunk + 16;
            n_start = tid * my_chunk;
            n_end = n_start + my_chunk;
        } else {
            n_start = (remainder / 16) * (base_chunk + 16) + (tid - remainder / 16) * base_chunk;
            n_end = n_start + base_chunk;
        }
        if (n_end > N) n_end = N;
        if (n_start < n_end) {
            gemv_tiled_chunk(x, W, bias, y, K, N, n_start, n_end);
        }
    }
}

// Register-blocked GEMM micro-kernel
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
        if (kk + 2 < K)
            for (int r = 0; r < RN; r++)
                _mm_prefetch((const char*)(B + (kk+2)*N + n_offset + r*VL), _MM_HINT_T0);
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

std::vector<torch::Tensor> gpt2_attn_fused_step(
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
        avx512_omp_gemv(x.data_ptr<float>(), W_attn.data_ptr<float>(),
                        b_attn.data_ptr<float>(), qkv.data_ptr<float>(),
                        embed_dim, 3 * embed_dim);
    } else {
        avx512_omp_gemm(x.data_ptr<float>(), W_attn.data_ptr<float>(),
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

    auto present_key = K.clone();
    auto present_value = V.clone();

    const int64_t total_len = K.size(2);

    // 4. Attention scores (still using torch ops for inner attention)
    auto K_t = K.transpose(-2, -1).contiguous();
    auto scores = torch::matmul(Q.contiguous(), K_t);
    float scale = 1.0f / std::sqrt((float)head_dim);
    scores.mul_(scale);

    // 5. Causal mask
    {
        auto causal = torch::full({S, total_len}, -1e4, scores.options());
        for (int64_t i = 0; i < S; i++) {
            int64_t max_pos = total_len - S + i;
            for (int64_t j = 0; j <= max_pos && j < total_len; j++) {
                causal[i][j] = 0.0f;
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
        avx512_omp_gemv(merged.data_ptr<float>(), W_proj.data_ptr<float>(),
                        b_proj.data_ptr<float>(), output.data_ptr<float>(),
                        embed_dim, embed_dim);
    } else {
        avx512_omp_gemm(merged.data_ptr<float>(), W_proj.data_ptr<float>(),
                        b_proj.data_ptr<float>(), output.data_ptr<float>(),
                        M, embed_dim, embed_dim);
    }

    return {output, present_key, present_value};
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_attn_fused_step", gpt2_attn_fused_step);
}
```
## How to Run

Enter the container
```
  docker exec -it mocha-bg bash -c 
```
In the container:
```
source /home/dockeruser/mochaenv/bin/activate
cd /home/dockeruser/step_cpu
PYTHONPATH=/home/dockeruser/step_cpu:\$PYTHONPATH \
python3 darpa/modified/causal_language_modeling_mlp.py --mode infer --prompt 'Why is the sky blue?' --cpu-only --replace gpt2mlp_fused6
```
