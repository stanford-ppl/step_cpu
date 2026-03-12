#!/usr/bin/env python
# --------------------------------------------------------------
#  Benchmark – fine‑tune a tiny GPT‑2 model on the ELI5 dataset
#
#  Required: torch, transformers, datasets
#  Optional: psutil (for CPU % in table)
#            tabulate (for pretty table formatting)
# --------------------------------------------------------------

import argparse
import os
import math
import sys
import time
import warnings
import threading
from collections import namedtuple
from typing import List, Dict

import pathlib as _pathlib
import torch

_PROJECT_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])  # mocha/

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# --------------------------------------------------------------
#  Optional pretty‑table printer
# --------------------------------------------------------------
try:
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None

# --------------------------------------------------------------
#  Silence irrelevant warnings
# --------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="loss_type=None was set in the config but it is unrecognised.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.cuda",
)


# --------------------------------------------------------------
#  Load a *small* slice of the ELI5 dataset.
# --------------------------------------------------------------
def load_small_eli5(num_examples: int = 200, test_frac: float = 0.1):
    raw = load_dataset("dany0407/eli5_category", split=f"train[:{num_examples}]")
    test_size = max(1, int(num_examples * test_frac))
    split = raw.train_test_split(test_size=test_size, seed=42)
    return split.flatten()


# --------------------------------------------------------------
#  Tokenisation helpers (need the global ``tokenizer`` variable)
# --------------------------------------------------------------
def tkn_preprocess(inputs):
    """Tokenise a batch of answer strings."""
    return tokenizer([" ".join(x) for x in inputs["answers.text"]])


def organize_texts(input_dict):
    """Chunk a flat list of token ids into ``block_size``‑long pieces."""
    result = {}
    for k, v in input_dict.items():
        concat_v = sum(v, [])
        total_len = len(concat_v)
        if total_len >= block_size:
            total_len = (total_len // block_size) * block_size
        result[k] = [
            concat_v[i : i + block_size] for i in range(0, total_len, block_size)
        ]
    # For causal LM the labels are a copy of the inputs
    result["labels"] = result["input_ids"].copy()
    return result


# --------------------------------------------------------------
#  Optional CPU / GPU monitoring utilities
# --------------------------------------------------------------
try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. CPU metrics will not be collected.")


def _cpu_percent() -> float:
    """Return current CPU utilization percentage."""
    if not _PSUTIL_AVAILABLE:
        return 0.0
    return psutil.cpu_percent(interval=0.0)  # Non-blocking, returns since last call


def _cpu_info_from_samples(samples: List[float]) -> Dict:
    """Calculate CPU info from sampled percentages."""
    if not _PSUTIL_AVAILABLE or not samples:
        return {"total_logical": None, "sys_cpu_pct": None, "cores_used": None}

    total_logical = psutil.cpu_count(logical=True) or 1
    avg_pct = sum(samples) / len(samples)  # Average across all samples
    cores_used = avg_pct / 100.0 * total_logical

    return {
        "total_logical": total_logical,
        "sys_cpu_pct": avg_pct,
        "cores_used": cores_used,
    }


def _gpu_utilization() -> List[float]:
    """Return utilisation % for each visible GPU."""
    if not torch.cuda.is_available():
        return []
    utils = []
    for i in range(torch.cuda.device_count()):
        try:
            import pynvml

            if not hasattr(_gpu_utilization, "_init"):
                pynvml.nvmlInit()
                _gpu_utilization._init = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utils.append(float(util.gpu))
        except Exception:  # pragma: no cover
            # Fallback: use 0.0 instead of conflating with memory
            utils.append(0.0)
    return utils


def _gpu_memory_used() -> List[int]:
    """Return memory used (MiB) for each visible GPU."""
    if not torch.cuda.is_available():
        return []
    mem = []
    for i in range(torch.cuda.device_count()):
        try:
            mem_bytes = torch.cuda.memory_allocated(i)
            mem.append(int(mem_bytes / (1024 * 1024)))
        except Exception:
            mem.append(0)
    return mem


def _gpu_peak_memory() -> List[int]:
    """Return peak memory (MiB) for each visible GPU since last reset."""
    if not torch.cuda.is_available():
        return []
    mem = []
    for i in range(torch.cuda.device_count()):
        try:
            mem_bytes = torch.cuda.max_memory_allocated(i)
            mem.append(int(mem_bytes / (1024 * 1024)))
        except Exception:
            mem.append(0)
    return mem


def _used_gpu_indices(mem_list: List[int]) -> List[int]:
    """Return GPU indices that have >0 MiB allocated."""
    return [i for i, m in enumerate(mem_list) if m > 0]


# --------------------------------------------------------------
#  Stage timer – records wall‑clock time + optional resource usage
# --------------------------------------------------------------
class StageTimer:
    """
    Context manager that measures elapsed wall‑clock time and (optionally)
    CPU / GPU utilisation and GPU peak memory. Used when ``--no-instrument`` is *not*
    supplied.
    """

    def __init__(self, name: str, records: List[Dict], monitor_resources: bool = True):
        self.name = name
        self.records = records
        self.monitor_resources = monitor_resources
        self._stop_event = threading.Event()
        self._gpu_samples: List[List[float]] = []
        self._gpu_mem_samples: List[List[int]] = []
        self._cpu_samples: List[float] = []

    # ------------------------------------------------------------------
    # background thread that samples GPU utilisation, memory, and CPU every 0.1 s
    # ------------------------------------------------------------------
    def _resource_sampler(self):
        # Initialize CPU monitoring with a blocking call
        if _PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)  # Initialize, ignore first value

        while not self._stop_event.is_set():
            # Sample GPU if available
            if torch.cuda.is_available():
                self._gpu_samples.append(_gpu_utilization())
                self._gpu_mem_samples.append(_gpu_memory_used())

            # Sample CPU if available
            if _PSUTIL_AVAILABLE:
                self._cpu_samples.append(_cpu_percent())

            time.sleep(0.1)

    def __enter__(self):
        self.start = time.perf_counter()

        # Reset peak memory stats at the start of each stage (only if CUDA is available)
        if self.monitor_resources and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
            except Exception:
                pass  # Silently ignore if reset fails

        # Start background sampling thread
        if self.monitor_resources:
            self._sampler_thread = threading.Thread(
                target=self._resource_sampler, daemon=True
            )
            self._sampler_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start

        gpu_pct = None
        gpu_mem = None
        gpu_peak_mem = None
        used_gpu_idxs = []
        cpu_info = {}

        if self.monitor_resources:
            # Stop sampling thread
            self._stop_event.set()
            if hasattr(self, "_sampler_thread"):
                self._sampler_thread.join()

            # Process GPU samples
            if torch.cuda.is_available():
                # Average GPU utilization across samples
                if self._gpu_samples:
                    transposed = list(zip(*self._gpu_samples))
                    gpu_pct = [sum(col) / len(col) for col in transposed]

                # Peak memory from PyTorch's built-in tracker
                gpu_peak_mem = _gpu_peak_memory()

                # Also track peak from our samples for comparison
                if self._gpu_mem_samples:
                    transposed_mem = list(zip(*self._gpu_mem_samples))
                    sampled_peak = [max(col) if col else 0 for col in transposed_mem]
                    # Use the maximum of PyTorch peak and sampled peak
                    if gpu_peak_mem:
                        gpu_mem = [
                            max(a, b) for a, b in zip(gpu_peak_mem, sampled_peak)
                        ]
                    else:
                        gpu_mem = sampled_peak
                else:
                    gpu_mem = gpu_peak_mem

                used_gpu_idxs = _used_gpu_indices(gpu_mem or [])

            # Process CPU samples
            cpu_info = _cpu_info_from_samples(self._cpu_samples)

        self.records.append(
            {
                "stage": self.name,
                "time_s": self.elapsed,
                "sys_cpu_pct": cpu_info.get("sys_cpu_pct"),
                "cores_used": cpu_info.get("cores_used"),
                "gpu_pct": gpu_pct,
                "gpu_mem_mib": gpu_mem,
                "used_gpu_idxs": used_gpu_idxs,
            }
        )


# --------------------------------------------------------------
#  No‑op timer – used when ``--no-instrument`` is set
# --------------------------------------------------------------
class NoOpTimer:
    """Context‑manager that does nothing – used when instrumentation is disabled."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# Named tuple for benchmark results
BenchmarkResult = namedtuple("BenchmarkResult", ["perplexity", "elapsed_time"])


# --------------------------------------------------------------
#  STeP module replacement infrastructure
# --------------------------------------------------------------


class GPT2MLPStepWrapper(torch.nn.Module):
    """Drop-in replacement for GPT2MLP using a STeP-compiled C++ kernel.

    In eval mode, uses the compiled kernel for inference.
    In training mode, falls through to the original HuggingFace forward
    (STeP has no autograd support).
    """

    def __init__(self, mlp_module, compiled_kernel):
        super().__init__()
        self.c_fc = mlp_module.c_fc
        self.c_proj = mlp_module.c_proj
        self.act = mlp_module.act
        self.dropout = mlp_module.dropout
        self._compiled_kernel = compiled_kernel

    def _original_forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)

    def forward(self, hidden_states):
        if self.training:
            return self._original_forward(hidden_states)
        shape = hidden_states.shape  # [batch, seq, 768]
        x2d = hidden_states.reshape(shape[0] * shape[1], shape[2]).contiguous()

        # print(f"GPT2MLPStepWrapper forward called")
        out = self._compiled_kernel(
            x2d,
            self.c_fc.weight,
            self.c_fc.bias,
            self.c_proj.weight,
            self.c_proj.bias,
        )
        return out.reshape(shape)


def _build_gpt2mlp_fused6():
    """Compile the AVX512+OpenMP tiled GEMV + Pade-tanh GELU fused GPT2MLP C++ kernel."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print(
        "Building AVX512+OpenMP tiled GEMV + Pade-tanh GELU fused GPT2MLP C++ kernel..."
    )

    _FUSED6_CPP_SOURCE = r"""
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

// ============================================================
// Tiled GEMV micro-kernel: RN=4 AVX512 vectors (64 floats) per tile
// with software prefetch 2 K-iterations ahead
// ============================================================
static void gemv_tiled_chunk(const float* __restrict__ x,
                             const float* __restrict__ W,
                             const float* __restrict__ bias,
                             float* __restrict__ y,
                             int64_t K, int64_t N,
                             int64_t n_start, int64_t n_end) {
    constexpr int RN = 4;  // 4 AVX512 registers = 64 floats per tile

    // Process in 64-float tiles
    int64_t n = n_start;
    for (; n + RN * 16 <= n_end; n += RN * 16) {
        __m512 acc0 = _mm512_loadu_ps(bias + n);
        __m512 acc1 = _mm512_loadu_ps(bias + n + 16);
        __m512 acc2 = _mm512_loadu_ps(bias + n + 32);
        __m512 acc3 = _mm512_loadu_ps(bias + n + 48);

        for (int64_t k = 0; k < K; k++) {
            __m512 xk = _mm512_set1_ps(x[k]);
            const float* w_row = W + k * N + n;

            // Prefetch weight row 2 iterations ahead
            if (k + 2 < K) {
                _mm_prefetch((const char*)(W + (k + 2) * N + n), _MM_HINT_T0);
                _mm_prefetch((const char*)(W + (k + 2) * N + n + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(W + (k + 2) * N + n + 32), _MM_HINT_T0);
                _mm_prefetch((const char*)(W + (k + 2) * N + n + 48), _MM_HINT_T0);
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

    // Handle remaining 16-float chunks (if chunk not multiple of 64)
    for (; n + 16 <= n_end; n += 16) {
        __m512 acc = _mm512_loadu_ps(bias + n);
        for (int64_t k = 0; k < K; k++) {
            __m512 xk = _mm512_set1_ps(x[k]);
            acc = _mm512_fmadd_ps(xk, _mm512_loadu_ps(W + k * N + n), acc);
        }
        _mm512_storeu_ps(y + n, acc);
    }

    // Scalar tail
    for (; n < n_end; n++) {
        float sum = bias[n];
        for (int64_t k = 0; k < K; k++) {
            sum += x[k] * W[k * N + n];
        }
        y[n] = sum;
    }
}

// OpenMP parallel GEMV: y[0..N) = x[0..K) * W[K,N] + bias[0..N)
static void avx512_omp_gemv(const float* x, const float* W, const float* bias,
                            float* y, int64_t K, int64_t N) {
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        // Partition N into 16-aligned chunks
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

// ============================================================
// Register-blocked GEMM micro-kernel: RM rows x RN vectors
// with bias fusion and software prefetch
// ============================================================
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

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused6_step", gpt2_mlp_fused6_step);
}
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_mlp_fused6"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_mlp_fused6_step.cpp"
    cpp_path.write_text(_FUSED6_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_mlp_fused6",
        sources=[str(cpp_path)],
        extra_cflags=[
            "-O3",
            "-std=c++17",
            "-march=native",
            "-fopenmp",
            "-mavx512f",
            "-mfma",
        ],
        extra_ldflags=["-fopenmp"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_mlp_fused6_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2mlp_fused6(model):
    """Replace all transformer block MLPs with GPT2MLPStepWrapper using the tiled GEMV + Pade-tanh GELU kernel."""
    compiled_kernel = _build_gpt2mlp_fused6()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.mlp = GPT2MLPStepWrapper(block.mlp, compiled_kernel)
    print(
        f"Replaced {len(blocks)} GPT2MLP blocks with tiled GEMV + Pade-tanh GELU kernel."
    )


_REPLACEMENT_REGISTRY = {
    "gpt2mlp_fused6": _apply_gpt2mlp_fused6,
}


def apply_replacements(model, replace_names):
    """Apply named submodule replacements. Call after model.eval(), before inference."""
    if not replace_names:
        return
    for name in replace_names:
        fn = _REPLACEMENT_REGISTRY.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown replacement: {name!r}. Known: {list(_REPLACEMENT_REGISTRY)}"
            )
        fn(model)


def parse_clm(args):
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a tiny GPT‑2 fine‑tuning run on the ELI5 dataset. "
            "Use --cpu-only to force CPU‑only, --no-instrument to skip per‑stage "
            "resource measurement."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force the whole pipeline onto the CPU (disables CUDA).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of raw ELI5 examples to load (default: 200).",
    )
    parser.add_argument(
        "--no-instrument",
        action="store_true",
        help="Disable per‑stage instrumentation – only total runtime will be printed.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        default="train",
        help="Run training ('train') or inference ('infer'). Default: train.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for inference mode (required when --mode infer).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="clm-example-model",
        help=(
            "Path to a fine‑tuned model directory. In inference mode the script "
            "will try this path first and fall back to the base distilgpt2. "
            "In training mode the model is saved here after training. "
            "(default: clm-example-model)"
        ),
    )
    parser.add_argument(
        "--replace",
        nargs="*",
        default=[],
        metavar="MODULE",
        help=(
            "Replace named model submodules with STeP-compiled kernels. "
            "Supported: gpt2mlp_fused6. "
            "Example: --replace gpt2mlp_fused6"
        ),
    )
    return parser.parse_args(args)


def main(args=None):
    args = parse_clm(args)
    # --------------------------------------------------------------
    # Choose timer implementation
    # --------------------------------------------------------------
    TimerClass = NoOpTimer if args.no_instrument else StageTimer

    # --------------------------------------------------------------
    # Container for timing data
    # --------------------------------------------------------------
    records: List[Dict] = []

    global tokenizer, block_size

    # --------------------------------------------------------------
    # Overall wall‑clock start
    # --------------------------------------------------------------
    overall_start = time.perf_counter()

    # --------------------------------------------------------------
    # System info
    # --------------------------------------------------------------
    print("\n=== System resources ===")
    print(f"Available CPUs : {os.cpu_count()}")
    if args.cpu_only:
        print("Running in CPU-only mode")
    else:
        print(f"Available GPUs : {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(
                    f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB)"
                )
    print("=" * 50 + "\n")

    # --------------------------------------------------------------
    # Inference mode
    # --------------------------------------------------------------
    if args.mode == "infer":
        if args.prompt is None:
            print("Error: --prompt is required when --mode infer")
            return BenchmarkResult(perplexity=None, elapsed_time=0.0)

        # Try to load fine‑tuned weights; fall back to the base model.
        use_finetuned = os.path.isdir(args.model_path) and os.path.isfile(
            os.path.join(args.model_path, "config.json")
        )
        model_id = args.model_path if use_finetuned else "distilbert/distilgpt2"

        if use_finetuned:
            print(f"Loading fine‑tuned model from {args.model_path}")
        else:
            print("No fine‑tuned model found — using base distilgpt2")

        with TimerClass("create_tokenizer", records):
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token

        with TimerClass("load_model", records):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            device = torch.device("cpu")
            if not args.cpu_only and torch.cuda.is_available():
                device = torch.device("cuda")
            model.to(device)

        if args.replace:
            print(f"\nApplying module replacements: {args.replace}")
            with TimerClass("apply_replacements", records):
                apply_replacements(model, args.replace)
        model.eval()

        with TimerClass("tokenize_prompt", records):
            inputs = tokenizer(args.prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with TimerClass("generate", records):
            with torch.no_grad():
                print("generate started")
                output_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=20,
                    do_sample=False,
                )
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"\n=== Generated Text ===\n{generated_text}\n")

        overall_end = time.perf_counter()
        total_elapsed = overall_end - overall_start

        if not args.no_instrument and records:
            all_used = set()
            for r in records:
                all_used.update(r.get("used_gpu_idxs", []))
            used_ordered = sorted(all_used)

            if used_ordered:
                print("\n=== Benchmark Summary (GPU memory shows PEAK) ===\n")
            else:
                print("\n=== Benchmark Summary (CPU-only mode) ===\n")

            headers = ["Stage", "Time (s)", "CPU %", "Cores Used"]
            for i in used_ordered:
                headers += [f"GPU {i} %", f"GPU {i} Peak (MiB)"]

            rows = []
            for r in records:
                row = [
                    r["stage"],
                    f"{r['time_s']:.2f}",
                    (
                        f"{r['sys_cpu_pct']:.1f}"
                        if r.get("sys_cpu_pct") is not None
                        else "-"
                    ),
                    (
                        f"{r['cores_used']:.1f}"
                        if r.get("cores_used") is not None
                        else "-"
                    ),
                ]
                gpu_pct = r.get("gpu_pct") or []
                gpu_mem = r.get("gpu_mem_mib") or []
                for i in used_ordered:
                    pct = gpu_pct[i] if i < len(gpu_pct) else None
                    mem = gpu_mem[i] if i < len(gpu_mem) else None
                    row.append(f"{pct:.1f}" if pct is not None else "-")
                    row.append(str(mem) if mem is not None else "-")
                rows.append(row)

            total_time = sum(r["time_s"] for r in records)
            total_row = ["TOTAL", f"{total_time:.2f}"] + ["-"] * (len(headers) - 2)
            rows.append(total_row)

            if tabulate:
                print(tabulate(rows, headers=headers, tablefmt="github"))
            else:
                col_widths = [
                    max(len(str(v)) for v in col) for col in zip(*([headers] + rows))
                ]
                fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
                print(fmt.format(*headers))
                print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
                for row in rows:
                    print(fmt.format(*row))
            print("\n" + "=" * 50)
        else:
            print("\n=== Total Runtime ===")
            print(f"{total_elapsed:.2f} seconds")

        return BenchmarkResult(perplexity=None, elapsed_time=total_elapsed)

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------
    with TimerClass("load_dataset", records):
        eli5 = load_small_eli5(num_examples=args.samples, test_frac=0.1)

    # --------------------------------------------------------------
    # Tokenizer creation
    # --------------------------------------------------------------
    with TimerClass("create_tokenizer", records):
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------------
    # Tokenise the dataset
    # --------------------------------------------------------------
    with TimerClass("tokenise", records):
        tokenized_eli5 = eli5.map(
            tkn_preprocess,
            batched=True,
            num_proc=4,
            remove_columns=eli5["train"].column_names,
        )

    # --------------------------------------------------------------
    # Chunk token streams into fixed‑size blocks
    # --------------------------------------------------------------
    block_size = 128
    with TimerClass("blockify", records):
        eli5_dataset = tokenized_eli5.map(organize_texts, batched=True, num_proc=4)

    # --------------------------------------------------------------
    # Load the model & move to the appropriate device
    # --------------------------------------------------------------
    with TimerClass("load_model", records):
        model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
        device = torch.device("cpu")
        if not args.cpu_only and torch.cuda.is_available():
            device = torch.device("cuda")
        model.to(device)

    # --------------------------------------------------------------
    # Data collator
    # --------------------------------------------------------------
    with TimerClass("create_collator", records):
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # --------------------------------------------------------------
    # TrainingArguments
    # --------------------------------------------------------------
    with TimerClass("training_args", records):
        training_args = TrainingArguments(
            output_dir="clm-example-model",
            eval_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=False,
            save_strategy="no",
            use_cpu=args.cpu_only,
        )

    # --------------------------------------------------------------
    # Trainer construction
    # --------------------------------------------------------------
    with TimerClass("create_trainer", records):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=eli5_dataset["train"],
            eval_dataset=eli5_dataset["test"],
            data_collator=data_collator,
            processing_class=tokenizer,  # needed for correct padding handling
        )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------
    with TimerClass("train", records):
        trainer.train()

    # --------------------------------------------------------------
    # Save fine‑tuned model
    # --------------------------------------------------------------
    with TimerClass("save_model", records):
        model.save_pretrained(args.model_path)
        tokenizer.save_pretrained(args.model_path)
        print(f"\n=== Model saved to {args.model_path} ===")

    # --------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------
    with TimerClass("evaluate", records):
        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results["eval_loss"])
        print(f"\n=== Final Perplexity: {perplexity:.2f} ===\n")

    # --------------------------------------------------------------
    # Overall wall‑clock end
    # --------------------------------------------------------------
    overall_end = time.perf_counter()
    total_elapsed = overall_end - overall_start

    # --------------------------------------------------------------
    # Benchmark summary (per‑stage)
    # --------------------------------------------------------------
    if not args.no_instrument and records:
        # Check if any GPU was actually used
        all_used = set()
        for r in records:
            all_used.update(r.get("used_gpu_idxs", []))
        used_ordered = sorted(all_used)

        if used_ordered:
            print("\n=== Benchmark Summary (GPU memory shows PEAK) ===\n")
        else:
            print("\n=== Benchmark Summary (CPU-only mode) ===\n")

        # Header
        headers = ["Stage", "Time (s)", "CPU %", "Cores Used"]
        for i in used_ordered:
            headers += [f"GPU {i} %", f"GPU {i} Peak (MiB)"]

        rows = []
        for r in records:
            row = [
                r["stage"],
                f"{r['time_s']:.2f}",
                f"{r['sys_cpu_pct']:.1f}" if r.get("sys_cpu_pct") is not None else "-",
                f"{r['cores_used']:.1f}" if r.get("cores_used") is not None else "-",
            ]

            gpu_pct = r.get("gpu_pct") or []
            gpu_mem = r.get("gpu_mem_mib") or []
            for i in used_ordered:
                pct = gpu_pct[i] if i < len(gpu_pct) else None
                mem = gpu_mem[i] if i < len(gpu_mem) else None
                row.append(f"{pct:.1f}" if pct is not None else "-")
                row.append(str(mem) if mem is not None else "-")
            rows.append(row)

        # TOTAL line
        total_time = sum(r["time_s"] for r in records)
        total_row = ["TOTAL", f"{total_time:.2f}"] + ["-"] * (len(headers) - 2)
        rows.append(total_row)

        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="github"))
        else:
            # very simple fallback formatting
            col_widths = [
                max(len(str(v)) for v in col) for col in zip(*([headers] + rows))
            ]
            fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
            print(fmt.format(*headers))
            print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
            for row in rows:
                print(fmt.format(*row))
        print("\n" + "=" * 50)

    else:
        print("\n=== Total Runtime ===")
        print(f"{total_elapsed:.2f} seconds")
    return BenchmarkResult(perplexity=perplexity, elapsed_time=total_elapsed)


if __name__ == "__main__":
    main()
