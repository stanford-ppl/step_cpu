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


def _build_gpt2mlp_replacement():
    """Lazily compile the GPT2MLP STeP kernel and return the compiled callable."""
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    import math as _math
    import step

    def gpt2_mlp_step(hidden_states, W_fc, b_fc, W_proj, b_proj):
        M = hidden_states.shape[0]
        K_tile = 256
        N_tile = 256

        # c_fc matmul: hidden_states @ W_fc + b_fc
        act_fc = step.tensor_to_stream(hidden_states, vec=[M, K_tile])
        act_fc = step.Flatten(act_fc, min_rank=0, max_rank=1)
        act_fc_buf = step.Bufferize(act_fc, rank=1)
        act_fc_rep = step.Streamify(act_fc_buf, repeat_factor=[12], rank=1)

        W_fc_T = W_fc.T.contiguous()
        wfc_stream = step.tensor_to_stream(W_fc_T, vec=[N_tile, K_tile])

        fc_partial = step.BinaryMap(
            act_fc_rep, wfc_stream, lambda a, w: torch.mm(a, w.T)
        )
        fc_accum = step.Accum(fc_partial, rank=1)

        bfc_stream = step.tensor_to_stream(b_fc.unsqueeze(0), vec=[1, N_tile])
        bfc_stream = step.Flatten(bfc_stream, min_rank=0, max_rank=1)

        fc_biased = step.BinaryMap(fc_accum, bfc_stream, lambda a, b: a + b)

        # GELU
        c_sqrt = _math.sqrt(2.0 / _math.pi)
        c_pow = 0.044715

        x3 = step.UnaryMap(fc_biased, lambda a: torch.pow(a, 3.0))
        x_in = step.BinaryMap(fc_biased, x3, lambda a, b: a + c_pow * b)
        t_in = step.UnaryMap(x_in, lambda a: c_sqrt * a)
        t = step.UnaryMap(t_in, lambda a: torch.tanh(a))
        onep = step.UnaryMap(t, lambda a: 1.0 + a)
        gelu_out = step.BinaryMap(fc_biased, onep, lambda a, b: 0.5 * a * b)

        # c_proj matmul: gelu_out @ W_proj + b_proj
        proj_buf = step.Bufferize(gelu_out, rank=1)
        proj_act_rep = step.Streamify(proj_buf, repeat_factor=[3], rank=1)

        W_proj_T = W_proj.T.contiguous()
        wproj_stream = step.tensor_to_stream(W_proj_T, vec=[N_tile, K_tile])

        proj_partial = step.BinaryMap(
            proj_act_rep, wproj_stream, lambda a, w: torch.mm(a, w.T)
        )
        proj_accum = step.Accum(proj_partial, rank=1)

        bproj_stream = step.tensor_to_stream(b_proj.unsqueeze(0), vec=[1, N_tile])
        bproj_stream = step.Flatten(bproj_stream, min_rank=0, max_rank=1)

        proj_biased = step.BinaryMap(proj_accum, bproj_stream, lambda a, b: a + b)

        output = step.stream_to_tensor(proj_biased, like_tensor=hidden_states)
        return output

    gpt2_mlp_step._param_ndims = [2, 2, 1, 2, 1]
    return step.cpu_compile(gpt2_mlp_step)


def _apply_gpt2mlp(model):
    """Replace all transformer block MLPs with GPT2MLPStepWrapper."""
    compiled_kernel = _build_gpt2mlp_replacement()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.mlp = GPT2MLPStepWrapper(block.mlp, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2MLP blocks with STeP-compiled kernel.")


def _build_gpt2mlp_fused():
    """Compile the hand-written fused GPT2MLP C++ kernel and return the callable."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building hand-written fused GPT2MLP C++ kernel...")

    _FUSED_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>

torch::Tensor gpt2_mlp_fused_step(
    torch::Tensor hidden_states, torch::Tensor W_fc,
    torch::Tensor b_fc, torch::Tensor W_proj, torch::Tensor b_proj) {

    // c_fc: hidden_states @ W_fc + b_fc  (Conv1D layout: W is [in, out])
    auto h = torch::mm(hidden_states, W_fc);
    h.add_(b_fc);

    // GELU (tanh approximation) — single fused call
    h = at::gelu(h, "tanh");

    // c_proj: h @ W_proj + b_proj
    auto output = torch::mm(h, W_proj);
    output.add_(b_proj);

    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused_step", gpt2_mlp_fused_step);
}
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_mlp_fused"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_mlp_fused_step.cpp"
    cpp_path.write_text(_FUSED_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_mlp_fused",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_mlp_fused_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2mlp_fused(model):
    """Replace all transformer block MLPs with GPT2MLPStepWrapper using the fused kernel."""
    compiled_kernel = _build_gpt2mlp_fused()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.mlp = GPT2MLPStepWrapper(block.mlp, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2MLP blocks with hand-written fused kernel.")


def _build_gpt2mlp_loopfused():
    """Compile the loop-fused GPT2MLP C++ kernel (no intermediate tensor)."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building loop-fused GPT2MLP C++ kernel...")

    _LOOPFUSED_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>
#include <cmath>
#include <omp.h>

#define K_TILE 256

torch::Tensor gpt2_mlp_loopfused_step(
    torch::Tensor hidden_states, torch::Tensor W_fc,
    torch::Tensor b_fc, torch::Tensor W_proj, torch::Tensor b_proj) {

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2-D");
    TORCH_CHECK(W_fc.dim() == 2, "W_fc must be 2-D");
    TORCH_CHECK(W_proj.dim() == 2, "W_proj must be 2-D");

    const int64_t M   = hidden_states.size(0);  // tokens
    const int64_t D   = hidden_states.size(1);  // hidden dim (768)
    const int64_t K   = W_fc.size(1);           // intermediate dim (3072)
    const int64_t N   = W_proj.size(1);         // output dim (768)

    auto output = torch::empty({M, N}, hidden_states.options());

    const float* hs_ptr    = hidden_states.data_ptr<float>();
    const float* wfc_ptr   = W_fc.data_ptr<float>();
    const float* bfc_ptr   = b_fc.data_ptr<float>();
    const float* wproj_ptr = W_proj.data_ptr<float>();
    const float* bproj_ptr = b_proj.data_ptr<float>();
    float* out_ptr         = output.data_ptr<float>();

    static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    static constexpr float GELU_COEFF     = 0.044715f;

    #pragma omp parallel for schedule(dynamic)
    for (int64_t m = 0; m < M; m++) {
        const float* hs_row = hs_ptr + m * D;
        float* out_row      = out_ptr + m * N;

        // Initialize output row with b_proj
        for (int64_t j = 0; j < N; j++) {
            out_row[j] = bproj_ptr[j];
        }

        // Tile over intermediate dimension K
        for (int64_t k0 = 0; k0 < K; k0 += K_TILE) {
            const int64_t k_end = std::min(k0 + (int64_t)K_TILE, K);

            for (int64_t k = k0; k < k_end; k++) {
                // Compute intermediate_val = b_fc[k] + dot(hs_row, W_fc[:, k])
                float intermediate_val = bfc_ptr[k];
                const float* wfc_col = wfc_ptr + k;  // stride is K
                for (int64_t d = 0; d < D; d++) {
                    intermediate_val += hs_row[d] * wfc_col[d * K];
                }

                // GELU tanh approximation
                float x3 = intermediate_val * intermediate_val * intermediate_val;
                float tanh_arg = SQRT_2_OVER_PI * (intermediate_val + GELU_COEFF * x3);
                float gelu_val = 0.5f * intermediate_val * (1.0f + std::tanh(tanh_arg));

                // Accumulate: output[m, j] += gelu_val * W_proj[k, j]
                const float* wproj_row = wproj_ptr + k * N;
                for (int64_t j = 0; j < N; j++) {
                    out_row[j] += gelu_val * wproj_row[j];
                }
            }
        }
    }

    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_loopfused_step", gpt2_mlp_loopfused_step);
}
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_mlp_loopfused"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_mlp_loopfused_step.cpp"
    cpp_path.write_text(_LOOPFUSED_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_mlp_loopfused",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17", "-fopenmp", "-march=native"],
        extra_ldflags=["-fopenmp"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_mlp_loopfused_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2mlp_loopfused(model):
    """Replace all transformer block MLPs with GPT2MLPStepWrapper using the loop-fused kernel."""
    compiled_kernel = _build_gpt2mlp_loopfused()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.mlp = GPT2MLPStepWrapper(block.mlp, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2MLP blocks with loop-fused kernel.")


def _build_gpt2mlp_fused2():
    """Compile the BLAS-backed fused GPT2MLP C++ kernel and return the callable."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building BLAS-backed fused GPT2MLP C++ kernel...")

    _FUSED2_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>
#include <cmath>
#include <cblas.h>

torch::Tensor gpt2_mlp_fused2_step(
    torch::Tensor hidden_states, torch::Tensor W_fc,
    torch::Tensor b_fc, torch::Tensor W_proj, torch::Tensor b_proj) {

    // Step 1: c_fc matmul — keep torch::mm (BLAS-backed)
    auto h = torch::mm(hidden_states, W_fc);  // [M, 3072]

    // Step 2: Fused bias-add + GELU in a single pass
    const int64_t M = h.size(0);
    const int64_t K = h.size(1);   // 3072
    const int64_t N = W_proj.size(1); // 768
    float* h_ptr = h.data_ptr<float>();
    const float* bfc_ptr = b_fc.data_ptr<float>();
    static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    static constexpr float GELU_COEFF = 0.044715f;

    for (int64_t i = 0; i < M * K; i++) {
        float x = h_ptr[i] + bfc_ptr[i % K];
        float x3 = x * x * x;
        float tanh_arg = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
        h_ptr[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
    }

    // Step 3: c_proj matmul via cblas_sgemm
    // output = h @ W_proj + b_proj
    // Broadcast b_proj into output, then sgemm with beta=1
    auto output = b_proj.unsqueeze(0).expand({M, N}).contiguous().clone();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, h_ptr, K,
                W_proj.data_ptr<float>(), N,
                1.0f, output.data_ptr<float>(), N);

    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused2_step", gpt2_mlp_fused2_step);
}
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_mlp_fused2"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_mlp_fused2_step.cpp"
    cpp_path.write_text(_FUSED2_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_mlp_fused2",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17", "-march=native"],
        extra_ldflags=["-lopenblas"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_mlp_fused2_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2mlp_fused2(model):
    """Replace all transformer block MLPs with GPT2MLPStepWrapper using the BLAS-backed fused kernel."""
    compiled_kernel = _build_gpt2mlp_fused2()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.mlp = GPT2MLPStepWrapper(block.mlp, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2MLP blocks with BLAS-backed fused kernel.")


def _build_gpt2mlp_fused3():
    """Compile the AVX512-vectorized fused bias+GELU GPT2MLP C++ kernel."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building AVX512 fused bias+GELU GPT2MLP C++ kernel...")

    _FUSED3_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>
#include <immintrin.h>
#include <cmath>
#include <omp.h>

// Fast tanh approximation for AVX-512: [7,6] Pade rational form
// tanh(x) ~ x*(135135 + 17325*x^2 + 378*x^4 + x^6) /
//             (135135 + 62370*x^2 + 3150*x^4 + 28*x^6)
// Clamp |x| > 4.97 to +/-1 (tanh(4.97) > 0.99999)
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

    // num = 135135 + 17325*x^2 + 378*x^4 + x^6
    __m512 num = _mm512_fmadd_ps(c17325, x2, c135135);
    num = _mm512_fmadd_ps(c378, x4, num);
    num = _mm512_add_ps(num, x6);

    // den = 135135 + 62370*x^2 + 3150*x^4 + 28*x^6
    __m512 den = _mm512_fmadd_ps(c62370, x2, c135135);
    den = _mm512_fmadd_ps(c3150, x4, den);
    den = _mm512_fmadd_ps(c28, x6, den);

    return _mm512_mul_ps(x_clamped, _mm512_div_ps(num, den));
}

torch::Tensor gpt2_mlp_fused3_step(
    torch::Tensor hidden_states, torch::Tensor W_fc,
    torch::Tensor b_fc, torch::Tensor W_proj, torch::Tensor b_proj) {

    auto h = torch::mm(hidden_states, W_fc);

    const int64_t M = h.size(0);
    const int64_t K = h.size(1);
    float* h_ptr = h.data_ptr<float>();
    const float* bfc_ptr = b_fc.data_ptr<float>();

    const __m512 sqrt2overpi = _mm512_set1_ps(0.7978845608028654f);
    const __m512 gelu_coeff  = _mm512_set1_ps(0.044715f);
    const __m512 half_v      = _mm512_set1_ps(0.5f);
    const __m512 one_v       = _mm512_set1_ps(1.0f);

    // Process each row; K=3072 is divisible by 16
    #pragma omp parallel for if(M > 1)
    for (int64_t m = 0; m < M; m++) {
        float* row = h_ptr + m * K;
        int64_t k = 0;
        for (; k + 15 < K; k += 16) {
            __m512 hv = _mm512_loadu_ps(row + k);
            __m512 bv = _mm512_loadu_ps(bfc_ptr + k);
            __m512 x  = _mm512_add_ps(hv, bv);

            __m512 x2    = _mm512_mul_ps(x, x);
            __m512 x3    = _mm512_mul_ps(x2, x);
            __m512 inner = _mm512_fmadd_ps(gelu_coeff, x3, x);
            __m512 targ  = _mm512_mul_ps(sqrt2overpi, inner);
            __m512 tval  = fast_tanh_avx512(targ);
            __m512 res   = _mm512_mul_ps(half_v, _mm512_mul_ps(x, _mm512_add_ps(one_v, tval)));

            _mm512_storeu_ps(row + k, res);
        }
        // Scalar tail (K=3072 is 16-aligned, so this won't execute)
        for (; k < K; k++) {
            float x = row[k] + bfc_ptr[k];
            float x3 = x * x * x;
            float ta = 0.7978845608028654f * (x + 0.044715f * x3);
            row[k] = 0.5f * x * (1.0f + std::tanh(ta));
        }
    }

    auto output = torch::mm(h, W_proj);
    output.add_(b_proj);
    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused3_step", gpt2_mlp_fused3_step);
}
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_mlp_fused3"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_mlp_fused3_step.cpp"
    cpp_path.write_text(_FUSED3_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_mlp_fused3",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17", "-march=native", "-fopenmp", "-mavx512f", "-mfma"],
        extra_ldflags=["-fopenmp"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_mlp_fused3_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2mlp_fused3(model):
    """Replace all transformer block MLPs with GPT2MLPStepWrapper using the AVX512 fused bias+GELU kernel."""
    compiled_kernel = _build_gpt2mlp_fused3()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.mlp = GPT2MLPStepWrapper(block.mlp, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2MLP blocks with AVX512 fused bias+GELU kernel.")


def _build_gpt2mlp_fused4():
    """Compile the AVX512+OpenMP GEMV fused GPT2MLP C++ kernel."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building AVX512+OpenMP GEMV fused GPT2MLP C++ kernel...")

    _FUSED4_CPP_SOURCE = r"""
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

// Per-thread GEMV inner kernel: y[n_start..n_end) = x[0..K) * W[0..K, n_start..n_end) + bias[n_start..n_end)
// W is row-major [K, N], so W[k] row is contiguous.
static void gemv_chunk(const float* __restrict__ x,
                       const float* __restrict__ W,
                       const float* __restrict__ bias,
                       float* __restrict__ y,
                       int64_t K, int64_t N,
                       int64_t n_start, int64_t n_end) {
    const int64_t chunk = n_end - n_start;
    const int64_t nvec = chunk / 16;

    // Fixed-size accumulator array (max 3072/16 = 192)
    __m512 acc[192];

    // Initialize accumulators from bias
    for (int64_t v = 0; v < nvec; v++) {
        acc[v] = _mm512_loadu_ps(bias + n_start + v * 16);
    }

    // Accumulate: for each k, broadcast x[k] and FMA with W row
    for (int64_t k = 0; k < K; k++) {
        __m512 xk = _mm512_set1_ps(x[k]);
        const float* w_row = W + k * N + n_start;
        for (int64_t v = 0; v < nvec; v++) {
            acc[v] = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w_row + v * 16), acc[v]);
        }
    }

    // Store results
    for (int64_t v = 0; v < nvec; v++) {
        _mm512_storeu_ps(y + n_start + v * 16, acc[v]);
    }

    // Scalar tail (N is 768 or 3072, both divisible by 16, so this won't execute)
    for (int64_t n = n_start + nvec * 16; n < n_end; n++) {
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
        int64_t base_chunk = (N / nthreads) & ~15LL;  // round down to multiple of 16
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
            gemv_chunk(x, W, bias, y, K, N, n_start, n_end);
        }
    }
}

torch::Tensor gpt2_mlp_fused4_step(
    torch::Tensor hidden_states, torch::Tensor W_fc,
    torch::Tensor b_fc, torch::Tensor W_proj, torch::Tensor b_proj) {

    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(W_fc.is_contiguous(), "W_fc must be contiguous");
    TORCH_CHECK(W_proj.is_contiguous(), "W_proj must be contiguous");

    const int64_t M = hidden_states.size(0);

    if (M == 1) {
        // Fast GEMV path for autoregressive decode
        const int64_t D = hidden_states.size(1);  // 768
        const int64_t K = W_fc.size(1);            // 3072

        const float* x_ptr = hidden_states.data_ptr<float>();
        const float* wfc_ptr = W_fc.data_ptr<float>();
        const float* bfc_ptr = b_fc.data_ptr<float>();
        const float* wproj_ptr = W_proj.data_ptr<float>();
        const float* bproj_ptr = b_proj.data_ptr<float>();

        // GEMV1: h = x * W_fc + b_fc  [1×768] × [768×3072] → [1×3072]
        auto h = torch::empty({1, K}, hidden_states.options());
        float* h_ptr = h.data_ptr<float>();
        avx512_omp_gemv(x_ptr, wfc_ptr, bfc_ptr, h_ptr, D, K);

        // Fused GELU on h (3072 floats = 12KB, fits in L1, single-threaded)
        const __m512 sqrt2overpi = _mm512_set1_ps(0.7978845608028654f);
        const __m512 gelu_coeff  = _mm512_set1_ps(0.044715f);
        const __m512 half_v      = _mm512_set1_ps(0.5f);
        const __m512 one_v       = _mm512_set1_ps(1.0f);

        int64_t k = 0;
        for (; k + 15 < K; k += 16) {
            __m512 x  = _mm512_loadu_ps(h_ptr + k);
            __m512 x2    = _mm512_mul_ps(x, x);
            __m512 x3    = _mm512_mul_ps(x2, x);
            __m512 inner = _mm512_fmadd_ps(gelu_coeff, x3, x);
            __m512 targ  = _mm512_mul_ps(sqrt2overpi, inner);
            __m512 tval  = fast_tanh_avx512(targ);
            __m512 res   = _mm512_mul_ps(half_v, _mm512_mul_ps(x, _mm512_add_ps(one_v, tval)));
            _mm512_storeu_ps(h_ptr + k, res);
        }
        for (; k < K; k++) {
            float xv = h_ptr[k];
            float x3 = xv * xv * xv;
            float ta = 0.7978845608028654f * (xv + 0.044715f * x3);
            h_ptr[k] = 0.5f * xv * (1.0f + std::tanh(ta));
        }

        // GEMV2: out = h * W_proj + b_proj  [1×3072] × [3072×768] → [1×768]
        auto output = torch::empty({1, D}, hidden_states.options());
        float* out_ptr = output.data_ptr<float>();
        avx512_omp_gemv(h_ptr, wproj_ptr, bproj_ptr, out_ptr, K, D);

        return output;
    }

    // Fallback: M > 1 (prefill) — use torch::mm + fused bias+GELU
    auto h = torch::mm(hidden_states, W_fc);

    const int64_t K = h.size(1);
    float* h_ptr = h.data_ptr<float>();
    const float* bfc_ptr = b_fc.data_ptr<float>();

    const __m512 sqrt2overpi = _mm512_set1_ps(0.7978845608028654f);
    const __m512 gelu_coeff  = _mm512_set1_ps(0.044715f);
    const __m512 half_v      = _mm512_set1_ps(0.5f);
    const __m512 one_v       = _mm512_set1_ps(1.0f);

    #pragma omp parallel for if(M > 1)
    for (int64_t m = 0; m < M; m++) {
        float* row = h_ptr + m * K;
        int64_t k = 0;
        for (; k + 15 < K; k += 16) {
            __m512 hv = _mm512_loadu_ps(row + k);
            __m512 bv = _mm512_loadu_ps(bfc_ptr + k);
            __m512 x  = _mm512_add_ps(hv, bv);
            __m512 x2    = _mm512_mul_ps(x, x);
            __m512 x3    = _mm512_mul_ps(x2, x);
            __m512 inner = _mm512_fmadd_ps(gelu_coeff, x3, x);
            __m512 targ  = _mm512_mul_ps(sqrt2overpi, inner);
            __m512 tval  = fast_tanh_avx512(targ);
            __m512 res   = _mm512_mul_ps(half_v, _mm512_mul_ps(x, _mm512_add_ps(one_v, tval)));
            _mm512_storeu_ps(row + k, res);
        }
        for (; k < K; k++) {
            float x = row[k] + bfc_ptr[k];
            float x3 = x * x * x;
            float ta = 0.7978845608028654f * (x + 0.044715f * x3);
            row[k] = 0.5f * x * (1.0f + std::tanh(ta));
        }
    }

    auto output = torch::mm(h, W_proj);
    output.add_(b_proj);
    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused4_step", gpt2_mlp_fused4_step);
}
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_mlp_fused4"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_mlp_fused4_step.cpp"
    cpp_path.write_text(_FUSED4_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_mlp_fused4",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17", "-march=native", "-fopenmp", "-mavx512f", "-mfma"],
        extra_ldflags=["-fopenmp"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_mlp_fused4_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2mlp_fused4(model):
    """Replace all transformer block MLPs with GPT2MLPStepWrapper using the AVX512+OpenMP GEMV kernel."""
    compiled_kernel = _build_gpt2mlp_fused4()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.mlp = GPT2MLPStepWrapper(block.mlp, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2MLP blocks with AVX512+OpenMP GEMV kernel.")


def _build_gpt2mlp_fused5():
    """Compile the AVX512+OpenMP tiled GEMV + exp-based GELU fused GPT2MLP C++ kernel."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building AVX512+OpenMP tiled GEMV + exp-based GELU fused GPT2MLP C++ kernel...")

    _FUSED5_CPP_SOURCE = r"""
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

torch::Tensor gpt2_mlp_fused5_step(
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

    // Fallback: M > 1 (prefill) — use torch::mm + fused bias+Pade-tanh GELU
    auto h = torch::mm(hidden_states, W_fc);

    const int64_t K = h.size(1);
    float* h_ptr = h.data_ptr<float>();
    const float* bfc_ptr = b_fc.data_ptr<float>();

    const __m512 sqrt2overpi = _mm512_set1_ps(0.7978845608028654f);
    const __m512 gelu_coeff  = _mm512_set1_ps(0.044715f);
    const __m512 half_v      = _mm512_set1_ps(0.5f);
    const __m512 one_v       = _mm512_set1_ps(1.0f);

    #pragma omp parallel for if(M > 1)
    for (int64_t m = 0; m < M; m++) {
        float* row = h_ptr + m * K;
        int64_t k = 0;
        for (; k + 15 < K; k += 16) {
            __m512 hv = _mm512_loadu_ps(row + k);
            __m512 bv = _mm512_loadu_ps(bfc_ptr + k);
            __m512 x  = _mm512_add_ps(hv, bv);
            __m512 x2    = _mm512_mul_ps(x, x);
            __m512 x3    = _mm512_mul_ps(x2, x);
            __m512 inner = _mm512_fmadd_ps(gelu_coeff, x3, x);
            __m512 targ  = _mm512_mul_ps(sqrt2overpi, inner);
            __m512 tval  = fast_tanh_avx512(targ);
            __m512 res   = _mm512_mul_ps(half_v, _mm512_mul_ps(x, _mm512_add_ps(one_v, tval)));
            _mm512_storeu_ps(row + k, res);
        }
        for (; k < K; k++) {
            float x = row[k] + bfc_ptr[k];
            float x3 = x * x * x;
            float ta = 0.7978845608028654f * (x + 0.044715f * x3);
            row[k] = 0.5f * x * (1.0f + std::tanh(ta));
        }
    }

    auto output = torch::mm(h, W_proj);
    output.add_(b_proj);
    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused5_step", gpt2_mlp_fused5_step);
}
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_mlp_fused5"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_mlp_fused5_step.cpp"
    cpp_path.write_text(_FUSED5_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_mlp_fused5",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17", "-march=native", "-fopenmp", "-mavx512f", "-mfma"],
        extra_ldflags=["-fopenmp"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_mlp_fused5_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2mlp_fused5(model):
    """Replace all transformer block MLPs with GPT2MLPStepWrapper using the tiled GEMV + Pade-tanh GELU kernel."""
    compiled_kernel = _build_gpt2mlp_fused5()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.mlp = GPT2MLPStepWrapper(block.mlp, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2MLP blocks with tiled GEMV + exp-GELU kernel.")

def _build_gpt2mlp_fused6():
    """Compile the AVX512+OpenMP tiled GEMV + Pade-tanh GELU fused GPT2MLP C++ kernel."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building AVX512+OpenMP tiled GEMV + Pade-tanh GELU fused GPT2MLP C++ kernel...")

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
        extra_cflags=["-O3", "-std=c++17", "-march=native", "-fopenmp", "-mavx512f", "-mfma"],
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
    print(f"Replaced {len(blocks)} GPT2MLP blocks with tiled GEMV + Pade-tanh GELU kernel.")


# --------------------------------------------------------------
#  GPT2Attention STeP replacement infrastructure
# --------------------------------------------------------------


class GPT2AttentionStepWrapper(torch.nn.Module):
    """Drop-in replacement for GPT2Attention using a STeP-compiled C++ kernel.

    In eval mode, uses the compiled kernel for inference.
    In training mode, falls through to the original HuggingFace forward.
    """

    def __init__(self, attn_module, compiled_kernel):
        super().__init__()
        self.c_attn = attn_module.c_attn       # Conv1D(768, 2304) — QKV projection
        self.c_proj = attn_module.c_proj        # Conv1D(768, 768) — output projection
        self.attn_dropout = attn_module.attn_dropout
        self.resid_dropout = attn_module.resid_dropout
        self.num_heads = attn_module.num_heads  # 12
        self.head_dim = attn_module.head_dim    # 64
        self.embed_dim = attn_module.embed_dim  # 768
        self.scale_attn_weights = attn_module.scale_attn_weights
        self.layer_idx = attn_module.layer_idx
        self._compiled_kernel = compiled_kernel
        self._original_attn = attn_module       # training fallback

    def forward(self, hidden_states, past_key_values=None, cache_position=None,
                attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, **kwargs):
        if self.training:
            return self._original_attn(
                hidden_states, past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask, head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions, **kwargs,
            )

        # Eval path: call C++ kernel
        shape = hidden_states.shape  # [B, S, 768]
        B = shape[0]
        S = shape[1]
        x2d = hidden_states.reshape(B * S, self.embed_dim).contiguous()

        # Extract past key/value from Cache object (empty tensors if no cache)
        past_key = torch.empty(0)
        past_value = torch.empty(0)
        if past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) > 0:
            # Cache stores [B, num_heads, seq_len, head_dim]
            past_key = past_key_values.layers[self.layer_idx].keys
            past_value = past_key_values.layers[self.layer_idx].values

        # Prepare attention mask (empty if None)
        if attention_mask is not None:
            attn_mask = attention_mask.contiguous()
        else:
            attn_mask = torch.empty(0)

        # C++ kernel returns [output, present_key, present_value]
        result = self._compiled_kernel(
            x2d,
            self.c_attn.weight,
            self.c_attn.bias,
            self.c_proj.weight,
            self.c_proj.bias,
            past_key,
            past_value,
            attn_mask,
            self.num_heads,
            B,
        )

        attn_output = result[0].reshape(shape)
        new_key = result[1]   # [B, num_heads, total_len, head_dim]
        new_value = result[2]

        # Update the Cache object with the full key/value (overwrite previous)
        if past_key_values is not None:
            # We need to set the cache directly since our kernel already concatenated
            # Ensure the cache has enough layers
            while len(past_key_values.layers) <= self.layer_idx:
                past_key_values.layers.append(past_key_values.layer_class_to_replicate())
            layer = past_key_values.layers[self.layer_idx]
            if not layer.is_initialized:
                layer.lazy_initialization(new_key)
            layer.keys = new_key
            layer.values = new_value

        # Return format matching GPT2Attention: (attn_output, attn_weights)
        return attn_output, None


def _build_gpt2attn():
    """Compile the baseline GPT2Attention C++ kernel (torch::mm + at::softmax)."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building baseline GPT2Attention C++ kernel...")

    _ATTN_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>
#include <cmath>
#include <limits>

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
) {
    const int64_t M = x.size(0);
    const int64_t embed_dim = x.size(1);       // 768
    const int64_t head_dim = embed_dim / num_heads;  // 64
    const int64_t S = M / batch_size;

    // 1. QKV projection: qkv = x @ W_attn + b_attn -> [M, 2304]
    auto qkv = torch::mm(x, W_attn);
    qkv.add_(b_attn);

    // 2. Split Q, K, V each [M, 768], reshape to [B, 12, S, 64]
    auto qkv_split = qkv.chunk(3, /*dim=*/1);
    auto Q = qkv_split[0].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto K = qkv_split[1].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto V = qkv_split[2].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3});

    // 3. Concatenate past K, V along dim=2 if provided
    bool has_past = past_key.numel() > 0;
    if (has_past) {
        K = torch::cat({past_key, K}, /*dim=*/2);
        V = torch::cat({past_value, V}, /*dim=*/2);
    }

    auto present_key = K.clone();
    auto present_value = V.clone();

    const int64_t total_len = K.size(2);

    // 4. scores = Q @ K^T * (1/sqrt(64)) -> [B, 12, S, total_len]
    auto K_t = K.transpose(-2, -1).contiguous();
    auto scores = torch::matmul(Q.contiguous(), K_t);
    float scale = 1.0f / std::sqrt((float)head_dim);
    scores.mul_(scale);

    // 5. Build causal mask: for new positions only
    //    Position i in Q (query) corresponds to position (total_len - S + i) in the full sequence
    //    It can attend to positions 0...(total_len - S + i)
    {
        auto causal = torch::full({S, total_len}, -1e4, scores.options());
        for (int64_t i = 0; i < S; i++) {
            int64_t max_pos = total_len - S + i;
            for (int64_t j = 0; j <= max_pos && j < total_len; j++) {
                causal[i][j] = 0.0f;
            }
        }
        // Broadcast: [1, 1, S, total_len]
        scores.add_(causal.unsqueeze(0).unsqueeze(0));
    }

    // 6. Add HuggingFace attention_mask if provided
    if (attn_mask.numel() > 0) {
        scores.add_(attn_mask);
    }

    // 7. Softmax
    auto attn_weights = at::softmax(scores, /*dim=*/-1);

    // 8. context = attn_weights @ V -> [B, 12, S, 64]
    auto context = torch::matmul(attn_weights, V.contiguous());

    // 9. Merge heads -> [M, 768]
    auto merged = context.permute({0, 2, 1, 3}).contiguous().reshape({M, embed_dim});

    // 10. Output projection: output = merged @ W_proj + b_proj
    auto output = torch::mm(merged, W_proj);
    output.add_(b_proj);

    return {output, present_key, present_value};
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_attn_step", gpt2_attn_step);
}
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_attn"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_attn_step.cpp"
    cpp_path.write_text(_ATTN_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_attn",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_attn_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2attn(model):
    """Replace all transformer block attentions with GPT2AttentionStepWrapper."""
    compiled_kernel = _build_gpt2attn()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.attn = GPT2AttentionStepWrapper(block.attn, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2Attention blocks with baseline C++ kernel.")


def _build_gpt2attn_fused():
    """Compile the AVX-512 projection GPT2Attention C++ kernel."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building AVX-512 projection GPT2Attention C++ kernel...")

    _ATTN_FUSED_CPP_SOURCE = r"""
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
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_attn_fused"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_attn_fused_step.cpp"
    cpp_path.write_text(_ATTN_FUSED_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_attn_fused",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17", "-march=native", "-fopenmp", "-mavx512f", "-mfma"],
        extra_ldflags=["-fopenmp"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_attn_fused_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2attn_fused(model):
    """Replace all transformer block attentions with AVX-512 projection kernel."""
    compiled_kernel = _build_gpt2attn_fused()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.attn = GPT2AttentionStepWrapper(block.attn, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2Attention blocks with AVX-512 projection kernel.")


def _build_gpt2attn_flash():
    """Compile the full flash attention GPT2Attention C++ kernel."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building flash attention GPT2Attention C++ kernel...")

    _ATTN_FLASH_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>
#include <immintrin.h>
#include <cmath>
#include <omp.h>
#include <limits>
#include <cstring>

// ============================================================
// AVX-512 fast exp (ported from llama.cpp ggml_v_expf)
// ============================================================
static inline __m512 fast_exp_avx512(__m512 x) {
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b =
        _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                         _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __mmask16 d =
        _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                        _mm512_set1_ps(0x1.573e2ep-5f)),
                        u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                        _mm512_set1_ps(0x1.fffdb6p-2f))),
        u,
        _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
    const __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d))
        return res;
    const __m512 zero = _mm512_setzero_ps();
    const __m512 alt = _mm512_mask_blend_ps(
        _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
    return _mm512_mask_blend_ps(d, res, alt);
}

// ============================================================
// Tiled GEMV (from fused6)
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

// ============================================================
// Flash attention core: tiled online softmax per head
// ============================================================
static void flash_attention_head(
    const float* Q_row,    // [S, head_dim] row-major
    const float* K_all,    // [total_len, head_dim] row-major
    const float* V_all,    // [total_len, head_dim] row-major
    float* output,         // [S, head_dim]
    const float* mask_row, // [S, total_len] causal+hf mask combined, or nullptr
    int64_t S, int64_t total_len, int64_t head_dim, float scale
) {
    constexpr int KV_TILE = 64;

    for (int64_t q = 0; q < S; q++) {
        const float* q_vec = Q_row + q * head_dim;
        float* acc = output + q * head_dim;

        float M_val = -INFINITY;
        float S_val = 0.0f;
        std::memset(acc, 0, head_dim * sizeof(float));

        for (int64_t kv0 = 0; kv0 < total_len; kv0 += KV_TILE) {
            const int64_t kv_end = std::min(kv0 + (int64_t)KV_TILE, total_len);
            const int64_t kv_tile_len = kv_end - kv0;

            // Check if entire tile is masked
            if (mask_row) {
                bool all_masked = true;
                for (int64_t t = 0; t < kv_tile_len; t++) {
                    if (mask_row[q * total_len + kv0 + t] > -1e3f) {
                        all_masked = false;
                        break;
                    }
                }
                if (all_masked) continue;
            }

            // Compute Q @ K^T for this tile: scores[kv_tile_len]
            float scores[KV_TILE];
            for (int64_t t = 0; t < kv_tile_len; t++) {
                const float* k_vec = K_all + (kv0 + t) * head_dim;
                float dot = 0.0f;

                // AVX-512 dot product for head_dim=64 (4 vectors of 16)
                int64_t d = 0;
                __m512 sum_v = _mm512_setzero_ps();
                for (; d + 15 < head_dim; d += 16) {
                    __m512 qv = _mm512_loadu_ps(q_vec + d);
                    __m512 kv = _mm512_loadu_ps(k_vec + d);
                    sum_v = _mm512_fmadd_ps(qv, kv, sum_v);
                }
                dot = _mm512_reduce_add_ps(sum_v);
                for (; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }

                scores[t] = dot * scale;

                // Apply mask
                if (mask_row) {
                    scores[t] += mask_row[q * total_len + kv0 + t];
                }
            }

            // Online softmax update
            float tile_max = -INFINITY;
            for (int64_t t = 0; t < kv_tile_len; t++) {
                if (scores[t] > tile_max) tile_max = scores[t];
            }

            if (tile_max == -INFINITY) continue;

            float M_new = std::fmax(M_val, tile_max);

            // Rescale existing accumulator
            if (M_val > -INFINITY) {
                float rescale = std::exp(M_val - M_new);
                S_val *= rescale;
                // Rescale acc using AVX-512
                __m512 rs = _mm512_set1_ps(rescale);
                int64_t d = 0;
                for (; d + 15 < head_dim; d += 16) {
                    __m512 av = _mm512_loadu_ps(acc + d);
                    _mm512_storeu_ps(acc + d, _mm512_mul_ps(av, rs));
                }
                for (; d < head_dim; d++) acc[d] *= rescale;
            }
            M_val = M_new;

            // Compute exp(scores - M_new) using fast_exp_avx512
            float exp_scores[KV_TILE];
            {
                int64_t t = 0;
                for (; t + 15 < kv_tile_len; t += 16) {
                    __m512 sv = _mm512_loadu_ps(scores + t);
                    __m512 shifted = _mm512_sub_ps(sv, _mm512_set1_ps(M_new));
                    __m512 ev = fast_exp_avx512(shifted);
                    _mm512_storeu_ps(exp_scores + t, ev);
                }
                for (; t < kv_tile_len; t++) {
                    exp_scores[t] = std::exp(scores[t] - M_new);
                }
            }

            // Accumulate S
            for (int64_t t = 0; t < kv_tile_len; t++) {
                S_val += exp_scores[t];
            }

            // Accumulate: acc += exp_scores @ V_tile
            for (int64_t t = 0; t < kv_tile_len; t++) {
                if (exp_scores[t] == 0.0f) continue;
                const float* v_vec = V_all + (kv0 + t) * head_dim;
                __m512 ev = _mm512_set1_ps(exp_scores[t]);
                int64_t d = 0;
                for (; d + 15 < head_dim; d += 16) {
                    __m512 av = _mm512_loadu_ps(acc + d);
                    __m512 vv = _mm512_loadu_ps(v_vec + d);
                    _mm512_storeu_ps(acc + d, _mm512_fmadd_ps(ev, vv, av));
                }
                for (; d < head_dim; d++) {
                    acc[d] += exp_scores[t] * v_vec[d];
                }
            }
        }

        // Normalize: acc /= S
        if (S_val > 0.0f) {
            float inv_s = 1.0f / S_val;
            __m512 inv_v = _mm512_set1_ps(inv_s);
            int64_t d = 0;
            for (; d + 15 < head_dim; d += 16) {
                __m512 av = _mm512_loadu_ps(acc + d);
                _mm512_storeu_ps(acc + d, _mm512_mul_ps(av, inv_v));
            }
            for (; d < head_dim; d++) acc[d] *= inv_s;
        }
    }
}

std::vector<torch::Tensor> gpt2_attn_flash_step(
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

    // 2. Split Q, K, V and reshape to [B, num_heads, S, head_dim]
    auto qkv_split = qkv.chunk(3, 1);
    auto Q = qkv_split[0].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous();
    auto K = qkv_split[1].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous();
    auto V = qkv_split[2].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous();

    // 3. Concatenate past KV
    bool has_past = past_key.numel() > 0;
    if (has_past) {
        K = torch::cat({past_key, K}, 2).contiguous();
        V = torch::cat({past_value, V}, 2).contiguous();
    }

    auto present_key = K.clone();
    auto present_value = V.clone();

    const int64_t total_len = K.size(2);
    float scale = 1.0f / std::sqrt((float)head_dim);

    // 4. Build combined causal + HF mask: [B, num_heads, S, total_len]
    //    (we flatten to per-query-row for the flash kernel)
    auto combined_mask = torch::full({S, total_len}, -1e4f, x.options());
    for (int64_t i = 0; i < S; i++) {
        int64_t max_pos = total_len - S + i;
        for (int64_t j = 0; j <= max_pos && j < total_len; j++) {
            combined_mask[i][j] = 0.0f;
        }
    }
    float* causal_ptr = combined_mask.data_ptr<float>();

    // If HF mask provided, we need per-batch masks
    bool has_hf_mask = attn_mask.numel() > 0;
    // For simplicity, expand HF mask per batch
    // HF mask shape: [B, 1, 1, total_len]
    // We'll handle it inside the parallel loop

    // 5. Flash attention: parallel over B * num_heads
    auto context = torch::empty({batch_size, num_heads, S, head_dim}, x.options());

    #pragma omp parallel for schedule(dynamic)
    for (int64_t bh = 0; bh < batch_size * num_heads; bh++) {
        int64_t b = bh / num_heads;
        int64_t h = bh % num_heads;

        const float* Q_ptr = Q.data_ptr<float>() + (b * num_heads + h) * S * head_dim;
        const float* K_ptr = K.data_ptr<float>() + (b * num_heads + h) * total_len * head_dim;
        const float* V_ptr = V.data_ptr<float>() + (b * num_heads + h) * total_len * head_dim;
        float* out_ptr = context.data_ptr<float>() + (b * num_heads + h) * S * head_dim;

        // Build per-head mask combining causal + HF
        // Thread-local mask buffer
        std::vector<float> mask_buf(S * total_len);
        std::memcpy(mask_buf.data(), causal_ptr, S * total_len * sizeof(float));

        if (has_hf_mask) {
            // HF mask: [B, 1, 1, total_len] — broadcast to [S, total_len]
            const float* hf_ptr = attn_mask.data_ptr<float>() + b * total_len;
            for (int64_t i = 0; i < S; i++) {
                for (int64_t j = 0; j < total_len; j++) {
                    mask_buf[i * total_len + j] += hf_ptr[j];
                }
            }
        }

        flash_attention_head(Q_ptr, K_ptr, V_ptr, out_ptr, mask_buf.data(),
                            S, total_len, head_dim, scale);
    }

    // 6. Merge heads: [B, num_heads, S, head_dim] -> [M, embed_dim]
    auto merged = context.permute({0, 2, 1, 3}).contiguous().reshape({M, embed_dim});

    // 7. Output projection with AVX-512
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
    m.def("gpt2_attn_flash_step", gpt2_attn_flash_step);
}
"""
    cache_dir = _pl.Path.home() / ".cache" / "mocha" / "gpt2_attn_flash"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = cache_dir / "gpt2_attn_flash_step.cpp"
    cpp_path.write_text(_ATTN_FLASH_CPP_SOURCE)

    torch.utils.cpp_extension.load(
        name="step_gpt2_attn_flash",
        sources=[str(cpp_path)],
        extra_cflags=["-O3", "-std=c++17", "-march=native", "-fopenmp", "-mavx512f", "-mfma"],
        extra_ldflags=["-fopenmp"],
        build_directory=str(cache_dir),
        verbose=False,
        is_python_module=False,
    )

    op_fn = torch.ops.step_ops.gpt2_attn_flash_step

    def wrapper(*args):
        return op_fn(*args)

    return wrapper


def _apply_gpt2attn_flash(model):
    """Replace all transformer block attentions with flash attention kernel."""
    compiled_kernel = _build_gpt2attn_flash()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        block.attn = GPT2AttentionStepWrapper(block.attn, compiled_kernel)
    print(f"Replaced {len(blocks)} GPT2Attention blocks with flash attention kernel.")


_REPLACEMENT_REGISTRY = {
    "gpt2mlp": _apply_gpt2mlp,              # hand-written scalar MLP (mm + manual loops)
    "gpt2mlp_fused": _apply_gpt2mlp_fused,  # torch::mm matmuls + separate add_ & at::gelu calls
    "gpt2mlp_loopfused": _apply_gpt2mlp_loopfused,  # fully loop-fused scalar (no intermediate tensor)
    "gpt2mlp_fused2": _apply_gpt2mlp_fused2,  # cblas_sgemm matmuls + at::gelu (slower than fused)
    "gpt2mlp_fused3": _apply_gpt2mlp_fused3,  # torch::mm matmuls + single-pass fused bias+GELU loop
    "gpt2mlp_fused4": _apply_gpt2mlp_fused4,  # AVX512+OpenMP hand-written GEMV for M=1 decode
    "gpt2mlp_fused5": _apply_gpt2mlp_fused5,  # tiled GEMV + Pade-tanh GELU + torch::mm
    "gpt2mlp_fused6": _apply_gpt2mlp_fused6,  # tiled GEMM + Pade-tanh GELU + avx512_omp_gemm
    "gpt2attn":       _apply_gpt2attn,        # baseline: torch::mm + at::softmax
    "gpt2attn_fused": _apply_gpt2attn_fused,  # AVX-512 projections + torch softmax
    "gpt2attn_flash": _apply_gpt2attn_flash,  # full tiled flash attention
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
            "Supported: gpt2mlp, gpt2mlp_fused, gpt2mlp_loopfused, gpt2mlp_fused2. "
            "Example: --replace gpt2mlp_fused2"
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
