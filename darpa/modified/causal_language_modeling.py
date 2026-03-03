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


_REPLACEMENT_REGISTRY = {
    "gpt2mlp": _apply_gpt2mlp,
    "gpt2mlp_fused": _apply_gpt2mlp_fused,
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
            "Supported: gpt2mlp. Example: --replace gpt2mlp"
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
