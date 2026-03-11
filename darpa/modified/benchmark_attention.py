#!/usr/bin/env python3
"""
Benchmark the 'generate' stage across attention kernel variants.

Runs causal_language_modeling.py inference N times per variant and reports
the generate-stage wall-clock time for each run plus the average.

Usage:
  python3 benchmark_attention.py                          # defaults: 3 runs, all variants, short prompt
  python3 benchmark_attention.py -n 5                     # 5 runs per variant
  python3 benchmark_attention.py --prompt-length long      # ~512 token prompt
  python3 benchmark_attention.py --sweep                   # run all prompt lengths (short/medium/long)
  python3 benchmark_attention.py --variants gpt2attn gpt2attn_fused   # subset of variants
  python3 benchmark_attention.py --prompt "Custom prompt"  # custom prompt (overrides --prompt-length)

Docker example:
  source /home/dockeruser/mochaenv/bin/activate
  cd /home/dockeruser/step_cpu/darpa/modified
  python3 benchmark_attention.py --sweep -n 3
"""

import argparse
import re
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLM_SCRIPT = os.path.join(SCRIPT_DIR, "causal_language_modeling.py")

DEFAULT_VARIANTS = ["gpt2attn", "gpt2attn_fused", "gpt2attn_flash"]

# Preset prompts at different lengths (approximate token counts for distilgpt2 tokenizer)
PRESET_PROMPTS = {
    "short": (
        "Why is the sky blue?",
        "~6 tokens",
    ),
    "medium": (
        "The history of artificial intelligence began in antiquity, with myths, stories "
        "and rumors of artificial beings endowed with intelligence or consciousness by "
        "master craftsmen. The seeds of modern AI were planted by philosophers who "
        "attempted to describe the process of human thinking as the mechanical "
        "manipulation of symbols. This work culminated in the invention of the "
        "programmable digital computer in the 1940s, a machine based on the abstract "
        "essence of mathematical reasoning. This device and the ideas behind it inspired "
        "a handful of scientists to begin seriously discussing the possibility of building "
        "an electronic brain. The field of AI research was founded at a workshop held on "
        "the campus of Dartmouth College during the summer of 1956. Those who attended "
        "would become the leaders of AI research for decades. Many of them predicted that "
        "a machine as intelligent as a human being would exist in no more than a "
        "generation, and they were given millions of dollars to make this vision come "
        "true. Eventually, it became obvious that commercial developers and researchers "
        "had grossly underestimated the difficulty of the project.",
        "~196 tokens",
    ),
    "long": (
        "The history of artificial intelligence began in antiquity, with myths, stories "
        "and rumors of artificial beings endowed with intelligence or consciousness by "
        "master craftsmen. The seeds of modern AI were planted by philosophers who "
        "attempted to describe the process of human thinking as the mechanical "
        "manipulation of symbols. This work culminated in the invention of the "
        "programmable digital computer in the 1940s, a machine based on the abstract "
        "essence of mathematical reasoning. This device and the ideas behind it inspired "
        "a handful of scientists to begin seriously discussing the possibility of building "
        "an electronic brain. The field of AI research was founded at a workshop held on "
        "the campus of Dartmouth College during the summer of 1956. Those who attended "
        "would become the leaders of AI research for decades. Many of them predicted that "
        "a machine as intelligent as a human being would exist in no more than a "
        "generation, and they were given millions of dollars to make this vision come "
        "true. Eventually, it became obvious that commercial developers and researchers "
        "had grossly underestimated the difficulty of the project. In the 1970s, AI was "
        "subjected to critiques and financial setbacks. AI researchers had failed to "
        "appreciate the difficulty of the problems they faced. Their tremendous optimism "
        "had raised expectations impossibly high, and when the promised results failed to "
        "materialize, funding for AI disappeared. At the same time, the connectionism "
        "movement achieved little success with simple neural network architectures. In "
        "the 1980s, expert systems became commercially successful and knowledge-based "
        "approaches gained momentum. The Japanese Fifth Generation Computer project "
        "spurred renewed investment worldwide. However, the market for specialized AI "
        "hardware collapsed in 1987, beginning the second AI winter. Research continued "
        "quietly through the 1990s, with advances in machine learning, intelligent "
        "agents, and statistical approaches replacing earlier symbolic methods. Deep "
        "learning emerged in the 2000s as a breakthrough approach, enabled by larger "
        "datasets and faster hardware. Convolutional neural networks revolutionized "
        "computer vision, while recurrent networks transformed natural language "
        "processing. The ImageNet competition in 2012 marked a turning point when deep "
        "learning dramatically outperformed traditional methods. Tech companies invested "
        "billions in AI research labs. Reinforcement learning achieved superhuman "
        "performance in games like Go and chess. Generative adversarial networks created "
        "realistic synthetic images. Transfer learning and pre-trained language models "
        "like BERT and GPT demonstrated that large neural networks trained on vast corpora "
        "could be fine-tuned for diverse downstream tasks with remarkable effectiveness "
        "across many benchmarks and application domains throughout the research community.",
        "~512 tokens",
    ),
}


def run_once(variant: str, prompt: str) -> float | None:
    """Run inference with the given variant and return the generate-stage time in seconds."""
    cmd = [
        sys.executable, CLM_SCRIPT,
        "--mode", "infer",
        "--prompt", prompt,
        "--cpu-only",
        "--replace", variant,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after 600s", file=sys.stderr)
        return None

    # Parse the benchmark table for the generate row.
    # The table row looks like: | generate  |  0.53 | 35.5  | 51.1  |
    # or plain text:            generate    0.53   35.5   51.1
    for line in result.stdout.splitlines():
        stripped = line.strip().strip("|")
        parts = [p.strip() for p in re.split(r"\s{2,}|\|", stripped) if p.strip()]
        if parts and parts[0] == "generate":
            try:
                return float(parts[1])
            except (IndexError, ValueError):
                pass

    # Fallback: look for "Total Runtime" line (used when --no-instrument is set)
    for line in result.stdout.splitlines():
        m = re.search(r"Total Runtime.*?([\d.]+)\s*seconds", line)
        if m:
            return float(m.group(1))

    print(f"    WARNING: could not parse generate time from output", file=sys.stderr)
    if result.stderr:
        for line in result.stderr.strip().splitlines()[-5:]:
            print(f"    stderr: {line}", file=sys.stderr)
    return None


def run_benchmark(variants, prompt, prompt_label, n_runs):
    """Run all variants with the given prompt and return results dict."""
    print(f"\n{'#'*60}")
    print(f"  Prompt: {prompt_label}")
    print(f"{'#'*60}")

    results: dict[str, list[float]] = {}
    for variant in variants:
        print(f"\n  --- {variant} ---")
        times = []
        for i in range(1, n_runs + 1):
            print(f"  Run {i}/{n_runs} ... ", end="", flush=True)
            t = run_once(variant, prompt)
            if t is not None:
                times.append(t)
                print(f"{t:.2f}s")
            else:
                print("FAILED")
        results[variant] = times
    return results


def print_summary(variants, results, n_runs, prompt_label=""):
    """Print a formatted summary table."""
    title = "SUMMARY: generate-stage time (seconds)"
    if prompt_label:
        title += f"  [{prompt_label}]"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

    header = f"  {'Variant':<20}"
    for i in range(1, n_runs + 1):
        header += f" {'Run '+str(i):>8}"
    header += f" {'Average':>10}"
    print(header)
    print(f"  {'─'*20}" + f" {'─'*8}" * n_runs + f" {'─'*10}")

    averages = {}
    for variant in variants:
        times = results.get(variant, [])
        row = f"  {variant:<20}"
        for t in times:
            row += f" {t:8.2f}"
        for _ in range(n_runs - len(times)):
            row += f" {'N/A':>8}"
        if times:
            avg = sum(times) / len(times)
            row += f" {avg:10.2f}"
            averages[variant] = avg
        else:
            row += f" {'N/A':>10}"
        print(row)

    print()
    return averages


def print_sweep_summary(variants, all_averages):
    """Print a cross-prompt-length comparison table."""
    print(f"\n{'='*60}")
    print(f"  CROSS-LENGTH COMPARISON (average generate time, seconds)")
    print(f"{'='*60}\n")

    lengths = list(all_averages.keys())
    header = f"  {'Variant':<20}"
    for length in lengths:
        header += f" {length:>12}"
    print(header)
    print(f"  {'─'*20}" + f" {'─'*12}" * len(lengths))

    for variant in variants:
        row = f"  {variant:<20}"
        for length in lengths:
            avg = all_averages[length].get(variant)
            if avg is not None:
                row += f" {avg:12.2f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Speedup vs baseline
    baseline = variants[0] if variants else None
    if baseline and len(variants) > 1:
        print(f"\n  Speedup vs {baseline}:")
        print(f"  {'Variant':<20}", end="")
        for length in lengths:
            print(f" {length:>12}", end="")
        print()
        print(f"  {'─'*20}" + f" {'─'*12}" * len(lengths))
        for variant in variants[1:]:
            row = f"  {variant:<20}"
            for length in lengths:
                base_avg = all_averages[length].get(baseline)
                var_avg = all_averages[length].get(variant)
                if base_avg and var_avg:
                    speedup = base_avg / var_avg
                    row += f" {speedup:11.2f}x"
                else:
                    row += f" {'N/A':>12}"
            print(row)

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention kernel variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-n", "--runs", type=int, default=3,
                        help="Number of runs per variant (default: 3)")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        help=f"Variants to benchmark (default: {DEFAULT_VARIANTS})")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt (overrides --prompt-length)")
    parser.add_argument("--prompt-length", choices=["short", "medium", "long"],
                        default="short",
                        help="Preset prompt length (default: short)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run all preset prompt lengths and compare")
    args = parser.parse_args()

    if args.sweep:
        all_averages = {}
        for length_name in ["short", "medium", "long"]:
            prompt, label = PRESET_PROMPTS[length_name]
            results = run_benchmark(args.variants, prompt, f"{length_name} ({label})", args.runs)
            averages = print_summary(args.variants, results, args.runs,
                                     f"{length_name} ({label})")
            all_averages[length_name] = averages
        print_sweep_summary(args.variants, all_averages)
    else:
        if args.prompt is not None:
            prompt = args.prompt
            prompt_label = "custom"
        else:
            prompt, token_info = PRESET_PROMPTS[args.prompt_length]
            prompt_label = f"{args.prompt_length} ({token_info})"
        results = run_benchmark(args.variants, prompt, prompt_label, args.runs)
        print_summary(args.variants, results, args.runs, prompt_label)


if __name__ == "__main__":
    main()
