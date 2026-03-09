Content of `causal_language_modeling.py`

# Changes made to `causal_language_modeling.py`

1. parse_clm() (lines 336–347) — Added --mode (choices: train/infer, default: train) and --prompt
(string, optional) arguments.
2. main() (lines 388–477) — Inserted an inference branch that fires before dataset loading when
args.mode == "infer":
- Validates --prompt is present, prints a clear error and returns early if not.
- Runs 4 timed stages: create_tokenizer → load_model → tokenize_prompt → generate.
- Prints the generated text, then the same benchmark summary table (or total runtime if
--no-instrument) that the training path uses.
- Returns BenchmarkResult(perplexity=None, elapsed_time=total_elapsed).
- The training path is unchanged — it runs when args.mode == "train" (the default) by falling
through the if args.mode == "infer": block.

# Usage:
## Training (unchanged)
python3 darpa/modified/causal_language_modeling.py --cpu-only --samples 50

## Inference
python3 darpa/modified/causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only

## Inference without instrumentation
python3 darpa/modified/causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only --no-instrument

## Missing prompt — prints error and exits cleanly
python3 darpa/modified/causal_language_modeling.py --mode infer --cpu-only
