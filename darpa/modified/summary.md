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


# Results

```
((.venv) ) (py312clean) ginasohn@lagos:~/research/mocha/darpa/modified$ python3 causal_language_modeling.py --cpu-only --samples 256 --no-instrument

=== System resources ===
Available CPUs : 144
Running in CPU-only mode
==================================================

Loading weights: 100%|████████████| 76/76 [00:00<00:00, 2557.63it/s, Materializing param=transformer.wte.weight]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 50256}.
  0%|                                                                                   | 0/264 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
{'eval_loss': '3.869', 'eval_runtime': '1.036', 'eval_samples_per_second': '41.52', 'eval_steps_per_second': '5.793', 'epoch': '1'}                                                                                             
{'eval_loss': '3.858', 'eval_runtime': '1.034', 'eval_samples_per_second': '41.58', 'eval_steps_per_second': '5.801', 'epoch': '2'}                                                                                             
{'eval_loss': '3.858', 'eval_runtime': '1.049', 'eval_samples_per_second': '41', 'eval_steps_per_second': '5.721', 'epoch': '3'}                                                                                                
{'train_runtime': '294.8', 'train_samples_per_second': '7.143', 'train_steps_per_second': '0.895', 'train_loss': '3.888', 'epoch': '3'}                                                                                         
100%|█████████████████████████████████████████████████████████████████████████| 264/264 [04:54<00:00,  1.12s/it]
100%|█████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00,  7.27it/s]

=== Final Perplexity: 47.38 ===


=== Total Runtime ===
299.43 seconds
```