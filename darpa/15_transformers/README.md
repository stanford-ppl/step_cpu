```
# uses gpu(s) if present
python3 causal_language_modeling.py --samples 4096
# verify instrumentation did not add excessive overhead
python3 causal_language_modeling.py --samples 4096 --no-instrument

# force a run on the CPU
python3 causal_language_modeling.py --cpu-only --samples 256
# verify instrumentation did not add excessive overhead
python3 causal_language_modeling.py --cpu-only --samples 256 --no-instrument
```

## Results
```
python3 causal_language_modeling.py --samples 4096

=== Final Perplexity: 43.53 ===


=== Benchmark Summary (GPU memory shows PEAK) ===

| Stage            |   Time (s) | CPU %   | Cores Used   | GPU 0 %   | GPU 0 Peak (MiB)   |
|------------------|------------|---------|--------------|-----------|--------------------|
| load_dataset     |       1.37 | 1.0     | 0.2          | 0.0       | 0                  |
| create_tokenizer |       0.22 | 1.6     | 0.4          | 0.0       | 0                  |
| tokenise         |       0.03 | 0.0     | 0.0          | 0.0       | 0                  |
| blockify         |       0.02 | 0.0     | 0.0          | 0.0       | 0                  |
| load_model       |       0.49 | 3.8     | 0.9          | 0.0       | 318                |
| create_collator  |       0    | -       | -            | -         | 318                |
| training_args    |       0.02 | 0.0     | 0.0          | 0.0       | 318                |
| create_trainer   |       0.01 | 0.0     | 0.0          | 0.0       | 318                |
| train            |     183.11 | 5.2     | 1.2          | 96.4      | 2104               |
| evaluate         |       1.84 | 4.6     | 1.1          | 87.5      | 1391               |
| TOTAL            |     187.11 | -       | -            | -         | -                  |

==================================================


python3 causal_language_modeling.py --samples 4096 --no-instrument
=== Final Perplexity: 43.53 ===


=== Total Runtime ===
186.47 seconds

# force a run on the CPU
python3 causal_language_modeling.py --cpu-only --samples 256
=== Final Perplexity: 47.38 ===


=== Benchmark Summary (CPU-only mode) ===

| Stage            |   Time (s) | CPU %   | Cores Used   |
|------------------|------------|---------|--------------|
| load_dataset     |       1.21 | 2.2     | 0.5          |
| create_tokenizer |       0.19 | 0.4     | 0.1          |
| tokenise         |       0.03 | 0.0     | 0.0          |
| blockify         |       0.02 | 0.0     | 0.0          |
| load_model       |       0.37 | 0.8     | 0.2          |
| create_collator  |       0    | -       | -            |
| training_args    |       0    | 0.0     | 0.0          |
| create_trainer   |       0.06 | 0.0     | 0.0          |
| train            |     297.88 | 94.5    | 22.7         |
| evaluate         |       1.06 | 100.0   | 24.0         |
| TOTAL            |     300.83 | -       | -            |

==================================================

python3 causal_language_modeling.py --cpu-only --samples 256 --no-instrument
=== Final Perplexity: 47.38 ===


=== Total Runtime ===
275.74 seconds
```
