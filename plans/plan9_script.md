1. Add a script to run the baseline and the replaced version (gpt2mlp_fused6, gpt2attn_flash) for N number of times (make N configurable in the command line or in the script) and report the summary and average of the time for running the 'generate' stage. After creating a script, add a documentation on how to use it.

As shown in the `main` function in /home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py, the instance of `StageTimer` class in `records` contains the time to run the 'generate' stage.
A printed benchmark summary will look like below. When I refer to the time for running the 'generate' stage, I mean the "Time (s)" column in the table below.

```
=== Benchmark Summary (CPU-only mode) ===

| Stage              |   Time (s) | CPU %   | Cores Used   |
|--------------------|------------|---------|--------------|
| create_tokenizer   |       0.52 | 0.6     | 0.9          |
| load_model         |       0.13 | 0.5     | 0.6          |
| apply_replacements |      32.72 | 1.2     | 1.7          |
| tokenize_prompt    |       0.01 | 0.0     | 0.0          |
| generate           |       0.53 | 35.5    | 51.1         |
| TOTAL              |      33.91 | -       | -            |
==================================================
```

2. Use this script to compare the time to run the "generate" stage for each three attention variants. Run 3 times each and report the summary and average.

To run tests, read below:
I implement it in a docker container. You can enter it using
```bash
docker exec -it mocha-bg bash
```
Edits on the host in ./step_cpu show up immediately in the container.
Edits in the container under /home/ginasohn/step_cpu write back to the host

In the container you can run the tests using:
```bash
# In the container
source /home/dockeruser/mochaenv/bin/activate
cd /home/dockeruser/step_cpu
python3 darpa/modified/causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only --replace gpt2mlp_fused
```