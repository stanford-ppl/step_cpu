  
    Script: darpa/modified/benchmark_attention.py — runs inference N times per variant and reports generate-stage times.

  Usage:
  python3 benchmark_attention.py              # 3 runs, all 3 variants
  python3 benchmark_attention.py -n 5         # 5 runs per variant
  python3 benchmark_attention.py --variants gpt2attn gpt2attn_flash  # subset

  Benchmark Results (3 runs each):
```bash
┌─────────────────────┬───────┬───────┬───────┬─────────┐
│       Variant       │ Run 1 │ Run 2 │ Run 3 │ Average │
├─────────────────────┼───────┼───────┼───────┼─────────┤
│ gpt2attn (baseline) │ 1.25s │ 0.91s │ 1.07s │ 1.08s   │
├─────────────────────┼───────┼───────┼───────┼─────────┤
│ gpt2attn_fused      │ 0.86s │ 0.73s │ 0.65s │ 0.75s   │
├─────────────────────┼───────┼───────┼───────┼─────────┤
│ gpt2attn_flash      │ 0.89s │ 0.97s │ 0.79s │ 0.88s   │
└─────────────────────┴───────┴───────┴───────┴─────────┘
```
  Key takeaway: The fused variant is fastest at 0.75s average (~30% faster than baseline). Flash comes in at 0.88s (~19% faster). Both optimized kernels
  meaningfully improve generate-stage latency over the baseline.

  What each variant does differently (attention core only — projections are identical):
```bash
  ┌──────────────────┬──────────────────────────────────┬─────────────────────────────────────────────────────┐
  │                  │          gpt2attn_fused          │                   gpt2attn_flash                    │
  ├──────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Attention matmul │ torch::matmul (optimized BLAS)   │ Custom tiled loop, KV_TILE=64                       │
  ├──────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Softmax          │ at::softmax (optimized library)  │ Custom online softmax per tile with fast_exp_avx512 │
  ├──────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Memory           │ Full [B,12,S,S] attention matrix │ Per-query accumulators only                         │
  └──────────────────┴──────────────────────────────────┴─────────────────────────────────────────────────────┘
```
  Why flash is slower here: The benchmark prompt "Why is the sky blue?" is ~7 tokens. At S≈7 with 12 heads, the full attention matrix is only 12 × 7 × 7 =
  588 floats (~2.3 KB) — it fits entirely in L1 cache. At this tiny size:

  1. PyTorch's matmul + softmax (used by fused) dispatch to highly optimized BLAS/MKL routines that handle small matrices very efficiently in a single call.
  2. Flash attention's tiled loop has per-tile bookkeeping overhead (running max, rescaling accumulators with exp(M_old - M_new), tile iteration logic) that
  costs more than it saves when there's no memory pressure to begin with.
  3. The fast_exp_avx512 approximation and per-query-position C++ loops add dispatch overhead that exceeds the cost of a single library softmax call on a
  7-element vector.

  Flash attention's advantage emerges at longer sequences (S=512, 1024+), where the O(S²) attention matrix no longer fits in cache and the memory savings
  from online softmax become significant.

  To verify this hypothesis, you could re-run the benchmark with a longer prompt (e.g., a few hundred tokens). Want me to do that