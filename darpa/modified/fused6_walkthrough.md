# Fused6: Fully Hand-Written AVX512 GPT2 MLP Execution Walkthrough

## 1. Overview

The GPT2 MLP computes: `fc_up -> GELU -> fc_down` with dimensions **768 -> 3072 -> 768**.

Fused6 replaces this entirely with hand-written AVX512 C++ kernels for **both** M=1 (decode/GEMV) and M>1 (prefill/GEMM) paths. There is no `torch::mm` or MKL fallback — every matrix multiply is a custom micro-kernel.

**Python wrapper** (`GPT2MLPStepWrapper`, lines 321-357): reshapes the input from `[batch, seq, 768]` to `[M, 768]` (where M = batch * seq), calls `gpt2_mlp_fused6_step`, then reshapes back. In training mode, falls through to the original HuggingFace forward.

**C++ entry point** (`gpt2_mlp_fused6_step`, lines 1544-1648): branches on `M == 1` to select the GEMV or GEMM path.

---

## 2. Two Execution Paths

| Condition | Path | GEMM1 | GELU | GEMM2 |
|-----------|------|-------|------|-------|
| M = 1 (decode) | GEMV | `avx512_omp_gemv` | Inline vectorized | `avx512_omp_gemv` |
| M > 1 (prefill) | GEMM | `avx512_omp_gemm` | OpenMP parallel vectorized | `avx512_omp_gemm` |

Both paths fuse bias into the matmul accumulators (no separate add pass). Both use the same Pade-tanh GELU implementation.

---

## 3. M=1 GEMV Path (lines 1554-1598)

### Execution flow

```
x[1,768] --GEMV1--> h[1,3072] --GELU--> h[1,3072] --GEMV2--> out[1,768]
```

1. **GEMV1**: `avx512_omp_gemv(x, W_fc, b_fc, h, D=768, K=3072)` — computes `h = x * W_fc + b_fc`
2. **GELU**: Inline loop over 3072 floats (no OpenMP — single row fits in L1)
3. **GEMV2**: `avx512_omp_gemv(h, W_proj, b_proj, out, K=3072, D=768)` — computes `out = h * W_proj + b_proj`

---

## 4. GEMV Tiled Micro-Kernel (`gemv_tiled_chunk`, lines 1387-1445)

Computes a chunk of the output vector `y[n_start..n_end)` for a single input row.

### 64-float main tiles (RN=4 x 16-wide)

```
for each tile of 64 output elements:
    acc0..acc3 = load bias[n..n+64)          // 4 zmm registers initialized with bias
    for k in 0..K:
        xk = broadcast x[k]                  // 1 scalar broadcast
        prefetch W[(k+2)*N + n .. n+64)       // 4 cache-line prefetches, 2 iterations ahead
        acc0 += xk * W[k*N + n]              // FMA
        acc1 += xk * W[k*N + n+16]
        acc2 += xk * W[k*N + n+32]
        acc3 += xk * W[k*N + n+48]
    store acc0..acc3 -> y[n..n+64)
```

Each tile uses **4 accumulator zmm registers** + 1 broadcast register = **5 zmm registers**.

### 16-float remainder

Single-register accumulator loop for leftover chunks not covered by the 64-float tiles.

### Scalar tail

Element-by-element fallback for any remaining elements (< 16).

---

## 5. OpenMP GEMV Work Distribution (`avx512_omp_gemv`, lines 1448-1471)

Partitions the **N dimension** (output columns) across threads:

- Base chunk size = `(N / nthreads) & ~15` — rounded down to 16-element alignment
- Remainder columns distributed as extra 16-element slices to the first few threads
- Each thread calls `gemv_tiled_chunk` on its `[n_start, n_end)` range

For GPT2 dimensions: GEMV1 partitions N=3072, GEMV2 partitions N=768. Both are divisible by 16.

---

## 6. M>1 GEMM Path (lines 1601-1648)

### Execution flow

```
x[M,768] --GEMM1--> h[M,3072] --GELU--> h[M,3072] --GEMM2--> out[M,768]
```

1. **GEMM1**: `avx512_omp_gemm(x, W_fc, b_fc, h, M, D=768, K=3072)` — bias fused
2. **GELU**: OpenMP parallel for over M rows, AVX512 vectorized within each row
3. **GEMM2**: `avx512_omp_gemm(h, W_proj, b_proj, out, M, K=3072, D=768)` — bias fused

This is the key difference from fused5: **no `torch::mm` / MKL fallback**. The entire M>1 path uses the custom `gemm_ukernel`.

---

## 7. GEMM Register-Blocked Micro-Kernel (`gemm_ukernel<RM,RN>`, lines 1477-1504)

Template parameters: **RM** (rows processed) x **RN** (AVX512 vectors per row). Default instantiation: **RM=4, RN=4** = 4 rows x 64 output floats per tile.

### Execution (default 4x4 tile)

```
// Initialize: RM*RN = 16 accumulator registers from bias (or zero)
for i in 0..RM:
    for r in 0..RN:
        acc[i][r] = load bias[n_offset + r*16]

// K-loop
for k in 0..K:
    // Load RN=4 weight vectors (shared across all RM rows)
    Bv[0..3] = load B[k*N + n_offset .. +64)

    // Prefetch weights 2 iterations ahead
    prefetch B[(k+2)*N + n_offset .. +64)       // RN=4 prefetches

    // For each of RM=4 rows, broadcast A[i,k] and FMA
    for i in 0..RM:
        a = broadcast A[i*K + k]
        for r in 0..RN:
            acc[i][r] = fma(a, Bv[r], acc[i][r])

// Store RM*RN = 16 result tiles
for i in 0..RM:
    for r in 0..RN:
        store acc[i][r] -> C[i*N + n_offset + r*16]
```

### Register budget (4x4 tile)

| Purpose | Count | Registers |
|---------|-------|-----------|
| Accumulators `acc[4][4]` | 16 | zmm0-zmm15 |
| Weight vectors `Bv[4]` | 4 | zmm16-zmm19 |
| Broadcast temp `a` | 1 | zmm20 |
| **Total** | **21** | of 32 available |

The 4x4 tile maximizes accumulator count to hide FMA latency (4-5 cycles on modern x86) while staying within the 32-register budget.

### Template instantiations used

| Shape | Where | Purpose |
|-------|-------|---------|
| `<4, 4>` | Main body | Full 4-row x 64-float tiles |
| `<4, 1>` | N remainder | 4-row x 16-float for leftover columns |
| `<1, 4>` | M tail | 1-row x 64-float for leftover rows |
| `<1, 1>` | Corner | 1-row x 16-float for both remainders |

---

## 8. OpenMP GEMM Work Distribution (`avx512_omp_gemm`, lines 1507-1542)

Partitions the **M dimension** (rows) across threads:

- Base chunk size = `(M / nthreads / RM) * RM` — rounded down to RM=4 alignment
- Extra rows distributed as RM-sized slices; last thread gets any final remainder
- Each thread processes its row range independently

### Per-thread loop structure

```
for ii in [m_start, m_end) step RM=4:      // full RM-row blocks
    for jj in [0, N) step RN*VL=64:         //   64-wide tiles -> gemm_ukernel<4,4>
    for jj in [last_64, N) step VL=16:      //   16-wide remainder -> gemm_ukernel<4,1>

for ii in [last_RM, m_end) step 1:          // row tail (< RM rows)
    for jj in [0, N) step RN*VL=64:         //   64-wide tiles -> gemm_ukernel<1,4>
    for jj in [last_64, N) step VL=16:      //   16-wide remainder -> gemm_ukernel<1,1>
```

For GPT2: N=3072 is divisible by 64 (no remainder), N=768 is divisible by 64 (no remainder). Row tail only matters when M is not divisible by 4.

---

## 9. GELU Activation (lines 1615-1641)

Separate pass (not fused into GEMM output), but bias was already applied during the GEMM.

### Formula

```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

### Vectorized implementation

- **M>1 path**: `#pragma omp parallel for` over M rows
- **M=1 path**: Single-threaded inline loop (one row, 3072 elements)
- **Inner loop**: 16-wide AVX512 — compute `v^3`, FMA for inner term, multiply by `sqrt(2/pi)`, call `fast_tanh_avx512`, final multiply
- **Scalar tail**: Standard `std::tanh` for remaining elements (< 16)

---

## 10. Fast Tanh: Pade [7,6] Rational Approximation (lines 1357-1381)

Avoids calling `exp()` entirely. Uses a [7,6] Pade rational form:

```
tanh(x) ~ x * P(x^2) / Q(x^2)

P(x2) = 135135 + 17325*x2 + 378*x4 + x6
Q(x2) = 135135 + 62370*x2 + 3150*x4 + 28*x6
```

- Input clamped to [-4.97, 4.97] (tanh saturates beyond this)
- Computes `x2 = x*x`, `x4 = x2*x2`, `x6 = x4*x2`
- Numerator and denominator built with FMA chains
- Final: `x_clamped * (num / den)`
- Uses `_mm512_div_ps` (one division per 16 floats)

---

## 11. Optimizations Catalog

| Optimization | Where | Effect |
|-------------|-------|--------|
| **Pade [7,6] rational tanh** | `fast_tanh_avx512` | Eliminates `exp()` from GELU; pure arithmetic |
| **Software prefetch** | GEMV + GEMM K-loops | Prefetch weight rows 2 iterations ahead into L1 (`_MM_HINT_T0`) |
| **FMA instructions** | Everywhere | Single-instruction multiply-add, doubles throughput vs separate mul+add |
| **Bias fused into accumulators** | GEMV + GEMM | Accumulators initialized from bias vector; no separate add pass |
| **Register blocking (GEMV)** | `gemv_tiled_chunk` | RN=4 accumulators (64 floats) hide FMA latency, reduce loop overhead |
| **Register blocking (GEMM)** | `gemm_ukernel<4,4>` | 4x4 tile = 16 accumulators; maximizes ILP for FMA pipeline |
| **Template-based micro-kernel** | `gemm_ukernel<RM,RN>` | Compile-time unrolling for 4 tile sizes; no runtime branching inside kernel |
| **16-aligned GEMV partitioning** | `avx512_omp_gemv` | Thread boundaries align to AVX512 vector width |
| **RM-aligned GEMM partitioning** | `avx512_omp_gemm` | Thread boundaries align to register-block rows |
| **No library fallback** | M>1 path | Avoids MKL/torch::mm overhead and dispatch; direct AVX512 everywhere |

### Compiler flags

```
-O3 -std=c++17 -march=native -mavx512f -mfma -fopenmp
```

- `-O3`: Aggressive optimization (auto-vectorization, inlining, loop unrolling)
- `-march=native`: Target the build machine's exact ISA
- `-mavx512f`: Enable AVX-512 Foundation intrinsics
- `-mfma`: Enable FMA3 instructions
- `-fopenmp`: Enable OpenMP parallel regions
- `-std=c++17`: Required for `if constexpr` and structured bindings

---

## 12. Source Reference

All code is in `causal_language_modeling.py`:

| Component | Lines |
|-----------|-------|
| `GPT2MLPStepWrapper` (Python) | 321-357 |
| `_build_gpt2mlp_fused6` (build) | 1343-1675 |
| `fast_tanh_avx512` | 1357-1381 |
| `gemv_tiled_chunk` | 1387-1445 |
| `avx512_omp_gemv` | 1448-1471 |
| `gemm_ukernel<RM,RN>` | 1477-1504 |
| `avx512_omp_gemm` | 1507-1542 |
| `gpt2_mlp_fused6_step` | 1544-1648 |
| `_apply_gpt2mlp_fused6` | 1678-1685 |
