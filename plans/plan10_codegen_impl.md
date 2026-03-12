Plan: AVX512+OpenMP Code Generator for STeP GPT2 MLP                                                                                                           │
│                                                                                                                                                                │
│ Step 1: STeP IR (DONE) — darpa/modified/step_kernels.py                                                                                                        │
│                                                                                                                                                                │
│ IR Classes (as implemented)                                                                                                                                    │
│                                                                                                                                                                │
│ - IndexVar(name, size, step, parallelized, register_block) — loop index. size is int or str (C++ expr). register_block = how many AVX512 regs to unroll.       │
│ - Buffer(index_vars, torch_tensor, name) — tensor with per-dim IndexVar annotations.                                                                           │
│ - LinearLoad(in_buff, iter_space) — load from buffer. iter_space defaults to in_buff.index_vars. Determines loop placement.                                    │
│ - LinearStore(input, out_buff, iter_space) — store to buffer after input op completes.                                                                         │
│ - BinaryMapAccum(in1, in2, init, func, rank) — fused multiply-accumulate. rank=1 means accumulate over innermost dataflow dim. init placed before inner loop,  │
│ body inside, result after.                                                                                                                                     │
│ - UnaryMap(input, func) — element-wise unary op. stream_shape/tile_shape inherited from input. Codegen inspects func (e.g., torch.nn.functional.gelu) to pick  │
│ AVX512 lowering.                                                                                                                                               │
│ - StepStage(stage_type, ops, dataflow_order, buffers) — one computation stage with its loop nest.                                                              │
│ - StepProgram(name, tensor_params, dim_bindings) — general program with stages list. Branching (M=1 vs M>1) handled by separate programs.                      │
│                                                                                                                                                                │
│ Program builders (as implemented)                                                                                                                              │
│                                                                                                                                                                │
│ - build_gpt2_mlp_gemv_program() → 3 stages: GEMV1 → GELU → GEMV2                                                                                               │
│ - build_gpt2_mlp_gemm_program() → 3 stages: GEMM1 → GELU → GEMM2                                                                                               │
│ - _build_gemv_stage(x, w, bias, out, n, k) — ops: [LinearLoad(x,[k]), LinearLoad(w,[n,k]), LinearLoad(bias,[n]), BinaryMapAccum(..., rank=1),                  │
│ LinearStore(...,[n])], dataflow [n,k]                                                                                                                          │
│ - _build_gelu_stage(buf, k_var, m_var=None) — ops: [LinearLoad(buf,[k]), UnaryMap(h, gelu), LinearStore(gelu, buf,[k])]                                        │
│ - _build_gemm_stage(a, w, bias, out, m, n, k) — ops: [LinearLoad(a,[m,k]), LinearLoad(w,[k,n]), LinearLoad(bias,[n]), BinaryMapAccum(..., rank=1),             │
│ LinearStore(...,[m,n])], dataflow [m,n,k]                                                                                                                      │
│                                                                                                                                                                │
│ ---                                                                                                                                                            │
│ Step 2: AVX512 Code Generator (DONE) — step/avx_codegen.py                                                                                                     │
│                                                                                                                                                                │
│ Codegen logic overview                                                                                                                                         │
│                                                                                                                                                                │
│ The codegen takes a StepProgram and walks each StepStage to produce C++ with AVX512 intrinsics. The core algorithm has two phases:                             │
│ 1. Operation placement: determine which loop level each op belongs to                                                                                          │
│ 2. Recursive visitation: visit each op's inputs first, collect their emitted variable names, then emit the current op using those names                        │
│                                                                                                                                                                │
│ Each visit_*(op) method emits C++ code at the appropriate loop level and returns the variable name(s) it produced. Parent ops use those returned names as      │
│ arguments — no hardcoded patterns.                                                                                                                             │
│                                                                                                                                                                │
│ Operation placement algorithm                                                                                                                                  │
│                                                                                                                                                                │
│ Given dataflow_order = [n, k] (outer to inner), for each op:                                                                                                   │
│ 1. Find the innermost dataflow dim present in the op's iter_space.                                                                                             │
│ 2. That dim's loop level is where the op is emitted.                                                                                                           │
│                                                                                                                                                                │
│ For BinaryMapAccum with rank=1:                                                                                                                                │
│ - init (bias load): iter_space [n] → n-loop level, emitted before k-loop opens                                                                                 │
│ - in1 (x load): iter_space [k] → k-loop level                                                                                                                  │
│ - in2 (w load): iter_space [n,k] → k-loop level                                                                                                                │
│ - body (fmadd): inside k-loop                                                                                                                                  │
│ - result + LinearStore: iter_space [n] → n-loop level, emitted after k-loop closes                                                                             │
│                                                                                                                                                                │
│ Recursive visitation — every node emits variables                                                                                                              │
│                                                                                                                                                                │
│ Uniform rule: every non-terminal node emits __m512 variable(s) when visited and returns the variable name(s). The consuming node uses those names as           │
│ arguments. No inlining, no pre-analysis — the C++ compiler handles copy propagation at -O2/-O3, so extra variables produce identical assembly.                 │
│                                                                                                                                                                │
│ Return type convention                                                                                                                                         │
│                                                                                                                                                                │
│ The return is always a list of variable names, shaped by register_block:                                                                                       │
│ - 1D (GEMV, GELU): List[str] — length = register_block of the relevant dim (or 1 for scalars)                                                                  │
│ - 2D (GEMM): List[List[str]] — shape [RM][RN], matching __m512 acc[RM][RN] in C++                                                                              │
│                                                                                                                                                                │
│ ┌──────────────────────┬───────────────────────┬───────────────────────────────┬─────────────────┐                                                             │
│ │         Node         │    GEMV (1D, RN=4)    │     GEMM (2D, RM=4, RN=4)     │ GELU (1D, RB=1) │                                                             │
│ ├──────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────┤                                                             │
│ │ LinearLoad(bias,[n]) │ ["bias0",...,"bias3"] │ ["bias0",...,"bias3"]         │ —               │                                                             │
│ ├──────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────┤                                                             │
│ │ LinearLoad(x,[k])    │ ["xk"]                │ —                             │ —               │                                                             │
│ ├──────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────┤                                                             │
│ │ LinearLoad(A,[m,k])  │ —                     │ [["a0"],...,["a3"]] (RM×1)    │ —               │                                                             │
│ ├──────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────┤                                                             │
│ │ LinearLoad(B,[k,n])  │ —                     │ ["b0",...,"b3"] (1×RN)        │ —               │                                                             │
│ ├──────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────┤                                                             │
│ │ LinearLoad(buf,[k])  │ —                     │ —                             │ ["v"]           │                                                             │
│ ├──────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────┤                                                             │
│ │ BinaryMapAccum       │ ["acc0",...,"acc3"]   │ [["acc_0_0",...],...] (RM×RN) │ —               │                                                             │
│ ├──────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────┤                                                             │
│ │ UnaryMap(gelu)       │ —                     │ —                             │ ["result"]      │                                                             │
│ └──────────────────────┴───────────────────────┴───────────────────────────────┴─────────────────┘                                                             │
│                                                                                                                                                                │
│ Node visitors                                                                                                                                                  │
│                                                                                                                                                                │
│ visit_LinearLoad(op) → emits variable(s), returns name list:                                                                                                   │
│                                                                                                                                                                │
│ Inspects the buffer's IndexVar to decide the intrinsic:                                                                                                        │
│ - Buffer dim has step=1 → broadcast: emits __m512 {name} = _mm512_set1_ps(ptr[idx]); → returns ["{name}"]                                                      │
│ - Buffer dim has step=16 with register_block=RN → vector load: emits RN lines __m512 {name}{i} = _mm512_loadu_ps(ptr + offset + i*16); → returns ["{name}0",   │
│ ..., "{name}{RN-1}"]                                                                                                                                           │
│ - GEMM 2D case: If iter_space has two register-blocked dims (m,n), return nested List[List[str]]                                                               │
│                                                                                                                                                                │
│ visit_BinaryMapAccum(op) → emits variable(s), returns name list:                                                                                               │
│                                                                                                                                                                │
│ GEMV (1D, dataflow [n,k]):                                                                                                                                     │
│ 1. Visit op.init → get init_vars: ["bias0", "bias1", "bias2", "bias3"]                                                                                         │
│ 2. Emit acc init: __m512 acc{i} = {init_vars[i]}; — placed before k-loop                                                                                       │
│ 3. Visit op.in1 → get in1_vars: ["xk"]                                                                                                                         │
│ 4. Visit op.in2 → get in2_vars: ["w0", "w1", "w2", "w3"]                                                                                                       │
│ 5. Emit body: acc{i} = _mm512_fmadd_ps({in1_vars[0]}, {in2_vars[i]}, acc{i}); — inside k-loop                                                                  │
│ 6. Returns ["acc0", ..., "acc3"]                                                                                                                               │
│                                                                                                                                                                │
│ GEMM (2D, dataflow [m,n,k]):                                                                                                                                   │
│ 1. Visit op.init → get init_vars: ["bias0", ..., "bias3"] (only n-dim)                                                                                         │
│ 2. Emit 2D acc init: __m512 acc_{mr}_{nr} = {init_vars[nr]}; for each (mr, nr) — before k-loop                                                                 │
│ 3. Visit op.in1 (A load, [m,k]) → get in1_vars: [["a0"], ["a1"], ["a2"], ["a3"]]                                                                               │
│ 4. Visit op.in2 (B load, [k,n]) → get in2_vars: ["b0", "b1", "b2", "b3"]                                                                                       │
│ 5. Emit nested body: acc_{mr}_{nr} = _mm512_fmadd_ps({in1_vars[mr][0]}, {in2_vars[nr]}, acc_{mr}_{nr});                                                        │
│ 6. Returns [["acc_0_0",...,"acc_0_3"], ..., ["acc_3_0",...,"acc_3_3"]]                                                                                         │
│                                                                                                                                                                │
│ visit_UnaryMap(op) → emits variable(s), returns name list:                                                                                                     │
│                                                                                                                                                                │
│ 1. Visit op.input → get input_vars                                                                                                                             │
│ 2. Inspect op.func to select lowering (e.g., gelu → multi-step GELU AVX512 sequence)                                                                           │
│ 3. Emit computation chain as named variables (e.g., v2 = mul(v,v); v3 = mul(v2,v); ...)                                                                        │
│ 4. Return final result var names                                                                                                                               │
│                                                                                                                                                                │
│ visit_LinearStore(op) → emits store, returns nothing (terminal):                                                                                               │
│                                                                                                                                                                │
│ 1. Visit op.input → get input_vars (e.g., ["acc0", ..., "acc3"])                                                                                               │
│ 2. Emit: _mm512_storeu_ps(ptr + offset + i*16, {input_vars[i]}); for each i                                                                                    │
│                                                                                                                                                                │
│ Example generated output (GEMV, RN=4)                                                                                                                          │
│                                                                                                                                                                │
│ // bias load: visit_LinearLoad(bias_load) → emits 4 vars                                                                                                       │
│ __m512 bias0 = _mm512_loadu_ps(bias_ptr + n);                                                                                                                  │
│ __m512 bias1 = _mm512_loadu_ps(bias_ptr + n + 16);                                                                                                             │
│ __m512 bias2 = _mm512_loadu_ps(bias_ptr + n + 32);                                                                                                             │
│ __m512 bias3 = _mm512_loadu_ps(bias_ptr + n + 48);                                                                                                             │
│ // accum init: visit_BinaryMapAccum uses init_vars                                                                                                             │
│ __m512 acc0 = bias0;                                                                                                                                           │
│ __m512 acc1 = bias1;                                                                                                                                           │
│ __m512 acc2 = bias2;                                                                                                                                           │
│ __m512 acc3 = bias3;                                                                                                                                           │
│ for (int64_t k = 0; k < K; k++) {                                                                                                                              │
│     // x broadcast: visit_LinearLoad(x_load) → emits 1 var                                                                                                     │
│     __m512 xk = _mm512_set1_ps(x_ptr[k]);                                                                                                                      │
│     // weight load: visit_LinearLoad(w_load) → emits 4 vars                                                                                                    │
│     __m512 w0 = _mm512_loadu_ps(w_ptr + k * N + n);                                                                                                            │
│     __m512 w1 = _mm512_loadu_ps(w_ptr + k * N + n + 16);                                                                                                       │
│     __m512 w2 = _mm512_loadu_ps(w_ptr + k * N + n + 32);                                                                                                       │
│     __m512 w3 = _mm512_loadu_ps(w_ptr + k * N + n + 48);                                                                                                       │
│     // fmadd body: visit_BinaryMapAccum uses in1_vars, in2_vars                                                                                                │
│     acc0 = _mm512_fmadd_ps(xk, w0, acc0);                                                                                                                      │
│     acc1 = _mm512_fmadd_ps(xk, w1, acc1);                                                                                                                      │
│     acc2 = _mm512_fmadd_ps(xk, w2, acc2);                                                                                                                      │
│     acc3 = _mm512_fmadd_ps(xk, w3, acc3);                                                                                                                      │
│ }                                                                                                                                                              │
│ // store: visit_LinearStore uses input_vars from BinaryMapAccum                                                                                                │
│ _mm512_storeu_ps(y_ptr + n, acc0);                                                                                                                             │
│ _mm512_storeu_ps(y_ptr + n + 16, acc1);                                                                                                                        │
│ _mm512_storeu_ps(y_ptr + n + 32, acc2);                                                                                                                        │
│ _mm512_storeu_ps(y_ptr + n + 48, acc3);                                                                                                                        │
│                                                                                                                                                                │
│ Documentation                                                                                                                                                  │
│                                                                                                                                                                │
│ Create plans/inlining_approaches.md documenting the two-pass vs pre-analysis inlining approaches — for future use if source-level inlining becomes needed for  │
│ more complex programs.                                                                                                                                         │
│                                                                                                                                                                │
│ AVX512 instruction selection                                                                                                                                   │
│                                                                                                                                                                │
│ The codegen decides which intrinsic to use based on the buffer's IndexVar steps:                                                                               │
│                                                                                                                                                                │
│ ┌──────────────────┬───────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐          │
│ │     Pattern      │         Condition         │                                          AVX512 intrinsic                                          │          │
│ ├──────────────────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤          │
│ │ Scalar broadcast │ Buffer dim has step=1     │ _mm512_set1_ps(ptr[idx])                                                                           │          │
│ ├──────────────────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤          │
│ │ Vector load      │ Buffer dim has step=16    │ RN × _mm512_loadu_ps(ptr + offset + r*16)                                                          │          │
│ ├──────────────────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤          │
│ │ FMA body         │ BinaryMapAccum func=a*b+c │ RN × _mm512_fmadd_ps(in1_result, in2_result[r], acc_r) — in1/in2 may be inlined exprs or var names │          │
│ ├──────────────────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤          │
│ │ Vector store     │ LinearStore               │ RN × _mm512_storeu_ps(ptr + offset + r*16, input_var[r])                                           │          │
│ ├──────────────────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤          │
│ │ GELU             │ UnaryMap func=gelu        │ load → v²→v³→fmadd→mul→fast_tanh→mul→store                                                         │          │
│ └──────────────────┴───────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘          │
│                                                                                                                                                                │
│ Register count per tile = IndexVar.register_block (RN for n-dim, RM for m-dim).                                                                                │
│                                                                                                                                                                │
│ Generated C++ structure                                                                                                                                        │
│                                                                                                                                                                │
│ #includes (torch, immintrin, omp)                                                                                                                              │
│ fast_tanh_avx512()          — fixed Pade [7,6] helper                                                                                                          │
│ gemv_tiled_chunk()          — from GEMV stage                                                                                                                  │
│ avx512_omp_gemv()           — OpenMP wrapper for GEMV                                                                                                          │
│ gemm_ukernel<RM,RN>()      — from GEMM stage (template)                                                                                                        │
│ avx512_omp_gemm()           — OpenMP wrapper for GEMM                                                                                                          │
│ gpt2_mlp_fused6_step()      — entry point: if(M==1) GEMV path else GEMM path                                                                                   │
│ TORCH_LIBRARY_FRAGMENT       — op registration                                                                                                                 │
│                                                                                                                                                                │
│ Key files                                                                                                                                                      │
│                                                                                                                                                                │
│ - Create: step/avx_codegen.py                                                                                                                                  │
│ - Reference target: darpa/modified/causal_language_modeling_mlp.py lines 369-667

---

## Step 3: Update `step/compile.py` - AVX512 flags

Add `avx512: bool = False` parameter to `build_extension()`. When True, append:
```python
extra_cflags += ["-march=native", "-fopenmp", "-mavx512f", "-mfma"]
extra_ldflags += ["-fopenmp"]
```

---

## Step 4: Update `darpa/modified/causal_language_modeling_mlp.py` - Wire codegen

Replace the hardcoded `_FUSED6_CPP_SOURCE` string in `_build_gpt2mlp_fused6()` with:
```python
def _build_gpt2mlp_fused6():
    from step.avx_codegen import AVXCodegen
    from step.compile import build_extension
    from darpa.modified.step_kernels import build_gpt2_mlp_gemv_program, build_gpt2_mlp_gemm_program

    gemv = build_gpt2_mlp_gemv_program()
    gemm = build_gpt2_mlp_gemm_program()
    codegen = AVXCodegen(gemv, gemm)
    cpp_source = codegen.generate()
    return build_extension(gemv.name, cpp_source, avx512=True)
```

---

## Step 5: Add tests

### `tests/test_avx_codegen.py` - Unit tests on generated C++ string

- **TestAVXCodegenStructure**: Verify includes, `fast_tanh_avx512`, `TORCH_LIBRARY_FRAGMENT`
- **TestAVXCodegenGEMV**: Verify gemv signature, 4 accumulators, broadcast/load/fmadd/store
- **TestAVXCodegenGEMM**: Verify template, 2D acc array, instantiations
- **TestAVXCodegenGELU**: Verify constants, fast_tanh call, parallel for
- **TestAVXCodegenEntryPoint**: Verify M==1 branch, dim extraction, contiguity checks

### Integration test in Docker:
```bash
docker exec -it mocha-bg bash
source /home/dockeruser/mochaenv/bin/activate
cd /home/dockeruser/step_cpu
python3 darpa/modified/causal_language_modeling_mlp.py --mode infer --prompt "Why is the sky blue?" --cpu-only --replace gpt2mlp_fused6
```

---

## Implementation Order

1. `step_kernels.py`: Add new IR classes, extend IndexVar, implement `build_gpt2_mlp_program()`
2. `step/avx_codegen.py`: Scaffold + GEMV lowering
3. `step/avx_codegen.py`: GELU lowering
4. `step/avx_codegen.py`: GEMM lowering
5. `step/compile.py`: AVX512 flag support
6. `causal_language_modeling_mlp.py`: Wire codegen
7. `tests/test_avx_codegen.py`: Tests

## Key Design Decisions

- **Separate codegen from existing `step/codegen.py`**: Existing emits PyTorch tensor ops; AVX512 emits raw intrinsics.
- **`UnaryMap(func=gelu)` for GELU**: General UnaryMap with codegen inspecting `func` to select AVX512 lowering (fast_tanh_avx512 Pade pattern).
- **`fast_tanh_avx512` emitted verbatim**: Pade approximation coefficients are fixed math.
- **M==1 / M>1 branch**: GEMV and GEMM have different register blocking/parallelization.
- **No remainder handling in v1**: GPT2 dims (768, 3072) divisible by 64.

## Key Reference: Operation-to-AVX512 Mapping

- LinearLoad buffer step=1 -> `_mm512_set1_ps(ptr[idx])` (scalar broadcast)
- LinearLoad buffer step=16*RN -> RN x `_mm512_loadu_ps(ptr + offset + r*16)` (vector loads)
- BinaryMapAccum init -> RN x `_mm512_loadu_ps(bias + n + r*16)` (accum init)
- BinaryMapAccum body `a*b+c` -> RN x `_mm512_fmadd_ps(a_vec, b_vec, acc_r)` (FMA)
- LinearStore -> RN x `_mm512_storeu_ps(ptr + n + r*16, acc_r)` (vector store)
- UnaryMap(func=gelu) -> vectorized GELU with fast_tanh_avx512

## Key Reference: Address Computation for GEMV

For weight W[K,N] (row-major, where K is reduction dim, N is output dim):
- `W[k, n] -> w_ptr + k * N + n` with register r: `w_ptr + k * N + n + r * 16`
For input x[K] (1D): `x[k] -> x_ptr[k]`
For bias[N]: `bias[n + r*16] -> bias_ptr + n + r * 16`

## Key Reference: Operation Placement Algorithm

For each op, find the innermost loop level containing ALL its iter_space IndexVars.
For BinaryMapAccum with rank=r:
- init operand: placed BEFORE the inner r-th loop (accumulator initialization)
- body operands: placed INSIDE the inner r-th loop
- result: available AFTER the inner r-th loop closes
- LinearStore consuming result: placed after the inner loop closes
