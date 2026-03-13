# Plan: Refactor AVX Codegen with Recursive Visitor Pattern

## Context

The existing `step/avx_codegen.py` (650 lines) is functionally correct — it generates the right C++ output and all test assertions match. However, the GEMV microkernel uses procedural emission with a hardcoded `_visit_linear_load_gemv(op, loop_var, buf_name, RN, VL, scalar, weight)` method that relies on boolean flags rather than inspecting the IR. The plan specification calls for a recursive visitor pattern where each op type has a `_visit_*()` method that derives behavior from the buffer's `IndexVar.step`/`register_block` attributes.

The refactoring applies the visitor pattern to **GEMV only**. GEMM keeps its template-based approach (the tests expect `__m512 acc[RM][RN]`, `Bv[r]`, etc. — fundamentally different from named-variable unrolling). All other methods (GELU, entry point, wrappers, registration) remain unchanged.

## Files to Modify

- `step/avx_codegen.py` — add visitor methods, refactor GEMV microkernel, delete old helper

## Implementation Steps

### Step 1: Add visitor infrastructure (~70 lines after `_fresh()`, before `generate()`)

**`_get_cpp_ptr(buf, ctx)`** — looks up `buf.name` in `ctx["buf_ptrs"]` to get C++ pointer name.

**`_build_gemv_ctx(stage)`** — builds context dict from stage:
```python
{
    "mode": "gemv",
    "buf_ptrs": {buf.name: key for key, buf in stage.buffers.items()},
    # e.g. {"x": "x", "wfc": "W", "bfc": "bias", "h": "y"}
    "RN": n_var.register_block,  # 4
    "VL": n_var.step,            # 16
    "n_loop_var": n_var.name,    # "n"
    "k_loop_var": k_var.name,    # "k"
    "K_size": "K",
    "N_size": "N",
}
```

**`_visit(op, ctx)`** — type dispatcher: routes to `_visit_linear_load`, `_visit_binary_map_accum`, `_visit_linear_store`, `_visit_unary_map`.

**`_visit_linear_load(op, ctx)`** — 3 cases derived from buffer IndexVars:
1. **All dims step=1** (scalar): `__m512 xk = _mm512_set1_ps(x[k]);` → returns `["xk"]`
2. **Single dim step=16** (1D vector): `__m512 bias{r} = _mm512_loadu_ps(bias + n + r*16);` → returns `["bias0",...,"bias3"]`
3. **Two dims: step=1 + step=16** (2D weight): `__m512 w{r} = _mm512_loadu_ps(W + k * N + n + r*16);` → returns `["w0",...,"w3"]`

Variable prefix convention: `cpp_ptr.lower()` for variable names, `cpp_ptr` for address expressions.

**`_visit_binary_map_accum(op, ctx)`** — recursive visitation:
1. `init_vars = _visit(op.init, ctx)` → `["bias0",...,"bias3"]` (emitted before k-loop)
2. Emit acc init: `__m512 acc{i} = {init_vars[i]};`
3. Open k-loop: `for (int64_t k = 0; k < K; k++) {`
4. `in1_vars = _visit(op.in1, ctx)` → `["xk"]` (emitted inside k-loop)
5. `in2_vars = _visit(op.in2, ctx)` → `["w0",...,"w3"]` (emitted inside k-loop)
6. Emit fmadd: `acc{i} = _mm512_fmadd_ps({in1_vars[0]}, {in2_vars[i]}, acc{i});`
7. Close k-loop
8. Return `["acc0",...,"acc3"]`

**`_visit_linear_store(op, ctx)`** — terminal:
1. `input_vars = _visit(op.input, ctx)` → triggers full recursive chain
2. Emit: `_mm512_storeu_ps(y + n + {i * VL}, {input_vars[i]});`

**`_visit_unary_map(op, ctx)`** — stub (GELU is emitted at entry-point level, not inside a microkernel).

### Step 2: Refactor `_emit_gemv_microkernel` (lines 148-223)

Keep the function signature and n-loop scaffolding (lines 148-173) unchanged. Replace the manual 3-phase orchestration (lines 175-219) with:
```python
ctx = self._build_gemv_ctx(stage)
store_op = next(op for op in stage.ops if isinstance(op, LinearStore))
self._visit(store_op, ctx)
```

The recursive `_visit(store_op)` call traverses: LinearStore → BinaryMapAccum → {LinearLoad(bias), LinearLoad(x), LinearLoad(W)}, emitting the entire GEMV body.

### Step 3: Delete `_visit_linear_load_gemv` (lines 225-255)

This method with its `scalar`/`weight` boolean flags is replaced by `_visit_linear_load` which derives behavior from `IndexVar.step`.

### Step 4: Run tests

```bash
python -m pytest tests/test_avx_codegen.py -v
```

All 30+ assertions must pass — the generated C++ strings are identical.

### Step 5: Docker integration test

```bash
docker exec -it mocha-bg bash -c "source /home/dockeruser/mochaenv/bin/activate && cd /home/dockeruser/step_cpu && python3 darpa/modified/causal_language_modeling_mlp.py --mode infer --prompt 'Why is the sky blue?' --cpu-only --replace gpt2mlp_fused6"
```

## What Stays Unchanged

- `generate()`, `_emit_includes()`, `_emit_fast_tanh()`, `_emit_registration()`
- `_emit_omp_gemv_wrapper()` — wraps gemv_tiled_chunk, unchanged
- `_emit_gemm_microkernel()`, `_emit_omp_gemm_wrapper()` — template approach, unchanged
- `_emit_gelu_vectorized()`, `_emit_gelu_parallel()` — GELU emission, unchanged
- `_emit_entry_point()`, `_emit_gemv_path()`, `_emit_gemm_path()` — unchanged
- All emit helpers: `_emit()`, `_indent_inc()`, `_indent_dec()`, `_fresh()`

## Net Change

~30 lines removed (old `_visit_linear_load_gemv` + manual orchestration), ~90 lines added (6 new visitor methods + context builder). Net: +60 lines.
