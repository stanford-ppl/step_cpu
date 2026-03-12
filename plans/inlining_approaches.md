# Inlining Approaches for AVX512 Code Generator

## Context

When the codegen visits `LinearLoad` nodes, it can either:
1. Emit a named variable and return the name (consumer uses the variable)
2. Return an expression string (consumer inlines it into its own emission)

For example, in GEMV with RN=4:
```cpp
// Option A: Named variables (current approach)
__m512 bias0 = _mm512_loadu_ps(bias + n);
__m512 acc0 = bias0;

// Option B: Inlined expression
__m512 acc0 = _mm512_loadu_ps(bias + n);  // no intermediate variable
```

## Performance Analysis

Both approaches produce **identical assembly** with `-O2` or `-O3`. The C++ compiler performs copy propagation on `__m512` types (which are register-width). An intermediate `__m512` variable used once compiles to either a register rename (free) or a single `vmovaps` (1 cycle latency, eliminated by the optimizer).

## Current Implementation: Always Emit Variables (v1)

We chose to always emit named variables because:
- Simplest codegen logic — each `visit_*` method is self-contained
- No pre-analysis or post-processing needed
- No performance penalty with compiler optimization
- Uniform pattern: every non-terminal node emits variables and returns names

## Alternative: Option 1 — Two-Pass with Statement IR

**How it works:**
1. First pass: emit all nodes as named variables into a lightweight intermediate representation (list of Statement objects, not raw C++ strings)
2. Second pass: count consumers of each variable; for single-use variables, substitute the RHS expression into the consumer and remove the variable declaration
3. Serialize the IR to C++ strings

**Pros:**
- Clean separation of concerns (emission vs optimization)
- Similar to LLVM's approach (emit SSA, then copy propagation pass)
- The inlining pass is a well-understood compiler optimization

**Cons:**
- Requires an intermediate Statement IR between the visitor and C++ serialization
- More engineering effort
- String-level substitution on raw C++ would be fragile (need to handle scoping, etc.)

**When to use:** If we need source-level inlining for readability of generated code, or if we extend to targets where the compiler doesn't optimize well (e.g., embedded compilers without copy propagation).

## Alternative: Option 2 — Pre-Analysis Consumer Counting

**How it works:**
1. Before codegen, walk the STeP op graph to count C++-level uses of each op's output
2. Key insight: `BinaryMapAccum` with `register_block=RN` uses `in1` result RN times (once per fmadd), but `init` and `in2` results once each per register
3. During `visit_LinearLoad`, check the use count: if 1 → return expression (inline); if >1 → emit variable, return name

**Pros:**
- Single-pass emission with pre-computed decisions
- Standard approach in DSL compilers (TVM, Halide, MLIR)
- No post-processing needed

**Cons:**
- Emission logic must be aware of use counts (slightly more complex)
- Consumer counting at C++-level (not graph-level) requires understanding register blocking

**When to use:** If source-level inlining becomes important for debugging or matching specific reference code patterns exactly.

## Migration Path

If we need to switch from v1 (always variables) to Option 1 or 2:

1. **To Option 2:** Add a `_count_uses(stage)` method that walks ops and computes `use_count[op_id]`. Modify `visit_LinearLoad` to check the count and conditionally return expression vs variable name. ~50 lines of code.

2. **To Option 1:** Create a `Statement` dataclass (`var_name`, `rhs_expr`, `is_declaration`). Change `_emit()` calls to append Statements to a list. Add an `_inline_single_use(statements)` pass. Add `_serialize(statements)` to convert to C++ strings. ~100 lines of code.
