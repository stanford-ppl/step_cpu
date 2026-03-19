"""AVX512+OpenMP C++ code generator for STeP programs.

Walks a StepProgram and emits C++ source with AVX512 intrinsics and OpenMP
parallelization. Each StepStage is lowered to a C++ function; the entry point
dispatches between GEMV (M=1) and GEMM (M>1) paths via separate programs.

The core algorithm:
  1. Operation placement: each op placed at the innermost loop level
     containing ALL its iter_space IndexVars.
  2. Recursive visitation: visit each op's inputs, emit variables,
     return variable names to the consuming op.
"""

from __future__ import annotations

import torch.nn.functional
from typing import List, Dict, Optional, Union

from step.step_kernels import (
    BinaryMapAccum,
    Buffer,
    IndexVar,
    LinearLoad,
    LinearStore,
    StepOps,
    StepProgram,
    StepStage,
    UnaryMap,
)


ISA_PROFILES = {
    "avx512": {
        "VL": 16,
        "vec_type": "__m512",
        "loadu": "_mm512_loadu_ps",
        "storeu": "_mm512_storeu_ps",
        "set1": "_mm512_set1_ps",
        "setzero": "_mm512_setzero_ps",
        "fmadd": "_mm512_fmadd_ps",
        "mul": "_mm512_mul_ps",
        "add": "_mm512_add_ps",
        "sub": "_mm512_sub_ps",
        "div": "_mm512_div_ps",
        "min": "_mm512_min_ps",
        "max": "_mm512_max_ps",
        "tanh_fn": "fast_tanh_avx512",
        "gemv_fn": "avx512_omp_gemv",
        "gemm_fn": "avx512_omp_gemm",
    },
    "avx2": {
        "VL": 8,
        "vec_type": "__m256",
        "loadu": "_mm256_loadu_ps",
        "storeu": "_mm256_storeu_ps",
        "set1": "_mm256_set1_ps",
        "setzero": "_mm256_setzero_ps",
        "fmadd": "_mm256_fmadd_ps",
        "mul": "_mm256_mul_ps",
        "add": "_mm256_add_ps",
        "sub": "_mm256_sub_ps",
        "div": "_mm256_div_ps",
        "min": "_mm256_min_ps",
        "max": "_mm256_max_ps",
        "tanh_fn": "fast_tanh_avx2",
        "gemv_fn": "avx2_omp_gemv",
        "gemm_fn": "avx2_omp_gemm",
    },
}


class AVXCodegen:
    """Generates AVX512/AVX2+OpenMP C++ from a decode StepProgram and a prefill StepProgram."""

    def __init__(
        self,
        decode: StepProgram,
        prefill: StepProgram,
        isa: str = "avx512",
    ):
        self.decode = decode
        self.prefill = prefill
        self.isa = ISA_PROFILES[isa]
        self.lines: list[str] = []
        self._indent = 0
        self._var_counter = 0

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def _emit(self, line: str = "") -> None:
        if line:
            self.lines.append("    " * self._indent + line)
        else:
            self.lines.append("")

    def _indent_inc(self) -> None:
        self._indent += 1

    def _indent_dec(self) -> None:
        self._indent -= 1

    def _fresh(self, prefix: str) -> str:
        self._var_counter += 1
        return f"{prefix}_{self._var_counter}"

    # ------------------------------------------------------------------
    # GEMV visitor infrastructure
    # ------------------------------------------------------------------

    def _get_cpp_ptr(self, buf, ctx: dict) -> str:
        """Look up the C++ pointer name for a buffer."""
        return ctx["buf_ptrs"][buf.name]

    def _build_gemv_ctx(self, stage: StepStage) -> dict:
        """Build visitor context from a GEMV stage."""
        n_var = stage.dataflow_order[0]
        k_var = stage.dataflow_order[1]
        return {
            "mode": "gemv",
            "buf_ptrs": {buf.name: key for key, buf in stage.buffers.items()},
            "RN": n_var.register_block,
            "VL": n_var.step,
            "n_loop_var": n_var.name,
            "k_loop_var": k_var.name,
            "K_size": "K",
            "N_size": "N",
        }

    def _visit(self, op, ctx: dict):
        """Type dispatcher: route to the appropriate visitor method."""
        if isinstance(op, LinearLoad):
            return self._visit_linear_load(op, ctx)
        elif isinstance(op, BinaryMapAccum):
            return self._visit_binary_map_accum(op, ctx)
        elif isinstance(op, LinearStore):
            return self._visit_linear_store(op, ctx)
        elif isinstance(op, UnaryMap):
            return self._visit_unary_map(op, ctx)
        else:
            raise ValueError(f"Unknown op type: {type(op)}")

    def _visit_linear_load(self, op, ctx: dict) -> List[str]:
        """Visit a LinearLoad, emit variables, return names.

        Derives behavior from the buffer's IndexVar steps:
          - All dims step=1  -> scalar broadcast (_mm512_set1_ps)
          - Single dim step=VL -> 1D vector load (_mm512_loadu_ps)
          - Two dims (step=1 + step=VL) -> 2D weight load
        """
        buf = op.in_buff
        cpp_ptr = self._get_cpp_ptr(buf, ctx)
        prefix = cpp_ptr.lower()
        RN = ctx["RN"]
        VL = ctx["VL"]
        n_var = ctx["n_loop_var"]
        k_var = ctx["k_loop_var"]

        steps = [iv.step for iv in buf.index_vars]

        vt = self.isa["vec_type"]
        set1 = self.isa["set1"]
        loadu = self.isa["loadu"]

        if all(s == 1 for s in steps):
            # Scalar broadcast
            name = f"{prefix}{k_var}"
            self._emit(f"{vt} {name} = {set1}({cpp_ptr}[{k_var}]);")
            return [name]
        elif len(buf.index_vars) == 1 and steps[0] == VL:
            # 1D vector load
            var_names = []
            for r in range(RN):
                name = f"{prefix}{r}"
                offset = f"{n_var} + {r * VL}" if r > 0 else n_var
                self._emit(f"{vt} {name} = {loadu}({cpp_ptr} + {offset});")
                var_names.append(name)
            return var_names
        else:
            # 2D weight load
            N_size = ctx["N_size"]
            var_names = []
            for r in range(RN):
                name = f"{prefix}{r}"
                offset = (f"{k_var} * {N_size} + {n_var} + {r * VL}"
                          if r > 0 else f"{k_var} * {N_size} + {n_var}")
                self._emit(f"{vt} {name} = {loadu}({cpp_ptr} + {offset});")
                var_names.append(name)
            return var_names

    def _visit_binary_map_accum(self, op, ctx: dict) -> List[str]:
        """Visit a BinaryMapAccum: recursive visitation with k-loop."""
        RN = ctx["RN"]
        K_size = ctx["K_size"]
        k_var = ctx["k_loop_var"]

        vt = self.isa["vec_type"]
        fmadd = self.isa["fmadd"]

        # Visit init (bias load) — emitted before k-loop
        init_vars = self._visit(op.init, ctx)

        # Emit acc init
        acc_vars = []
        for i in range(RN):
            acc_name = f"acc{i}"
            self._emit(f"{vt} {acc_name} = {init_vars[i]};")
            acc_vars.append(acc_name)

        self._emit("")

        # Open k-loop
        self._emit(f"for (int64_t {k_var} = 0; {k_var} < {K_size}; {k_var}++) {{")
        self._indent_inc()

        # Visit in1 (x load — scalar broadcast)
        in1_vars = self._visit(op.in1, ctx)

        # Visit in2 (W load — vector loads)
        in2_vars = self._visit(op.in2, ctx)

        # Emit fmadd body
        for i in range(RN):
            self._emit(f"{acc_vars[i]} = {fmadd}({in1_vars[0]}, {in2_vars[i]}, {acc_vars[i]});")

        # Prefetch next k iteration's weight data into L1
        w_ptr = self._get_cpp_ptr(op.in2.in_buff, ctx)
        N_size = ctx["N_size"]
        n_var = ctx["n_loop_var"]
        VL = ctx["VL"]
        self._emit(f"if ({k_var} + 1 < {K_size}) {{")
        self._indent_inc()
        for r in range(RN):
            offset = f"({k_var}+1) * {N_size} + {n_var} + {r * VL}" if r > 0 else f"({k_var}+1) * {N_size} + {n_var}"
            self._emit(f"_mm_prefetch((const char*)({w_ptr} + {offset}), _MM_HINT_T0);")
        self._indent_dec()
        self._emit("}")

        self._indent_dec()
        self._emit("}")
        self._emit("")

        return acc_vars

    def _visit_linear_store(self, op, ctx: dict) -> None:
        """Visit a LinearStore: terminal node, emit stores."""
        VL = ctx["VL"]
        n_var = ctx["n_loop_var"]
        cpp_ptr = self._get_cpp_ptr(op.out_buff, ctx)

        # Visit input — triggers full recursive chain
        input_vars = self._visit(op.input, ctx)

        # Emit stores
        storeu = self.isa["storeu"]
        for i in range(len(input_vars)):
            self._emit(f"{storeu}({cpp_ptr} + {n_var} + {i * VL}, {input_vars[i]});")

    def _visit_unary_map(self, op, ctx: dict):
        """Stub: GELU is emitted at entry-point level, not inside a microkernel."""
        pass

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self) -> str:
        self.lines = []
        self._indent = 0
        self._var_counter = 0

        self._emit_includes()
        self._emit_fast_tanh()
        self._emit("")

        # GEMV functions
        for stage in self.decode.stages:
            if stage.stage_type == "gemv":
                self._emit_gemv_microkernel(stage)
                self._emit_omp_gemv_wrapper(stage)
                break  # all GEMV stages share the same microkernel shape

        # GEMM functions
        for stage in self.prefill.stages:
            if stage.stage_type == "gemm":
                self._emit_gemm_microkernel(stage)
                self._emit_omp_gemm_wrapper(stage)
                break

        # Entry point with M==1 / M>1 branch
        self._emit_entry_point()
        self._emit_registration()

        return "\n".join(self.lines) + "\n"

    def generate_kernels_only(self, extra_includes: list[str] | None = None) -> str:
        """Generate only kernel functions (GEMV + GEMM), no entry point or registration.

        Used when the entry point will be provided as a hardcoded C++ string
        (e.g., attention kernel with custom inner logic).
        """
        self.lines = []
        self._indent = 0
        self._var_counter = 0

        self._emit_includes()
        if extra_includes:
            for inc in extra_includes:
                self._emit(inc)
            self._emit("")
        self._emit_fast_tanh()
        self._emit("")

        # GEMV functions
        for stage in self.decode.stages:
            if stage.stage_type == "gemv":
                self._emit_gemv_microkernel(stage)
                self._emit_omp_gemv_wrapper(stage)
                break

        # GEMM functions
        for stage in self.prefill.stages:
            if stage.stage_type == "gemm":
                self._emit_gemm_microkernel(stage)
                self._emit_omp_gemm_wrapper(stage)
                break

        return "\n".join(self.lines) + "\n"

    # ------------------------------------------------------------------
    # Includes
    # ------------------------------------------------------------------

    def _emit_includes(self) -> None:
        self._emit("#include <torch/extension.h>")
        self._emit("#include <torch/library.h>")
        self._emit("#include <immintrin.h>")
        self._emit("#include <cmath>")
        self._emit("#include <omp.h>")
        self._emit("")

    # ------------------------------------------------------------------
    # fast_tanh_avx512 — fixed Pade [7,6] rational approximation
    # ------------------------------------------------------------------

    def _emit_fast_tanh(self) -> None:
        vt = self.isa["vec_type"]
        set1 = self.isa["set1"]
        setzero = self.isa["setzero"]
        mul = self.isa["mul"]
        add = self.isa["add"]
        sub = self.isa["sub"]
        div = self.isa["div"]
        fmadd = self.isa["fmadd"]
        min_ps = self.isa["min"]
        max_ps = self.isa["max"]
        tanh_fn = self.isa["tanh_fn"]

        self._emit(f"// Fast tanh approximation: [7,6] Pade rational form")
        self._emit(f"static inline {vt} {tanh_fn}({vt} x) {{")
        self._indent_inc()
        self._emit(f"{vt} lim = {set1}(4.97f);")
        self._emit(f"{vt} x_clamped = {min_ps}({max_ps}(x, {sub}({setzero}(), lim)), lim);")
        self._emit(f"{vt} x2 = {mul}(x_clamped, x_clamped);")
        self._emit(f"{vt} x4 = {mul}(x2, x2);")
        self._emit(f"{vt} x6 = {mul}(x4, x2);")
        self._emit("")
        self._emit(f"{vt} c135135 = {set1}(135135.0f);")
        self._emit(f"{vt} c17325  = {set1}(17325.0f);")
        self._emit(f"{vt} c378    = {set1}(378.0f);")
        self._emit(f"{vt} c62370  = {set1}(62370.0f);")
        self._emit(f"{vt} c3150   = {set1}(3150.0f);")
        self._emit(f"{vt} c28     = {set1}(28.0f);")
        self._emit("")
        self._emit(f"{vt} num = {fmadd}(c17325, x2, c135135);")
        self._emit(f"num = {fmadd}(c378, x4, num);")
        self._emit(f"num = {add}(num, x6);")
        self._emit("")
        self._emit(f"{vt} den = {fmadd}(c62370, x2, c135135);")
        self._emit(f"den = {fmadd}(c3150, x4, den);")
        self._emit(f"den = {fmadd}(c28, x6, den);")
        self._emit("")
        self._emit(f"return {mul}(x_clamped, {div}(num, den));")
        self._indent_dec()
        self._emit("}")

    # ------------------------------------------------------------------
    # GEMV microkernel
    # ------------------------------------------------------------------

    def _emit_gemv_microkernel(self, stage: StepStage) -> None:
        """Emit gemv_tiled_chunk from the GEMV stage ops via recursive visitation."""
        dataflow = stage.dataflow_order  # [n, k]
        n_var = dataflow[0]
        k_var = dataflow[1]
        RN = n_var.register_block
        VL = n_var.step  # 16

        # Collect buffer names from stage
        bufs = stage.buffers  # {"x": ..., "W": ..., "bias": ..., "y": ...}

        self._emit(f"// GEMV tiled micro-kernel: RN={RN} vectors of {VL} ({RN * VL} floats) per tile")
        self._emit("static void gemv_tiled_chunk(const float* __restrict__ x,")
        self._emit("                             const float* __restrict__ W,")
        self._emit("                             const float* __restrict__ bias,")
        self._emit("                             float* __restrict__ y,")
        self._emit("                             int64_t K, int64_t N,")
        self._emit("                             int64_t n_start, int64_t n_end) {")
        self._indent_inc()
        self._emit(f"constexpr int RN = {RN};")
        self._emit("")

        # Main tiled loop
        self._emit("int64_t n = n_start;")
        self._emit(f"for (; n + RN * {VL} <= n_end; n += RN * {VL}) {{")
        self._indent_inc()

        # Recursive visitation: LinearStore -> BinaryMapAccum -> LinearLoads
        ctx = self._build_gemv_ctx(stage)
        store_op = next(op for op in stage.ops if isinstance(op, LinearStore))
        self._visit(store_op, ctx)

        self._indent_dec()
        self._emit("}")

        self._indent_dec()
        self._emit("}")
        self._emit("")

    # ------------------------------------------------------------------
    # OpenMP GEMV wrapper
    # ------------------------------------------------------------------

    def _emit_omp_gemv_wrapper(self, stage: StepStage) -> None:
        """Emit avx512_omp_gemv with tile-aligned chunk partitioning."""
        dataflow = stage.dataflow_order  # [n, k]
        n_var = dataflow[0]
        RN = n_var.register_block
        VL = n_var.step  # 16
        TILE = RN * VL  # 64 — minimum chunk per thread

        gemv_fn = self.isa["gemv_fn"]
        self._emit("// OpenMP parallel GEMV: y[0..N) = x[0..K) * W[K,N] + bias[0..N)")
        self._emit(f"static void {gemv_fn}(const float* x, const float* W, const float* bias,")
        self._emit(f"{'':>{len(gemv_fn) + 13}}float* y, int64_t K, int64_t N) {{")
        self._indent_inc()
        self._emit(f"constexpr int64_t TILE = {TILE};  // RN * VL")
        self._emit("int64_t ntiles = N / TILE;")
        self._emit("#pragma omp parallel")
        self._emit("{")
        self._indent_inc()
        self._emit("int nthreads = omp_get_num_threads();")
        self._emit("int tid = omp_get_thread_num();")
        self._emit("// Distribute tiles evenly; excess threads idle")
        self._emit("int64_t tiles_per = ntiles / nthreads;")
        self._emit("int64_t extra = ntiles % nthreads;")
        self._emit("int64_t my_tiles = tiles_per + (tid < extra ? 1 : 0);")
        self._emit("int64_t n_start = (tid < extra ? tid * (tiles_per + 1)")
        self._emit("                               : extra * (tiles_per + 1) + (tid - extra) * tiles_per) * TILE;")
        self._emit("int64_t n_end = n_start + my_tiles * TILE;")
        self._emit("if (n_start < n_end) {")
        self._indent_inc()
        self._emit("gemv_tiled_chunk(x, W, bias, y, K, N, n_start, n_end);")
        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")
        self._emit("")

    # ------------------------------------------------------------------
    # GEMM microkernel (template)
    # ------------------------------------------------------------------

    def _emit_gemm_microkernel(self, stage: StepStage) -> None:
        """Emit template<int RM, int RN> gemm_ukernel from the GEMM stage ops."""
        dataflow = stage.dataflow_order  # [m, n, k]
        m_var = dataflow[0]
        n_var = dataflow[1]
        k_var = dataflow[2]
        VL = n_var.step  # 16

        self._emit("// Register-blocked GEMM micro-kernel: RM rows x RN vectors")
        self._emit("template <int RM, int RN>")
        self._emit("static inline void gemm_ukernel(")
        self._emit("    float* __restrict__ C, const float* __restrict__ A,")
        self._emit("    const float* __restrict__ B, const float* __restrict__ bias,")
        self._emit("    int64_t K, int64_t N, int64_t n_offset) {")
        self._indent_inc()
        self._emit(f"constexpr int VL = {VL};")

        vt = self.isa["vec_type"]
        loadu = self.isa["loadu"]
        storeu = self.isa["storeu"]
        set1 = self.isa["set1"]
        setzero = self.isa["setzero"]
        fmadd = self.isa["fmadd"]

        # Accumulator init from bias (visit init → bias load)
        # Using template parameters RM, RN as loop bounds
        self._emit(f"{vt} acc[RM][RN];")
        self._emit("for (int i = 0; i < RM; i++)")
        self._indent_inc()
        self._emit("for (int r = 0; r < RN; r++)")
        self._indent_inc()
        self._emit(f"acc[i][r] = bias ? {loadu}(bias + n_offset + r * VL)")
        self._emit(f"                 : {setzero}();")
        self._indent_dec()
        self._indent_dec()

        # k-loop: visit in1 (A broadcast), in2 (B vector load), fmadd
        self._emit("for (int64_t kk = 0; kk < K; kk++) {")
        self._indent_inc()

        # Visit in2: B vector loads
        self._emit(f"{vt} Bv[RN];")
        self._emit("for (int r = 0; r < RN; r++)")
        self._indent_inc()
        self._emit(f"Bv[r] = {loadu}(B + kk * N + n_offset + r * VL);")
        self._indent_dec()

        # Prefetch next k iteration's weight data into L1
        self._emit("if (kk + 1 < K) {")
        self._indent_inc()
        self._emit("for (int r = 0; r < RN; r++)")
        self._indent_inc()
        self._emit("_mm_prefetch((const char*)(B + (kk+1) * N + n_offset + r * VL), _MM_HINT_T0);")
        self._indent_dec()
        self._indent_dec()
        self._emit("}")

        # Visit in1 + fmadd body: for each row, broadcast A scalar, fmadd with Bv
        self._emit("for (int i = 0; i < RM; i++) {")
        self._indent_inc()
        self._emit(f"{vt} a = {set1}(A[i * K + kk]);")
        self._emit("for (int r = 0; r < RN; r++)")
        self._indent_inc()
        self._emit(f"acc[i][r] = {fmadd}(a, Bv[r], acc[i][r]);")
        self._indent_dec()
        self._indent_dec()
        self._emit("}")

        self._indent_dec()
        self._emit("}")

        # Store: visit LinearStore
        self._emit("for (int i = 0; i < RM; i++)")
        self._indent_inc()
        self._emit("for (int r = 0; r < RN; r++)")
        self._indent_inc()
        self._emit(f"{storeu}(C + i * N + n_offset + r * VL, acc[i][r]);")
        self._indent_dec()
        self._indent_dec()

        self._indent_dec()
        self._emit("}")
        self._emit("")

    # ------------------------------------------------------------------
    # OpenMP GEMM wrapper
    # ------------------------------------------------------------------

    def _emit_omp_gemm_wrapper(self, stage: StepStage) -> None:
        """Emit avx512_omp_gemm with RM-aligned chunk partitioning over M."""
        dataflow = stage.dataflow_order  # [m, n, k]
        m_var = dataflow[0]
        n_var = dataflow[1]
        RM = m_var.register_block
        RN = n_var.register_block
        VL = n_var.step

        gemm_fn = self.isa["gemm_fn"]
        self._emit("// OpenMP parallel GEMM: C[M,N] = A[M,K] * B[K,N] + bias[N]")
        self._emit(f"static void {gemm_fn}(")
        self._emit("    const float* A, const float* B, const float* bias,")
        self._emit("    float* C, int64_t M, int64_t K, int64_t N) {")
        self._indent_inc()
        self._emit(f"constexpr int RM = {RM}, RN = {RN}, VL = {VL};")
        self._emit("#pragma omp parallel")
        self._emit("{")
        self._indent_inc()
        self._emit("int nthreads = omp_get_num_threads();")
        self._emit("int tid = omp_get_thread_num();")
        self._emit("int64_t base = (M / nthreads / RM) * RM;")
        self._emit("int64_t extra = M - base * nthreads;")
        self._emit("int64_t extra_threads = extra / RM;")
        self._emit("int64_t m_start, m_end;")
        self._emit("if (tid < extra_threads) {")
        self._indent_inc()
        self._emit("int64_t chunk = base + RM;")
        self._emit("m_start = tid * chunk;")
        self._emit("m_end = m_start + chunk;")
        self._indent_dec()
        self._emit("} else {")
        self._indent_inc()
        self._emit("m_start = extra_threads * (base + RM) + (tid - extra_threads) * base;")
        self._emit("m_end = m_start + base;")
        self._indent_dec()
        self._emit("}")
        self._emit("if (tid == nthreads - 1) m_end = M;")

        # Main RM-tiled loop with RN-tiled inner loop
        self._emit("int64_t ii = m_start;")
        self._emit("for (; ii + RM <= m_end; ii += RM) {")
        self._indent_inc()
        self._emit("for (int64_t jj = 0; jj + RN*VL <= N; jj += RN*VL)")
        self._indent_inc()
        self._emit("gemm_ukernel<RM, RN>(C + ii*N, A + ii*K, B, bias, K, N, jj);")
        self._indent_dec()
        # Remainder columns (RN=1 tail)
        self._emit("for (int64_t jj = (N/(RN*VL))*(RN*VL); jj + VL <= N; jj += VL)")
        self._indent_inc()
        self._emit("gemm_ukernel<RM, 1>(C + ii*N, A + ii*K, B, bias, K, N, jj);")
        self._indent_dec()
        self._indent_dec()
        self._emit("}")
        # Remainder rows (RM=1 tail)
        self._emit("for (; ii < m_end; ii++) {")
        self._indent_inc()
        self._emit("for (int64_t jj = 0; jj + RN*VL <= N; jj += RN*VL)")
        self._indent_inc()
        self._emit("gemm_ukernel<1, RN>(C + ii*N, A + ii*K, B, bias, K, N, jj);")
        self._indent_dec()
        self._emit("for (int64_t jj = (N/(RN*VL))*(RN*VL); jj + VL <= N; jj += VL)")
        self._indent_inc()
        self._emit("gemm_ukernel<1, 1>(C + ii*N, A + ii*K, B, bias, K, N, jj);")
        self._indent_dec()
        self._indent_dec()
        self._emit("}")

        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")
        self._emit("")

    # ------------------------------------------------------------------
    # GELU emission
    # ------------------------------------------------------------------

    def _emit_gelu_vectorized(self, ptr_expr: str, size_expr: str) -> None:
        """Emit vectorized GELU on a contiguous float buffer.

        Args:
            ptr_expr: C++ expression for the float* pointer (e.g., "h_ptr")
            size_expr: C++ expression for the number of elements (e.g., "K")
        """
        vt = self.isa["vec_type"]
        VL = self.isa["VL"]
        set1 = self.isa["set1"]
        loadu = self.isa["loadu"]
        storeu = self.isa["storeu"]
        mul = self.isa["mul"]
        add = self.isa["add"]
        fmadd = self.isa["fmadd"]
        tanh_fn = self.isa["tanh_fn"]

        self._emit(f"const {vt} sqrt2overpi = {set1}(0.7978845608028654f);")
        self._emit(f"const {vt} gelu_coeff  = {set1}(0.044715f);")
        self._emit(f"const {vt} half_v      = {set1}(0.5f);")
        self._emit(f"const {vt} one_v       = {set1}(1.0f);")
        self._emit("")
        self._emit("int64_t k = 0;")
        self._emit(f"for (; k + {VL - 1} < {size_expr}; k += {VL}) {{")
        self._indent_inc()
        self._emit(f"{vt} v    = {loadu}({ptr_expr} + k);")
        self._emit(f"{vt} v2   = {mul}(v, v);")
        self._emit(f"{vt} v3   = {mul}(v2, v);")
        self._emit(f"{vt} inner = {fmadd}(gelu_coeff, v3, v);")
        self._emit(f"{vt} targ = {mul}(sqrt2overpi, inner);")
        self._emit(f"{vt} tval = {tanh_fn}(targ);")
        self._emit(f"{vt} res  = {mul}(half_v, {mul}(v, {add}(one_v, tval)));")
        self._emit(f"{storeu}({ptr_expr} + k, res);")
        self._indent_dec()
        self._emit("}")
        # Scalar tail
        self._emit(f"for (; k < {size_expr}; k++) {{")
        self._indent_inc()
        self._emit(f"float xv = {ptr_expr}[k];")
        self._emit("float x3 = xv * xv * xv;")
        self._emit("float ta = 0.7978845608028654f * (xv + 0.044715f * x3);")
        self._emit(f"{ptr_expr}[k] = 0.5f * xv * (1.0f + std::tanh(ta));")
        self._indent_dec()
        self._emit("}")

    def _emit_gelu_parallel(self, ptr_expr: str, m_expr: str, k_expr: str) -> None:
        """Emit parallelized GELU over M rows."""
        vt = self.isa["vec_type"]
        VL = self.isa["VL"]
        set1 = self.isa["set1"]
        loadu = self.isa["loadu"]
        storeu = self.isa["storeu"]
        mul = self.isa["mul"]
        add = self.isa["add"]
        fmadd = self.isa["fmadd"]
        tanh_fn = self.isa["tanh_fn"]

        self._emit(f"const {vt} sqrt2overpi = {set1}(0.7978845608028654f);")
        self._emit(f"const {vt} gelu_coeff  = {set1}(0.044715f);")
        self._emit(f"const {vt} half_v      = {set1}(0.5f);")
        self._emit(f"const {vt} one_v       = {set1}(1.0f);")
        self._emit("")
        self._emit(f"#pragma omp parallel for if({m_expr} > 1)")
        self._emit(f"for (int64_t m = 0; m < {m_expr}; m++) {{")
        self._indent_inc()
        self._emit(f"float* row = {ptr_expr} + m * {k_expr};")
        self._emit("int64_t k = 0;")
        self._emit(f"for (; k + {VL - 1} < {k_expr}; k += {VL}) {{")
        self._indent_inc()
        self._emit(f"{vt} v     = {loadu}(row + k);")
        self._emit(f"{vt} v2    = {mul}(v, v);")
        self._emit(f"{vt} v3    = {mul}(v2, v);")
        self._emit(f"{vt} inner = {fmadd}(gelu_coeff, v3, v);")
        self._emit(f"{vt} targ  = {mul}(sqrt2overpi, inner);")
        self._emit(f"{vt} tval  = {tanh_fn}(targ);")
        self._emit(f"{vt} res   = {mul}(half_v, {mul}(v, {add}(one_v, tval)));")
        self._emit(f"{storeu}(row + k, res);")
        self._indent_dec()
        self._emit("}")
        self._emit(f"for (; k < {k_expr}; k++) {{")
        self._indent_inc()
        self._emit("float xv = row[k];")
        self._emit("float x3 = xv * xv * xv;")
        self._emit("float ta = 0.7978845608028654f * (xv + 0.044715f * x3);")
        self._emit("row[k] = 0.5f * xv * (1.0f + std::tanh(ta));")
        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _emit_entry_point(self) -> None:
        """Emit the main function: dispatch M==1 (GEMV) vs M>1 (GEMM)."""
        prog = self.decode  # use GEMV program for signature (same params)
        func_name = prog.name + "_step"
        params = ", ".join(f"torch::Tensor {p}" for p in prog.tensor_params)

        self._emit(f"torch::Tensor {func_name}(")
        self._emit(f"    {params}) {{")
        self._indent_inc()
        self._emit("")

        # Contiguity checks
        self._emit('TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");')
        self._emit('TORCH_CHECK(W_fc.is_contiguous(), "W_fc must be contiguous");')
        self._emit('TORCH_CHECK(W_proj.is_contiguous(), "W_proj must be contiguous");')
        self._emit("")

        self._emit("const int64_t M = hidden_states.size(0);")
        self._emit("")

        # M == 1 branch (GEMV path)
        self._emit("if (M == 1) {")
        self._indent_inc()
        self._emit_gemv_path()
        self._indent_dec()
        self._emit("}")
        self._emit("")

        # M > 1 branch (GEMM path)
        self._emit_gemm_path()

        self._indent_dec()
        self._emit("}")
        self._emit("")

    def _emit_gemv_path(self) -> None:
        """Emit the M==1 GEMV path body inside the entry point."""
        # Dimension extraction from dim_bindings
        for dim_name, (tensor_param, dim_idx) in self.decode.dim_bindings.items():
            self._emit(f"const int64_t {dim_name} = {tensor_param}.size({dim_idx});")
        self._emit("")

        # Data pointers
        self._emit("const float* x_ptr = hidden_states.data_ptr<float>();")
        self._emit("const float* wfc_ptr = W_fc.data_ptr<float>();")
        self._emit("const float* bfc_ptr = b_fc.data_ptr<float>();")
        self._emit("const float* wproj_ptr = W_proj.data_ptr<float>();")
        self._emit("const float* bproj_ptr = b_proj.data_ptr<float>();")
        self._emit("")

        # Walk GEMV stages sequentially
        for i, stage in enumerate(self.decode.stages):
            if stage.stage_type == "gemv":
                if i == 0:
                    # GEMV1: h = x * W_fc + b_fc
                    self._emit("// GEMV1: h = x * W_fc + b_fc")
                    self._emit("auto h = torch::empty({1, K}, hidden_states.options());")
                    self._emit("float* h_ptr = h.data_ptr<float>();")
                    self._emit(f"{self.isa['gemv_fn']}(x_ptr, wfc_ptr, bfc_ptr, h_ptr, D, K);")
                else:
                    # GEMV2: out = h * W_proj + b_proj
                    self._emit("// GEMV2: out = h * W_proj + b_proj")
                    self._emit("auto output = torch::empty({1, D}, hidden_states.options());")
                    self._emit("float* out_ptr = output.data_ptr<float>();")
                    self._emit(f"{self.isa['gemv_fn']}(h_ptr, wproj_ptr, bproj_ptr, out_ptr, K, D);")
                self._emit("")
            elif stage.stage_type == "gelu":
                self._emit("// GELU activation (in-place on h)")
                self._emit_gelu_vectorized("h_ptr", "K")
                self._emit("")

        self._emit("return output;")

    def _emit_gemm_path(self) -> None:
        """Emit the M>1 GEMM path body inside the entry point."""
        self._emit("// M > 1 (prefill) — GEMM with fused bias")
        # Dimension extraction
        for dim_name, (tensor_param, dim_idx) in self.prefill.dim_bindings.items():
            if dim_name == "M":
                continue  # already extracted above
            self._emit(f"const int64_t {dim_name} = {tensor_param}.size({dim_idx});")
        self._emit("")

        # Data pointers
        self._emit("const float* x_ptr = hidden_states.data_ptr<float>();")
        self._emit("const float* wfc_ptr = W_fc.data_ptr<float>();")
        self._emit("const float* bfc_ptr = b_fc.data_ptr<float>();")
        self._emit("const float* wproj_ptr = W_proj.data_ptr<float>();")
        self._emit("const float* bproj_ptr = b_proj.data_ptr<float>();")
        self._emit("")

        # Walk GEMM stages sequentially
        for i, stage in enumerate(self.prefill.stages):
            if stage.stage_type == "gemm":
                if i == 0:
                    self._emit("// GEMM1: h = hidden_states * W_fc + b_fc")
                    self._emit("auto h = torch::empty({M, K}, hidden_states.options());")
                    self._emit("float* h_ptr = h.data_ptr<float>();")
                    self._emit(f"{self.isa['gemm_fn']}(x_ptr, wfc_ptr, bfc_ptr, h_ptr, M, D, K);")
                else:
                    self._emit("// GEMM2: output = h * W_proj + b_proj")
                    self._emit("auto output = torch::empty({M, D}, hidden_states.options());")
                    self._emit("float* out_ptr = output.data_ptr<float>();")
                    self._emit(f"{self.isa['gemm_fn']}(h_ptr, wproj_ptr, bproj_ptr, out_ptr, M, K, D);")
                self._emit("")
            elif stage.stage_type == "gelu":
                self._emit("// GELU activation (in-place, parallelized over M)")
                self._emit_gelu_parallel("h_ptr", "M", "K")
                self._emit("")

        self._emit("return output;")

    # ------------------------------------------------------------------
    # TORCH_LIBRARY registration
    # ------------------------------------------------------------------

    def _emit_registration(self) -> None:
        func_name = self.decode.name + "_step"
        self._emit("TORCH_LIBRARY_FRAGMENT(step_ops, m) {")
        self._indent_inc()
        self._emit(f'm.def("{func_name}", {func_name});')
        self._indent_dec()
        self._emit("}")
        self._emit("")
