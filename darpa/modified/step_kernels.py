from typing import Callable, List, Tuple, Dict, Any, Union
from abc import ABC, abstractmethod
import torch


# ============================================================
# STeP IR Classes for AVX512+OpenMP Code Generation
# ============================================================


class IndexVar:
    """Loop index variable for a STeP dataflow dimension.

    Attributes:
        name: Variable name (used in generated C++ loop variable).
        size: Loop bound. Can be an int literal or a str C++ expression
              (e.g., "K") that will be resolved from dim_bindings at codegen time.
        step: Tile size per iteration. For AVX512 float32, step=16 means one
              __m512 register width. step=1 means scalar access.
        parallelized: If True, the codegen wraps this dimension in an
                      OpenMP parallel region.
        register_block: Number of AVX512 registers to unroll per tile.
                        For GEMV, n has register_block=4 → RN=4 (64 floats/tile).
                        For GEMM, m has register_block=4 → RM=4 rows,
                                  n has register_block=4 → RN=4 vectors.
    """

    def __init__(
        self,
        name: str,
        size: Union[int, str],
        step: int = 1,
        parallelized: bool = False,
        register_block: int = 1,
    ):
        self.name = name
        self.size = size
        self.step = step
        self.parallelized = parallelized
        self.register_block = register_block

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"size={self.size!r}", f"step={self.step}"]
        if self.parallelized:
            parts.append("parallelized=True")
        if self.register_block != 1:
            parts.append(f"register_block={self.register_block}")
        return f"IndexVar({', '.join(parts)})"


class InputBuffer(ABC):
    """Will be lowered to 'const float* __restrict__ <buffer_name>'"""

    pass


class OutputBuffer(ABC):
    """Will be lowered to 'float* __restrict__ <buffer_name>'"""

    pass


class Buffer(InputBuffer, OutputBuffer):
    """A tensor annotated with index variables that describe how it is accessed.

    Each index variable maps 1:1 to a dimension of the tensor in order.
    The IndexVar.step for each dimension tells the codegen the access granularity:
      - step=1 on a dim → scalar access (broadcast to vector via _mm512_set1_ps)
      - step=16 on a dim → one AVX512 vector load per register_block

    Attributes:
        index_vars: List of IndexVar, one per tensor dimension.
        torch_tensor: The underlying torch.Tensor (or None at trace time).
        name: Optional name for the buffer in generated code (e.g., "x", "W", "bias").
    """

    def __init__(
        self,
        index_vars: List[IndexVar],
        torch_tensor: torch.Tensor = None,
        name: str = None,
    ):
        self.index_vars = index_vars
        self.torch_tensor = torch_tensor
        self.name = name


class StepOps(ABC):
    @abstractmethod
    def stream_shape(self) -> List[IndexVar]:
        pass

    def tile_shape(self) -> List[int]:
        pass


class LinearLoad(StepOps):
    """Load data from a buffer.

    The iter_space determines which loop levels this load lives in.
    The buffer's index_vars and their step sizes determine whether the
    codegen emits a scalar load+broadcast or vector loads.

    If iter_space is omitted, it defaults to the buffer's own index_vars.
    """

    def __init__(
        self, in_buff: InputBuffer, iter_space: List[IndexVar] = None
    ):
        self.in_buff = in_buff
        self.iter_space = (
            iter_space if iter_space is not None else in_buff.index_vars
        )

    def stream_shape(self) -> List[IndexVar]:
        return self.iter_space

    def tile_shape(self) -> List[int]:
        return [var.step for var in self.in_buff.index_vars]


class LinearStore(StepOps):
    """Store data to an output buffer.

    The iter_space determines which loop level the store is placed at.
    Data dependency on `input` means the store goes after the input's
    producing loop completes.
    """

    def __init__(
        self, input: StepOps, out_buff: OutputBuffer, iter_space: List[IndexVar]
    ):
        self.input = input
        self.out_buff = out_buff
        self.iter_space = iter_space

    def stream_shape(self) -> List[IndexVar]:
        return []

    def tile_shape(self) -> List[int]:
        return []


class BinaryMapAccum(StepOps):
    """Binary map with accumulation over inner dimensions.

    For GEMV/GEMM, this represents the fused multiply-accumulate:
      acc = init  (before inner loop)
      for each inner tile:
          acc = func(in1_tile, in2_tile, acc)

    The `rank` parameter specifies how many innermost dataflow dimensions
    are accumulated over. For rank=1 with dataflow_order=[n, k]:
      - init (bias) is loaded at the n-loop level, BEFORE the k-loop
      - body (fmadd) executes INSIDE the k-loop
      - result is available AFTER the k-loop closes

    Attributes:
        in1: First input operand (e.g., activation scalar).
        in2: Second input operand (e.g., weight vector).
        init: Accumulator initializer (e.g., bias vector).
        func: Lambda (a, b, c) -> result. For fmadd: lambda a, b, c: a * b + c
        rank: Number of innermost dimensions to accumulate over.
    """

    def __init__(
        self,
        in1: StepOps,
        in2: StepOps,
        init: StepOps,
        func: Callable[[Any, Any, Any], Any],
        rank: int,
    ):
        self.in1 = in1
        self.in2 = in2
        self.init = init
        self.func = func
        self.rank = rank

    def stream_shape(self) -> List[IndexVar]:
        return self.in1.stream_shape()[: -self.rank]

    def tile_shape(self) -> List[int]:
        return self.init.tile_shape()


class UnaryMap(StepOps):
    """Apply a unary function element-wise to each tile in the stream.

    The codegen inspects `func` to determine the AVX512 lowering.
    For example, `torch.nn.functional.gelu` maps to the fast_tanh_avx512
    Pade rational approximation pattern.

    Attributes:
        input: The input operand.
        func: A callable (e.g., torch.nn.functional.gelu, lambda a: a * 2).
        iter_space: Index variables determining loop placement.
    """

    def __init__(
        self,
        input: StepOps,
        func: Callable
    ):
        self.input = input
        self.func = func

    def stream_shape(self) -> List[IndexVar]:
        return self.input.stream_shape()    

    def tile_shape(self) -> List[int]:
        return self.input.tile_shape()


# ============================================================
# Stage and Program containers
# ============================================================


class StepStage:
    """One computation stage within a StepProgram.

    Each stage has its own loop nest (defined by dataflow_order) and
    a list of STeP operations. The codegen processes stages sequentially.

    Attributes:
        stage_type: 'gemv', 'gemm', or 'gelu'.
        ops: Ordered list of STeP operations in this stage.
        dataflow_order: List of IndexVars defining the loop nesting
                        (outer to inner).
        buffers: Named buffers used in this stage.
    """

    def __init__(
        self,
        stage_type: str,
        ops: List[StepOps],
        dataflow_order: List[IndexVar],
        buffers: Dict[str, Buffer] = None,
    ):
        self.stage_type = stage_type
        self.ops = ops
        self.dataflow_order = dataflow_order
        self.buffers = buffers or {}

    def __repr__(self) -> str:
        return (
            f"StepStage(type={self.stage_type!r}, "
            f"ops={len(self.ops)}, "
            f"dataflow={[iv.name for iv in self.dataflow_order]})"
        )


class StepProgram:
    """Container for a complete STeP program with multiple stages.

    A general-purpose program representation: an ordered list of stages
    that the codegen processes sequentially. Any branching logic
    (e.g., M=1 vs M>1) is handled at a higher level by constructing
    separate StepPrograms for each path.

    Attributes:
        name: Function name for the generated C++ entry point.
        tensor_params: Ordered list of input tensor parameter names
                       (determines the C++ function signature).
        dim_bindings: Maps symbolic dimension names to (tensor_param, dim_index)
                      pairs for runtime size extraction.
                      E.g., {"D": ("hidden_states", 1), "K": ("W_fc", 1)}
        stages: Ordered list of StepStages executed sequentially.
    """

    def __init__(
        self,
        name: str,
        tensor_params: List[str],
        dim_bindings: Dict[str, Tuple[str, int]] = None,
    ):
        self.name = name
        self.tensor_params = tensor_params
        self.dim_bindings = dim_bindings or {}
        self.stages: List[StepStage] = []

    def __repr__(self) -> str:
        return (
            f"StepProgram(name={self.name!r}, "
            f"params={self.tensor_params}, "
            f"stages={len(self.stages)})"
        )


# ============================================================
# Helper: build a GEMV stage from dataflow + buffers
# ============================================================


def _build_gemv_stage(
    x_buf: Buffer,
    w_buf: Buffer,
    bias_buf: Buffer,
    out_buf: Buffer,
    n: IndexVar,
    k: IndexVar,
) -> StepStage:
    """Construct a GEMV StepStage: y[N] = x[K] @ W[K,N] + bias[N].

    The dataflow order is [n, k]: outer loop over N (parallelized),
    inner loop over K (reduction).

    Operation placement (derived from iter_space + data dependencies):
      n-loop level, BEFORE k-loop:
        - LinearLoad(bias, [n])        → accumulator init
      k-loop level (inside both n and k):
        - LinearLoad(x, [k])           → broadcast scalar
        - LinearLoad(W, [n, k])        → vector load
        - BinaryMapAccum body           → fmadd
      n-loop level, AFTER k-loop:
        - LinearStore(result, out, [n]) → vector store
    """
    x_load = LinearLoad(x_buf, [k])
    w_load = LinearLoad(w_buf, [n, k])
    bias_load = LinearLoad(bias_buf, [n])

    accum = BinaryMapAccum(
        in1=x_load,
        in2=w_load,
        init=bias_load,
        func=lambda a, b, c: a * b + c,
        rank=1,
    )

    store = LinearStore(accum, out_buf, [n])

    return StepStage(
        stage_type="gemv",
        ops=[x_load, w_load, bias_load, accum, store],
        dataflow_order=[n, k],
        buffers={
            "x": x_buf,
            "W": w_buf,
            "bias": bias_buf,
            "y": out_buf,
        },
    )


def _build_gelu_stage(
    buf: Buffer,
    k_var: IndexVar,
    m_var: IndexVar = None,
) -> StepStage:
    """Construct a GELU StepStage: buf = gelu(buf) in-place.

    For M=1 (GEMV path): single loop over k_var.
    For M>1 (GEMM path): outer parallelized loop over m_var, inner loop over k_var.

    The stage is: load → gelu → store (back to same buffer, in-place).
    The codegen recognizes torch.nn.functional.gelu and emits the AVX512
    fast_tanh_avx512 Pade approximation pattern as below:
    
        // Fused Pade-tanh GELU on h (3072 floats, fits in L1)
        const __m512 sqrt2overpi = _mm512_set1_ps(0.7978845608028654f);
        const __m512 gelu_coeff  = _mm512_set1_ps(0.044715f);
        const __m512 half_v      = _mm512_set1_ps(0.5f);
        const __m512 one_v       = _mm512_set1_ps(1.0f);

        int64_t k = 0;
        for (; k + 15 < K; k += 16) {
            __m512 v    = _mm512_loadu_ps(h_ptr + k);
            __m512 v2   = _mm512_mul_ps(v, v);
            __m512 v3   = _mm512_mul_ps(v2, v);
            __m512 inner = _mm512_fmadd_ps(gelu_coeff, v3, v);
            __m512 targ = _mm512_mul_ps(sqrt2overpi, inner);
            __m512 tval = fast_tanh_avx512(targ);
            __m512 res  = _mm512_mul_ps(half_v, _mm512_mul_ps(v, _mm512_add_ps(one_v, tval)));
            _mm512_storeu_ps(h_ptr + k, res);
        }
        for (; k < K; k++) {
            float xv = h_ptr[k];
            float x3 = xv * xv * xv;
            float ta = 0.7978845608028654f * (xv + 0.044715f * x3);
            h_ptr[k] = 0.5f * xv * (1.0f + std::tanh(ta));
        }
    """
    h = LinearLoad(buf, [k_var])
    gelu = UnaryMap(
        input=h,
        func=torch.nn.functional.gelu,
    )
    store = LinearStore(gelu, buf, [k_var])

    if m_var is not None:
        dataflow_order = [m_var, k_var]
    else:
        dataflow_order = [k_var]

    return StepStage(
        stage_type="gelu",
        ops=[h, gelu, store],
        dataflow_order=dataflow_order,
        buffers={"h": buf},
    )


def _build_gemm_stage(
    a_buf: Buffer,
    w_buf: Buffer,
    bias_buf: Buffer,
    out_buf: Buffer,
    m: IndexVar,
    n: IndexVar,
    k: IndexVar,
) -> StepStage:
    """Construct a GEMM StepStage: C[M,N] = A[M,K] @ W[K,N] + bias[N].

    The dataflow order is [m, n, k]: parallelized over m (rows),
    tiled over n (columns with register blocking), reduction over k.

    For the GEMM microkernel with RM x RN register blocking:
      - m has register_block=RM (rows per tile)
      - n has register_block=RN (AVX vectors per tile)
      - k is the reduction dimension (step=1)
    """
    a_load = LinearLoad(a_buf, [m, k])
    w_load = LinearLoad(w_buf, [k, n])
    bias_load = LinearLoad(bias_buf, [n])

    accum = BinaryMapAccum(
        in1=a_load,
        in2=w_load,
        init=bias_load,
        func=lambda a, b, c: a * b + c,
        rank=1,
    )

    store = LinearStore(accum, out_buf, [m, n])

    return StepStage(
        stage_type="gemm",
        ops=[a_load, w_load, bias_load, accum, store],
        dataflow_order=[m, n, k],
        buffers={
            "A": a_buf,
            "W": w_buf,
            "bias": bias_buf,
            "C": out_buf,
        },
    )


# ============================================================
# build_gpt2_mlp_program: the full MLP in STeP
# ============================================================


def build_gpt2_mlp_gemv_program() -> StepProgram:
    """Build the StepProgram for the GPT2 MLP GEMV path (M=1).

    The MLP computes:
        h = x @ W_fc + b_fc          (GEMV stage 1)
        h = gelu(h)                   (GELU stage 2)
        output = h @ W_proj + b_proj  (GEMV stage 3)

    Tensor parameter order matches _build_gpt2mlp_fused6:
        hidden_states [1, D], W_fc [D, K], b_fc [K],
        W_proj [K, D], b_proj [D]

    Symbolic dimensions:
        D = hidden_states.size(1)   (768 for GPT2)
        K = W_fc.size(1)            (3072 for GPT2)
    """
    program = StepProgram(
        name="gpt2_mlp_fused6",
        tensor_params=["hidden_states", "W_fc", "b_fc", "W_proj", "b_proj"],
        dim_bindings={
            "D": ("hidden_states", 1),
            "K": ("W_fc", 1),
        },
    )

    # Stage 1: GEMV h = x[1,D] @ W_fc[D,K] + b_fc[K]
    gemv1_n = IndexVar("n", size="K", step=16, parallelized=True, register_block=4)
    gemv1_k = IndexVar("k", size="D", step=1)

    gemv1_x = Buffer([gemv1_k], name="x")             # hidden_states: accessed as x[k], scalar
    gemv1_w = Buffer([gemv1_k, gemv1_n], name="wfc")   # W_fc[D,K]: accessed as W[k, n], vector
    gemv1_bias = Buffer([gemv1_n], name="bfc")          # b_fc[K]: accessed as bias[n], vector
    gemv1_out = Buffer([gemv1_n], name="h")             # h[K]: output, vector

    program.stages.append(
        _build_gemv_stage(gemv1_x, gemv1_w, gemv1_bias, gemv1_out, gemv1_n, gemv1_k)
    )

    # Stage 2: GELU h = gelu(h) in-place
    gelu_k = IndexVar("k", size="K", step=16)
    gelu_buf = Buffer([gelu_k], name="h")

    program.stages.append(_build_gelu_stage(gelu_buf, gelu_k))

    # Stage 3: GEMV out = h[1,K] @ W_proj[K,D] + b_proj[D]
    gemv2_n = IndexVar("n", size="D", step=16, parallelized=True, register_block=4)
    gemv2_k = IndexVar("k", size="K", step=1)

    gemv2_x = Buffer([gemv2_k], name="h")               # h[K]: accessed as x[k], scalar
    gemv2_w = Buffer([gemv2_k, gemv2_n], name="wproj")   # W_proj[K,D]: accessed as W[k, n], vector
    gemv2_bias = Buffer([gemv2_n], name="bproj")          # b_proj[D]: accessed as bias[n], vector
    gemv2_out = Buffer([gemv2_n], name="out")             # output[D], vector

    program.stages.append(
        _build_gemv_stage(gemv2_x, gemv2_w, gemv2_bias, gemv2_out, gemv2_n, gemv2_k)
    )

    return program


def build_gpt2_mlp_gemm_program() -> StepProgram:
    """Build the StepProgram for the GPT2 MLP GEMM path (M>1).

    The MLP computes:
        h = A @ W_fc + b_fc          (GEMM stage 1)
        h = gelu(h)                   (GELU stage 2)
        output = h @ W_proj + b_proj  (GEMM stage 3)

    Tensor parameter order matches _build_gpt2mlp_fused6:
        hidden_states [M, D], W_fc [D, K], b_fc [K],
        W_proj [K, D], b_proj [D]

    Symbolic dimensions:
        M = hidden_states.size(0)   (batch*seq, runtime)
        D = hidden_states.size(1)   (768 for GPT2)
        K = W_fc.size(1)            (3072 for GPT2)
    """
    program = StepProgram(
        name="gpt2_mlp_fused6",
        tensor_params=["hidden_states", "W_fc", "b_fc", "W_proj", "b_proj"],
        dim_bindings={
            "M": ("hidden_states", 0),
            "D": ("hidden_states", 1),
            "K": ("W_fc", 1),
        },
    )

    # Stage 1: GEMM h[M,K] = A[M,D] @ W_fc[D,K] + b_fc[K]
    gemm1_m = IndexVar("m", size="M", step=1, parallelized=True, register_block=4)
    gemm1_n = IndexVar("n", size="K", step=16, register_block=4)
    gemm1_k = IndexVar("k", size="D", step=1)

    gemm1_a = Buffer([gemm1_m, gemm1_k], name="x")     # hidden_states[M,D]
    gemm1_w = Buffer([gemm1_k, gemm1_n], name="wfc")   # W_fc[D,K]
    gemm1_bias = Buffer([gemm1_n], name="bfc")          # b_fc[K]
    gemm1_out = Buffer([gemm1_m, gemm1_n], name="h")    # h[M,K]

    program.stages.append(
        _build_gemm_stage(
            gemm1_a, gemm1_w, gemm1_bias, gemm1_out,
            gemm1_m, gemm1_n, gemm1_k,
        )
    )

    # Stage 2: GELU h = gelu(h) in-place, parallelized over M
    gelu_m = IndexVar("m", size="M", step=1, parallelized=True)
    gelu_k = IndexVar("k", size="K", step=16)
    gelu_buf = Buffer([gelu_m, gelu_k], name="h")

    program.stages.append(_build_gelu_stage(gelu_buf, gelu_k, m_var=gelu_m))

    # Stage 3: GEMM out[M,D] = h[M,K] @ W_proj[K,D] + b_proj[D]
    gemm2_m = IndexVar("m", size="M", step=1, parallelized=True, register_block=4)
    gemm2_n = IndexVar("n", size="D", step=16, register_block=4)
    gemm2_k = IndexVar("k", size="K", step=1)

    gemm2_a = Buffer([gemm2_m, gemm2_k], name="h")       # h[M,K]
    gemm2_w = Buffer([gemm2_k, gemm2_n], name="wproj")   # W_proj[K,D]
    gemm2_bias = Buffer([gemm2_n], name="bproj")          # b_proj[D]
    gemm2_out = Buffer([gemm2_m, gemm2_n], name="out")    # output[M,D]

    program.stages.append(
        _build_gemm_stage(
            gemm2_a, gemm2_w, gemm2_bias, gemm2_out,
            gemm2_m, gemm2_n, gemm2_k,
        )
    )

    return program


# ============================================================
# GPT2MLPStepWrapper (kept from causal_language_modeling_mlp.py
# for reference; the actual runtime wrapper lives there)
# ============================================================


class GPT2MLPStepWrapper(torch.nn.Module):
    """Drop-in replacement for GPT2MLP using a STeP-compiled C++ kernel.

    In eval mode, uses the compiled kernel for inference.
    In training mode, falls through to the original HuggingFace forward
    (STeP has no autograd support).
    """

    def __init__(self, mlp_module):
        super().__init__()
        self.c_fc = mlp_module.c_fc
        self.c_proj = mlp_module.c_proj
        self.act = mlp_module.act
        self.dropout = mlp_module.dropout

    def _original_forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)

    def forward(self, hidden_states):
        if self.training:
            return self._original_forward(hidden_states)
        shape = hidden_states.shape  # [batch, seq, 768]
        x2d = hidden_states.reshape(shape[0] * shape[1], shape[2]).contiguous()

        _compiled_kernel = compile(self.step_impl)
        out = _compiled_kernel(
            x2d,
            self.c_fc.weight,
            self.c_fc.bias,
            self.c_proj.weight,
            self.c_proj.bias,
        )
        return out.reshape(shape)
