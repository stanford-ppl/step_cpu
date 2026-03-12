from typing import Callable, List, Tuple, Dict, Any
from abc import ABC, abstractmethod
import torch

# from sympy import symbols, Symbol


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

        # print(f"GPT2MLPStepWrapper forward called")
        _compiled_kernel = compile(self.step_impl)
        out = _compiled_kernel(
            x2d,
            self.c_fc.weight,
            self.c_fc.bias,
            self.c_proj.weight,
            self.c_proj.bias,
        )
        return out.reshape(shape)

    def step_impl(
        self, hidden_states, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias
    ):
        def gemv_tiled_chunk(
            hidden_states: InputBuffer,
            weight: InputBuffer,
            bias: InputBuffer,
            output: OutputBuffer,
            n: IndexVar,
            k: IndexVar,
        ):
            x = LinearLoad(
                hidden_states,
            )  # shape is [1] as the InputBuffer is [K] and the step size for K is 1
            # [N,K] ([1])

            W = LinearLoad(weight, [n, k])  # [N,K] ([1,64])
            bias = LinearLoad(bias, [n])  # [N] ([64])

            # in1: [N,K] ([1])
            # in2: [N,K] ([1,64])
            # init: [N] ([64])
            binary_map_accum = BinaryMapAccum(x, W, bias, lambda a, b, c: a * b + c, 1)

            _ = LinearStore(binary_map_accum, output, [n])  # [N] ([64])

        M = hidden_states.shape[0]

        if M == 1:
            # Call the STeP version of avx512_omp_gemv
            D = hidden_states.shape[1] // 768
            K = c_fc_weight.shape[1] // 3072
            y = torch.empty(
                [1, K], dtype=hidden_states.dtype, device=hidden_states.device
            )

            n = IndexVar(
                "n",
                size=K,
                step=16,
                parallelized=True,
            )
            k = IndexVar("k", size=K)
            dataflow_order = [n, k]

            in_buffer = Buffer(
                [k],
                hidden_states,
            )
            w_buffer = Buffer(
                [k, n],
                c_fc_weight,
            )
            bias_buffer = Buffer(
                [n],
                c_fc_bias,
            )
            out_buffer = Buffer(
                [n],
                y,
            )

            gemv_tiled_chunk(in_buffer, w_buffer, bias_buffer, out_buffer, n, k)
        else:
            # Call the STeP version of avx512_omp_gemm
            pass


class IndexVar:
    name: str
    step: int
    size: int
    parallelized: bool

    def __init__(self, name: str, size: int, step: int = 1, parallelized: bool = False):
        self.name = name
        self.step = step
        self.parallelized = parallelized
        self.size = size


class InputBuffer(ABC):
    """Will be lowered to 'const float* __restrict__ <buffer_name>'"""

    pass


class OutputBuffer(ABC):
    """Will be lowered to 'float* __restrict__ <buffer_name>'"""

    pass


class Buffer(InputBuffer, OutputBuffer):
    def __init__(
        self,
        index_var: List[IndexVar],
        torch_tensor: torch.Tensor,
    ):
        self.index_vars = index_var
        self.torch_tensor = torch_tensor


class StepOps(ABC):
    @abstractmethod
    def stream_shape(self) -> List[IndexVar]:
        pass

    def tile_shape(self) -> List[int]:
        pass


class LinearLoad(StepOps):
    in_buff: InputBuffer
    iter_space: List[IndexVar]

    def __init__(self, in_buff: InputBuffer, iter_space: List[IndexVar]):
        self.in_buff = in_buff
        self.iter_space = iter_space

    def stream_shape(self) -> List[IndexVar]:
        return self.iter_space

    def tile_shape(self) -> List[int]:
        return [var.step for var in self.in_buff.index_vars]


class LinearStore(StepOps):
    out_buff: OutputBuffer
    iter_space: List[IndexVar]

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
    """Initialize the accumulator with the value from the init operator after every 'sel.rank' number of inner loops finish.
    The callable here applies a binary operation between in1, in2 and an accumulation function
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
