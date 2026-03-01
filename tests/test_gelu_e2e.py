"""End-to-end test: compile + run GELU, check numerics."""

import math

import torch
import torch.nn.functional as F

import step


def gelu_kernel(input):
    vec = [1, 32]
    x = step.tensor_to_stream(input, vec)

    c_sqrt = math.sqrt(2.0 / math.pi)
    c_pow = 0.044715

    x3 = step.UnaryMap(x, lambda a: torch.pow(a, 3.0))
    x_in = step.BinaryMap(x, x3, lambda a, b: a + c_pow * b)
    t_in = step.UnaryMap(x_in, lambda a: c_sqrt * a)
    t = step.UnaryMap(t_in, lambda a: torch.tanh(a))
    onep = step.UnaryMap(t, lambda a: 1.0 + a)
    y = step.BinaryMap(x, onep, lambda a, b: 0.5 * a * b)

    return step.stream_to_tensor(y, like_tensor=input)


# Compile once at module level
compiled_gelu = step.cpu_compile(gelu_kernel)


class TestGeluE2E:
    def _check(self, shape: tuple[int, ...]):
        x = torch.randn(*shape, dtype=torch.float32)
        result = compiled_gelu(x)
        expected = F.gelu(x, approximate="tanh")
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_shape_4x64(self):
        self._check((4, 64))

    def test_shape_16x128(self):
        self._check((16, 128))

    def test_shape_3x100_non_divisible(self):
        """Non-divisible by tile size 32."""
        self._check((3, 100))

    def test_shape_1x1(self):
        """Edge case: single element."""
        self._check((1, 1))

    def test_shape_1x32_exact_tile(self):
        """Exactly one tile."""
        self._check((1, 32))

    def test_shape_large(self):
        """Larger tensor."""
        self._check((64, 256))
