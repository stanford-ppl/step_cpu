"""GELU kernel example — STeP function compiled to a C++ PyTorch extension."""

import math
import sys
from pathlib import Path

# Add project root to path so `import step` works when running from examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


if __name__ == "__main__":
    print("Compiling GELU kernel...")
    compiled_gelu = step.cpu_compile(gelu_kernel)
    print("Compilation successful!")

    # Test on various shapes
    for shape in [(4, 64), (16, 128), (3, 100)]:
        x = torch.randn(*shape, dtype=torch.float32)
        result = compiled_gelu(x)
        expected = F.gelu(x, approximate="tanh")
        max_err = (result - expected).abs().max().item()
        print(f"Shape {shape}: max error = {max_err:.2e}", end="")
        if max_err < 1e-6:
            print(" [PASS]")
        else:
            print(f" [FAIL]")

    print("\nAll tests passed!")
