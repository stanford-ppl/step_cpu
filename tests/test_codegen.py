"""Unit tests for generated C++ code structure."""

import ast
import math

from step.codegen import CppCodegen
from step.ir import (
    BinaryMapOp,
    StepGraph,
    StreamToTensor,
    TensorToStream,
    UnaryMapOp,
    reset_id_counter,
)


def _build_gelu_graph() -> StepGraph:
    """Build the GELU IR graph for testing."""
    reset_id_counter()
    graph = StepGraph("gelu_kernel", ["input"])

    x = TensorToStream("input", [1, 32], tensor_ndim=2)
    graph.add_node(x)

    c_sqrt = math.sqrt(2.0 / math.pi)
    c_pow = 0.044715

    x3_body = ast.parse("torch.pow(a, 3.0)", mode="eval").body
    x3 = UnaryMapOp(x, "x3", x3_body, "a", {})
    graph.add_node(x3)

    x_in_body = ast.parse("a + c_pow * b", mode="eval").body
    x_in = BinaryMapOp(x, x3, "x_in", x_in_body, ("a", "b"), {"c_pow": c_pow})
    graph.add_node(x_in)

    t_in_body = ast.parse("c_sqrt * a", mode="eval").body
    t_in = UnaryMapOp(x_in, "t_in", t_in_body, "a", {"c_sqrt": c_sqrt})
    graph.add_node(t_in)

    t_body = ast.parse("torch.tanh(a)", mode="eval").body
    t = UnaryMapOp(t_in, "t", t_body, "a", {})
    graph.add_node(t)

    onep_body = ast.parse("1.0 + a", mode="eval").body
    onep = UnaryMapOp(t, "onep", onep_body, "a", {})
    graph.add_node(onep)

    y_body = ast.parse("0.5 * a * b", mode="eval").body
    y = BinaryMapOp(x, onep, "y", y_body, ("a", "b"), {})
    graph.add_node(y)

    out = StreamToTensor(y, "input")
    graph.add_node(out)

    return graph


class TestCppCodegen:
    def test_includes(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "#include <torch/extension.h>" in cpp
        assert "#include <torch/library.h>" in cpp

    def test_function_signature(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "torch::Tensor gelu_kernel_step(torch::Tensor input)" in cpp

    def test_contiguity_check(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert 'TORCH_CHECK(input.is_contiguous()' in cpp

    def test_output_allocation(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "auto output = torch::empty_like(input);" in cpp

    def test_dimension_variables(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "int64_t dim0 = input.size(0);" in cpp
        assert "int64_t dim1 = input.size(1);" in cpp

    def test_loop_nest(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "for (int64_t i0 = 0; i0 < dim0; i0 += 1)" in cpp
        assert "for (int64_t i1 = 0; i1 < dim1; i1 += 32)" in cpp
        assert "std::min(i0 + (int64_t)1, dim0)" in cpp
        assert "std::min(i1 + (int64_t)32, dim1)" in cpp

    def test_tile_load(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "auto s_input = input.slice(0, i0, i0_end).slice(1, i1, i1_end);" in cpp

    def test_computation_nodes(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "auto x3 = torch::pow(s_input, 3.0);" in cpp
        assert "auto x_in =" in cpp
        assert "0.044715" in cpp
        assert "auto t_in =" in cpp
        assert "0.7978845608028654" in cpp
        assert "auto t = torch::tanh(t_in);" in cpp
        assert "auto onep = (1.0 + t);" in cpp
        assert "auto y =" in cpp

    def test_tile_store(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "output.slice(0, i0, i0_end).slice(1, i1, i1_end).copy_(y);" in cpp

    def test_registration(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "TORCH_LIBRARY_FRAGMENT(step_ops, m)" in cpp
        assert '"gelu_kernel_step"' in cpp

    def test_return_output(self):
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        assert "return output;" in cpp

    def test_full_structure(self):
        """Verify the overall structure of the generated code."""
        graph = _build_gelu_graph()
        cpp = CppCodegen(graph).generate()
        lines = cpp.strip().split("\n")
        # Should start with includes
        assert lines[0].startswith("#include")
        # Should end with registration block closing brace
        non_empty = [l for l in lines if l.strip()]
        assert non_empty[-1].strip() == "}"
