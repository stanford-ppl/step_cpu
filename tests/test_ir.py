"""Unit tests for IR graph construction."""

import ast
import math

from step.ir import (
    BinaryMapOp,
    StepGraph,
    StreamShape,
    StreamToTensor,
    TensorToStream,
    UnaryMapOp,
    reset_id_counter,
)


class TestStreamShape:
    def test_from_vec(self):
        shape = StreamShape.from_vec([1, 32], tensor_ndim=2)
        assert shape.rank == 2
        assert shape.tile_shape == [1, 32]
        assert shape.tensor_ndim == 2


class TestIRNodes:
    def setup_method(self):
        reset_id_counter()

    def test_tensor_to_stream(self):
        node = TensorToStream("input", [1, 32], tensor_ndim=2)
        assert node.tensor_param == "input"
        assert node.vec == [1, 32]
        assert node.stream_shape.rank == 2
        assert node.name == "s_input"

    def test_unary_map(self):
        source = TensorToStream("input", [1, 32], tensor_ndim=2)
        body = ast.parse("torch.tanh(a)", mode="eval").body
        unary = UnaryMapOp(source, "t", body, "a", {})
        assert unary.input_node is source
        assert source.users == [unary]
        assert unary.name == "t"

    def test_binary_map(self):
        source = TensorToStream("input", [1, 32], tensor_ndim=2)
        body = ast.parse("a + b", mode="eval").body
        binary = BinaryMapOp(source, source, "y", body, ("a", "b"), {})
        assert binary.input_node1 is source
        assert binary.input_node2 is source
        assert source.users == [binary, binary]

    def test_stream_to_tensor(self):
        source = TensorToStream("input", [1, 32], tensor_ndim=2)
        sink = StreamToTensor(source, "input")
        assert sink.input_node is source
        assert sink.like_tensor_param == "input"


class TestStepGraph:
    def setup_method(self):
        reset_id_counter()

    def test_gelu_graph_structure(self):
        """Build the GELU graph and verify structure."""
        graph = StepGraph("gelu_kernel", ["input"])

        # tensor_to_stream
        x = TensorToStream("input", [1, 32], tensor_ndim=2)
        graph.add_node(x)

        c_sqrt = math.sqrt(2.0 / math.pi)
        c_pow = 0.044715

        # x3 = UnaryMap(x, lambda a: torch.pow(a, 3.0))
        x3_body = ast.parse("torch.pow(a, 3.0)", mode="eval").body
        x3 = UnaryMapOp(x, "x3", x3_body, "a", {})
        graph.add_node(x3)

        # x_in = BinaryMap(x, x3, lambda a, b: a + c_pow * b)
        x_in_body = ast.parse("a + c_pow * b", mode="eval").body
        x_in = BinaryMapOp(x, x3, "x_in", x_in_body, ("a", "b"), {"c_pow": c_pow})
        graph.add_node(x_in)

        # t_in = UnaryMap(x_in, lambda a: c_sqrt * a)
        t_in_body = ast.parse("c_sqrt * a", mode="eval").body
        t_in = UnaryMapOp(x_in, "t_in", t_in_body, "a", {"c_sqrt": c_sqrt})
        graph.add_node(t_in)

        # t = UnaryMap(t_in, lambda a: torch.tanh(a))
        t_body = ast.parse("torch.tanh(a)", mode="eval").body
        t = UnaryMapOp(t_in, "t", t_body, "a", {})
        graph.add_node(t)

        # onep = UnaryMap(t, lambda a: 1.0 + a)
        onep_body = ast.parse("1.0 + a", mode="eval").body
        onep = UnaryMapOp(t, "onep", onep_body, "a", {})
        graph.add_node(onep)

        # y = BinaryMap(x, onep, lambda a, b: 0.5 * a * b)
        y_body = ast.parse("0.5 * a * b", mode="eval").body
        y = BinaryMapOp(x, onep, "y", y_body, ("a", "b"), {})
        graph.add_node(y)

        # stream_to_tensor
        out = StreamToTensor(y, "input")
        graph.add_node(out)

        # Verify counts
        assert len(graph.nodes) == 8
        assert len(graph.sources) == 1
        assert len(graph.sinks) == 1

        # Verify topo sort
        order = graph.topo_sort()
        assert len(order) == 8
        # Source must come first
        assert order[0] is x
        # Sink must come last
        assert order[-1] is out
        # x3 must come before x_in
        assert order.index(x3) < order.index(x_in)
        # t_in before t
        assert order.index(t_in) < order.index(t)
        # onep before y
        assert order.index(onep) < order.index(y)
        # y before out
        assert order.index(y) < order.index(out)

    def test_topo_sort_simple(self):
        graph = StepGraph("simple", ["input"])
        src = TensorToStream("input", [1, 32], tensor_ndim=2)
        graph.add_node(src)

        body = ast.parse("torch.tanh(a)", mode="eval").body
        m = UnaryMapOp(src, "m", body, "a", {})
        graph.add_node(m)

        sink = StreamToTensor(m, "input")
        graph.add_node(sink)

        order = graph.topo_sort()
        assert order == [src, m, sink]
