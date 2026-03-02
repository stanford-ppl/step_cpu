"""IR node classes and StepGraph container for STeP codegen."""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass, field
from typing import Any


_id_counter = itertools.count()


def _next_id() -> int:
    return next(_id_counter)


def reset_id_counter() -> None:
    """Reset the global ID counter (useful for tests)."""
    global _id_counter
    _id_counter = itertools.count()


@dataclass
class StreamShape:
    """Metadata describing a stream's iteration structure.

    rank: number of loop nesting levels
    tile_shape: chunk size at each level (the vec parameter)
    tensor_ndim: number of dimensions of the source tensor
    """
    rank: int
    tile_shape: list[int]
    tensor_ndim: int

    @classmethod
    def from_vec(cls, vec: list[int], tensor_ndim: int) -> StreamShape:
        return cls(rank=len(vec), tile_shape=list(vec), tensor_ndim=tensor_ndim)


class IRNode:
    """Base class for all IR nodes."""

    def __init__(self, name: str, stream_shape: StreamShape | None = None):
        self.id: int = _next_id()
        self.name: str = name
        self.stream_shape: StreamShape | None = stream_shape
        self.users: list[IRNode] = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, name={self.name!r})"


class TensorToStream(IRNode):
    """Source node: converts a tensor parameter into a tiled stream."""

    def __init__(self, tensor_param: str, vec: list[int], tensor_ndim: int):
        shape = StreamShape.from_vec(vec, tensor_ndim)
        super().__init__(name=f"s_{tensor_param}", stream_shape=shape)
        self.tensor_param: str = tensor_param
        self.vec: list[int] = list(vec)
        self.tensor_ndim: int = tensor_ndim


class StreamToTensor(IRNode):
    """Sink node: materializes a stream back into a tensor."""

    def __init__(self, input_node: IRNode, like_tensor_param: str):
        super().__init__(name="output", stream_shape=input_node.stream_shape)
        self.input_node: IRNode = input_node
        self.like_tensor_param: str = like_tensor_param
        input_node.users.append(self)


class UnaryMapOp(IRNode):
    """Apply a unary function element-wise to each tile in the stream."""

    def __init__(
        self,
        input_node: IRNode,
        name: str,
        func_ast: ast.expr,
        func_param: str,
        closures: dict[str, Any],
    ):
        super().__init__(name=name, stream_shape=input_node.stream_shape)
        self.input_node: IRNode = input_node
        self.func_ast: ast.expr = func_ast
        self.func_param: str = func_param
        self.closures: dict[str, Any] = closures
        input_node.users.append(self)


class BinaryMapOp(IRNode):
    """Apply a binary function element-wise to corresponding tile pairs."""

    def __init__(
        self,
        input_node1: IRNode,
        input_node2: IRNode,
        name: str,
        func_ast: ast.expr,
        func_params: tuple[str, str],
        closures: dict[str, Any],
    ):
        super().__init__(name=name, stream_shape=input_node1.stream_shape)
        self.input_node1: IRNode = input_node1
        self.input_node2: IRNode = input_node2
        self.func_ast: ast.expr = func_ast
        self.func_params: tuple[str, str] = func_params
        self.closures: dict[str, Any] = closures
        input_node1.users.append(self)
        input_node2.users.append(self)


class FlattenOp(IRNode):
    """Logical reshape: merge stream dimensions. CPU no-op."""

    def __init__(self, input_node: IRNode, min_rank: int, max_rank: int):
        super().__init__(name=f"flat_{input_node.name}", stream_shape=input_node.stream_shape)
        self.input_node: IRNode = input_node
        self.min_rank: int = min_rank
        self.max_rank: int = max_rank
        input_node.users.append(self)


class BufferizeOp(IRNode):
    """Materialization boundary: collect a stream into a buffer."""

    def __init__(self, input_node: IRNode, rank: int, buffer_id: int = 0):
        super().__init__(name=f"buf_{buffer_id}", stream_shape=input_node.stream_shape)
        self.input_node: IRNode = input_node
        self.rank: int = rank
        self.buffer_id: int = buffer_id
        input_node.users.append(self)


class StreamifyOp(IRNode):
    """Replay boundary: replay a buffered stream with repetition."""

    def __init__(self, input_node: IRNode, repeat_factor: list[int], rank: int):
        super().__init__(name=f"stfy_{input_node.name}", stream_shape=input_node.stream_shape)
        self.input_node: IRNode = input_node
        self.repeat_factor: list[int] = list(repeat_factor)
        self.rank: int = rank
        input_node.users.append(self)


class AccumOp(IRNode):
    """Reduction over innermost stream dimensions."""

    def __init__(self, input_node: IRNode, rank: int):
        super().__init__(name=f"acc_{input_node.name}", stream_shape=input_node.stream_shape)
        self.input_node: IRNode = input_node
        self.rank: int = rank
        input_node.users.append(self)


class ConstantNode(IRNode):
    """A captured scalar constant."""

    def __init__(self, value: float | int, name: str):
        super().__init__(name=name, stream_shape=None)
        self.value = value


class StepGraph:
    """Container for the full IR graph of a STeP function."""

    def __init__(self, func_name: str, tensor_params: list[str]):
        self.func_name: str = func_name
        self.tensor_params: list[str] = tensor_params
        self.nodes: list[IRNode] = []
        self.sources: list[TensorToStream] = []
        self.sinks: list[StreamToTensor] = []
        self.buffers: list[BufferizeOp] = []
        self._next_buffer_id: int = 0

    def add_node(self, node: IRNode) -> IRNode:
        self.nodes.append(node)
        if isinstance(node, TensorToStream):
            self.sources.append(node)
        elif isinstance(node, StreamToTensor):
            self.sinks.append(node)
        elif isinstance(node, BufferizeOp):
            self.buffers.append(node)
        return node

    def topo_sort(self) -> list[IRNode]:
        """Return nodes in topological order."""
        visited: set[int] = set()
        order: list[IRNode] = []

        def visit(node: IRNode) -> None:
            if node.id in visited:
                return
            visited.add(node.id)
            # Visit dependencies first
            if isinstance(node, UnaryMapOp):
                visit(node.input_node)
            elif isinstance(node, BinaryMapOp):
                visit(node.input_node1)
                visit(node.input_node2)
            elif isinstance(node, StreamToTensor):
                visit(node.input_node)
            elif isinstance(node, FlattenOp):
                visit(node.input_node)
            elif isinstance(node, BufferizeOp):
                visit(node.input_node)
            elif isinstance(node, StreamifyOp):
                visit(node.input_node)
            elif isinstance(node, AccumOp):
                visit(node.input_node)
            order.append(node)

        for sink in self.sinks:
            visit(sink)
        # Include any orphan sources
        for source in self.sources:
            if source.id not in visited:
                visit(source)
        return order
