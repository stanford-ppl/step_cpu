"""STeP (Streaming Tensor Programs) — Python frontend and CPU code generator.

Public API:
    tensor_to_stream  — Convert a tensor to a tiled stream
    stream_to_tensor  — Materialize a stream back into a tensor
    UnaryMap          — Apply a unary function to each tile
    BinaryMap         — Apply a binary function to tile pairs
    cpu_compile       — Trace a STeP function and compile to a C++ kernel
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from .ir import (
    AccumOp,
    BinaryMapOp,
    BufferizeOp,
    FlattenOp,
    IRNode,
    StepGraph,
    StreamifyOp,
    StreamToTensor,
    TensorToStream,
    UnaryMapOp,
    reset_id_counter,
)
from .lambda_parser import parse_lambda

# Sentinel: "entire dimension, no tiling"
FULL = -1

# ---------------------------------------------------------------------------
# Thread-local tracing context
# ---------------------------------------------------------------------------

_trace_ctx = threading.local()


def _is_tracing() -> bool:
    return getattr(_trace_ctx, "graph", None) is not None


def _get_graph() -> StepGraph:
    g = getattr(_trace_ctx, "graph", None)
    if g is None:
        raise RuntimeError("Not in a tracing context. Use cpu_compile() to trace.")
    return g


def _get_name_counter() -> dict[str, int]:
    c = getattr(_trace_ctx, "name_counter", None)
    if c is None:
        _trace_ctx.name_counter = {}
        c = _trace_ctx.name_counter
    return c


def _fresh_name(base: str) -> str:
    counter = _get_name_counter()
    idx = counter.get(base, 0)
    counter[base] = idx + 1
    if idx == 0:
        return base
    return f"{base}_{idx}"


# ---------------------------------------------------------------------------
# Proxy object — stands in for a stream during tracing
# ---------------------------------------------------------------------------

class StreamProxy:
    """Proxy returned by STeP operators during tracing. Wraps an IR node."""

    def __init__(self, node: IRNode):
        self._node = node

    @property
    def node(self) -> IRNode:
        return self._node


class _SymbolicShape:
    """Symbolic shape that returns FULL for any dimension index."""

    def __init__(self, ndim: int):
        self._ndim = ndim

    def __getitem__(self, idx: int):
        return FULL

    def __len__(self):
        return self._ndim


class TensorProxy:
    """Proxy standing in for a tensor parameter during tracing."""

    def __init__(self, name: str, ndim: int, transforms: list | None = None):
        self.name = name
        self.ndim = ndim
        self.transforms: list = transforms if transforms is not None else []

    @property
    def T(self) -> "TensorProxy":
        return TensorProxy(self.name, self.ndim, self.transforms + [("T",)])

    def contiguous(self) -> "TensorProxy":
        return TensorProxy(self.name, self.ndim, self.transforms + [("contiguous",)])

    def unsqueeze(self, dim: int) -> "TensorProxy":
        return TensorProxy(self.name, self.ndim + 1, self.transforms + [("unsqueeze", dim)])

    @property
    def shape(self) -> _SymbolicShape:
        return _SymbolicShape(self.ndim)

    def size(self, dim: int | None = None):
        return FULL


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def tensor_to_stream(tensor, vec: list[int]) -> StreamProxy:
    """Convert a tensor into a tiled stream.

    During tracing: creates a TensorToStream IR node and returns a StreamProxy.
    """
    if not _is_tracing():
        raise RuntimeError("tensor_to_stream() can only be called during tracing (via cpu_compile).")

    graph = _get_graph()

    if isinstance(tensor, TensorProxy):
        param_name = tensor.name
        ndim = tensor.ndim
        transforms = tensor.transforms
    else:
        raise TypeError(f"Expected TensorProxy during tracing, got {type(tensor).__name__}")

    node = TensorToStream(tensor_param=param_name, vec=vec, tensor_ndim=ndim)
    node.transforms = list(transforms)
    graph.add_node(node)
    return StreamProxy(node)


def stream_to_tensor(stream: StreamProxy, like_tensor) -> TensorProxy:
    """Materialize a stream back into a tensor.

    During tracing: creates a StreamToTensor IR node.
    """
    if not _is_tracing():
        raise RuntimeError("stream_to_tensor() can only be called during tracing (via cpu_compile).")

    graph = _get_graph()

    if not isinstance(stream, StreamProxy):
        raise TypeError(f"Expected StreamProxy, got {type(stream).__name__}")

    if isinstance(like_tensor, TensorProxy):
        like_name = like_tensor.name
    else:
        raise TypeError(f"Expected TensorProxy for like_tensor, got {type(like_tensor).__name__}")

    node = StreamToTensor(input_node=stream.node, like_tensor_param=like_name)
    graph.add_node(node)
    return TensorProxy(name="output", ndim=like_tensor.ndim)


def UnaryMap(stream: StreamProxy, func: Callable) -> StreamProxy:
    """Apply a unary function element-wise to each tile.

    During tracing: parses the lambda, creates a UnaryMapOp IR node.
    """
    if not _is_tracing():
        raise RuntimeError("UnaryMap() can only be called during tracing (via cpu_compile).")

    graph = _get_graph()

    if not isinstance(stream, StreamProxy):
        raise TypeError(f"Expected StreamProxy, got {type(stream).__name__}")

    body_ast, param_names, closures = parse_lambda(func)
    if len(param_names) != 1:
        raise ValueError(f"UnaryMap lambda must take exactly 1 parameter, got {len(param_names)}")

    name = _fresh_name("u")
    node = UnaryMapOp(
        input_node=stream.node,
        name=name,
        func_ast=body_ast,
        func_param=param_names[0],
        closures=closures,
    )
    graph.add_node(node)
    return StreamProxy(node)


def BinaryMap(stream1: StreamProxy, stream2: StreamProxy, func: Callable) -> StreamProxy:
    """Apply a binary function element-wise to corresponding tile pairs.

    During tracing: parses the lambda, creates a BinaryMapOp IR node.
    """
    if not _is_tracing():
        raise RuntimeError("BinaryMap() can only be called during tracing (via cpu_compile).")

    graph = _get_graph()

    if not isinstance(stream1, StreamProxy):
        raise TypeError(f"Expected StreamProxy for stream1, got {type(stream1).__name__}")
    if not isinstance(stream2, StreamProxy):
        raise TypeError(f"Expected StreamProxy for stream2, got {type(stream2).__name__}")

    body_ast, param_names, closures = parse_lambda(func)
    if len(param_names) != 2:
        raise ValueError(f"BinaryMap lambda must take exactly 2 parameters, got {len(param_names)}")

    name = _fresh_name("b")
    node = BinaryMapOp(
        input_node1=stream1.node,
        input_node2=stream2.node,
        name=name,
        func_ast=body_ast,
        func_params=(param_names[0], param_names[1]),
        closures=closures,
    )
    graph.add_node(node)
    return StreamProxy(node)


def Flatten(stream: StreamProxy, min_rank: int, max_rank: int) -> StreamProxy:
    """Flatten stream dimensions [min_rank, max_rank) into one."""
    if not _is_tracing():
        raise RuntimeError("Flatten() can only be called during tracing.")
    graph = _get_graph()
    if not isinstance(stream, StreamProxy):
        raise TypeError(f"Expected StreamProxy, got {type(stream).__name__}")
    node = FlattenOp(input_node=stream.node, min_rank=min_rank, max_rank=max_rank)
    graph.add_node(node)
    return StreamProxy(node)


def Bufferize(stream: StreamProxy, rank: int) -> StreamProxy:
    """Materialize a stream into a buffer for reuse."""
    if not _is_tracing():
        raise RuntimeError("Bufferize() can only be called during tracing.")
    graph = _get_graph()
    if not isinstance(stream, StreamProxy):
        raise TypeError(f"Expected StreamProxy, got {type(stream).__name__}")
    buf_id = graph._next_buffer_id
    graph._next_buffer_id += 1
    node = BufferizeOp(input_node=stream.node, rank=rank, buffer_id=buf_id)
    graph.add_node(node)
    return StreamProxy(node)


def Streamify(buffer: StreamProxy, repeat_factor: list[int], rank: int) -> StreamProxy:
    """Replay a buffered stream with repetition."""
    if not _is_tracing():
        raise RuntimeError("Streamify() can only be called during tracing.")
    graph = _get_graph()
    if not isinstance(buffer, StreamProxy):
        raise TypeError(f"Expected StreamProxy, got {type(buffer).__name__}")
    node = StreamifyOp(input_node=buffer.node, repeat_factor=repeat_factor, rank=rank)
    graph.add_node(node)
    return StreamProxy(node)


def Accum(stream: StreamProxy, rank: int) -> StreamProxy:
    """Accumulate (reduce) over innermost stream dimensions."""
    if not _is_tracing():
        raise RuntimeError("Accum() can only be called during tracing.")
    graph = _get_graph()
    if not isinstance(stream, StreamProxy):
        raise TypeError(f"Expected StreamProxy, got {type(stream).__name__}")
    node = AccumOp(input_node=stream.node, rank=rank)
    graph.add_node(node)
    return StreamProxy(node)


# ---------------------------------------------------------------------------
# cpu_compile — trace and compile
# ---------------------------------------------------------------------------

def cpu_compile(func: Callable) -> Callable:
    """Trace a STeP function and compile it to an optimized C++ kernel.

    Returns a callable that accepts the same tensor arguments as `func`
    and dispatches to the compiled C++ kernel.
    """
    import inspect

    from .codegen import CppCodegen
    from .compile import build_extension

    # Determine tensor parameter names from the function signature
    sig = inspect.signature(func)
    tensor_params = list(sig.parameters.keys())

    # Reset IR IDs for deterministic codegen
    reset_id_counter()

    # Set up tracing context
    graph = StepGraph(func_name=func.__name__, tensor_params=tensor_params)
    _trace_ctx.graph = graph
    _trace_ctx.name_counter = {}

    try:
        # Call func with proxy objects.
        # Infer ndim from parameter annotations or hints if available,
        # otherwise default to 2D. The function may call .unsqueeze() etc.
        # which updates the proxy's ndim dynamically.
        ndim_hints = getattr(func, '_param_ndims', None)
        proxies = []
        for i, p in enumerate(tensor_params):
            nd = ndim_hints[i] if ndim_hints else 2
            proxies.append(TensorProxy(name=p, ndim=nd))
        func(*proxies)
    finally:
        _trace_ctx.graph = None
        _trace_ctx.name_counter = None

    # Generate C++ code
    codegen = CppCodegen(graph)
    cpp_source = codegen.generate()

    # Build and load the extension
    compiled_func = build_extension(graph.func_name, cpp_source)
    return compiled_func
