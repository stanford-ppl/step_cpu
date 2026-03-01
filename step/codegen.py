"""C++ code emitter — walks a StepGraph and produces a complete .cpp source file."""

from __future__ import annotations

from .ir import (
    BinaryMapOp,
    IRNode,
    StepGraph,
    StreamToTensor,
    TensorToStream,
    UnaryMapOp,
)
from .lambda_parser import LambdaTranslator


class CppCodegen:
    """Generates a complete C++ source file from a StepGraph."""

    def __init__(self, graph: StepGraph):
        self.graph = graph
        self.lines: list[str] = []
        self._indent = 0

    # ------------------------------------------------------------------
    # Helpers
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

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self) -> str:
        """Generate the complete C++ source string."""
        self.lines = []
        self._indent = 0

        self._emit_preamble()
        self._emit_function()
        self._emit_registration()

        return "\n".join(self.lines) + "\n"

    # ------------------------------------------------------------------
    # Phase 1: Preamble
    # ------------------------------------------------------------------

    def _emit_preamble(self) -> None:
        self._emit("#include <torch/extension.h>")
        self._emit("#include <torch/library.h>")
        self._emit("#include <algorithm>")
        self._emit()

    # ------------------------------------------------------------------
    # Phase 2: Function signature + body
    # ------------------------------------------------------------------

    def _emit_function(self) -> None:
        func_name = self.graph.func_name + "_step"
        params = ", ".join(f"torch::Tensor {p}" for p in self.graph.tensor_params)
        self._emit(f"torch::Tensor {func_name}({params}) {{")
        self._indent_inc()

        # Contiguity checks
        for p in self.graph.tensor_params:
            self._emit(f'TORCH_CHECK({p}.is_contiguous(), "{p} must be contiguous");')

        # Output tensor allocation (from the first sink)
        sink = self.graph.sinks[0]
        self._emit(f"auto output = torch::empty_like({sink.like_tensor_param});")
        self._emit()

        # Determine loop structure from the first source
        source = self.graph.sources[0]
        ndim = source.tensor_ndim
        vec = source.vec
        tensor_param = source.tensor_param

        # Dimension variables
        for d in range(ndim):
            self._emit(f"int64_t dim{d} = {tensor_param}.size({d});")
        self._emit()

        # Emit tiled loop nest
        self._emit_loop_nest(ndim, vec)

        self._emit()
        self._emit("return output;")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _emit_loop_nest(self, ndim: int, vec: list[int]) -> None:
        """Emit nested for-loops with tile boundary clamping."""
        for d in range(ndim):
            v = vec[d]
            self._emit(f"for (int64_t i{d} = 0; i{d} < dim{d}; i{d} += {v}) {{")
            self._indent_inc()
            self._emit(f"int64_t i{d}_end = std::min(i{d} + (int64_t){v}, dim{d});")

        # Emit computation body (topo-sorted)
        self._emit()
        sorted_nodes = self.graph.topo_sort()
        for node in sorted_nodes:
            self._emit_node(node, ndim)

        # Close loops
        for d in range(ndim - 1, -1, -1):
            self._indent_dec()
            self._emit("}")

    def _emit_node(self, node: IRNode, ndim: int) -> None:
        """Emit C++ code for a single IR node."""
        if isinstance(node, TensorToStream):
            slices = _make_slices(node.tensor_param, ndim)
            self._emit(f"auto {node.name} = {slices};")

        elif isinstance(node, UnaryMapOp):
            cpp_expr = _translate_unary(node)
            self._emit(f"auto {node.name} = {cpp_expr};")

        elif isinstance(node, BinaryMapOp):
            cpp_expr = _translate_binary(node)
            self._emit(f"auto {node.name} = {cpp_expr};")

        elif isinstance(node, StreamToTensor):
            slices = _make_slices("output", ndim)
            self._emit(f"{slices}.copy_({node.input_node.name});")

    # ------------------------------------------------------------------
    # Phase 3: torch::Library registration
    # ------------------------------------------------------------------

    def _emit_registration(self) -> None:
        func_name = self.graph.func_name + "_step"
        self._emit("TORCH_LIBRARY_FRAGMENT(step_ops, m) {")
        self._indent_inc()
        self._emit(f'm.def("{func_name}", {func_name});')
        self._indent_dec()
        self._emit("}")
        self._emit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slices(tensor_name: str, ndim: int) -> str:
    """Build chained .slice() calls for tile extraction."""
    expr = tensor_name
    for d in range(ndim):
        expr = f"{expr}.slice({d}, i{d}, i{d}_end)"
    return expr


def _translate_unary(node: UnaryMapOp) -> str:
    """Translate a UnaryMapOp's lambda to a C++ expression."""
    param_to_cpp = {node.func_param: node.input_node.name}
    translator = LambdaTranslator(
        func_params=(node.func_param,),
        closures=node.closures,
        param_to_cpp=param_to_cpp,
    )
    return translator.translate(node.func_ast)


def _translate_binary(node: BinaryMapOp) -> str:
    """Translate a BinaryMapOp's lambda to a C++ expression."""
    param_to_cpp = {
        node.func_params[0]: node.input_node1.name,
        node.func_params[1]: node.input_node2.name,
    }
    translator = LambdaTranslator(
        func_params=node.func_params,
        closures=node.closures,
        param_to_cpp=param_to_cpp,
    )
    return translator.translate(node.func_ast)
