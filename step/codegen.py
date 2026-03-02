"""C++ code emitter — walks a StepGraph and produces a complete .cpp source file."""

from __future__ import annotations

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

        if self._has_multistage_nodes():
            return self._generate_multistage()
        return self._generate_elementwise()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _has_multistage_nodes(self) -> bool:
        for node in self.graph.nodes:
            if isinstance(node, (AccumOp, BufferizeOp)):
                return True
        return False

    # ------------------------------------------------------------------
    # Elementwise path (existing GELU codegen, unchanged)
    # ------------------------------------------------------------------

    def _generate_elementwise(self) -> str:
        self._emit_preamble()
        self._emit_elementwise_function()
        self._emit_registration()
        return "\n".join(self.lines) + "\n"

    def _emit_preamble(self) -> None:
        self._emit("#include <torch/extension.h>")
        self._emit("#include <torch/library.h>")
        self._emit("#include <algorithm>")
        self._emit()

    def _emit_elementwise_function(self) -> None:
        func_name = self.graph.func_name + "_step"
        params = ", ".join(f"torch::Tensor {p}" for p in self.graph.tensor_params)
        self._emit(f"torch::Tensor {func_name}({params}) {{")
        self._indent_inc()

        for p in self.graph.tensor_params:
            self._emit(f'TORCH_CHECK({p}.is_contiguous(), "{p} must be contiguous");')

        sink = self.graph.sinks[0]
        self._emit(f"auto output = torch::empty_like({sink.like_tensor_param});")
        self._emit()

        source = self.graph.sources[0]
        ndim = source.tensor_ndim
        vec = source.vec
        tensor_param = source.tensor_param

        for d in range(ndim):
            self._emit(f"int64_t dim{d} = {tensor_param}.size({d});")
        self._emit()

        self._emit_loop_nest(ndim, vec)

        self._emit()
        self._emit("return output;")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _emit_loop_nest(self, ndim: int, vec: list[int]) -> None:
        for d in range(ndim):
            v = vec[d]
            self._emit(f"for (int64_t i{d} = 0; i{d} < dim{d}; i{d} += {v}) {{")
            self._indent_inc()
            self._emit(f"int64_t i{d}_end = std::min(i{d} + (int64_t){v}, dim{d});")

        self._emit()
        sorted_nodes = self.graph.topo_sort()
        for node in sorted_nodes:
            self._emit_node(node, ndim)

        for d in range(ndim - 1, -1, -1):
            self._indent_dec()
            self._emit("}")

    def _emit_node(self, node: IRNode, ndim: int) -> None:
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

    def _emit_registration(self) -> None:
        func_name = self.graph.func_name + "_step"
        self._emit("TORCH_LIBRARY_FRAGMENT(step_ops, m) {")
        self._indent_inc()
        self._emit(f'm.def("{func_name}", {func_name});')
        self._indent_dec()
        self._emit("}")
        self._emit()

    # ------------------------------------------------------------------
    # Multi-stage path (GPT2MLP codegen)
    # ------------------------------------------------------------------

    def _generate_multistage(self) -> str:
        self._emit_preamble()
        self._emit_multistage_function()
        self._emit_registration()
        return "\n".join(self.lines) + "\n"

    def _emit_multistage_function(self) -> None:
        func_name = self.graph.func_name + "_step"
        params = ", ".join(f"torch::Tensor {p}" for p in self.graph.tensor_params)
        self._emit(f"torch::Tensor {func_name}({params}) {{")
        self._indent_inc()

        # Analyze the graph to find stages
        sorted_nodes = self.graph.topo_sort()
        stages = self._analyze_stages(sorted_nodes)

        # Find the primary activation parameter (first param) for M dim
        primary_param = self.graph.tensor_params[0]
        self._emit(f"int64_t M = {primary_param}.size(0);")

        # Output allocation
        sink = self.graph.sinks[0]
        self._emit(f"auto output = torch::empty_like({sink.like_tensor_param});")

        # Emit tensor transforms
        self._emit_transforms(sorted_nodes)

        # Emit intermediate buffer allocations
        self._emit_buffer_allocs(stages, primary_param)

        # Emit each computation stage
        for stage in stages:
            self._emit()
            self._emit_stage(stage, primary_param)

        self._emit()
        self._emit("return output;")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _analyze_stages(self, sorted_nodes: list[IRNode]) -> list[dict]:
        """Split the topo-sorted graph into stages at Bufferize/StreamToTensor boundaries.

        Each stage is a matmul+post-processing pipeline:
        StreamifyOp → BinaryMap(mm) → AccumOp → [bias add] → [GELU chain] → BufferizeOp/StreamToTensor

        Returns list of stage dicts with keys:
            - 'streamify': StreamifyOp for activation replay
            - 'weight_source': TensorToStream for weight
            - 'matmul': BinaryMapOp for torch.mm
            - 'accum': AccumOp
            - 'post_accum': list of nodes after accum (bias, GELU, etc.)
            - 'sink': BufferizeOp or StreamToTensor
            - 'bias_source': TensorToStream for bias (if any)
            - 'outer_N': total N dimension size
            - 'N_tile': tile size for N
            - 'inner_K': total K dimension (number of inner iterations * tile)
            - 'K_tile': tile size for K
            - 'buffer_id': output buffer id (or -1 for final output)
        """
        stages = []

        # Find all AccumOps — each one anchors a matmul stage
        accum_nodes = [n for n in sorted_nodes if isinstance(n, AccumOp)]

        for accum in accum_nodes:
            stage = self._trace_stage(accum, sorted_nodes)
            stages.append(stage)

        return stages

    def _trace_stage(self, accum: AccumOp, sorted_nodes: list[IRNode]) -> dict:
        """Trace backward from an AccumOp to find all stage components."""
        stage: dict = {
            'accum': accum,
            'post_accum': [],
            'sink': None,
            'streamify': None,
            'weight_source': None,
            'matmul': None,
            'bias_source': None,
            'bias_flatten': None,
        }

        # The matmul BinaryMapOp feeds into AccumOp
        matmul = accum.input_node
        assert isinstance(matmul, BinaryMapOp), f"Expected BinaryMapOp before AccumOp, got {type(matmul)}"
        stage['matmul'] = matmul

        # matmul inputs: input_node1 = activation (StreamifyOp), input_node2 = weight (TensorToStream)
        act_input = matmul.input_node1
        weight_input = matmul.input_node2

        # Trace activation back to StreamifyOp
        streamify = act_input
        while not isinstance(streamify, StreamifyOp):
            if hasattr(streamify, 'input_node'):
                streamify = streamify.input_node
            else:
                break
        stage['streamify'] = streamify

        # Weight source: trace back to TensorToStream
        wt_src = weight_input
        while not isinstance(wt_src, TensorToStream):
            if hasattr(wt_src, 'input_node'):
                wt_src = wt_src.input_node
            else:
                break
        stage['weight_source'] = wt_src

        # Compute loop bounds from StreamifyOp
        # repeat_factor gives outer N iterations, rank gives inner K iterations
        # Weight vec gives tile sizes
        wt_vec = wt_src.vec  # [N_tile, K_tile]
        stage['N_tile'] = wt_vec[0]
        stage['K_tile'] = wt_vec[1]

        # The StreamifyOp's repeat_factor[0] * N_tile = total N
        stage['outer_N'] = streamify.repeat_factor[0] * wt_vec[0]

        # Inner K: trace the bufferize that feeds the streamify
        buf_input = streamify.input_node
        assert isinstance(buf_input, BufferizeOp)
        # Trace back through flatten to the TensorToStream of activations
        act_src = buf_input.input_node
        while isinstance(act_src, FlattenOp):
            act_src = act_src.input_node
        if isinstance(act_src, TensorToStream):
            # K comes from the activation tensor's K dimension
            act_vec = act_src.vec  # [M, K_tile] for activation
            # Total K = tensor K dim. We can infer from vec and the fact that
            # K_tile from weight matches the activation tile's second dim.
            # The number of K iterations is encoded in the bufferize rank
            # For now, compute from weight: inner K iterations * K_tile
            # The StreamifyOp has rank=1 meaning 1 inner dimension is replayed
            pass

        # Inner K: total_K = number of inner loop iterations * K_tile
        # The bufferize collects all K tiles. The number of K tiles can be
        # computed from the weight tensor dimensions:
        # Weight is [N, K] after transpose, tiled as [N_tile, K_tile]
        # So K iterations = K / K_tile
        # We store the total K size
        stage['inner_K'] = stage['outer_N']  # placeholder, we'll compute from weight

        # Actually: for c_fc, weight_source has tensor W_fc_T of shape [3072, 768]
        # with vec=[256, 256]. The stream shape is [12, 3]. outer=12 (N/N_tile), inner=3 (K/K_tile)
        # So inner_K = (K/K_tile) * K_tile = K. But we don't have K directly.
        # We need to get it from the weight's original shape.
        # The weight vec is [N_tile, K_tile]. The weight stream is [N/N_tile, K/K_tile].
        # We need K. Let's trace from the activation source.
        # For activation: vec=[M, K_tile] → stream [1, K/K_tile], tile [M, K_tile]
        # After flatten: stream [K/K_tile]
        # After bufferize: all K/K_tile tiles collected
        # After streamify(repeat=[N/N_tile]): stream [N/N_tile, K/K_tile]
        # So repeat_factor[0] = N/N_tile, and the buffered count = K/K_tile
        # The K/K_tile is stored implicitly. We need total_K for the inner loop.

        # Strategy: we'll emit the inner loop using the weight dimensions directly.
        # Total N = repeat_factor[0] * N_tile
        # Total K = we need from another source. Let's look at the activation TensorToStream.
        if isinstance(act_src, TensorToStream):
            act_vec = act_src.vec
            # act_vec = [M, K_tile]. M is FULL (-1), K_tile = 256
            # The original tensor is [M, K]. K is the second dim.
            # We'll use the tensor param to get K at runtime.
            stage['act_source'] = act_src
            stage['act_param'] = act_src.tensor_param
        else:
            # Activation comes from a previous buffer
            stage['act_source'] = None
            stage['act_param'] = None

        # Find post-accum chain: nodes that consume accum's output
        # Walk forward from accum through the sorted_nodes
        post_nodes = []
        current_names = {accum.name}
        # Track the accum's position in sorted_nodes
        accum_idx = None
        for i, n in enumerate(sorted_nodes):
            if n is accum:
                accum_idx = i
                break

        for n in sorted_nodes[accum_idx + 1:]:
            if isinstance(n, (BufferizeOp, StreamToTensor)):
                # Check if this node depends on our chain
                if isinstance(n, StreamToTensor) and n.input_node.name in current_names:
                    stage['sink'] = n
                    stage['sink_type'] = 'output'
                    break
                elif isinstance(n, BufferizeOp) and n.input_node.name in current_names:
                    stage['sink'] = n
                    stage['sink_type'] = 'buffer'
                    stage['buffer_id'] = n.buffer_id
                    break
                else:
                    continue

            # Check if this node consumes something in our chain
            is_in_chain = False
            if isinstance(n, UnaryMapOp) and n.input_node.name in current_names:
                is_in_chain = True
            elif isinstance(n, BinaryMapOp):
                if n.input_node1.name in current_names or n.input_node2.name in current_names:
                    is_in_chain = True
                    # If one input is a Flatten feeding a bias TensorToStream
                    if n.input_node1.name not in current_names:
                        stage['bias_node'] = n.input_node1
                    elif n.input_node2.name not in current_names:
                        stage['bias_node'] = n.input_node2
            elif isinstance(n, FlattenOp) and n.input_node.name in current_names:
                is_in_chain = True

            if is_in_chain:
                post_nodes.append(n)
                current_names.add(n.name)

        stage['post_accum'] = post_nodes

        # If we haven't found a sink yet, look further
        if stage['sink'] is None:
            for n in sorted_nodes:
                if isinstance(n, BufferizeOp) and n.input_node.name in current_names:
                    stage['sink'] = n
                    stage['sink_type'] = 'buffer'
                    stage['buffer_id'] = n.buffer_id
                    break
                elif isinstance(n, StreamToTensor) and n.input_node.name in current_names:
                    stage['sink'] = n
                    stage['sink_type'] = 'output'
                    break

        # Find bias TensorToStream if any in post_accum BinaryMapOps
        for n in post_nodes:
            if isinstance(n, BinaryMapOp):
                for inp in [n.input_node1, n.input_node2]:
                    # Trace back to find bias TensorToStream
                    src = inp
                    while isinstance(src, FlattenOp):
                        src = src.input_node
                    if isinstance(src, TensorToStream) and src is not stage.get('weight_source'):
                        if src is not stage.get('act_source'):
                            stage['bias_source'] = src
                            break

        return stage

    def _emit_transforms(self, sorted_nodes: list[IRNode]) -> None:
        """Emit tensor transform declarations at function start."""
        for node in sorted_nodes:
            if not isinstance(node, TensorToStream):
                continue
            transforms = getattr(node, 'transforms', [])
            if not transforms:
                continue

            param = node.tensor_param
            cpp_var = param
            suffix_parts = []

            i = 0
            while i < len(transforms):
                t = transforms[i]
                if t[0] == 'T':
                    suffix_parts.append('t')
                    # Check if next is contiguous — combine them
                    if i + 1 < len(transforms) and transforms[i + 1][0] == 'contiguous':
                        cpp_expr = f"{param}.t().contiguous()"
                        suffix_parts.append('c')
                        i += 2
                    else:
                        cpp_expr = f"{param}.t()"
                        i += 1
                elif t[0] == 'contiguous':
                    suffix_parts.append('c')
                    cpp_expr = f"{cpp_var}.contiguous()"
                    i += 1
                elif t[0] == 'unsqueeze':
                    dim = t[1]
                    suffix_parts.append(f'us{dim}')
                    cpp_expr = f"{param}.unsqueeze({dim})"
                    i += 1
                else:
                    i += 1
                    continue

                cpp_var_name = f"{param}_{'_'.join(suffix_parts)}_"
                self._emit(f"auto {cpp_var_name} = {cpp_expr};")
                node._cpp_transformed_name = cpp_var_name

    def _emit_buffer_allocs(self, stages: list[dict], primary_param: str) -> None:
        """Emit intermediate buffer allocations."""
        for stage in stages:
            sink = stage.get('sink')
            if sink is not None and isinstance(sink, BufferizeOp):
                buf_id = sink.buffer_id
                total_cols = stage['outer_N']
                self._emit(f"auto buf_{buf_id} = torch::empty({{M, {total_cols}}}, {primary_param}.options());")

    def _get_transformed_name(self, src: TensorToStream) -> str:
        """Get the C++ variable name for a (possibly transformed) TensorToStream."""
        if hasattr(src, '_cpp_transformed_name'):
            return src._cpp_transformed_name
        return src.tensor_param

    def _emit_stage(self, stage: dict, primary_param: str) -> None:
        """Emit a single matmul+post-processing stage."""
        streamify = stage['streamify']
        weight_src = stage['weight_source']
        matmul = stage['matmul']
        accum = stage['accum']
        post_accum = stage['post_accum']
        sink = stage.get('sink')

        N_tile = stage['N_tile']
        K_tile = stage['K_tile']
        outer_N = stage['outer_N']

        # Determine weight C++ name
        weight_name = self._get_transformed_name(weight_src)

        # Determine activation source name
        # The StreamifyOp's input is a BufferizeOp. That bufferize's input
        # traces back to a TensorToStream (for stage 1) or a previous buffer.
        buf_op = streamify.input_node
        assert isinstance(buf_op, BufferizeOp)
        act_chain = buf_op.input_node
        while isinstance(act_chain, FlattenOp):
            act_chain = act_chain.input_node

        if isinstance(act_chain, TensorToStream):
            act_name = self._get_transformed_name(act_chain)
            inner_K_expr = f"{act_name}.size(1)"
        else:
            # Intermediate buffer from a previous stage — use the BufferizeOp directly
            act_name = f"buf_{buf_op.buffer_id}"
            inner_K_expr = f"{act_name}.size(1)"

        # Emit comment
        self._emit(f"// Stage: matmul {act_name} @ {weight_name}")

        # Outer N loop
        self._emit(f"for (int64_t i0 = 0; i0 < {outer_N}; i0 += {N_tile}) {{")
        self._indent_inc()
        self._emit(f"int64_t i0_end = std::min(i0 + (int64_t){N_tile}, (int64_t){outer_N});")

        # Accumulator
        self._emit(f"auto acc = torch::zeros({{M, i0_end - i0}}, {primary_param}.options());")

        # Inner K loop
        self._emit(f"for (int64_t i1 = 0; i1 < {inner_K_expr}; i1 += {K_tile}) {{")
        self._indent_inc()
        self._emit(f"int64_t i1_end = std::min(i1 + (int64_t){K_tile}, {inner_K_expr});")

        # Tile loads
        self._emit(f"auto act = {act_name}.slice(1, i1, i1_end);")
        self._emit(f"auto w = {weight_name}.slice(0, i0, i0_end).slice(1, i1, i1_end);")

        # Matmul accumulation — translate the BinaryMap lambda
        # The lambda is `lambda a, w: torch.mm(a, w.T)` which translates to
        # `torch::mm(act, w.t())`
        param_to_cpp = {
            matmul.func_params[0]: "act",
            matmul.func_params[1]: "w",
        }
        translator = LambdaTranslator(
            func_params=matmul.func_params,
            closures=matmul.closures,
            param_to_cpp=param_to_cpp,
        )
        mm_expr = translator.translate(matmul.func_ast)
        self._emit(f"acc.add_({mm_expr});")

        # Close inner K loop
        self._indent_dec()
        self._emit("}")

        # Post-accumulation operations (bias, GELU chain, etc.)
        # Track variable name mapping: the accum result is "acc"
        var_map: dict[str, str] = {accum.name: "acc"}

        for node in post_accum:
            if isinstance(node, BinaryMapOp):
                # Determine input names
                name1 = var_map.get(node.input_node1.name, node.input_node1.name)
                name2 = var_map.get(node.input_node2.name, node.input_node2.name)

                # Check if either input is a bias (TensorToStream, possibly through Flatten)
                for inp, var_name_key in [(node.input_node1, 'name1'), (node.input_node2, 'name2')]:
                    src = inp
                    while isinstance(src, FlattenOp):
                        src = src.input_node
                    if isinstance(src, TensorToStream) and src is not weight_src:
                        if src is not stage.get('act_source'):
                            # This is a bias — emit slice
                            bias_name = self._get_transformed_name(src)
                            bias_slice = f"{bias_name}.slice(1, i0, i0_end)"
                            if var_name_key == 'name1':
                                name1 = bias_slice
                            else:
                                name2 = bias_slice

                p2c = {
                    node.func_params[0]: name1,
                    node.func_params[1]: name2,
                }
                t = LambdaTranslator(
                    func_params=node.func_params,
                    closures=node.closures,
                    param_to_cpp=p2c,
                )
                cpp_expr = t.translate(node.func_ast)
                self._emit(f"auto {node.name} = {cpp_expr};")
                var_map[node.name] = node.name

            elif isinstance(node, UnaryMapOp):
                input_name = var_map.get(node.input_node.name, node.input_node.name)
                p2c = {node.func_param: input_name}
                t = LambdaTranslator(
                    func_params=(node.func_param,),
                    closures=node.closures,
                    param_to_cpp=p2c,
                )
                cpp_expr = t.translate(node.func_ast)
                self._emit(f"auto {node.name} = {cpp_expr};")
                var_map[node.name] = node.name

        # Write result to output or buffer
        # Find the last variable in our chain
        if post_accum:
            result_var = post_accum[-1].name
        else:
            result_var = "acc"

        if sink is not None:
            if isinstance(sink, StreamToTensor):
                self._emit(f"output.slice(1, i0, i0_end).copy_({result_var});")
            elif isinstance(sink, BufferizeOp):
                self._emit(f"buf_{sink.buffer_id}.slice(1, i0, i0_end).copy_({result_var});")

        # Close outer N loop
        self._indent_dec()
        self._emit("}")


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
