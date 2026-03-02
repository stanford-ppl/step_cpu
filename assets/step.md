# STeP Operator Semantics for CPU C++ Codegen

This document defines how each STeP (Streaming Tensor Programs) operator lowers to C++ code targeting CPU execution via PyTorch C++ extensions.

## Core Concepts

### Streams on CPU

On hardware, STeP uses streams of tiles with stop tokens for synchronization. On CPU, streams become **tiled loop nests** over contiguous tensors:

- **Stream rank** = number of loop nesting levels
- **Tile shape** = the chunk size at each level (the `vec` parameter)
- **Stop tokens** = loop bounds derived from `ceil(dim_size / tile_size)` at each level

### Tiles on CPU

A tile is a contiguous sub-tensor extracted via slicing. For a 2D tensor of shape `[M, N]` with `vec = [v0, v1]`, each tile is `input[i0:i0+v0, i1:i1+v1]` — a view into the original tensor. The last tile along each dimension may be smaller if the dimension is not evenly divisible by the tile size.

### Fusion Model

When a chain of UnaryMap/BinaryMap operators all originate from the same `tensor_to_stream` source(s), the entire chain **fuses into a single tiled loop**. Intermediate results are local tile-sized tensors that live in registers/cache. Only the final `stream_to_tensor` writes results back to the output tensor. This eliminates intermediate memory traffic — the key optimization STeP enables.

---

## Operators

### `tensor_to_stream(tensor, vec)`

**Semantics:** Convert a tensor into a stream by tiling it according to `vec`.

**Parameters:**
- `tensor`: Input PyTorch tensor (must be contiguous)
- `vec`: List of tile sizes, one per dimension. `vec[i]` is the tile size for dimension `i`.

**CPU Lowering:** Generates the outer loop nest:
```cpp
// For tensor shape [M, N] with vec = [v0, v1]:
for (int64_t i0 = 0; i0 < M; i0 += v0) {
    int64_t i0_end = std::min(i0 + v0, M);
    for (int64_t i1 = 0; i1 < N; i1 += v1) {
        int64_t i1_end = std::min(i1 + v1, N);
        auto tile = input.slice(0, i0, i0_end).slice(1, i1, i1_end);
        // ... computation body ...
    }
}
```

**Stream shape:** `rank = len(vec)`, `tile_shape = vec`, `extents[i] = ceil(tensor.shape[i] / vec[i])`.

### `stream_to_tensor(stream, like_tensor)`

**Semantics:** Materialize a stream back into a tensor by writing each output tile to its corresponding position.

**Parameters:**
- `stream`: The input stream (result of Map operations)
- `like_tensor`: Reference tensor whose shape and dtype determine the output tensor

**CPU Lowering:** Writes the final computed tile into the output tensor:
```cpp
output.slice(0, i0, i0_end).slice(1, i1, i1_end).copy_(result_tile);
```

The output tensor is allocated once as `torch::empty_like(like_tensor)` before the loop nest.

### `UnaryMap(stream, func)`

**Semantics:** Apply `func` element-wise to each tile in the stream. Produces a new stream with the same shape.

**Parameters:**
- `stream`: Input stream
- `func`: A Python lambda `lambda a: expr` where `a` is a tile-sized tensor

**CPU Lowering:** Inserts a computation line inside the tile loop body:
```cpp
auto result = /* C++ translation of func applied to input tile */;
```

No new loops are generated — UnaryMap fuses into the existing tile loop.

**Stop token propagation:** Output stream has the same shape and extents as the input. In C++ this is implicit since both share the same loop.

### `BinaryMap(stream1, stream2, func)`

**Semantics:** Consume two streams in lockstep and apply `func` element-wise to corresponding tile pairs. Both streams must have the same stream shape.

**Parameters:**
- `stream1`, `stream2`: Two input streams with identical stream shapes
- `func`: A Python lambda `lambda a, b: expr` where `a` and `b` are tile-sized tensors

**CPU Lowering:** Both inputs are available as local variables in the same loop iteration:
```cpp
auto result = /* C++ translation of func applied to tile1 and tile2 */;
```

**Constraint:** Both input streams must originate from `tensor_to_stream` calls with the same `vec` on tensors of the same shape. This ensures they iterate in lockstep.

### Fused Map Chains (GELU Example)

When a linear chain of UnaryMap/BinaryMap operations shares the same loop structure, the entire chain fuses:

```python
x = tensor_to_stream(input, [1, 32])
x3 = UnaryMap(x, lambda a: torch.pow(a, 3.0))
x_in = BinaryMap(x, x3, lambda a, b: a + 0.044715 * b)
t_in = UnaryMap(x_in, lambda a: 0.7978845608028654 * a)
t = UnaryMap(t_in, lambda a: torch.tanh(a))
onep = UnaryMap(t, lambda a: 1.0 + a)
y = BinaryMap(x, onep, lambda a, b: 0.5 * a * b)
out = stream_to_tensor(y, like_tensor=input)
```

Lowers to a **single fused loop**:
```cpp
for (int64_t i0 = 0; i0 < dim0; i0 += 1) {
    int64_t i0_end = std::min(i0 + (int64_t)1, dim0);
    for (int64_t i1 = 0; i1 < dim1; i1 += 32) {
        int64_t i1_end = std::min(i1 + (int64_t)32, dim1);

        auto x = input.slice(0, i0, i0_end).slice(1, i1, i1_end);
        auto x3 = torch::pow(x, 3.0);
        auto x_in = x + 0.044715 * x3;
        auto t_in = 0.7978845608028654 * x_in;
        auto t = torch::tanh(t_in);
        auto onep = 1.0 + t;
        auto y = 0.5 * x * onep;

        output.slice(0, i0, i0_end).slice(1, i1, i1_end).copy_(y);
    }
}
```

All intermediate tensors (`x3`, `x_in`, `t_in`, `t`, `onep`) are tile-sized and short-lived — they fit in cache and are never written to main memory.

---

## Lambda-to-C++ Translation

Python lambdas in Map operators are translated to C++ expressions:

| Python | C++ |
|---|---|
| `a + b`, `a * b`, `a - b`, `a / b` | Same operators |
| `torch.pow(a, 3.0)` | `torch::pow(a, 3.0)` |
| `torch.tanh(a)` | `torch::tanh(a)` |
| `torch.exp(a)` | `torch::exp(a)` |
| `torch.sqrt(a)` | `torch::sqrt(a)` |
| `0.5 * a` | `0.5 * a` (literal constants pass through) |
| Closure variable `c = 0.797...` | Inlined as `0.7978845608028654` |

---

## Constraints (v1)

- All tensors must be contiguous (row-major)
- All `tensor_to_stream` calls in a fused chain must use the same `vec`
- Only 2D tensors are supported in v1 (extensible to N-D)
- Supported element-wise ops: arithmetic, pow, tanh, exp, sqrt
- No dynamic shapes: tensor shapes are fixed at compile time for the generated kernel, but the generated code reads shapes at runtime via `tensor.size(dim)`
