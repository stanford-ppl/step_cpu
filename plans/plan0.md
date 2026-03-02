We want to implement a code generator in Python that will generate optimized C++ code of the STeP abstraction.

We will precompile the generated c++ code for a PyTorch module and replace the previous call to a torch.nn.module with our PyTorch module that calls this precompiled kernel that is created by lowering a STeP program that corresponds to the torch.nn.module.

We will use precompiled CPU library + PyTorch C++ extension (pybind / torch::Library) to do this (preferably via torch::Library custom op rather than raw pybind calls).

Read the STeP paper: /home/ginasohn/research/mocha/STeP_ASPLOS_2026_camera_ready.pdf

STeP simulator location: 
* /home/ginasohn/step_tl/step-perf/src/operator
* /home/ginasohn/step_tl/step-perf/src/memory


Since we do not use stop tokens in the CPU implementation of STeP nodes, consider them as loop bounds derived from the input stream/tensor shape.

Just for more context, we are hoping to feed in a STeP function like this and generate a c++ implementation by generating code for each STeP operator.
```
def gelu_kernel(input: torch.Tensor):
    vec = [1, 32]
    x = step.tensor_to_stream(input, vec)

    c_sqrt = math.sqrt(2.0 / math.pi)
    c_pow  = 0.044715

    x3   = UnaryMap(x,  lambda a: torch.pow(a, 3.0))
    x_in = BinaryMap(x, x3, lambda a, b: a + c_pow * b)          # x + 0.044715*x^3
    t_in = UnaryMap(x_in, lambda a: c_sqrt * a)                  # sqrt(2/pi) * ...
    t    = UnaryMap(t_in, lambda a: torch.tanh(a))               # tanh(...)
    onep = UnaryMap(t,    lambda a: 1.0 + a)                     # 1 + tanh(...)
    y    = BinaryMap(x, onep, lambda a, b: 0.5 * a * b)           # 0.5*x*(...)

    return step.stream_to_tensor(y, like_tensor=input)

step.cpu_compile(gelu_kernel)

def cpu_compile(f): (Python function => C++ code)
	traverse(f) => Implement a Python AST visitor for each STeP operator 
```
(`tensor_to_stream` converts tensor into a stream (with a specific stream shape and tile size assuming row-major order). Can add additional fields like layout if that’s necessary for codegen.)



Generate a step.md to generate the semantics that will be used to generate c++ implementation of each STeP nodes.