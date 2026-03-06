
The cpp implementation in /home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py L436-L438 (code below) matche the performance of the default PyTorch implementation without any replacement.
However, this implementation still materializes the intermediate tensor between `c_fc - GELU` and `GELU - c_proj`.

Can you generate a new version of a fused implementation that fuses the three operations (c_fc, GELU, c_proj) into a single nested loop?
I assume this would be more performant than the current code since it saves the latency to read and write the intermediate tensors.


```
def _build_gpt2mlp_fused():
    """Compile the hand-written fused GPT2MLP C++ kernel and return the callable."""
    import pathlib as _pl
    import torch.utils.cpp_extension

    print("Building hand-written fused GPT2MLP C++ kernel...")

    _FUSED_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>

torch::Tensor gpt2_mlp_fused_step(
    torch::Tensor hidden_states, torch::Tensor W_fc,
    torch::Tensor b_fc, torch::Tensor W_proj, torch::Tensor b_proj) {

    // c_fc: hidden_states @ W_fc + b_fc  (Conv1D layout: W is [in, out])
    auto h = torch::mm(hidden_states, W_fc);
    h.add_(b_fc);

    // GELU (tanh approximation) — single fused call
    h = at::gelu(h, "tanh");

    // c_proj: h @ W_proj + b_proj
    auto output = torch::mm(h, W_proj);
    output.add_(b_proj);

    return output;
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_mlp_fused_step", gpt2_mlp_fused_step);
}
"""
```

I implement it in a docker container. You can enter it using
```bash
docker exec -it mocha-bg bash
```
Edits on the host in ./step_cpu show up immediately in the container.
Edits in the container under /home/ginasohn/step_cpu write back to the host

In the container you can run the tests using:
```bash
# In the container
source /home/dockeruser/mochaenv/bin/activate
cd /home/dockeruser/step_cpu
python3 darpa/modified/causal_language_modeling.py --mode infer --prompt "Why is the sky blue?" --cpu-only --replace gpt2mlp_fused
```

