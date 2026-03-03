Read /home/ginasohn/research/mocha/darpa/modified/summary.md to understand the inefficiencies in the current codegen and try to fix them.

Here are some suggestions (but take it with a grain of salt and you can figure out how to solve it yourself too):

1. For the `tensor_to_stream`, would adding stride and shape field help avoid copying the weights? Also, if changing the dataflow order helps, you can update the STeP implementation accodingly.
2. /home/ginasohn/research/llama.cpp has optimized cpp implementations for LLMs. Try to use their utility functions or use their implementation as reference.
3. At the end of the day, I want to update the code generator so that I can generate optimized fused implementation from the STeP program. To do so, maybe first implementing an optimized fused GPT2MLP implementation for CPU first and trying to figure out how to update the STeP program and code generation can be a better approach.
