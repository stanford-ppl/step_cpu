Consider you are a genius compiler expert in writing c++ lowering and code generators and also are famililar with using einsum notations.

Now I want to generate `_build_gpt2mlp_fused6` from a STeP program.

STeP is a stream-centric abstraction targetting dataflow hardware. However, I am trying to generate CPU code from a STeP representation. 
On dataflow hardware, STeP uses streams of tiles with stop tokens for synchronization. On CPU, streams become **(potentially tiled/vectorized) loop nests** over contiguous tensors. For CPUs,
- **Stream rank** = number of loop nesting levels
- **Tile shape** = the chunk size at each level (the `vec` parameter)
- **Stop tokens** = loop bounds derived from `ceil(dim_size / tile_size)` at each level

For more information on the operators, read section 3 in /home/ginasohn/step_cpu/assets/STeP_ASPLOS_2026_camera_ready.pdf.

Previously I have tried some experiments to do this, where the codegen can be found in /home/ginasohn/step_cpu/step/codegen.py. However this is pretty outdated, so skim it and if you don't think it's that relevant, don't waste tokens reading this.

I want to generate optimized code shown in `_build_gpt2mlp_fused6`(/home/ginasohn/step_cpu/darpa/modified/causal_language_modeling_mlp.py) from the `step_impl` function in /home/ginasohn/step_cpu/darpa/modified/step_kernels.py.
To get something working first, you can ignore handling cases where there are remainders because the dimension is not an exact multiple of some factor and prefetching. We can integrate them later. But you should try to keep the other performance critical syntaxes like SIMD instructions and openMP parallellizations.

To make the nested loop instantiation easier, I added index variable information to each buffer and operator.
I was thinking of a lowering method like this:
1. Look at the dataflow order and whether they are parallelized. If a certain dim is parallelized, generate proper OpenMP code.
2. Once parallelized, for the code to run in each thread, instantiate nested for loop for each index variable. At the high-level this would be (1) Instantiate nested for loops for the index variables based on the dataflow order, (2) Recursively visit the parent STeP node and stage them in the right loop level based on the iteration space info and data dependencies. 
In more detail, visit from the last node in the STeP graph. Recursively visit the parents and once all the parents are all visited, stage the current node in the right loop level. For example, LinearStore for the bias only contains the index variable `n` in its iteration space. So, it can be staged in the `for` loop for `n` and outside of the `for` loop for `k`. In the `for` loop for `n`, it has several positions it can go. This will be determined based on the data-dependency. As the parent is the BinaryMapAccum, where the input operands's iteration sapce holds the `k` index variable, this has to be placed in the innermost loop for the `k` dimension. Therefore, the code for `LinearStore` (which will be the `_mm512_storeu_ps`s) will come after the loop for the `k` dimension.


**You can change how the STeP program is written.If you need the STeP program to be written in a more specific way, do so and introduce new classes if necessary.**
Also, some of the functions assume the shape transformation happens implicitly. For example, in the BinaryMapAccum function, I'm expecting the code generator will notice
1. x is [1], so to do elementwise multiplication with a [1,64] shape weight
2. Then I first need to broadcast x with `_mm512_set1_ps` and split the weight into four `_mm512_loadu_ps` operator.
3. And then use xk to do `_mm512_fmadd_ps` as the lambda function expresses a fmadd.
However, if this is too much heavy lifting, feel free to make scalar to [16] avx vector broadcast or staging 4 avx vectors explicit by updating the /home/ginasohn/step_cpu/darpa/modified/step_kernels.py.

If you update things, state why and what you changed.


To run tests, read below:
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
python3 darpa/modified/causal_language_modeling_mlp.py --mode infer --prompt "Why is the sky blue?" --cpu-only --replace gpt2mlp_fused6
```