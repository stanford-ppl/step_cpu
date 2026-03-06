llama.cpp is an open-source C++ library and inference engine that efficiently runs large language models (LLMs) on standard consumer hardware, including CPUs.
You can find the codebase in /home/ginasohn/llama.cpp.

Read the document that organizes type of optimizations in llama.cpp(/home/ginasohn/llama.cpp/cpu-optimizations.md).
See if the implementation in `_build_gpt2mlp_fused4` (/home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py) can be further optimized by calling llama.cpp implementations or applying similar methodologies.
If so, implement and test that by adding a new fused5.

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

----
Similar to how you optimized for decode phase (M = 1), are there things you can do for the prefill phase too?
Try to leverage the optimizations you did for M=1 case and the ones listed in /home/ginasohn/llama.cpp/cpu-optimizations.md.
If those are applicable to c_fc and c_proj, try to replace torch::mm with a more optimized implementation.