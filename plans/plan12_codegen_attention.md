## Goal
Through plan 10, I wrote STeP programs (/home/ginasohn/step_cpu/darpa/modified/step_kernels.py) that is equivalent to the c++ implementation in _apply_gpt2mlp_fused6 (
/home/ginasohn/step_cpu/darpa/modified/causal_language_modeling_codegen.py) and was able to codegen a similar c++ code from the STeP implementation.

You can see the code for the code generated implementation of GEMV and GEMM by:
```bash
# Enter the container
docker exec -it mocha-bg bash -c 
# In the container:
cat /root/.cache/mocha/gpt2_mlp_fused6/gpt2_mlp_fused6_step.cpp
```


Since the _build_gpt2attn_fused (
/home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py) uses similar GEMV and GEMM code generated from the codegen pass, I would like to generate that part of the code from the STeP implementation.

Since I currently only have code for generating gemv and gemm, you can keep parts of the cpp code in _build_gpt2attn_fused (/home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py) for the other parts outside of gemv and gemm.

In /home/ginasohn/step_cpu/darpa/modified/causal_language_modeling_codegen.py, implement a GPT2AttentionStepWrapper and functions to build and apply the kernel compiled from the combination of (1) code generated from gemv and gemm like in gpt2mlp_fused6 and (2) code from below (brought from _build_gpt2attn_fused in /home/ginasohn/step_cpu/darpa/modified/causal_language_modeling.py) for the other parts outside of gemv and gemm.
Since you're using the cpp code in _build_gpt2attn_fused for parts other than gemv and gemm, you can hardcode those parts for now.


## How to Run

Enter the container
```
docker exec -it mocha-bg bash
```
In the container:
```
source /home/dockeruser/mochaenv/bin/activate
cd /home/dockeruser/step_cpu
PYTHONPATH=/home/dockeruser/step_cpu:\$PYTHONPATH \
python3 darpa/modified/causal_language_modeling_mlp.py --mode infer --prompt 'Why is the sky blue?' --cpu-only --replace gpt2mlp_fused6
```
