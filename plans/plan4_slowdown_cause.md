The performance of running the replaced GPT2MLP is 10 times slower than the default PyTorch version. However, in the STeP implementation, it fuses the addmm, activation, and addmm which I believe should give a performance boost.

Before trying to optimize things, I want to understand the cause first. Focus on reporting the potential causes.

1. Identify why it is so slow.
2. Does the tensor to stream transformation cause inefficiencies?

Location for the code generation logic: /home/ginasohn/research/mocha/step/codegen.py

How to run code:
```
conda activate py312clean
source /home/ginasohn/research/mocha/.venv/bin/activate
python3 darpa/modified/causal_language_modeling.py --mode infer --prompt "The performance of running the replaced GPT2MLP is 10 times slower than the default PyTorch version. However, in the STeP implementation, it fuses the addmm, activation, and addmm which I believe should give a performance boost.

Before trying to optimize things, I want to understand the cause first. Focus on reporting the potential causes.

1. Identify why it is so slow.
2. Does the tensor to stream transformation cause inefficiencies?" --cpu-only --replace gpt2mlp
```