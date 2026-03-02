Now add Flatten, Bufferize, Streamify, and Accum operators to the step module.

Update the code generator you previously made for the /home/ginasohn/research/mocha/examples/gelu_kernel.py example so that is works for /home/ginasohn/research/mocha/examples/gpt2_mlp_step.py too.

We will precompile the generated c++ code for the step kernel (/home/ginasohn/research/mocha/examples/gpt2_mlp_step.py) and replace the previous call to a torch.nn.module with our PyTorch module that calls this precompiled kernel. We will use precompiled CPU library + PyTorch C++ extension.