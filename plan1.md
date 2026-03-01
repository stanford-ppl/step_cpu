Now create a Python function with the STeP implementation for GPT2MLP.

Implementation for GPT2MLP:
```
class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
```

To see the configuration used, see /home/ginasohn/research/mocha/distilgpt2_config.json.

Read step.md for necessary information about STeP.

Read /home/ginasohn/step_tl/dyn_tiling/test_mixtral_sweep.py to gain more insight on how STeP nodes can be composed end-to-end.
To see more information about the operators used in this implementation, read /home/ginasohn/step_tl/src/step_py/ops.py.

As STeP requires knowing the tile size and dataflow order for matrix multiplication, use the following:
- Dataflow: Inner-produce dataflow
- Tiling: For each of the two matrix muliplications in c_fc and c_proj, the matrix multiplication will have three dimensions (M,N,K) where the matrix multiplication is `[M,K]*[K,N] = [M,N]`. For M,N,K use tile size M, 256, 256 respectively. 

As we focus on inference only, ignore the `nn.dropout`.

Also, add the stream shape as comments for easier debugging.