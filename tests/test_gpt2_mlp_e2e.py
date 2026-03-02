"""End-to-end test: compile + run GPT2 MLP, check numerics."""

import math

import torch

import step


def gpt2_mlp_step(hidden_states, W_fc, b_fc, W_proj, b_proj):
    M = hidden_states.shape[0]
    K_tile = 256
    N_tile = 256

    # c_fc matmul: hidden_states @ W_fc + b_fc
    act_fc = step.tensor_to_stream(hidden_states, vec=[M, K_tile])
    act_fc = step.Flatten(act_fc, min_rank=0, max_rank=1)
    act_fc_buf = step.Bufferize(act_fc, rank=1)
    act_fc_rep = step.Streamify(act_fc_buf, repeat_factor=[12], rank=1)

    W_fc_T = W_fc.T.contiguous()
    wfc_stream = step.tensor_to_stream(W_fc_T, vec=[N_tile, K_tile])

    fc_partial = step.BinaryMap(act_fc_rep, wfc_stream,
                                lambda a, w: torch.mm(a, w.T))
    fc_accum = step.Accum(fc_partial, rank=1)

    bfc_stream = step.tensor_to_stream(b_fc.unsqueeze(0), vec=[1, N_tile])
    bfc_stream = step.Flatten(bfc_stream, min_rank=0, max_rank=1)

    fc_biased = step.BinaryMap(fc_accum, bfc_stream, lambda a, b: a + b)

    # GELU
    c_sqrt = math.sqrt(2.0 / math.pi)
    c_pow = 0.044715

    x3 = step.UnaryMap(fc_biased, lambda a: torch.pow(a, 3.0))
    x_in = step.BinaryMap(fc_biased, x3, lambda a, b: a + c_pow * b)
    t_in = step.UnaryMap(x_in, lambda a: c_sqrt * a)
    t = step.UnaryMap(t_in, lambda a: torch.tanh(a))
    onep = step.UnaryMap(t, lambda a: 1.0 + a)
    gelu_out = step.BinaryMap(fc_biased, onep, lambda a, b: 0.5 * a * b)

    # c_proj matmul: gelu_out @ W_proj + b_proj
    proj_buf = step.Bufferize(gelu_out, rank=1)
    proj_act_rep = step.Streamify(proj_buf, repeat_factor=[3], rank=1)

    W_proj_T = W_proj.T.contiguous()
    wproj_stream = step.tensor_to_stream(W_proj_T, vec=[N_tile, K_tile])

    proj_partial = step.BinaryMap(proj_act_rep, wproj_stream,
                                  lambda a, w: torch.mm(a, w.T))
    proj_accum = step.Accum(proj_partial, rank=1)

    bproj_stream = step.tensor_to_stream(b_proj.unsqueeze(0), vec=[1, N_tile])
    bproj_stream = step.Flatten(bproj_stream, min_rank=0, max_rank=1)

    proj_biased = step.BinaryMap(proj_accum, bproj_stream, lambda a, b: a + b)

    output = step.stream_to_tensor(proj_biased, like_tensor=hidden_states)
    return output


# Set param ndims: hidden_states=2D, W_fc=2D, b_fc=1D, W_proj=2D, b_proj=1D
gpt2_mlp_step._param_ndims = [2, 2, 1, 2, 1]

compiled_mlp = step.cpu_compile(gpt2_mlp_step)


def _reference_gpt2_mlp(hidden_states, W_fc, b_fc, W_proj, b_proj):
    """Pure PyTorch reference implementation."""
    # c_fc: x @ W + b
    h = hidden_states @ W_fc + b_fc
    # NewGELU
    c = math.sqrt(2.0 / math.pi)
    h = 0.5 * h * (1.0 + torch.tanh(c * (h + 0.044715 * torch.pow(h, 3.0))))
    # c_proj
    h = h @ W_proj + b_proj
    return h


class TestGpt2MlpE2E:
    def _check(self, M: int):
        torch.manual_seed(42)
        hidden = torch.randn(M, 768, dtype=torch.float32)
        W_fc = torch.randn(768, 3072, dtype=torch.float32) * 0.02
        b_fc = torch.randn(3072, dtype=torch.float32) * 0.02
        W_proj = torch.randn(3072, 768, dtype=torch.float32) * 0.02
        b_proj = torch.randn(768, dtype=torch.float32) * 0.02

        result = compiled_mlp(hidden, W_fc, b_fc, W_proj, b_proj)
        expected = _reference_gpt2_mlp(hidden, W_fc, b_fc, W_proj, b_proj)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-3)

    def test_M4(self):
        self._check(4)

    def test_M16(self):
        self._check(16)
