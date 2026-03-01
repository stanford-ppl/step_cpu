"""
STeP implementation of GPT2MLP (inference only).

GPT2MLP forward pass:
    hidden_states = c_fc(hidden_states)      # [M, 768] @ [768, 3072] + bias
    hidden_states = NewGELU(hidden_states)   # element-wise
    hidden_states = c_proj(hidden_states)    # [M, 3072] @ [3072, 768] + bias

Conv1D(nf, nx):
    weight: [nx, nf]   (note: transposed from nn.Linear)
    bias:   [nf]
    forward: x @ weight + bias

Dataflow:  Inner-product
Tiling:    M_tile = M (full batch), N_tile = 256, K_tile = 256

DistilGPT2 config:
    n_embd = 768
    intermediate_size = 4 * 768 = 3072
    activation_function = gelu_new (tanh-approximation GELU)

Reference: step_py/ops.py (Bufferize, Streamify, Accum, BinaryMap, UnaryMap, Flatten)
Reference: dyn_tiling/test_weight_stationary_gemm.py (Mixtral MLP composition)
"""

import math

import torch

import step


# ---------------------------------------------------------------------------
# Inner-product matmul via STeP operators
#
#   C = A @ B    where A: [M, K], B: [K, N]
#
#   Operator chain:
#     1. tensor_to_stream(A, vec=[M, K_tile])       → stream [1, K/K_tile], tile [M, K_tile]
#     2. Flatten(min_rank=0, max_rank=1)             → stream [K/K_tile], tile [M, K_tile]
#     3. Bufferize(rank=1)                           → stream (), buffer [K/K_tile] of [M, K_tile]
#     4. Streamify(repeat=[N/N_tile], rank=1)        → stream [N/N_tile, K/K_tile], tile [M, K_tile]
#     5. tensor_to_stream(B.T, vec=[N_tile, K_tile]) → stream [N/N_tile, K/K_tile], tile [N_tile, K_tile]
#     6. BinaryMap(Matmul, weight_transposed=True)   → stream [N/N_tile, K/K_tile], tile [M, N_tile]
#     7. Accum(rank=1, fn=Add, init=Zero)            → stream [N/N_tile], tile [M, N_tile]
# ---------------------------------------------------------------------------


def gpt2_mlp_step(hidden_states, W_fc, b_fc, W_proj, b_proj):
    """
    STeP implementation of GPT2MLP.

    Parameters:
        hidden_states: [M, 768]    — input activations
        W_fc:          [768, 3072] — c_fc weight  (Conv1D: [nx, nf])
        b_fc:          [3072]      — c_fc bias
        W_proj:        [3072, 768] — c_proj weight (Conv1D: [nx, nf])
        b_proj:        [768]       — c_proj bias

    Returns:
        output: [M, 768]
    """
    M = hidden_states.shape[0]
    K_tile = 256
    N_tile = 256

    # ==================================================================
    # c_fc: hidden_states @ W_fc + b_fc
    #   [M, K=768] @ [K=768, N=3072] + [3072] = [M, 3072]
    #   K/K_tile = 768/256 = 3,   N/N_tile = 3072/256 = 12
    # ==================================================================

    # --- Step 1: Stream input activations over K dimension ---
    # hidden_states: [M, K=768]
    # vec = [M, K_tile=256]: one tile per K-block, full M in each tile
    act_fc = step.tensor_to_stream(hidden_states, vec=[M, K_tile])
    # stream shape: [1, 3] (tile: [M, 256])

    # --- Step 2: Flatten trivial M-tile dimension ---
    act_fc = step.Flatten(act_fc, min_rank=0, max_rank=1)
    # stream shape: [3] (tile: [M, 256])

    # --- Step 3: Buffer activation for reuse across N tiles ---
    # Inner-product dataflow: each K-tile is reused for every output N-block
    act_fc_buf = step.Bufferize(act_fc, rank=1)
    # stream shape: () — single buffer holding [3] tiles of [M, 256]
    # On-chip footprint: 3 × M × 256 × sizeof(float) = M × 768 × 4 bytes

    # --- Step 4: Replay buffered activations for each N tile ---
    act_fc_rep = step.Streamify(act_fc_buf, repeat_factor=[12], rank=1)
    # stream shape: [12, 3] (tile: [M, 256])
    # For each of 12 output N-blocks, replay all 3 K-tiles

    # --- Step 5: Stream c_fc weight tiles (N-outer, K-inner) ---
    # W_fc stored as [K=768, N=3072]. Transpose to [N=3072, K=768] for
    # N-outer, K-inner iteration order (inner-product dataflow).
    W_fc_T = W_fc.T.contiguous()  # [3072, 768]
    wfc_stream = step.tensor_to_stream(W_fc_T, vec=[N_tile, K_tile])
    # stream shape: [3072/256, 768/256] = [12, 3] (tile: [N_tile=256, K_tile=256])

    # --- Step 6: Tile matmul (partial products) ---
    # Each pair: act[M, K_tile=256] @ weight[N_tile=256, K_tile=256].T
    #          = act[M, 256] @ [256, 256] = [M, 256]
    fc_partial = step.BinaryMap(act_fc_rep, wfc_stream,
                                lambda a, w: torch.mm(a, w.T))
    # stream shape: [12, 3] (tile: [M, N_tile=256])

    # --- Step 7: Accumulate over K dimension (reduction) ---
    # Sum 3 partial products per output N-block:
    #   acc = partial_k0 + partial_k1 + partial_k2
    fc_accum = step.Accum(fc_partial, rank=1)
    # stream shape: [12] (tile: [M, N_tile=256])
    # Each tile is the fully-accumulated result for one N-block of [M, 3072]

    # --- Step 8: Add c_fc bias ---
    # b_fc: [3072] → reshape to [1, 3072] for 2D tiling
    bfc_stream = step.tensor_to_stream(b_fc.unsqueeze(0), vec=[1, N_tile])
    # stream shape: [1, 12] (tile: [1, 256])
    bfc_stream = step.Flatten(bfc_stream, min_rank=0, max_rank=1)
    # stream shape: [12] (tile: [1, 256])

    fc_biased = step.BinaryMap(fc_accum, bfc_stream, lambda a, b: a + b)
    # Broadcasting: [M, 256] + [1, 256] → [M, 256]
    # stream shape: [12] (tile: [M, 256])

    # ==================================================================
    # NewGELU activation (element-wise, fuses into the stream)
    #   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    # ==================================================================

    c_sqrt = math.sqrt(2.0 / math.pi)  # 0.7978845608028654
    c_pow = 0.044715

    # --- Step 9: x³ ---
    x3 = step.UnaryMap(fc_biased, lambda a: torch.pow(a, 3.0))
    # stream shape: [12] (tile: [M, 256])

    # --- Step 10: x + 0.044715 * x³ ---
    x_in = step.BinaryMap(fc_biased, x3, lambda a, b: a + c_pow * b)
    # stream shape: [12] (tile: [M, 256])

    # --- Step 11: sqrt(2/π) * (x + 0.044715 * x³) ---
    t_in = step.UnaryMap(x_in, lambda a: c_sqrt * a)
    # stream shape: [12] (tile: [M, 256])

    # --- Step 12: tanh(...) ---
    t = step.UnaryMap(t_in, lambda a: torch.tanh(a))
    # stream shape: [12] (tile: [M, 256])

    # --- Step 13: 1 + tanh(...) ---
    onep = step.UnaryMap(t, lambda a: 1.0 + a)
    # stream shape: [12] (tile: [M, 256])

    # --- Step 14: 0.5 * x * (1 + tanh(...)) ---
    gelu_out = step.BinaryMap(fc_biased, onep, lambda a, b: 0.5 * a * b)
    # stream shape: [12] (tile: [M, 256])
    #
    # Note: fc_biased feeds 3 consumers (steps 9, 10, 14). In the STeP
    # hardware model, output FIFOs replicate tiles to each consumer.
    # The GELU chain fuses entirely — no intermediate memory writes.

    # ==================================================================
    # c_proj: gelu_out @ W_proj + b_proj
    #   [M, K=3072] @ [K=3072, N=768] + [768] = [M, 768]
    #   K/K_tile = 3072/256 = 12,   N/N_tile = 768/256 = 3
    # ==================================================================

    # --- Step 15: Buffer GELU output for reuse across c_proj N tiles ---
    # This is a synchronization point: all 12 GELU tiles must be produced
    # before c_proj can begin (Bufferize collects the full stream).
    proj_buf = step.Bufferize(gelu_out, rank=1)
    # stream shape: () — single buffer holding [12] tiles of [M, 256]
    # On-chip footprint: 12 × M × 256 × sizeof(float) = M × 3072 × 4 bytes

    # --- Step 16: Replay for each c_proj output N-block ---
    proj_act_rep = step.Streamify(proj_buf, repeat_factor=[3], rank=1)
    # stream shape: [3, 12] (tile: [M, 256])
    # For each of 3 output N-blocks, replay all 12 K-tiles

    # --- Step 17: Stream c_proj weight tiles ---
    # W_proj stored as [K=3072, N=768]. Transpose for inner-product order.
    W_proj_T = W_proj.T.contiguous()  # [768, 3072]
    wproj_stream = step.tensor_to_stream(W_proj_T, vec=[N_tile, K_tile])
    # stream shape: [768/256, 3072/256] = [3, 12] (tile: [N_tile=256, K_tile=256])

    # --- Step 18: Tile matmul (partial products) ---
    proj_partial = step.BinaryMap(proj_act_rep, wproj_stream,
                                  lambda a, w: torch.mm(a, w.T))
    # act[M, K_tile=256] @ weight[N_tile=256, K_tile=256].T = [M, N_tile=256]
    # stream shape: [3, 12] (tile: [M, 256])

    # --- Step 19: Accumulate over K dimension ---
    proj_accum = step.Accum(proj_partial, rank=1)
    # stream shape: [3] (tile: [M, N_tile=256])

    # --- Step 20: Add c_proj bias ---
    bproj_stream = step.tensor_to_stream(b_proj.unsqueeze(0), vec=[1, N_tile])
    # stream shape: [1, 3] (tile: [1, 256])
    bproj_stream = step.Flatten(bproj_stream, min_rank=0, max_rank=1)
    # stream shape: [3] (tile: [1, 256])

    proj_biased = step.BinaryMap(proj_accum, bproj_stream, lambda a, b: a + b)
    # Broadcasting: [M, 256] + [1, 256] → [M, 256]
    # stream shape: [3] (tile: [M, 256])

    # --- Step 21: Materialize output tensor ---
    output = step.stream_to_tensor(proj_biased, like_tensor=hidden_states)
    # output: [M, 768]

    return output


# ---------------------------------------------------------------------------
# Summary: Operator count and on-chip memory
#
# Operators:
#   tensor_to_stream ×4 (hidden_states, W_fc.T, W_proj.T, biases ×2)
#   Flatten          ×4
#   Bufferize        ×2 (activation reuse between c_fc→GELU and GELU→c_proj)
#   Streamify        ×2
#   BinaryMap(Matmul)×2 (c_fc, c_proj)
#   Accum            ×2 (c_fc, c_proj K-reduction)
#   BinaryMap(Add)   ×2 (bias additions)
#   UnaryMap         ×4 (GELU: pow, scale, tanh, add-one)
#   BinaryMap        ×3 (GELU: x+c*x3, 0.5*x*onep; plus x reuse)
#   stream_to_tensor ×1
#
# On-chip buffer requirements:
#   c_fc activation buffer:   M × 768 × 4 bytes   (hidden_states, 3 tiles)
#   c_proj activation buffer: M × 3072 × 4 bytes  (post-GELU, 12 tiles)
#   Tile working set per stage: O(M × 256 × 4) bytes
#
# Data movement (inner-product dataflow):
#   W_fc:   768 × 3072 × 4 bytes  = 9.0 MB  (streamed once from DRAM)
#   W_proj: 3072 × 768 × 4 bytes  = 9.0 MB  (streamed once from DRAM)
#   Input:  M × 768 × 4 bytes     (loaded once, buffered on-chip)
#   Output: M × 768 × 4 bytes     (written once to DRAM)
# ---------------------------------------------------------------------------
