1. See if the attn_fused is okay with large sequences
=> Fix the script or prompt to use

2. I don't have an idea on how to express the other parts in STePProgram, so my plan is 

For the `GPT2AttentionStepWrapper`, write a 
```
std::vector<torch::Tensor> gpt2_attn_fused_step(
    torch::Tensor x,
    torch::Tensor W_attn,
    torch::Tensor b_attn,
    torch::Tensor W_proj,
    torch::Tensor b_proj,
    torch::Tensor past_key,
    torch::Tensor past_value,
    torch::Tensor attn_mask,
    int64_t num_heads,
    int64_t batch_size
) {
    const int64_t M = x.size(0);
    const int64_t embed_dim = x.size(1);
    const int64_t head_dim = embed_dim / num_heads;
    const int64_t S = M / batch_size;

    // 1. QKV projection with AVX-512 GEMM/GEMV
    auto qkv = torch::empty({M, 3 * embed_dim}, x.options());
    if (M == 1) {
        avx512_omp_gemv(x.data_ptr<float>(), W_attn.data_ptr<float>(),
                        b_attn.data_ptr<float>(), qkv.data_ptr<float>(),
                        embed_dim, 3 * embed_dim);
    } else {
        avx512_omp_gemm(x.data_ptr<float>(), W_attn.data_ptr<float>(),
                        b_attn.data_ptr<float>(), qkv.data_ptr<float>(),
                        M, embed_dim, 3 * embed_dim);
    }

    // 2. Split Q, K, V
    auto qkv_split = qkv.chunk(3, 1);
    auto Q = qkv_split[0].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto K = qkv_split[1].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto V = qkv_split[2].reshape({batch_size, S, num_heads, head_dim}).permute({0, 2, 1, 3});

    // 3. Concatenate past KV
    bool has_past = past_key.numel() > 0;
    if (has_past) {
        K = torch::cat({past_key, K}, 2);
        V = torch::cat({past_value, V}, 2);
    }

    auto present_key = K.clone();
    auto present_value = V.clone();

    const int64_t total_len = K.size(2);

    // 4. Attention scores (still using torch ops for inner attention)
    auto K_t = K.transpose(-2, -1).contiguous();
    auto scores = torch::matmul(Q.contiguous(), K_t);
    float scale = 1.0f / std::sqrt((float)head_dim);
    scores.mul_(scale);

    // 5. Causal mask
    {
        auto causal = torch::full({S, total_len}, -1e4, scores.options());
        for (int64_t i = 0; i < S; i++) {
            int64_t max_pos = total_len - S + i;
            for (int64_t j = 0; j <= max_pos && j < total_len; j++) {
                causal[i][j] = 0.0f;
            }
        }
        scores.add_(causal.unsqueeze(0).unsqueeze(0));
    }

    // 6. Add HF mask
    if (attn_mask.numel() > 0) scores.add_(attn_mask);

    // 7. Softmax
    auto attn_weights = at::softmax(scores, -1);

    // 8. Context
    auto context = torch::matmul(attn_weights, V.contiguous());
    auto merged = context.permute({0, 2, 1, 3}).contiguous().reshape({M, embed_dim});

    // 9. Output projection with AVX-512
    auto output = torch::empty({M, embed_dim}, x.options());
    if (M == 1) {
        avx512_omp_gemv(merged.data_ptr<float>(), W_proj.data_ptr<float>(),
                        b_proj.data_ptr<float>(), output.data_ptr<float>(),
                        embed_dim, embed_dim);
    } else {
        avx512_omp_gemm(merged.data_ptr<float>(), W_proj.data_ptr<float>(),
                        b_proj.data_ptr<float>(), output.data_ptr<float>(),
                        M, embed_dim, embed_dim);
    }

    return {output, present_key, present_value};
}

TORCH_LIBRARY_FRAGMENT(step_ops, m) {
    m.def("gpt2_attn_fused_step", gpt2_attn_fused_step);
}
```

--------
1. Get MLP done
    AvxcodegenPrefillDecode(prefill_step, decode_step) => Fills in the condition in cpp
2. Get ATtention done
3. Use the test dataset or write the prompt lengths that are realistic and better.