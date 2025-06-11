#pragma once
#include <vector>

void transformer_decoder(
    float* d_input,
    float* d_output,
    float* residual_copy,
    int block_size,
    int n_heads,
    int d_model,
    int head_dim,
    int n_blocks,
    int vocab_size,
    const std::vector<float*>& qkv_weights,
    const std::vector<float*>& mha_proj_weights,
    const std::vector<float*>& ln1_weights,
    const std::vector<float*>& ln2_weights,
    const std::vector<float*>& ffwd_weights,
    const std::vector<float*>& lnf_weights,
    const std::vector<float*>& lm_head_weights
);