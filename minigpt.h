#pragma once
#include <vector>

struct PositionalEncodingResources;

struct TransformerWeights {
    const float* d_token_table;
    const float* d_pos_table;

    std::vector<float*> qkv_weights;
    std::vector<float*> mha_proj_weights;
    std::vector<float*> ln1_weights;
    std::vector<float*> ln2_weights;
    std::vector<float*> ffwd_weights;
    std::vector<float*> lnf_weights;
    std::vector<float*> lm_head_weights;

    TransformerWeights(
        const float* d_token_table,
        const float* d_pos_table,
        const std::vector<float*>& qkv_weights,
        const std::vector<float*>& mha_proj_weights,
        const std::vector<float*>& ln1_weights,
        const std::vector<float*>& ln2_weights,
        const std::vector<float*>& ffwd_weights,
        const std::vector<float*>& lnf_weights,
        const std::vector<float*>& lm_head_weights
    );
};

class MiniGPT {
public:
    int block_size;
    int n_heads;
    int d_model;
    int hidden_size;
    int n_blocks;
    int vocab_size;
    TransformerWeights weights;

    MiniGPT(
        int block_size,
        int n_heads,
        int d_model,
        int hidden_size,
        int n_blocks,
        int vocab_size,
        TransformerWeights& weights
    );

    ~MiniGPT();

    void forward_pass(
        int seq_len,
        int max_seq_len,
        const int* h_tokens,
        PositionalEncodingResources& pos_resources,
        float* d_input,
        float* d_output,
        int block_size,
        int n_heads,
        int d_model,
        int head_dim,
        int n_blocks,
        int vocab_size
    );
};