#include "minigpt.h"
#include "transformer_block.h"
#include "positional_encoding.h"
#include "tools.h"

TransformerWeights::TransformerWeights(
    const float* d_token_table,
    const float* d_pos_table,
    const std::vector<float*>& qkv_weights,
    const std::vector<float*>& mha_proj_weights,
    const std::vector<float*>& ln1_weights,
    const std::vector<float*>& ln2_weights,
    const std::vector<float*>& ffwd_weights,
    const std::vector<float*>& lnf_weights,
    const std::vector<float*>& lm_head_weights
)
    : d_token_table(d_token_table),
      d_pos_table(d_pos_table),
      qkv_weights(qkv_weights),
      mha_proj_weights(mha_proj_weights),
      ln1_weights(ln1_weights),
      ln2_weights(ln2_weights),
      ffwd_weights(ffwd_weights),
      lnf_weights(lnf_weights),
      lm_head_weights(lm_head_weights)
{}

MiniGPT::MiniGPT(
    int block_size,
    int n_heads,
    int d_model,
    int hidden_size,
    int n_blocks,
    int vocab_size,
    TransformerWeights& weights
)
    : block_size(block_size),
      n_heads(n_heads),
      d_model(d_model),
      hidden_size(hidden_size),
      n_blocks(n_blocks),
      vocab_size(vocab_size),
      weights(weights)
{}

MiniGPT::~MiniGPT() {
    // cleanup if needed
}

void MiniGPT::forward_pass(
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
) {
    embed_sequence_sgemm(
        d_input,
        weights.d_token_table,
        weights.d_pos_table,
        h_tokens,
        seq_len,
        d_model,
        vocab_size,
        max_seq_len,
        &pos_resources
    );

    // create residual copy
    float* residual_copy; // for residual layer later
    cudaMalloc(&residual_copy, sizeof(float)* block_size*d_model);
    cudaMemcpy(residual_copy, d_input, sizeof(float)* block_size*d_model, cudaMemcpyDeviceToDevice);
    

    transformer_decoder(
        d_input,
        d_output,
        residual_copy,
        block_size,
        n_heads,
        d_model,
        head_dim,
        n_blocks,
        vocab_size,
        weights.qkv_weights,
        weights.mha_proj_weights,
        weights.ln1_weights,
        weights.ln2_weights,
        weights.ffwd_weights,
        weights.lnf_weights,
        weights.lm_head_weights
    );
}