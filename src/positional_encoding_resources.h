#pragma once

struct PositionalEncodingResources {
    float* d_token_onehot;
    float* d_pos_onehot;
    float* d_token_embeddings;
    float* d_pos_embeddings;
    float* h_token_onehot;
    float* h_pos_onehot;
    int max_seq_len;
    int vocab_size;
    int d_model;
};

// Function declarations
void initialize_positional_encoding_resources(
    PositionalEncodingResources* resources,
    int max_seq_len,
    int vocab_size,
    int d_model
);

void cleanup_positional_encoding_resources(PositionalEncodingResources* resources);