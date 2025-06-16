#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "positional_encoding_resources.h"

void initialize_positional_encoding_resources(
    PositionalEncodingResources* resources,
    int max_seq_len,
    int vocab_size,
    int d_model
) {
    resources->max_seq_len = max_seq_len;
    resources->vocab_size = vocab_size;
    resources->d_model = d_model;
    
    cudaMalloc(&resources->d_token_onehot, max_seq_len * vocab_size * sizeof(float));
    cudaMalloc(&resources->d_pos_onehot, max_seq_len * max_seq_len * sizeof(float));
    cudaMalloc(&resources->d_token_embeddings, max_seq_len * d_model * sizeof(float));
    cudaMalloc(&resources->d_pos_embeddings, max_seq_len * d_model * sizeof(float));
    
    resources->h_token_onehot = (float*)calloc(max_seq_len * vocab_size, sizeof(float));
    resources->h_pos_onehot = (float*)calloc(max_seq_len * max_seq_len, sizeof(float));
    
    printf("Positional encoding resources initialized for max_seq_len=%d, vocab_size=%d, d_model=%d\n", 
           max_seq_len, vocab_size, d_model);
}

void cleanup_positional_encoding_resources(PositionalEncodingResources* resources) {
    cudaFree(resources->d_token_onehot);
    cudaFree(resources->d_pos_onehot);
    cudaFree(resources->d_token_embeddings);
    cudaFree(resources->d_pos_embeddings);
    free(resources->h_token_onehot);
    free(resources->h_pos_onehot);
}