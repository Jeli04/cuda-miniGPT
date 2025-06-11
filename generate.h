#pragma once

#include <curand_kernel.h>
#include "minigpt.h"
#include "positional_encoding_resources.h"

// Time measurement function
double get_wall_time();

// CUDA kernel declarations
__global__ void setup_random_states(curandState* states, unsigned long seed, int n);
__global__ void multinomial_sample_kernel(
    const float* probs,
    int* selected_token,
    curandState* states,
    int vocab_size
);

// Token processing functions
int* text_to_tokens(char** vocab, int vocab_size, const char* text, int* num_tokens);

// Main generation function
void generate_tokens_contextual(
    int block_size,
    int d_model,
    int n_heads,
    int head_dim,
    int n_blocks,
    int* input_tokens,
    int input_length,
    int max_new_tokens,
    int vocab_size,
    char** vocab,
    curandState* d_states,
    PositionalEncodingResources& pos_resources,
    MiniGPT& gpt_model
);