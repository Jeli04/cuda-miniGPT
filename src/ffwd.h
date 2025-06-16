#pragma once

void ffwd(
    float* d_input, 
    int block_size,
    int d_model,
    int hidden_size,
    const float* d_bias1,
    const float* d_weights1,
    const float* d_bias2,
    const float* d_weights2
);

__global__ void add_bias(
    const float* input, 
    const float* bias, 
    float* output, 
    int total, 
    int cols
);
