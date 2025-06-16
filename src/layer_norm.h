#pragma once

__global__ void layer_norm(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int rows,
    int cols
);