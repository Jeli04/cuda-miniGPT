#pragma once

void softmax(float* input, float* output, int rows, int cols);
__global__ void softmax_kernel(float* input, float* output, int rows, int cols);