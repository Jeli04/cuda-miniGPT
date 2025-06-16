#pragma once

void basicSgemm(int m, int n, int k, bool A_t, bool B_t, const float *A, const float *B, float *C);
__global__ void mysgemm(int m, int n, int k, bool A_t, bool B_t, const float *A, const float *B, float* C);