// cublas_gemm.cu
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>


__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // One float per warp
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); // Each warp reduces its values

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val); // Final reduction by first warp

    return val;
}

__global__ void layer_norm(float* input, float* output, const float* gamma, const float* beta, int num_rows, int num_cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    extern __shared__ float shared_stats[];  // use dynamically allocated shared memory
    float* shared_mean = &shared_stats[0];
    float* shared_var  = &shared_stats[1];

    float val = 0.0f;
    if (col < num_cols)
        val = input[row * num_cols + col];

    float sum = blockReduceSum(col < num_cols ? val : 0.0f);
    if (col == 0) shared_mean[0] = sum / num_cols;
    __syncthreads();
    float mean = shared_mean[0];

    float diff = (col < num_cols) ? val - mean : 0.0f;
    float sq_diff = diff * diff;
    float var_sum = blockReduceSum(col < num_cols ? sq_diff : 0.0f);
    if (col == 0) shared_var[0] = var_sum / num_cols;
    __syncthreads();
    float variance = shared_var[0];

    if (col < num_cols) {
        float norm = (val - mean) / sqrtf(variance + 1e-5f);
        output[row * num_cols + col] = gamma[col] * norm + beta[col];
    }
}

