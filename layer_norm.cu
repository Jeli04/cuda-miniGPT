// cublas_gemm.cu
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "tools.cu"


// __inline__ __device__ float warpReduceSum(float val) {
//     for (int offset = warpSize/2; offset > 0; offset >>= 1)
//         val += __shfl_down_sync(0xffffffff, val, offset);
//     return val;
// }

// __inline__ __device__ float blockReduceSum(float val) {
//     static __shared__ float shared[32]; // One float per warp
//     int lane = threadIdx.x % warpSize;
//     int wid = threadIdx.x / warpSize;

//     val = warpReduceSum(val); // Each warp reduces its values

//     if (lane == 0) shared[wid] = val; // Write reduced value to shared memory
//     __syncthreads();

//     val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
//     if (wid == 0) val = warpReduceSum(val); // Final reduction by first warp

//     return val;
// }

// __global__ void layer_norm(float* input, float* output, const float* gamma, const float* beta, int num_rows, int num_cols) {
//     int row = blockIdx.x;
//     int col = threadIdx.x;

//     extern __shared__ float shared_stats[];  // use dynamically allocated shared memory
//     float* shared_mean = &shared_stats[0];
//     float* shared_var  = &shared_stats[1];

//     float val = 0.0f;
//     if (col < num_cols)
//         val = input[row * num_cols + col];

//     float sum = blockReduceSum(col < num_cols ? val : 0.0f);
//     if (col == 0) shared_mean[0] = sum / num_cols;
//     __syncthreads();
//     float mean = shared_mean[0];

//     float diff = (col < num_cols) ? val - mean : 0.0f;
//     float sq_diff = diff * diff;
//     float var_sum = blockReduceSum(col < num_cols ? sq_diff : 0.0f);
//     if (col == 0) shared_var[0] = var_sum / num_cols;
//     __syncthreads();
//     float variance = shared_var[0];

//     if (col < num_cols) {
//         float norm = (val - mean) / sqrtf(variance + 1e-5f);
//         output[row * num_cols + col] = gamma[col] * norm + beta[col];
//     }
// }


__global__ void layer_norm(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int rows,
    int cols
) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    const float* x = input + row * cols;

    // compute mean
    float s = 0.0f;
    for (int i = tid; i < cols; i += stride) {
        s += x[i];
    }
    shared[tid] = s;
    for (int offset = stride / 2; offset > 0; offset /= 2) {
        __syncthreads();
        if (tid < offset) shared[tid] += shared[tid + offset];
    }
    __syncthreads();
    float mean = shared[0] / cols;

    // compute variance
    float vs = 0.0f;
    for (int i = tid; i < cols; i += stride) {
        float d = x[i] - mean;
        vs += d * d;
    }
    shared[tid] = vs;
    for (int offset = stride / 2; offset > 0; offset /= 2) {
        __syncthreads();
        if (tid < offset) shared[tid] += shared[tid + offset];
    }
    __syncthreads();
    float var = shared[0] / cols;

    // normalize + scale/shift
    for (int i = tid; i < cols; i += stride) {
        float norm = (x[i] - mean) / sqrtf(var + 1e-5f);
        output[row * cols + i] = gamma[i] * norm + beta[i];
    }
}

// int main(){
//     int d_model = 128;
//     int head_dim = 16;

//     std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
//     // head 0 weights
//     std::string file = "block.0.mha.attn_heads.0.query.weight.txt";
//     std::string source = folder + file;
//     const float* h_Q_w_0 = loadMatrix(head_dim, d_model, source); // load the data
//     float* d_Q_w_0;
//     cudaMalloc(&d_Q_w_0, head_dim * d_model * sizeof(float));
//     cudaMemcpy(d_Q_w_0, h_Q_w_0, sizeof(float) * head_dim * d_model, cudaMemcpyHostToDevice);

//     // gamma and beta
//     file = "block.0.ln1.weight.txt";
//     source = folder + file;
//     float* h_gamma = loadMatrix(d_model, 1, source); // load the data
//     float* d_gamma;
//     cudaMalloc(&d_gamma, d_model * sizeof(float));
//     cudaMemcpy(d_gamma, h_gamma, sizeof(float) * d_model, cudaMemcpyHostToDevice);
//     // printMatrix(h_gamma, d_model, 1);

//     file = "block.0.ln1.bias.txt";
//     source = folder + file;
//     float* h_beta = loadMatrix(d_model, 1, source); // load the data
//     float* d_beta;
//     cudaMalloc(&d_beta, d_model * sizeof(float));
//     cudaMemcpy(d_beta, h_beta, sizeof(float) * d_model, cudaMemcpyHostToDevice);

//     float* d_output;
//     cudaMalloc(&d_output, head_dim * d_model * sizeof(float));
//     dim3 grid(head_dim);       
//     dim3 block(d_model);             
//     size_t shmem = d_model * sizeof(float);  
//     layer_norm<<<grid, block, shmem>>>(
//         d_Q_w_0,
//         d_output,
//         d_gamma,
//         d_beta,
//         head_dim,
//         d_model
//     );
//     cudaDeviceSynchronize();

//     std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
//     float* h_output = (float*)malloc(head_dim * d_model * sizeof(float));
//     cudaMemcpy(h_output, d_output,
//                head_dim * d_model * sizeof(float),
//                cudaMemcpyDeviceToHost);
//     dumpMatrix(h_output, head_dim, d_model, loc);
// }