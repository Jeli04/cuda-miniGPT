// cublas_gemm.cu
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "tools.h"
#include "layer_norm.h"

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
//     int block_size = 64; 

//     std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
//     // // head 0 weights
//     // std::string file = "block.0.mha.attn_heads.0.query.weight.txt";
//     // std::string source = folder + file;
//     // const float* h_Q_w_0 = loadMatrix(head_dim, d_model, source); // load the data
//     // float* d_Q_w_0;
//     // cudaMalloc(&d_Q_w_0, head_dim * d_model * sizeof(float));
//     // cudaMemcpy(d_Q_w_0, h_Q_w_0, sizeof(float) * head_dim * d_model, cudaMemcpyHostToDevice);


//     std::string file = "qkv_dump.txt";
//     const float* h_input = loadMatrix(block_size, d_model, file); // load the data
//     float* d_input;
//     cudaMalloc(&d_input, block_size * d_model * sizeof(float));
//     cudaMemcpy(d_input, h_input, sizeof(float) * block_size * d_model, cudaMemcpyHostToDevice);

//     // gamma and beta
//     file = "block.0.ln1.weight.txt";
//     std::string source = folder + file;
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
//     cudaMalloc(&d_output, block_size * d_model * sizeof(float));
//     dim3 grid(block_size);       
//     dim3 block(d_model);             
//     size_t shmem = d_model * sizeof(float);  
//     layer_norm<<<grid, block, shmem>>>(
//         d_input,
//         d_output,
//         d_gamma,
//         d_beta,
//         head_dim,
//         d_model
//     );
//     cudaDeviceSynchronize();

//     std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/layernorm.txt";
//     float* h_output = (float*)malloc(head_dim * d_model * sizeof(float));
//     cudaMemcpy(h_output, d_output,
//                head_dim * d_model * sizeof(float),
//                cudaMemcpyDeviceToHost);
//     dumpMatrix(h_output, head_dim, d_model, loc);
// }