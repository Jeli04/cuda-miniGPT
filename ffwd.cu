#include <stdio.h>
#include "sgemm.h"
#include "tools.h"
#include "ffwd.h"

#define TILE_SIZE 64

__global__ void add_bias(const float* input, const float* bias, float* output, int total, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int i = idx % cols;
        output[idx] = input[idx] + bias[i];
    }
}

__global__ void relu(float* a, int total_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = a[idx];
        a[idx] = (val < 0.0f) ? 0.0f : val;
    }
}

void ffwd(
    float* d_input, 
    int block_size,
    int d_model,
    int hidden_size,
    const float* d_bias1,
    const float* d_weights1,
    const float* d_bias2,
    const float* d_weights2
){
    const unsigned int BLOCK_SIZE = TILE_SIZE;
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE); 

    // First layer: input_size x hidden_size
    float* d_hidden_output;
    cudaMalloc(&d_hidden_output, sizeof(float)* block_size* hidden_size);
    mysgemm<<<dim_grid, dim_block>>>(block_size, hidden_size, d_model, false, true, d_input, d_weights1, d_hidden_output);

    int threads = BLOCK_SIZE;
    int blocks = (block_size*hidden_size + threads - 1) / threads;
    add_bias<<<blocks, threads>>>(d_hidden_output, d_bias1, d_hidden_output, block_size*hidden_size, hidden_size);
    cudaDeviceSynchronize();

    // Apply activation function 
    dim_block = dim3(BLOCK_SIZE);
    dim_grid = dim3((block_size * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    relu<<<dim_grid, dim_block>>>(d_hidden_output, block_size * hidden_size);
    cudaDeviceSynchronize();

    // Second layer: hidden_size x output_size
    dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim_grid = dim3((d_model + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    mysgemm<<<dim_grid, dim_block>>>(block_size, d_model, hidden_size, false, true, d_hidden_output, d_weights2, d_input);
    
    blocks = (block_size*d_model + threads - 1) / threads;
    add_bias<<<blocks, threads>>>(d_input, d_bias2, d_input, block_size*d_model, d_model);
    cudaDeviceSynchronize();

    cudaFree(d_hidden_output);
}


// int main(){
//     const int d_model = 128; 
//     const int n_heads = 8;
//     const int block_size = 64;
//     const int head_dim = 16;
//     const int n_blocks = 6;
//     const unsigned int BLOCK_SIZE = TILE_SIZE;

//     std::string folder = "./weights_dump/";
//     std::vector<std::string> ffwd_dump_path = get_ffwd_paths(n_blocks, folder);
//     std::vector<float*> ffwd_weights = load_ffwd_weights(
//         n_blocks,
//         d_model,
//         d_model*4,         
//         ffwd_dump_path
//     );

//     float* input = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
//     for(int i = 0; i < block_size * n_heads * head_dim; i++) input[i] = 2.0f; // fill with ones
//     float* d_input;
//     cudaMalloc(&d_input, sizeof(float)* block_size*n_heads*head_dim);
//     cudaMemcpy(d_input, input, sizeof(float)* block_size*n_heads*head_dim, cudaMemcpyHostToDevice);

//     ffwd(
//         d_input,
//         block_size,
//         d_model,
//         d_model*4, // hidden size is 4 times the model size
//         ffwd_weights[0], // d_weights1
//         ffwd_weights[1], // d_bias1
//         ffwd_weights[2], // d_weights2
//         ffwd_weights[3]  // d_bias2
//     );

//     float* output_h = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
//     cudaMemcpy(output_h, d_input, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToHost);
//     std::string loc = "./ffwd_dump.txt";
//     dumpMatrix(output_h, block_size, n_heads * head_dim, loc);
//     return 0;

// }