// transformer_decoder.cu

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include "layer_norm.cu"  // Make sure this is accessible and correct!

#define EMBED_DIM 102  
#define NUM_HEADS 4
#define HEAD_DIM (EMBED_DIM / NUM_HEADS)
#define MLP_HIDDEN_DIM 512
#define SEQ_LEN 32
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = gelu(input[idx]);
}

__global__ void add_bias(float* data, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        data[idx] += bias[col];
    }
}

__global__ void add_residual(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

void linear(cublasHandle_t handle, const float* input, const float* weight, float* output, 
    int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // For the operation: output = input × weight
    // input is M×K, weight is K×N, output is M×N
    cublasStatus_t stat = cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose operations needed
        N, M, K,                    // N = num cols out, M = num rows out, K = num cols in
        &alpha,
        weight, N,                  // Leading dimension of weight matrix
        input, K,                   // Leading dimension of input matrix
        &beta,
        output, N);                 // Leading dimension of output matrix

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", stat);
        exit(1);
    }
}

// Decoder block (MLP + residual, no attention)
void run_decoder_block(
    float* input, float* output,
    float* ln1_weight, float* ln1_bias,
    float* mlp_fc1_weight, float* mlp_fc1_bias,
    float* mlp_fc2_weight, float* mlp_fc2_bias,
    int seq_len, int embed_dim,
    cublasHandle_t handle
) 
{
    float *ln_output = nullptr, *fc1_out = nullptr, *gelu_out = nullptr;
    
    // Allocate memory with error checking
    CHECK_CUDA(cudaMalloc(&ln_output, sizeof(float) * seq_len * embed_dim));
    CHECK_CUDA(cudaMalloc(&fc1_out, sizeof(float) * seq_len * MLP_HIDDEN_DIM));
    CHECK_CUDA(cudaMalloc(&gelu_out, sizeof(float) * seq_len * MLP_HIDDEN_DIM));

    // Layer normalization
    dim3 block_ln(32, 32);
    dim3 grid_ln((seq_len + 31) / 32, (embed_dim + 31) / 32);
    layer_norm<<<grid_ln, block_ln>>>(input, ln_output, ln1_weight, ln1_bias, seq_len, embed_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // First linear layer
    linear(handle, ln_output, mlp_fc1_weight, fc1_out, seq_len, MLP_HIDDEN_DIM, embed_dim);
    
    // Add bias and apply GELU
    int fc1_size = seq_len * MLP_HIDDEN_DIM;
    int block_size = 256;
    int grid_size = (fc1_size + block_size - 1) / block_size;
    
    add_bias<<<grid_size, block_size>>>(fc1_out, mlp_fc1_bias, seq_len, MLP_HIDDEN_DIM);
    CHECK_CUDA(cudaGetLastError());
    
    gelu_kernel<<<grid_size, block_size>>>(fc1_out, gelu_out, fc1_size);
    CHECK_CUDA(cudaGetLastError());

    // Second linear layer
    linear(handle, gelu_out, mlp_fc2_weight, output, seq_len, embed_dim, MLP_HIDDEN_DIM);
    
    int output_size = seq_len * embed_dim;
    grid_size = (output_size + block_size - 1) / block_size;
    
    add_bias<<<grid_size, block_size>>>(output, mlp_fc2_bias, seq_len, embed_dim);
    CHECK_CUDA(cudaGetLastError());

    // Residual connection
    add_residual<<<grid_size, block_size>>>(input, output, output, output_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Free temporary buffers
    CHECK_CUDA(cudaFree(ln_output));
    CHECK_CUDA(cudaFree(fc1_out));
    CHECK_CUDA(cudaFree(gelu_out));
}


// Add this function at the top of the file after includes
float* loadMatrix(int rows, int cols, const std::string& filename) {
    float* data = new float[rows * cols];
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < rows * cols; i++) {
        file >> data[i];
    }
    file.close();
    return data;
}

void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    float* host_matrix = new float[rows * cols];
    CHECK_CUDA(cudaMemcpy(host_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\n%s:\n", name);
    // Print first 5 rows and columns only to avoid cluttering the output
    for (int i = 0; i < std::min(rows, 5); i++) {
        for (int j = 0; j < std::min(cols, 5); j++) {
            printf("%.4f ", host_matrix[i * cols + j]);
        }
        printf("\n");
    }
    delete[] host_matrix;
}

// In the main function, modify the folder path:
int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Model dimensions from your Python model
    const int seq_len = SEQ_LEN;
    const int embed_dim = EMBED_DIM;
    
    std::string folder = "./weights_dump/";

    // Allocate device memory
    float *d_input, *d_output;
    float *d_ln1_weight, *d_ln1_bias;
    float *d_mlp_fc1_weight, *d_mlp_fc1_bias;
    float *d_mlp_fc2_weight, *d_mlp_fc2_bias;

    CHECK_CUDA(cudaMalloc(&d_input, seq_len * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, seq_len * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln1_weight, embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln1_bias, embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_fc1_weight, embed_dim * MLP_HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_fc1_bias, MLP_HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_fc2_weight, MLP_HIDDEN_DIM * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_fc2_bias, embed_dim * sizeof(float)));

    // Load weights from dumps
    float *h_input, *h_ln1_weight, *h_ln1_bias;
    float *h_mlp_fc1_weight, *h_mlp_fc1_bias;
    float *h_mlp_fc2_weight, *h_mlp_fc2_bias;

    // Load weights from dumps
    h_input = loadMatrix(seq_len, embed_dim, "./logits.txt");
    h_ln1_weight = loadMatrix(1, embed_dim, folder + "block.0.ln1.weight.txt");
    h_ln1_bias = loadMatrix(1, embed_dim, folder + "block.0.ln1.bias.txt");
    h_mlp_fc1_weight = loadMatrix(MLP_HIDDEN_DIM, embed_dim, folder + "block.0.ffwd.0.weight.txt");
    h_mlp_fc1_bias = loadMatrix(1, MLP_HIDDEN_DIM, folder + "block.0.ffwd.0.bias.txt");
    h_mlp_fc2_weight = loadMatrix(embed_dim, MLP_HIDDEN_DIM, folder + "block.0.ffwd.2.weight.txt");
    h_mlp_fc2_bias = loadMatrix(1, embed_dim, folder + "block.0.ffwd.2.bias.txt");

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, seq_len * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln1_weight, h_ln1_weight, embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln1_bias, h_ln1_bias, embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_fc1_weight, h_mlp_fc1_weight, embed_dim * MLP_HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_fc1_bias, h_mlp_fc1_bias, MLP_HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_fc2_weight, h_mlp_fc2_weight, MLP_HIDDEN_DIM * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_fc2_bias, h_mlp_fc2_bias, embed_dim * sizeof(float), cudaMemcpyHostToDevice));

    // Print input
    printf("Input tensor:");
    print_matrix(d_input, seq_len, embed_dim, "Input");

    // Run decoder block
    run_decoder_block(
        d_input, d_output,
        d_ln1_weight, d_ln1_bias,
        d_mlp_fc1_weight, d_mlp_fc1_bias,
        d_mlp_fc2_weight, d_mlp_fc2_bias,
        seq_len, embed_dim,
        handle
    );

    // Print output and save to file for comparison
    printf("\nOutput tensor:");
    print_matrix(d_output, seq_len, embed_dim, "Output");

    // Save output to file for comparison with Python
    float* h_output = new float[seq_len * embed_dim];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, seq_len * embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::ofstream outfile("cuda_decoder_output.txt");
    for (int i = 0; i < seq_len * embed_dim; i++) {
        outfile << h_output[i] << " ";
    }
    outfile.close();

    // Cleanup
    delete[] h_input;
    delete[] h_ln1_weight;
    delete[] h_ln1_bias;
    delete[] h_mlp_fc1_weight;
    delete[] h_mlp_fc1_bias;
    delete[] h_mlp_fc2_weight;
    delete[] h_mlp_fc2_bias;
    delete[] h_output;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_ln1_weight));
    CHECK_CUDA(cudaFree(d_ln1_bias));
    CHECK_CUDA(cudaFree(d_mlp_fc1_weight));
    CHECK_CUDA(cudaFree(d_mlp_fc1_bias));
    CHECK_CUDA(cudaFree(d_mlp_fc2_weight));
    CHECK_CUDA(cudaFree(d_mlp_fc2_bias));
    
    cublasDestroy(handle);
    return 0;
}