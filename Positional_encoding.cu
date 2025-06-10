#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "tools.cpp"

__global__ void embed_kernel(
    float* output,
    const float* token_table, 
    const float* pos_table,  
    int token_id,
    int position,
    int d_model
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_model) {
        output[i] = token_table[token_id * d_model + i] + pos_table[position * d_model + i];
    }
}

void embed_gpu(
    float* d_output,
    const float* d_token_table,
    const float* d_pos_table,
    int token_id,
    int position,
    int d_model
) {
    int num_threads = 256;
    int num_blocks = (d_model + num_threads - 1) / num_threads;
    embed_kernel<<<num_blocks, num_threads>>>(
        d_output, d_token_table, d_pos_table, token_id, position, d_model
    );
    cudaDeviceSynchronize();
}

// Test function
void test_positional_encoding() {
    printf("=== SIMPLE POSITIONAL ENCODING TEST ===\n");
    
    // Parameters
    int vocab_size = 84;
    int d_model = 128;
    int max_seq_len = 64;
    
    // Load embedding tables
    std::string weights_folder = "./weights_dump/";
    std::string token_file = weights_folder + "token_embedding_table.weight.txt";
    std::string pos_file = weights_folder + "position_embedding_table.weight.txt";
    
    float* h_token_table = loadMatrix(vocab_size, d_model, token_file);
    float* h_pos_table = loadMatrix(max_seq_len, d_model, pos_file);
    
    printf("Loaded token embedding table: %d x %d\n", vocab_size, d_model);
    printf("Loaded position embedding table: %d x %d\n", max_seq_len, d_model);
    
    // Allocate device memory
    float* d_token_table;
    float* d_pos_table;
    float* d_output;
    
    cudaMalloc(&d_token_table, vocab_size * d_model * sizeof(float));
    cudaMalloc(&d_pos_table, max_seq_len * d_model * sizeof(float));
    cudaMalloc(&d_output, d_model * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_token_table, h_token_table, vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_table, h_pos_table, max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test first token (token_id = 45, position = 0)
    int test_token = 45;
    int test_position = 0;
    
    printf("\n=== TESTING TOKEN %d AT POSITION %d ===\n", test_token, test_position);
    
    // Run embedding
    embed_gpu(d_output, d_token_table, d_pos_table, test_token, test_position, d_model);
    
    // Copy result back
    float* h_result = (float*)malloc(d_model * sizeof(float));
    cudaMemcpy(h_result, d_output, d_model * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Show breakdown
    printf("Token embedding (first 8 dims): ");
    for (int i = 0; i < 8; i++) {
        printf("%.6f ", h_token_table[test_token * d_model + i]);
    }
    printf("\n");
    
    printf("Position embedding (first 8 dims): ");
    for (int i = 0; i < 8; i++) {
        printf("%.6f ", h_pos_table[test_position * d_model + i]);
    }
    printf("\n");
    
    printf("Combined result (first 8 dims): ");
    for (int i = 0; i < 8; i++) {
        printf("%.6f ", h_result[i]);
    }
    printf("\n");
    
    // Save result
    dumpMatrix(h_result, 1, d_model, "./positional_embedding_result.txt");
    printf("\nSaved result to: positional_embedding_result.txt\n");
        
    // Cleanup
    cudaFree(d_token_table);
    cudaFree(d_pos_table);
    cudaFree(d_output);
    free(h_token_table);
    free(h_pos_table);
    free(h_result);
}

int main() {
    test_positional_encoding();
    return 0;
}