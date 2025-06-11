#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "tools.cu"
#include "sgemm.cu"  


__global__ void add_embeddings(const float* token_emb, const float* pos_emb, float* output, int seq_len, int d_model) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // sequence position
    int j = blockIdx.x * blockDim.x + threadIdx.x; // embedding dimension
    
    if (i < seq_len && j < d_model) {
        output[i * d_model + j] = token_emb[i * d_model + j] + pos_emb[i * d_model + j];
    }
}

// Matrix-based embedding using your basicSgemm
void embed_sequence_sgemm(
    float* d_output,           // [seq_len, d_model]
    const float* d_token_table, // [vocab_size, d_model] 
    const float* d_pos_table,   // [max_seq_len, d_model]
    const int* h_token_sequence, // [seq_len] - host memory
    int seq_len,
    int d_model,
    int vocab_size,
    int max_seq_len
) {
    // Create one-hot matrices for the sequence
    float* d_token_onehot;
    float* d_pos_onehot;
    float* d_token_embeddings;
    float* d_pos_embeddings;
    
    cudaMalloc(&d_token_onehot, seq_len * vocab_size * sizeof(float));
    cudaMalloc(&d_pos_onehot, seq_len * max_seq_len * sizeof(float));
    cudaMalloc(&d_token_embeddings, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_pos_embeddings, seq_len * d_model * sizeof(float));
    
    // Zero out one-hot matrices
    cudaMemset(d_token_onehot, 0, seq_len * vocab_size * sizeof(float));
    cudaMemset(d_pos_onehot, 0, seq_len * max_seq_len * sizeof(float));
    
    // Fill one-hot matrices on host then copy
    float* h_token_onehot = (float*)calloc(seq_len * vocab_size, sizeof(float));
    float* h_pos_onehot = (float*)calloc(seq_len * max_seq_len, sizeof(float));
    
    for (int i = 0; i < seq_len; i++) {
        h_token_onehot[i * vocab_size + h_token_sequence[i]] = 1.0f;
        h_pos_onehot[i * max_seq_len + i] = 1.0f; // position = sequence index
    }
    
    cudaMemcpy(d_token_onehot, h_token_onehot, seq_len * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_onehot, h_pos_onehot, seq_len * max_seq_len * sizeof(float), cudaMemcpyHostToDevice);
    
    // Token embeddings: [seq_len, vocab_size] × [vocab_size, d_model] = [seq_len, d_model]
    basicSgemm(seq_len, d_model, vocab_size, false, false, d_token_onehot, d_token_table, d_token_embeddings);
    
    // Position embeddings: [seq_len, max_seq_len] × [max_seq_len, d_model] = [seq_len, d_model]  
    basicSgemm(seq_len, d_model, max_seq_len, false, false, d_pos_onehot, d_pos_table, d_pos_embeddings);
    
    // Add token and position embeddings
    dim3 block(16, 16);
    dim3 grid((d_model + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);
    add_embeddings<<<grid, block>>>(d_token_embeddings, d_pos_embeddings, d_output, seq_len, d_model);
    
    // Cleanup
    cudaFree(d_token_onehot);
    cudaFree(d_pos_onehot);
    cudaFree(d_token_embeddings);
    cudaFree(d_pos_embeddings);
    free(h_token_onehot);
    free(h_pos_onehot);
}



// Test function using sgemm
void test_positional_encoding_sgemm() {
    printf("=== POSITIONAL ENCODING WITH SGEMM ===\n");
    
    // Parameters
    int vocab_size = 84;
    int d_model = 128;
    int max_seq_len = 64;
    int seq_len = 18; // "To be or not to be" length
    
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
    cudaMalloc(&d_output, seq_len * d_model * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_token_table, h_token_table, vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_table, h_pos_table, max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    int h_tokens[18] = {45, 70, 1, 57, 60, 1, 70, 73, 1, 69, 70, 75, 1, 75, 70, 1, 57, 60};
    
    printf("Processing sequence using SGEMM matrix multiplication...\n");
    
    // Process sequence using sgemm
    embed_sequence_sgemm(
        d_output,
        d_token_table,
        d_pos_table,
        h_tokens,
        seq_len,
        d_model,
        vocab_size,
        max_seq_len
    );
    
    // Copy results back
    float* h_result = (float*)malloc(seq_len * d_model * sizeof(float));
    cudaMemcpy(h_result, d_output, seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Show results for first token
    printf("Sequence embedding (first token, first 8 dims): ");
    for (int i = 0; i < 8; i++) {
        printf("%.6f ", h_result[i]);
    }
    printf("\n");
    
    // Save results
    dumpMatrix(h_result, seq_len, d_model, "./positional_embedding_result.txt");
    printf("Saved positional results to: positional_embedding_result.txt\n");
    
    // Cleanup
    cudaFree(d_token_table);
    cudaFree(d_pos_table);
    cudaFree(d_output);
    free(h_token_table);
    free(h_pos_table);
    free(h_result);
}

// int main() {
//     test_positional_encoding_sgemm();
//     return 0;
// }