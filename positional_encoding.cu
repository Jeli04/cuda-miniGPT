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

// Structure to hold pre-allocated positional encoding resources
struct PositionalEncodingResources {
    float* d_token_onehot;
    float* d_pos_onehot;
    float* d_token_embeddings;
    float* d_pos_embeddings;
    float* h_token_onehot;
    float* h_pos_onehot;
    int max_seq_len;
    int vocab_size;
    int d_model;
};

// Initialize positional encoding resources outside the main function
void initialize_positional_encoding_resources(
    PositionalEncodingResources* resources,
    int max_seq_len,
    int vocab_size,
    int d_model
) {
    resources->max_seq_len = max_seq_len;
    resources->vocab_size = vocab_size;
    resources->d_model = d_model;
    
    // Allocate device memory for maximum possible sequence length
    cudaMalloc(&resources->d_token_onehot, max_seq_len * vocab_size * sizeof(float));
    cudaMalloc(&resources->d_pos_onehot, max_seq_len * max_seq_len * sizeof(float));
    cudaMalloc(&resources->d_token_embeddings, max_seq_len * d_model * sizeof(float));
    cudaMalloc(&resources->d_pos_embeddings, max_seq_len * d_model * sizeof(float));
    
    // Allocate host memory for maximum possible sequence length
    resources->h_token_onehot = (float*)calloc(max_seq_len * vocab_size, sizeof(float));
    resources->h_pos_onehot = (float*)calloc(max_seq_len * max_seq_len, sizeof(float));
    
    printf("Positional encoding resources initialized for max_seq_len=%d, vocab_size=%d, d_model=%d\n", 
           max_seq_len, vocab_size, d_model);
}

// Cleanup positional encoding resources
void cleanup_positional_encoding_resources(PositionalEncodingResources* resources) {
    cudaFree(resources->d_token_onehot);
    cudaFree(resources->d_pos_onehot);
    cudaFree(resources->d_token_embeddings);
    cudaFree(resources->d_pos_embeddings);
    free(resources->h_token_onehot);
    free(resources->h_pos_onehot);
}

// Modified embed_sequence_sgemm that uses pre-allocated resources
void embed_sequence_sgemm(
    float* d_output,           // [seq_len, d_model]
    const float* d_token_table, // [vocab_size, d_model] 
    const float* d_pos_table,   // [max_seq_len, d_model]
    const int* h_token_sequence, // [seq_len] - host memory
    int seq_len,
    int d_model,
    int vocab_size,
    int max_seq_len,
    PositionalEncodingResources* resources // Pre-allocated resources
) {
    // Clear the one-hot matrices for current sequence length
    cudaMemset(resources->d_token_onehot, 0, seq_len * vocab_size * sizeof(float));
    cudaMemset(resources->d_pos_onehot, 0, seq_len * max_seq_len * sizeof(float));
    
    // Clear host one-hot matrices for current sequence length
    memset(resources->h_token_onehot, 0, seq_len * vocab_size * sizeof(float));
    memset(resources->h_pos_onehot, 0, seq_len * max_seq_len * sizeof(float));
    
    // Fill one-hot matrices on host
    for (int i = 0; i < seq_len; i++) {
        resources->h_token_onehot[i * vocab_size + h_token_sequence[i]] = 1.0f;
        resources->h_pos_onehot[i * max_seq_len + i] = 1.0f; // position = sequence index
    }
    
    // Copy to device (only the portions we need)
    cudaMemcpy(resources->d_token_onehot, resources->h_token_onehot, 
               seq_len * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(resources->d_pos_onehot, resources->h_pos_onehot, 
               seq_len * max_seq_len * sizeof(float), cudaMemcpyHostToDevice);
    
    // Token embeddings: [seq_len, vocab_size] × [vocab_size, d_model] = [seq_len, d_model]
    basicSgemm(seq_len, d_model, vocab_size, false, false, 
               resources->d_token_onehot, d_token_table, resources->d_token_embeddings);
    
    // Position embeddings: [seq_len, max_seq_len] × [max_seq_len, d_model] = [seq_len, d_model]  
    basicSgemm(seq_len, d_model, max_seq_len, false, false, 
               resources->d_pos_onehot, d_pos_table, resources->d_pos_embeddings);
    
    // Add token and position embeddings
    dim3 block(16, 16);
    dim3 grid((d_model + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);
    add_embeddings<<<grid, block>>>(resources->d_token_embeddings, resources->d_pos_embeddings, 
                                   d_output, seq_len, d_model);
    cudaDeviceSynchronize();
}

// Backward compatibility: original function that allocates resources internally
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
    // Create temporary resources for this call
    PositionalEncodingResources temp_resources;
    initialize_positional_encoding_resources(&temp_resources, max_seq_len, vocab_size, d_model);
    
    // Use the optimized version
    embed_sequence_sgemm(
        d_output, d_token_table, d_pos_table, h_token_sequence,
        seq_len, d_model, vocab_size, max_seq_len, &temp_resources
    );
    
    // Cleanup temporary resources
    cleanup_positional_encoding_resources(&temp_resources);
}

// Test function using sgemm with pre-allocated resources
void test_positional_encoding_sgemm() {
    printf("=== POSITIONAL ENCODING WITH SGEMM ===\n");
    
    // Parameters
    int vocab_size = 84;
    int d_model = 128;
    int max_seq_len = 64;
    int seq_len = 18; // "To be or not to be" length
    
    // Initialize positional encoding resources
    PositionalEncodingResources pos_resources;
    initialize_positional_encoding_resources(&pos_resources, max_seq_len, vocab_size, d_model);
    
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
    
    // Process sequence using sgemm with pre-allocated resources
    embed_sequence_sgemm(
        d_output,
        d_token_table,
        d_pos_table,
        h_tokens,
        seq_len,
        d_model,
        vocab_size,
        max_seq_len,
        &pos_resources
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
    cleanup_positional_encoding_resources(&pos_resources);
    cudaFree(d_token_table);
    cudaFree(d_pos_table);
    cudaFree(d_output);
    free(h_token_table);
    free(h_pos_table);
    free(h_result);
}
