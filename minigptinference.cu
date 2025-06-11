#include <stdio.h>
#include "sgemm.cu"
#include "softmax.cu"
#include "tools.cu"
#include <vector>
#include <cstring>
#include "transformer_block.cu"

// Global weight vectors - initialized once outside main
std::vector<float*> qkv_weights;
std::vector<float*> ln1_weights;
std::vector<float*> ln2_weights;
std::vector<float*> ffwd_weights;
std::vector<float*> mha_proj_weights;

// Function to load all weights
void load_all_weights(int n_blocks, int n_heads, int d_model, int head_dim, const std::string& folder) {
    printf("Loading transformer weights...\n");
    
    std::vector<std::string> qkv_dump_path = get_qkv_path(n_blocks, n_heads, folder);
    qkv_weights = load_qkv_weights(
        n_blocks, 
        n_heads, 
        d_model, 
        head_dim,
        qkv_dump_path
    );
    
    std::vector<std::string> ln1_dump_path = get_layernorm_paths(n_blocks, 1, folder);
    ln1_weights = load_layernorm_weights(
        n_blocks, 
        n_heads, 
        d_model, 
        head_dim,
        ln1_dump_path
    );
    
    std::vector<std::string> ln2_dump_path = get_layernorm_paths(n_blocks, 2, folder);
    ln2_weights = load_layernorm_weights(
        n_blocks, 
        n_heads, 
        d_model, 
        head_dim,
        ln2_dump_path
    );
    
    std::vector<std::string> ffwd_dump_path = get_ffwd_paths(n_blocks, folder);
    ffwd_weights = load_ffwd_weights(
        n_blocks,
        d_model,
        d_model*4,         
        ffwd_dump_path
    );
    
    std::vector<std::string> mha_proj_dump_path = get_mha_proj_paths(n_blocks, folder);
    mha_proj_weights = load_mha_proj_weights(
        n_blocks,
        d_model,     
        mha_proj_dump_path
    );
    
    printf("All transformer weights loaded successfully.\n");
}

int main(){
    const int d_model = 128; 
    const int n_heads = 8;
    const int block_size = 64;
    const int head_dim = 16;
    const int n_blocks = 6;
    int vocab_size = 84;
    int max_seq_len = 64;
    int seq_len = 16; // "To be or not to be" length
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    // Load vocabulary for generation
    int gen_vocab_size;
    char** gen_vocab = load_vocab_json("vocab.json", &gen_vocab_size);
    if (!gen_vocab) {
        printf("Failed to load vocabulary for generation\n");
        return 1;
    }
    printf("Loaded vocabulary for generation: %d tokens\n", gen_vocab_size);
    
    PositionalEncodingResources pos_resources;
    initialize_positional_encoding_resources(&pos_resources, max_seq_len, vocab_size, d_model);

    // Initialize generation device memory
    float *d_gen_logits, *d_gen_probs;
    int *d_selected_token;
    curandState *d_states;
    
    initialize_generation_resources(
        &d_gen_logits,
        &d_gen_probs, 
        &d_selected_token,
        &d_states,
        gen_vocab_size
    );
    
    // Prepare generation prompt tokens
    const char* generation_prompt_text = "To be or not to be";
    int generation_prompt_length;
    int* generation_prompt_tokens = text_to_tokens(gen_vocab, gen_vocab_size, generation_prompt_text, &generation_prompt_length);
    
    printf("Generation prompt: '%s' (%d tokens)\n", generation_prompt_text, generation_prompt_length);

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
    float* d_input;

    cudaMalloc(&d_token_table, vocab_size * d_model * sizeof(float));
    cudaMalloc(&d_pos_table, max_seq_len * d_model * sizeof(float));
    cudaMalloc(&d_input, seq_len * d_model * sizeof(float));

    // Copy to device
    cudaMemcpy(d_token_table, h_token_table, vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_table, h_pos_table, max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);

    int h_tokens[18] = {45, 70, 1, 57, 60, 1, 70, 73, 1, 69, 70, 75, 1, 75, 70, 1, 57, 60};

    printf("Processing sequence using SGEMM matrix multiplication...\n");

    // Process sequence using sgemm
    embed_sequence_sgemm(
        d_input,
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
    cudaMemcpy(h_result, d_input, seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

    // Show results for first token
    printf("Sequence embedding (first token, first 8 dims): ");
    for (int i = 0; i < 8; i++) {
        printf("%.6f ", h_result[i]);
    }
    printf("\n");

    // Save results
    dumpMatrix(h_result, seq_len, d_model, "./positional_embedding_result.txt");
    printf("Saved positional results to: positional_embedding_result.txt\n");

    // Load all weights once - moved outside of main processing
    load_all_weights(n_blocks, n_heads, d_model, head_dim, weights_folder);

    // setup input and output
    float* input = (float*) malloc(sizeof(float) * block_size * d_model);
    for(int i = 0; i < block_size * d_model; i++){
        if(i < 10) input[i] = 10.0f; // fill first 10 with tens
        else input[i] = 1.0f; // fill with ones
    }
    float* output = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
    for(int i = 0; i < block_size * n_heads * head_dim; i++) output[i] = 2.0f; // fill with ones

    float* d_output;
    cudaMalloc(&d_output, sizeof(float)* block_size*n_heads*head_dim);
    cudaMemcpy(d_output, output, sizeof(float)* block_size*n_heads*head_dim, cudaMemcpyHostToDevice);
    // for residual layer
    float* residual_copy; // for residual layer later
    cudaMalloc(&residual_copy, sizeof(float)* block_size*d_model);
    cudaMemcpy(residual_copy, d_input, sizeof(float)* block_size*d_model, cudaMemcpyDeviceToDevice);

    for(int b = 0; b < n_blocks; b++) {
        dim3 grid(block_size);      
        dim3 block(d_model);  

        // layer norm     
        size_t shmem = d_model * sizeof(float);  
        layer_norm<<<grid, block, shmem>>>(
            d_input,
            d_input,
            ln1_weights[b * 2], // gamma
            ln1_weights[b * 2 + 1], // beta
            head_dim,
            d_model
        );
        cudaDeviceSynchronize();

        // launch mha
        multi_head_attention(
            block_size,
            n_heads,
            d_model,
            head_dim,
            qkv_weights[b], // QKV weights
            d_input, // input
            d_output // output
        );

        // residual connection
        grid = dim3((block_size+BLOCK_SIZE-1)/BLOCK_SIZE,  (n_heads * head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        block= dim3(BLOCK_SIZE, BLOCK_SIZE);
        add_residual<<<grid, block>>>(residual_copy, d_output, d_output, block_size, n_heads * head_dim);
        cudaDeviceSynchronize();

        // copy new residual
        cudaMemcpy(residual_copy, d_output, sizeof(float)* block_size*d_model, cudaMemcpyDeviceToDevice);

        // layer norm     
        layer_norm<<<grid, block, shmem>>>(
            d_output,
            d_output,
            ln2_weights[b * 2], // gamma
            ln2_weights[b * 2 + 1], // beta
            head_dim,
            d_model
        );
        cudaDeviceSynchronize();

        // feed forward
        ffwd(
            d_output, // input
            block_size, // batch size
            d_model, // d_model
            d_model * 4, // hidden size is 4 times the model size
            ffwd_weights[b * 4], // d_bias1
            ffwd_weights[b * 4 + 1], // d_weights1
            ffwd_weights[b * 4 + 2], // d_bias2
            ffwd_weights[b * 4 + 3]  // d_weights2
        );

        // residual connection
        add_residual<<<grid, block>>>(residual_copy, d_output, d_output, block_size, n_heads * head_dim);
        cudaDeviceSynchronize();

        if (b < n_blocks - 1) {
            cudaMemcpy(residual_copy, d_output, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_input, d_output, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToDevice);
        }

        printf("Block %d processed.\n", b);

        float* output_h = (float*) malloc(block_size *  n_heads * head_dim * sizeof(float));
        cudaMemcpy(output_h, d_output, block_size *  n_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        std::string loc = "./test.txt";
        dumpMatrix(output_h, block_size, n_heads * head_dim, loc);
        free(output_h); // Free immediately after use
    }

    // Now call generation with pre-initialized resources
    int max_new_gen_tokens = 50; 
    generate_tokens_contextual(
        generation_prompt_tokens, 
        generation_prompt_length, 
        max_new_gen_tokens, 
        gen_vocab_size, 
        gen_vocab,
        d_output,           // Pass transformer output
        d_gen_logits,       // Pre-allocated
        d_gen_probs,        // Pre-allocated  
        d_selected_token,   // Pre-allocated
        d_states            // Pre-initialized
    );

    printf("Text generation finished.\n");

    // === CLEANUP ===
    // Cleanup generation resources
    cleanup_generation_resources(d_gen_logits, d_gen_probs, d_selected_token, d_states);
    
    // Cleanup generation vocabulary and tokens
    free(generation_prompt_tokens);
    for (int i = 0; i < gen_vocab_size; i++) {
        if (gen_vocab[i]) {
            free(gen_vocab[i]);
        }
    }
    free(gen_vocab);

    // Cleanup transformer resources
    cudaFree(d_token_table);
    cudaFree(d_pos_table);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(residual_copy);
    
    free(h_token_table);
    free(h_pos_table);
    free(h_result);
    free(input);
    free(output);

    return 0;
}