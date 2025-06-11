#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "softmax.cu"

#define VOCAB_SIZE 84

// Fixed vocabulary loading - no duplicate space tokens
char** load_vocab_c(const char* filename, int* vocab_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open vocab file %s\n", filename);
        return NULL;
    }
    
    // Read vocabulary directly from file
    int count = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), file)) count++;
    rewind(file);
    
    char** vocab = (char**)malloc(count * sizeof(char*));
    int index = 0;
    
    while (fgets(buffer, sizeof(buffer), file) && index < count) {
        // Remove line endings
        size_t len = strlen(buffer);
        while (len > 0 && (buffer[len-1] == '\n' || buffer[len-1] == '\r')) {
            buffer[len-1] = '\0';
            len--;
        }
        
        vocab[index] = strdup(buffer);
        
        // Debug first 10 tokens
        if (index < 10) {
            if (vocab[index][0] == ' ') {
                printf("Vocab[%d]: 'SPACE'\n", index);
            } else {
                printf("Vocab[%d]: '%s'\n", index, vocab[index]);
            }
        }
        index++;
    }
    
    *vocab_size = index;
    fclose(file);
    printf("Loaded %d tokens total\n", index);
    return vocab;
}

// Simple JSON parser for vocabulary
char** load_vocab_json(const char* filename, int* vocab_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open vocab file %s\n", filename);
        return NULL;
    }
    
    // Read entire file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* json_content = (char*)malloc(file_size + 1);
    fread(json_content, 1, file_size, file);
    json_content[file_size] = '\0';
    fclose(file);
    
    // Allocate vocabulary array for 84 tokens
    char** vocab = (char**)calloc(84, sizeof(char*));
    
    // Parse JSON manually
    char* pos = json_content;
    
    while ((pos = strstr(pos, "\"")) != NULL) {
        pos++; // Skip opening quote
        
        // Find token ID
        char* id_end = strchr(pos, '"');
        if (!id_end) break;
        
        // Extract token ID
        char id_str[10];
        int id_len = id_end - pos;
        strncpy(id_str, pos, id_len);
        id_str[id_len] = '\0';
        int token_id = atoi(id_str);
        
        // Skip to value
        pos = strstr(id_end, ": \"");
        if (!pos) break;
        pos += 3; // Skip ": "
        
        // Find end of value
        char* value_end = pos;
        while (*value_end && *value_end != '"') {
            if (*value_end == '\\') value_end++; // Skip escaped chars
            value_end++;
        }
        
        // Extract token value
        int value_len = value_end - pos;
        char* token_value = (char*)malloc(value_len + 1);
        strncpy(token_value, pos, value_len);
        token_value[value_len] = '\0';
        
        // Handle escape sequences
        if (strcmp(token_value, "\\n") == 0) {
            free(token_value);
            token_value = strdup("\n");
        }
        
        // Store in correct position
        if (token_id >= 0 && token_id < 84) {
            vocab[token_id] = token_value;
        }
        
        pos = value_end + 1;
    }
    
    *vocab_size = 84;
    free(json_content);
    return vocab;
}

// Setup random states
__global__ void setup_random_states(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void multinomial_sample_kernel(
    const float* probs,
    int* selected_token,
    curandState* states,
    int vocab_size
    // Remove step_number parameter - don't re-seed each step!
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    curandState* state = &states[idx];
    float coin = curand_uniform(state);

    float total_prob = 0.0f;
    float max_prob = 0.0f;
    int max_idx = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        total_prob += probs[i];
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = i;
        }
    }
    
    float cumulative_prob = 0.0f;
    int selected = 0;
    
    for(int i = 0; i < vocab_size; i++){
        cumulative_prob += probs[i];

        if(coin <= cumulative_prob){
            selected = i;
            break;
        }
    }

    if (coin > cumulative_prob) {
        selected = vocab_size - 1;
    }

    *selected_token = selected;
}

// Convert text to tokens
int* text_to_tokens(char** vocab, int vocab_size, const char* text, int* num_tokens) {
    int text_len = strlen(text);
    int* token_ids = (int*)malloc(text_len * sizeof(int));
    int valid_tokens = 0;
    
    for (int i = 0; i < text_len; i++) {
        char target_char = text[i];
        int token_id = -1;
        
        // Find matching token
        for (int j = 0; j < vocab_size; j++) {
            if (vocab[j] && strlen(vocab[j]) == 1 && vocab[j][0] == target_char) {
                token_id = j;
                break;
            }
        }
        
        if (token_id >= 0) {
            token_ids[valid_tokens] = token_id;
        } else {
            // Find space token as fallback
            for (int j = 0; j < vocab_size; j++) {
                if (vocab[j] && strlen(vocab[j]) == 1 && vocab[j][0] == ' ') {
                    token_ids[valid_tokens] = j;
                    break;
                }
            }
        }
        valid_tokens++;
    }
    
    *num_tokens = valid_tokens;
    return token_ids;
}

// Extract last token logits from full logits tensor (matching logits[:, -1, :])
float* extract_last_token_logits(float* full_logits, int seq_len, int vocab_size) {
    // Equivalent to PyTorch: logits[:, -1, :]
    // In row-major storage: last_token_start = (seq_len - 1) * vocab_size
    int last_token_start = (seq_len - 1) * vocab_size;
    
    float* last_logits = (float*)malloc(vocab_size * sizeof(float));
    
    for (int i = 0; i < vocab_size; i++) {
        last_logits[i] = full_logits[last_token_start + i];
    }
    
    return last_logits;
}

// Updated load function that handles both cases
float* load_and_extract_logits_c(const char* filename, int vocab_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return NULL;
    }
    
    // Count total numbers
    float temp;
    int total_numbers = 0;
    while (fscanf(file, "%f", &temp) == 1) {
        total_numbers++;
    }
    rewind(file);
    
    if (total_numbers == vocab_size) {
        // Already extracted - load directly
        float* logits = (float*)malloc(vocab_size * sizeof(float));
        for (int i = 0; i < vocab_size; i++) {
            fscanf(file, "%f", &logits[i]);
        }
        fclose(file);
        return logits;
    }
    else {
        // Full tensor - load and extract
        int seq_len = total_numbers / vocab_size;
        float* full_logits = (float*)malloc(total_numbers * sizeof(float));
        
        for (int i = 0; i < total_numbers; i++) {
            fscanf(file, "%f", &full_logits[i]);
        }
        fclose(file);
        
        // Extract last token logits
        float* last_logits = extract_last_token_logits(full_logits, seq_len, vocab_size);
        free(full_logits);
        
        return last_logits;
    }
}

// Simplified generation that matches the Python function exactly
void generate_tokens_contextual(
    int* input_tokens,
    int input_length,
    int max_new_tokens,
    int vocab_size,
    char** vocab
) {
    // initialize weights
    const int d_model = 128; 
    const int n_heads = 8;
    const int block_size = 64;
    const int head_dim = 16;
    const int n_blocks = 6;
    int vocab_size = 84;
    int max_seq_len = 64;
    int seq_len = 16; // "To be or not to be" length
    const unsigned int BLOCK_SIZE = TILE_SIZE;


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
        max_seq_len
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



    // load the weights
    std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
    std::vector<std::string> qkv_dump_path = get_qkv_path(n_blocks, n_heads, folder);
    std::vector<float*> qkv_weights = load_qkv_weights(
        n_blocks, 
        n_heads, 
        d_model, 
        head_dim,
        qkv_dump_path
    );
    std::vector<std::string> ln1_dump_path = get_layernorm_paths(n_blocks, 1, folder);
    std::vector<float*> ln1_weights = load_layernorm_weights(
        n_blocks, 
        n_heads, 
        d_model, 
        head_dim,
        ln1_dump_path
    );
    std::vector<std::string> ln2_dump_path = get_layernorm_paths(n_blocks, 2, folder);
    std::vector<float*> ln2_weights = load_layernorm_weights(
        n_blocks, 
        n_heads, 
        d_model, 
        head_dim,
        ln2_dump_path
    );
    std::vector<std::string> ffwd_dump_path = get_ffwd_paths(n_blocks, folder);
    std::vector<float*> ffwd_weights = load_ffwd_weights(
        n_blocks,
        d_model,
        d_model*4,         
        ffwd_dump_path
    );
    std::vector<std::string> mha_proj_dump_path = get_mha_proj_paths(n_blocks, folder);
    std::vector<float*> mha_proj_weights = load_mha_proj_weights(
        n_blocks,
        d_model,     
        mha_proj_dump_path
    );


    // input and output
    float* input = (float*) malloc(sizeof(float) * block_size * d_model);
    for(int i = 0; i < block_size * d_model; i++){
        if(i < 10) input[i] = 10.0f; // fill first 10 with tens
        else input[i] = 1.0f; // fill with ones
    }
    float* output = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
    for(int i = 0; i < block_size * n_heads * head_dim; i++) output[i] = 2.0f; // fill with ones


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

        // float* output_h = (float*) malloc(block_size*d_model * sizeof(float));
        // cudaMemcpy(output_h, d_output, block_size*d_model * sizeof(float), cudaMemcpyDeviceToHost);
        // std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/layernorm.txt";
        // dumpMatrix(output_h, block_size, d_model, loc);

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
        std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
        dumpMatrix(output_h, block_size, n_heads * head_dim, loc);
    }


    // Device memory
    float *d_logits, *d_probs;
    int *d_selected_token;
    curandState *d_states;
    
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_probs, vocab_size * sizeof(float));
    cudaMalloc(&d_selected_token, sizeof(int));
    cudaMalloc(&d_states, sizeof(curandState));
    
    // Synchronize to ensure random states are initialized before sampling
    cudaDeviceSynchronize();
    
    // Use 42 as fixed seed to match Python torch.manual_seed(42)
    setup_random_states<<<1, 1>>>(d_states, 42, 1);
    cudaDeviceSynchronize();
    
    // Create dynamic token sequence (grows with each generation)
    int max_sequence_length = input_length + max_new_tokens;
    int* token_sequence = (int*)malloc(max_sequence_length * sizeof(int));
    
    // Copy initial tokens
    memcpy(token_sequence, input_tokens, input_length * sizeof(int));
    int current_length = input_length;
    
    // Create full text buffer
    char* full_text = (char*)malloc(10000);
    strcpy(full_text, "");
    
    // Add initial tokens to text
    for (int i = 0; i < input_length; i++) {
        if (input_tokens[i] >= 0 && input_tokens[i] < vocab_size && vocab[input_tokens[i]]) {
            strcat(full_text, vocab[input_tokens[i]]);
        }
    }
    
    // Generate tokens one by one
    for (int step = 0; step < max_new_tokens; step++) {
        // Load logits for this step
        char logits_filename[256];
        sprintf(logits_filename, "./logits/logits_%d.txt", step);
        
        float* logits = load_and_extract_logits_c(logits_filename, vocab_size);
        if (!logits) {
            break;
        }

        // Apply softmax to get probabilities
        cudaMemcpy(d_logits, logits, vocab_size * sizeof(float), cudaMemcpyHostToDevice);
        softmax(d_logits, d_probs, 1, vocab_size);

        float* h_probs = (float*)malloc(vocab_size * sizeof(float));
        cudaMemcpy(h_probs, d_probs, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

        float prob_sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            prob_sum += h_probs[i];
        }

        // Sample token using multinomial
        multinomial_sample_kernel<<<1, 1>>>(d_probs, d_selected_token, d_states, vocab_size);
        cudaDeviceSynchronize();

        // Get result and validate
        int next_token;
        cudaMemcpy(&next_token, d_selected_token, sizeof(int), cudaMemcpyDeviceToHost);

        // Show what the top 3 highest probability tokens were for comparison
        int top_indices[3] = {-1, -1, -1};
        float top_probs[3] = {-1.0f, -1.0f, -1.0f};
        
        // Find top 3 by scanning all tokens
        for (int i = 0; i < vocab_size; i++) {
            if (h_probs[i] > top_probs[2]) {
                // New token beats 3rd place
                if (h_probs[i] > top_probs[1]) {
                    // Beats 2nd place
                    if (h_probs[i] > top_probs[0]) {
                        // Beats 1st place - shift everything down
                        top_probs[2] = top_probs[1];
                        top_indices[2] = top_indices[1];
                        top_probs[1] = top_probs[0];
                        top_indices[1] = top_indices[0];
                        top_probs[0] = h_probs[i];
                        top_indices[0] = i;
                    } else {
                        // Beats 2nd place only
                        top_probs[2] = top_probs[1];
                        top_indices[2] = top_indices[1];
                        top_probs[1] = h_probs[i];
                        top_indices[1] = i;
                    }
                } else {
                    // Beats 3rd place only
                    top_probs[2] = h_probs[i];
                    top_indices[2] = i;
                }
            }
        }

        free(h_probs);
        
        // Validate token
        if (next_token < 0 || next_token >= vocab_size) {
            break;
        }

        // Add token to sequence
        token_sequence[current_length] = next_token;
        current_length++;
        
        // Add to text
        if (vocab[next_token]) {
            strcat(full_text, vocab[next_token]);
        }
        
        free(logits);
    }
    
    printf("\n=== GENERATED TEXT ===\n");
    printf("'%s'\n", full_text);
    printf("======================\n");
    
    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_probs);
    cudaFree(d_selected_token);
    cudaFree(d_states);
    free(token_sequence);
    free(full_text);
}

int main() {
    // Load vocabulary from JSON
    int vocab_size;
    char** vocab = load_vocab_json("vocab.json", &vocab_size);  // Changed from load_vocab_c
    if (!vocab) {
        printf("Failed to load vocabulary\n");
        return 1;
    }
    
    // Convert input text to tokens
    const char* input_text = "To be or not to be";
    int input_length;
    int* input_tokens = text_to_tokens(vocab, vocab_size, input_text, &input_length);
    
    printf("Loaded vocabulary: %d tokens\n", vocab_size);
    printf("Input: '%s' (%d tokens)\n\n", input_text, input_length);
    
    // Generate text
    generate_tokens_contextual(input_tokens, input_length, 50, vocab_size, vocab);
    
    // Cleanup
    free(input_tokens);
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i]) { // Add check for NULL before freeing
            free(vocab[i]);
        }
    }
    free(vocab);
    
    return 0;
}