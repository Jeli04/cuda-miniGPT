#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "softmax.cu"
#include "transformer_block.cu"
#include "tools.cu"

#define VOCAB_SIZE 84


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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    curandState* state = &states[idx];
    float coin = curand_uniform(state);

    float total_prob = 0.0f;

    
    for (int i = 0; i < vocab_size; i++) {
        total_prob += probs[i];
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

float* extract_last_token_logits(float* full_logits, int seq_len, int vocab_size) {
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

void generate_tokens_contextual(
    int* input_tokens,
    int input_length,
    int max_new_tokens,
    int vocab_size,
    char** vocab,
    float* d_transformer_output,
    float* d_logits,
    int* d_selected_token,
    curandState* d_states,
    TransformerBlockConfig config,
) {
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
        // // Load logits for this step
        // char logits_filename[256];
        // sprintf(logits_filename, "./logits/logits_%d.txt", step);
        
        // float* logits = load_and_extract_logits_c(logits_filename, vocab_size);
        // if (!logits) {
        //     break;
        // }

        float* logits = (float*)malloc(vocab_size * sizeof(float));
        for (int i = 0; i < vocab_size; i++) {
            logits[i] = 1.0f; // Placeholder for actual logits
        }
        // Copy logits to device
        cudaMemcpy(d_logits, config.logits, vocab_size * sizeof(float), cudaMemcpyHostToDevice);

        // for residual layer
        float* residual_copy; // for residual layer later
        cudaMalloc(&residual_copy, sizeof(float)* block_size*d_model);
        cudaMemcpy(residual_copy, d_input, sizeof(float)* block_size*d_model, cudaMemcpyDeviceToDevice);

        // Apply transformer decoder
        transformer_decoder(
            d_input
            d_output
            residual_copy
            config.block_size
            config.n_heads
            config.d_model
            config.head_dim
            config.n_blocks
            config.vocab_size
            config.qkv_weights
            config.mha_proj_weights
            config.ln1_weights
            config.ln2_weights
            config.ffwd_weights
            config.lnf_weights
            config.lm_head_weights
        );

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
    
}



// Helper function to cleanup generation resources
void cleanup_generation_resources(
    float* d_logits,
    float* d_probs,
    int* d_selected_token, 
    curandState* d_states
) {
    cudaFree(d_logits);
    cudaFree(d_probs);
    cudaFree(d_selected_token);
    cudaFree(d_states);
}

// int main() {
//     // Load vocabulary from JSON
//     int vocab_size;
//     char** vocab = load_vocab_json("vocab.json", &vocab_size);  // Changed from load_vocab_c
//     if (!vocab) {
//         printf("Failed to load vocabulary\n");
//         return 1;
//     }
    
//     // Convert input text to tokens
//     const char* input_text = "To be or not to be";
//     int input_length;
//     int* input_tokens = text_to_tokens(vocab, vocab_size, input_text, &input_length);
    
//     printf("Loaded vocabulary: %d tokens\n", vocab_size);
//     printf("Input: '%s' (%d tokens)\n\n", input_text, input_length);
    
//     // Generate text
//     generate_tokens_contextual(input_tokens, input_length, 50, vocab_size, vocab);
    
//     // Cleanup
//     free(input_tokens);
//     for (int i = 0; i < vocab_size; i++) {
//         if (vocab[i]) { // Add check for NULL before freeing
//             free(vocab[i]);
//         }
//     }
//     free(vocab);
    
//     return 0;
// }
