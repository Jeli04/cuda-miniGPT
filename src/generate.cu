#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include <time.h>
#include "generate.h"
#include "minigpt.h"
#include "softmax.h"
#include "tools.h"


double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

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

    curandState localState = states[idx];
    float coin = curand_uniform(&localState);

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

    states[idx] = localState; 
    *selected_token = selected;
}

int* text_to_tokens(char** vocab, int vocab_size, const char* text, int* num_tokens) {
    int text_len = strlen(text);
    int* token_ids = (int*)malloc(text_len * sizeof(int));
    int valid_tokens = 0;
    
    for (int i = 0; i < text_len; i++) {
        char target_char = text[i];
        int token_id = -1;
        
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

void generate_tokens_contextual(
    int block_size,
    int d_model,
    int n_heads,
    int head_dim,
    int n_blocks,
    int* input_tokens,
    int input_length,
    int max_new_tokens,
    int vocab_size,
    char** vocab,
    curandState* d_states,
    PositionalEncodingResources& pos_resources,
    MiniGPT& gpt_model
) {
    printf("\n=== TOKEN GENERATION WITH CONDITIONING CHECK ===\n");
    double start_time = get_wall_time();

    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    float* d_probs;
    cudaMalloc(&d_probs, vocab_size * sizeof(float));
    int* d_selected_token;
    cudaMalloc(&d_selected_token, sizeof(int));
    float* d_input;
    cudaMalloc(&d_input, block_size * d_model * sizeof(float));

    int* conditioned_tokens;
    int conditioned_length;
    
    if (input_length > block_size) {
        conditioned_length = block_size;
        conditioned_tokens = (int*)malloc(conditioned_length * sizeof(int));
        int start_idx = input_length - block_size;
        memcpy(conditioned_tokens, input_tokens + start_idx, conditioned_length * sizeof(int));
    } else {
        conditioned_length = input_length;
        conditioned_tokens = (int*)malloc(conditioned_length * sizeof(int));
        memcpy(conditioned_tokens, input_tokens, conditioned_length * sizeof(int));
    }

    int max_seq_len = conditioned_length + max_new_tokens;
    int* token_sequence = (int*)malloc(max_seq_len * sizeof(int));
    memcpy(token_sequence, conditioned_tokens, conditioned_length * sizeof(int));
    int current_length = conditioned_length;

    char* full_text = (char*)malloc(10000);
    strcpy(full_text, "");
    for (int i = 0; i < conditioned_length; i++) {
        if (conditioned_tokens[i] >= 0 && conditioned_tokens[i] < vocab_size && vocab[conditioned_tokens[i]]) {
            strcat(full_text, vocab[conditioned_tokens[i]]);
        }
    }
    printf("Starting text: '%s'\n", full_text);

    for (int step = 0; step < max_new_tokens; step++) {
        printf("Step %d/%d: ", step + 1, max_new_tokens);
        
        int context_start = (current_length > block_size) ? current_length - block_size : 0;
        int context_length = current_length - context_start;
        int* current_context = token_sequence + context_start;
        
        printf("Processing context of length %d... ", context_length);

        cudaMemset(d_input, 0, block_size * d_model * sizeof(float));

        gpt_model.forward_pass(
            conditioned_length,
            max_seq_len,
            current_context,
            pos_resources,
            d_input,
            d_logits, 
            block_size,
            n_heads,
            d_model,
            head_dim,
            n_blocks,
            vocab_size
        );

        printf("Forward pass done, ");

        softmax(d_logits, d_probs, 1, vocab_size);
        printf("Softmax done, ");

        unsigned long step_seed = time(NULL) + step * 1000 + current_length;
        setup_random_states<<<1, 1>>>(d_states, step_seed, 1);
        cudaDeviceSynchronize();

        multinomial_sample_kernel<<<1, 1>>>(d_probs, d_selected_token, d_states, vocab_size);
        cudaDeviceSynchronize();

        int next_token;
        cudaMemcpy(&next_token, d_selected_token, sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("Sampled token %d", next_token);
        if (next_token >= 32 && next_token <= 126) {
            printf(" ('%c')", (char)next_token);
        }
        printf("\n");
        
        if (next_token < 0 || next_token >= vocab_size) {
            printf("ERROR: Invalid token %d\n", next_token);
            break;
        }

        token_sequence[current_length] = next_token;
        current_length++;
        
        if (vocab[next_token]) {
            strcat(full_text, vocab[next_token]);
        }
    }
    
    double end_time = get_wall_time();
    double total_time = end_time - start_time;
    
    printf("\n=== GENERATED TEXT ===\n");
    printf("'%s'\n", full_text);
    printf("======================\n");
    printf("Generation time: %.3f ms\n", total_time * 1000);
    
    // Proper cleanup
    cudaFree(d_logits);
    cudaFree(d_probs);
    cudaFree(d_selected_token);
    cudaFree(d_input);
    free(conditioned_tokens);
    free(token_sequence);
    free(full_text);
}

// char** load_vocab_json(const char* filename, int* vocab_size) {
//     *vocab_size = 128;
//     char** vocab = (char**)malloc(*vocab_size * sizeof(char*));
    
//     // Create ASCII character vocabulary (0-127)
//     for (int i = 0; i < *vocab_size; i++) {
//         vocab[i] = (char*)malloc(2 * sizeof(char));
//         vocab[i][0] = (char)i;
//         vocab[i][1] = '\0';
//     }
    
//     return vocab;
// }


// int main() {
    // cublasHandle_t cublasHandle;
    // cublasCreate(&cublasHandle);
    
    // int vocab_size;
    // char** vocab = load_vocab_json("vocab.json", &vocab_size);
    // if (!vocab) {
    //     printf("Failed to load vocabulary\n");
    //     return 1;
    // }
    
    // printf("=== TRANSFORMER VALIDATION TEST ===\n");
    // printf("Loaded vocabulary: %d tokens\n", vocab_size);
    
    // TransformerBlockCofig config;
    // config.block_size = 32;
    // config.n_heads = 4;
    // config.d_model = 102;
    // config.head_dim = config.d_model / config.n_heads;
    // config.n_blocks = 1;
    // config.vocab_size = vocab_size;
    
    // printf("Config: block_size=%d, n_heads=%d, d_model=%d, head_dim=%d\n", 
    //        config.block_size, config.n_heads, config.d_model, config.head_dim);
    
    // float *d_input, *d_output, *d_residual;
    // float *d_logits, *d_probs;
    // int *d_selected_token;
    // curandState *d_states;
    
    // cudaMalloc(&d_input, config.block_size * config.d_model * sizeof(float));
    // cudaMalloc(&d_output, config.block_size * config.d_model * sizeof(float));
    // cudaMalloc(&d_residual, config.block_size * config.d_model * sizeof(float));
    // cudaMalloc(&d_logits, config.vocab_size * sizeof(float));
    // cudaMalloc(&d_probs, config.vocab_size * sizeof(float));
    // cudaMalloc(&d_selected_token, sizeof(int));
    // cudaMalloc(&d_states, config.vocab_size * sizeof(curandState));
    
    // unsigned long fixed_seed = 12345;
    // setup_random_states<<<(config.vocab_size + 255) / 256, 256>>>(d_states, fixed_seed, config.vocab_size);
    // cudaDeviceSynchronize();
    
    // // Multiple test cases
    // const char* test_inputs[] = {
    //     "Hi",                                         
    //     "Hello world",                             
    //     "very long text text text text text text text text text text text text text text text text text text text text text ", 
    //     "A",                                         
    //     "text text text text text text text text text text text text text text "
    // };
    // int num_tests = sizeof(test_inputs) 
    
    // for (int test_idx = 0; test_idx < num_tests; test_idx++) {
    //     printf("\n=== TEST CASE %d ===\n", test_idx + 1);
    //     const char* input_text = test_inputs[test_idx];
    //     int input_length;
    //     int* input_tokens = text_to_tokens(vocab, vocab_size, input_text, &input_length);
        
    //     printf("Input: '%s' -> %d tokens: ", input_text, input_length);
    //     for (int i = 0; i < input_length; i++) {
    //         printf("%d ", input_tokens[i]);
    //     }
    //     printf("\n");
        
    //     setup_random_states<<<(config.vocab_size + 255) / 256, 256>>>(d_states, fixed_seed + test_idx, config.vocab_size);
    //     cudaDeviceSynchronize();
        
    //     generate_tokens_contextual(
    //         input_tokens,
    //         input_length,
    //         5,  
    //         vocab_size,
    //         vocab,
    //         d_output,
    //         d_logits,
    //         d_selected_token,
    //         d_states,
    //         config
    //     );
        
    //     free(input_tokens);
    // }
    
    // cudaFree(d_input);
    // cudaFree(d_output);
    // cudaFree(d_residual);
    // cudaFree(d_logits);
    // cudaFree(d_probs);
    // cudaFree(d_selected_token);
    // cudaFree(d_states);
    
    // for (int i = 0; i < vocab_size; i++) {
    //     if (vocab[i]) {
    //         free(vocab[i]);
    //     }
    // }
    // free(vocab);
    
    // cublasDestroy(cublasHandle);
    
    // printf("\n=== VALIDATION COMPLETE ===\n");
//     return 0;
// }
