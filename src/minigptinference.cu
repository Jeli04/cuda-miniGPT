#include <stdio.h>
#include "tools.h"
#include <vector>
#include <cstring>
#include "generate.h"
#include "minigpt.h"
#include "positional_encoding_resources.h"

std::vector<float*> qkv_weights;
std::vector<float*> ln1_weights;
std::vector<float*> ln2_weights;
std::vector<float*> ffwd_weights;
std::vector<float*> mha_proj_weights;
std::vector<float*> lnf_weights;
std::vector<float*> lm_head_weights;

void load_all_weights(int n_blocks, int n_heads, int d_model, int head_dim, int vocab_size, const std::string& folder) {
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
    std::vector<std::string> lnf_dump_path = get_ln_f_paths(folder);
    lnf_weights = load_ln_f_weights(
        d_model,     
        lnf_dump_path
    );
    std::vector<std::string> lm_head_paths = get_lm_head_paths(folder);
    lm_head_weights = load_lm_head_weights(vocab_size, d_model, lm_head_paths);
    printf("All transformer weights loaded successfully.\n");
}

char** load_vocab_json(const char* filename, int* vocab_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open vocab file %s\n", filename);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* json_content = (char*)malloc(file_size + 1);
    fread(json_content, 1, file_size, file);
    json_content[file_size] = '\0';
    fclose(file);
    
    char** vocab = (char**)calloc(84, sizeof(char*));
    
    char* pos = json_content;
    
    while ((pos = strstr(pos, "\"")) != NULL) {
        pos++;
        
        char* id_end = strchr(pos, '"');
        if (!id_end) break;
        
        char id_str[10];
        int id_len = id_end - pos;
        strncpy(id_str, pos, id_len);
        id_str[id_len] = '\0';
        int token_id = atoi(id_str);
        
        pos = strstr(id_end, ": \"");
        if (!pos) break;
        pos += 3; 
        
        char* value_end = pos;
        while (*value_end && *value_end != '"') {
            if (*value_end == '\\') value_end++; 
            value_end++;
        }
        
        int value_len = value_end - pos;
        char* token_value = (char*)malloc(value_len + 1);
        strncpy(token_value, pos, value_len);
        token_value[value_len] = '\0';
        
        if (strcmp(token_value, "\\n") == 0) {
            free(token_value);
            token_value = strdup("\n");
        }
        
        if (token_id >= 0 && token_id < 84) {
            vocab[token_id] = token_value;
        }
        
        pos = value_end + 1;
    }
    
    *vocab_size = 84;
    free(json_content);
    return vocab;
}

void initialize_generation_resources(
    curandState** d_states
) {
    cudaMalloc(d_states, sizeof(curandState));
    
    setup_random_states<<<1, 1>>>(*d_states, 42, 1);
    cudaDeviceSynchronize();
}



int main() {
    const int d_model = 128;
    const int n_heads = 8;
    const int block_size = 64;
    const int head_dim = 16;
    const int n_blocks = 6;
    int vocab_size = 84;
    int max_seq_len = 64;
    int seq_len = block_size;

    int gen_vocab_size;
    char** gen_vocab = load_vocab_json("vocab.json", &gen_vocab_size);
    if (!gen_vocab) return 1;

    PositionalEncodingResources pos_resources;
    initialize_positional_encoding_resources(&pos_resources, max_seq_len, vocab_size, d_model);

    curandState* d_states;
    initialize_generation_resources(&d_states);

    const char* prompt = "To be or not to be";
    int prompt_length;
    int* prompt_tokens = text_to_tokens(gen_vocab, gen_vocab_size, prompt, &prompt_length);
    if (!prompt_tokens) {
        printf("Error: prompt_tokens is NULL!\n");
        return 1;
    }

    std::string weights_folder = "./weights_dump/";
    std::string location = weights_folder + "token_embedding_table.weight.txt";
    float* h_token_table = loadMatrix(vocab_size, d_model, location);
    location = weights_folder + "position_embedding_table.weight.txt";
    float* h_pos_table = loadMatrix(max_seq_len, d_model, location);

    float* d_token_table;
    float* d_pos_table;
    cudaMalloc(&d_token_table, vocab_size * d_model * sizeof(float));
    cudaMalloc(&d_pos_table, max_seq_len * d_model * sizeof(float));
    cudaMemcpy(d_token_table, h_token_table, vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_table, h_pos_table, max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);

    load_all_weights(n_blocks, n_heads, d_model, head_dim, vocab_size, weights_folder);
    
    TransformerWeights model_weights(
        d_token_table,
        d_pos_table,
        qkv_weights,
        mha_proj_weights,
        ln1_weights,
        ln2_weights,
        ffwd_weights,
        lnf_weights,
        lm_head_weights
    );

    MiniGPT gpt_model(
        block_size,
        n_heads,
        d_model,
        d_model * 4,
        n_blocks,
        vocab_size,
        model_weights
    );

    int max_new_gen_tokens = 50;
    generate_tokens_contextual(
        block_size,
        d_model,
        n_heads,
        head_dim,
        n_blocks,
        prompt_tokens,
        prompt_length,
        max_new_gen_tokens,
        gen_vocab_size,
        gen_vocab,
        d_states,
        pos_resources,
        gpt_model
    );

    printf("Text generation finished.\n");

    return 0;
}