#include <stdio.h>
#include "sgemm.cu"
#include "softmax.cu"
#include "tools.cu"
#include <vector>
#include <cstring>
#include "layer_norm.cu"
#include "ffwd.cu"
#include "positional_encoding.cu"

#define TILE_SIZE 16\

#define CHECK_CUDA(msg) \
  { cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      printf("CUDA ERROR after %s: %s\n", msg, cudaGetErrorString(err)); \
      exit(1); \
    } \
  }

__global__ void splitQKV(const float* QKV, float* Q, float* K, float* V, int block_size, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < block_size * head_dim) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        Q[idx] = QKV[row * (3 * head_dim) + col];
        K[idx] = QKV[row * (3 * head_dim) + (head_dim) + col];
        V[idx] = QKV[row * (3 * head_dim) + (2 * head_dim) + col];
    }
}

__global__ void combineQKV(
    const float* q_w,   // [head_dim × d_model]
    const float* k_w,   // [head_dim × d_model]
    const float* v_w,   // [head_dim × d_model]
    float* QKV_w,   // [d_model × (3*head_dim)]
    int d_model,
    int head_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d_model * head_dim;
    if (tid >= total) return;

    int i = tid % head_dim;           
    int j = tid / head_dim;          

    // Q part
    QKV_w[i * d_model + j] = q_w[i * d_model + j];
    // K part
    QKV_w[(i + head_dim) * d_model + j] = k_w[i * d_model + j];
    // V part
    QKV_w[(i + 2 * head_dim) * d_model + j] = v_w[i * d_model + j];
}


// good coalesced explanation 
// https://stackoverflow.com/questions/5041328/in-cuda-what-is-memory-coalescing-and-how-is-it-achieved
__global__ void matrixMultiplyConstant(float* input, float factor, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        input[idx] *= factor;
    }
}

__global__ void add_residual(const float* a, const float* b, float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        // printf("i: %d, j: %d\n", i, j);
        out[i * cols + j] = a[i * cols + j] + b[i * cols + j];
    } 
}

void multi_head_attention(
    int block_size,
    int num_heads,
    int d_model,
    int head_dim,
    const float* qkv_w, 
    float* d_input,
    float* d_output
){
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    float* d_qkv_proj;
    cudaMalloc(&d_qkv_proj, sizeof(float) * block_size * num_heads * 3 * head_dim);
    
    // get QKV projections
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dim_grid(num_heads, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_heads*3*head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    dim3 dim_grid((num_heads*3*head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    mysgemm<<<dim_grid, dim_block>>>(block_size, num_heads*3*head_dim, d_model, false, true, d_input, qkv_w, d_qkv_proj);
    cudaDeviceSynchronize();

    // split QKV into Q, K, V
    float* d_Q;
    cudaMalloc(&d_Q, sizeof(float)*block_size*num_heads*head_dim); // allocate Q
    float* d_K;
    cudaMalloc(&d_K, sizeof(float)*block_size*num_heads*head_dim); // allocate K
    float* d_V;
    cudaMalloc(&d_V, sizeof(float)*block_size*num_heads*head_dim); // allovate V 
    dim_block= dim3(BLOCK_SIZE); // create the block dim 
    dim_grid=dim3((block_size*3*num_heads*head_dim+BLOCK_SIZE)/BLOCK_SIZE); // create the grid dim
    splitQKV<<<dim_grid, dim_block>>>(d_qkv_proj, d_Q, d_K, d_V, block_size, num_heads*head_dim);
    cudaDeviceSynchronize();

    // Compute attention scores 
    float* attn_scores;
    cudaMalloc(&attn_scores, sizeof(float)*block_size*block_size);
    basicSgemm(block_size, block_size, num_heads*head_dim, false, true, d_Q, d_K, attn_scores);
    cudaDeviceSynchronize();

    // attention scaling + softmax
    float scale = 1.0f / sqrtf((float)head_dim);
    matrixMultiplyConstant<<<(block_size*block_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(attn_scores, scale, block_size*block_size);
    cudaDeviceSynchronize();

    softmax(attn_scores, attn_scores, block_size, block_size);

    // multply by values 
    basicSgemm(block_size, num_heads*head_dim, block_size, false, false, attn_scores, d_V, d_output);
    cudaDeviceSynchronize();

    // float* output_h = (float*) malloc(block_size *  num_heads * head_dim * sizeof(float));
    // cudaMemcpy(output_h, d_output, block_size *  num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    // std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test3.txt";
    // dumpMatrix(output_h, block_size, num_heads * head_dim, loc);

    // dealloc non static values
    // cudaFree(d_Q);
    // cudaFree(d_K);
    // cudaFree(d_V);
    // cudaFree(attn_scores);
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

    // setup input and output
    float* input = (float*) malloc(sizeof(float) * block_size * d_model);
    for(int i = 0; i < block_size * d_model; i++){
        if(i < 10) input[i] = 10.0f; // fill first 10 with tens
        else input[i] = 1.0f; // fill with ones
    }
    float* output = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
    for(int i = 0; i < block_size * n_heads * head_dim; i++) output[i] = 2.0f; // fill with ones

    // move input and output 
    // float* d_input;
    // cudaMalloc(&d_input, sizeof(float)* block_size*d_model);
    // cudaMemcpy(d_input, input, sizeof(float)* block_size*d_model, cudaMemcpyHostToDevice);
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


    float* output_h = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
    cudaMemcpy(output_h, d_output, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToHost);
    // printMatrix(h_input, block_size, n_heads * head_dim);
    std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
    dumpMatrix(output_h, block_size, n_heads * head_dim, loc);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}