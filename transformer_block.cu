#include <stdio.h>
#include <vector>
#include <cstring>
#include "sgemm.h"
#include "softmax.h"
#include "tools.h"
#include "transformer_block.h"
#include "layer_norm.h"
#include "ffwd.h"

#define TILE_SIZE 16\

#define CHECK_CUDA(msg) \
  { cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      printf("CUDA ERROR after %s: %s\n", msg, cudaGetErrorString(err)); \
      exit(1); \
    } \
  }

__global__ void splitQKV_vertical(const float* QKV, float* Q, float* K, float* V, int block_size, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < block_size * head_dim) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        Q[idx] = QKV[row * (3 * head_dim) + col];
        K[idx] = QKV[row * (3 * head_dim) + (head_dim) + col];
        V[idx] = QKV[row * (3 * head_dim) + (2 * head_dim) + col];
    }
}

__global__ void splitQKV_horizontal(
    const float* __restrict__ QKV, 
    float* __restrict__ Q,        
    float* __restrict__ K,
    float* __restrict__ V,
    int block_size,
    int head_dim,
    int n_heads)
{
    int total = block_size * head_dim * n_heads;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int col_out = idx % block_size;
    int row_out = idx / block_size;

    int head = row_out / head_dim;
    int d = row_out % head_dim;

    int row_stride_in = 3 * n_heads * head_dim;
    int q_col_in = head * head_dim + d;
    int k_col_in = n_heads * head_dim + head * head_dim + d;
    int v_col_in = 2 * n_heads * head_dim + head * head_dim + d;

    const float* row_ptr = QKV + col_out * row_stride_in;
    Q[idx] = row_ptr[q_col_in];
    K[idx] = row_ptr[k_col_in];
    V[idx] = row_ptr[v_col_in];
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

__global__ void apply_causal_mask(float* attn_scores, int block_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < block_size && col < block_size) {
        if (col > row) {  // Upper triangular part (future positions)
            attn_scores[row * block_size + col] = -INFINITY;
        }
    }
}
void multi_head_attention(
    int block_size,
    int num_heads,
    int d_model,
    int head_dim,
    const float* qkv_w, 
    const float* o_proj_w, 
    const float* o_proj_b, 
    float* d_input,
    float* d_output
){
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    // QKV projection: (block_size, 3*num_heads*head_dim)
    float* d_qkv_proj;
    cudaMalloc(&d_qkv_proj, sizeof(float) * block_size * 3 * num_heads * head_dim);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((3 * num_heads * head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    mysgemm<<<dim_grid, dim_block>>>(block_size, 3 * num_heads * head_dim, d_model, false, true, d_input, qkv_w, d_qkv_proj);
    cudaDeviceSynchronize();

    // Split QKV into Q, K, V
    float *d_Q, *d_K, *d_V;
    cudaMalloc(&d_Q, sizeof(float) * block_size * num_heads * head_dim);
    cudaMalloc(&d_K, sizeof(float) * block_size * num_heads * head_dim);
    cudaMalloc(&d_V, sizeof(float) * block_size * num_heads * head_dim);
    dim3 split_block(BLOCK_SIZE);
    dim3 split_grid((block_size * num_heads * head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    splitQKV_horizontal<<<split_grid, split_block>>>(
        d_qkv_proj, d_Q, d_K, d_V, block_size, head_dim, num_heads
    );
    cudaDeviceSynchronize();

    int stride_h = block_size * head_dim;
    for (int h = 0; h < num_heads; h++) {
        float* d_Qh = d_Q + h * stride_h;
        float* d_Kh = d_K + h * stride_h;
        float* d_Vh = d_V + h * stride_h;
        float* Oh = d_output + h * stride_h;

        // Attention scores (block_size x block_size)
        float* attn_scores;
        cudaMalloc(&attn_scores, sizeof(float) * block_size * block_size);
        basicSgemm(block_size, block_size, head_dim, false, true, d_Qh, d_Kh, attn_scores);
        cudaDeviceSynchronize();

        // Scale and softmax
        float scale = 1.0f / sqrtf((float)head_dim);
        matrixMultiplyConstant<<<(block_size * block_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            attn_scores, scale, block_size * block_size
        );
        cudaDeviceSynchronize();
        softmax(attn_scores, attn_scores, block_size, block_size);

        // Weighted sum of values
        basicSgemm(block_size, head_dim, block_size, false, false, attn_scores, d_Vh, Oh);
        cudaDeviceSynchronize();

        cudaFree(attn_scores);
    }

    // Output projection
    dim3 proj_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 proj_grid((d_model + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    mysgemm<<<proj_grid, proj_block>>>(block_size, d_model, num_heads * head_dim, false, true, d_output, o_proj_w, d_output);
    cudaDeviceSynchronize();

    // Optionally, add output projection bias if provided
    if (o_proj_b) {
        int blocks = (block_size * d_model + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bias<<<blocks, BLOCK_SIZE>>>(d_output, o_proj_b, d_output, block_size * d_model, d_model);
        cudaDeviceSynchronize();
    }

    // Free memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_qkv_proj);
}


void transformer_decoder(
    float* d_input,
    float* d_output,
    float* residual_copy,
    int block_size,
    int n_heads,
    int d_model,
    int head_dim,
    int n_blocks,
    int vocab_size,
    const std::vector<float*>& qkv_weights,
    const std::vector<float*>& mha_proj_weights,
    const std::vector<float*>& ln1_weights,
    const std::vector<float*>& ln2_weights,
    const std::vector<float*>& ffwd_weights,
    const std::vector<float*>& lnf_weights,
    const std::vector<float*>& lm_head_weights
) {
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    for(int b = 0; b < n_blocks; b++) {
        dim3 ln_grid(block_size);
        dim3 ln_block(d_model);
        size_t shmem = d_model * sizeof(float);

        // Layer norm 1
        layer_norm<<<ln_grid, ln_block, shmem>>>(
            d_input, d_input,
            ln1_weights[b * 2],
            ln1_weights[b * 2 + 1],
            block_size, d_model
        );
        cudaDeviceSynchronize();

        // Multi-head attention
        multi_head_attention(
            block_size, n_heads, d_model, head_dim,
            qkv_weights[b],
            mha_proj_weights[2*b+1],
            mha_proj_weights[2*b],
            d_input, d_output
        );

        // Residual
        dim3 res_grid((block_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (n_heads * head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 res_block(BLOCK_SIZE, BLOCK_SIZE);
        add_residual<<<res_grid, res_block>>>(
            residual_copy, d_output, d_output,
            block_size, n_heads * head_dim
        );
        cudaDeviceSynchronize();

        // Copy new residual
        cudaMemcpy(residual_copy, d_output, sizeof(float) * block_size * d_model, cudaMemcpyDeviceToDevice);

        // Layer norm 2
        layer_norm<<<ln_grid, ln_block, shmem>>>(
            d_output, d_output,
            ln2_weights[b * 2],
            ln2_weights[b * 2 + 1],
            block_size, d_model
        );
        cudaDeviceSynchronize();

        // Feed forward
        ffwd(
            d_output, block_size, d_model, d_model * 4,
            ffwd_weights[b * 4],
            ffwd_weights[b * 4 + 1],
            ffwd_weights[b * 4 + 2],
            ffwd_weights[b * 4 + 3]
        );

        // Residual
        add_residual<<<res_grid, res_block>>>(
            residual_copy, d_output, d_output,
            block_size, n_heads * head_dim
        );
        cudaDeviceSynchronize();

        if (b < n_blocks - 1) {
            cudaMemcpy(residual_copy, d_output, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_input, d_output, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToDevice);
        }

        printf("Block %d processed.\n", b);
    }

    // Final layer norm
    dim3 ln_grid(block_size);
    dim3 ln_block(d_model);
    size_t shmem = d_model * sizeof(float);
    layer_norm<<<ln_grid, ln_block, shmem>>>(
        d_output, d_output,
        lnf_weights[0], lnf_weights[1],
        block_size, d_model
    );
    cudaDeviceSynchronize();

    // Final linear and bias
    dim3 grid((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    mysgemm<<<grid, block>>>(block_size, vocab_size, d_model, false, true, d_output, lm_head_weights[1], d_output);

    int blocks = (block_size * vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias<<<blocks, BLOCK_SIZE>>>(d_output, lm_head_weights[0], d_output, block_size * vocab_size, vocab_size);
    cudaDeviceSynchronize();

    float* output_h = (float*) malloc(sizeof(float) * block_size * vocab_size);
    cudaMemcpy(output_h, d_output, sizeof(float) * block_size * vocab_size, cudaMemcpyDeviceToHost);
    std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/block_output.txt";
    dumpMatrix(output_h, block_size, vocab_size, loc);
}


// int main(){
//     const int d_model = 128; 
//     const int n_heads = 8;
//     const int block_size = 64;
//     const int head_dim = 16;
//     const int n_blocks = 6;
//     int vocab_size = 84;
//     int max_seq_len = 64;
//     int seq_len = 64; // "To be or not to be" length
//     const unsigned int BLOCK_SIZE = TILE_SIZE;

//     // load the weights
//     std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
//     std::vector<std::string> qkv_dump_path = get_qkv_path(n_blocks, n_heads, folder);
//     std::vector<float*> qkv_weights = load_qkv_weights(
//         n_blocks, 
//         n_heads, 
//         d_model, 
//         head_dim,
//         qkv_dump_path
//     );
//     std::vector<std::string> ln1_dump_path = get_layernorm_paths(n_blocks, 1, folder);
//     std::vector<float*> ln1_weights = load_layernorm_weights(
//         n_blocks, 
//         n_heads, 
//         d_model, 
//         head_dim,
//         ln1_dump_path
//     );
//     std::vector<std::string> ln2_dump_path = get_layernorm_paths(n_blocks, 2, folder);
//     std::vector<float*> ln2_weights = load_layernorm_weights(
//         n_blocks, 
//         n_heads, 
//         d_model, 
//         head_dim,
//         ln2_dump_path
//     );
//     std::vector<std::string> ffwd_dump_path = get_ffwd_paths(n_blocks, folder);
//     std::vector<float*> ffwd_weights = load_ffwd_weights(
//         n_blocks,
//         d_model,
//         d_model*4,         
//         ffwd_dump_path
//     );
//     std::vector<std::string> mha_proj_dump_path = get_mha_proj_paths(n_blocks, folder);
//     std::vector<float*> mha_proj_weights = load_mha_proj_weights(
//         n_blocks,
//         d_model,     
//         mha_proj_dump_path
//     );
//     std::vector<std::string> lnf_dump_path = get_ln_f_paths(folder);
//     std::vector<float*> lnf_weights = load_ln_f_weights(
//         d_model,     
//         lnf_dump_path
//     );
//     std::vector<std::string> lm_head_paths = get_lm_head_paths(folder);
//     std::vector<float*> lm_head_weights = load_lm_head_weights(vocab_size, d_model, lm_head_paths);

//     // setup input and output
//     float* input = (float*) malloc(sizeof(float) * block_size * d_model);
//     for(int i = 0; i < block_size * d_model; i++){
//         if(i < 10) input[i] = 10.0f; // fill first 10 with tens
//         else input[i] = 1.0f; // fill with ones
//     }
//     float* output = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
//     for(int i = 0; i < block_size * n_heads * head_dim; i++) output[i] = 2.0f; // fill with ones

//     // move input and output 
//     float* d_input;
//     cudaMalloc(&d_input, sizeof(float)* block_size*d_model);
//     cudaMemcpy(d_input, input, sizeof(float)* block_size*d_model, cudaMemcpyHostToDevice);
//     float* d_output;
//     cudaMalloc(&d_output, sizeof(float)* block_size*n_heads*head_dim);
//     cudaMemcpy(d_output, output, sizeof(float)* block_size*n_heads*head_dim, cudaMemcpyHostToDevice);
//     // for residual layer
//     float* residual_copy; // for residual layer later
//     cudaMalloc(&residual_copy, sizeof(float)* block_size*d_model);
//     cudaMemcpy(residual_copy, d_input, sizeof(float)* block_size*d_model, cudaMemcpyDeviceToDevice);

//     for(int b = 0; b < n_blocks; b++) {
//         dim3 grid(block_size);      
//         dim3 block(d_model);  

//         // layer norm     
//         size_t shmem = d_model * sizeof(float);  
//         layer_norm<<<grid, block, shmem>>>(
//             d_input,
//             d_input,
//             ln1_weights[b * 2], // gamma
//             ln1_weights[b * 2 + 1], // beta
//             block_size,
//             d_model
//         );
//         cudaDeviceSynchronize();

//         // float* output_h = (float*) malloc(block_size*d_model * sizeof(float));
//         // cudaMemcpy(output_h, d_output, block_size*d_model * sizeof(float), cudaMemcpyDeviceToHost);
//         // std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/layernorm.txt";
//         // dumpMatrix(output_h, block_size, d_model, loc);

//         // launch mha
//         multi_head_attention(
//             block_size,
//             n_heads,
//             d_model,
//             head_dim,
//             qkv_weights[b], // QKV weights
//             mha_proj_weights[2*b+1], // output projection weights
//             mha_proj_weights[2*b],
//             d_input, // input
//             d_output // output
//         );

//         // residual connection
//         grid = dim3((block_size+BLOCK_SIZE-1)/BLOCK_SIZE,  (n_heads * head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
//         block= dim3(BLOCK_SIZE, BLOCK_SIZE);
//         add_residual<<<grid, block>>>(residual_copy, d_output, d_output, block_size, n_heads * head_dim);
//         cudaDeviceSynchronize();

//         // copy new residual
//         cudaMemcpy(residual_copy, d_output, sizeof(float)* block_size*d_model, cudaMemcpyDeviceToDevice);

//         // layer norm     
//         dim3 ln_grid(block_size);
//         dim3 ln_block(d_model);
//         shmem = d_model*sizeof(float);
//         layer_norm<<<ln_grid, ln_block, shmem>>>(
//             d_output,
//             d_output,
//             ln2_weights[b * 2], // gamma
//             ln2_weights[b * 2+1], // beta
//             block_size,
//             d_model
//         );
//         cudaDeviceSynchronize();

//         // feed forward
//         ffwd(
//             d_output, // input
//             block_size, // batch size
//             d_model, // d_model
//             d_model * 4, // hidden size is 4 times the model size
//             ffwd_weights[b * 4], // d_bias1
//             ffwd_weights[b * 4 + 1], // d_weights1
//             ffwd_weights[b * 4 + 2], // d_bias2
//             ffwd_weights[b * 4 + 3]  // d_weights2
//         );

//         // residual connection
//         add_residual<<<grid, block>>>(residual_copy, d_output, d_output, block_size, n_heads * head_dim);
//         cudaDeviceSynchronize();

//         if (b < n_blocks - 1) {
//             cudaMemcpy(residual_copy, d_output, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToDevice);
//             cudaMemcpy(d_input, d_output, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToDevice);
//         }

//         printf("Block %d processed.\n", b);

//         float* output_h = (float*) malloc(block_size *  n_heads * head_dim * sizeof(float));
//         cudaMemcpy(output_h, d_output, block_size *  n_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
//         std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
//         dumpMatrix(output_h, block_size, n_heads * head_dim, loc);
//     }

//     // final layer norm 
//     dim3 ln_grid(block_size);
//     dim3 ln_block(d_model);
//     size_t shmem = d_model * sizeof(float);
//     layer_norm<<<ln_grid, ln_block, shmem>>>(
//         d_output,
//         d_output,
//         lnf_weights[1], // gamma
//         lnf_weights[0], // beta
//         head_dim,
//         d_model
//     );

//     // final linear transformation
//     // dim3 grid((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     dim3 grid((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     dim3 block(BLOCK_SIZE, BLOCK_SIZE);
//     float* final_output;
//     cudaMalloc(&final_output, sizeof(float) * block_size * vocab_size);
//     mysgemm<<<grid, block>>>(block_size, vocab_size, d_model, false, true, d_output, lm_head_weights[1], final_output);
//     cudaDeviceSynchronize();

//     float* output_h = (float*) malloc(sizeof(float) * d_model * vocab_size);
//     cudaMemcpy(output_h, lm_head_weights[1], sizeof(float) * d_model * vocab_size, cudaMemcpyDeviceToHost);
//     // printMatrix(h_input, block_size, n_heads * head_dim);
//     std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/output_lm.txt";
//     dumpMatrix(output_h, vocab_size, d_model, loc);

//     // add bias
//     int blocks = (block_size*vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     add_bias<<<blocks, BLOCK_SIZE>>>(final_output, lm_head_weights[0], final_output, block_size*vocab_size, vocab_size);
//     cudaDeviceSynchronize();

//     // float* output_h = (float*) malloc(sizeof(float) * block_size * vocab_size);
//     // cudaMemcpy(output_h, d_output, sizeof(float) * block_size * vocab_size, cudaMemcpyDeviceToHost);
//     // // printMatrix(h_input, block_size, n_heads * head_dim);
//     // std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
//     // dumpMatrix(output_h, block_size, vocab_size, loc);


//     // float* output_h = (float*) malloc(sizeof(float) * block_size * vocab_size);
//     // cudaMemcpy(output_h, d_output, sizeof(float) * block_size * vocab_size, cudaMemcpyDeviceToHost);
//     // // printMatrix(h_input, block_size, n_heads * head_dim);
//     // std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test2.txt";
//     // dumpMatrix(output_h, block_size, vocab_size, loc);

//     output_h = (float*) malloc(sizeof(float) * block_size * vocab_size);
//     cudaMemcpy(output_h, final_output, sizeof(float) * block_size * vocab_size, cudaMemcpyDeviceToHost);
//     // printMatrix(h_input, block_size, n_heads * head_dim);
//     loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test2.txt";
//     dumpMatrix(output_h, block_size, vocab_size, loc);

//     cudaFree(d_input);
//     cudaFree(d_output);

//     return 0;
// }