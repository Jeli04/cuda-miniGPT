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

__global__ void scatter_head(
    const float* __restrict__ head_out, 
    float* __restrict__ concat,
    int block_size,
    int num_heads,
    int head_dim,
    int head_idx
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = block_size * head_dim;
    if (tid >= total) return;
    int row = tid / head_dim;
    int col = tid % head_dim;
    // destination column offset = head_idx * head_dim
    concat[row * (num_heads*head_dim) + head_idx*head_dim + col] = head_out[tid];
}


void multi_head_attention(
    int block_size,
    int num_heads,
    int d_model,
    int head_dim,
    const float* qkv_w, 
    const float* o_proj_w, 
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
    splitQKV_horizontal<<<dim_grid, dim_block>>>(d_qkv_proj, d_Q, d_K, d_V, block_size, head_dim, num_heads);
    cudaDeviceSynchronize();

    int stride_h = block_size * head_dim; // stride for each head
    for (int h = 0; h < num_heads; h++){
        float* d_Qh = d_Q + h * stride_h;    // (B × Dh)
        float* d_Kh = d_K + h * stride_h;    // (B × Dh)
        float* d_Vh = d_V + h * stride_h;    // (B × Dh)
        float* Oh = d_output + h * stride_h;   // write‑back location

        // Compute attention scores 
        float* attn_scores;
        cudaMalloc(&attn_scores, sizeof(float)*block_size*block_size);
        basicSgemm(block_size, block_size, num_heads*head_dim, true, false, d_Qh, d_Kh, attn_scores);
        cudaDeviceSynchronize();
  
        // attention scaling + softmax
        float scale = 1.0f / sqrtf((float)head_dim);
        matrixMultiplyConstant<<<(block_size*block_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(attn_scores, scale, block_size*block_size);
        cudaDeviceSynchronize();

        softmax(attn_scores, attn_scores, block_size, block_size);
        cudaDeviceSynchronize();

        basicSgemm(block_size, head_dim, block_size, false, false, attn_scores, d_Vh, Oh);
        cudaDeviceSynchronize();
    }

    // // Compute attention scores 
    // float* attn_scores;
    // cudaMalloc(&attn_scores, sizeof(float)*block_size*block_size);
    // basicSgemm(block_size, block_size, num_heads*head_dim, false, true, d_Q, d_K, attn_scores);
    // cudaDeviceSynchronize();

    // float* output_h = (float*) malloc(block_size *  num_heads * head_dim * sizeof(float));
    // cudaMemcpy(output_h, d_Q, block_size *  num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    // std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/q2_dump.txt";
    // dumpMatrix(output_h, num_heads * head_dim, block_size, loc);


    // // float* attn_scores;
    // // cudaMalloc(&attn_scores, sizeof(float)*block_size*block_size);
    // // basicSgemm(block_size, block_size, num_heads*head_dim, false, true, d_Q, d_K, attn_scores);
    // // cudaDeviceSynchronize();

    // // attention scaling + softmax
    // float scale = 1.0f / sqrtf((float)head_dim);
    // matrixMultiplyConstant<<<(block_size*block_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(attn_scores, scale, block_size*block_size);
    // cudaDeviceSynchronize();

    // softmax(attn_scores, attn_scores, block_size, block_size);

    // // multply by values 
    // float* attn_output;
    // cudaMalloc(&attn_output, sizeof(float) * block_size * d_model);
    // basicSgemm(block_size, num_heads*head_dim, block_size, false, false, attn_scores, d_V, attn_output);
    // cudaDeviceSynchronize();

    // apply output projection
    dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim_grid = dim3((d_model + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    mysgemm<<<dim_grid, dim_block>>>(block_size, d_model, d_model, false, true, d_output, o_proj_w, d_output);

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

void transformer_decoder(
    float* d_input, // input to the block
    float* d_output, // output of the block
    float* residual_copy, // for residual connection
    int block_size, // batch size
    int n_heads, // number of heads
    int d_model, // model dimension
    int head_dim, // head dimension
    int n_blocks, // number of blocks
    int vocab_size, // vocab size
    const std::vector<float*>& qkv_weights, // QKV weights for each block
    const std::vector<float*>& mha_proj_weights, // MHA projection weights for each block
    const std::vector<float*>& ln1_weights, // layer norm 1 weights for each block
    const std::vector<float*>& ln2_weights, // layer norm 2 weights for each block
    const std::vector<float*>& ffwd_weights, // feed forward weights for each block
    const std::vector<float*>& lnf_weights, // final layer norm weights
    const std::vector<float*>& lm_head_weights // language model head weights
){
    const unsigned int BLOCK_SIZE = TILE_SIZE;

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
            mha_proj_weights[b], // output projection weights
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
    
    }

    // final layer norm 
    dim3 ln_grid(block_size);
    dim3 ln_block(d_model);
    size_t shmem = d_model * sizeof(float);
    layer_norm<<<ln_grid, ln_block, shmem>>>(
        d_output,
        d_output,
        lnf_weights[0], // gamma
        lnf_weights[1], // beta
        head_dim,
        d_model
    );

    // final linear transformation
    dim3 grid((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    mysgemm<<<grid, block>>>(block_size, vocab_size, d_model, false, true, d_output, lm_head_weights[1], d_output);
    // add bias
    int blocks = (block_size*d_model + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias<<<blocks, BLOCK_SIZE>>>(d_output, lm_head_weights[0], d_output, block_size*vocab_size, vocab_size);
    cudaDeviceSynchronize();

    float* output_h = (float*) malloc(sizeof(float) * block_size * vocab_size);
    cudaMemcpy(output_h, d_output, sizeof(float) * block_size * vocab_size, cudaMemcpyDeviceToHost);
    // printMatrix(h_input, block_size, n_heads * head_dim);
    std::string loc = "./block_output.txt";
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
//             head_dim,
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
//             mha_proj_weights[b], // output projection weights
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
//         layer_norm<<<grid, block, shmem>>>(
//             d_output,
//             d_output,
//             ln2_weights[b * 2], // gamma
//             ln2_weights[b * 2 + 1], // beta
//             head_dim,
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
//         lnf_weights[0], // gamma
//         lnf_weights[1], // beta
//         head_dim,
//         d_model
//     );

//     // final linear transformation
//     dim3 grid((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     dim3 block(BLOCK_SIZE, BLOCK_SIZE);
//     mysgemm<<<grid, block>>>(block_size, vocab_size, d_model, false, true, d_output, lm_head_weights[1], d_output);
//     // add bias
//     int blocks = (block_size*d_model + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     add_bias<<<blocks, BLOCK_SIZE>>>(d_output, lm_head_weights[0], d_output, block_size*vocab_size, vocab_size);
//     cudaDeviceSynchronize();

//     float* output_h = (float*) malloc(sizeof(float) * block_size * vocab_size);
//     cudaMemcpy(output_h, d_output, sizeof(float) * block_size * vocab_size, cudaMemcpyDeviceToHost);
//     // printMatrix(h_input, block_size, n_heads * head_dim);
//     std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/block_output.txt";
//     dumpMatrix(output_h, block_size, vocab_size, loc);


//     // float* output_h = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
//     // cudaMemcpy(output_h, d_output, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToHost);
//     // // printMatrix(h_input, block_size, n_heads * head_dim);
//     // std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
//     // dumpMatrix(output_h, block_size, n_heads * head_dim, loc);

//     cudaFree(d_input);
//     cudaFree(d_output);

//     return 0;
// }