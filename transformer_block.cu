#include <stdio.h>
#include "sgemm.cu"
#include "softmax.cu"
#include "tools.cu"
#include <vector>
#include <cstring>

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

    // get QKV projections
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dim_grid(num_heads, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_heads*3*head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    dim3 dim_grid((num_heads*3*head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    mysgemm<<<dim_grid, dim_block>>>(block_size, num_heads*3*head_dim, d_model, false, true, d_input, qkv_w, d_output);
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
    splitQKV<<<dim_grid, dim_block>>>(d_output, d_Q, d_K, d_V, block_size, num_heads*head_dim);
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

    float* output_h = (float*) malloc(block_size *  num_heads * head_dim * sizeof(float));
    cudaMemcpy(output_h, d_output, block_size *  num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
    dumpMatrix(output_h, block_size, num_heads * head_dim, loc);

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
    const unsigned int BLOCK_SIZE = TILE_SIZE;

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

    // setup input and output
    float* input = (float*) malloc(sizeof(float) * block_size * d_model);
    for(int i = 0; i < block_size * d_model; i++){
        if(i < 10) input[i] = 10.0f; // fill first 10 with tens
        else input[i] = 1.0f; // fill with ones
    }
    float* output = (float*) malloc(sizeof(float) * block_size * 3 * n_heads * head_dim);
    for(int i = 0; i < block_size * 3 * n_heads * head_dim; i++) output[i] = 2.0f; // fill with ones

    // move input and output 
    float* d_input;
    cudaMalloc(&d_input, sizeof(float)* block_size*d_model);
    cudaMemcpy(d_input, input, sizeof(float)* block_size*d_model, cudaMemcpyHostToDevice);
    float* d_output;
    cudaMalloc(&d_output, sizeof(float)* block_size*n_heads*3*head_dim);
    cudaMemcpy(d_output, output, sizeof(float)* block_size*n_heads*3*head_dim, cudaMemcpyHostToDevice);

    for(int b = 0; b < n_blocks; b++) {\
        // if(b == 3){
        //     float* h_input = (float*) malloc(sizeof(float) *  n_heads*3*head_dim*d_model);
        //     cudaMemcpy(h_input, qkv_weights[b], sizeof(float) * n_heads*3*head_dim*d_model, cudaMemcpyDeviceToHost);
        //     printMatrix(h_input, n_heads*3*head_dim, d_model);

        //     float* output_h = (float*) malloc(n_heads*3*head_dim*d_model * sizeof(float));
        //     cudaMemcpy(output_h, qkv_weights[b], n_heads*3*head_dim*d_model* sizeof(float), cudaMemcpyDeviceToHost);
        //     std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
        //     dumpMatrix(output_h, n_heads*3*head_dim, d_model, loc);
        // }

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

        cudaMemcpy(d_input, d_output, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToDevice);
        printf("Block %d processed.\n", b);
        // float* h_input = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
        // cudaMemcpy(h_input, d_input, sizeof(float) * block_size * n_heads * head_dim, cudaMemcpyDeviceToHost);
        // printMatrix(h_input, block_size, n_heads * head_dim);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}