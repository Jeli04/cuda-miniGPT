#include <stdio.h>
#include "sgemm.cu"
#include "softmax.cu"
#include "tools.cpp"
#include <vector>
#include <cstring>
#define TILE_SIZE 16

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

    int row_off = j * (3 * head_dim);

    QKV_w[row_off +      i] = q_w[j * head_dim + i];
    QKV_w[row_off + head_dim + i] = k_w[j * head_dim + i];
    QKV_w[row_off + 2*head_dim + i] = v_w[j * head_dim + i];
}

void combineHeadWeightsHost(
    const std::vector<const float*>& head_weights,
    int H,
    int head_dim,
    int hidden_dim,
    float* out
) {
    size_t per_head_bytes = size_t(head_dim) * hidden_dim * sizeof(float);
    for (int h = 0; h < H; ++h) {
        // destination offset: h*head_dim rows of hidden_dim columns
        float* dst = out + size_t(h) * head_dim * hidden_dim;
        // copy that head's entire [head_dim × hidden_dim] block
        std::memcpy(dst, head_weights[h], per_head_bytes);
    }
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
    const float* q_w, 
    const float* k_w, 
    const float* v_w, 
    float* input,
    float* output
){
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    // move input and output 
    float* input_d;
    cudaMalloc(&input_d, sizeof(float)* block_size*d_model);
    cudaMemcpy(input_d, input, sizeof(float)* block_size*d_model, cudaMemcpyHostToDevice);
    float* output_d;
    cudaMalloc(&output_d, sizeof(float)* block_size*d_model);
    cudaMemcpy(output_d, output, sizeof(float)* block_size*d_model, cudaMemcpyHostToDevice);

    // get attention scores
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid(num_heads, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_model  + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    mysgemm<<<dim_grid, dim_block>>>(block_size, num_heads*head_dim, d_model, false, true, input_d, q_w, output_d);

    float* output_h = (float*) malloc(block_size * num_heads*head_dim * sizeof(float));
    cudaMemcpy(output_h, output_d, block_size * num_heads*head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
    dumpMatrix(output_h, block_size, num_heads*head_dim, loc);
}

void self_attention(
    int block_size,
    int d_model,
    int head_dim,
    const float* q_w, 
    const float* k_w, 
    const float* v_w, 
    float* input,
    float* output
){
    /*
        sa = softmax(QK / sqrt(dim_head)) * V
        q_w: d_model x head_dim (loaded is head_dim x d_model)
        k_w: d_model x head_dim (loaded is head_dim x d_model)
        v_w: d_model x head_dim (loaded is head_dim x d_model)

        Q: B x block_size x head_dim
        K: B x block_size x head_dim
        V: B x block_size x head_dim
        attn_scores (Q x K.T): B x block_size x block_size

        input: block_size x d_model 
        output: block_size x head_dim
    */
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    // will move the allocation out of this later, allocation will all be done once in the beginning

    // move input and output 
    float* input_d;
    cudaMalloc(&input_d, sizeof(float)* block_size*d_model);
    cudaMemcpy(input_d, input, sizeof(float)* block_size*d_model, cudaMemcpyHostToDevice);
    float* output_d;
    cudaMalloc(&output_d, sizeof(float)* block_size*head_dim);
    cudaMemcpy(output_d, output, sizeof(float)* block_size*head_dim, cudaMemcpyHostToDevice);

    // allocate combined qkv  
    float* QKV_d;
    cudaMalloc(&QKV_d, sizeof(float)* block_size * head_dim * 3);

    // combine weights into single matrix
    float* QKV_w;
    QKV_w = (float*) malloc( sizeof(float)*d_model*head_dim*3);
    for (int i = 0; i < head_dim; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float q = q_w[i * d_model + j];
            float k = k_w[i * d_model + j];
            float v = v_w[i * d_model + j];

            QKV_w[j * (3*head_dim) + i] = q;
            QKV_w[j * (3*head_dim) + head_dim + i] = k;
            QKV_w[j * (3*head_dim) + 2*head_dim + i] = v;
        }
    }

    // printMatrix(QKV_w, d_model, head_dim*3);
    std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/qkv_dump.txt";
    dumpMatrix(QKV_w, d_model, head_dim*3, loc);

    // move qkv weights onto device
    float* QKV_w_d; 
    cudaMalloc(&QKV_w_d, sizeof(float)* d_model * 3 * head_dim);
    cudaMemcpy(QKV_w_d, QKV_w, sizeof(float)* d_model*head_dim*3, cudaMemcpyHostToDevice);

    // get QKV values
    basicSgemm(block_size, head_dim*3, d_model, false, false, input_d, QKV_w_d, QKV_d);

    float* QKV_h = (float*) malloc(block_size * 3 * head_dim * sizeof(float));
    cudaMemcpy(QKV_h, QKV_d, block_size * 3 * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    dumpMatrix(QKV_h, block_size, 3*head_dim, loc);

    // split QKV into Q, K, V
    float* Q_d;
    cudaMalloc(&Q_d, sizeof(float)*block_size*head_dim); // allocate Q
    float* K_d;
    cudaMalloc(&K_d, sizeof(float)*block_size*head_dim); // allocate K
    float* V_d;
    cudaMalloc(&V_d, sizeof(float)*block_size*head_dim); // allovate V 
    dim3 dim_block(BLOCK_SIZE); // create the block dim 
    dim3 dim_grid((block_size*head_dim+BLOCK_SIZE)/BLOCK_SIZE); // create the grid dim
    splitQKV<<<dim_grid, dim_block>>>(QKV_d, Q_d, K_d, V_d, block_size, head_dim);

    float* K_h = (float*) malloc(block_size * head_dim * sizeof(float));
    cudaMemcpy(K_h, V_d, block_size * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/v_dump.txt";
    dumpMatrix(K_h, block_size, head_dim, loc);


    // get attention scores
    float* attn_scores;
    cudaMalloc(&attn_scores, sizeof(float)*block_size*block_size);
    basicSgemm(block_size, block_size, head_dim, false, true, Q_d, K_d, attn_scores);

    // attention scaling + softmax
    float scale = 1.0f / sqrtf((float)head_dim);
    matrixMultiplyConstant<<<(block_size*block_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(attn_scores, scale, block_size*block_size);
    softmax(attn_scores, attn_scores, block_size, block_size);
    
    // multply by values 
    basicSgemm(block_size, head_dim, block_size, false, false, attn_scores, V_d, output_d);

    float* output_h = (float*) malloc(block_size * head_dim * sizeof(float));
    cudaMemcpy(output_h, output_d, block_size * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/output_dump.txt";
    dumpMatrix(output_h, block_size, head_dim, loc);


    // dealloc
    cudaFree(QKV_d);
    cudaFree(QKV_w_d);
}



int main(){
    const int d_model = 128; 
    const int n_heads = 8;
    const int block_size = 64;
    const int head_dim = d_model / n_heads;

    std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
    // head 0 weights
    std::string file = "block.0.mha.attn_heads.0.query.weight.txt";
    std::string source = folder + file;
    const float* h_Q_w_0 = loadMatrix(d_model/n_heads, d_model, source); // load the data

    file = "block.0.mha.attn_heads.0.key.weight.txt";
    source = folder + file;
    const float* h_K_w_0 = loadMatrix(d_model/n_heads, d_model, source); // load the data

    file = "block.0.mha.attn_heads.0.value.weight.txt";
    source = folder + file;
    const float* h_V_w_0 = loadMatrix(d_model/n_heads, d_model, source); // load the data

    // head 1 weights 
    file = "block.0.mha.attn_heads.1.query.weight.txt";
    source = folder + file;
    const float* h_Q_w_1 = loadMatrix(d_model/n_heads, d_model, source); // load the data

    file = "block.0.mha.attn_heads.1.key.weight.txt";
    source = folder + file;
    const float* h_K_w_1 = loadMatrix(d_model/n_heads, d_model, source); // load the data

    file = "block.0.mha.attn_heads.1.value.weight.txt";
    source = folder + file;
    const float* h_V_w_1 = loadMatrix(d_model/n_heads, d_model, source); // load the data


    // combine the weights across each set of Q, K and V 
    std::vector<const float*> Q_heads = {
        h_Q_w_0, h_Q_w_1
    };
    
    // 2) Allocate a big host buffer:
    float* h_Q_combined = (float*)malloc(2 * head_dim * d_model * sizeof(float));
    
    // 3) Combine on the host:
    combineHeadWeightsHost(
        Q_heads, 2, head_dim, d_model, h_Q_combined
    );
    
    // 4) Copy up to device in one shot:
    float* d_Q_combined;
    cudaMalloc(&d_Q_combined, 2 * head_dim * d_model * sizeof(float));
    cudaMemcpy(d_Q_combined, h_Q_combined,
               2 * head_dim * d_model * sizeof(float),
               cudaMemcpyHostToDevice);

    // float* output_h = (float*) malloc(d_model * 2 * head_dim * sizeof(float));
    // cudaMemcpy(output_h, d_Q_combined, d_model * 2 * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    // std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
    // dumpMatrix(output_h, 2*head_dim, d_model, loc);

    std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/query.txt";
    float* copy = (float*) malloc(sizeof(float)*d_model/n_heads*d_model);
    for(int i = 0; i < d_model/n_heads*d_model; i++) copy[i] = h_Q_w_0[i];
    dumpMatrix(copy, d_model/n_heads, d_model, loc);

    float* input = (float*) malloc(sizeof(float)*block_size*d_model);
    for(int i = 0; i < block_size*d_model; i++) input[i] = 1.0f;
    float* output = (float*) malloc(sizeof(float)*block_size*d_model);
    for(int i = 0; i < block_size*2*head_dim; i++) input[i] = 1.0f;

    // self_attention(
    //     block_size,
    //     d_model,
    //     head_dim,
    //     h_Q_w, 
    //     h_K_w, 
    //     h_V_w, 
    //     input,
    //     output
    // );

    combineQKV<<<

    multi_head_attention(
        block_size,
        2,
        d_model,
        head_dim,
        d_Q_combined, 
        d_Q_combined, 
        d_Q_combined, 
        input,
        output
    );

    return 0;
}