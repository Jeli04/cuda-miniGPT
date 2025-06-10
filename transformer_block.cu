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
    const float* q_w, 
    const float* k_w, 
    const float* v_w, 
    float* input,
    float* output
){
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    // move input and output 
    float* d_input;
    cudaMalloc(&d_input, sizeof(float)* block_size*d_model);
    cudaMemcpy(d_input, input, sizeof(float)* block_size*d_model, cudaMemcpyHostToDevice);
    float* d_output;
    cudaMalloc(&d_output, sizeof(float)* block_size*num_heads*3*head_dim);
    cudaMemcpy(d_output, output, sizeof(float)* block_size*num_heads*3*head_dim, cudaMemcpyHostToDevice);

    // get QKV projections
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid(num_heads, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_heads*3*head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    mysgemm<<<dim_grid, dim_block>>>(block_size, num_heads*3*head_dim, d_model, false, true, d_input, q_w, d_output);

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

    // float* output_h = (float*) malloc(block_size * num_heads*head_dim * sizeof(float));
    // cudaMemcpy(output_h, d_Q, block_size * num_heads*head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    // std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
    // dumpMatrix(output_h, block_size, num_heads*head_dim, loc);
    // printMatrix(output_h, block_size, num_heads*head_dim);

    // Compute attention scores 
    float* attn_scores;
    cudaMalloc(&attn_scores, sizeof(float)*block_size*block_size);
    basicSgemm(block_size, block_size, num_heads*head_dim, false, true, d_Q, d_K, attn_scores);

    // attention scaling + softmax
    float scale = 1.0f / sqrtf((float)head_dim);
    matrixMultiplyConstant<<<(block_size*block_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(attn_scores, scale, block_size*block_size);
    softmax(attn_scores, attn_scores, block_size, block_size);

    // multply by values 
    basicSgemm(block_size, num_heads*head_dim, block_size, false, false, attn_scores, d_V, d_output);

    float* output_h = (float*) malloc(block_size * num_heads*head_dim * sizeof(float));
    cudaMemcpy(output_h, d_output, block_size * num_heads*head_dim * sizeof(float), cudaMemcpyDeviceToHost);
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
    float* d_input;
    cudaMalloc(&d_input, sizeof(float)* block_size*d_model);
    cudaMemcpy(d_input, input, sizeof(float)* block_size*d_model, cudaMemcpyHostToDevice);
    float* d_output;
    cudaMalloc(&d_output, sizeof(float)* block_size*head_dim);
    cudaMemcpy(d_output, output, sizeof(float)* block_size*head_dim, cudaMemcpyHostToDevice);

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
    basicSgemm(block_size, head_dim*3, d_model, false, false, d_input, QKV_w_d, QKV_d);

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
    basicSgemm(block_size, head_dim, block_size, false, false, attn_scores, V_d, d_output);

    float* output_h = (float*) malloc(block_size * head_dim * sizeof(float));
    cudaMemcpy(output_h, d_output, block_size * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/d_outputump.txt";
    dumpMatrix(output_h, block_size, head_dim, loc);


    // dealloc
    cudaFree(QKV_d);
    cudaFree(QKV_w_d);
}



int main(){
    const int d_model = 128; 
    const int n_heads = 2;
    const int block_size = 64;
    const int head_dim = 16;
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    std::vector<std::string> weights_dump = {
        "block.0.mha.attn_heads.0.query.weight.txt",
        "block.0.mha.attn_heads.0.key.weight.txt",
        "block.0.mha.attn_heads.0.value.weight.txt",
        "block.0.mha.attn_heads.1.query.weight.txt",
        "block.0.mha.attn_heads.1.key.weight.txt",
        "block.0.mha.attn_heads.1.value.weight.txt"
    };

    float* h_W_qkv;
    cudaHostAlloc(&h_W_qkv, sizeof(float) * d_model * n_heads * head_dim * 3, cudaHostAllocDefault);

    float* h_Q_w = h_W_qkv;
    float* h_K_w = h_W_qkv + head_dim *  n_heads * d_model;
    float* h_V_w = h_W_qkv + 2 * head_dim * n_heads * d_model;

    // load the QKV weights
    std::string file, source;
    for(int i = 0; i < n_heads; i++) {
        file = weights_dump[3*i + 0]; // query weight
        source = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/" + file;
        float* dst_q = h_Q_w + i * head_dim * d_model; // destination for query weight
        loadQKVCombined(
            source,
            dst_q,
            head_dim, 
            d_model
        );
        printf("%d\n", i * head_dim * d_model);

        file = weights_dump[3*i + 1]; // key weight
        source = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/" + file;
        float* dst_k = h_K_w + i * head_dim * d_model; // destination for key weight
        loadQKVCombined(
            source,
            dst_k,
            head_dim, 
            d_model
        );
        printf("%d\n", i * head_dim * d_model);

        file = weights_dump[3*i + 2]; // value weight
        source = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/" + file;
        float* dst_v = h_V_w + i * head_dim * d_model; // destination for value weight
        loadQKVCombined(
            source,
            dst_v,
            head_dim, 
            d_model
        );
    }

    dumpMatrix(h_W_qkv, n_heads * 3 * head_dim, d_model, "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/combined_qkv.txt");

    // move the weights to device
    float* d_W_qkv;
    cudaMalloc(&d_W_qkv, sizeof(float) * d_model * n_heads * head_dim * 3);
    cudaMemcpy(d_W_qkv, h_W_qkv, sizeof(float) * d_model * n_heads * head_dim * 3, cudaMemcpyHostToDevice);

    // setup input and output
    float* input = (float*) malloc(sizeof(float) * block_size * d_model);
    for(int i = 0; i < block_size * d_model; i++){
        if(i < 10) input[i] = 10.0f; // fill first 10 with tens
        else input[i] = 1.0f; // fill with ones
    }
    float* output = (float*) malloc(sizeof(float) * block_size * 3 * n_heads * head_dim);
    for(int i = 0; i < block_size * 3 * n_heads * head_dim; i++) output[i] = 1.0f; // fill with ones

    // launch mha
    multi_head_attention(
        block_size,
        n_heads,
        d_model,
        head_dim,
        d_W_qkv, // QKV weights
        d_W_qkv, // QKV weights
        d_W_qkv, // QKV weights
        input,   // input
        output   // output
    );


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

    return 0;
}