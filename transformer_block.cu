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

__global__ void combineHeadWeightsKernel(
    const float** head_weights,  // [H] (array of device pointers, itself on device)
    int H,
    int head_dim,
    int hidden_dim,
    float* out                   // [H * head_dim * hidden_dim]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * head_dim * hidden_dim;
    if (idx >= total) return;

    // Decode flat idx into (head, row, col)
    int col = idx % hidden_dim;
    int tmp = idx / hidden_dim;
    int row = tmp % head_dim;
    int head = tmp / head_dim;

    // Pointer to this head's weight block
    const float* src = head_weights[head];
    int src_idx = row * hidden_dim + col;

    out[idx] = src[src_idx];
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
    dim3 dim_grid(num_heads, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (num_heads*3*head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    mysgemm<<<dim_grid, dim_block>>>(block_size, num_heads*3*head_dim, d_model, false, true, input_d, q_w, output_d);

    float* output_h = (float*) malloc(block_size * num_heads*3*head_dim * sizeof(float));
    cudaMemcpy(output_h, output_d, block_size * num_heads*3*head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/test.txt";
    dumpMatrix(output_h, block_size, num_heads*3*head_dim, loc);
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
    const int n_heads = 2;
    const int block_size = 64;
    const int head_dim = 16;
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
    // head 0 weights
    std::string file = "block.0.mha.attn_heads.0.query.weight.txt";
    std::string source = folder + file;
    const float* h_Q_w_0 = loadMatrix(head_dim, d_model, source); // load the data
    const float* d_Q_w_0;
    cudaMalloc(&d_Q_w_0, sizeof(float)*head_dim*d_model);
    cudaMemcpy((void*)d_Q_w_0, (void*)h_Q_w_0, sizeof(float)*head_dim*d_model, cudaMemcpyHostToDevice);

    file = "block.0.mha.attn_heads.0.key.weight.txt";
    source = folder + file;
    const float* h_K_w_0 = loadMatrix(head_dim, d_model, source); // load the data
    const float* d_K_w_0;
    cudaMalloc(&d_K_w_0, sizeof(float)*head_dim*d_model);
    cudaMemcpy((void*)d_K_w_0, (void*)h_K_w_0, sizeof(float)*head_dim*d_model, cudaMemcpyHostToDevice);

    file = "block.0.mha.attn_heads.0.value.weight.txt";
    source = folder + file;
    const float* h_V_w_0 = loadMatrix(head_dim, d_model, source); // load the data
    const float* d_V_w_0;
    cudaMalloc(&d_V_w_0, sizeof(float)*head_dim*d_model);
    cudaMemcpy((void*)d_V_w_0, (void*)h_V_w_0, sizeof(float)*head_dim*d_model, cudaMemcpyHostToDevice);

    float* QKV_d_0;
    cudaMalloc(&QKV_d_0, sizeof(float)* head_dim * 3 * d_model);

    dim3 dim_block(BLOCK_SIZE); // create the block dim 
    dim3 dim_grid((3*head_dim*d_model+BLOCK_SIZE)/BLOCK_SIZE); // create the grid dim
    combineQKV<<<dim_grid, dim_block>>>(
        d_Q_w_0, d_K_w_0, d_V_w_0, QKV_d_0,
        d_model, head_dim
    );
    cudaDeviceSynchronize();

    float* QKV_h_0 = (float*) malloc(sizeof(float) * 3 * head_dim * d_model);
    cudaMemcpy(QKV_h_0, QKV_d_0, sizeof(float) * 3 * head_dim * d_model, cudaMemcpyDeviceToHost);
    std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/combined_qkv_head0.txt";
    dumpMatrix(QKV_h_0, 3*head_dim, d_model, loc);


    // head 1 weights 
    file = "block.0.mha.attn_heads.1.query.weight.txt";
    source = folder + file;
    const float* h_Q_w_1 = loadMatrix(head_dim, d_model, source); // load the data
    const float* d_Q_w_1;
    cudaMalloc(&d_Q_w_1, sizeof(float)*head_dim*d_model);
    cudaMemcpy((void*)d_Q_w_1, (void*)h_Q_w_1, sizeof(float)*head_dim*d_model, cudaMemcpyHostToDevice);

    file = "block.0.mha.attn_heads.1.key.weight.txt";
    source = folder + file;
    const float* h_K_w_1 = loadMatrix(head_dim, d_model, source); // load the data
    const float* d_K_w_1;
    cudaMalloc(&d_K_w_1, sizeof(float)*head_dim*d_model);
    cudaMemcpy((void*)d_K_w_1, (void*)h_K_w_1, sizeof(float)*head_dim*d_model, cudaMemcpyHostToDevice);

    file = "block.0.mha.attn_heads.1.value.weight.txt";
    source = folder + file;
    const float* h_V_w_1 = loadMatrix(head_dim, d_model, source); // load the data
    const float* d_V_w_1;
    cudaMalloc(&d_V_w_1, sizeof(float)*head_dim*d_model);
    cudaMemcpy((void*)d_V_w_1, (void*)h_V_w_1, sizeof(float)*head_dim*d_model, cudaMemcpyHostToDevice);

    float* QKV_d_1;
    cudaMalloc(&QKV_d_1, sizeof(float) * head_dim * 3 * d_model);

    combineQKV<<<dim_grid, dim_block>>>(
        d_Q_w_1, d_K_w_1, d_V_w_1, QKV_d_1,
        d_model, head_dim
    );
    cudaDeviceSynchronize();

    float* QKV_h_1 = (float*) malloc(sizeof(float) * 3 * head_dim * d_model);
    cudaMemcpy(QKV_h_1, QKV_d_1, sizeof(float) * 3 * head_dim * d_model, cudaMemcpyDeviceToHost);
    loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/combined_qkv_head1.txt";
    dumpMatrix(QKV_h_1, 3*head_dim, d_model, loc);


    float* h_head_weights[n_heads];
    h_head_weights[0] = QKV_d_0;
    h_head_weights[1] = QKV_d_1;
    const float** d_head_weights;
    cudaMalloc(&d_head_weights, n_heads * sizeof(float*));
    cudaMemcpy(d_head_weights, h_head_weights, n_heads * sizeof(float*), cudaMemcpyHostToDevice);

    // Output allocation
    float* d_QKV_combined;
    cudaMalloc(&d_QKV_combined, n_heads * 3 * head_dim * d_model * sizeof(float));

    // reset the grid
    dim_grid = dim3((n_heads * 3 * head_dim * d_model + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch
    combineHeadWeightsKernel<<<dim_grid, dim_block>>>(d_head_weights, n_heads, 3*head_dim, d_model, d_QKV_combined);

    float* h_QKV_combined = (float*) malloc(sizeof(float) * n_heads * 3 * head_dim * d_model);
    cudaMemcpy(h_QKV_combined, d_QKV_combined, sizeof(float) * n_heads * 3 * head_dim * d_model, cudaMemcpyDeviceToHost);
    loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/combined_qkv.txt";
    dumpMatrix(h_QKV_combined, n_heads*3*head_dim, d_model, loc);

    float* input = (float*) malloc(sizeof(float)*block_size*d_model);
    for(int i = 0; i < block_size*d_model; i++) input[i] = 1.0f;
    float* output = (float*) malloc(sizeof(float)*block_size*n_heads*3*head_dim);
    for(int i = 0; i < block_size*n_heads*3*head_dim; i++) input[i] = 1.0f;

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

    multi_head_attention(
        block_size,
        n_heads,
        d_model,
        head_dim,
        d_QKV_combined, 
        d_QKV_combined, 
        d_QKV_combined, 
        input,
        output
    );

    return 0;
}