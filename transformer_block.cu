#include <stdio.h>
#include "sgemm.cu"
#include "tools.cpp"

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
    float* K_d;
    float* V_d;
    cudaMalloc(&Q_d, sizeof(float)*block_size*head_dim); // allocate K
    cudaMalloc(&K_d, sizeof(float)*block_size*head_dim); // allocate K
    cudaMalloc(&V_d, sizeof(float)*block_size*head_dim); // allovate V 
    dim3 dim_block(BLOCK_SIZE); // create the block dim 
    dim3 dim_grid((block_size*head_dim+BLOCK_SIZE)/BLOCK_SIZE); // create the grid dim
    splitQKV<<<dim_grid, dim_block>>>(QKV_d, Q_d, K_d, V_d, block_size, head_dim);

    float* K_h = (float*) malloc(block_size * head_dim * sizeof(float));
    cudaMemcpy(K_h, Q_d, block_size * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/q_dump.txt";
    dumpMatrix(K_h, block_size, head_dim, loc);


    // get attention scores
    float* attn_scores;
    cudaMalloc(&attn_scores, sizeof(float)*block_size*block_size);
    basicSgemm(block_size, block_size, head_dim, false, true, Q_d, K_d, attn_scores);

    float* QK_h = (float*) malloc(block_size * block_size * sizeof(float));
    cudaMemcpy(QK_h, attn_scores, block_size * block_size * sizeof(float), cudaMemcpyDeviceToHost);
    loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/qk_dump.txt";
    dumpMatrix(QK_h, block_size, block_size, loc);


    // softmax + attention scaling

    // multply values 


    // dealloc
    cudaFree(QKV_d);
    cudaFree(QKV_w_d);
}


// class TransformerBlock {
// public:   

// }

int main(){
    const int d_model = 128; 
    const int n_heads = 8;
    const int block_size = 64;
    const int head_dim = d_model / n_heads;

    std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
    std::string file = "block.0.mha.attn_heads.0.query.weight.txt";
    std::string source = folder + file;
    const float* h_Q_w = loadMatrix(d_model/n_heads, d_model, source); // load the data

    file = "block.0.mha.attn_heads.0.key.weight.txt";
    source = folder + file;
    const float* h_K_w = loadMatrix(d_model/n_heads, d_model, source); // load the data

    file = "block.0.mha.attn_heads.0.value.weight.txt";
    source = folder + file;
    const float* h_V_w = loadMatrix(d_model/n_heads, d_model, source); // load the data

    float* input = (float*) malloc(sizeof(float)*block_size*d_model);
    for(int i = 0; i < block_size*d_model; i++) input[i] = 1.0f;
    float* output = (float*) malloc(sizeof(float)*block_size*head_dim);
    for(int i = 0; i < block_size*d_model; i++) input[i] = 1.0f;

    self_attention(
        block_size,
        d_model,
        head_dim,
        h_Q_w, 
        h_K_w, 
        h_V_w, 
        input,
        output
    );

    return 0;
}