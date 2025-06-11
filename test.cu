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

    // projections 
    float* d_qkv_projection;
    cudaMalloc(&d_qkv_projection, sizeof(float) * num_heads * 3 * head_dim * d_model); // allocate QKV projection

    // get QKV projections
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((num_heads*3*head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, (block_size + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    mysgemm<<<dim_grid, dim_block>>>(block_size, num_heads*3*head_dim, d_model, false, false, d_input, qkv_w, d_qkv_projection);
    cudaDeviceSynchronize();

    // split QKV into Q, K, V
    float* d_qkv_batched;
    cudaMalloc(&d_qkv_batched, sizeof(float) * 3 * num_heads * block_size * head_dim);

    // Rearrange to QKV layout
    dim3 reorder_grid((3 * num_heads * block_size * head_dim + BLOCK_SIZE) / BLOCK_SIZE);
    dim3 reorder_block(BLOCK_SIZE);
    splitQKVBatched<<<reorder_grid, reorder_block>>>(d_qkv_projection, d_qkv_batched, block_size, num_heads, head_dim);
    cudaDeviceSynchronize();
    cudaFree(d_qkv_projection); 

    // pointers to each QKV batch
    const float* d_Q_batched = d_qkv_batched;
    const float* d_K_batched = d_qkv_batched + (long)num_heads * block_size * head_dim;
    const float* d_V_batched = d_qkv_batched + 2L * (long)num_heads * block_size * head_dim;

    // Compute attention scores 
    float* attn_scores;
    cudaMalloc(&attn_scores, sizeof(float)*num_heads*block_size*block_size);
    // basicSgemm(block_size, block_size, num_heads*head_dim, false, true, d_Q, d_K, attn_scores);

    dim3 dim_block_batch(TILE_SIZE, TILE_SIZE, 1);
    dim3 dim_grid_batch((block_size + TILE_SIZE - 1) / TILE_SIZE, (block_size + TILE_SIZE - 1) / TILE_SIZE, num_heads); 
    batched_sgemm<<<dim_grid_batch, dim_block_batch>>>(
        block_size, block_size, head_dim, 
        false, 
        true,             
        d_Q_batched, 
        d_K_batched, 
        attn_scores,
        num_heads);
    cudaDeviceSynchronize();

    // attention scaling + softmax
    dim3 softmax_grid(block_size, num_heads); 
    dim3 softmax_block(BLOCK_SIZE); 
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
    batched_scaled_softmax_kernel<<<softmax_grid, softmax_block, shared_mem_size>>>(
        attn_scores,
        num_heads,
        block_size, // row
        block_size, // col
        scale
    );

    // float scale = 1.0f / sqrtf((float)head_dim);
    // matrixMultiplyConstant<<<(block_size*block_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(attn_scores, scale, block_size*block_size);
    // cudaDeviceSynchronize();
    // softmax(attn_scores, attn_scores, block_size, block_size);

    // multply by values 
    float* d_attn_output_batched; 
    cudaMalloc(&d_attn_output_batched, sizeof(float) * num_heads * block_size * head_dim);
    dim_grid_batch = dim3((head_dim + TILE_SIZE - 1) / TILE_SIZE, (block_size + TILE_SIZE - 1) / TILE_SIZE, num_heads); 
    batched_sgemm<<<dim_grid_batch, dim_block_batch>>>(
        block_size, 
        head_dim, 
        block_size, 
        false, 
        false,                     
        attn_scores, 
        d_V_batched, 
        d_attn_output_batched,
        num_heads);
    cudaDeviceSynchronize();      
    
    // basicSgemm(block_size, num_heads*head_dim, block_size, false, false, attn_scores, d_V, d_output);
    // cudaDeviceSynchronize();

    // combine the heads
    dim3 merge_grid((num_heads * block_size * head_dim + BLOCK_SIZE) / BLOCK_SIZE);
    dim3 merge_block(BLOCK_SIZE);
    combineBatchedHeads<<<merge_grid, merge_block>>>(d_attn_output_batched, d_output, block_size, num_heads, head_dim);
    cudaDeviceSynchronize();

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
    const int n_blocks = 1;
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


    // setup input and output
    float* input = (float*) malloc(sizeof(float) * block_size * d_model);
    for(int i = 0; i < block_size * d_model; i++){
        if(i < 10) input[i] = 10.0f; // fill first 10 with tens
        else input[i] = 1.0f; // fill with ones
    }
    float* output = (float*) malloc(sizeof(float) * block_size * n_heads * head_dim);
    for(int i = 0; i < block_size * n_heads * head_dim; i++) output[i] = 2.0f; // fill with ones

    // move input and output 
    float* d_input;
    cudaMalloc(&d_input, sizeof(float)* block_size*d_model);
    cudaMemcpy(d_input, input, sizeof(float)* block_size*d_model, cudaMemcpyHostToDevice);
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

        // // layer norm     
        // size_t shmem = d_model * sizeof(float);  
        // layer_norm<<<grid, block, shmem>>>(
        //     d_input,
        //     d_input,
        //     ln1_weights[b * 2], // gamma
        //     ln1_weights[b * 2 + 1], // beta
        //     head_dim,
        //     d_model
        // );
        // cudaDeviceSynchronize();

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

        // // residual connection
        // grid = dim3((block_size+BLOCK_SIZE-1)/BLOCK_SIZE,  (n_heads * head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        // block= dim3(BLOCK_SIZE, BLOCK_SIZE);
        // add_residual<<<grid, block>>>(residual_copy, d_output, d_output, block_size, n_heads * head_dim);
        // cudaDeviceSynchronize();

        // // copy new residual
        // cudaMemcpy(residual_copy, d_output, sizeof(float)* block_size*d_model, cudaMemcpyDeviceToDevice);

        // // layer norm     
        // layer_norm<<<grid, block, shmem>>>(
        //     d_output,
        //     d_output,
        //     ln2_weights[b * 2], // gamma
        //     ln2_weights[b * 2 + 1], // beta
        //     head_dim,
        //     d_model
        // );
        // cudaDeviceSynchronize();

        // // feed forward
        // ffwd(
        //     d_output, // input
        //     block_size, // batch size
        //     d_model, // d_model
        //     d_model * 4, // hidden size is 4 times the model size
        //     ffwd_weights[b * 4], // d_bias1
        //     ffwd_weights[b * 4 + 1], // d_weights1
        //     ffwd_weights[b * 4 + 2], // d_bias2
        //     ffwd_weights[b * 4 + 3]  // d_weights2
        // );

        // // residual connection
        // add_residual<<<grid, block>>>(residual_copy, d_output, d_output, block_size, n_heads * head_dim);
        // cudaDeviceSynchronize();

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