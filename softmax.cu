#include <stdio.h>
#include "sgemm.h"
#include "tools.h"
#include "softmax.h"

#define TILE_SIZE 16
#pragma once

// https://youtu.be/IpHjDoW4ffw?si=oVyTlxgyjhpEv39F

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
  extern __shared__ float shared[];
  int row_idx = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  const float* x = input + row_idx * cols;

  // get the max values with coarsing + reduce (regular reduction)
  float max_val = -INFINITY;
  for (unsigned int i = tid; i < cols; i+=block_size){
    max_val = fmaxf(max_val, x[i]);
  }

  shared[tid] = max_val;

  // reduce (we do / 2 since we are reducing the array by 2 at each step)
  for(unsigned int stride = block_size / 2; stride>=1; stride /= 2){
    __syncthreads();

    if(tid < stride){
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
  }
  __syncthreads();

  // numerator expoential value
  float max_offest = shared[0];

  for(unsigned int i = tid; i < cols; i+=block_size){
    // printf("here \n");
    output[row_idx * cols + i] = expf(x[i] - max_offest);
  }

  __syncthreads();

  // get the sum values + reduce (coarse reduction)
  x = output + row_idx * cols;
  float sum_val = 0.0f;

  for (unsigned int i = tid; i < cols; i+=block_size){
    sum_val += x[i];
  }
  shared[tid] = sum_val; 

  // reduce (we do / 2 since we are reducing the array by 2 at each step)
  for(unsigned int stride = block_size / 2; stride>=1; stride /= 2){
    __syncthreads();

    if(tid < stride){
      shared[tid] = shared[tid] + shared[tid + stride];
    }
  }  

  // divide numerator by denominator
  __syncthreads();
  float sum = shared[0];
  for (unsigned int i = tid; i < cols; i+=block_size){
    output[row_idx * cols + i] /= sum;
  }
  
}

void softmax(
    float* input,
    float* output,
    int rows,
    int cols
){
  int BLOCK_SIZE = 16; // length of the row
  dim3 grid(rows);
  dim3 block(BLOCK_SIZE);

  softmax_kernel<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(input, output, rows, cols);
}

/*
int main(){
  const int d_model = 128; 
  const int n_heads = 8;
  const int block_size = 64;
  const int head_dim = d_model / n_heads;

  std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
  std::string file = "block.0.mha.attn_heads.0.query.weight.txt";
  std::string source = folder + file;
  float* h_Q_w = loadMatrix(head_dim, d_model, source); // load the data
  float* h_Q_w_d;
  float* output_d; 

  cudaMallocHost(&h_Q_w_d, head_dim * d_model * sizeof(float));
  cudaMemcpy(h_Q_w_d, h_Q_w, sizeof(float)* head_dim*d_model, cudaMemcpyHostToDevice);

  float* output = (float*) malloc(sizeof(float)*head_dim*d_model);
  for(int i = 0; i < head_dim*d_model; i++) output[i] = 1.0f;

  cudaMallocHost(&output_d, head_dim * d_model * sizeof(float));
  cudaMemcpy(output_d, output, sizeof(float)* head_dim*d_model, cudaMemcpyHostToDevice);

  softmax(
    h_Q_w_d,
    output_d,
    head_dim,
    d_model 
  );

  float* output_h = (float*) malloc(head_dim * d_model * sizeof(float));
  cudaMemcpy(output_h, output_d, head_dim * d_model * sizeof(float), cudaMemcpyDeviceToHost);
  std::string loc = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/dumps/softmax_dump.txt";
  dumpMatrix(output_h, head_dim, d_model, loc);

  return 0;
}
*/