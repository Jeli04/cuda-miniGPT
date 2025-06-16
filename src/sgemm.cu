#include <stdio.h>
#include <iostream>
#include "sgemm.h"

#define TILE_SIZE 64
#pragma once


__global__ void mysgemm(int m, int n, int k, bool A_t, bool B_t, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ float As[TILE_SIZE][TILE_SIZE]; // shared memory for A
    __shared__ float Bs[TILE_SIZE][TILE_SIZE]; // shared memory for B

    int row, col;
    const float *A_ptr;
    const float *B_ptr;
    float *C_ptr;

    row = blockIdx.y * blockDim.y + threadIdx.y; 
    col = blockIdx.x * blockDim.x + threadIdx.x; 

    A_ptr = A;
    B_ptr = B;
    C_ptr = C;

    float Cvalue = 0.0f;
    // iterate across the tiles 
    for(int tile_idx = 0; tile_idx < (k + TILE_SIZE - 1) / TILE_SIZE; tile_idx++) {
        int A_row = A_t ? tile_idx * TILE_SIZE + threadIdx.x : row;
        int A_col = A_t ? row : tile_idx * TILE_SIZE + threadIdx.x;
        if (A_row < (A_t ? k : m) && A_col < (A_t ? m : k)){
            As[threadIdx.y][threadIdx.x] = A_ptr[(A_row * k) + A_col];
        }
        else{
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int B_stride = B_t ? k : n; 
        int B_col = B_t ? tile_idx * TILE_SIZE + threadIdx.y : col; 
        int B_row = B_t ? col : tile_idx * TILE_SIZE + threadIdx.y;
        if (B_row < (B_t ? n : k) && B_col < (B_t ? k : n)){
            Bs[threadIdx.y][threadIdx.x] = B_ptr[B_row * B_stride + B_col]; 
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for(int i = 0; i < TILE_SIZE; i++) {
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();

    }

    if (row < m && col < n){
        C_ptr[row*n + col] = Cvalue;
    }
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, bool A_t, bool B_t, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE); // create the block dim 
    // we have n as first dimension and m as second dimension so easier to access for x and y (row and col)
    dim3 dim_grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE); // create the grid dim

    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    mysgemm<<<dim_grid, dim_block>>>(m, n, k, A_t, B_t, A, B, C);

    /*************************************************************************/
}

