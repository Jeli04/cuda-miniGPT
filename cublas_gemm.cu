// cublas_gemm.cu
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>


float* loadMatrix(int rows, int cols, std::string& source){
  float* data = new float[rows * cols]; // or float data[rows * cols];

  std::ifstream infile(source);
  if (!infile) {
      std::cerr << "Could not open file.\n";
      exit(1);
  }

  std::string line;
  int row = 0;
  while (std::getline(infile, line) && row < rows) {
      if (line.empty()) continue;
      std::istringstream iss(line);
      std::string val;
      int col = 0;
      while (iss >> val && col < cols) {
          data[row * cols + col] = std::stof(val);
          ++col;
      }
      ++row;
  }

  for (int i = 0; i < std::min(5, rows * cols); ++i)
      std::cout << data[i] << " ";
  std::cout << std::endl;

  return data;
}

int main() {
  const int rows = 84;
  const int cols = 128;
  std::string folder = "/home/csmaj/jeli/final-project-sp2025-guys-performing-transformations-gpt/weights_dump/";
  std::string file = "token_embedding_table.weight.txt";
  std::string source = folder + file;

  float* h_A = loadMatrix(rows, cols, source); // load the data

  float h_B[cols * rows], h_C[rows * rows];
  // fill dummy values
  for(int i=0;i<rows*cols;++i) h_B[i] = 1.0f;

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, rows*cols*sizeof(float));
  cudaMalloc(&d_B, cols*rows*sizeof(float));
  cudaMalloc(&d_C, rows*rows*sizeof(float));

  cudaMemcpy(d_A, h_A, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, cols*rows*sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  const float alpha = 1.0f, beta = 0.0f;
  // C = alpha * A * B + beta * C
  // cublas by default works as column major so we have to tranpose to work in row
  cublasSgemm(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_T, // transpose
    rows, rows, cols,
    &alpha,
    d_A, cols,
    d_B, rows,
    &beta,
    d_C, rows
  );

  cudaMemcpy(h_C, d_C, rows*rows*sizeof(float), cudaMemcpyDeviceToHost);

  // allocate a row-major copy
  float* C_row = new float[rows * rows];

  // transpose from column-major h_C into row-major C_row
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < rows; ++c) {
      // h_C is column-major, so element (r,c) lives at h_C[ c*rows + r ]
      C_row[r * rows + c] = h_C[c * rows + r];
    }
  }

  // now you can print the first 5 in true row-major order:
  for (int i = 0; i < 5; ++i) {
    std::cout << C_row[i] << " ";
  }
  std::cout << std::endl;

  // cleanup
  delete[] C_row;
  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  return 0;
}



/* 
Block class - SA + MLP
SA class - 
Layer Norm 
Decoder 


*/