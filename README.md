[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5-zirqst)


do this for setting up 
git clone https://github.com/NVIDIA/cutlass.git extern/cutlass

run the enviorment 
apptainer shell --nv /singularity/cs217/cs217.2024-12-12.sif

nvcc cublas_gemm.cu     -o cublas_gemm     -I/usr/local/cuda-11.3/include     -L/usr/local/cuda-11.3/lib64     -lcublas -lcudart     -arch=sm_75     -std=c++14

