nvcc -c transformer_block.cu -o transformer_block.o
nvcc -c sgemm.cu -o sgemm.o
nvcc -c softmax.cu -o softmax.o
nvcc -c tools.cu -o tools.o
nvcc -c layer_norm.cu -o layer_norm.o
nvcc -c ffwd.cu -o ffwd.o
nvcc transformer_block.o sgemm.o softmax.o tools.o layer_norm.o ffwd.o \
    -o transformer_block \
    -I/usr/local/cuda-11.3/include \
    -L/usr/local/cuda-11.3/lib64 \
    -lcublas -lcudart -arch=sm_75 -std=c++14