nvcc -c generate.cu -o generate.o
nvcc -c minigptinference.cu -o minigptinference.o
nvcc minigptinference.o minigpt.o transformer_block.o positional_encoding.o softmax.o sgemm.o generate.o ffwd.o tools.o positional_encoding_resources.o layer_norm.o \
    -o minigptinference \
    -I/usr/local/cuda-11.3/include \
    -L/usr/local/cuda-11.3/lib64 \
    -lcublas -lcudart -arch=sm_75 -std=c++14