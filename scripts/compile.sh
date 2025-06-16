nvcc -c minigpt.cu -o minigpt.o
nvcc -c transformer_block.cu -o transformer_block.o
nvcc -c positional_encoding.cu -o positional_encoding.o
nvcc -c softmax.cu -o softmax.o
nvcc -c sgemm.cu -o sgemm.o
nvcc -c generate.cu -o generate.o
nvcc -c ffwd.cu -o ffwd.o
nvcc -c tools.cu -o tools.o
nvcc -c positional_encoding_resources.cu -o positional_encoding_resources.o
nvcc -c layer_norm.cu -o layer_norm.o
nvcc -c minigptinference.cu -o minigptinference.o
nvcc minigptinference.o minigpt.o transformer_block.o positional_encoding.o softmax.o sgemm.o generate.o ffwd.o tools.o positional_encoding_resources.o layer_norm.o \
    -o minigptinference \
    -I/usr/local/cuda-11.3/include \
    -L/usr/local/cuda-11.3/lib64 \
    -lcublas -lcudart -arch=sm_75 -std=c++14