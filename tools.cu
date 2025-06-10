#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>

#pragma once

std::vector<std::string> get_qkv_path(int n_blocks, int n_heads, const std::string& folder) {
    std::vector<std::string> paths;
    for (int b = 0; b < n_blocks; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            for (const auto& proj : {"query", "key", "value"}) {
                std::ostringstream oss;
                oss << folder
                    << "block." << b
                    << ".mha.attn_heads." << h
                    << "." << proj << ".weight.txt";
                paths.push_back(oss.str());
            }
        }
    }
    return paths;
}

std::vector<std::string> get_layernorm_paths(
    int n_blocks,
    int ln, 
    const std::string& folder)
{
    std::vector<std::string> paths;
    for (int b = 0; b < n_blocks; ++b) {
        for (const auto& param : {"weight", "bias"}) {
            std::ostringstream oss;
            oss << folder
                << "block." << b
                << ".ln" << ln
                << "." << param << ".txt";
            paths.push_back(oss.str());
        }
    }
    return paths;
}

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

void loadQKVCombined(
    const std::string& source,
    float* dst,
    int rows, 
    int cols
){
    // This function loads the pretrained QKV weights from the disk on host 
    // and combines them into a single matrix for each head.
    // This is a host function, so it will not be run on the device.

    std::ifstream infile(source);
    if (!infile) {
        std::cerr << "Could not open file: " << source << "\n";
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
            dst[row * cols + col] = std::stof(val);
            ++col;
        }
        ++row;
    }
}

void dumpMatrix(float* matrix, int rows, int cols, const std::string& destination) {
    std::ofstream outfile(destination);
    if (!outfile.is_open()) {
        std::cerr << "Could not open file for writing: " << destination << std::endl;
        return;
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            outfile << matrix[r * cols + c];
            if (c < cols - 1)
                outfile << " ";
        }
        outfile << "\n";
    }
    outfile.close();
}


void printMatrix(float* matrix, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // printf("%.4f ", matrix[r * cols + c]);
            printf("Row %d, Col %d: %.4f ", r, c, matrix[r * cols + c]);        
        }
        printf("\n");
    }
}

std::vector<float*> load_qkv_weights(
    int n_blocks, 
    int n_heads, 
    int d_model, 
    int head_dim,
    const std::vector<std::string>& weights_dump
) {
    std::vector<float*> all_weights(n_blocks);

    for(int b = 0; b < n_blocks; b++){
        float* h_W_qkv;
        cudaHostAlloc(&h_W_qkv, sizeof(float) * d_model * n_heads * head_dim * 3, cudaHostAllocDefault);

        float* h_Q_w = h_W_qkv;
        float* h_K_w = h_W_qkv + head_dim *  n_heads * d_model;
        float* h_V_w = h_W_qkv + 2 * head_dim * n_heads * d_model;

        // load the QKV weights for block b
        for(int i = 0; i < n_heads; i++) {
            int base = 3 * n_heads * b + 3 * i;
            loadQKVCombined(weights_dump[base + 0], h_Q_w + i * head_dim * d_model, head_dim, d_model);
            loadQKVCombined(weights_dump[base + 1], h_K_w + i * head_dim * d_model, head_dim, d_model);
            loadQKVCombined(weights_dump[base + 2], h_V_w + i * head_dim * d_model, head_dim, d_model);
            // printf("%s\n", weights_dump[base + 0].c_str());
            // printf("%s\n", weights_dump[base + 1].c_str());
            // printf("%s\n", weights_dump[base + 2].c_str());
        }

        float* d_W_qkv;
        cudaMalloc(&d_W_qkv, sizeof(float) * d_model * n_heads * head_dim * 3);
        cudaMemcpy(d_W_qkv, h_W_qkv, sizeof(float) * d_model * n_heads * head_dim * 3, cudaMemcpyHostToDevice);
        all_weights[b] = d_W_qkv;
        cudaFreeHost(h_W_qkv);  // free the host
    }
    return all_weights; // returns a host-side vector of device pointers
}


std::vector<float*> load_layernorm_weights(
    int n_blocks,
    int n_heads,
    int d_model,
    int head_dim,
    const std::vector<std::string>& weights_dump
) {
    std::vector<float*> all_weights(n_blocks);

    for(int b = 0; b < n_blocks; b++){
        std::string gamma_path = weights_dump[2 * b];
        float* h_gamma = loadMatrix(n_heads*head_dim, d_model, gamma_path);
        float* d_gamma;
        cudaMalloc(&d_gamma, sizeof(float) * n_heads * head_dim * d_model);
        cudaMemcpy(d_gamma, h_gamma, sizeof(float) * n_heads * head_dim * d_model, cudaMemcpyHostToDevice);
        all_weights[2 * b] = d_gamma;

        std::string beta_path = weights_dump[2 * b + 1];
        float* h_beta = loadMatrix(n_heads * head_dim, d_model, beta_path);
        float* d_beta;
        cudaMalloc(&d_beta, sizeof(float) * n_heads * head_dim * d_model);
        cudaMemcpy(d_beta, h_beta, sizeof(float) * n_heads * head_dim * d_model, cudaMemcpyHostToDevice);
        all_weights[2 * b + 1] = d_beta;
    }

    return all_weights; // returns a host-side vector of device pointers
}