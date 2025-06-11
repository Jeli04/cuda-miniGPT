#pragma once
#include <vector>
#include <string>
#include "positional_encoding_resources.h"

float* loadMatrix(int rows, int cols, std::string& source);
void loadQKVCombined(const std::string& source, float* dst, int rows, int cols);
void dumpMatrix(float* matrix, int rows, int cols, const std::string& destination);
void printMatrix(float* matrix, int rows, int cols);

std::vector<std::string> get_qkv_path(int n_blocks, int n_heads, const std::string& folder);
std::vector<std::string> get_layernorm_paths(int n_blocks, int ln, const std::string& folder);
std::vector<std::string> get_ffwd_paths(int n_blocks, const std::string& folder);
std::vector<std::string> get_mha_proj_paths(int n_blocks, const std::string& folder);
std::vector<std::string> get_ln_f_paths(const std::string& folder);
std::vector<std::string> get_lm_head_paths(const std::string& folder);

std::vector<float*> load_qkv_weights(int n_blocks, int n_heads, int d_model, int head_dim, const std::vector<std::string>& weights_dump);
std::vector<float*> load_layernorm_weights(int n_blocks, int n_heads, int d_model, int head_dim, const std::vector<std::string>& weights_dump);
std::vector<float*> load_ffwd_weights(int n_blocks, int d_model, int hidden_dim, const std::vector<std::string>& weights_dump);
std::vector<float*> load_mha_proj_weights(int n_blocks, int d_model, const std::vector<std::string>& weights_dump);
std::vector<float*> load_ln_f_weights(int d_model, const std::vector<std::string>& weights_dump);
std::vector<float*> load_lm_head_weights(int vocab_size, int d_model, const std::vector<std::string>& weights_dump);