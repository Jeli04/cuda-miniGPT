#pragma once
#include "positional_encoding_resources.h"

void embed_sequence_sgemm(
    float* d_output,
    const float* d_token_table,
    const float* d_pos_table,
    const int* h_token_sequence,
    int seq_len,
    int d_model,
    int vocab_size,
    int max_seq_len,
    PositionalEncodingResources* resources
);