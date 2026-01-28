/**
 * @file memory_first_attention.cpp
 * [Documentation available in English]
 */

#include "memory_first_attention.h"
#include "mobile_safe_matmul.h"
#include <cstring>
#include <iostream>

namespace ops {
namespace memory_first {

TensorPtr memory_first_attention_forward(
    const TensorPtr& query,
    const TensorPtr& key,
    const TensorPtr& value,
    bool causal,
    int row_block_size,
    int col_block_size
) {
    // inputshape：[batch, seq_len, num_heads, head_dim]
    auto q_shape = query->shape();
    if (q_shape.size() != 4) {
        throw TensorError("memory_first_attention: query must be 4D [batch, seq_len, num_heads, head_dim]");
    }
    
    int64_t batch = q_shape[0];
    int64_t seq_len = q_shape[1];
    int64_t num_heads = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    // createoutputtensor
    auto output = zeros(q_shape, query->dtype(), query->device());
    
    const float* q_data = query->data<float>();
    const float* k_data = key->data<float>();
    const float* v_data = value->data<float>();
    float* out_data = output->data<float>();
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // aseach batch and head independentprocess
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
                        // [Translated]
            int64_t head_offset = (b * seq_len * num_heads + h) * head_dim;
            int64_t stride_seq = num_heads * head_dim;
            
                        // [Translated]
            std::vector<float> scores_block(row_block_size * col_block_size);
            
                        // [Translated]
            for (int64_t row_start = 0; row_start < seq_len; row_start += row_block_size) {
                int64_t row_end = std::min(row_start + row_block_size, seq_len);
                int64_t actual_row_block = row_end - row_start;
                
                // [Translated comment removed - see documentation]
                std::vector<OnlineSoftmaxState> row_states;
                row_states.reserve(actual_row_block);
                for (int64_t i = 0; i < actual_row_block; ++i) {
                    row_states.emplace_back(head_dim);
                }
                
                                // [Translated]
                for (int64_t col_start = 0; col_start < seq_len; col_start += col_block_size) {
                    int64_t col_end = std::min(col_start + col_block_size, seq_len);
                    int64_t actual_col_block = col_end - col_start;
                    
                                        // [Translated]
                    // Qi: [actual_row_block, head_dim]
                    // Kj: [actual_col_block, head_dim]
                    // Sij: [actual_row_block, actual_col_block]
                    
                    std::memset(scores_block.data(), 0, 
                               actual_row_block * actual_col_block * sizeof(float));
                    
                    for (int64_t i = 0; i < actual_row_block; ++i) {
                        int64_t qi_pos = row_start + i;
                        const float* qi_ptr = q_data + head_offset + qi_pos * stride_seq;
                        
                        for (int64_t j = 0; j < actual_col_block; ++j) {
                            int64_t kj_pos = col_start + j;
                            const float* kj_ptr = k_data + head_offset + kj_pos * stride_seq;
                            
                            // dot product：qi · kj
                            float dot = 0.0f;
                            for (int64_t d = 0; d < head_dim; ++d) {
                                dot += qi_ptr[d] * kj_ptr[d];
                            }
                            
                            float score = dot * scale;
                            
                            // applycausal mask
                            if (causal && kj_pos > qi_pos) {
                                score = -std::numeric_limits<float>::infinity();
                            }
                            
                            scores_block[i * actual_col_block + j] = score;
                        }
                    }
                    
                                        // [Translated]
                    for (int64_t i = 0; i < actual_row_block; ++i) {
                        const float* row_scores = &scores_block[i * actual_col_block];
                        
                                                // [Translated]
                        std::vector<float> value_block(actual_col_block * head_dim);
                        for (int64_t j = 0; j < actual_col_block; ++j) {
                            int64_t vj_pos = col_start + j;
                            const float* vj_ptr = v_data + head_offset + vj_pos * stride_seq;
                            std::memcpy(&value_block[j * head_dim], vj_ptr, head_dim * sizeof(float));
                        }
                        
                                                // [Translated]
                        row_states[i].update(row_scores, value_block.data(), 
                                           actual_col_block, head_dim);
                    }
                }
                
                                // [Translated]
                for (int64_t i = 0; i < actual_row_block; ++i) {
                    row_states[i].normalize();
                    
                    int64_t qi_pos = row_start + i;
                    float* out_ptr = out_data + head_offset + qi_pos * stride_seq;
                    
                    std::memcpy(out_ptr, row_states[i].weighted_output.data(), 
                               head_dim * sizeof(float));
                }
            }
        }
    }
    
    // [Translated comment removed - see documentation]
    if (query->requires_grad() || key->requires_grad() || value->requires_grad()) {
        output->set_requires_grad(true);
                // [Translated]
    }
    
    return output;
}

TensorPtr memory_first_multihead_attention(
    const TensorPtr& x,
    const TensorPtr& q_weight,
    const TensorPtr& k_weight,
    const TensorPtr& v_weight,
    const TensorPtr& o_weight,
    int num_heads,
    bool causal
) {
    // x: [batch, seq_len, n_embd]
    auto x_shape = x->shape();
    if (x_shape.size() != 3) {
        throw TensorError("memory_first_multihead_attention: x must be 3D [batch, seq_len, n_embd]");
    }
    
    int64_t batch = x_shape[0];
    int64_t seq_len = x_shape[1];
    int64_t n_embd = x_shape[2];
    int64_t head_dim = n_embd / num_heads;
    
    // Q/K/V project
    auto q = matmul(x, q_weight);  // [batch, seq_len, n_embd]
    auto k = matmul(x, k_weight);
    auto v = matmul(x, v_weight);
    
    // Reshape: [batch, seq_len, n_embd] -> [batch, seq_len, num_heads, head_dim]
    auto q_reshaped = reshape(q, {batch, seq_len, num_heads, head_dim});
    auto k_reshaped = reshape(k, {batch, seq_len, num_heads, head_dim});
    auto v_reshaped = reshape(v, {batch, seq_len, num_heads, head_dim});
    
        // [Translated]
    auto attn_out = memory_first_attention_forward(
        q_reshaped, k_reshaped, v_reshaped, causal, 32, 32
    );
    
        // [Translated]
    auto attn_flat = reshape(attn_out, {batch, seq_len, n_embd});
    
    // Output project
    auto output = matmul(attn_flat, o_weight);
    
    return output;
}

} // namespace memory_first
} // namespace ops

