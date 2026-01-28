/**
 * @file memory_first_attention.h
 * [Documentation in English - see separate docs]
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 */

#pragma once

#include "tensor.h"
#include "ops.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace ops {
namespace memory_first {

/**
 * [Documentation available in English]
 * [Documentation in English - see separate docs]
 */
struct OnlineSoftmaxState {
    float max_val;          // [Translated]
    float exp_sum;      // exp(x - max) accumulateand
    std::vector<float> weighted_output;     // [Translated]
    
    OnlineSoftmaxState(int output_dim) 
        : max_val(-std::numeric_limits<float>::infinity()), 
          exp_sum(0.0f),
          weighted_output(output_dim, 0.0f) {}
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     * @param new_values corresponding Value block [block_cols, head_dim]
     * [Documentation available in English]
     * [Documentation available in English]
     */
    void update(const float* new_scores, const float* new_values, 
                int block_cols, int head_dim) {
                // [Translated]
        float new_max = max_val;
        for (int j = 0; j < block_cols; ++j) {
            new_max = std::max(new_max, new_scores[j]);
        }
        
        // [Translated comment removed - see documentation]
        if (new_max > max_val) {
            float scale = std::exp(max_val - new_max);
            exp_sum *= scale;
            for (auto& val : weighted_output) {
                val *= scale;
            }
            max_val = new_max;
        }
        
                // [Translated]
        for (int j = 0; j < block_cols; ++j) {
            float exp_score = std::exp(new_scores[j] - max_val);
            exp_sum += exp_score;
            
            // weighted_output += exp_score * new_values[j, :]
            for (int d = 0; d < head_dim; ++d) {
                weighted_output[d] += exp_score * new_values[j * head_dim + d];
            }
        }
    }
    
    /**
     * @brief finalnormalizationoutput
     */
    void normalize() {
        if (exp_sum > 0.0f) {
            for (auto& val : weighted_output) {
                val /= exp_sum;
            }
        }
    }
};

/**
 * [Documentation available in English]
 * 
 * @param query Query tensor [batch, seq_len, num_heads, head_dim]
 * @param key Key tensor [batch, seq_len, num_heads, head_dim]
 * @param value Value tensor [batch, seq_len, num_heads, head_dim]
 * @param causal is notusecausal mask
 * [Documentation available in English]
 * [Documentation available in English]
 * @return attentionoutput [batch, seq_len, num_heads, head_dim]
 * 
 * algorithmCore: 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * 4. applycausal mask（ifenabled）
 * [Documentation available in English]
 * [Documentation available in English]
 * 7. finalnormalization
 * 
 * memoryusage：
 * - Sij temporaryblock：row_block * col_block * 4B（default 32*32*4 = 4KB）
 * [Documentation available in English]
 * - no O(S²) intermediatematrix
 */
TensorPtr memory_first_attention_forward(
    const TensorPtr& query,
    const TensorPtr& key,
    const TensorPtr& value,
    bool causal = false,
    int row_block_size = 32,
    int col_block_size = 32
);

/**
 * [Documentation available in English]
 * 
 * @param x inputtensor [batch, seq_len, n_embd]
 * @param q_weight Query projectweight [n_embd, n_embd]
 * @param k_weight Key projectweight [n_embd, n_embd]
 * @param v_weight Value projectweight [n_embd, n_embd]
 * @param o_weight Output projectweight [n_embd, n_embd]
 * [Documentation available in English]
 * @param causal is notusecausal mask
 * @return attentionoutput [batch, seq_len, n_embd]
 */
TensorPtr memory_first_multihead_attention(
    const TensorPtr& x,
    const TensorPtr& q_weight,
    const TensorPtr& k_weight,
    const TensorPtr& v_weight,
    const TensorPtr& o_weight,
    int num_heads,
    bool causal = false
);

} // namespace memory_first
} // namespace ops

