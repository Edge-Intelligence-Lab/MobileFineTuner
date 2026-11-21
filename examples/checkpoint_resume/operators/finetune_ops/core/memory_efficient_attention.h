/**
 * @file memory_efficient_attention.h
 * @brief Memory-efficient attention implementation (streaming softmax, avoids materializing S×S matrix)
 * 
 * Core idea:
 * - Don't explicitly build full scores/probs matrix [B,H,S,S]
 * - Use online/streaming softmax algorithm, chunked computation, space complexity reduced from O(S²) to O(S)
 * - Two-pass scan: first pass computes row-wise max/sumExp, second pass normalizes and accumulates to context
 * 
 * References:
 * - FlashAttention (Dao et al., 2022)
 * - Online normalizer for softmax
 * - PyTorch SDPA memory-efficient kernel
 */

#pragma once

#include "tensor.h"
#include <vector>

namespace ops {

/**
 * @brief Memory-efficient attention configuration
 */
struct MemoryEfficientAttentionConfig {
    bool use_causal_mask = true;      // Whether to use causal mask
    float scale = -1.0f;              // Scaling factor (-1 means auto: 1/sqrt(head_dim))
    int chunk_size = 512;             // Chunk size (for very long sequences, current impl uses full sequence)
    bool save_probs = false;          // Whether to save probabilities (for debugging, default false)
};

/**
 * @brief Memory-efficient scaled dot-product attention
 * 
 * @param q [batch, n_head, seq_len, head_dim]
 * @param k [batch, n_head, seq_len, head_dim]
 * @param v [batch, n_head, seq_len, head_dim]
 * @param causal_mask [seq_len, seq_len] optional, upper triangle is -inf
 * @param config Configuration options
 * @return context [batch, n_head, seq_len, head_dim]
 * 
 * Features:
 * - Doesn't materialize full scores/probs matrix
 * - Numerically stable (max-normalization)
 * - Memory usage O(B*H*S*D) vs original O(B*H*S^2 + B*H*S*D)
 * - CPU implementation, pure C++, no external dependencies
 */
TensorPtr memory_efficient_attention(
    const TensorPtr& q,
    const TensorPtr& k,
    const TensorPtr& v,
    const TensorPtr& causal_mask = nullptr,
    const TensorPtr& pad_mask = nullptr,
    const MemoryEfficientAttentionConfig& config = {}
);

/**
 * @brief Online/streaming softmax (single-row version, for attention)
 * 
 * Given a row of logits [S], compute softmax weights and accumulate to output
 * without materializing the full exp row.
 * Uses Welford/Kahan style online algorithm, numerically stable.
 * 
 * @param logits Input logits (one row of S values)
 * @param values Corresponding values [S, D]
 * @param seq_len Sequence length S
 * @param head_dim Head dimension D
 * @param output Output accumulation buffer [D]
 * @param max_val Maximum logit in this row (for numerical stability)
 */
void online_softmax_weighted_sum(
    const float* logits,
    const float* values,
    int64_t seq_len,
    int64_t head_dim,
    float* output,
    float max_val
);

} // namespace ops
