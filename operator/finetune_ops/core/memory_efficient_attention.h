/**
 * @file memory_efficient_attention.h
 * @brief Memory-efficient attention (streaming softmax, avoids materializing S×S)
 * 
 * Key ideas:
 * - Do not explicitly build full scores/probs matrix [B,H,S,S]
 * - Use online/streaming softmax; blockwise; reduce space from O(S²) to O(S)
 * - Two passes: compute row-wise max/sumExp, then normalize and accumulate context
 * 
 * 参考：
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
    bool use_causal_mask = true;      // Whether to apply causal mask
    float scale = -1.0f;              // Scale factor (-1 = auto: 1/sqrt(head_dim))
    int chunk_size = 512;             // Chunk size (for very long seqs; current impl uses full seq)
    bool save_probs = false;          // Whether to save probs (debug only, default off)
};

/**
 * @brief Memory-efficient scaled dot-product attention
 * 
 * @param q [batch, n_head, seq_len, head_dim]
 * @param k [batch, n_head, seq_len, head_dim]
 * @param v [batch, n_head, seq_len, head_dim]
 * @param causal_mask [seq_len, seq_len] optional, upper triangle = -inf
 * @param config options
 * @return context [batch, n_head, seq_len, head_dim]
 * 
 * Features:
 * - Does not materialize full scores/probs matrix
 * - Numerically stable (max-normalization)
 * - Memory: O(B·H·S·D) vs original O(B·H·S² + B·H·S·D)
 * - CPU implementation, pure C++, no external deps
 */
TensorPtr memory_efficient_attention(
    const TensorPtr& q,
    const TensorPtr& k,
    const TensorPtr& v,
    const TensorPtr& causal_mask = nullptr,
    const MemoryEfficientAttentionConfig& config = {}
);

/**
 * @brief Online/streaming softmax (single-row version for attention)
 * 
 * Given one logits row [S], compute softmax weights and accumulate output without materializing exp row.
 * Uses Welford/Kahan-style online algorithm for numerical stability.
 * 
 * @param logits input logits (one row of S values)
 * @param values corresponding values [S, D]
 * @param seq_len sequence length S
 * @param head_dim head dimension D
 * @param output output accumulation buffer [D]
 * @param max_val max logit of the row (for stability)
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

