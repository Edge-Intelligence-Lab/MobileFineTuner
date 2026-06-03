/**
 * @file lm_loss.h
 * @brief Language-model-specific loss functions (supports 3D logits and ignore_index)
 */

#pragma once

#include "tensor.h"
#include <string>
#include <cmath>

namespace ops {

/**
 * @brief Cross-Entropy loss for language modeling
 * 
 * @param logits [B, S, V] float32
 * @param labels [B, S] int32, PAD positions use ignore_index
 * @param ignore_index Label value to ignore (default -100)
 * @param reduction "mean" | "sum" | "none"
 * @return 
 *   - "mean": scalar loss averaged over valid tokens
 *   - "sum": scalar loss summed over valid tokens
 *   - "none": [B,S] per-token NLL
 * 
 * Features:
 * - Numerically stable (logsumexp)
 * - Automatically skips ignore_index (excluded from loss/grad)
 * - Autograd supported
 */
TensorPtr lm_cross_entropy(const TensorPtr& logits,
                          const TensorPtr& labels,
                          int ignore_index = -100,
                          const std::string& reduction = "mean");

/**
 * @brief LM cross entropy without materializing dense [B,S,V] logits.
 *
 * Computes the same shifted objective for "mean" and "sum" reductions as:
 *   lm_cross_entropy(matmul_rhs_T(hidden, lm_head_weight), labels, ignore_index, reduction)
 *
 * Shapes:
 * - hidden: [B, S, H]
 * - lm_head_weight: [V, H] (Qwen tied embedding weight layout)
 * - labels: [B, S], with shifted labels and ignore_index support
 *
 * This is intended for LoRA / frozen-base fine-tuning. It preserves dense-CE
 * gradients w.r.t. hidden states while avoiding allocation of [B,S,V] logits
 * and gradients. For full-token labels this still supervises every shifted
 * token; for masked labels it naturally skips ignore_index positions.
 */
TensorPtr selected_token_lm_cross_entropy(const TensorPtr& hidden,
                                         const TensorPtr& lm_head_weight,
                                         const TensorPtr& labels,
                                         int ignore_index = -100,
                                         const std::string& reduction = "mean");

/**
 * @brief Alias with clearer semantics for dense/full-token CE.
 *
 * This computes the same objective as lm_cross_entropy(matmul_rhs_T(...)) but
 * streams vocab rows instead of materializing dense logits. It is suitable for
 * full-token causal LM training and for masked-label tasks.
 */
TensorPtr streaming_lm_cross_entropy(const TensorPtr& hidden,
                                    const TensorPtr& lm_head_weight,
                                    const TensorPtr& labels,
                                    int ignore_index = -100,
                                    const std::string& reduction = "mean");

/**
 * @brief Compute perplexity from mean NLL
 */
inline float perplexity_from_loss(float mean_nll) {
    return std::exp(mean_nll);
}

}  // namespace ops
