/**
 * @file lora_ops.h
 * @brief LoRA (Low-Rank Adaptation) operations implementsation
 * 
 * This file providess the core LoRA operations for parameter-efficient
 * fine-tuning. LoRA adapts pre-trained models by learning low-rank
 * matrices that are added to existing weights.
 */

#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>

namespace ops {

/**
 * @brief Configuration structure for LoRA operations
 * 
 * Contains all parameters needed to configure LoRA adaptation
 * including rank, scaling factor, and regularization options.
 */
struct LoRAOpsConfig {
    int rank;        /**< Rank of the low-rank matrices */
    float alpha;     /**< Scaling factor for LoRA adaptation */
    float dropout;   /**< Dropout rate for LoRA layers */
    bool use_bias;   /**< Whether to use bias in LoRA layers */
    bool residual;   /**< Whether to add residual connection */

    /**
     * @brief Default LoRA configuration
     * 
     * Creates a configuration with commonly used default values:
     * - rank: 8
     * - alpha: 32.0
     * - dropout: 0.1
     * - use_bias: false
     * - residual: true
     */
    LoRAOpsConfig() : rank(8), alpha(32.0f), dropout(0.1f),
                      use_bias(false), residual(true) {}

    /**
     * @brief Construct LoRA configuration with custom parameters
     * @param rank_ Rank of the low-rank matrices
     * @param alpha_ Scaling factor (default: 32.0)
     * @param dropout_ Dropout rate (default: 0.1)
     * @param use_bias_ Whether to use bias (default: false)
     * @param residual_ Whether to add residual connection (default: true)
     */
    LoRAOpsConfig(int rank_, float alpha_ = 32.0f, float dropout_ = 0.1f,
                  bool use_bias_ = false, bool residual_ = true)
        : rank(rank_), alpha(alpha_), dropout(dropout_),
          use_bias(use_bias_), residual(residual_) {}
};

namespace lora_ops {

TensorPtr forward(const Tensor& input,
                  const Tensor& lora_A,
                  const Tensor& lora_B,
                  const Tensor& bias,
                  const LoRAOpsConfig& config,
                  bool training = false);

std::vector<TensorPtr> backward(const Tensor& grad_output,
                                const Tensor& input,
                                const Tensor& lora_A,
                                const Tensor& lora_B,
                                const Tensor& hidden_state,
                                const LoRAOpsConfig& config);

TensorPtr apply_dropout(const Tensor& x, float dropout_rate, bool training);

float get_scale_factor(const LoRAOpsConfig& config);

void init_lora_weights(Tensor& lora_A, Tensor& lora_B,
                      int in_features, int out_features, int rank);

TensorPtr transpose(const Tensor& input, int dim1, int dim2);

TensorPtr batch_matmul(const Tensor& A, const Tensor& B,
                      int batch_size, int M, int N, int K);

}

}
