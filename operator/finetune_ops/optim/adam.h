/**
 * @file adam.h
 * @brief Adam optimizer implementsation
 * 
 * This file providess the Adam optimizer implementsation with support
 * for adaptive learning rates, momentum, and optional AMSGrad variant.
 * Adam is widely used for training deep neural networks.
 */

#pragma once

#include "optimizer.h"
#include <unordered_map>

namespace ops {

/**
 * @brief Configuration structure for Adam optimizer
 * 
 * Extends the base OptimizerConfig with Adam-specific parameters
 * including beta coefficients and epsilon for numerical stability.
 */
struct AdamConfig : public OptimizerConfig {
    float beta1;    /**< First moment decay rate (default: 0.9) */
    float beta2;    /**< Second moment decay rate (default: 0.999) */
    float epsilon;  /**< Small constant for numerical stability (default: 1e-8) */

    /**
     * @brief Default Adam configuration
     * 
     * Creates a configuration with commonly used default values:
     * - beta1: 0.9
     * - beta2: 0.999
     * - epsilon: 1e-8
     * - learning_rate: 0.001 (from base class)
     */
    AdamConfig() : OptimizerConfig(),
        beta1(0.9f), beta2(0.999f), epsilon(1e-8f) {}

    /**
     * @brief Construct Adam configuration with custom parameters
     * @param learning_rate_ Learning rate for parameter updates
     * @param beta1_ First moment decay rate (default: 0.9)
     * @param beta2_ Second moment decay rate (default: 0.999)
     * @param epsilon_ Small constant for numerical stability (default: 1e-8)
     * @param weight_decay_ L2 regularization coefficient (default: 0.0)
     * @param clip_grad_norm_ Gradient clipping threshold (default: 1.0)
     * @param amsgrad_ Whether to use AMSGrad variant (default: false)
     */
    AdamConfig(
        float learning_rate_,
        float beta1_ = 0.9f,
        float beta2_ = 0.999f,
        float epsilon_ = 1e-8f,
        float weight_decay_ = 0.0f,
        float clip_grad_norm_ = 1.0f,
        bool amsgrad_ = false
    ) : OptimizerConfig(learning_rate_, weight_decay_,
                       clip_grad_norm_, amsgrad_),
        beta1(beta1_), beta2(beta2_), epsilon(epsilon_) {}
};

struct AdamState : public OptimizerState {
    size_t step;
    std::vector<TensorPtr> m;
    std::vector<TensorPtr> v;
    std::vector<TensorPtr> v_hat;  // Fix: v_max -> v_hat

    void to_file(const std::string& path) const override;
    void from_file(const std::string& path) override;
};

class Adam : public Optimizer {
public:
    explicit Adam(const AdamConfig& config);
    ~Adam() override = default;

    void step(const std::vector<TensorPtr>& parameters,
             const std::vector<TensorPtr>& gradients) override;

    void save_state(const std::string& path) const override;
    void load_state(const std::string& path) override;
    
    void set_learning_rate(float new_lr) override {
        adam_config_.learning_rate = new_lr;
    }
    
    float get_learning_rate() const override {
        return adam_config_.learning_rate;
    }

protected:
    void update_param(TensorPtr param,
                     const TensorPtr& grad,
                     OptimizerState& state) override;

private:
    AdamConfig adam_config_;
    std::unordered_map<TensorPtr, AdamState> states_;

    void init_state(const TensorPtr& param);
    float compute_bias_correction1(size_t step) const;
    float compute_bias_correction2(size_t step) const;
};

}
