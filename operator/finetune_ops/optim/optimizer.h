/**
 * @file optimizer.h
 * @brief Base optimizer class and configuration
 * 
 * This file defines the base optimizer interface and configuration
 * structures used by all optimization algorithms in the framework.
 * It providess a unified interface for parameter updates during training.
 */

#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>
#include <string>

namespace ops {

/**
 * @brief Configuration structure for optimizers
 * 
 * Contains common hyperparameters used by most optimization algorithms
 * including learning rate, weight decay, and gradient clipping.
 */
struct OptimizerConfig {
    float learning_rate;    /**< Learning rate for parameter updates */
    float weight_decay;     /**< L2 regularization coefficient */
    float clip_grad_norm;   /**< Gradient clipping threshold */
    bool amsgrad;           /**< Whether to use AMSGrad variant (for Adam) */

    /**
     * @brief Default optimizer configuration
     * 
     * Creates a configuration with commonly used default values:
     * - learning_rate: 0.001
     * - weight_decay: 0.0
     * - clip_grad_norm: 1.0
     * - amsgrad: false
     */
    OptimizerConfig() : learning_rate(0.001f),
        weight_decay(0.0f), clip_grad_norm(1.0f),
        amsgrad(false) {}

    /**
     * @brief Construct optimizer configuration with custom parameters
     * @param learning_rate_ Learning rate for parameter updates
     * @param weight_decay_ L2 regularization coefficient (default: 0.0)
     * @param clip_grad_norm_ Gradient clipping threshold (default: 1.0)
     * @param amsgrad_ Whether to use AMSGrad variant (default: false)
     */
    OptimizerConfig(
        float learning_rate_,
        float weight_decay_ = 0.0f,
        float clip_grad_norm_ = 1.0f,
        bool amsgrad_ = false
    ) : learning_rate(learning_rate_),
        weight_decay(weight_decay_),
        clip_grad_norm(clip_grad_norm_),
        amsgrad(amsgrad_) {}

    virtual ~OptimizerConfig() = default;
};

struct OptimizerState {
    virtual ~OptimizerState() = default;
    virtual void to_file(const std::string& path) const = 0;
    virtual void from_file(const std::string& path) = 0;
};

class Optimizer {
public:
    explicit Optimizer(const OptimizerConfig& config)
        : config_(config) {}
    virtual ~Optimizer() = default;

    virtual void step(const std::vector<TensorPtr>& parameters,
                     const std::vector<TensorPtr>& gradients) = 0;

    virtual void zero_grad(const std::vector<TensorPtr>& parameters) {
        for (auto& param : parameters) {
            param->zero_grad();
        }
    }

    virtual void clip_grad_norm(const std::vector<TensorPtr>& parameters,
                              float max_norm);

    virtual void save_state(const std::string& path) const = 0;
    virtual void load_state(const std::string& path) = 0;

    virtual float get_learning_rate() const { return config_.learning_rate; }
    virtual void set_learning_rate(float lr) { config_.learning_rate = lr; }

    const OptimizerConfig& get_config() const { return config_; }

protected:
    OptimizerConfig config_;

    virtual void update_param(TensorPtr param,
                            const TensorPtr& grad,
                            OptimizerState& state) = 0;

    virtual float compute_grad_norm(const std::vector<TensorPtr>& parameters);
    virtual void scale_gradients(const std::vector<TensorPtr>& parameters,
                               float scale);
};

}
