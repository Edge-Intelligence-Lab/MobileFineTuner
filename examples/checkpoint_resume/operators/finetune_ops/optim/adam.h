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
#include "../../third_party/nlohmann/json.hpp"
#include <map>
#include <string>
#include <utility>
#include <unordered_map>

namespace ops {

using NamedParameters = std::vector<std::pair<std::string, TensorPtr>>;

struct AdamSnapshot {
    nlohmann::json metadata;
    std::map<std::string, TensorPtr> tensors;
};

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
    bool decoupled_weight_decay;  /**< AdamW-style decoupled weight decay */

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
        beta1(0.9f), beta2(0.999f), epsilon(1e-8f),
        decoupled_weight_decay(false) {}

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
        bool amsgrad_ = false,
        bool decoupled_weight_decay_ = false
    ) : OptimizerConfig(learning_rate_, weight_decay_,
                       clip_grad_norm_, amsgrad_),
        beta1(beta1_), beta2(beta2_), epsilon(epsilon_),
        decoupled_weight_decay(decoupled_weight_decay_) {}
};

struct AdamWConfig : public AdamConfig {
    AdamWConfig() : AdamConfig() {
        decoupled_weight_decay = true;
    }

    AdamWConfig(
        float learning_rate_,
        float beta1_ = 0.9f,
        float beta2_ = 0.999f,
        float epsilon_ = 1e-8f,
        float weight_decay_ = 0.0f,
        float clip_grad_norm_ = 1.0f,
        bool amsgrad_ = false
    ) : AdamConfig(learning_rate_, beta1_, beta2_, epsilon_, weight_decay_,
                   clip_grad_norm_, amsgrad_, true) {}
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

    // Opt-in CPU FP32 resume. Export borrows tensors: serialize at a quiescent
    // update boundary before stepping again. Legacy save/load is not exact resume.
    AdamSnapshot export_named_state(const NamedParameters& parameters) const;
    // Names, not vector positions, bind state to parameters. The caller preserves
    // its saved training order separately. Imported storage must outlive training
    // (not a step arena) and must not subsequently be modified by the caller.
    void import_named_state(const NamedParameters& parameters, const AdamSnapshot& snapshot);
    
    void set_learning_rate(float new_lr) override {
        adam_config_.learning_rate = new_lr;
    }
    
    float get_learning_rate() const override {
        return adam_config_.learning_rate;
    }

    bool decoupled_weight_decay() const {
        return adam_config_.decoupled_weight_decay;
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

class AdamW : public Adam {
public:
    explicit AdamW(const AdamWConfig& config) : Adam(config) {}
    ~AdamW() override = default;
};

}
