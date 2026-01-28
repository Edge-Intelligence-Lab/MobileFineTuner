/**
 * @file adam_amp.h
 * @brief Adam optimizer with Automatic Mixed Precision (AMP) support
 * 
 * Implements Adam optimizer with FP32 master weights for stable FP16 training.
 * Follows PyTorch AMP best practices.
 */

#pragma once

#include "optimizer.h"
#include "../utils/grad_scaler.h"
#include <unordered_map>
#include <memory>

namespace ops {

/**
 * @brief Configuration for AdamAMP optimizer
 */
struct AdamAMPConfig : public OptimizerConfig {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    bool use_fp16_params = true;      // Store parameters in FP16
    bool use_fp32_master = true;      // Keep FP32 master weights
    bool enable_grad_scaling = true;  // Enable automatic grad scaling
    GradScalerConfig scaler_config;   // Grad scaler configuration
    
    AdamAMPConfig() : OptimizerConfig() {}
    
    AdamAMPConfig(float lr, float b1 = 0.9f, float b2 = 0.999f, 
                  float eps = 1e-8f, float wd = 0.0f)
        : OptimizerConfig(lr, wd, 1.0f, false),
          beta1(b1), beta2(b2), epsilon(eps) {}
};

/**
 * @brief Per-parameter optimizer state for AdamAMP
 */
struct AdamAMPState : public OptimizerState {
    size_t step = 0;
    TensorPtr master_weight;  // FP32 master weight (if enabled)
    TensorPtr m;              // First moment (FP32)
    TensorPtr v;              // Second moment (FP32)
    TensorPtr v_hat;          // Max second moment for AMSGrad (FP32)
    
    void to_file(const std::string& path) const override;
    void from_file(const std::string& path) override;
};

/**
 * @brief Adam optimizer with mixed precision training support
 * 
 * Key features:
 * - FP32 master weights for numerical stability
 * - FP16 parameter storage for memory efficiency
 * - Integrated gradient scaling
 * - Compatible with existing training loops
 * 
 * Usage:
 *   AdamAMP optimizer(config);
 *   GradScaler scaler(config.scaler_config);
 *   
 *   auto loss = model.forward(input);
 *   auto scaled_loss = scaler.scale(loss);
 *   scaled_loss->backward();
 *   
 *   if (scaler.unscale(parameters)) {
 *       optimizer.step(parameters);
 *   }
 *   scaler.update();
 */
class AdamAMP : public Optimizer {
public:
    explicit AdamAMP(const AdamAMPConfig& config);
    ~AdamAMP() override = default;
    
    /**
     * @brief Perfor optimization step (implementsation of base class method)
     */
    void step(const std::vector<TensorPtr>& parameters,
             const std::vector<TensorPtr>& gradients) override;
    
    /**
     * @brief Perfor optimization step using gradients from parameters
     */
    void step_with_params(const std::vector<TensorPtr>& parameters);
    
    /**
     * @brief Zero out all gradients
     */
    void zero_grad(const std::vector<TensorPtr>& parameters) override;
    
    /**
     * @brief Save optimizer state to file
     */
    void save_state(const std::string& path) const override;
    
    /**
     * @brief Load optimizer state from file
     */
    void load_state(const std::string& path) override;
    
    /**
     * @brief Get FP32 master weight for a parameter
     */
    TensorPtr get_master_weight(const TensorPtr& param) const;
    
    /**
     * @brief Sync FP16 parameters from FP32 master weights
     * Call this before inference or evaluation
     */
    void sync_fp16_from_master();
    
    /**
     * @brief Get current configuration
     */
    const AdamAMPConfig& get_config() const { return amp_config_; }
    
    /**
     * @brief Set learning rate (override to sync both config_ and amp_config_)
     */
    void set_learning_rate(float lr) override {
        config_.learning_rate = lr;
        amp_config_.learning_rate = lr;
    }
    
    /**
     * @brief Get learning rate
     */
    float get_learning_rate() const override {
        return amp_config_.learning_rate;
    }

protected:
    void update_param(TensorPtr param,
                     const TensorPtr& grad,
                     OptimizerState& state) override;

private:
    AdamAMPConfig amp_config_;
    std::unordered_map<TensorPtr, AdamAMPState> states_;
    
    void init_state(const TensorPtr& param);
    float compute_bias_correction1(size_t step) const;
    float compute_bias_correction2(size_t step) const;
    
    // Cast parameter to FP16 if needed
    TensorPtr ensure_fp16(const TensorPtr& param);
    
    // Cast parameter to FP32 if needed
    TensorPtr ensure_fp32(const TensorPtr& param);
};

} // namespace ops
