/**
 * @file grad_scaler.h
 * @brief Automatic Mixed Precision (AMP) Gradient Scaler
 * 
 * Implements dynamic loss scaling similar to to torch.cuda.amp.GradScaler
 * to prevent gradient underflow in FP16 training.
 */

#pragma once

#include "../core/tensor.h"
#include "../core/ops.h"
#include <memory>
#include <vector>
#include <cmath>

namespace ops {

/**
 * @brief Configuration for gradient scaling
 */
struct GradScalerConfig {
    float init_scale = 65536.0f;        // Initial loss scale (2^16)
    float growth_factor = 2.0f;         // Scale growth rate
    float backoff_factor = 0.5f;        // Scale reduction rate on overflow
    int growth_interval = 2000;         // Steps between scale increases
    bool enabled = true;                // Enable/disable scaling
    
    GradScalerConfig() = default;
};

/**
 * @brief Gradient Scaler for Automatic Mixed Precision Training
 * 
 * Usage (similar to to PyTorch):
 *   GradScaler scaler;
 *   auto loss = model.forward(input);
 *   auto scaled_loss = scaler.scale(loss);
 *   scaled_loss->backward();
 *   scaler.unscale(optimizer);  // Unscale gradients
 *   scaler.step(optimizer);     // Update parameters if no overflow
 *   scaler.update();            // Update scale for next iteration
 */
class GradScaler {
public:
    explicit GradScaler(const GradScalerConfig& config = GradScalerConfig())
        : config_(config), scale_(config.init_scale), 
          growth_tracker_(0), found_inf_(false) {}
    
    /**
     * @brief Scale loss before backward pass
     * @param loss Unscaled loss tensor
     * @return Scaled loss tensor (loss * scale)
     */
    TensorPtr scale(const TensorPtr& loss) {
        if (!config_.enabled) return loss;
        
        // Scale loss by scalar (simpler, avoid full() call)
        return ops::mul(loss, scale_);
    }
    
    /**
     * @brief Unscale gradients after backward pass
     * @param parameters List of parameters with gradients
     * @return true if no inf/nan found in gradients
     */
    bool unscale(const std::vector<TensorPtr>& parameters) {
        if (!config_.enabled) return true;
        
        found_inf_ = false;
        float inv_scale = 1.0f / scale_;
        
        for (auto& param : parameters) {
            if (!param->grad()) continue;
            
            auto grad = param->grad();
            float* grad_data = grad->data<float>();
            size_t numel = static_cast<size_t>(grad->numel());
            
            // Check for inf/nan and unscale simultaneously
            for (size_t i = 0; i < numel; ++i) {
                float val = grad_data[i];
                if (std::isinf(val) || std::isnan(val)) {
                    found_inf_ = true;
                    return false;
                }
                grad_data[i] = val * inv_scale;
            }
        }
        
        return true;
    }
    
    /**
     * @brief Update parameters if no overflow detected
     * @param optimizer Optimizer to step
     * @param parameters List of parameters
     * @return true if step was perfored
     */
    template<typename Optimizer>
    bool step(Optimizer& optimizer, const std::vector<TensorPtr>& parameters) {
        if (!config_.enabled) {
            optimizer.step();
            return true;
        }
        
        if (!found_inf_) {
            optimizer.step();
            return true;
        }
        
        // Skip optimizer step if overflow detected
        return false;
    }
    
    /**
     * @brief Update scale factor after each iteration
     * Call this at the end of each training step
     */
    void update() {
        if (!config_.enabled) return;
        
        if (found_inf_) {
            // Reduce scale on overflow
            scale_ *= config_.backoff_factor;
            growth_tracker_ = 0;
        } else {
            // Increase scale periodically if no overflow
            growth_tracker_++;
            if (growth_tracker_ >= config_.growth_interval) {
                scale_ *= config_.growth_factor;
                growth_tracker_ = 0;
            }
        }
        
        // Clamp scale to reasonable range
        scale_ = std::max(1.0f, std::min(scale_, 65536.0f));
    }
    
    /**
     * @brief Get current scale factor
     */
    float get_scale() const { return scale_; }
    
    /**
     * @brief Check if overflow was detected in last unscale
     */
    bool found_inf() const { return found_inf_; }
    
    /**
     * @brief Enable or disable gradient scaling
     */
    void set_enabled(bool enabled) { config_.enabled = enabled; }
    
    /**
     * @brief Reset scaler to initial state
     */
    void reset() {
        scale_ = config_.init_scale;
        growth_tracker_ = 0;
        found_inf_ = false;
    }

private:
    GradScalerConfig config_;
    float scale_;           // Current scale factor
    int growth_tracker_;    // Steps since last scale increase
    bool found_inf_;        // Whether inf/nan was found in last unscale
};

} // namespace ops
