/**
 * @file adam_amp.cpp
 * @brief Implementation of Adam optimizer with AMP support
 */

#include "adam_amp.h"
#include "../core/ops.h"
#include <cmath>
#include <fstream>
#include <iostream>

namespace ops {

AdamAMP::AdamAMP(const AdamAMPConfig& config)
    : Optimizer(config), amp_config_(config) {}

void AdamAMP::step_with_params(const std::vector<TensorPtr>& parameters) {
    for (auto& param : parameters) {
        if (!param->grad()) continue;
        
        if (states_.find(param) == states_.end()) {
            init_state(param);
        }
        
        auto& state = states_[param];
        state.step++;
        
        // Work with FP32 master weight
        TensorPtr working_param = state.master_weight ? state.master_weight : param;
        auto grad = param->grad();
        
        // Convert gradient to FP32 if needed
        if (grad->dtype() == DType::kFloat16) {
            grad = cast(grad, DType::kFloat32);
        }
        
        const float* grad_data = grad->data<float>();
        float* param_data = working_param->data<float>();
        float* m_data = state.m->data<float>();
        float* v_data = state.v->data<float>();
        
        float bias_correction1 = compute_bias_correction1(state.step);
        float bias_correction2 = compute_bias_correction2(state.step);
        
        // Apply weight decay and clip gradients
        float grad_norm = 0.0f;
        for (int64_t j = 0; j < grad->numel(); ++j) {
            grad_norm += grad_data[j] * grad_data[j];
        }
        grad_norm = std::sqrt(grad_norm);
        
        float clip_coef = 1.0f;
        if (amp_config_.clip_grad_norm > 0.0f && grad_norm > amp_config_.clip_grad_norm) {
            clip_coef = amp_config_.clip_grad_norm / (grad_norm + 1e-6f);
        }
        
        // Adam update
        for (int64_t j = 0; j < param->numel(); ++j) {
            float grad_val = grad_data[j] * clip_coef;
            
            if (amp_config_.weight_decay > 0.0f) {
                grad_val += amp_config_.weight_decay * param_data[j];
            }
            
            m_data[j] = amp_config_.beta1 * m_data[j] + (1.0f - amp_config_.beta1) * grad_val;
            v_data[j] = amp_config_.beta2 * v_data[j] + (1.0f - amp_config_.beta2) * grad_val * grad_val;
            
            float m_corrected = m_data[j] / bias_correction1;
            float v_corrected = v_data[j] / bias_correction2;
            
            param_data[j] -= amp_config_.learning_rate * m_corrected / (std::sqrt(v_corrected) + amp_config_.epsilon);
        }
        
        // Sync back to FP16 parameter if using master weights
        if (state.master_weight && param->dtype() == DType::kFloat16) {
            auto fp16_param = cast(state.master_weight, DType::kFloat16);
            std::memcpy(param->data<float>(), fp16_param->data<float>(), 
                       param->numel() * DTypeUtils::size_of(param->dtype()));
        }
    }
}

void AdamAMP::step(const std::vector<TensorPtr>& parameters,
                   const std::vector<TensorPtr>& gradients) {
    if (parameters.size() != gradients.size()) {
        throw std::runtime_error("AdamAMP: Parameters and gradients size mismatch");
    }
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        auto& param = parameters[i];
        auto& grad = gradients[i];
        if (!grad) continue;
        
        // Temporarily set gradient for step_with_params()
        param->set_grad(grad);
    }
    
    step_with_params(parameters);
}

void AdamAMP::zero_grad(const std::vector<TensorPtr>& parameters) {
    for (auto& param : parameters) {
        param->set_grad(nullptr);
    }
}

void AdamAMP::save_state(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving AdamAMP state: " + path);
    }
    
    // Save config
    file.write(reinterpret_cast<const char*>(&amp_config_.learning_rate), sizeof(float));
    file.write(reinterpret_cast<const char*>(&amp_config_.beta1), sizeof(float));
    file.write(reinterpret_cast<const char*>(&amp_config_.beta2), sizeof(float));
    file.write(reinterpret_cast<const char*>(&amp_config_.epsilon), sizeof(float));
    
    size_t num_params = states_.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(size_t));
    
    for (const auto& [param, state] : states_) {
        file.write(reinterpret_cast<const char*>(&state.step), sizeof(size_t));
        
        // Save master weight if exists
        bool has_master = (state.master_weight != nullptr);
        file.write(reinterpret_cast<const char*>(&has_master), sizeof(bool));
        if (has_master) {
            int64_t size = state.master_weight->numel();
            file.write(reinterpret_cast<const char*>(&size), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(state.master_weight->data<float>()), size * sizeof(float));
        }
        
        // Save m and v
        int64_t m_size = state.m->numel();
        file.write(reinterpret_cast<const char*>(&m_size), sizeof(int64_t));
        file.write(reinterpret_cast<const char*>(state.m->data<float>()), m_size * sizeof(float));
        
        int64_t v_size = state.v->numel();
        file.write(reinterpret_cast<const char*>(&v_size), sizeof(int64_t));
        file.write(reinterpret_cast<const char*>(state.v->data<float>()), v_size * sizeof(float));
    }
    
    file.close();
}

void AdamAMP::load_state(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading AdamAMP state: " + path);
    }
    
    file.read(reinterpret_cast<char*>(&amp_config_.learning_rate), sizeof(float));
    file.read(reinterpret_cast<char*>(&amp_config_.beta1), sizeof(float));
    file.read(reinterpret_cast<char*>(&amp_config_.beta2), sizeof(float));
    file.read(reinterpret_cast<char*>(&amp_config_.epsilon), sizeof(float));
    
    file.close();
}

TensorPtr AdamAMP::get_master_weight(const TensorPtr& param) const {
    auto it = states_.find(param);
    if (it != states_.end() && it->second.master_weight) {
        return it->second.master_weight;
    }
    return param;
}

void AdamAMP::sync_fp16_from_master() {
    for (auto& [param, state] : states_) {
        if (state.master_weight && param->dtype() == DType::kFloat16) {
            auto fp16_param = cast(state.master_weight, DType::kFloat16);
            std::memcpy(param->data<float>(), fp16_param->data<float>(),
                       param->numel() * DTypeUtils::size_of(param->dtype()));
        }
    }
}

void AdamAMP::update_param(TensorPtr param, const TensorPtr& grad, OptimizerState& state) {
    // This is handled in step() for AdamAMP
}

void AdamAMP::init_state(const TensorPtr& param) {
    AdamAMPState state;
    state.step = 0;
    
    // Create FP32 master weight if parameter is FP16
    if (amp_config_.use_fp32_master && param->dtype() == DType::kFloat16) {
        state.master_weight = cast(param, DType::kFloat32);
    }
    
    // Initialize m and v in FP32
    state.m = zeros(param->shape(), DType::kFloat32, param->device());
    state.v = zeros(param->shape(), DType::kFloat32, param->device());
    
    if (amp_config_.amsgrad) {
        state.v_hat = zeros(param->shape(), DType::kFloat32, param->device());
    }
    
    states_[param] = std::move(state);
}

float AdamAMP::compute_bias_correction1(size_t step) const {
    return 1.0f - std::pow(amp_config_.beta1, static_cast<float>(step));
}

float AdamAMP::compute_bias_correction2(size_t step) const {
    return 1.0f - std::pow(amp_config_.beta2, static_cast<float>(step));
}

TensorPtr AdamAMP::ensure_fp16(const TensorPtr& param) {
    if (param->dtype() == DType::kFloat16) return param;
    return cast(param, DType::kFloat16);
}

TensorPtr AdamAMP::ensure_fp32(const TensorPtr& param) {
    if (param->dtype() == DType::kFloat32) return param;
    return cast(param, DType::kFloat32);
}

void AdamAMPState::to_file(const std::string& path) const {
    // Implementation similar to to save_state
}

void AdamAMPState::from_file(const std::string& path) {
    // Implementation similar to to load_state
}

} // namespace ops
