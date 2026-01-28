/**
 * @file adam.cpp
 * @brief Implementation of Adam optimizer
 * 
 * This file providess the implementsation of the Adam optimizer algorithm
 * with support for adaptive learning rates, momentum, and optional AMSGrad variant.
 */

#include "adam.h"
#include "../memory/arena_allocator.h"
#include "../core/logger.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace ops {

Adam::Adam(const AdamConfig& config) 
    : Optimizer(config), adam_config_(config) {
}

// [Translated comment removed - see documentation]

void Adam::step(const std::vector<TensorPtr>& parameters,
               const std::vector<TensorPtr>& gradients) {
    if (parameters.size() != gradients.size()) {
        throw std::runtime_error("Parameters and gradients size mismatch");
    }
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        auto& param = parameters[i];
        auto& grad = gradients[i];
        
        if (!grad) continue;  // Skip if no gradient
        
        // Initialize state if needed
        if (states_.find(param) == states_.end()) {
            init_state(param);
        }
        
        auto& state = states_[param];
        state.step++;
        
        // Get current parameter and gradient data
        const float* grad_data = grad->data<float>();
        float* param_data = param->data<float>();
        float* m_data = state.m[0]->data<float>();
        float* v_data = state.v[0]->data<float>();
        float* v_hat_data = nullptr;
        
        if (adam_config_.amsgrad && !state.v_hat.empty()) {
            v_hat_data = state.v_hat[0]->data<float>();
        }
        
        // Compute bias correction factors
        float bias_correction1 = compute_bias_correction1(state.step);
        float bias_correction2 = compute_bias_correction2(state.step);
        
        // Update parameters
        for (int64_t j = 0; j < param->numel(); ++j) {
            float grad_val = grad_data[j];
            
            // Apply weight decay if specified
            if (adam_config_.weight_decay > 0.0f) {
                grad_val += adam_config_.weight_decay * param_data[j];
            }
            
            // Update biased first moment estimate
            m_data[j] = adam_config_.beta1 * m_data[j] + (1.0f - adam_config_.beta1) * grad_val;
            
            // Update biased second raw moment estimate
            v_data[j] = adam_config_.beta2 * v_data[j] + (1.0f - adam_config_.beta2) * grad_val * grad_val;
            
            float v_corrected = v_data[j] / bias_correction2;
            
            // AMSGrad variant
            if (adam_config_.amsgrad && v_hat_data) {
                v_hat_data[j] = std::max(v_hat_data[j], v_corrected);
                v_corrected = v_hat_data[j];
            }
            
            // Compute bias-corrected first moment estimate
            float m_corrected = m_data[j] / bias_correction1;
            
            // Update parameters
            param_data[j] -= adam_config_.learning_rate * m_corrected / (std::sqrt(v_corrected) + adam_config_.epsilon);
        }
    }
}

void Adam::save_state(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving Adam state: " + path);
    }
    
    // Save configuration
    file.write(reinterpret_cast<const char*>(&adam_config_.learning_rate), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.beta1), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.beta2), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.epsilon), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.weight_decay), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.amsgrad), sizeof(bool));
    
    // Save number of parameters
    size_t num_params = states_.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(size_t));
    
    // Save states for each parameter
    for (const auto& [param, state] : states_) {
        file.write(reinterpret_cast<const char*>(&state.step), sizeof(size_t));
        
        // Save first moment
        if (!state.m.empty()) {
            auto& m_tensor = state.m[0];
            int64_t m_size = m_tensor->numel();
            file.write(reinterpret_cast<const char*>(&m_size), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(m_tensor->data<float>()), m_size * sizeof(float));
        }
        
        // Save second moment
        if (!state.v.empty()) {
            auto& v_tensor = state.v[0];
            int64_t v_size = v_tensor->numel();
            file.write(reinterpret_cast<const char*>(&v_size), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(v_tensor->data<float>()), v_size * sizeof(float));
        }
        
        // Save v_hat for AMSGrad
        if (adam_config_.amsgrad && !state.v_hat.empty()) {
            auto& v_hat_tensor = state.v_hat[0];
            int64_t v_hat_size = v_hat_tensor->numel();
            file.write(reinterpret_cast<const char*>(&v_hat_size), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(v_hat_tensor->data<float>()), v_hat_size * sizeof(float));
        }
    }
    
    file.close();
}

void Adam::load_state(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading Adam state: " + path);
    }
    
    // Load configuration
    file.read(reinterpret_cast<char*>(&adam_config_.learning_rate), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.beta1), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.beta2), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.epsilon), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.weight_decay), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.amsgrad), sizeof(bool));
    
    // Load number of parameters
    size_t num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(size_t));
    
    // Note: This is a simplified implementsation
    // In a complete implementsation, we would need to match parameters by name or index
    
    file.close();
}

void Adam::update_param(TensorPtr /* param */, const TensorPtr& /* grad */, OptimizerState& /* state */) {
    // This method is called by the base class
    // The main logic is in the step() method above
}

void Adam::init_state(const TensorPtr& param) {
    AdamState state;
    state.step = 0;
    
    // 🔧 Key fix: use StaticWeightArena for optimizer state (long-lived, not step-level arena)
    #ifdef USE_ARENA_ALLOCATOR
    size_t state_bytes = param->numel() * sizeof(float);
    auto& static_arena = memory::ArenaManager::instance().static_weights();
    
    // Allocate m buffer (momentum)
    float* m_buffer = static_cast<float*>(static_arena.allocate_static(state_bytes, "adam_m"));
    std::memset(m_buffer, 0, state_bytes);
    auto m = std::make_shared<Tensor>(param->shape(), m_buffer, kFloat32, param->device(), true);
    state.m.push_back(m);
    
    // Allocate v buffer (variance)
    float* v_buffer = static_cast<float*>(static_arena.allocate_static(state_bytes, "adam_v"));
    std::memset(v_buffer, 0, state_bytes);
    auto v = std::make_shared<Tensor>(param->shape(), v_buffer, kFloat32, param->device(), true);
    state.v.push_back(v);
    
    // Allocate v_hat for AMSGrad if needed
    if (adam_config_.amsgrad) {
        float* v_hat_buffer = static_cast<float*>(static_arena.allocate_static(state_bytes, "adam_v_hat"));
        std::memset(v_hat_buffer, 0, state_bytes);
        auto v_hat = std::make_shared<Tensor>(param->shape(), v_hat_buffer, kFloat32, param->device(), true);
        state.v_hat.push_back(v_hat);
    }
    #else
    // Fallback: traditional allocation (may enter step-level arena; riskier)
    auto m = std::make_shared<Tensor>(param->shape(), kFloat32, param->device());
    std::memset(m->data<float>(), 0, m->numel() * sizeof(float));
    state.m.push_back(m);
    
    auto v = std::make_shared<Tensor>(param->shape(), kFloat32, param->device());
    std::memset(v->data<float>(), 0, v->numel() * sizeof(float));
    state.v.push_back(v);
    
    if (adam_config_.amsgrad) {
        auto v_hat = std::make_shared<Tensor>(param->shape(), kFloat32, param->device());
        std::memset(v_hat->data<float>(), 0, v_hat->numel() * sizeof(float));
        state.v_hat.push_back(v_hat);
    }
    #endif
    
    states_[param] = std::move(state);
    
    // Silent init (avoid log noise)
    // OPS_LOG_DEBUG_F("Adam state initialized: %zu bytes (m+v)", state_bytes * 2);
}

float Adam::compute_bias_correction1(size_t step) const {
    return 1.0f - std::pow(adam_config_.beta1, static_cast<float>(step));
}

float Adam::compute_bias_correction2(size_t step) const {
    return 1.0f - std::pow(adam_config_.beta2, static_cast<float>(step));
}

void AdamState::to_file(const std::string& path) const {
        // [Translated]
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&step), sizeof(step));
        file.close();
    }
}

void AdamState::from_file(const std::string& path) {
        // [Translated]
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&step), sizeof(step));
        file.close();
    }
}

} // namespace ops
