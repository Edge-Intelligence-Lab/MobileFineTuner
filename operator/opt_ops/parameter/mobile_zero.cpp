/**
 * @file mobile_zero.cpp
 * @brief Mobile-optimized ZeRO implementsation for single-device training
 * 
 * Industrial-grade implementsation of ZeRO optimizer adapted for mobile platfors
 */

#include "mobile_zero.h"
#include "../core/logger.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace ops {
namespace memory {

// ============================================================================
// GradientAccumulator Implementation
// ============================================================================

GradientAccumulator::GradientAccumulator(int accumulation_steps)
    : target_accumulation_steps_(accumulation_steps), 
      gradient_scale_(1.0f / accumulation_steps) {
}

void GradientAccumulator::accumulate_gradient(size_t param_id, const TensorPtr& grad) {
    if (!grad) return;
    
    if (accumulated_grads_.find(param_id) == accumulated_grads_.end()) {
        // First accumulation: clone the gradient
        accumulated_grads_[param_id] = grad->clone();
        accumulation_counts_[param_id] = 1;
    } else {
        // Subsequent accumulations: add to existing
        auto& accumulated = accumulated_grads_[param_id];
        auto grad_data = grad->data<float>();
        auto accum_data = accumulated->data<float>();
        size_t numel = grad->numel();
        
        for (size_t i = 0; i < numel; ++i) {
            accum_data[i] += grad_data[i];
        }
        accumulation_counts_[param_id]++;
    }
}

TensorPtr GradientAccumulator::get_accumulated_gradient(size_t param_id) {
    auto it = accumulated_grads_.find(param_id);
    if (it == accumulated_grads_.end()) {
        return nullptr;
    }
    
    // Apply gradient scaling
    auto& grad = it->second;
    auto grad_data = grad->data<float>();
    size_t numel = grad->numel();
    
    for (size_t i = 0; i < numel; ++i) {
        grad_data[i] *= gradient_scale_;
    }
    
    return grad;
}

bool GradientAccumulator::is_ready_for_update(size_t param_id) const {
    auto it = accumulation_counts_.find(param_id);
    if (it == accumulation_counts_.end()) {
        return false;
    }
    return it->second >= target_accumulation_steps_;
}

void GradientAccumulator::clear_accumulated_gradient(size_t param_id) {
    accumulated_grads_.erase(param_id);
    accumulation_counts_.erase(param_id);
}

void GradientAccumulator::reset() {
    accumulated_grads_.clear();
    accumulation_counts_.clear();
}

// ============================================================================
// MobileZeROOptimizer Implementation
// ============================================================================

MobileZeROOptimizer::MobileZeROOptimizer(
    std::unique_ptr<MobileParameterManager> param_manager,
    MobileZeROStage stage,
    float learning_rate,
    float weight_decay,
    int grad_accumulation_steps)
    : param_manager_(std::move(param_manager)),
      zero_stage_(stage),
      base_learning_rate_(learning_rate),
      weight_decay_(weight_decay),
      beta1_(0.9f),
      beta2_(0.999f),
      epsilon_(1e-8f),
      gradient_accumulation_steps_(grad_accumulation_steps),
      current_step_(0),
      max_active_parameters_(10000),
      current_memory_usage_(0) {
    
    grad_accumulator_ = std::make_unique<GradientAccumulator>(grad_accumulation_steps);
    
    // Initialize statistics
    stats_.total_parameters = 0;
    stats_.active_parameters = 0;
    stats_.memory_saved_bytes = 0;
    stats_.average_load_time_ms = 0.0;
    stats_.average_compute_time_ms = 0.0;
    stats_.parameter_swaps = 0;
    
    OPS_LOG_INFO_F("MobileZeRO initialized: stage=%d, lr=%.2e, weight_decay=%.2e",
                   static_cast<int>(stage), learning_rate, weight_decay);
}

size_t MobileZeROOptimizer::register_parameter(
    const std::string& name,
    const TensorPtr& tensor,
    const std::string& group_name,
    bool requires_grad) {
    
    // Register with parameter manager
    size_t param_id = param_manager_->register_parameter(name, tensor, requires_grad);
    
    // Create or find parameter group
    size_t group_id;
    auto it = name_to_group_map_.find(group_name);
    if (it == name_to_group_map_.end()) {
        // Create new group
        auto group = std::make_unique<ParameterGroup>(group_name);
        group_id = parameter_groups_.size();
        name_to_group_map_[group_name] = group_id;
        parameter_groups_.push_back(std::move(group));
    } else {
        group_id = it->second;
    }
    
    // Add parameter to group
    auto& group = parameter_groups_[group_id];
    group->param_ids.push_back(param_id);
    group->param_names.push_back(name);
    group->total_size_bytes += tensor->numel() * sizeof(float);
    group->requires_grad = requires_grad;
    
    param_to_group_map_[param_id] = group_id;
    
    stats_.total_parameters++;
    
    return param_id;
}

TensorPtr MobileZeROOptimizer::get_parameter(size_t param_id) {
    // Ensure parameter is loaded
    auto group_id_it = param_to_group_map_.find(param_id);
    if (group_id_it != param_to_group_map_.end()) {
        auto& group = parameter_groups_[group_id_it->second];
        if (!group->is_active) {
            load_parameter_group(group->group_name);
        }
    }
    
    return param_manager_->get_parameter(param_id);
}

TensorPtr MobileZeROOptimizer::get_parameter(const std::string& name) {
    // Find parameter by name through parameter manager
    return param_manager_->get_parameter(name);
}

void MobileZeROOptimizer::backward(size_t param_id, const TensorPtr& gradient) {
    grad_accumulator_->accumulate_gradient(param_id, gradient);
}

void MobileZeROOptimizer::step(const std::vector<size_t>& param_ids) {
    current_step_++;
    
    std::vector<size_t> params_to_update = param_ids;
    if (params_to_update.empty()) {
        // Update all parameters
        for (const auto& group : parameter_groups_) {
            if (group->requires_grad) {
                params_to_update.insert(params_to_update.end(),
                                       group->param_ids.begin(),
                                       group->param_ids.end());
            }
        }
    }
    
    // Bias correction for Adam
    float bias_correction1 = 1.0f - std::pow(beta1_, current_step_);
    float bias_correction2 = 1.0f - std::pow(beta2_, current_step_);
    
    for (auto param_id : params_to_update) {
        if (!grad_accumulator_->is_ready_for_update(param_id)) {
            continue;
        }
        
        auto grad = grad_accumulator_->get_accumulated_gradient(param_id);
        if (!grad) continue;
        
        auto param = get_parameter(param_id);
        if (!param) continue;
        
        // Initialize optimizer states if needed
        if (optimizer_states_m_.find(param_id) == optimizer_states_m_.end()) {
            optimizer_states_m_[param_id] = zeros(param->shape());
            optimizer_states_v_[param_id] = zeros(param->shape());
        }
        
        auto& m = optimizer_states_m_[param_id];
        auto& v = optimizer_states_v_[param_id];
        
        auto param_data = param->data<float>();
        auto grad_data = grad->data<float>();
        auto m_data = m->data<float>();
        auto v_data = v->data<float>();
        
        size_t numel = param->numel();
        
        // Get learning rate (per-parameter or global)
        float lr = base_learning_rate_;
        auto lr_it = learning_rates_.find(param_id);
        if (lr_it != learning_rates_.end()) {
            lr = lr_it->second;
        }
        
        // Adam update with weight decay (AdamW style)
        for (size_t i = 0; i < numel; ++i) {
            float g = grad_data[i];
            
            // Weight decay
            if (weight_decay_ > 0.0f) {
                g += weight_decay_ * param_data[i];
            }
            
            // Update biased first moment estimate
            m_data[i] = beta1_ * m_data[i] + (1.0f - beta1_) * g;
            
            // Update biased second raw moment estimate  
            v_data[i] = beta2_ * v_data[i] + (1.0f - beta2_) * g * g;
            
            // Compute bias-corrected moments
            float m_hat = m_data[i] / bias_correction1;
            float v_hat = v_data[i] / bias_correction2;
            
            // Update parameters
            param_data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
        
        // Clear accumulated gradient
        grad_accumulator_->clear_accumulated_gradient(param_id);
    }
}

void MobileZeROOptimizer::zero_grad(const std::vector<size_t>& param_ids) {
    if (param_ids.empty()) {
        grad_accumulator_->reset();
    } else {
        for (auto param_id : param_ids) {
            grad_accumulator_->clear_accumulated_gradient(param_id);
        }
    }
}

void MobileZeROOptimizer::load_parameter_group(const std::string& group_name) {
    auto it = name_to_group_map_.find(group_name);
    if (it == name_to_group_map_.end()) {
        // Warning log using standard macro
        if (LogManager::get_logger()) {
            LogManager::get_logger()->warning("Parameter group not found: " + group_name);
        }
        return;
    }
    
    auto& group = parameter_groups_[it->second];
    if (group->is_active) {
        return; // Already loaded
    }
    
    // Prefetch all parameters in the group to CPU memory
    param_manager_->prefetch_parameters(group->param_ids, MemoryTier::CPU_MEMORY);
    
    group->is_active = true;
    stats_.active_parameters += group->param_ids.size();
    
    OPS_LOG_DEBUG_F("Loaded parameter group: %s (%zu params)", 
                    group_name.c_str(), group->param_ids.size());
}

void MobileZeROOptimizer::unload_parameter_group(const std::string& group_name) {
    auto it = name_to_group_map_.find(group_name);
    if (it == name_to_group_map_.end()) {
        return;
    }
    
    auto& group = parameter_groups_[it->second];
    if (!group->is_active) {
        return; // Already unloaded
    }
    
    // Release parameters (parameter manager will handle tier migration)
    if (zero_stage_ >= MobileZeROStage::PARAMETERS) {
        for (auto param_id : group->param_ids) {
            param_manager_->release_parameter(param_id, false);
        }
    }
    
    group->is_active = false;
    stats_.active_parameters -= group->param_ids.size();
    stats_.parameter_swaps++;
    
    OPS_LOG_DEBUG_F("Unloaded parameter group: %s", group_name.c_str());
}

void MobileZeROOptimizer::optimize_for_inference() {
    // For inference, load all parameters and freeze
    for (auto& group : parameter_groups_) {
        load_parameter_group(group->group_name);
        group->requires_grad = false;
    }
    
    // Clear optimizer states to save memory
    optimizer_states_m_.clear();
    optimizer_states_v_.clear();
    
    OPS_LOG_INFO("Optimized for inference mode");
}

void MobileZeROOptimizer::optimize_for_training() {
    // For training, enable gradient computation
    for (auto& group : parameter_groups_) {
        group->requires_grad = true;
    }
    
    OPS_LOG_INFO("Optimized for training mode");
}

void MobileZeROOptimizer::set_learning_rate(size_t param_id, float lr) {
    learning_rates_[param_id] = lr;
}

void MobileZeROOptimizer::set_learning_rate(float lr) {
    base_learning_rate_ = lr;
}

void MobileZeROOptimizer::set_gradient_accumulation_steps(int steps) {
    gradient_accumulation_steps_ = steps;
    grad_accumulator_ = std::make_unique<GradientAccumulator>(steps);
}

void MobileZeROOptimizer::save_checkpoint(const std::string& checkpoint_path) {
    std::ofstream file(checkpoint_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file: " + checkpoint_path);
    }
    
    // Write version and metadata
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&current_step_), sizeof(current_step_));
    
    // Write optimizer hyperparameters
    file.write(reinterpret_cast<const char*>(&base_learning_rate_), sizeof(base_learning_rate_));
    file.write(reinterpret_cast<const char*>(&weight_decay_), sizeof(weight_decay_));
    file.write(reinterpret_cast<const char*>(&beta1_), sizeof(beta1_));
    file.write(reinterpret_cast<const char*>(&beta2_), sizeof(beta2_));
    
    // Write optimizer states
    uint64_t num_states = optimizer_states_m_.size();
    file.write(reinterpret_cast<const char*>(&num_states), sizeof(num_states));
    
    for (const auto& [param_id, m_state] : optimizer_states_m_) {
        file.write(reinterpret_cast<const char*>(&param_id), sizeof(param_id));
        
        size_t numel = m_state->numel();
        file.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
        file.write(reinterpret_cast<const char*>(m_state->data<float>()), numel * sizeof(float));
        
        auto v_state = optimizer_states_v_[param_id];
        file.write(reinterpret_cast<const char*>(v_state->data<float>()), numel * sizeof(float));
    }
    
    file.close();
    OPS_LOG_INFO_F("Checkpoint saved: %s", checkpoint_path.c_str());
}

void MobileZeROOptimizer::load_checkpoint(const std::string& checkpoint_path) {
    std::ifstream file(checkpoint_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file: " + checkpoint_path);
    }
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error("Unsupported checkpoint version");
    }
    
    file.read(reinterpret_cast<char*>(&current_step_), sizeof(current_step_));
    
    // Read hyperparameters
    file.read(reinterpret_cast<char*>(&base_learning_rate_), sizeof(base_learning_rate_));
    file.read(reinterpret_cast<char*>(&weight_decay_), sizeof(weight_decay_));
    file.read(reinterpret_cast<char*>(&beta1_), sizeof(beta1_));
    file.read(reinterpret_cast<char*>(&beta2_), sizeof(beta2_));
    
    // Read optimizer states
    uint64_t num_states;
    file.read(reinterpret_cast<char*>(&num_states), sizeof(num_states));
    
    for (uint64_t i = 0; i < num_states; ++i) {
        size_t param_id;
        file.read(reinterpret_cast<char*>(&param_id), sizeof(param_id));
        
        size_t numel;
        file.read(reinterpret_cast<char*>(&numel), sizeof(numel));
        
        auto m_state = zeros({static_cast<int64_t>(numel)});
        file.read(reinterpret_cast<char*>(m_state->data<float>()), numel * sizeof(float));
        optimizer_states_m_[param_id] = m_state;
        
        auto v_state = zeros({static_cast<int64_t>(numel)});
        file.read(reinterpret_cast<char*>(v_state->data<float>()), numel * sizeof(float));
        optimizer_states_v_[param_id] = v_state;
    }
    
    file.close();
    OPS_LOG_INFO_F("Checkpoint loaded: %s", checkpoint_path.c_str());
}

MemoryStats MobileZeROOptimizer::get_memory_stats() const {
    return param_manager_->get_memory_stats();
}

} // namespace memory
} // namespace ops
