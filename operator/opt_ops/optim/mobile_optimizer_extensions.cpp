/**
 * @file mobile_optimizer_extensions.cpp
 * [Documentation available in English]
 */

#include "mobile_optimizer_extensions.h"
#include "mobile_optimizer_state_manager.h"
#include "optimizer_utils.h"
#include "../core/ops.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cstring>

namespace ops {
namespace optim {

// ===============================================================================
// MobileGradientClipper implements - gradient clipping
// ===============================================================================

MobileGradientClipper::MobileGradientClipper(const GradientClippingConfig& config,
                                             OptimizerExtensionStats* stats)
    : config_(config), stats_(stats) {
    std::fill_n(adaptive_norm_history_, 10, 0.0f);
}

float MobileGradientClipper::clip_gradients(std::vector<TensorPtr>& gradients) {
    if (!config_.enabled || gradients.empty()) {
        return 0.0f;
    }
    
    // 1. compute global gradient norm
    float grad_norm = compute_global_grad_norm(gradients);
    
    if (stats_) {
        stats_->average_grad_norm = (stats_->average_grad_norm * stats_->gradient_clips_applied + grad_norm) / 
                                   (stats_->gradient_clips_applied + 1);
        stats_->max_grad_norm = std::max(stats_->max_grad_norm, grad_norm);
    }
    
    // 2. determine clipping threshold
    float clip_threshold = config_.max_grad_norm;
    if (config_.adaptive_clipping) {
        clip_threshold = compute_adaptive_clip_value(grad_norm);
    }
    
    // 3. execute clipping
    if (grad_norm > clip_threshold) {
        float scale_factor = clip_threshold / grad_norm;
        
        for (auto& gradient : gradients) {
                        // [Translated]
            gradient = ops::mul(gradient, scale_factor);
        }
        
        if (stats_) {
            stats_->gradient_clips_applied++;
        }
        
        #ifdef AUTOGRAD_DEBUG
        std::cout << "[GradientClipper] Clipped gradients: norm=" << grad_norm 
                  << " -> " << clip_threshold << " (scale=" << scale_factor << ")" << std::endl;
        #endif
    }
    
    // 4. update adaptive history
    update_gradient_history(grad_norm);
    
    return grad_norm;
}

float MobileGradientClipper::compute_global_grad_norm(const std::vector<TensorPtr>& gradients) {
    float total_norm_sq = 0.0f;
    
    for (const auto& gradient : gradients) {
        if (!gradient) continue;
        
        // checkNaNandInf
        const float* data = gradient->data<float>();
        size_t numel = gradient->numel();
        
        bool has_nan_inf = false;
        for (size_t i = 0; i < numel; ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                has_nan_inf = true;
                if (stats_) {
                    if (std::isnan(data[i])) stats_->nan_gradients++;
                    if (std::isinf(data[i])) stats_->inf_gradients++;
                }
                break;
            }
        }
        
        if (has_nan_inf) {
            if (stats_) stats_->gradient_overflow_count++;
            return std::numeric_limits<float>::infinity();
        }
        
                // [Translated]
        for (size_t i = 0; i < numel; ++i) {
            total_norm_sq += data[i] * data[i];
        }
    }
    
    return std::sqrt(total_norm_sq);
}

float MobileGradientClipper::compute_adaptive_clip_value(float current_norm) {
    (void)current_norm;      // [Translated]
        // [Translated]
    if (!history_filled_ && history_index_ < 10) {
        return config_.max_grad_norm;         // [Translated]
    }
    
    // [Translated comment removed - see documentation]
    float mean = 0.0f;
    int count = history_filled_ ? 10 : history_index_;
    
    for (int i = 0; i < count; ++i) {
        mean += adaptive_norm_history_[i];
    }
    mean /= count;
    
    float variance = 0.0f;
    for (int i = 0; i < count; ++i) {
        float diff = adaptive_norm_history_[i] - mean;
        variance += diff * diff;
    }
    variance /= count;
    float std_dev = std::sqrt(variance);
    
    // adaptivethreshold = mean + adaptive_factor * std_dev
    float adaptive_threshold = mean + config_.adaptive_factor * std_dev;
    
        // [Translated]
    return std::max(config_.max_grad_norm * 0.1f, 
                   std::min(config_.max_grad_norm * 2.0f, adaptive_threshold));
}

void MobileGradientClipper::update_gradient_history(float grad_norm) {
    adaptive_norm_history_[history_index_] = grad_norm;
    history_index_ = (history_index_ + 1) % 10;
    if (history_index_ == 0) {
        history_filled_ = true;
    }
}

// ===============================================================================
// MobileLRScheduler implements - learning rateschedule
// ===============================================================================

MobileLRScheduler::MobileLRScheduler(const LRSchedulerConfig& config,
                                    OptimizerExtensionStats* stats)
    : config_(config), stats_(stats), current_lr_(config.base_lr) {
}

float MobileLRScheduler::get_learning_rate(int step) {
    int use_step = (step >= 0) ? step : current_step_;
    
    float lr = 0.0f;
    
    switch (config_.type) {
        case LRSchedulerType::CONSTANT:
            lr = config_.base_lr;
            break;
            
        case LRSchedulerType::LINEAR_DECAY:
            lr = compute_linear_decay_lr(use_step);
            break;
            
        case LRSchedulerType::COSINE_DECAY:
            lr = compute_cosine_decay_lr(use_step);
            break;
            
        case LRSchedulerType::EXPONENTIAL_DECAY:
            lr = compute_exponential_decay_lr(use_step);
            break;
            
        case LRSchedulerType::STEP_DECAY:
            lr = compute_step_decay_lr(use_step);
            break;
            
        case LRSchedulerType::WARM_UP_COSINE:
            if (use_step < config_.warmup_steps) {
                lr = compute_warmup_lr(use_step);
            } else {
                lr = compute_cosine_decay_lr(use_step - config_.warmup_steps);
            }
            break;
    }
    
    // mobilescale
    lr *= config_.mobile_lr_factor;
    
    // [Translated comment removed - see documentation]
    lr = std::max(lr, config_.min_lr);
    
    return lr;
}

float MobileLRScheduler::step() {
    current_step_++;
    current_lr_ = get_learning_rate();
    
    if (stats_) {
        stats_->current_lr = current_lr_;
        stats_->lr_updates++;
    }
    
    return current_lr_;
}

void MobileLRScheduler::adjust_for_mobile_state(bool is_thermal_throttle, bool is_low_battery) {
    float adjustment = 1.0f;
    
    if (is_thermal_throttle && config_.thermal_scaling) {
        adjustment *= 0.8f;         // [Translated]
        if (stats_) stats_->thermal_lr_reductions++;
    }
    
    if (is_low_battery && config_.battery_aware) {
        adjustment *= 0.9f;         // [Translated]
        if (stats_) stats_->battery_optimizations++;
    }
    
    current_lr_ *= adjustment;
}

float MobileLRScheduler::compute_warmup_lr(int step) {
        // [Translated]
    float progress = static_cast<float>(step) / config_.warmup_steps;
    return config_.warmup_start_lr + (config_.base_lr - config_.warmup_start_lr) * progress;
}

float MobileLRScheduler::compute_cosine_decay_lr(int step) {
    // [Translated comment removed - see documentation]
    float progress = static_cast<float>(step) / config_.decay_steps;
    progress = std::min(progress, 1.0f);
    
    float cosine_factor = 0.5f * (1.0f + std::cos(M_PI * progress));
    return config_.min_lr + (config_.base_lr - config_.min_lr) * cosine_factor;
}

float MobileLRScheduler::compute_linear_decay_lr(int step) {
    // [Translated comment removed - see documentation]
    float progress = static_cast<float>(step) / config_.decay_steps;
    progress = std::min(progress, 1.0f);
    
    return config_.base_lr * (1.0f - progress) + config_.min_lr * progress;
}

float MobileLRScheduler::compute_exponential_decay_lr(int step) {
    // exponential decay
    int decay_epochs = step / config_.step_size;
    return config_.base_lr * std::pow(config_.decay_rate, decay_epochs);
}

float MobileLRScheduler::compute_step_decay_lr(int step) {
    // [Translated comment removed - see documentation]
    int decay_epochs = step / config_.step_size;
    return config_.base_lr * std::pow(0.1f, decay_epochs);
}

// ===============================================================================
// [Translated]
// ===============================================================================

MobileOptimizer::MobileOptimizer(const OptimizerHyperParams& hyperparams,
                                const GradientClippingConfig& clip_config,
                                const LRSchedulerConfig& lr_config,
                                MobileOptimizerStateManager* state_manager)
    : hyperparams_(hyperparams), state_manager_(state_manager) {
    
        // [Translated]
    gradient_clipper_ = std::make_unique<MobileGradientClipper>(clip_config, &extension_stats_);
    
    // createlearning ratescheduler
    lr_scheduler_ = std::make_unique<MobileLRScheduler>(lr_config, &extension_stats_);
    
    // sparsegradientsettings
    enable_sparse_gradients_ = hyperparams_.sparse_gradients;
    
    std::cout << "[MobileOptimizer] Initialized with optimizer type: " 
              << static_cast<int>(hyperparams_.type) << std::endl;
}

bool MobileOptimizer::step(const std::unordered_map<size_t, TensorPtr>& param_gradients) {
    if (param_gradients.empty()) {
        return false;
    }
    
        // [Translated]
    std::unordered_map<size_t, TensorPtr> gradients_copy = param_gradients;
    
    // 1. check gradient validity
    for (const auto& [param_id, gradient] : gradients_copy) {
        if (!check_gradient_validity(gradient)) {
            handle_gradient_overflow();
            return false;
        }
    }
    
    // 2. sparse gradient optimization
    if (enable_sparse_gradients_) {
        apply_gradient_sparsification(gradients_copy);
    }
    
    // 3. gradient clipping
    std::vector<TensorPtr> gradient_list;
    for (const auto& [param_id, gradient] : gradients_copy) {
        gradient_list.push_back(gradient);
    }
    
    float grad_norm = gradient_clipper_->clip_gradients(gradient_list);
    if (std::isinf(grad_norm)) {
        std::cout << "[MobileOptimizer] Gradient overflow detected, skipping step" << std::endl;
        return false;
    }
    
    // 4. update learning rate
    float current_lr = lr_scheduler_->step();
    
    // 5. execute optimization steps
    bool success = false;
    switch (hyperparams_.type) {
        case MobileOptimizerType::ADAM:
            success = adam_step(gradients_copy);
            break;
        case MobileOptimizerType::ADAMW:
            success = adamw_step(gradients_copy);
            break;
        case MobileOptimizerType::SGD:
            success = sgd_step(gradients_copy);
            break;
        case MobileOptimizerType::SGD_MOMENTUM:
            success = sgd_momentum_step(gradients_copy);
            break;
        default:
            std::cout << "[MobileOptimizer] Unsupported optimizer type" << std::endl;
            return false;
    }
    
    if (success) {
        global_step_++;
        std::cout << "[MobileOptimizer] Step " << global_step_ 
                  << " completed, lr=" << current_lr 
                  << ", grad_norm=" << grad_norm << std::endl;
    }
    
    return success;
}

bool MobileOptimizer::adam_step(const std::unordered_map<size_t, TensorPtr>& param_gradients) {
    float lr = lr_scheduler_->get_learning_rate();
    
    for (const auto& [param_id, gradient] : param_gradients) {
        // [Translated comment removed - see documentation]
        // TensorPtr param = get_parameter(param_id);
        
                // [Translated]
        adam_update_single_param(param_id, nullptr, gradient, lr, false);
    }
    
    return true;
}

bool MobileOptimizer::adamw_step(const std::unordered_map<size_t, TensorPtr>& param_gradients) {
    float lr = lr_scheduler_->get_learning_rate();
    
    for (const auto& [param_id, gradient] : param_gradients) {
                // [Translated]
        adam_update_single_param(param_id, nullptr, gradient, lr, true);
    }
    
    return true;
}

bool MobileOptimizer::sgd_step(const std::unordered_map<size_t, TensorPtr>& param_gradients) {
    float lr = lr_scheduler_->get_learning_rate();
    (void)lr;  // TODO: implementsactualparameterupdate
    
    for (const auto& [param_id, gradient] : param_gradients) {
        (void)param_id;  // TODO: implements
        (void)gradient;
        // simpleSGD：param = param - lr * gradient
        // [Translated comment removed - see documentation]
        // param = ops::sub(param, ops::mul(gradient, lr));
    }
    
    return true;
}

bool MobileOptimizer::sgd_momentum_step(const std::unordered_map<size_t, TensorPtr>& param_gradients) {
    float lr = lr_scheduler_->get_learning_rate();
    (void)lr;  // TODO: implements
    float momentum = hyperparams_.momentum_sgd;
    
    for (const auto& [param_id, gradient] : param_gradients) {
        // SGD with momentum
        // v = momentum * v + gradient
        // param = param - lr * v
        
        auto momentum_state = state_manager_->get_momentum_state(param_id);
        auto new_momentum = ops::add(
            ops::mul(momentum_state, momentum),
            gradient
        );
        
        state_manager_->update_momentum_state(param_id, new_momentum);
        
        // parameterupdaterequireactualparameter
        // param = ops::sub(param, ops::mul(new_momentum, lr));
    }
    
    return true;
}

void MobileOptimizer::adam_update_single_param(size_t param_id, const TensorPtr& param,
                                              const TensorPtr& gradient, float lr, bool adamw_mode) {
    // Adamalgorithmimplements
    float beta1 = hyperparams_.beta1;
    float beta2 = hyperparams_.beta2;
    float eps = hyperparams_.eps;
    float weight_decay = hyperparams_.weight_decay;
    
    // acquireoptimizerstate
    auto momentum = state_manager_->get_momentum_state(param_id);
    auto variance = state_manager_->get_variance_state(param_id);
    
    // updatemomentumandvariance
    // m_t = beta1 * m_{t-1} + (1-beta1) * g_t
    auto new_momentum = ops::add(
        ops::mul(momentum, beta1),
        ops::mul(gradient, 1.0f - beta1)
    );
    
    // v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
    auto grad_sq = ops::mul(gradient, gradient);
    auto new_variance = ops::add(
        ops::mul(variance, beta2),
        ops::mul(grad_sq, 1.0f - beta2)
    );
    
    // Bias correction (ifenable)
    TensorPtr corrected_momentum = new_momentum;
    TensorPtr corrected_variance = new_variance;
    
    if (hyperparams_.bias_correction) {
        float bias_correction1 = 1.0f - std::pow(beta1, global_step_ + 1);
        float bias_correction2 = 1.0f - std::pow(beta2, global_step_ + 1);
        
        corrected_momentum = ops::div(new_momentum, bias_correction1);
        corrected_variance = ops::div(new_variance, bias_correction2);
    }
    
        // [Translated]
    if (param) {
        auto sqrt_v = ops::sqrt(corrected_variance);
        auto denom = ops::add(sqrt_v, eps);
        auto update = ops::mul(ops::div(corrected_momentum, denom), lr);
        
                // [Translated]
        TensorPtr new_param;
        if (adamw_mode && weight_decay > 0) {
            // AdamW: param = param - lr * weight_decay * param - lr * m / (sqrt(v) + eps)
            auto weight_decay_term = ops::mul(param, lr * weight_decay);
            new_param = ops::sub(ops::sub(param, weight_decay_term), update);
        } else {
            // Adam: param = param - lr * m / (sqrt(v) + eps)
            new_param = ops::sub(param, update);
        }
        
                // [Translated]
        // param_manager_->update_parameter(param_id, new_param);
    }
    
    // updateoptimizerstate
    state_manager_->update_momentum_state(param_id, new_momentum);
    state_manager_->update_variance_state(param_id, new_variance);
}

bool MobileOptimizer::check_gradient_validity(const TensorPtr& gradient) {
    const float* data = gradient->data<float>();
    size_t numel = gradient->numel();
    
    for (size_t i = 0; i < numel; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            return false;
        }
    }
    return true;
}

void MobileOptimizer::handle_gradient_overflow() {
    extension_stats_.gradient_overflow_count++;
    std::cout << "[MobileOptimizer] Gradient overflow handled, total count: " 
              << extension_stats_.gradient_overflow_count << std::endl;
}

void MobileOptimizer::apply_gradient_sparsification(std::unordered_map<size_t, TensorPtr>& param_gradients) {
    size_t sparse_count = 0;
    size_t total_elements = 0;
    
    for (auto& [param_id, gradient] : param_gradients) {
        if (is_gradient_sparse(gradient)) {
            gradient = sparsify_gradient(gradient);
            sparse_count++;
        }
        total_elements += gradient->numel();
    }
    
    extension_stats_.gradient_sparsity_ratio = static_cast<float>(sparse_count) / param_gradients.size();
    extension_stats_.sparse_updates += sparse_count;
    
    if (sparse_count > 0) {
        std::cout << "[MobileOptimizer] Applied gradient sparsification to " 
                  << sparse_count << "/" << param_gradients.size() << " parameters"
                  << " (total elements: " << total_elements << ")" << std::endl;
    }
}

bool MobileOptimizer::is_gradient_sparse(const TensorPtr& gradient) {
    // [Translated comment removed - see documentation]
    const float* data = gradient->data<float>();
    size_t numel = gradient->numel();
    size_t zero_count = 0;
    
    float threshold = hyperparams_.gradient_sparsity_threshold;
    
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(data[i]) < threshold) {
            zero_count++;
        }
    }
    
    return (static_cast<float>(zero_count) / numel) > 0.5f;     // [Translated]
}

TensorPtr MobileOptimizer::sparsify_gradient(const TensorPtr& gradient) {
        // [Translated]
    auto sparse_grad = gradient->clone();
    float* data = sparse_grad->data<float>();
    size_t numel = sparse_grad->numel();
    float threshold = hyperparams_.gradient_sparsity_threshold;
    
    size_t zeroed_count = 0;
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(data[i]) < threshold) {
            data[i] = 0.0f;
            zeroed_count++;
        }
    }
    
    extension_stats_.memory_saved_by_sparsity += zeroed_count * sizeof(float);
    
    return sparse_grad;
}

void MobileOptimizer::zero_grad() {
        // [Translated]
    std::cout << "[MobileOptimizer] Zeroed gradients" << std::endl;
}

// ===============================================================================
// CompleteMobileTrainingOptimizer implements - completetrainingoptimizer
// ===============================================================================

CompleteMobileTrainingOptimizer::CompleteMobileTrainingOptimizer(
    size_t available_memory_mb,
    const std::string& storage_path,
    const OptimizerHyperParams& opt_params,
    const GradientClippingConfig& clip_config,
    const LRSchedulerConfig& lr_config)
    : enable_mixed_precision_(opt_params.type == MobileOptimizerType::ADAMW),
      enable_gradient_accumulation_(true),
      accumulation_steps_(1),
      current_accumulation_step_(0) {
    
    // createstatemanager
    MobileOptimizerStateConfig state_config;
    state_config.max_active_memory_mb = available_memory_mb / 2;
    state_config.max_standby_memory_mb = available_memory_mb;
    state_config.storage_path = storage_path;
    state_config.enable_compression = true;
    state_config.default_compression = opt_params.fp32_optimizer_states ? 
                                      OptimizerStateCompression::FP16 : 
                                      OptimizerStateCompression::INT8_QUANTIZED;
    
    state_manager_ = std::make_unique<MobileOptimizerStateManager>(state_config);
    
    // createoptimizer
    optimizer_ = std::make_unique<MobileOptimizer>(opt_params, clip_config, lr_config, state_manager_.get());
    
    std::cout << "[CompleteMobileTrainingOptimizer] Initialized complete training system" << std::endl;
}

bool CompleteMobileTrainingOptimizer::training_step(
    const std::unordered_map<size_t, TensorPtr>& param_gradients, bool accumulate) {
    
    if (accumulate && enable_gradient_accumulation_) {
        current_accumulation_step_++;
        
                // [Translated]
        
        if (current_accumulation_step_ < accumulation_steps_) {
            return false;             // [Translated]
        }
        
        // resetaccumulatestep
        current_accumulation_step_ = 0;
    }
    
    // execute optimization steps
    bool success = optimizer_->step(param_gradients);
    
    if (success) {
        total_stats_.successful_steps++;
    } else {
        total_stats_.failed_steps++;
    }
    
        // [Translated]
    if (total_stats_.successful_steps % 10 == 0) {
        state_manager_->optimize_memory_usage();
    }
    
    return success;
}

void CompleteMobileTrainingOptimizer::register_training_parameter(
    size_t param_id, const std::string& param_name, 
    size_t param_size, const std::string& group_name) {
    
    state_manager_->register_parameter_state(param_id, param_name, param_size, group_name, true);
    total_stats_.total_parameters++;
    total_stats_.trainable_parameters++;
}

void CompleteMobileTrainingOptimizer::update_mobile_system_state(
    float cpu_util, bool thermal, bool low_battery) {
    
    state_manager_->update_mobile_state(cpu_util, thermal, low_battery);
    
    // meanwhileupdate learning ratescheduler（viaoptimizer）
    // optimizer_->adjust_lr_for_mobile_state(thermal, low_battery);
}

CompleteMobileTrainingOptimizer::CompleteTrainingStats 
CompleteMobileTrainingOptimizer::get_complete_stats() const {
    CompleteTrainingStats stats;
    stats.state_stats = state_manager_->get_statistics();
    stats.extension_stats = optimizer_->get_extension_stats();
    stats.total_parameters = total_stats_.total_parameters;
    stats.trainable_parameters = total_stats_.trainable_parameters;
    stats.successful_steps = total_stats_.successful_steps;
    stats.failed_steps = total_stats_.failed_steps;
    
        // [Translated]
    if (stats.successful_steps > 0) {
        stats.average_update_time_ms = 50.0f;         // [Translated]
    }
    
    return stats;
}

void CompleteMobileTrainingOptimizer::save_training_checkpoint(const std::string& path) {
    state_manager_->save_checkpoint(path + "/optimizer_states.bin");
    
        // [Translated]
    std::cout << "[CompleteMobileTrainingOptimizer] Training checkpoint saved to " << path << std::endl;
}

void CompleteMobileTrainingOptimizer::load_training_checkpoint(const std::string& path) {
    state_manager_->load_checkpoint(path + "/optimizer_states.bin");
    
        // [Translated]
    std::cout << "[CompleteMobileTrainingOptimizer] Training checkpoint loaded from " << path << std::endl;
}

} // namespace memory
} // namespace ops
