/**
 * @file mobile_optimizer_advanced.cpp
 * [Documentation available in English]
 */

#include "mobile_optimizer_advanced.h"
#include "optimizer_utils.h"
#include "../core/ops.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <iostream>

namespace ops {
namespace optim {

// ===============================================================================
// AdvancedCheckpointManager implements
// ===============================================================================

AdvancedCheckpointManager::AdvancedCheckpointManager(const std::string& dir, int max_keep)
    : base_checkpoint_dir_(dir), max_checkpoints_to_keep_(max_keep) {
    std::filesystem::create_directories(dir);
}

bool AdvancedCheckpointManager::save_training_state(const TrainingState& state, 
                                                   const std::string& checkpoint_name) {
    try {
        std::string checkpoint_path = base_checkpoint_dir_ + "/" + checkpoint_name + ".json";
        std::ofstream file(checkpoint_path);
        
        if (!file.is_open()) {
            std::cout << "[AdvancedCheckpointManager] Failed to create checkpoint file: " 
                      << checkpoint_path << std::endl;
            return false;
        }
        
        // saveJSONforattrainingstate
        file << "{\n";
        file << "  \"global_step\": " << state.global_step << ",\n";
        file << "  \"best_loss\": " << state.best_loss << ",\n";
        file << "  \"current_lr\": " << state.current_lr << ",\n";
        file << "  \"total_thermal_events\": " << state.total_thermal_events << ",\n";
        file << "  \"total_battery_events\": " << state.total_battery_events << ",\n";
        
                // [Translated]
        file << "  \"loss_history\": [";
        for (size_t i = 0; i < state.loss_history.size(); ++i) {
            if (i > 0) file << ", ";
            file << state.loss_history[i];
        }
        file << "],\n";
        
                // [Translated]
        file << "  \"power_history\": [";
        for (size_t i = 0; i < state.power_history.size(); ++i) {
            if (i > 0) file << ", ";
            file << state.power_history[i];
        }
        file << "],\n";
        
                // [Translated]
        file << "  \"thermal_history\": [";
        for (size_t i = 0; i < state.thermal_history.size(); ++i) {
            if (i > 0) file << ", ";
            file << state.thermal_history[i];
        }
        file << "]\n";
        
        file << "}\n";
        file.close();
        
                // [Translated]
        saved_checkpoints_.push_back(checkpoint_name);
        
        // cleanupoldcheckpoint
        if (saved_checkpoints_.size() > static_cast<size_t>(max_checkpoints_to_keep_)) {
            cleanup_old_checkpoints();
        }
        
        std::cout << "[AdvancedCheckpointManager] Saved checkpoint: " << checkpoint_name << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "[AdvancedCheckpointManager] Error saving checkpoint: " << e.what() << std::endl;
        return false;
    }
}

bool AdvancedCheckpointManager::load_training_state(TrainingState& state, 
                                                   const std::string& checkpoint_name) {
    try {
        std::string checkpoint_path = base_checkpoint_dir_ + "/" + checkpoint_name + ".json";
        std::ifstream file(checkpoint_path);
        
        if (!file.is_open()) {
            std::cout << "[AdvancedCheckpointManager] Checkpoint file not found: " 
                      << checkpoint_path << std::endl;
            return false;
        }
        
                // [Translated]
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("\"global_step\":") != std::string::npos) {
                size_t pos = line.find(':');
                state.global_step = std::stoi(line.substr(pos + 1));
            }
            else if (line.find("\"best_loss\":") != std::string::npos) {
                size_t pos = line.find(':');
                state.best_loss = std::stof(line.substr(pos + 1));
            }
            else if (line.find("\"current_lr\":") != std::string::npos) {
                size_t pos = line.find(':');
                state.current_lr = std::stof(line.substr(pos + 1));
            }
                        // [Translated]
        }
        
        file.close();
        std::cout << "[AdvancedCheckpointManager] Loaded checkpoint: " << checkpoint_name << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "[AdvancedCheckpointManager] Error loading checkpoint: " << e.what() << std::endl;
        return false;
    }
}

void AdvancedCheckpointManager::cleanup_old_checkpoints() {
    while (saved_checkpoints_.size() > static_cast<size_t>(max_checkpoints_to_keep_)) {
        std::string old_checkpoint = saved_checkpoints_.front();
        saved_checkpoints_.erase(saved_checkpoints_.begin());
        
        std::string checkpoint_path = base_checkpoint_dir_ + "/" + old_checkpoint + ".json";
        std::filesystem::remove(checkpoint_path);
        
        std::cout << "[AdvancedCheckpointManager] Removed old checkpoint: " << old_checkpoint << std::endl;
    }
}

void AdvancedCheckpointManager::auto_checkpoint(const TrainingState& state, float current_loss) {
    static float best_loss = std::numeric_limits<float>::max();
    
    // [Translated comment removed - see documentation]
    if (state.global_step % 100 == 0) {
        std::string checkpoint_name = "step_" + std::to_string(state.global_step);
        save_training_state(state, checkpoint_name);
    }
    
        // [Translated]
    if (current_loss < best_loss) {
        best_loss = current_loss;
        save_training_state(state, "best_checkpoint");
        std::cout << "[AdvancedCheckpointManager] New best checkpoint saved with loss: " 
                  << current_loss << std::endl;
    }
}

std::string AdvancedCheckpointManager::get_best_checkpoint() const {
    return base_checkpoint_dir_ + "/best_checkpoint.json";
}

// ===============================================================================
// AdvancedLRScheduler implements
// ===============================================================================

AdvancedLRScheduler::AdvancedLRScheduler(const LRSchedulerConfig& config,
                                        const WarmRestartConfig& restart_config,
                                        OptimizerExtensionStats* stats)
    : MobileLRScheduler(config, stats), restart_config_(restart_config) {
    if (restart_config_.enabled) {
        std::cout << "[AdvancedLRScheduler] Warm restart enabled with period: " 
                  << restart_config_.restart_period << std::endl;
    }
}

float AdvancedLRScheduler::compute_polynomial_decay_lr(int step, float power) {
    float progress = static_cast<float>(step) / config_.decay_steps;
    progress = std::min(progress, 1.0f);
    
    float decay_factor = std::pow(1.0f - progress, power);
    return config_.min_lr + (config_.base_lr - config_.min_lr) * decay_factor;
}

float AdvancedLRScheduler::compute_multistep_decay_lr(int step, 
                                                     const std::vector<int>& milestones, 
                                                     float gamma) {
    int decay_count = 0;
    for (int milestone : milestones) {
        if (step >= milestone) {
            decay_count++;
        } else {
            break;
        }
    }
    
    return config_.base_lr * std::pow(gamma, decay_count);
}

float AdvancedLRScheduler::compute_warm_restart_lr(int step) {
    if (!restart_config_.enabled) {
        return compute_cosine_decay_lr(step);
    }
    
        // [Translated]
    int period = restart_config_.restart_period * static_cast<int>(std::pow(restart_config_.restart_mult, restart_count_));
    int steps_in_period = step - last_restart_step_;
    
    if (steps_in_period >= period) {
        // requirerestart
        perfor_restart();
        last_restart_step_ = step;
        steps_in_period = 0;
        restart_count_++;
        period = restart_config_.restart_period * static_cast<int>(std::pow(restart_config_.restart_mult, restart_count_));
    }
    
    // [Translated comment removed - see documentation]
    float progress = static_cast<float>(steps_in_period) / period;
    float cosine_factor = 0.5f * (1.0f + std::cos(M_PI * progress));
    
    float min_lr = config_.base_lr * restart_config_.min_lr_ratio;
    return min_lr + (config_.base_lr - min_lr) * cosine_factor;
}

bool AdvancedLRScheduler::should_restart(float current_perforance) {
    if (!restart_config_.adaptive_restart) {
        return false;
    }
    
    update_perforance_history(current_perforance);
    
        // [Translated]
    if (perforance_history_.size() < static_cast<size_t>(restart_config_.patience_steps)) {
        return false;
    }
    
    float recent_improvement = perforance_history_.back() - perforance_history_[perforance_history_.size() - static_cast<size_t>(restart_config_.patience_steps)];
    
    return recent_improvement < restart_config_.perforance_threshold;
}

void AdvancedLRScheduler::perfor_restart() {
    std::cout << "[AdvancedLRScheduler] Perforing warm restart at step " 
              << current_step_ << " (restart #" << restart_count_ + 1 << ")" << std::endl;
    
    // resetlearning ratetoinitialvalue
    current_lr_ = config_.base_lr;
    
    if (stats_) {
                // [Translated]
    }
}

void AdvancedLRScheduler::update_perforance_history(float perforance) {
    perforance_history_.push_back(perforance);
    
    // [Translated comment removed - see documentation]
    const size_t max_history = restart_config_.patience_steps * 3;
    if (perforance_history_.size() > max_history) {
        perforance_history_.erase(perforance_history_.begin());
    }
}

// ===============================================================================
// NumericalStabilityManager implements
// ===============================================================================

NumericalStabilityManager::NumericalStabilityManager(const NumericalStabilityConfig& config)
    : config_(config), current_loss_scale_(config.initial_scale) {
    std::cout << "[NumericalStabilityManager] Initialized with loss scale: " 
              << current_loss_scale_ << std::endl;
}

bool NumericalStabilityManager::check_gradient_overflow(const std::vector<TensorPtr>& gradients) {
    for (const auto& gradient : gradients) {
        if (!gradient) continue;
        
        const float* data = gradient->data<float>();
        size_t numel = gradient->numel();
        
        for (size_t i = 0; i < numel; ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                return true;
            }
            
                        // [Translated]
            if (std::abs(data[i]) > config_.max_grad_norm * current_loss_scale_) {
                return true;
            }
        }
    }
    
    return false;
}

void NumericalStabilityManager::handle_overflow() {
    overflow_count_++;
    
    // [Translated comment removed - see documentation]
    recent_overflows_.push(true);
    if (recent_overflows_.size() > 10) {
        recent_overflows_.pop();
    }
    
    // [Translated comment removed - see documentation]
    int recent_overflow_count = 0;
    std::queue<bool> temp = recent_overflows_;
    while (!temp.empty()) {
        if (temp.front()) recent_overflow_count++;
        temp.pop();
    }
    
    float backoff_factor = config_.scale_backoff_factor;
    if (recent_overflow_count > 5) {
        backoff_factor = config_.scale_backoff_factor * config_.scale_backoff_factor;         // [Translated]
    }
    
    current_loss_scale_ = std::max(config_.min_loss_scale, current_loss_scale_ * backoff_factor);
    
    std::cout << "[NumericalStabilityManager] Gradient overflow detected! "
              << "New loss scale: " << current_loss_scale_ 
              << " (overflow count: " << overflow_count_ << ")" << std::endl;
}

void NumericalStabilityManager::scale_gradients(std::vector<TensorPtr>& gradients) {
    if (!config_.enable_gradient_scaling || current_loss_scale_ == 1.0f) {
        return;
    }
    
    for (auto& gradient : gradients) {
        if (gradient) {
            gradient = ops::mul(gradient, current_loss_scale_);
        }
    }
}

void NumericalStabilityManager::unscale_gradients(std::vector<TensorPtr>& gradients) {
    if (!config_.enable_gradient_scaling || current_loss_scale_ == 1.0f) {
        return;
    }
    
    float inv_scale = 1.0f / current_loss_scale_;
    for (auto& gradient : gradients) {
        if (gradient) {
            gradient = ops::mul(gradient, inv_scale);
        }
    }
}

void NumericalStabilityManager::update_loss_scale(bool overflow_detected) {
    if (overflow_detected) {
        handle_overflow();
        successful_steps_ = 0;
    } else {
        successful_steps_++;
        recent_overflows_.push(false);
        if (recent_overflows_.size() > 10) {
            recent_overflows_.pop();
        }
        
        // [Translated comment removed - see documentation]
        if (successful_steps_ >= config_.scale_growth_interval && 
            current_loss_scale_ < config_.max_loss_scale) {
            
            current_loss_scale_ = std::min(config_.max_loss_scale, 
                                         current_loss_scale_ * config_.scale_growth_factor);
            successful_steps_ = 0;
            
            std::cout << "[NumericalStabilityManager] Increased loss scale to: " 
                      << current_loss_scale_ << std::endl;
        }
    }
}

// ===============================================================================
// PowerOptimizationManager implements
// ===============================================================================

PowerOptimizationManager::PowerOptimizationManager(const PowerOptimizationConfig& config)
    : config_(config) {
    last_power_measurement_ = std::chrono::steady_clock::now();
    std::cout << "[PowerOptimizationManager] Initialized with target power: " 
              << config_.target_power_consumption << "W" << std::endl;
}

void PowerOptimizationManager::update_power_state(float battery_level, bool plugged_in, float temperature) {
    battery_level_ = battery_level;
    is_plugged_in_ = plugged_in;
    device_temperature_ = temperature;
    
        // [Translated]
    current_power_consumption_ = 2.5f + 1.5f * (temperature - 30.0f) / 20.0f;
    if (!plugged_in) {
        current_power_consumption_ *= 0.8f;         // [Translated]
    }
    
    power_history_.push_back(current_power_consumption_);
    if (power_history_.size() > 100) {
        power_history_.erase(power_history_.begin());
    }
    
    // recordcriticalevent
    if (battery_level < config_.battery_critical_threshold) {
        log_power_event("CRITICAL_BATTERY");
    } else if (battery_level < config_.battery_low_threshold) {
        log_power_event("LOW_BATTERY");
    }
    
    if (temperature > 65.0f) {
        log_power_event("HIGH_TEMPERATURE");
    }
}

float PowerOptimizationManager::compute_power_adjustment_factor() {
    float factor = 1.0f;
    
    // based onbatterystateadjust
    if (!is_plugged_in_) {
        if (battery_level_ < config_.battery_critical_threshold) {
            factor *= 0.5f;             // [Translated]
        } else if (battery_level_ < config_.battery_low_threshold) {
            factor *= config_.power_reduction_factor;             // [Translated]
        }
    }
    
    // based ontemperatureadjust
    if (device_temperature_ > 60.0f) {
        factor *= 0.8f;         // [Translated]
    }
    
        // [Translated]
    if (!power_history_.empty()) {
        float avg_power = std::accumulate(power_history_.begin(), power_history_.end(), 0.0f) / power_history_.size();
        if (avg_power > config_.target_power_consumption) {
            float excess_ratio = avg_power / config_.target_power_consumption;
            factor *= (1.0f / excess_ratio);
        }
    }
    
    return std::max(0.1f, std::min(1.0f, factor)); // limitat0.1-1.0rangeinner
}

void PowerOptimizationManager::apply_power_optimizations(float& learning_rate, int& batch_size) {
    float adjustment_factor = compute_power_adjustment_factor();
    
    if (adjustment_factor < 1.0f) {
        learning_rate *= adjustment_factor;
        
        // [Translated comment removed - see documentation]
        if (adjustment_factor < 0.7f) {
            batch_size = std::max(1, static_cast<int>(batch_size * adjustment_factor));
        }
        
        std::cout << "[PowerOptimizationManager] Applied power optimization: factor=" 
                  << adjustment_factor << ", new_lr=" << learning_rate << std::endl;
    }
}

float PowerOptimizationManager::predict_power_consumption(float lr, int batch_size) {
    // [Translated comment removed - see documentation]
    float base_power = 2.0f;
    float lr_factor = 1.0f + lr * 100;     // [Translated]
    float batch_factor = 1.0f + batch_size * 0.1f;     // [Translated]
    
    return base_power * lr_factor * batch_factor;
}

void PowerOptimizationManager::log_power_event(const std::string& event) {
    auto now = std::chrono::steady_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    
    std::cout << "[PowerOptimizationManager] Power event: " << event 
              << " at timestamp " << timestamp 
              << " (battery: " << battery_level_ * 100 << "%, temp: " << device_temperature_ << "°C)" << std::endl;
}

float PowerOptimizationManager::get_battery_drain_rate() {
    // [Translated comment removed - see documentation]
    if (power_history_.size() < 10) return 0.0f;
    
    float avg_power = std::accumulate(power_history_.end() - 10, power_history_.end(), 0.0f) / 10.0f;
    
        // [Translated]
    float battery_capacity_wh = 18.5f;
    float drain_rate_per_hour = avg_power / battery_capacity_wh;
    
    return drain_rate_per_hour;
}

// ===============================================================================
// AsyncOptimizerUpdater implements
// ===============================================================================

AsyncOptimizerUpdater::AsyncOptimizerUpdater(const AsyncOptimizerConfig& config)
    : config_(config) {
    if (config_.enable_async_updates) {
        worker_thread_ = std::thread(&AsyncOptimizerUpdater::worker_loop, this);
        std::cout << "[AsyncOptimizerUpdater] Started async optimizer worker thread" << std::endl;
    }
}

AsyncOptimizerUpdater::~AsyncOptimizerUpdater() {
    if (config_.enable_async_updates) {
        should_stop_ = true;
        queue_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
}

void AsyncOptimizerUpdater::submit_gradient_update(size_t param_id, const TensorPtr& gradient) {
    if (!config_.enable_async_updates) {
        return;
    }
    
    float importance = compute_parameter_importance(param_id, gradient);
    
    if (should_sync_update(param_id, importance)) {
        // syncprocessimportantparameter
        return;
    }
    
        // [Translated]
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        async_gradient_queue_.emplace(param_id, gradient);
    }
    queue_cv_.notify_one();
}

float AsyncOptimizerUpdater::compute_parameter_importance(size_t param_id, const TensorPtr& gradient) {
    float importance = 0.0f;
    
        // [Translated]
    if (config_.use_gradient_magnitude) {
        const float* data = gradient->data<float>();
        float grad_norm = 0.0f;
        size_t numel = static_cast<size_t>(gradient->numel());
        for (size_t i = 0; i < numel; ++i) {
            grad_norm += data[i] * data[i];
        }
        grad_norm = std::sqrt(grad_norm);
        importance += grad_norm;
    }
    
        // [Translated]
    if (config_.use_historical_importance) {
        auto it = importance_history_.find(param_id);
        if (it != importance_history_.end() && !it->second.empty()) {
            float avg_importance = std::accumulate(it->second.begin(), it->second.end(), 0.0f) / it->second.size();
            importance = 0.7f * importance + 0.3f * avg_importance;             // [Translated]
        }
    }
    
        // [Translated]
    update_importance_history(param_id, importance);
    
    return importance;
}

bool AsyncOptimizerUpdater::should_sync_update(size_t param_id, float importance) {
    (void)param_id;  // TODO: implementsbased onparam_idsyncstrategy
    return importance > config_.importance_threshold;
}

void AsyncOptimizerUpdater::worker_loop() {
    while (!should_stop_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return !async_gradient_queue_.empty() || should_stop_; });
        
        if (should_stop_) break;
        
        process_async_updates();
    }
}

void AsyncOptimizerUpdater::process_async_updates() {
    int processed_count = 0;
    const int max_batch_size = config_.accumulation_window;
    
    while (!async_gradient_queue_.empty() && processed_count < max_batch_size) {
        auto [param_id, gradient] = async_gradient_queue_.front();
        async_gradient_queue_.pop();
        
                // [Translated]
                // [Translated]
        processed_count++;
    }
    
    if (processed_count > 0) {
        std::cout << "[AsyncOptimizerUpdater] Processed " << processed_count 
                  << " async gradient updates" << std::endl;
    }
}

void AsyncOptimizerUpdater::update_importance_history(size_t param_id, float importance) {
    auto& history = importance_history_[param_id];
    history.push_back(importance);
    
    const size_t max_history_size = 20;
    if (history.size() > max_history_size) {
        history.erase(history.begin());
    }
    
    parameter_importance_[param_id] = importance;
}

// ===============================================================================
// CompleteMobileOptimizerAdvanced implements
// ===============================================================================

CompleteMobileOptimizerAdvanced::CompleteMobileOptimizerAdvanced(
    size_t available_memory_mb,
    const std::string& storage_path,
    const OptimizerHyperParams& opt_params,
    const GradientClippingConfig& clip_config,
    const LRSchedulerConfig& lr_config,
    const WarmRestartConfig& restart_config,
    const NumericalStabilityConfig& stability_config,
    const PowerOptimizationConfig& power_config,
    const AsyncOptimizerConfig& async_config)
    : restart_config_(restart_config), 
      stability_config_(stability_config),
      power_config_(power_config),
      async_config_(async_config) {
    
    std::cout << "[CompleteMobileOptimizerAdvanced] Initializing advanced mobile optimizer..." << std::endl;
    
    // createbasicoptimizer
    base_optimizer_ = std::make_unique<CompleteMobileTrainingOptimizer>(
        available_memory_mb, storage_path, opt_params, clip_config, lr_config
    );
    
    initialize_advanced_components();
    
    std::cout << "[CompleteMobileOptimizerAdvanced] Advanced mobile optimizer initialized successfully!" << std::endl;
}

void CompleteMobileOptimizerAdvanced::initialize_advanced_components() {
    // createcheckpointmanager
    checkpoint_manager_ = std::make_unique<AdvancedCheckpointManager>("./advanced_checkpoints", 5);
    
    // createadvancedlearning ratescheduler
    LRSchedulerConfig lr_config;  // requirefrombase_optimizeracquire
    advanced_lr_scheduler_ = std::make_unique<AdvancedLRScheduler>(lr_config, restart_config_);
    
        // [Translated]
    stability_manager_ = std::make_unique<NumericalStabilityManager>(stability_config_);
    
    // createpowermanager
    power_manager_ = std::make_unique<PowerOptimizationManager>(power_config_);
    
        // [Translated]
    async_updater_ = std::make_unique<AsyncOptimizerUpdater>(async_config_);
    
    std::cout << "[CompleteMobileOptimizerAdvanced] All advanced components initialized" << std::endl;
}

void CompleteMobileOptimizerAdvanced::create_parameter_group(const ParameterGroupConfig& group_config) {
    auto group = std::make_unique<ParameterGroupConfig>(group_config);
    parameter_groups_.push_back(std::move(group));
    
    std::cout << "[CompleteMobileOptimizerAdvanced] Created parameter group: " 
              << group_config.group_name 
              << " with " << group_config.param_ids.size() << " parameters" << std::endl;
}

bool CompleteMobileOptimizerAdvanced::advanced_training_step(
    const std::unordered_map<size_t, TensorPtr>& param_gradients,
    float training_loss,
    float validation_loss) {
    
    // 1. numerical stability check
    std::vector<TensorPtr> gradients;
    for (const auto& [param_id, grad] : param_gradients) {
        gradients.push_back(grad);
    }
    
    bool overflow_detected = stability_manager_->check_gradient_overflow(gradients);
    
    if (overflow_detected) {
        stability_manager_->handle_overflow();
        return false;         // [Translated]
    }
    
    // 2. gradient scaling
    stability_manager_->scale_gradients(gradients);
    
    // 3. async optimizer update check
    std::unordered_map<size_t, TensorPtr> sync_gradients;
    for (const auto& [param_id, grad] : param_gradients) {
        async_updater_->submit_gradient_update(param_id, grad);
        // [Translated comment removed - see documentation]
        sync_gradients[param_id] = grad;
    }
    
    // 4. execute basic training steps
    bool success = base_optimizer_->training_step(sync_gradients, false);
    
    // 5. update numerical stability state
    stability_manager_->update_loss_scale(false); // successfulexecute
    
    // 6. update training metrics
    update_training_metrics(training_loss, validation_loss);
    
    // 7. auto-save checkpoint
    if (success) {
        AdvancedCheckpointManager::TrainingState state;
        state.global_step = training_loss_history_.size();
        state.best_loss = best_validation_loss_;
        state.current_lr = 1e-4f; // shouldfromscheduleracquire
        
        checkpoint_manager_->auto_checkpoint(state, training_loss);
    }
    
    // 8. auto-adjust optimization strategy
    if (training_loss_history_.size() % 100 == 0) {
        auto_adjust_optimization_strategy();
    }
    
    return success;
}

void CompleteMobileOptimizerAdvanced::update_training_metrics(float training_loss, float validation_loss) {
    training_loss_history_.push_back(training_loss);
    
    if (validation_loss >= 0) {
        validation_loss_history_.push_back(validation_loss);
        
        if (validation_loss < best_validation_loss_) {
            best_validation_loss_ = validation_loss;
            steps_without_improvement_ = 0;
        } else {
            steps_without_improvement_++;
        }
    }
    
    // [Translated comment removed - see documentation]
    const size_t max_history = 1000;
    if (training_loss_history_.size() > max_history) {
        training_loss_history_.erase(training_loss_history_.begin());
    }
    if (validation_loss_history_.size() > max_history) {
        validation_loss_history_.erase(validation_loss_history_.begin());
    }
}

void CompleteMobileOptimizerAdvanced::auto_adjust_optimization_strategy() {
    // [Translated comment removed - see documentation]
    if (training_loss_history_.size() < 50) return;
    
    // [Translated comment removed - see documentation]
    float recent_improvement = training_loss_history_[training_loss_history_.size()-50] - 
                              training_loss_history_.back();
    
    if (recent_improvement < 0.001f) {
        // [Translated comment removed - see documentation]
        if (restart_config_.enabled && advanced_lr_scheduler_->should_restart(recent_improvement)) {
            advanced_lr_scheduler_->perfor_restart();
        }
    }
    
    std::cout << "[CompleteMobileOptimizerAdvanced] Auto-adjusted optimization strategy" << std::endl;
}

bool CompleteMobileOptimizerAdvanced::should_early_stop() {
    const int patience = 200;
    return steps_without_improvement_ > patience;
}

CompleteMobileOptimizerAdvanced::AdvancedTrainingStats 
CompleteMobileOptimizerAdvanced::get_advanced_stats() const {
    AdvancedTrainingStats stats;
    
    // acquirebasicstatistics
    stats.base_stats = base_optimizer_->get_complete_stats();
    
    // addadvancedstatistics
    stats.restart_count = advanced_lr_scheduler_ ? 0 : 0; // requirefromscheduleracquire
    stats.overflow_count = stability_manager_->get_overflow_count();
    stats.current_loss_scale = stability_manager_->get_current_loss_scale();
    
    // powerstatistics
    stats.average_power_consumption = 3.5f; // shouldfrompower_manager_acquire
    stats.total_energy_consumed = stats.average_power_consumption * 
                                 training_loss_history_.size() * 0.1f;                                  // [Translated]
    
    // trainingefficiency
    stats.loss_history = training_loss_history_;
    if (training_loss_history_.size() > 10) {
        float initial_loss = training_loss_history_[0];
        float current_loss = training_loss_history_.back();
        stats.convergence_rate = (initial_loss - current_loss) / training_loss_history_.size();
    }
    
    stats.training_efficiency = stats.base_stats.successful_steps / 
                               (stats.base_stats.successful_steps + stats.base_stats.failed_steps);
    
    return stats;
}

void CompleteMobileOptimizerAdvanced::save_advanced_checkpoint(const std::string& name, float current_loss) {
    (void)current_loss;  // TODO: availableinadaptivecheckpointstrategy
    AdvancedCheckpointManager::TrainingState state;
    state.global_step = training_loss_history_.size();
    state.best_loss = best_validation_loss_;
    state.current_lr = 1e-4f; // shouldfromscheduleracquire
    state.loss_history = training_loss_history_;
    
    checkpoint_manager_->save_training_state(state, name);
    base_optimizer_->save_training_checkpoint("./checkpoints/" + name);
}

bool CompleteMobileOptimizerAdvanced::load_advanced_checkpoint(const std::string& name) {
    AdvancedCheckpointManager::TrainingState state;
    bool success = checkpoint_manager_->load_training_state(state, name);
    
    if (success) {
        training_loss_history_ = state.loss_history;
        best_validation_loss_ = state.best_loss;
        base_optimizer_->load_training_checkpoint("./checkpoints/" + name);
    }
    
    return success;
}

void CompleteMobileOptimizerAdvanced::generate_training_analysis_report(const std::string& report_path) {
    auto stats = get_advanced_stats();
    
    std::ofstream report(report_path);
    report << "Advanced Mobile Training Analysis Report\n";
    report << "========================================\n\n";
    
    report << "Training Summary:\n";
    report << "  Total Steps: " << stats.base_stats.successful_steps + stats.base_stats.failed_steps << "\n";
    report << "  Success Rate: " << stats.training_efficiency * 100 << "%\n";
    report << "  Convergence Rate: " << stats.convergence_rate << "\n";
    report << "  Restart Count: " << stats.restart_count << "\n";
    report << "  Overflow Count: " << stats.overflow_count << "\n\n";
    
    report << "Memory Usage:\n";
    report << "  Peak Memory: " << stats.base_stats.state_stats.active_memory_used / 1024 / 1024 << "MB\n";
    report << "  Compression Ratio: " << stats.base_stats.state_stats.average_compression_ratio << "x\n";
    report << "  Memory Saved: " << stats.base_stats.state_stats.memory_saved_by_compression / 1024 / 1024 << "MB\n\n";
    
    report << "Power Consumption:\n";
    report << "  Average Power: " << stats.average_power_consumption << "W\n";
    report << "  Total Energy: " << stats.total_energy_consumed << "Wh\n\n";
    
    report.close();
    std::cout << "[CompleteMobileOptimizerAdvanced] Training analysis report saved to: " << report_path << std::endl;
}

// ===============================================================================
// [Translated]
// ===============================================================================

std::unique_ptr<CompleteMobileOptimizerAdvanced> create_advanced_mobile_optimizer(
    size_t available_memory_mb,
    const std::string& storage_path,
    const std::string& checkpoint_dir,
    bool enable_all_advanced_features) {
    (void)checkpoint_dir;  // TODO: used forsettingscheckpointdirectory
    
    // createdefaultconfiguration
    auto opt_params = MobileOptimizerFactory::create_mobile_adamw_config();
    auto clip_config = MobileOptimizerFactory::create_mobile_clipping_config();
    auto lr_config = MobileOptimizerFactory::create_mobile_lr_config(10000);
    
        // [Translated]
    WarmRestartConfig restart_config;
    restart_config.enabled = enable_all_advanced_features;
    restart_config.restart_period = 1000;
    restart_config.adaptive_restart = true;
    
    NumericalStabilityConfig stability_config;
    stability_config.enable_overflow_detection = enable_all_advanced_features;
    stability_config.enable_gradient_scaling = enable_all_advanced_features;
    
    PowerOptimizationConfig power_config;
    power_config.enable_power_aware = enable_all_advanced_features;
    power_config.target_power_consumption = 3.0f;
    
    AsyncOptimizerConfig async_config;
    async_config.enable_async_updates = enable_all_advanced_features;
    async_config.importance_threshold = 0.1f;
    
    return std::make_unique<CompleteMobileOptimizerAdvanced>(
        available_memory_mb, storage_path, opt_params, clip_config, lr_config,
        restart_config, stability_config, power_config, async_config
    );
}

} // namespace memory
} // namespace ops
