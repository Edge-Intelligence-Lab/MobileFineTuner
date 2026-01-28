/**
 * @file mobile_optimizer_extensions.h
 * [Documentation in English - see separate docs]
 * 
 * [Documentation in English - see separate docs]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * 6. sparse gradient optimization - mobilememorycritical
 */

#pragma once

#include "mobile_optimizer_state_manager.h"
#include <cmath>
#include <algorithm>

namespace ops {
namespace optim {

/**
 * @brief supportoptimizertype
 */
enum class MobileOptimizerType {
    ADAM = 0,
    ADAMW = 1,
    SGD = 2,
    SGD_MOMENTUM = 3,
    ADAGRAD = 4,
    RMSPROP = 5
};

/**
 * @brief learning rateschedulertype  
 */
enum class LRSchedulerType {
    CONSTANT = 0,               // [Translated]
    LINEAR_DECAY = 1,           // [Translated]
    COSINE_DECAY = 2,           // [Translated]
    EXPONENTIAL_DECAY = 3,  // exponential decay
    STEP_DECAY = 4,             // [Translated]
    WARM_UP_COSINE = 5          // [Translated]
};

/**
 * @brief gradient clippingconfiguration
 */
struct GradientClippingConfig {
    bool enabled = true;
    float max_grad_norm = 1.0f;            // [Translated]
    float clip_value = 0.0f;               // [Translated]
    bool use_global_norm = true;           // [Translated]
    bool adaptive_clipping = false;        // [Translated]
    float adaptive_factor = 0.01f;         // [Translated]
};

/**
 * @brief learning rateschedulerconfiguration
 */
struct LRSchedulerConfig {
    LRSchedulerType type = LRSchedulerType::WARM_UP_COSINE;
    float base_lr = 1e-4f;             // basiclearning rate
    float min_lr = 1e-6f;                  // [Translated]
    
    // warmupconfiguration
    int warmup_steps = 1000;           // warmupstep
    float warmup_start_lr = 1e-6f;         // [Translated]
    
        // [Translated]
    int decay_steps = 10000;               // [Translated]
    float decay_rate = 0.95f;          // decay rate（exponential decay）
    int step_size = 1000;                  // [Translated]
    
        // [Translated]
    bool thermal_scaling = true;           // [Translated]
    bool battery_aware = true;             // [Translated]
    float mobile_lr_factor = 0.8f;     // mobilelrscaling factor
};

/**
 * [Documentation available in English]
 */
struct OptimizerHyperParams {
    MobileOptimizerType type = MobileOptimizerType::ADAMW;
    
    // Adam/AdamWparameter
    float lr = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;        // L2regularization
    bool amsgrad = false;
    bool bias_correction = true;       // Important: bias correction
    bool adamw_mode = true;            // AdamW vs Adam
    
    // SGDparameter
    float momentum_sgd = 0.9f;
    bool nesterov = false;
    
    // Adagradparameter
    float adagrad_eps = 1e-10f;
    
    // RMSpropparameter
    float rmsprop_alpha = 0.99f;
    
        // [Translated]
    bool fp32_optimizer_states = false;     // [Translated]
    bool sparse_gradients = false;      // sparse gradient optimization
    float gradient_sparsity_threshold = 0.01f; // sparsethreshold
};

/**
 * @brief optimizerstatisticsexpand
 */
struct OptimizerExtensionStats {
    // gradientstatistics
    size_t gradient_clips_applied = 0;
    float average_grad_norm = 0.0f;
    float max_grad_norm = 0.0f;
    size_t gradient_overflow_count = 0;
    
    // learning ratestatistics
    float current_lr = 0.0f;
    size_t lr_updates = 0;
    
    // sparsegradientstatistics
    float gradient_sparsity_ratio = 0.0f;
    size_t sparse_updates = 0;
    size_t memory_saved_by_sparsity = 0;
    
        // [Translated]
    size_t nan_gradients = 0;
    size_t inf_gradients = 0;
    
    // mobilerelated
    size_t thermal_lr_reductions = 0;
    size_t battery_optimizations = 0;
    
    // trainingstatistics
    size_t total_parameters = 0;
    size_t trainable_parameters = 0;
    size_t successful_steps = 0;
    size_t failed_steps = 0;
};

/**
 * [Documentation available in English]
 */
class MobileGradientClipper {
private:
    GradientClippingConfig config_;
    OptimizerExtensionStats* stats_;
    
    // mobileadaptiveclipping
    float adaptive_norm_history_[10];      // [Translated]
    int history_index_ = 0;
    bool history_filled_ = false;

public:
    explicit MobileGradientClipper(const GradientClippingConfig& config,
                                  OptimizerExtensionStats* stats = nullptr);
    
    /**
     * @brief clippinggradient
     * @param gradients gradientlist
     * @return clippingfrontgradientnorm
     */
    float clip_gradients(std::vector<TensorPtr>& gradients);
    
    /**
     * @brief compute global gradient norm
     */
    float compute_global_grad_norm(const std::vector<TensorPtr>& gradients);
    
    /**
     * [Documentation available in English]
     */
    float compute_adaptive_clip_value(float current_norm);

private:
    void update_gradient_history(float grad_norm);
};

/**
 * @brief learning ratescheduler
 */
class MobileLRScheduler {
protected:
    LRSchedulerConfig config_;  // Protected for derived classes
    OptimizerExtensionStats* stats_;
    int current_step_ = 0;
    float current_lr_;

public:
    explicit MobileLRScheduler(const LRSchedulerConfig& config,
                              OptimizerExtensionStats* stats = nullptr);
    
    /**
     * @brief acquirecurrentlearning rate
     */
    float get_learning_rate(int step = -1);
    
    /**
     * [Documentation available in English]
     */
    float step();
    
    /**
     * [Documentation available in English]
     */
    void adjust_for_mobile_state(bool is_thermal_throttle, bool is_low_battery);

protected:
    float compute_warmup_lr(int step);
    float compute_cosine_decay_lr(int step);
    float compute_linear_decay_lr(int step);
    float compute_exponential_decay_lr(int step);
    float compute_step_decay_lr(int step);
};

/**
 * [Documentation available in English]
 */
class MobileOptimizer {
private:
    OptimizerHyperParams hyperparams_;
    std::unique_ptr<MobileGradientClipper> gradient_clipper_;
    std::unique_ptr<MobileLRScheduler> lr_scheduler_;
    MobileOptimizerStateManager* state_manager_;
    
    OptimizerExtensionStats extension_stats_;
    int global_step_ = 0;
    
    // sparsegradientsupport
    bool enable_sparse_gradients_;
    std::vector<bool> sparse_mask_;

public:
    MobileOptimizer(const OptimizerHyperParams& hyperparams,
                   const GradientClippingConfig& clip_config,
                   const LRSchedulerConfig& lr_config,
                   MobileOptimizerStateManager* state_manager);
    
    /**
     * @brief execute optimization steps
     * @param param_gradients parameterIDandcorrespondinggradientmapping
     * @return is notsuccessfulupdate
     */
    bool step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    
    /**
     * @brief zerogradient
     */
    void zero_grad();
    
    /**
     * @brief acquirecurrentlearning rate
     */
    float get_current_lr() const { return lr_scheduler_->get_learning_rate(); }
    
    /**
     * @brief acquireexpandstatisticsinforation
     */
    const OptimizerExtensionStats& get_extension_stats() const { return extension_stats_; }

private:
        // [Translated]
    bool adam_step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    bool adamw_step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    bool sgd_step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    bool sgd_momentum_step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    
    // Adam/AdamWcoreimplements
    void adam_update_single_param(size_t param_id, const TensorPtr& param, 
                                 const TensorPtr& gradient, float lr, bool adamw_mode);
    
    // numerical stability check
    bool check_gradient_validity(const TensorPtr& gradient);
    void handle_gradient_overflow();
    
    // sparse gradient optimization
    void apply_gradient_sparsification(std::unordered_map<size_t, TensorPtr>& param_gradients);
    bool is_gradient_sparse(const TensorPtr& gradient);
    TensorPtr sparsify_gradient(const TensorPtr& gradient);
};

/**
 * [Documentation available in English]
 */
class MobileOptimizerFactory {
public:
    /**
     * [Documentation available in English]
     */
    static OptimizerHyperParams create_mobile_adamw_config(float lr = 1e-4f) {
        OptimizerHyperParams config;
        config.type = MobileOptimizerType::ADAMW;
        config.lr = lr;
        config.weight_decay = 0.01f;
        config.adamw_mode = true;
        config.bias_correction = true;
        config.fp32_optimizer_states = false; // mobileuseFP16
        return config;
    }
    
    /**
     * [Documentation available in English]
     */
    static GradientClippingConfig create_mobile_clipping_config() {
        GradientClippingConfig config;
        config.enabled = true;
        config.max_grad_norm = 1.0f;
        config.adaptive_clipping = true;         // [Translated]
        config.adaptive_factor = 0.01f;
        return config;
    }
    
    /**
     * [Documentation available in English]
     */
    static LRSchedulerConfig create_mobile_lr_config(int total_steps) {
        LRSchedulerConfig config;
        config.type = LRSchedulerType::WARM_UP_COSINE;
        config.base_lr = 1e-4f;
        config.min_lr = 1e-6f;
        config.warmup_steps = total_steps / 20; // 5%warmup
        config.decay_steps = total_steps;
        config.thermal_scaling = true;
        config.battery_aware = true;
        config.mobile_lr_factor = 0.8f;
        return config;
    }
};

/**
 * [Documentation available in English]
 */
class CompleteMobileTrainingOptimizer {
private:
    std::unique_ptr<MobileOptimizerStateManager> state_manager_;
    std::unique_ptr<MobileOptimizer> optimizer_;
    
        // [Translated]
    [[maybe_unused]] bool enable_mixed_precision_;
    bool enable_gradient_accumulation_;
    int accumulation_steps_;
    int current_accumulation_step_;
    
    // statisticsandmonitor
    OptimizerExtensionStats total_stats_;

public:
    CompleteMobileTrainingOptimizer(
        size_t available_memory_mb,
        const std::string& storage_path,
        const OptimizerHyperParams& opt_params,
        const GradientClippingConfig& clip_config,
        const LRSchedulerConfig& lr_config
    );
    
    /**
     * @brief completetrainingsteps
     * @param param_gradients parametergradient
     * @param accumulate is notaccumulategradient
     * @return is notexecuteparameterupdate
     */
    bool training_step(const std::unordered_map<size_t, TensorPtr>& param_gradients,
                      bool accumulate = false);
    
    /**
     * @brief registertrainingparameter
     */
    void register_training_parameter(size_t param_id, const std::string& param_name, 
                                   size_t param_size, const std::string& group_name);
    
    /**
     * @brief updatemobilestate
     */
    void update_mobile_system_state(float cpu_util, bool thermal, bool low_battery);
    
    /**
     * @brief acquirecompletestatisticsinforation
     */
    struct CompleteTrainingStats {
        OptimizerStateStats state_stats;
        OptimizerExtensionStats extension_stats;
        size_t total_parameters;
        size_t trainable_parameters;
        float average_update_time_ms;
        size_t successful_steps;
        size_t failed_steps;
    };
    
    CompleteTrainingStats get_complete_stats() const;
    
    /**
     * @brief save/loadcheckpoint
     */
    void save_training_checkpoint(const std::string& path);
    void load_training_checkpoint(const std::string& path);
};

} // namespace memory
} // namespace ops
