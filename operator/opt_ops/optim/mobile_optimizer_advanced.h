/**
 * @file mobile_optimizer_advanced.h
 * [Documentation in English - see separate docs]
 * 
 * [Documentation in English - see separate docs]
 * [Documentation available in English]
 * 2. Warm Restart - learning raterestartandtrainingresume
 * 3. advancedcheckpointsystem - completestatesave/load
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 */

#pragma once

#include "mobile_optimizer_extensions.h"
#include <fstream>
#include <thread>
#include <queue>
#include <condition_variable>

namespace ops {
namespace optim {

/**
 * [Documentation available in English]
 */
struct ParameterGroupConfig {
    std::string group_name;                        // [Translated]
    std::vector<size_t> param_ids;             // parameterIDlist
    
    // [Translated comment removed - see documentation]
    float group_lr_multiplier = 1.0f;             // [Translated]
    float group_weight_decay = -1.0f;             // [Translated]
    float group_beta1 = -1.0f;                // Beta1 (-1representuseglobalvalue)
    float group_beta2 = -1.0f;                // Beta2 (-1representuseglobalvalue)
    
        // [Translated]
    OptimizerStateCompression group_compression = OptimizerStateCompression::ADAPTIVE;
    bool freeze_group = false;                     // [Translated]
    bool use_sparse_updates = false;          // usesparseupdate
    
        // [Translated]
    float mobile_priority = 1.0f;             // mobilepriority (0-10)
    bool thermal_sensitive = true;                // [Translated]
    bool battery_sensitive = true;                // [Translated]
    
    ParameterGroupConfig(const std::string& name) 
        : group_name(name) {}
};

/**
 * @brief trainingresumeconfiguration
 */
struct WarmRestartConfig {
    bool enabled = false;                     // enablewarm restart
    int restart_period = 1000;                   // [Translated]
    float restart_mult = 2.0f;                   // [Translated]
    float min_lr_ratio = 0.1f;                   // [Translated]
    
        // [Translated]
    bool adaptive_restart = true;             // adaptiverestart
    float perforance_threshold = 0.001f;     // perforancethreshold
    int patience_steps = 100;                    // [Translated]
};

/**
 * [Documentation available in English]
 */
struct NumericalStabilityConfig {
    bool enable_overflow_detection = true;       // [Translated]
    bool enable_gradient_scaling = true;     // gradient scaling
    float initial_scale = 65536.0f;          // initialscaling factor
    float scale_growth_factor = 2.0f;            // [Translated]
    float scale_backoff_factor = 0.5f;           // [Translated]
    int scale_growth_interval = 2000;           // [Translated]
    
        // [Translated]
    float max_grad_norm = 10.0f;                // [Translated]
    float min_loss_scale = 1.0f;                // [Translated]
    float max_loss_scale = 65536.0f;            // [Translated]
};

/**
 * [Documentation available in English]
 */
struct PowerOptimizationConfig {
    bool enable_power_aware = true;              // [Translated]
    float target_power_consumption = 3.0f;       // [Translated]
    float max_power_consumption = 5.0f;          // [Translated]
    
    // powerschedulestrategy
    bool enable_dynamic_voltage_scaling = true;        // [Translated]
    bool enable_frequency_scaling = true;              // [Translated]
    bool enable_core_migration = true;                 // [Translated]
    
    // batteryoptimization
    float battery_critical_threshold = 0.15f;          // [Translated]
    float battery_low_threshold = 0.30f;           // batterylow batterythreshold (30%)
    float power_reduction_factor = 0.7f;               // [Translated]
};

/**
 * [Documentation available in English]
 */
struct AsyncOptimizerConfig {
    bool enable_async_updates = false;       // enableasyncupdate
    float importance_threshold = 0.1f;           // [Translated]
    int accumulation_window = 4;                 // [Translated]
    int max_async_ops = 2;                       // [Translated]
    
        // [Translated]
    bool use_gradient_magnitude = true;      // usegradientsize
    bool use_parameter_magnitude = true;     // useparametersize
    bool use_historical_importance = true;       // [Translated]
};

/**
 * @brief advancedcheckpointmanager
 */
class AdvancedCheckpointManager {
private:
    std::string base_checkpoint_dir_;
    int max_checkpoints_to_keep_;
    std::vector<std::string> saved_checkpoints_;
    
public:
    explicit AdvancedCheckpointManager(const std::string& dir, int max_keep = 5);
    
    /**
     * @brief savecompletetrainingstate
     */
    struct TrainingState {
        int global_step;
        float best_loss;
        float current_lr;
        std::unordered_map<std::string, float> group_lrs;
        OptimizerExtensionStats optimizer_stats;
        NumericalStabilityConfig stability_config;
        std::vector<float> loss_history;
        
        // mobilestate
        std::vector<float> power_history;
        std::vector<float> thermal_history;
        int total_thermal_events;
        int total_battery_events;
    };
    
    bool save_training_state(const TrainingState& state, const std::string& checkpoint_name);
    bool load_training_state(TrainingState& state, const std::string& checkpoint_name);
    
    /**
     * [Documentation available in English]
     */
    void auto_checkpoint(const TrainingState& state, float current_loss);
    void cleanup_old_checkpoints();
    std::string get_best_checkpoint() const;
};

/**
 * [Documentation available in English]
 */
class AdvancedLRScheduler : public MobileLRScheduler {
private:
    WarmRestartConfig restart_config_;
    std::vector<float> perforance_history_;
    int restart_count_ = 0;
    int last_restart_step_ = 0;
    
public:
    AdvancedLRScheduler(const LRSchedulerConfig& config, 
                       const WarmRestartConfig& restart_config,
                       OptimizerExtensionStats* stats = nullptr);
    
    /**
     * [Documentation available in English]
     */
    float compute_polynomial_decay_lr(int step, float power = 1.0f);
    
    /**
     * [Documentation available in English]
     */
    float compute_multistep_decay_lr(int step, const std::vector<int>& milestones, float gamma = 0.1f);
    
    /**
     * @brief Warm restart
     */
    float compute_warm_restart_lr(int step);
    
    /**
     * @brief adaptiverestartcheck
     */
    bool should_restart(float current_perforance);
    
    /**
     * @brief executerestart
     */
    void perfor_restart();
    
private:
    void update_perforance_history(float perforance);
};

/**
 * [Documentation available in English]
 */
class NumericalStabilityManager {
private:
    NumericalStabilityConfig config_;
    float current_loss_scale_ = 1.0f;
    int overflow_count_ = 0;
    int successful_steps_ = 0;
    std::queue<bool> recent_overflows_;
    
public:
    explicit NumericalStabilityManager(const NumericalStabilityConfig& config);
    
    /**
     * [Documentation available in English]
     */
    bool check_gradient_overflow(const std::vector<TensorPtr>& gradients);
    
    /**
     * [Documentation available in English]
     */
    void handle_overflow();
    
    /**
     * @brief scalegradient
     */
    void scale_gradients(std::vector<TensorPtr>& gradients);
    
    /**
     * [Documentation available in English]
     */
    void unscale_gradients(std::vector<TensorPtr>& gradients);
    
    /**
     * @brief updatelossscale
     */
    void update_loss_scale(bool overflow_detected);
    
    float get_current_loss_scale() const { return current_loss_scale_; }
    int get_overflow_count() const { return overflow_count_; }
};

/**
 * @brief poweroptimizationmanager
 */
class PowerOptimizationManager {
private:
    PowerOptimizationConfig config_;
    std::vector<float> power_history_;
    float current_power_consumption_ = 0.0f;
    std::chrono::steady_clock::time_point last_power_measurement_;
    
    // mobilesysteminterface
    bool is_plugged_in_ = false;
    float battery_level_ = 1.0f;
    float device_temperature_ = 30.0f;
    
public:
    explicit PowerOptimizationManager(const PowerOptimizationConfig& config);
    
    /**
     * @brief updatepowerstate
     */
    void update_power_state(float battery_level, bool plugged_in, float temperature);
    
    /**
     * [Documentation available in English]
     */
    float compute_power_adjustment_factor();
    
    /**
     * @brief applypoweroptimization
     */
    void apply_power_optimizations(float& learning_rate, int& batch_size);
    
    /**
     * [Documentation available in English]
     */
    float predict_power_consumption(float lr, int batch_size);
    
private:
    void log_power_event(const std::string& event);
    float get_battery_drain_rate();
};

/**
 * [Documentation available in English]
 */
class AsyncOptimizerUpdater {
private:
    AsyncOptimizerConfig config_;
    std::thread worker_thread_;
    std::queue<std::pair<size_t, TensorPtr>> async_gradient_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> should_stop_{false};
    
        // [Translated]
    std::unordered_map<size_t, float> parameter_importance_;
    std::unordered_map<size_t, std::vector<float>> importance_history_;
    
public:
    explicit AsyncOptimizerUpdater(const AsyncOptimizerConfig& config);
    ~AsyncOptimizerUpdater();
    
    /**
     * [Documentation available in English]
     */
    void submit_gradient_update(size_t param_id, const TensorPtr& gradient);
    
    /**
     * [Documentation available in English]
     */
    float compute_parameter_importance(size_t param_id, const TensorPtr& gradient);
    
    /**
     * @brief is notshouldsynchronousupdate
     */
    bool should_sync_update(size_t param_id, float importance);
    
private:
    void worker_loop();
    void process_async_updates();
    void update_importance_history(size_t param_id, float importance);
};

/**
 * @brief completeadvancedmoveoptimizer
 */
class CompleteMobileOptimizerAdvanced {
private:
    std::unique_ptr<CompleteMobileTrainingOptimizer> base_optimizer_;
    
        // [Translated]
    std::vector<std::unique_ptr<ParameterGroupConfig>> parameter_groups_;
    std::unique_ptr<AdvancedCheckpointManager> checkpoint_manager_;
    std::unique_ptr<AdvancedLRScheduler> advanced_lr_scheduler_;
    std::unique_ptr<NumericalStabilityManager> stability_manager_;
    std::unique_ptr<PowerOptimizationManager> power_manager_;
    std::unique_ptr<AsyncOptimizerUpdater> async_updater_;
    
    // advancedconfiguration
    WarmRestartConfig restart_config_;
    NumericalStabilityConfig stability_config_;
    PowerOptimizationConfig power_config_;
    AsyncOptimizerConfig async_config_;
    
    // advancedstatistics
    std::vector<float> training_loss_history_;
    std::vector<float> validation_loss_history_;
    float best_validation_loss_ = std::numeric_limits<float>::max();
    int steps_without_improvement_ = 0;
    
public:
    CompleteMobileOptimizerAdvanced(
        size_t available_memory_mb,
        const std::string& storage_path,
        const OptimizerHyperParams& opt_params,
        const GradientClippingConfig& clip_config,
        const LRSchedulerConfig& lr_config,
        const WarmRestartConfig& restart_config,
        const NumericalStabilityConfig& stability_config,
        const PowerOptimizationConfig& power_config,
        const AsyncOptimizerConfig& async_config
    );
    
    /**
     * [Documentation available in English]
     */
    void create_parameter_group(const ParameterGroupConfig& group_config);
    
    /**
     * @brief advancedtrainingsteps
     */
    bool advanced_training_step(
        const std::unordered_map<size_t, TensorPtr>& param_gradients,
        float training_loss,
        float validation_loss = -1.0f
    );
    
    /**
     * [Documentation available in English]
     */
    void auto_adjust_optimization_strategy();
    
    /**
     * @brief acquirecompleteadvancedstatistics
     */
    struct AdvancedTrainingStats {
        CompleteMobileTrainingOptimizer::CompleteTrainingStats base_stats;
        int restart_count;
        int overflow_count;
        float current_loss_scale;
        float average_power_consumption;
        float total_energy_consumed;
        std::vector<float> loss_history;
        float convergence_rate;
        float training_efficiency;
    };
    
    AdvancedTrainingStats get_advanced_stats() const;
    
    /**
     * @brief advancedcheckpoint
     */
    void save_advanced_checkpoint(const std::string& name, float current_loss);
    bool load_advanced_checkpoint(const std::string& name);
    
    /**
     * @brief trainingcompletedbackanalyze
     */
    void generate_training_analysis_report(const std::string& report_path);
    
private:
    void initialize_advanced_components();
    void update_training_metrics(float training_loss, float validation_loss);
    bool should_early_stop();
    void apply_parameter_group_optimizations();
};

/**
 * [Documentation available in English]
 */
std::unique_ptr<CompleteMobileOptimizerAdvanced> create_advanced_mobile_optimizer(
    size_t available_memory_mb,
    const std::string& storage_path,
    const std::string& checkpoint_dir,
    bool enable_all_advanced_features = true
);

} // namespace memory  
} // namespace ops
