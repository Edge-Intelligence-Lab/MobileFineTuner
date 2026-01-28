/**
 * @file activation_checkpointer.h
 * @brief Mobile-optimized activation checkpointing system
 * 
 * This component implementss gradient checkpointing specifically optimized for mobile
 * training scenarios. Unlike DeepSpeed's checkpointing which targets data center GPUs,
 * this system considers mobile constraints like limited memory, battery life, and
 * thermal management.
 * 
 * Key Features:
 * - Adaptive checkpointing based on mobile system state
 * - Smart checkpoint selection using layer importance analysis
 * - Mobile-aware recomputation scheduling
 * - Integration with mobile power and thermal management
 */

#pragma once

#include "../core/tensor.h"
#include "mobile_activation_manager.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <mutex>
#include <queue>
#include <chrono>
#include <future>
#include <any>

namespace ops {
namespace memory {

using ops::TensorPtr;
using ops::Tensor;

/**
 * @brief Checkpoint importance levels for mobile optimization
 */
enum class CheckpointImportance {
    LOW = 0,        // Can be dropped under pressure
    NORMAL = 1,     // Standard checkpoint
    HIGH = 2,       // Important for efficiency
    CRITICAL = 3    // Never drop (e.g., gradient accumulation boundaries)
};

/**
 * @brief Recomputation cost levels
 */
enum class RecomputationCost {
    VERY_LOW = 1,   // Simple operations (add, mul)
    LOW = 2,        // Basic activations (ReLU, sigmoid)
    MEDIUM = 3,     // Linear layers, normalization
    HIGH = 4,       // Attention computation
    VERY_HIGH = 5   // Complex operations (softmax over large dims)
};

/**
 * @brief Checkpoint metadata for mobile optimization
 */
struct CheckpointMetadata {
    size_t checkpoint_id;
    std::string layer_name;
    size_t layer_id;
    CheckpointImportance importance;
    RecomputationCost recomputation_cost;
    
    // Timing inforation
    std::chrono::steady_clock::time_point creation_time;
    std::chrono::steady_clock::time_point last_access_time;
    std::chrono::steady_clock::time_point expected_use_time;
    
    // Memory inforation
    size_t memory_footprint;
    size_t recomputation_memory_needed;
    
    // Mobile-specific metadata
    bool is_ui_critical;              // Critical for UI responsiveness
    bool is_battery_sensitive;        // Expensive to recompute on battery
    bool is_thermal_sensitive;        // Generates heat when recomputed
    float power_consumption_estimate; // Estimated power for recomputation (mW)
    
    // Graph topology inforation
    std::vector<size_t> dependent_checkpoints;   // Checkpoints this depends on
    std::vector<size_t> dependent_activations;   // Activations that depend on this
    int depth_from_input;                        // Distance from model input
    int depth_to_output;                         // Distance to model output
    
    CheckpointMetadata(size_t id, const std::string& name, size_t layer)
        : checkpoint_id(id), layer_name(name), layer_id(layer),
          importance(CheckpointImportance::NORMAL), recomputation_cost(RecomputationCost::MEDIUM),
          creation_time(std::chrono::steady_clock::now()), last_access_time(creation_time),
          memory_footprint(0), recomputation_memory_needed(0),
          is_ui_critical(false), is_battery_sensitive(false), is_thermal_sensitive(false),
          power_consumption_estimate(0.0f), depth_from_input(0), depth_to_output(0) {}
};

/**
 * @brief Recomputation function signature
 */
using RecomputationFunction = std::function<TensorPtr(const std::vector<TensorPtr>&)>;

/**
 * @brief Checkpoint entry storing tensor and recomputation info
 */
struct CheckpointEntry {
    TensorPtr activation;
    std::unique_ptr<CheckpointMetadata> metadata;
    RecomputationFunction recomputation_fn;
    std::vector<size_t> input_checkpoint_ids;  // Input checkpoints for recomputation
    
    CheckpointEntry(const TensorPtr& act, size_t id, const std::string& name, size_t layer)
        : activation(act), metadata(std::make_unique<CheckpointMetadata>(id, name, layer)) {}
};

/**
 * @brief Configuration for mobile checkpointing
 */
struct CheckpointConfig {
    // Basic checkpointing parameters
    CheckpointStrategy strategy = CheckpointStrategy::MOBILE_SMART;
    int unifor_checkpoint_interval = 4;        // Unifor checkpointing interval
    float memory_pressure_threshold = 0.8f;     // Start aggressive checkpointing at 80%
    
    // Mobile-specific configuration
    bool enable_adaptive_checkpointing = true;   // Enable adaptive checkpointing
    bool enable_importance_based_selection = true; // Use importance for checkpoint selection
    bool enable_mobile_power_awareness = true;   // Consider power consumption
    bool enable_thermal_awareness = true;        // Consider thermal state
    
    // Recomputation configuration
    bool enable_smart_recomputation = true;     // Enable smart recomputation scheduling
    int max_recomputation_threads = 2;          // Max concurrent recomputation threads
    float max_recomputation_memory_ratio = 0.3f; // Max memory to use for recomputation
    bool enable_async_recomputation = true;     // Enable async recomputation prefetch
    
    // Quality vs efficiency tradeoffs
    float memory_vs_compute_tradeoff = 0.7f;    // 0.0=minimize memory, 1.0=minimize compute
    float accuracy_vs_efficiency_tradeoff = 0.8f; // 0.0=max efficiency, 1.0=max accuracy
    
    // Mobile system integration
    bool respond_to_memory_pressure = true;     // Respond to system memory pressure
    bool respond_to_battery_state = true;       // Respond to battery state
    bool respond_to_thermal_state = true;       // Respond to thermal state
    bool maintain_ui_responsiveness = true;     // Never block UI thread
    
    // Advanced optimization
    bool enable_checkpoint_prefetch = true;     // Prefetch likely-needed checkpoints
    bool enable_checkpoint_compression = true;  // Compress checkpoints when needed
    bool enable_multi_level_checkpointing = true; // Multiple checkpoint importance levels
    
    // Perforance monitoring and profiling
    bool enable_checkpoint_profiling = false;   // Enable detailed profiling
    bool log_checkpoint_events = false;         // Log checkpointing events
    std::string profiling_output_path = "./checkpoint_profile.json";
};

/**
 * @brief Statistics for checkpoint management
 */
struct CheckpointStats {
    size_t total_checkpoints_created;
    size_t total_checkpoints_dropped;
    size_t total_recomputations;
    size_t cache_hits;
    size_t cache_misses;
    
    double total_checkpointing_time_ms;
    double total_recomputation_time_ms;
    double average_recomputation_time_ms;
    
    size_t memory_saved_by_checkpointing;
    size_t additional_computation_cost;
    
    // Mobile-specific stats
    size_t battery_aware_optimizations;
    size_t thermal_aware_optimizations;
    size_t ui_responsiveness_protections;
    size_t memory_pressure_adaptations;
};

/**
 * @brief Mobile-optimized activation checkpointer
 */
class ActivationCheckpointer {
private:
    CheckpointConfig config_;
    std::unordered_map<size_t, std::unique_ptr<CheckpointEntry>> checkpoints_;
    std::queue<size_t> checkpoint_creation_order_;  // Order of checkpoint creation
    
    // Mobile state monitoring
    std::atomic<float> current_memory_pressure_;
    std::atomic<int> current_battery_level_;
    std::atomic<float> current_temperature_;
    std::atomic<bool> is_app_foreground_;
    
    // Threading and synchronization
    mutable std::mutex checkpoint_mutex_;
    std::thread recomputation_worker_;
    std::queue<std::pair<size_t, std::promise<TensorPtr>>> recomputation_queue_;
    std::mutex recomputation_queue_mutex_;
    std::condition_variable recomputation_cv_;
    std::atomic<bool> worker_running_;
    
    // Statistics
    CheckpointStats stats_;
    mutable std::mutex stats_mutex_;
    
    // Adaptive checkpointing state
    std::vector<float> layer_importance_scores_;
    std::vector<float> layer_recomputation_costs_;
    std::deque<std::pair<size_t, std::chrono::steady_clock::time_point>> recent_recomputation_times_;

public:
    explicit ActivationCheckpointer(const CheckpointConfig& config);
    ~ActivationCheckpointer();
    
    /**
     * @brief Create a checkpoint for an activation
     * @param activation The activation tensor to checkpoint
     * @param layer_name Name of the layer producing this activation
     * @param layer_id Numeric layer ID
     * @param importance Importance level of this checkpoint
     * @param recomputation_fn Function to recompute this activation
     * @param input_checkpoint_ids Checkpoints needed for recomputation
     * @return Checkpoint ID
     */
    size_t create_checkpoint(
        const TensorPtr& activation,
        const std::string& layer_name,
        size_t layer_id,
        CheckpointImportance importance = CheckpointImportance::NORMAL,
        RecomputationFunction recomputation_fn = nullptr,
        const std::vector<size_t>& input_checkpoint_ids = {}
    );
    
    /**
     * @brief Get an activation from checkpoint (may trigger recomputation)
     * @param checkpoint_id Checkpoint ID
     * @param async_recompute Whether to allow async recomputation
     * @return The activation tensor
     */
    TensorPtr get_checkpoint_activation(size_t checkpoint_id, bool async_recompute = true);
    
    /**
     * @brief Remove checkpoints before a given checkpoint ID
     * @param before_checkpoint_id Remove checkpoints created before this ID
     * @param preserve_important Whether to preserve important checkpoints
     */
    void clear_checkpoints_before(size_t before_checkpoint_id, bool preserve_important = true);
    
    /**
     * @brief Optimize checkpoint selection based on current system state
     * This analyzes the current checkpoint graph and drops less important
     * checkpoints to free memory while minimizing recomputation cost
     */
    void optimize_checkpoint_selection();
    
    /**
     * @brief Update mobile system state for adaptive checkpointing
     * @param memory_pressure Current memory pressure (0.0-1.0)
     * @param battery_level Current battery level (0-100)
     * @param temperature Current device temperature
     * @param is_foreground Whether app is in foreground
     */
    void update_mobile_state(float memory_pressure, int battery_level, 
                           float temperature, bool is_foreground);
    
    /**
     * @brief Set layer importance scores for smart checkpointing
     * @param importance_scores Importance score for each layer (0.0-1.0)
     */
    void set_layer_importance_scores(const std::vector<float>& importance_scores);
    
    /**
     * @brief Set layer recomputation costs for smart checkpointing
     * @param recomputation_costs Relative recomputation cost for each layer
     */
    void set_layer_recomputation_costs(const std::vector<float>& recomputation_costs);
    
    /**
     * @brief Get current checkpoint statistics
     * @return Current checkpoint statistics
     */
    CheckpointStats get_checkpoint_stats() const;
    
    /**
     * @brief Configure checkpointing parameters
     * @param config New checkpoint configuration
     */
    void configure_checkpointing(const CheckpointConfig& config);
    
    /**
     * @brief Export detailed profiling report
     * @param report_path Path to save profiling report
     */
    void export_profiling_report(const std::string& report_path) const;

private:
    // Checkpoint selection algorithms
    std::vector<size_t> select_checkpoints_unifor(size_t layer_count);
    std::vector<size_t> select_checkpoints_adaptive(size_t layer_count);
    std::vector<size_t> select_checkpoints_mobile_smart(size_t layer_count);
    std::vector<size_t> select_checkpoints_importance_based(size_t layer_count);
    
    // Checkpoint dropping algorithms
    std::vector<size_t> select_checkpoints_to_drop(size_t memory_needed);
    float calculate_checkpoint_dropping_cost(size_t checkpoint_id);
    bool can_drop_checkpoint_safely(size_t checkpoint_id);
    
    // Recomputation management
    void recomputation_worker_loop();
    TensorPtr recompute_activation_sync(size_t checkpoint_id);
    void schedule_async_recomputation(size_t checkpoint_id);
    void update_recomputation_costs();
    
    // Mobile optimization methods
    void adapt_checkpointing_for_memory_pressure();
    void adapt_checkpointing_for_battery_state();
    void adapt_checkpointing_for_thermal_state();
    bool should_defer_recomputation_for_ui();
    
    // Analysis and profiling methods
    void analyze_checkpoint_graph();
    void update_layer_importance_scores();
    float calculate_layer_importance(size_t layer_id);
    float estimate_recomputation_power_cost(size_t checkpoint_id);
    
    // Utility methods
    bool is_checkpoint_valid(size_t checkpoint_id) const;
    void cleanup_expired_checkpoints();
    void update_checkpoint_metadata(size_t checkpoint_id);
    void log_checkpoint_event(const std::string& event, size_t checkpoint_id, double duration_ms = 0.0);
};

/**
 * @brief Smart checkpoint selection utilities for mobile
 */
namespace mobile_checkpoint_utils {
    
    /**
     * @brief Calculate optimal checkpoint interval based on model architecture
     * @param layer_count Total number of layers
     * @param memory_budget Available memory budget
     * @param compute_budget Available compute budget
     * @return Recommended checkpoint interval
     */
    int calculate_optimal_checkpoint_interval(int layer_count, size_t memory_budget, size_t compute_budget);
    
    /**
     * @brief Analyze model architecture to determine checkpoint importance
     * @param layer_types Types of layers in the model
     * @param layer_sizes Memory footprint of each layer
     * @return Importance scores for each layer
     */
    std::vector<float> analyze_layer_importance(const std::vector<std::string>& layer_types,
                                              const std::vector<size_t>& layer_sizes);
    
    /**
     * @brief Estimate recomputation cost for different layer types
     * @param layer_type Type of layer (e.g., "attention", "mlp", "norm")
     * @param input_size Input tensor size
     * @return Estimated recomputation cost
     */
    RecomputationCost estimate_layer_recomputation_cost(const std::string& layer_type, size_t input_size);
    
    /**
     * @brief Create optimal checkpointing schedule for mobile training
     * @param model_config Model architecture configuration
     * @param mobile_config Mobile system configuration
     * @return Optimized checkpointing schedule
     */
    struct CheckpointingSchedule {
        std::vector<int> checkpoint_layers;
        std::vector<CheckpointImportance> checkpoint_importance;
        float expected_memory_savings;
        float expected_compute_overhead;
    };
    CheckpointingSchedule create_mobile_checkpointing_schedule(
        const std::unordered_map<std::string, std::any>& model_config,
        const MobileActivationConfig& mobile_config
    );
}

} // namespace memory
} // namespace ops
