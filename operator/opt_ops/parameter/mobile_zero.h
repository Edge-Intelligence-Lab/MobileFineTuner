/**
 * @file mobile_zero.h
 * @brief Mobile-optimized ZeRO (Zero Redundancy Optimizer) implementsation
 * 
 * This file implementss a mobile-specific version of Microsoft DeepSpeed's ZeRO technology
 * adapted for single-device training on resource-constrained mobile platfors.
 * 
 * Key adaptations from DeepSpeed ZeRO:
 * - Single-device parameter partitioning (temporal partitioning instead of spatial)
 * - Layer-wise parameter loading/unloading
 * - Memory-aware gradient accumulation
 * - Mobile-specific memory hierarchy optimization
 */

#pragma once

#include "mobile_param_manager.h"
#include "../core/tensor.h"
#include "../core/ops.h"
#include "../optim/adam.h"
#include <unordered_set>
#include <map>

namespace ops {
namespace memory {

/**
 * @brief ZeRO stage configuration for mobile devices
 */
enum class MobileZeROStage {
    DISABLED = 0,           // No ZeRO optimizations
    OPTIMIZER_STATES = 1,   // Partition optimizer states only
    GRADIENTS = 2,          // Partition gradients + optimizer states  
    PARAMETERS = 3          // Partition parameters + gradients + optimizer states
};

/**
 * @brief Parameter group for layer-wise management
 */
struct ParameterGroup {
    std::string group_name;
    std::vector<size_t> param_ids;
    std::vector<std::string> param_names;
    size_t total_size_bytes;
    bool is_active;              // Currently loaded in memory
    bool requires_grad;          // Needs gradients
    int access_priority;         // Higher = more important to keep in memory
    
    ParameterGroup(const std::string& name) 
        : group_name(name), total_size_bytes(0), is_active(false), requires_grad(true), access_priority(0) {}
};

/**
 * @brief Gradient accumulation buffer for memory efficiency
 */
class GradientAccumulator {
private:
    std::unordered_map<size_t, TensorPtr> accumulated_grads_;
    std::unordered_map<size_t, int> accumulation_counts_;
    int target_accumulation_steps_;
    float gradient_scale_;

public:
    explicit GradientAccumulator(int accumulation_steps = 1);
    
    void accumulate_gradient(size_t param_id, const TensorPtr& grad);
    TensorPtr get_accumulated_gradient(size_t param_id);
    bool is_ready_for_update(size_t param_id) const;
    void clear_accumulated_gradient(size_t param_id);
    void reset();
    
    void set_gradient_scale(float scale) { gradient_scale_ = scale; }
    float get_gradient_scale() const { return gradient_scale_; }
};

/**
 * @brief Mobile-optimized ZeRO optimizer
 * 
 * This class implementss a ZeRO-inspired optimization strategy adapted for mobile devices:
 * 1. Temporal parameter partitioning - load/unload parameters as needed
 * 2. Layer-wise gradient management - process gradients layer by layer
 * 3. Optimizer state compression - use lower precision for momentum/variance
 * 4. Memory-aware scheduling - prioritize parameters based on memory constraints
 */
class MobileZeROOptimizer {
private:
    std::unique_ptr<MobileParameterManager> param_manager_;
    std::unique_ptr<GradientAccumulator> grad_accumulator_;
    MobileZeROStage zero_stage_;
    
    // Parameter organization
    std::vector<std::unique_ptr<ParameterGroup>> parameter_groups_;
    std::unordered_map<size_t, size_t> param_to_group_map_;
    std::unordered_map<std::string, size_t> name_to_group_map_;
    
    // Optimizer state management
    std::unordered_map<size_t, TensorPtr> optimizer_states_m_;  // Momentum (Adam)
    std::unordered_map<size_t, TensorPtr> optimizer_states_v_;  // Variance (Adam)
    std::unordered_map<size_t, float> learning_rates_;
    
    // Training configuration
    float base_learning_rate_;
    float weight_decay_;
    float beta1_, beta2_;
    float epsilon_;
    int gradient_accumulation_steps_;
    int current_step_;
    
    // Memory management
    size_t max_active_parameters_;
    size_t current_memory_usage_;
    std::priority_queue<std::pair<int, size_t>> eviction_queue_; // priority, group_id
    
    // Statistics
    struct ZeROStats {
        size_t total_parameters;
        size_t active_parameters;
        size_t memory_saved_bytes;
        double average_load_time_ms;
        double average_compute_time_ms;
        size_t parameter_swaps;
    } stats_;

public:
    /**
     * @brief Construct mobile ZeRO optimizer
     * @param param_manager Parameter manager for memory optimization
     * @param stage ZeRO stage to use
     * @param config Optimizer configuration
     */
    explicit MobileZeROOptimizer(
        std::unique_ptr<MobileParameterManager> param_manager,
        MobileZeROStage stage = MobileZeROStage::PARAMETERS,
        float learning_rate = 1e-3,
        float weight_decay = 0.01f,
        int grad_accumulation_steps = 1
    );
    
    ~MobileZeROOptimizer() = default;
    
    /**
     * @brief Register model parameters with the optimizer
     * @param name Parameter name (e.g., "transforer.layer.0.weight")  
     * @param tensor Parameter tensor
     * @param group_name Group name (e.g., "transforer.layer.0")
     * @param requires_grad Whether parameter needs gradients
     * @return Parameter ID
     */
    size_t register_parameter(
        const std::string& name,
        const TensorPtr& tensor, 
        const std::string& group_name = "default",
        bool requires_grad = true
    );
    
    /**
     * @brief Get parameter for forward pass computation
     * @param param_id Parameter ID
     * @return Parameter tensor (automatically loaded)
     */
    TensorPtr get_parameter(size_t param_id);
    TensorPtr get_parameter(const std::string& name);
    
    /**
     * @brief Perfor backward pass and gradient accumulation
     * @param param_id Parameter ID
     * @param gradient Computed gradient
     */
    void backward(size_t param_id, const TensorPtr& gradient);
    
    /**
     * @brief Perfor optimizer step (parameter update)
     * @param param_ids Parameters to update (empty = all active parameters)
     */
    void step(const std::vector<size_t>& param_ids = {});
    
    /**
     * @brief Zero gradients for specified parameters
     * @param param_ids Parameters to zero (empty = all parameters)
     */
    void zero_grad(const std::vector<size_t>& param_ids = {});
    
    /**
     * @brief Load specific parameter group into memory
     * @param group_name Name of parameter group
     */
    void load_parameter_group(const std::string& group_name);
    
    /**
     * @brief Unload parameter group from memory  
     * @param group_name Name of parameter group
     */
    void unload_parameter_group(const std::string& group_name);
    
    /**
     * @brief Optimize memory layout for inference
     */
    void optimize_for_inference();
    
    /**
     * @brief Optimize memory layout for training
     */
    void optimize_for_training();
    
    /**
     * @brief Set learning rate schedule
     * @param param_id Parameter ID (or SIZE_MAX for all parameters)
     * @param lr New learning rate
     */
    void set_learning_rate(size_t param_id, float lr);
    void set_learning_rate(float lr); // Set for all parameters
    
    /**
     * @brief Enable/disable gradient accumulation
     * @param steps Number of steps to accumulate (1 = no accumulation)
     */
    void set_gradient_accumulation_steps(int steps);
    
    /**
     * @brief Save optimizer state to checkpoint
     * @param checkpoint_path Path to save checkpoint
     */
    void save_checkpoint(const std::string& checkpoint_path);
    
    /**
     * @brief Load optimizer state from checkpoint
     * @param checkpoint_path Path to load checkpoint
     */
    void load_checkpoint(const std::string& checkpoint_path);
    
    /**
     * @brief Get current optimizer statistics
     */
    const ZeROStats& get_stats() const { return stats_; }
    
    /**
     * @brief Get memory statistics
     */
    MemoryStats get_memory_stats() const;

private:
    // Internal parameter management
    void initialize_optimizer_states(size_t param_id, const TensorPtr& param);
    void update_parameter_with_adam(size_t param_id, const TensorPtr& gradient);
    void manage_memory_pressure();
    void evict_least_recently_used_group();
    void update_access_priorities();
    
    // Group management
    size_t create_parameter_group(const std::string& group_name);
    void add_parameter_to_group(size_t param_id, size_t group_id);
    bool is_group_active(size_t group_id) const;
    void activate_group(size_t group_id);
    void deactivate_group(size_t group_id);
    
    // Optimizer state compression
    void compress_optimizer_states();
    void decompress_optimizer_states();
    
    // Statistics tracking
    void update_training_stats();
    void log_memory_usage();
};

/**
 * @brief RAII helper for automatic parameter group lifecycle management
 */
class ScopedParameterGroup {
private:
    MobileZeROOptimizer* optimizer_;
    std::string group_name_;
    bool was_loaded_;

public:
    ScopedParameterGroup(MobileZeROOptimizer* optimizer, const std::string& group_name);
    ~ScopedParameterGroup();
    
    // Non-copyable, movable
    ScopedParameterGroup(const ScopedParameterGroup&) = delete;
    ScopedParameterGroup& operator=(const ScopedParameterGroup&) = delete;
    ScopedParameterGroup(ScopedParameterGroup&& other) noexcept;
    ScopedParameterGroup& operator=(ScopedParameterGroup&& other) noexcept;
};

/**
 * @brief Memory-efficient training loop helper
 * 
 * This class providess utilities to coordinate forward/backward passes
 * with parameter loading/unloading for maximum memory efficiency.
 */
class MobileTrainingCoordinator {
private:
    MobileZeROOptimizer* optimizer_;
    std::vector<std::string> layer_execution_order_;
    size_t current_layer_index_;
    bool is_forward_pass_;

public:
    explicit MobileTrainingCoordinator(MobileZeROOptimizer* optimizer);
    
    /**
     * @brief Define layer execution order for training
     * @param layer_names Ordered list of layer group names
     */
    void set_layer_order(const std::vector<std::string>& layer_names);
    
    /**
     * @brief Begin forward pass
     */
    void begin_forward_pass();
    
    /**
     * @brief Begin backward pass
     */
    void begin_backward_pass();
    
    /**
     * @brief Move to next layer in execution order
     * @return True if more layers remain, false if finished
     */
    bool next_layer();
    
    /**
     * @brief Get current layer name
     */
    std::string get_current_layer() const;
    
    /**
     * @brief Execute training step with automatic memory management
     * @param forward_fn Forward pass function for current layer
     * @param backward_fn Backward pass function for current layer
     */
    void execute_training_step(
        const std::function<TensorPtr(const std::string&)>& forward_fn,
        const std::function<void(const std::string&, const TensorPtr&)>& backward_fn
    );
};

/**
 * @brief Factory functions for creating mobile ZeRO optimizers
 */
std::unique_ptr<MobileZeROOptimizer> create_mobile_zero_optimizer(
    size_t available_memory_mb = 1024,
    MobileZeROStage stage = MobileZeROStage::PARAMETERS,
    float learning_rate = 1e-3,
    int grad_accumulation_steps = 1
);

/**
 * @brief Helper function to estimate memory requirements
 * @param model_params_mb Size of model parameters in MB
 * @param stage ZeRO stage to use
 * @return Estimated memory usage in MB
 */
size_t estimate_memory_usage(
    size_t model_params_mb,
    MobileZeROStage stage,
    bool enable_gradient_accumulation = true
);

} // namespace memory
} // namespace ops

/**
 * @brief Usage example and integration guide
 * 
 * Example usage in training loop:
 * 
 * ```cpp
 * // Create mobile ZeRO optimizer
 * auto optimizer = create_mobile_zero_optimizer(
 *     1024,  // 1GB available memory
 *     MobileZeROStage::PARAMETERS,
 *     1e-4   // learning rate
 * );
 * 
 * // Register model parameters
 * for (auto& [name, param] : model->named_parameters()) {
 *     std::string layer_name = extract_layer_name(name);
 *     optimizer->register_parameter(name, param, layer_name);
 * }
 * 
 * // Training loop with automatic memory management
 * MobileTrainingCoordinator coordinator(optimizer.get());
 * coordinator.set_layer_order({"embed", "layer.0", "layer.1", ..., "head"});
 * 
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     coordinator.execute_training_step(
 *         [&](const std::string& layer) -> TensorPtr {
 *             // Forward pass for layer
 *             return forward_layer(layer, input);
 *         },
 *         [&](const std::string& layer, const TensorPtr& grad) {
 *             // Backward pass for layer  
 *             backward_layer(layer, grad);
 *         }
 *     );
 * }
 * ```
 */
