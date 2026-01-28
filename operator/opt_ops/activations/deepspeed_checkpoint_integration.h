/**
 * @file deepspeed_checkpoint_integration.h
 * [Documentation in English - see separate docs]
 * 
 * [Documentation in English - see separate docs]
 * [Documentation in English - see separate docs]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 */

#pragma once

#include "../core/tensor.h"
#include "../core/ops.h"
#include "activation_checkpointer.h"
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <future>
#include <atomic>
#include <stack>

namespace ops {
namespace memory {

using ops::TensorPtr;
using ops::GradFn;

/**
 * [Documentation available in English]
 */
using DeepSpeedForwardFunction = std::function<std::vector<TensorPtr>(const std::vector<TensorPtr>&)>;

/**
 * [Documentation available in English]
 */
class CheckpointContext {
private:
    std::vector<TensorPtr> saved_tensors_;
    DeepSpeedForwardFunction forward_fn_;
    size_t checkpoint_id_;
    bool has_random_state_;
    std::vector<uint8_t> random_state_;
    
public:
    CheckpointContext(size_t checkpoint_id, DeepSpeedForwardFunction forward_fn)
        : forward_fn_(forward_fn), checkpoint_id_(checkpoint_id), has_random_state_(false) {}
    
    void save_for_backward(const std::vector<TensorPtr>& tensors);
    std::vector<TensorPtr> get_saved_tensors() const { return saved_tensors_; }
    void save_random_state();
    void restore_random_state();
    
    DeepSpeedForwardFunction get_forward_function() const { return forward_fn_; }
    size_t get_checkpoint_id() const { return checkpoint_id_; }
};

/**
 * [Documentation available in English]
 */
class DeepSpeedCheckpointFunction {
private:
    static thread_local std::stack<std::shared_ptr<CheckpointContext>> context_stack_;
    
public:
    /**
     * @brief forwardpropagate - savecheckpointinforation
     */
    static std::vector<TensorPtr> forward(
        DeepSpeedForwardFunction forward_fn,
        const std::vector<TensorPtr>& inputs,
        bool preserve_random_state = true
    );
    
    /**
     * [Documentation available in English]
     */
    static std::vector<TensorPtr> backward(
        const std::shared_ptr<CheckpointContext>& ctx,
        const std::vector<TensorPtr>& grad_outputs
    );
    
    /**
     * [Documentation available in English]
     */
    static std::shared_ptr<CheckpointContext> get_current_context();
};

/**
 * [Documentation available in English]
 */
class MobileRecomputationScheduler {
private:
    struct RecomputationTask {
        size_t checkpoint_id;
        std::shared_ptr<CheckpointContext> context;
        std::vector<TensorPtr> grad_outputs;
        std::promise<std::vector<TensorPtr>> promise;
        std::chrono::steady_clock::time_point deadline;
        int priority;
        
        RecomputationTask(size_t id, std::shared_ptr<CheckpointContext> ctx, 
                         const std::vector<TensorPtr>& grads, int prio)
            : checkpoint_id(id), context(ctx), grad_outputs(grads), priority(prio) {}
    };
    
    std::priority_queue<std::unique_ptr<RecomputationTask>, 
                       std::vector<std::unique_ptr<RecomputationTask>>,
                       std::function<bool(const std::unique_ptr<RecomputationTask>&, 
                                        const std::unique_ptr<RecomputationTask>&)>> task_queue_;
    
    std::vector<std::thread> worker_threads_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_flag_;
    
    // mobilestatemonitor
    std::atomic<float> current_battery_level_;
    std::atomic<float> current_temperature_;
    std::atomic<bool> is_ui_thread_blocked_;
    std::atomic<size_t> active_recomputations_;
    
public:
    MobileRecomputationScheduler(int num_threads = 2);
    ~MobileRecomputationScheduler();
    
    /**
     * [Documentation available in English]
     */
    std::future<std::vector<TensorPtr>> schedule_recomputation(
        size_t checkpoint_id,
        std::shared_ptr<CheckpointContext> context,
        const std::vector<TensorPtr>& grad_outputs,
        int priority = 5
    );
    
    /**
     * [Documentation available in English]
     */
    std::vector<TensorPtr> recompute_sync(
        std::shared_ptr<CheckpointContext> context,
        const std::vector<TensorPtr>& grad_outputs
    );
    
    /**
     * @brief updatemobilesystemstate
     */
    void update_mobile_state(float battery_level, float temperature, bool ui_blocked);
    
    /**
     * @brief acquireschedulestatisticsinforation
     */
    struct SchedulerStats {
        size_t total_recomputations;
        size_t async_recomputations;
        size_t sync_recomputations;
        double average_recomputation_time_ms;
        size_t battery_optimized_tasks;
        size_t thermal_deferred_tasks;
        size_t ui_responsive_tasks;
    };
    SchedulerStats get_scheduler_stats() const;
    
private:
    void worker_loop();
    int calculate_task_priority(const RecomputationTask& task);
    bool should_defer_recomputation_for_mobile();
    void optimize_recomputation_for_battery();
    void handle_thermal_throttling();
};

/**
 * [Documentation available in English]
 */
class DeepSpeedActivationCheckpointing {
private:
    std::unique_ptr<ActivationCheckpointer> base_checkpointer_;
    std::unique_ptr<MobileRecomputationScheduler> recomputation_scheduler_;
    
    // Checkpointmanage
    std::unordered_map<size_t, std::shared_ptr<CheckpointContext>> checkpoint_contexts_;
    std::atomic<size_t> next_checkpoint_id_;
    mutable std::mutex checkpoint_mutex_;
    
    // mobileoptimizationparameter
    float memory_vs_compute_tradeoff_;      // [Translated]
    bool enable_smart_checkpointing_;
    bool enable_mobile_optimizations_;
    
    // statisticsinfo
    struct CheckpointingStats {
        size_t total_checkpoints;
        size_t active_checkpoints;
        size_t total_recomputations;
        size_t memory_saved_bytes;
        double total_recomputation_time_ms;
        size_t mobile_optimized_checkpoints;
    };
    mutable CheckpointingStats stats_;
    mutable std::mutex stats_mutex_;
    
public:
    DeepSpeedActivationCheckpointing(const CheckpointConfig& config = CheckpointConfig{});
    ~DeepSpeedActivationCheckpointing();
    
    /**
     * [Documentation available in English]
     * 
     * usemethod：
     * auto output = checkpointing.checkpoint(forward_function, inputs);
     * 
     * equivalentinPyTorch:
     * output = torch.utils.checkpoint.checkpoint(forward_function, *inputs)
     */
    std::vector<TensorPtr> checkpoint(
        DeepSpeedForwardFunction forward_fn,
        const std::vector<TensorPtr>& inputs,
        bool preserve_random_state = true
    );
    
    /**
     * [Documentation available in English]
     * 
     * similar toDeepSpeedcheckpoint_sequential
     */
    std::vector<TensorPtr> checkpoint_sequential(
        const std::vector<DeepSpeedForwardFunction>& functions,
        const std::vector<TensorPtr>& inputs,
        int segments = -1          // [Translated]
    );
    
    /**
     * [Documentation in English - see separate docs]
     */
    std::vector<TensorPtr> smart_checkpoint(
        const std::vector<DeepSpeedForwardFunction>& functions,
        const std::vector<TensorPtr>& inputs,
        float memory_budget_mb = 512.0f
    );
    
    /**
     * [Documentation in English - see separate docs]
     */
    std::vector<TensorPtr> mobile_aware_checkpoint(
        DeepSpeedForwardFunction forward_fn,
        const std::vector<TensorPtr>& inputs,
        MobileActivationState system_state = MobileActivationState::NORMAL
    );
    
    /**
     * @brief configurationcheckpointingparameter
     */
    void configure(
        float memory_vs_compute_tradeoff = 0.7f,          // [Translated]
        bool enable_smart_checkpointing = true,
        bool enable_mobile_optimizations = true
    );
    
    /**
     * @brief cleanupexpiredcheckpoint
     */
    void cleanup_expired_checkpoints(size_t before_checkpoint_id);
    
    /**
     * @brief acquirememorystatisticsinforation
     */
    CheckpointingStats get_checkpointing_stats() const;
    
    /**
     * [Documentation available in English]
     */
    void optimize_memory_usage(size_t target_memory_mb);
    
    /**
     * @brief settingsmobilesystemstatecallback
     */
    void set_mobile_state_callback(std::function<MobileActivationState()> callback);

private:
    // Checkpointselectalgorithm
    std::vector<int> calculate_optimal_checkpoint_points(
        const std::vector<DeepSpeedForwardFunction>& functions,
        const std::vector<TensorPtr>& inputs,
        float memory_budget_mb
    );
    
    // mobileoptimizationalgorithm
    bool should_checkpoint_for_mobile_state(MobileActivationState state, size_t memory_footprint);
    RecomputationCost estimate_recomputation_cost_mobile(const DeepSpeedForwardFunction& fn);
    
    // memoryanalyze
    size_t estimate_function_memory_footprint(
        const DeepSpeedForwardFunction& fn,
        const std::vector<TensorPtr>& inputs
    );
    
    // statisticsupdate
    void update_checkpointing_stats(size_t checkpoint_id, size_t memory_saved, double recomputation_time);
};

/**
 * @brief globalcheckpointmanager
 * 
 * [Documentation available in English]
 */
class GlobalCheckpointManager {
private:
    static std::unique_ptr<DeepSpeedActivationCheckpointing> instance_;
    static std::once_flag init_flag_;
    
public:
    static DeepSpeedActivationCheckpointing& get_instance();
    static void initialize(const CheckpointConfig& config = CheckpointConfig{});
    static void shutdown();
};

/**
 * [Documentation available in English]
 */
namespace checkpoint_utils {
    
    /**
     * [Documentation available in English]
     */
    inline std::vector<TensorPtr> checkpoint(
        DeepSpeedForwardFunction forward_fn,
        const std::vector<TensorPtr>& inputs,
        bool preserve_random_state = true
    ) {
        return GlobalCheckpointManager::get_instance().checkpoint(
            forward_fn, inputs, preserve_random_state
        );
    }
    
    /**
     * [Documentation available in English]
     */
    inline std::vector<TensorPtr> checkpoint_sequential(
        const std::vector<DeepSpeedForwardFunction>& functions,
        const std::vector<TensorPtr>& inputs,
        int segments = -1
    ) {
        return GlobalCheckpointManager::get_instance().checkpoint_sequential(
            functions, inputs, segments
        );
    }
    
    /**
     * [Documentation available in English]
     */
    inline std::vector<TensorPtr> mobile_checkpoint(
        DeepSpeedForwardFunction forward_fn,
        const std::vector<TensorPtr>& inputs,
        float memory_budget_mb = 512.0f
    ) {
        // TODO: Use memory_budget_mb for budget-aware checkpointing
        (void)memory_budget_mb; // Mark as used to prevent warning
        return GlobalCheckpointManager::get_instance().mobile_aware_checkpoint(
            forward_fn, inputs
        );
    }
}

} // namespace memory
} // namespace ops
