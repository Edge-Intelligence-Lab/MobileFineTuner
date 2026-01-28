/**
 * @file activation_partitioning.h
 * [Documentation in English - see separate docs]
 * 
 * [Documentation in English - see separate docs]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation in English - see separate docs]
 * [Documentation available in English]
 * 
 * [Documentation in English - see separate docs]
 * [Documentation available in English]
 */

#pragma once

#include "../core/tensor.h"
#include "../core/device.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <functional>
#include <queue>
#include <future>

namespace ops {
namespace memory {

using ops::TensorPtr;
using ops::Tensor;
using ops::Device;
using ops::kCPU;

// [Translated comment removed - see documentation]
/**
 * @brief Activation compression modes optimized for mobile
 */
#ifndef ACTIVATION_COMPRESSION_MODE_DEFINED
#define ACTIVATION_COMPRESSION_MODE_DEFINED
enum class ActivationCompressionMode {
    NONE = 0,              // No compression
    QUANTIZE_INT8 = 1,     // 8-bit quantization (3-4x compression)
    QUANTIZE_INT4 = 2,     // 4-bit quantization (6-8x compression)
    SPARSE_50 = 3,         // 50% sparsification
    SPARSE_75 = 4,         // 75% sparsification  
    LOSSY_COMPRESS = 5,    // Lossy compression for non-critical activations
    ADAPTIVE = 6           // Adaptive compression based on system state
};
#endif

/**
 * [Documentation available in English]
 */
enum class PartitionLocation {
    GPU_PRIMARY = 0,         // [Translated]
    GPU_SECONDARY = 1,   // secondaryGPUmemory（ifhavemultipleGPU）
    CPU_MEMORY = 2,      // CPUmemory
    COMPRESSED = 3,      // compressionmemory
    PERSISTENT = 4           // [Translated]
};

/**
 * [Documentation available in English]
 */
enum class PartitionStrategy {
    UNIFORM = 0,             // [Translated]
    ADAPTIVE = 1,            // [Translated]
    IMPORTANCE_BASED = 2,     // [Translated]
    MOBILE_OPTIMIZED = 3      // [Translated]
};

/**
 * [Documentation available in English]
 */
enum class GatherStrategy {
    EAGER = 0,               // [Translated]
    LAZY = 1,                // [Translated]
    STREAMING = 2,           // [Translated]
    MOBILE_AWARE = 3         // [Translated]
};

/**
 * [Documentation available in English]
 */
struct ActivationPartition {
    size_t partition_id;                        // [Translated]
    size_t activation_id;                       // [Translated]
    std::vector<int64_t> partition_shape;       // [Translated]
    std::vector<int64_t> offset;                // [Translated]
    PartitionLocation location;             // storageposition
    
    // memoryinfo
    size_t size_bytes;                          // [Translated]
    bool is_compressed;                     // is notcompression
    float compression_ratio;                    // [Translated]
    
        // [Translated]
    std::chrono::steady_clock::time_point last_access_time;
    size_t access_count;                        // [Translated]
    float access_frequency;                     // [Translated]
    
    // mobileoptimizationinfo
    bool is_ui_critical;                        // [Translated]
    float power_cost_analysis;              // 🚀 PRODUCTION: poweranalyzemetrics
    int priority_level;                     // priority（0-10）
    
    // [Translated comment removed - see documentation]
    std::vector<size_t> dependent_partitions;     // [Translated]
    bool can_be_prefetched;                     // [Translated]
    
    ActivationPartition(size_t part_id, size_t act_id, const std::vector<int64_t>& shape)
        : partition_id(part_id), activation_id(act_id), partition_shape(shape),
          location(PartitionLocation::CPU_MEMORY), size_bytes(0), is_compressed(false),
          compression_ratio(1.0f), access_count(0), access_frequency(0.0f),
          is_ui_critical(false), power_cost_analysis(0.0f), priority_level(5),
          can_be_prefetched(true) {
        last_access_time = std::chrono::steady_clock::now();
    }
};

/**
 * [Documentation available in English]
 */
struct GatherContext {
    size_t activation_id;
    std::vector<std::shared_ptr<ActivationPartition>> partitions;
    GatherStrategy strategy;
    std::vector<int64_t> target_shape;
    Device target_device;
    std::function<void(const TensorPtr&)> completion_callback;
    
    // perforancemonitor
    std::chrono::steady_clock::time_point start_time;
    std::atomic<size_t> gathered_partitions;
    std::atomic<bool> is_complete;
    
    GatherContext(size_t act_id, GatherStrategy strat, const std::vector<int64_t>& shape, const Device& device = kCPU)
        : activation_id(act_id), strategy(strat), target_shape(shape), target_device(device),
          gathered_partitions(0), is_complete(false) {
        start_time = std::chrono::steady_clock::now();
    }
};

/**
 * [Documentation available in English]
 */
struct PartitionConfig {
        // [Translated]
    PartitionStrategy partition_strategy = PartitionStrategy::MOBILE_OPTIMIZED;
    size_t max_partition_size_mb = 64;          // [Translated]
    size_t min_partition_size_mb = 4;           // [Translated]
    int max_partitions_per_activation = 16;     // [Translated]
    
    // storagelayerlevelconfiguration
    size_t gpu_partition_quota_mb = 256;        // [Translated]
    size_t cpu_partition_quota_mb = 512;        // [Translated]
    size_t compressed_partition_quota_mb = 1024;     // [Translated]
    
        // [Translated]
    GatherStrategy default_gather_strategy = GatherStrategy::MOBILE_AWARE;
    bool enable_async_gather = true;            // [Translated]
    bool enable_prefetch_gather = true;         // [Translated]
    int max_concurrent_gathers = 3;             // [Translated]
    
    // mobileoptimizationconfiguration
    bool enable_mobile_optimizations = true; // enablemobileoptimization
    bool prioritize_ui_responsiveness = true;     // [Translated]
    bool enable_battery_aware_partitioning = true;     // [Translated]
    bool enable_thermal_aware_partitioning = true;     // [Translated]
    
    // compressionconfiguration
    bool enable_partition_compression = true;     // [Translated]
    float compression_threshold = 0.8f;         // [Translated]
    ActivationCompressionMode default_compression = ActivationCompressionMode::QUANTIZE_INT8;
    
        // [Translated]
    float partition_vs_gather_tradeoff = 0.6f;     // [Translated]
    bool enable_intelligent_prefetch = true;       // [Translated]
    bool enable_adaptive_partition_sizing = true;     // [Translated]
    
    // Perforance analysis and monitoring
    bool enable_partition_profiling = false;     // [Translated]
    bool log_partition_events = false;           // [Translated]
    std::string profiling_output_path = "./partition_profile.json";
};

/**
 * [Documentation available in English]
 */
struct PartitionStats {
    // basicstatistics
    size_t total_activations_partitioned;
    size_t total_partitions_created;
    size_t total_partitions_active;
    size_t total_gather_operations;
    
    // memorystatistics
    size_t gpu_memory_used_by_partitions;
    size_t cpu_memory_used_by_partitions;
    size_t compressed_memory_used_by_partitions;
    size_t total_memory_saved_by_partitioning;
    
    // perforancestatistics
    double average_partition_time_ms;
    double average_gather_time_ms;
    double average_compression_ratio;
    size_t cache_hits;
    size_t cache_misses;
    
    // mobilestatistics
    size_t battery_optimized_partitions;
    size_t thermal_optimized_partitions;
    size_t ui_responsive_gathers;
    size_t prefetch_hits;
    size_t prefetch_misses;
};

/**
 * [Documentation available in English]
 */
class MobileActivationPartitioner {
private:
    PartitionConfig config_;
    
        // [Translated]
    std::unordered_map<size_t, std::vector<std::shared_ptr<ActivationPartition>>> activation_partitions_;
    std::unordered_map<size_t, TensorPtr> partition_tensors_;  // partition_id -> tensor
    std::unordered_map<PartitionLocation, size_t> location_memory_usage_;
    
        // [Translated]
    std::unordered_map<size_t, std::shared_ptr<GatherContext>> active_gathers_;
    std::queue<std::shared_ptr<GatherContext>> pending_gathers_;
    
    // asyncprocess
    std::vector<std::thread> worker_threads_;
    std::mutex partition_mutex_;
    std::mutex gather_mutex_;
    std::condition_variable gather_cv_;
    std::atomic<bool> shutdown_flag_;
    
    // mobilestatemonitor
    std::atomic<float> current_memory_pressure_;
    std::atomic<int> current_battery_level_;
    std::atomic<float> current_temperature_;
    std::atomic<bool> is_app_foreground_;
    
        // [Translated]
    struct PrefetchCandidate {
        size_t activation_id;
        float probability;
        std::chrono::steady_clock::time_point predicted_access_time;
        
        PrefetchCandidate(size_t id, float prob) 
            : activation_id(id), probability(prob) {
            predicted_access_time = std::chrono::steady_clock::now() + 
                                  std::chrono::milliseconds(static_cast<int>(1000 * (1.0f - prob)));
        }
    };
    std::priority_queue<PrefetchCandidate, std::vector<PrefetchCandidate>,
                       std::function<bool(const PrefetchCandidate&, const PrefetchCandidate&)>> prefetch_queue_;
    
    // statisticsinfo
    PartitionStats stats_;
    mutable std::mutex stats_mutex_;
    
    std::atomic<size_t> next_partition_id_;

public:
    explicit MobileActivationPartitioner(const PartitionConfig& config);
    ~MobileActivationPartitioner();
    
    /**
     * [Documentation available in English]
     * @param activation_id activationvalueID
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation available in English]
     */
    std::vector<size_t> partition_activation(
        size_t activation_id,
        const TensorPtr& activation,
        PartitionStrategy strategy = PartitionStrategy::MOBILE_OPTIMIZED
    );
    
    /**
     * [Documentation available in English]
     * @param activation_id activationvalueID
     * [Documentation available in English]
     * @param target_device targetdevice
     * [Documentation available in English]
     */
    TensorPtr gather_activation(
        size_t activation_id,
        GatherStrategy strategy = GatherStrategy::MOBILE_AWARE,
        Device target_device = kCPU
    );
    
    /**
     * [Documentation available in English]
     * @param activation_id activationvalueID
     * [Documentation available in English]
     * @param target_device targetdevice
     * [Documentation available in English]
     */
    std::future<TensorPtr> gather_activation_async(
        size_t activation_id,
        GatherStrategy strategy = GatherStrategy::MOBILE_AWARE,
        Device target_device = kCPU
    );
    
    /**
     * [Documentation in English - see separate docs]
     * [Documentation available in English]
     * [Documentation available in English]
     */
    void prefetch_activations(
        const std::vector<size_t>& activation_ids,
        Device target_device = kCPU
    );
    
    /**
     * [Documentation available in English]
     * @param activation_id activationvalueID
     * [Documentation in English - see separate docs]
     */
    void release_activation_partitions(size_t activation_id, bool force_release = false);
    
    /**
     * [Documentation in English - see separate docs]
     */
    void optimize_partition_layout();
    
    /**
     * @brief updatemobilesystemstate
     * [Documentation available in English]
     * @param battery_level batterybattery level (0-100)
     * [Documentation available in English]
     * @param is_foreground is notforegroundrunning
     */
    void update_mobile_state(float memory_pressure, int battery_level, 
                            float temperature, bool is_foreground);
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     */
    void configure_partitioning(const PartitionConfig& config);
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     */
    PartitionStats get_partition_stats() const;
    
    /**
     * [Documentation available in English]
     * @param activation activationvaluetensor
     * [Documentation available in English]
     * [Documentation available in English]
     */
    struct OptimalPartitionPlan {
        PartitionStrategy strategy;
        std::vector<std::vector<int64_t>> partition_shapes;
        std::vector<PartitionLocation> partition_locations;
        float estimated_memory_savings;
        float estimated_gather_overhead;
    };
    OptimalPartitionPlan calculate_optimal_partition_plan(
        const TensorPtr& activation,
        size_t current_memory_usage
    );
    
    /**
     * [Documentation available in English]
     * @param report_path reportsavepath
     */
    void export_profiling_report(const std::string& report_path) const;

private:
        // [Translated]
    std::vector<std::shared_ptr<ActivationPartition>> partition_unifor(
        size_t activation_id, const TensorPtr& activation);
    std::vector<std::shared_ptr<ActivationPartition>> partition_adaptive(
        size_t activation_id, const TensorPtr& activation);
    std::vector<std::shared_ptr<ActivationPartition>> partition_mobile_optimized(
        size_t activation_id, const TensorPtr& activation);
    
        // [Translated]
    TensorPtr gather_eager(const std::shared_ptr<GatherContext>& context);
    TensorPtr gather_lazy(const std::shared_ptr<GatherContext>& context);
    TensorPtr gather_streaming(const std::shared_ptr<GatherContext>& context);
    TensorPtr gather_mobile_aware(const std::shared_ptr<GatherContext>& context);
    
    // storagepositionselect
    PartitionLocation select_optimal_location(
        const std::shared_ptr<ActivationPartition>& partition);
    bool can_store_in_location(PartitionLocation location, size_t size_bytes);
    void migrate_partition_to_location(
        std::shared_ptr<ActivationPartition> partition, PartitionLocation new_location);
    
    // mobileoptimizationmethod
    void apply_mobile_optimizations_to_partition(
        std::shared_ptr<ActivationPartition> partition);
    void adapt_partitioning_for_battery_state();
    void adapt_partitioning_for_thermal_state();
    void adapt_partitioning_for_memory_pressure();
    
    // [Translated comment removed - see documentation]
    void update_access_patterns(size_t activation_id);
    std::vector<size_t> predict_next_accesses(size_t current_activation_id);
    void schedule_intelligent_prefetch();
    
        // [Translated]
    void gather_worker_loop();
    void prefetch_worker_loop();
    void optimization_worker_loop();
    
    // toolmethod
    std::vector<int64_t> calculate_partition_shape(
        const std::vector<int64_t>& original_shape, int partition_index, int total_partitions);
    size_t calculate_tensor_size_bytes(const std::vector<int64_t>& shape);
    bool is_partition_hot(const std::shared_ptr<ActivationPartition>& partition);
    void update_partition_statistics(const std::shared_ptr<ActivationPartition>& partition);
    
    // memorymanagetool
    void cleanup_expired_partitions();
    void handle_memory_pressure();
    void compress_cold_partitions();
    
    // Logging and analytics
    void log_partition_event(const std::string& event, size_t activation_id, 
                            const std::string& details = "");
    void update_perforance_metrics();
};

/**
 * [Documentation available in English]
 */
namespace partition_utils {
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation available in English]
     */
    int calculate_optimal_partition_count(
        size_t tensor_size, 
        size_t available_memory, 
        size_t target_partition_size
    );
    
    /**
     * [Documentation available in English]
     * @param shape tensorshape
     * [Documentation in English - see separate docs]
     */
    float analyze_partition_friendliness(const std::vector<int64_t>& shape);
    
    /**
     * [Documentation available in English]
     * @param original_size originalsize
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation available in English]
     */
    std::pair<double, size_t> estimate_partition_overhead(
        size_t original_size, 
        int partition_count, 
        float gather_frequency
    );
    
    /**
     * [Documentation available in English]
     * @param original_shape originaltensorshape
     * [Documentation available in English]
     * [Documentation available in English]
     */
    std::vector<std::vector<int64_t>> optimize_partition_shapes_for_mobile(
        const std::vector<int64_t>& original_shape,
        int target_partition_count
    );
}

} // namespace memory
} // namespace ops
