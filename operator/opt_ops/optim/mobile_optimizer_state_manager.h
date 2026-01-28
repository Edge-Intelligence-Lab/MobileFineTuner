/**
 * @file mobile_optimizer_state_manager.h
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation in English - see separate docs]
 * 2. statecompression: useFP16/INT8compressionmomentumandvariancestate
 * [Documentation available in English]
 * [Documentation in English - see separate docs]
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation in English - see separate docs]
 * 
 * @author Your Name
 * @date 2025
 */

#pragma once

#include "../core/tensor.h"
#include "../core/device.h"
#include "param_manager_lite.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <chrono>
#include <string>

namespace ops {
namespace optim {

// Forward declarations (actualdefinesat param_manager_lite.h)
// [Translated]

/**
 * @brief Optimizerstatetype（correspondingAdamoptimizer）
 */
enum class OptimizerStateType {
    MOMENTUM = 0,              // [Translated]
    VARIANCE = 1,              // [Translated]
    MASTER_WEIGHTS = 2,        // [Translated]
    STEP_COUNT = 3             // [Translated]
};

/**
 * @brief Optimizerstatestorageposition
 */
enum class OptimizerStateTier {
    ACTIVE_MEMORY = 0,         // [Translated]
    STANDBY_MEMORY = 1,        // [Translated]
    COMPRESSED = 2,            // [Translated]
    DISK_STORAGE = 3           // [Translated]
};

/**
 * @brief Optimizerstatecompressionmode
 */
enum class OptimizerStateCompression {
    NONE = 0,              // nocompression (FP32)
    FP16 = 1,                  // [Translated]
    BFLOAT16 = 2,              // [Translated]
    INT8_QUANTIZED = 3,        // [Translated]
    INT8_SPARSE = 4,           // [Translated]
    ADAPTIVE = 5               // [Translated]
};

/**
 * [Documentation available in English]
 */
struct OptimizerStateMetadata {
    size_t param_id;                              // parameterID
    std::string param_name;                       // parametername
    size_t param_size;                                // [Translated]
    
        // [Translated]
    OptimizerStateTier momentum_tier;             // Momentumstateposition
    OptimizerStateTier variance_tier;             // Variancestateposition
    OptimizerStateCompression compression_mode;    // compressionmode
    
    // memoryusage
    size_t momentum_size_bytes;                   // Momentummemoryusage
    size_t variance_size_bytes;                   // Variancememoryusage
    size_t original_size_bytes;                       // [Translated]
    float compression_ratio;                          // [Translated]
    
        // [Translated]
    std::chrono::steady_clock::time_point last_access_time;
    std::chrono::steady_clock::time_point creation_time;
    size_t access_count;                              // [Translated]
    int priority;                                 // priority（0-10）
    
    // stateflag
    bool is_loaded;                                   // [Translated]
    bool is_dirty;                                // is notbemodify
    bool requires_grad;                           // is notrequiregradient
    bool is_trainable;                            // is nottrainable
    
        // [Translated]
    bool is_cpu_optimized;                        // is notforCPUoptimization
    bool use_simd_acceleration;                       // [Translated]
    bool is_cache_aligned;                            // [Translated]
    
    // [Translated comment removed - see documentation]
    std::string momentum_storage_path;
    std::string variance_storage_path;
    
    OptimizerStateMetadata(size_t id, const std::string& name, size_t size)
        : param_id(id), param_name(name), param_size(size),
          momentum_tier(OptimizerStateTier::ACTIVE_MEMORY),
          variance_tier(OptimizerStateTier::ACTIVE_MEMORY),
          compression_mode(OptimizerStateCompression::NONE),
          momentum_size_bytes(size * sizeof(float)),
          variance_size_bytes(size * sizeof(float)),
          original_size_bytes(size * sizeof(float)),
          compression_ratio(1.0f),
          access_count(0), priority(5),
          is_loaded(false), is_dirty(false),
          requires_grad(true), is_trainable(true),
          is_cpu_optimized(false), use_simd_acceleration(false),
          is_cache_aligned(false) {
        creation_time = std::chrono::steady_clock::now();
        last_access_time = creation_time;
    }
};

/**
 * [Documentation available in English]
 * [Documentation available in English]
 */
struct OptimizerStateGroup {
    std::string group_name;                           // [Translated]
    std::vector<size_t> param_ids;                    // [Translated]
    
        // [Translated]
    bool is_active;                                   // [Translated]
    size_t total_memory_usage;                        // [Translated]
    OptimizerStateCompression group_compression;       // [Translated]
    
        // [Translated]
    bool enable_group_prefetch;                       // [Translated]
    bool enable_group_offload;                        // [Translated]
    int access_priority;                              // [Translated]
    
    // statisticsinfo
    std::chrono::steady_clock::time_point last_group_access;
    size_t group_access_count;
    
    OptimizerStateGroup(const std::string& name)
        : group_name(name), is_active(false), total_memory_usage(0),
          group_compression(OptimizerStateCompression::NONE),
          enable_group_prefetch(true), enable_group_offload(true),
          access_priority(5), group_access_count(0) {
        last_group_access = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Optimizerstatebuffer - correspondingDeepSpeedContiguous Buffer
 * [Documentation in English - see separate docs]
 */
class OptimizerStateBuffer {
private:
    void* buffer_ptr_;                            // bufferpointer
    size_t buffer_size_;                              // [Translated]
    size_t used_size_;                                // [Translated]
    [[maybe_unused]] bool is_cache_aligned_;          // [Translated]
    mutable std::mutex buffer_mutex_;                 // [Translated]
    
        // [Translated]
    std::vector<std::pair<size_t, size_t>> free_chunks_; // (offset, size)
    
public:
    OptimizerStateBuffer(size_t size_bytes, bool cache_align = true);
    ~OptimizerStateBuffer();
    
    // allocateandrelease
    void* allocate(size_t size_bytes);
    void deallocate(void* ptr, size_t size_bytes);
    
    // [Translated comment removed - see documentation]
    void defragment();
    float get_fragmentation_ratio() const;
    
    // statisticsinfo
    size_t get_used_size() const { return used_size_; }
    size_t get_free_size() const { return buffer_size_ - used_size_; }
    size_t get_total_size() const { return buffer_size_; }
};

/**
 * [Documentation available in English]
 * [Documentation available in English]
 */
class OptimizerStateCompressor {
private:
    [[maybe_unused]] OptimizerStateCompression default_compression_;
    
public:
    explicit OptimizerStateCompressor(OptimizerStateCompression mode = OptimizerStateCompression::FP16);
    
    /**
     * @brief compressionoptimizerstate
     * @param input inputtensor (FP32)
     * @param mode compressionmode
     * [Documentation available in English]
     */
    std::pair<TensorPtr, float> compress(const TensorPtr& input, OptimizerStateCompression mode);
    
    /**
     * @brief decompressoptimizerstate
     * @param compressed compressionbacktensor
     * @param mode compressionmode
     * @return decompressbacktensor (FP32)
     */
    TensorPtr decompress(const TensorPtr& compressed, OptimizerStateCompression mode);
    
    /**
     * [Documentation available in English]
     * @param input inputtensor
     * [Documentation available in English]
     * [Documentation available in English]
     */
    std::tuple<TensorPtr, OptimizerStateCompression, float> 
    adaptive_compress(const TensorPtr& input, float importance);
    
private:
        // [Translated]
    TensorPtr compress_fp16(const TensorPtr& input);
    TensorPtr decompress_fp16(const TensorPtr& compressed);
    
    TensorPtr compress_int8_quantized(const TensorPtr& input);
    TensorPtr decompress_int8_quantized(const TensorPtr& compressed);
    
    // CPUoptimizationSIMDimplements
    void compress_fp32_to_fp16_simd(const float* src, uint16_t* dst, size_t count);
    void decompress_fp16_to_fp32_simd(const uint16_t* src, float* dst, size_t count);
};

/**
 * @brief OptimizerstateI/Omanager
 * [Documentation in English - see separate docs]
 */
class OptimizerStateIOManager {
private:
    std::string storage_path_;
    [[maybe_unused]] bool enable_compression_;
    std::atomic<size_t> total_io_operations_;
    std::atomic<size_t> total_bytes_written_;
    std::atomic<size_t> total_bytes_read_;
    
public:
    explicit OptimizerStateIOManager(const std::string& path, bool compress = true);
    
    /**
     * [Documentation available in English]
     * @param state_id stateID
     * @param state_type statetype (MOMENTUM/VARIANCE)
     * @param data statetensor
     * @return savepath
     */
    std::string save_state_to_disk(size_t state_id, OptimizerStateType state_type, const TensorPtr& data);
    
    /**
     * [Documentation available in English]
     * @param path storagepath
     * @return statetensor
     */
    TensorPtr load_state_from_disk(const std::string& path);
    
    /**
     * [Documentation available in English]
     * @param path storagepath
     */
    void delete_state_file(const std::string& path);
    
    /**
     * @brief acquireI/Ostatisticsinforation
     */
    struct IOStats {
        size_t total_operations;
        size_t total_bytes_written;
        size_t total_bytes_read;
        double average_write_speed_mbps;
        double average_read_speed_mbps;
    };
    IOStats get_io_stats() const;
};

/**
 * @brief configurationstructure
 */
struct MobileOptimizerStateConfig {
    // basicconfiguration
    size_t max_active_memory_mb = 256;                // [Translated]
    size_t max_standby_memory_mb = 512;               // [Translated]
    std::string storage_path = "./optimizer_states"; // storagepath
    
    // compressionconfiguration
    bool enable_compression = true;               // enablecompression
    OptimizerStateCompression default_compression = OptimizerStateCompression::FP16;
    bool enable_adaptive_compression = true;      // adaptivecompression
    float compression_threshold = 0.7f;           // compressionthreshold（70%memoryuse）
    
    // CPUoptimizationconfiguration
    bool enable_cpu_simd = true;                  // enableSIMDoptimization
    bool enable_cache_alignment = true;               // [Translated]
    size_t cache_line_size = 64;                      // [Translated]
    bool enable_prefetch = true;                      // [Translated]
    
        // [Translated]
    bool enable_disk_offload = true;                  // [Translated]
    float offload_threshold = 0.8f;                   // [Translated]
    bool enable_async_io = true;                  // asyncI/O
    
    // memorymanageconfiguration
    bool use_contiguous_buffers = true;               // [Translated]
    size_t buffer_size_mb = 128;                  // buffersize
    bool enable_defragmentation = true;               // [Translated]
    float defrag_threshold = 0.3f;                    // [Translated]
    
        // [Translated]
    bool optimize_for_mobile_cpu = true;          // moveCPUoptimization
    bool respect_thermal_limits = true;               // [Translated]
    bool respect_battery_limits = true;               // [Translated]
    float cpu_utilization_target = 0.7f;              // [Translated]
    
        // [Translated]
    bool enable_group_management = true;              // [Translated]
    bool enable_group_prefetch = true;                // [Translated]
    bool enable_group_offload = true;                 // [Translated]
    
    // advancedoptimizationconfiguration
    bool enable_gradient_accumulation = true;     // gradientaccumulate
    int gradient_accumulation_steps = 1;          // accumulatestep
    bool enable_mixed_precision = true;           // mixed precision
    bool enable_loss_scaling = false;             // lossscale
};

/**
 * @brief statisticsinforation
 */
struct OptimizerStateStats {
    // memorystatistics
    size_t total_states;                              // [Translated]
    size_t active_states;                             // [Translated]
    size_t compressed_states;                         // [Translated]
    size_t offloaded_states;                          // [Translated]
    
    size_t active_memory_used;                        // [Translated]
    size_t standby_memory_used;                       // [Translated]
    size_t compressed_memory_used;                // compressionmemoryuse
    size_t disk_storage_used;                         // [Translated]
    
    // compressionstatistics
    size_t total_compressions;                        // [Translated]
    size_t total_decompressions;                      // [Translated]
    float average_compression_ratio;                  // [Translated]
    size_t memory_saved_by_compression;               // [Translated]
    
    // I/Ostatistics
    size_t total_loads;                               // [Translated]
    size_t total_offloads;                            // [Translated]
    double average_load_time_ms;                      // [Translated]
    double average_offload_time_ms;                   // [Translated]
    
    // perforancestatistics
    size_t cache_hits;                                // [Translated]
    size_t cache_misses;                              // [Translated]
    float cache_hit_ratio;                            // [Translated]
    size_t defragmentation_count;                     // [Translated]
    
    // mobilestatistics
    size_t thermal_throttle_events;                   // [Translated]
    size_t battery_optimization_events;           // batteryoptimizationevent
    float cpu_utilization;                            // [Translated]
};

/**
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * 3. compressionanddecompressoptimizerstate
 * 4. CPUoptimizationmemorymanage
 * [Documentation available in English]
 */
class MobileOptimizerStateManager {
private:
    MobileOptimizerStateConfig config_;
    
    // statestorage
    std::unordered_map<size_t, std::unique_ptr<OptimizerStateMetadata>> state_metadata_;
    std::unordered_map<size_t, TensorPtr> momentum_states_;     // param_id -> momentum tensor
    std::unordered_map<size_t, TensorPtr> variance_states_;     // param_id -> variance tensor
    std::unordered_map<size_t, TensorPtr> master_weights_;      // param_id -> FP32 master weight
    
        // [Translated]
    std::vector<std::unique_ptr<OptimizerStateGroup>> state_groups_;
    std::unordered_map<size_t, size_t> param_to_group_map_;     // param_id -> group_id
    std::unordered_map<std::string, size_t> group_name_to_id_;  // group_name -> group_id
    
    // memorymanage
    std::unique_ptr<OptimizerStateBuffer> active_buffer_;
    std::unique_ptr<OptimizerStateBuffer> standby_buffer_;
    std::unique_ptr<OptimizerStateCompressor> compressor_;
    std::unique_ptr<OptimizerStateIOManager> io_manager_;
    
        // [Translated]
    MobileParameterManager* param_manager_;
    
    // memoryusetrace
    std::atomic<size_t> active_memory_used_;
    std::atomic<size_t> standby_memory_used_;
    std::atomic<size_t> compressed_memory_used_;
    
    // statisticsinfo
    OptimizerStateStats stats_;
    mutable std::mutex stats_mutex_;
    mutable std::mutex manager_mutex_;
    
    // mobilemonitor
    std::atomic<float> current_cpu_utilization_;
    std::atomic<bool> is_thermal_throttling_;
    std::atomic<bool> is_low_battery_;

public:
    /**
     * @brief constructfunction
     * @param config configuration
     * [Documentation available in English]
     */
    explicit MobileOptimizerStateManager(
        const MobileOptimizerStateConfig& config,
        MobileParameterManager* param_manager = nullptr
    );
    
    ~MobileOptimizerStateManager();
    
    // ============================================================================
        // [Translated]
    // ============================================================================
    
    /**
     * @brief registerparameteroptimizerstate
     * @param param_id parameterID
     * @param param_name parametername
     * [Documentation available in English]
     * [Documentation available in English]
     * @param requires_grad is notrequiregradient
     */
    void register_parameter_state(
        size_t param_id,
        const std::string& param_name,
        size_t param_size,
        const std::string& group_name = "default",
        bool requires_grad = true
    );
    
    /**
     * @brief acquireparametermomentumstate
     * @param param_id parameterID
     * [Documentation available in English]
     */
    TensorPtr get_momentum_state(size_t param_id);
    
    /**
     * @brief acquireparametervariancestate
     * @param param_id parameterID
     * [Documentation available in English]
     */
    TensorPtr get_variance_state(size_t param_id);
    
    /**
     * @brief updatemomentumstate
     * @param param_id parameterID
     * @param new_momentum newmomentumvalue
     */
    void update_momentum_state(size_t param_id, const TensorPtr& new_momentum);
    
    /**
     * @brief updatevariancestate
     * @param param_id parameterID
     * @param new_variance newvariancevalue
     */
    void update_variance_state(size_t param_id, const TensorPtr& new_variance);
    
    /**
     * [Documentation available in English]
     * @param param_id parameterID
     */
    void release_parameter_state(size_t param_id);
    
    // ============================================================================
        // [Translated]
    // ============================================================================
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     */
    void load_group_states(const std::string& group_name);
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation in English - see separate docs]
     */
    void offload_group_states(const std::string& group_name, bool force = false);
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     * @param compression compressionmode
     */
    void set_group_compression(const std::string& group_name, OptimizerStateCompression compression);
    
    // ============================================================================
    // memoryoptimizationAPI
    // ============================================================================
    
    /**
     * [Documentation available in English]
     * @param param_id parameterID
     * @param compression compressionmode
     * [Documentation available in English]
     */
    size_t compress_parameter_state(size_t param_id, OptimizerStateCompression compression);
    
    /**
     * [Documentation available in English]
     * @param param_id parameterID
     * [Documentation available in English]
     */
    size_t offload_parameter_state(size_t param_id);
    
    /**
     * [Documentation available in English]
     * [Documentation in English - see separate docs]
     */
    void optimize_memory_usage();
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     */
    size_t defragment_memory();
    
    /**
     * [Documentation available in English]
     * [Documentation in English - see separate docs]
     * [Documentation available in English]
     */
    size_t emergency_memory_cleanup();
    
    // ============================================================================
        // [Translated]
    // ============================================================================
    
    /**
     * @brief updatemobilesystemstate
     * [Documentation available in English]
     * [Documentation available in English]
     * @param is_low_battery is notlow battery
     */
    void update_mobile_state(float cpu_util, bool is_thermal_throttle, bool is_low_battery);
    
    /**
     * @brief enabled/disabledCPU SIMDoptimization
     * @param enable is notenabled
     */
    void enable_cpu_simd_optimization(bool enable);
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     */
    void set_cpu_utilization_target(float target);
    
    // ============================================================================
    // statisticsandmonitorAPI
    // ============================================================================
    
    /**
     * @brief acquirestatisticsinforation
     */
    OptimizerStateStats get_statistics() const;
    
    /**
     * [Documentation available in English]
     */
    const OptimizerStateMetadata* get_state_metadata(size_t param_id) const;
    
    /**
     * [Documentation available in English]
     * @param report_path reportsavepath
     */
    void export_detailed_report(const std::string& report_path) const;
    
    // ============================================================================
    // checkpointAPI
    // ============================================================================
    
    /**
     * @brief savealloptimizerstatetocheckpoint
     * @param checkpoint_path checkpointpath
     */
    void save_checkpoint(const std::string& checkpoint_path);
    
    /**
     * @brief fromcheckpointloadoptimizerstate
     * @param checkpoint_path checkpointpath
     */
    void load_checkpoint(const std::string& checkpoint_path);
    
    // ============================================================================
    // configurationAPI
    // ============================================================================
    
    /**
     * @brief acquirecurrentconfiguration
     */
    const MobileOptimizerStateConfig& get_config() const { return config_; }
    
    /**
     * [Documentation available in English]
     */
    void set_parameter_manager(MobileParameterManager* param_manager);

private:
    // ============================================================================
    // internalimplementsmethod
    // ============================================================================
    
    // initializemethod
    void initialize_components();
    void cleanup_components();
    
        // [Translated]
    void load_state_internal(size_t param_id, OptimizerStateType state_type);
    void offload_state_internal(size_t param_id, OptimizerStateType state_type);
    
    // memorymanage
    void* allocate_from_buffer(size_t size_bytes, bool is_active);
    void deallocate_from_buffer(void* ptr, size_t size_bytes, bool is_active);
    size_t calculate_memory_pressure() const;
    
    // selectalgorithm
    std::vector<size_t> select_states_to_compress(size_t target_memory_reduction);
    std::vector<size_t> select_states_to_offload(size_t target_memory_reduction);
    OptimizerStateCompression select_optimal_compression(size_t param_id);
    
    // statisticsupdate
    void update_statistics();
    void update_access_pattern(size_t param_id);
    
    // mobileoptimization
    void apply_cpu_optimization();
    void apply_thermal_optimization();
    void apply_battery_optimization();
};

/**
 * [Documentation available in English]
 */
std::unique_ptr<MobileOptimizerStateManager> create_mobile_optimizer_state_manager(
    size_t available_memory_mb = 256,
    const std::string& storage_path = "./optimizer_states",
    MobileParameterManager* param_manager = nullptr
);

} // namespace memory
} // namespace ops

