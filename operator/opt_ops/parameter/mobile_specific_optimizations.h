/**
 * @file mobile_specific_optimizations.h
 * @brief Mobile-specific optimizations that go beyond DeepSpeed
 * 
 * This file contains optimizations that are unique to mobile environments
 * and not covered by DeepSpeed. These are critical for stable, efficient
 * operation on resource-constrained mobile devices.
 */

#pragma once

#include "mobile_param_manager.h"
#ifdef __ANDROID__
#include <jni.h>        // For Android integration
#else
// Placeholder JNI types for non-Android builds
typedef struct _JNIEnv JNIEnv;
typedef void* jobject;
typedef void* jclass;
#endif
#ifdef __linux__
#include <sys/sysinfo.h> // For Linux system info
#endif
#include <unistd.h>     // For system calls

namespace ops {
namespace memory {

/**
 * @brief Android-specific memory management integration
 */
class AndroidMemoryManager {
private:
    JNIEnv* jni_env_;
    jobject activity_ref_;
    jclass memory_info_class_;
    bool is_initialized_;

public:
    AndroidMemoryManager();
    ~AndroidMemoryManager();
    
    /**
     * @brief Initialize JNI integration
     */
    bool initialize(JNIEnv* env, jobject activity);
    
    /**
     * @brief Get Android memory info
     */
    struct AndroidMemoryInfo {
        size_t total_memory_mb;
        size_t available_memory_mb;
        size_t low_memory_threshold_mb;
        double memory_pressure_ratio;
        bool is_low_memory_state;
    };
    AndroidMemoryInfo get_android_memory_info() const;
    
    /**
     * @brief Register for Android Low Memory notifications
     */
    void register_low_memory_callback(const std::function<void()>& callback);
    
    /**
     * @brief Request garbage collection from Java runtime
     */
    void request_java_gc();
    
    /**
     * @brief Get Android app memory class
     */
    size_t get_app_memory_class_mb() const;
    
    /**
     * @brief Check if app is in background
     */
    bool is_app_in_background() const;
};

/**
 * @brief ZRAM and swap file optimization
 */
class ZRAMOptimizer {
private:
    bool zram_available_;
    size_t zram_size_mb_;
    std::string swap_file_path_;
    bool swap_file_active_;

public:
    ZRAMOptimizer();
    ~ZRAMOptimizer();
    
    /**
     * @brief Check if ZRAM is available on system
     */
    bool is_zram_available() const;
    
    /**
     * @brief Get ZRAM statistics
     */
    struct ZRAMStats {
        size_t compressed_size_mb;
        size_t original_size_mb;
        double compression_ratio;
        size_t swap_used_mb;
    };
    ZRAMStats get_zram_stats() const;
    
    /**
     * @brief Create and activate swap file
     * @param size_mb Swap file size in MB
     * @param path Path for swap file
     */
    bool create_swap_file(size_t size_mb, const std::string& path = "/data/data/swap");
    
    /**
     * @brief Optimize parameter for ZRAM compression
     * @param data Parameter data
     * @param size Data size
     * @return Optimized data layout for better compression
     */
    std::vector<uint8_t> optimize_for_zram_compression(const void* data, size_t size);
};

/**
 * @brief CPU big.LITTLE scheduler optimization
 */
class BigLittleOptimizer {
private:
    std::vector<size_t> big_core_ids_;
    std::vector<size_t> little_core_ids_;
    bool big_little_available_;

public:
    BigLittleOptimizer();
    
    /**
     * @brief Detect big.LITTLE CPU configuration
     */
    bool detect_big_little_configuration();
    
    /**
     * @brief Schedule computation on appropriate cores
     * @param computation_type Type of computation (memory-intensive, compute-intensive, etc.)
     * @param thread_id Thread to schedule
     */
    void schedule_on_optimal_cores(const std::string& computation_type, std::thread::id thread_id);
    
    /**
     * @brief Get optimal core assignment for parameter operations
     */
    std::vector<size_t> get_optimal_cores_for_param_ops() const;
    
    /**
     * @brief Enable/disable big cores based on thermal state
     */
    void thermal_aware_core_management(bool enable_big_cores);
};

/**
 * @brief Mobile GPU memory bandwidth optimizer
 */
class MobileGPUOptimizer {
private:
    MobileGPUVendor gpu_vendor_;
    size_t memory_bandwidth_gbps_;
    bool unified_memory_architecture_;

public:
    explicit MobileGPUOptimizer(MobileGPUVendor vendor);
    
    /**
     * @brief Optimize memory transfer patterns for mobile GPU
     */
    void optimize_transfer_pattern(void* src, void* dst, size_t size);
    
    /**
     * @brief Get optimal transfer chunk size for this GPU
     */
    size_t get_optimal_transfer_chunk_size() const;
    
    /**
     * @brief Check if unified memory architecture is available
     */
    bool has_unified_memory_architecture() const;
    
    /**
     * @brief Get GPU-specific memory alignment requirements
     */
    size_t get_gpu_memory_alignment() const;
};

/**
 * @brief Battery-aware computation scheduler
 */
class BatteryAwareScheduler {
private:
    std::atomic<size_t> battery_level_;
    std::atomic<bool> is_charging_;
    std::atomic<size_t> estimated_power_consumption_mw_;
    
public:
    BatteryAwareScheduler();
    
    /**
     * @brief Update battery state
     */
    void update_battery_state(size_t level_percent, bool charging);
    
    /**
     * @brief Check if operation should be throttled due to battery
     */
    bool should_throttle_for_battery(size_t estimated_power_mw, size_t duration_ms) const;
    
    /**
     * @brief Get battery-optimal computation strategy
     */
    enum class BatteryStrategy {
        PERFORMANCE,    // Battery > 50% or charging
        BALANCED,       // Battery 20-50%
        POWER_SAVER,    // Battery 10-20%
        EMERGENCY       // Battery < 10%
    };
    BatteryStrategy get_optimal_strategy() const;
    
    /**
     * @brief Estimate power cost of parameter operation
     */
    size_t estimate_operation_power_cost(size_t param_size_mb, const std::string& operation) const;
};

/**
 * @brief Network-aware parameter offloading
 */
class NetworkAwareOffloader {
private:
    std::atomic<bool> is_on_cellular_;
    std::atomic<bool> is_metered_;
    std::atomic<size_t> data_usage_bytes_;
    size_t daily_limit_bytes_;

public:
    NetworkAwareOffloader();
    
    /**
     * @brief Set network state
     */
    void set_network_state(bool cellular, bool metered);
    
    /**
     * @brief Check if parameter should be offloaded considering network
     */
    bool should_offload_parameter(size_t param_size_bytes) const;
    
    /**
     * @brief Get network-optimal compression for offloading
     */
    double get_network_optimal_compression_ratio() const;
    
    /**
     * @brief Track data usage
     */
    void track_data_usage(size_t bytes_transferred);
    
    /**
     * @brief Check remaining data budget
     */
    size_t get_remaining_data_budget_bytes() const;
};

/**
 * @brief UI responsiveness monitor and optimizer
 */
class UIResponsivenessOptimizer {
private:
    std::atomic<float> target_fps_;
    std::atomic<size_t> frame_drops_;
    std::chrono::steady_clock::time_point last_frame_time_;
    std::vector<double> frame_times_;
    
public:
    explicit UIResponsivenessOptimizer(float target_fps = 60.0f);
    
    /**
     * @brief Record frame timing
     */
    void record_frame_timing();
    
    /**
     * @brief Check if operation would cause frame drop
     */
    bool would_cause_frame_drop(size_t estimated_duration_ms) const;
    
    /**
     * @brief Get UI-safe operation time budget
     */
    size_t get_ui_safe_time_budget_ms() const;
    
    /**
     * @brief Optimize operation scheduling for UI smoothness
     */
    void schedule_ui_safe_operation(const std::function<void()>& operation);
    
    /**
     * @brief Get current UI perforance metrics
     */
    struct UIMetrics {
        float average_fps;
        size_t total_frame_drops;
        double average_frame_time_ms;
        bool is_ui_smooth;
    };
    UIMetrics get_ui_metrics() const;
};

/**
 * @brief Mobile-specific cache optimization strategies
 */
class MobileCacheOptimizer {
private:
    struct CacheConfig {
        size_t l1_size_kb;
        size_t l2_size_kb; 
        size_t l3_size_kb;
        size_t cache_line_size;
        size_t associativity;
    } cache_config_;
    
public:
    MobileCacheOptimizer();
    
    /**
     * @brief Detect mobile CPU cache configuration
     */
    bool detect_cache_configuration();
    
    /**
     * @brief Optimize parameter layout for cache efficiency
     */
    void optimize_parameter_layout(std::vector<TensorPtr>& parameters);
    
    /**
     * @brief Get cache-optimal access pattern
     */
    std::vector<size_t> get_cache_optimal_access_order(const std::vector<size_t>& param_ids);
    
    /**
     * @brief Align parameter to cache boundaries
     */
    size_t get_cache_aligned_offset(size_t base_offset, size_t param_size) const;
    
    /**
     * @brief Prefetch cache lines for parameters
     */
    void prefetch_parameter_cache_lines(const TensorPtr& param);
};

/**
 * @brief Comprehensive mobile optimization coordinator
 */
class MobileOptimizationCoordinator {
private:
    std::unique_ptr<AndroidMemoryManager> android_manager_;
    std::unique_ptr<ZRAMOptimizer> zram_optimizer_;
    std::unique_ptr<BigLittleOptimizer> big_little_optimizer_;
    std::unique_ptr<MobileGPUOptimizer> gpu_optimizer_;
    std::unique_ptr<BatteryAwareScheduler> battery_scheduler_;
    std::unique_ptr<NetworkAwareOffloader> network_offloader_;
    std::unique_ptr<UIResponsivenessOptimizer> ui_optimizer_;
    std::unique_ptr<MobileCacheOptimizer> cache_optimizer_;
    
    MobileSystemState current_state_;
    MobileOptimizationStrategy current_strategy_;

public:
    MobileOptimizationCoordinator();
    ~MobileOptimizationCoordinator();
    
    /**
     * @brief Initialize all mobile optimizations
     */
    bool initialize_mobile_optimizations(JNIEnv* env = nullptr, jobject activity = nullptr);
    
    /**
     * @brief Coordinate all mobile optimizations for parameter operation
     */
    void coordinate_mobile_optimization(size_t param_id, const std::string& operation_type);
    
    /**
     * @brief Adapt all optimizations to current mobile conditions
     */
    void adapt_to_current_conditions();
    
    /**
     * @brief Get comprehensive mobile optimization report
     */
    struct MobileOptimizationReport {
        // Memory optimizations
        bool android_integration_active;
        bool zram_optimization_active;
        double memory_pressure_level;
        
        // Perforance optimizations  
        bool big_little_optimization_active;
        bool gpu_optimization_active;
        bool cache_optimization_active;
        
        // Power optimizations
        BatteryAwareScheduler::BatteryStrategy battery_strategy;
        bool thermal_throttling_active;
        size_t estimated_power_savings_mw;
        
        // UI optimizations
        bool ui_optimization_active;
        float current_fps_impact;
        
        // Network optimizations
        bool network_optimization_active;
        size_t data_usage_savings_bytes;
        
        // Overall effectiveness
        double optimization_effectiveness_score; // 0.0 to 1.0
    };
    MobileOptimizationReport get_optimization_report() const;
    
    /**
     * @brief Emergency mobile optimization for critical situations
     */
    void trigger_emergency_mobile_optimization();
};

} // namespace memory
} // namespace ops
