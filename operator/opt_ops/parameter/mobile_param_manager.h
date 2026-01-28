/**
 * @file mobile_param_manager.h
 * @brief Mobile-optimized parameter management system inspired by DeepSpeed ZeRO
 * 
 * This file implementss a mobile-specific parameter management system that optimizes
 * memory usage for training and fine-tuning on resource-constrained devices.
 * Key features:
 * - Parameter partitioning for reduced memory footprint
 * - Dynamic parameter loading/unloading
 * - Memory pool management with defragmentation
 * - CPU/Storage offloading for large models
 * - LoRA-aware parameter management
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <chrono>
#include <deque>
#include "../core/tensor.h"
#include "../core/device.h"

namespace ops {
namespace memory {

// Import necessary types from ops namespace to avoid qualification issues
using ops::TensorPtr;
using ops::Tensor;
using ops::Device;
using ops::DType;

// Forward declarations for missing critical components
class ParameterQuantizer;
class PinnedMemoryManager;
class ContiguousMemoryAllocator;
class ParameterPersistenceManager;
class ParameterReuseTracker;
class MobilePlatforOptimizer;
class AdvancedPrefetchSystem;
class AsyncIOManager;

// Forward declarations for mobile-specific optimization components  
class AndroidMemoryManager;
class ZRAMOptimizer;
class BigLittleOptimizer;
class MobileGPUOptimizer;
class BatteryAwareScheduler;
class NetworkAwareOffloader;
class UIResponsivenessOptimizer;
class MobileCacheOptimizer;
class MobileOptimizationCoordinator;

/**
 * @brief Quantization modes for mobile optimization
 */
enum class QuantizationMode {
    NONE = 0,              // No quantization
    FP16 = 1,              // Half precision
    INT8 = 2,              // 8-bit integer quantization
    INT4 = 3,              // 4-bit integer quantization
    DYNAMIC = 4,           // Dynamic quantization based on parameter importance
    MIXED = 5              // Mixed precision (different modes for different parameters)
};

// ============= CRITICAL MOBILE-SPECIFIC MISSING COMPONENTS =============

/**
 * @brief Mobile system state for adaptive optimization
 */
enum class MobileSystemState {
    FOREGROUND_ACTIVE,      // App is actively used
    FOREGROUND_INACTIVE,    // App is visible but not active
    BACKGROUND,            // App is in background
    LOW_MEMORY_WARNING,    // System low memory warning
    THERMAL_WARNING,       // Device thermal warning
    BATTERY_LOW            // Battery critically low
};

/**
 * @brief Mobile GPU vendor for hardware-specific optimizations
 */
enum class MobileGPUVendor {
    UNKNOWN = 0,
    QUALCOMM_ADRENO = 1,   // Snapdragon devices
    ARM_MALI = 2,          // Samsung, MediaTek devices
    IMG_POWERVR = 3,       // Some older devices
    APPLE_GPU = 4          // iOS devices
};

/**
 * @brief Memory pressure levels for mobile optimization
 */
enum class MemoryPressureLevel {
    NORMAL = 0,            // < 60% memory usage
    MODERATE = 1,          // 60-80% memory usage
    HIGH = 2,              // 80-90% memory usage  
    CRITICAL = 3           // > 90% memory usage
};

/**
 * @brief Mobile-specific memory optimization strategies
 */
enum class MobileOptimizationStrategy {
    PERFORMANCE_FIRST,     // Prioritize speed over memory
    BALANCED,             // Balance speed and memory
    MEMORY_FIRST,         // Prioritize memory over speed
    BATTERY_SAVER,        // Minimize power consumption
    THERMAL_AWARE         // Prevent overheating
};

/**
 * @brief Parameter status enum for tracking parameter lifecycle
 */
enum class ParameterStatus {
    NOT_AVAILABLE,     // Parameter not loaded in memory
    LOADING,          // Parameter currently being loaded
    AVAILABLE,        // Parameter available for computation
    UNLOADING,        // Parameter being unloaded
    OFFLOADED         // Parameter stored in secondary storage
};

/**
 * @brief Memory tier enum for hierarchical storage
 */
enum class MemoryTier {
    GPU_MEMORY,       // Primary GPU/mobile GPU memory (fastest)
    CPU_MEMORY,       // System RAM (medium speed)
    STORAGE           // Persistent storage/file system (slowest)
};

/**
 * @brief Parameter partition inforation with mobile-specific metadata
 */
struct ParameterPartition {
    size_t partition_id;
    size_t start_offset;
    size_t size_bytes;
    MemoryTier current_tier;
    ParameterStatus status;
    TensorPtr tensor;
    std::string storage_path;  // For persistent storage
    
    // ===== MOBILE-SPECIFIC ADDITIONS =====
    
    // Access pattern analytics
    size_t access_count;
    std::chrono::steady_clock::time_point last_access_time;
    std::chrono::steady_clock::time_point creation_time;
    double access_frequency_score;      // Higher = more frequently accessed
    
    // Memory optimization metadata
    bool is_quantized;
    QuantizationMode quantization_mode;
    float compression_ratio;            // Actual compression achieved
    size_t original_size_bytes;         // Before compression
    
    // Mobile-specific flags
    bool is_persistent;                 // Never evict (critical parameters)
    bool is_thermal_sensitive;          // Reduce access when overheating
    bool is_battery_optimized;          // Use power-efficient access patterns
    bool prefer_cpu_over_gpu;          // CPU processing preferred for this param
    
    // Cache optimization
    size_t cache_line_alignment_offset; // Offset for optimal cache alignment
    bool is_prefetch_candidate;         // Good candidate for prefetching
    int priority_level;                 // 0=lowest, 10=highest priority
    
    // Hardware-specific metadata
    MobileGPUVendor optimal_gpu_vendor; // Best GPU vendor for this parameter
    bool supports_neon_acceleration;    // Can use ARM NEON SIMD
    
    ParameterPartition(size_t id, size_t offset, size_t size) 
        : partition_id(id), start_offset(offset), size_bytes(size),
          current_tier(MemoryTier::CPU_MEMORY), 
          status(ParameterStatus::NOT_AVAILABLE),
          // Mobile-specific initializations
          access_count(0), access_frequency_score(0.0),
          is_quantized(false), quantization_mode(QuantizationMode::NONE),
          compression_ratio(1.0f), original_size_bytes(size),
          is_persistent(false), is_thermal_sensitive(false),
          is_battery_optimized(false), prefer_cpu_over_gpu(false),
          cache_line_alignment_offset(0), is_prefetch_candidate(true),
          priority_level(5), optimal_gpu_vendor(MobileGPUVendor::UNKNOWN),
          supports_neon_acceleration(false) {
        
        creation_time = std::chrono::steady_clock::now();
        last_access_time = creation_time;
    }
};

/**
 * @brief Memory statistics for monitoring
 */
struct MemoryStats {
    size_t total_params_size;
    size_t active_params_size;
    size_t gpu_memory_used;
    size_t cpu_memory_used;
    size_t storage_used;
    double memory_fragmentation_ratio;
    size_t total_alloc_requests;
    size_t cache_hit_count;
    size_t cache_miss_count;
};

/**
 * @brief Configuration for mobile parameter manager
 */
struct MobileParamConfig {
    size_t max_gpu_memory_mb = 1024;      // Maximum GPU memory (1GB default)
    size_t max_cpu_memory_mb = 2048;      // Maximum CPU memory (2GB default)
    size_t partition_size_mb = 64;        // Default partition size (64MB)
    size_t prefetch_buffer_mb = 128;      // Prefetch buffer size
    size_t min_free_memory_mb = 256;      // Minimum free memory to maintain
    bool enable_storage_offload = true;    // Enable storage offloading
    bool enable_compression = false;       // Enable parameter compression
    bool enable_prefetch = true;          // Enable parameter prefetching
    double eviction_threshold = 0.8;      // Memory usage threshold for eviction
    std::string storage_path = "./param_cache";  // Storage path for offloaded params
    
    // ======= MISSING CRITICAL FEATURES =======
    
    // Quantization settings
    bool enable_quantization = true;      // Enable parameter quantization
    QuantizationMode default_quantization = QuantizationMode::INT8;
    bool enable_nontrainable_quantization = true;  // Quantize non-trainable params more aggressively
    
    // Pin memory settings  
    bool enable_pin_memory = true;        // Enable pinned memory for GPU transfers
    size_t max_pinned_memory_mb = 256;    // Maximum pinned memory allocation
    
    // Persistence thresholds (CRITICAL MISSING)
    size_t param_persistence_threshold = 100000;   // Don't partition params smaller than 100K elements (~400KB)
    size_t model_persistence_threshold = 1000000;  // Max 1M elements (~4MB) of persistent params
    
    // Reuse distance tracking (CRITICAL MISSING)
    size_t max_reuse_distance = 1000000000;        // Maximum reuse distance in elements
    bool enable_reuse_tracking = true;             // Enable parameter reuse tracking
    
    // Advanced prefetch settings (PERFORMANCE CRITICAL)
    size_t prefetch_bucket_size = 50000000;       // 50MB prefetch buckets
    size_t max_prefetch_buckets = 4;              // Maximum concurrent prefetch buckets
    bool enable_predictive_prefetch = true;       // Enable ML-based prefetch prediction
    
    // Mobile-specific optimizations (MOBILE CRITICAL)
    bool enable_thermal_monitoring = true;        // Monitor device temperature
    float thermal_throttle_threshold = 70.0f;     // Throttle at 70°C
    bool enable_low_power_mode = false;           // Low power mode for battery
    bool enable_arm_neon_optimizations = true;    // ARM NEON SIMD optimizations
    
    // Async I/O settings (PERFORMANCE CRITICAL)
    size_t async_io_threads = 2;                  // Number of async I/O threads
    bool enable_batch_io = true;                  // Enable batch I/O operations
    size_t io_buffer_size_mb = 64;               // I/O buffer size
    
    // Memory alignment (CACHE CRITICAL)
    size_t memory_alignment = 64;                 // Memory alignment in bytes (64 for cache lines)
    bool enable_memory_defragmentation = true;    // Enable automatic defragmentation
    size_t defrag_threshold_mb = 128;            // Defragment when fragmentation > 128MB
    
    // ======= CRITICAL MOBILE-SPECIFIC MISSING SETTINGS =======
    
    // Android/iOS system integration (MOBILE CRITICAL)
    bool enable_system_memory_monitoring = true;  // Monitor Android Low Memory Killer
    bool enable_lifecycle_awareness = true;       // Respond to app lifecycle changes
    bool enable_zram_optimization = true;         // Optimize for ZRAM/swap files
    size_t max_zram_usage_mb = 512;              // Maximum ZRAM usage (default 512MB)
    bool prioritize_foreground_allocation = true; // Prioritize when app is foreground
    
    // Mobile hardware detection and optimization (HARDWARE CRITICAL)
    bool enable_auto_hardware_detection = true;   // Auto-detect mobile GPU vendor
    MobileGPUVendor target_gpu_vendor = MobileGPUVendor::UNKNOWN; // Override auto-detection
    bool enable_big_little_optimization = true;   // Optimize for big.LITTLE CPU architecture
    bool enable_memory_bandwidth_adaptation = true; // Adapt to memory bandwidth
    
    // Advanced mobile memory management (MEMORY CRITICAL)
    bool enable_oom_killer_avoidance = true;      // Avoid Android OOM killer
    double memory_pressure_warning_threshold = 0.75; // Warn at 75% system memory usage
    double memory_pressure_critical_threshold = 0.9;  // Critical at 90% system memory usage
    size_t emergency_free_memory_mb = 128;        // Always keep 128MB free for system
    
    // Mobile perforance optimization (PERFORMANCE CRITICAL)
    bool enable_fps_aware_scheduling = true;      // Avoid blocking UI rendering
    size_t max_blocking_time_ms = 16;            // Max 16ms blocking (60 FPS)
    bool enable_cpu_cache_optimization = true;    // Optimize for CPU cache hierarchy
    bool enable_memory_prefetch_hints = true;     // Use ARM prefetch instructions
    
    // Power and thermal management (MOBILE CRITICAL)
    bool enable_dynamic_voltage_scaling = true;   // DVFS-aware optimizations
    float cpu_utilization_target = 0.8f;         // Target 80% CPU utilization
    bool enable_idle_power_optimization = true;   // Optimize power in idle periods
    size_t thermal_monitoring_interval_ms = 1000; // Check temperature every 1s
    
    // Mobile-specific data types and precision (PRECISION CRITICAL)
    bool enable_bfloat16_support = true;          // Enable BFloat16 if supported
    bool enable_mixed_precision_inference = true; // Mixed precision for inference
    bool enable_dynamic_precision_scaling = true; // Adjust precision based on system state
    
    // Network and connectivity awareness (MOBILE SPECIFIC)
    bool enable_network_aware_offloading_basic = true;  // Avoid offloading on cellular
    bool prefer_local_storage_on_metered = true;  // Use local storage on metered connections
    size_t max_cellular_transfer_mb = 100;       // Max cellular data transfer
    
    // Mobile UI/UX integration (UX CRITICAL)
    bool maintain_ui_responsiveness = true;       // Never block UI thread
    size_t ui_thread_max_wait_ms = 5;            // Max 5ms wait for UI operations
    bool enable_background_processing_limits = true; // Limit processing when backgrounded
    
    // Advanced mobile caching strategies (CACHE CRITICAL)
    bool enable_cache_line_prefetching_basic = true;    // Prefetch entire cache lines
    size_t l1_cache_size_kb = 32;                // L1 cache size (typical mobile)
    size_t l2_cache_size_kb = 256;               // L2 cache size (typical mobile) 
    size_t l3_cache_size_kb = 2048;              // L3 cache size (typical mobile)
    bool enable_cache_conscious_allocation = true; // Allocate considering cache sizes
    
    // Mobile-specific memory pattern optimization (PATTERN CRITICAL)
    bool enable_sequential_access_optimization = true; // Optimize sequential access patterns
    bool enable_spatial_locality_optimization = true;  // Group related parameters
    bool enable_temporal_locality_optimization = true; // Keep recent parameters hot
    size_t access_pattern_history_size = 1000;   // Track last 1000 access patterns
    
    // Mobile debugging and profiling
    bool enable_detailed_mobile_profiling = false; // Detailed mobile perforance profiling
    bool log_memory_pressure_events = true;       // Log memory pressure changes
    bool log_thermal_events = true;               // Log thermal throttling events
    bool enable_crash_safe_operation = true;      // Graceful handling of system kills
    std::string mobile_profile_output_path = "./mobile_profile.json"; // Mobile profiling output
    
    // ======= MOBILE-SPECIFIC OPTIMIZATION COMPONENTS (FINAL MISSING PIECES) =======
    
    // Android integration (ANDROID CRITICAL)
    bool enable_android_integration = true;       // Enable Android-specific optimizations
    bool enable_jni_integration = true;           // Enable JNI for Android integration
    bool request_java_gc_on_pressure = true;      // Request Java GC during memory pressure
    
    // ZRAM and swap optimization (MEMORY CRITICAL)
    bool enable_zram_aware_compression = true;    // Optimize compression for ZRAM
    bool enable_swap_file_creation = false;       // Create swap file if needed (requires root)
    size_t max_swap_usage_mb = 256;              // Maximum swap usage
    
    // Big.LITTLE CPU optimization (PERFORMANCE CRITICAL)
    bool enable_big_little_awareness = true;      // Optimize for big.LITTLE architecture
    bool prefer_little_cores_for_memory = true;   // Use little cores for memory operations
    bool prefer_big_cores_for_compute = true;     // Use big cores for compute operations
    
    // Mobile GPU vendor-specific optimization (GPU CRITICAL)
    bool enable_gpu_vendor_optimization = true;   // Enable GPU vendor-specific optimizations
    bool enable_unified_memory_optimization = true; // Optimize for UMA if available
    size_t gpu_transfer_chunk_size_kb = 64;      // Optimal GPU transfer chunk size
    
    // Battery-aware optimization (POWER CRITICAL)  
    bool enable_battery_aware_scheduling = true;  // Schedule operations based on battery
    size_t battery_critical_threshold = 15;       // Critical battery level (%)
    size_t max_power_budget_mw = 1000;           // Maximum power budget (1W)
    
    // Network-aware offloading (CONNECTIVITY CRITICAL)
    bool enable_network_aware_offloading = true;  // Consider network state for offloading
    size_t cellular_data_limit_mb = 100;         // Daily cellular data limit
    double cellular_compression_factor = 0.5;     // Extra compression on cellular
    
    // UI responsiveness (UX CRITICAL)
    bool enable_ui_responsiveness_optimization = true; // Optimize for UI smoothness
    float target_fps = 60.0f;                    // Target UI frame rate
    size_t max_ui_blocking_time_ms = 8;          // Maximum UI blocking time (8ms for 120Hz)
    
    // Mobile cache optimization (CACHE CRITICAL)
    bool enable_mobile_cache_optimization = true; // Enable mobile-specific cache optimizations
    bool enable_cache_line_prefetching = true;    // Prefetch entire cache lines
    bool enable_cache_aware_parameter_layout = true; // Layout parameters for cache efficiency
    
    // Comprehensive mobile coordinator (COORDINATION CRITICAL)
    bool enable_mobile_optimization_coordinator = true; // Enable comprehensive mobile optimization
    size_t optimization_update_interval_ms = 5000;      // Update optimization strategy every 5s
    double optimization_aggressiveness = 0.7;           // 0.0 = conservative, 1.0 = aggressive
};

/**
 * @brief LRU cache for parameter access tracking
 */
class ParameterCache {
private:
    struct CacheNode {
        size_t param_id;
        size_t last_access_time;
        size_t access_count;
        CacheNode* prev;
        CacheNode* next;
        
        CacheNode(size_t id) : param_id(id), last_access_time(0), 
                               access_count(0), prev(nullptr), next(nullptr) {}
    };
    
    std::unordered_map<size_t, CacheNode*> cache_map_;
    CacheNode* head_;
    CacheNode* tail_;
    size_t max_size_;
    size_t current_time_;
    mutable std::mutex cache_mutex_;

public:
    explicit ParameterCache(size_t max_size);
    ~ParameterCache();
    
    void access(size_t param_id);
    std::vector<size_t> get_eviction_candidates(size_t count);
    void remove(size_t param_id);
    double get_hit_rate() const;
    void clear();

private:
    void move_to_head(CacheNode* node);
    void remove_node(CacheNode* node);
};

/**
 * @brief Asynchronous parameter loader/unloader
 */
class AsyncParameterHandler {
private:
    std::thread worker_thread_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool stop_flag_;

public:
    AsyncParameterHandler();
    ~AsyncParameterHandler();
    
    void schedule_load(size_t param_id, const std::function<void()>& load_fn);
    void schedule_unload(size_t param_id, const std::function<void()>& unload_fn);
    void wait_for_completion();

private:
    void worker_loop();
};

/**
 * @brief Main mobile parameter manager class
 * 
 * This class manages model parameters with mobile-specific optimizations:
 * 1. Memory-efficient parameter storage and access
 * 2. Dynamic loading/unloading based on usage patterns
 * 3. Hierarchical storage across GPU/CPU/Storage tiers
 * 4. Automatic memory defragmentation
 * 5. LoRA parameter integration
 */
class MobileParameterManager {
private:
    MobileParamConfig config_;
    std::vector<std::unique_ptr<ParameterPartition>> partitions_;
    std::unordered_map<std::string, size_t> param_name_to_id_;
    std::unique_ptr<ParameterCache> param_cache_;
    std::unique_ptr<AsyncParameterHandler> async_handler_;
    
    // ======= CRITICAL MISSING COMPONENTS =======
    
    // Quantization system (CRITICAL MISSING)
    std::unique_ptr<ParameterQuantizer> parameter_quantizer_;
    std::unordered_map<size_t, QuantizationMode> param_quantization_modes_;
    std::atomic<size_t> total_quantization_savings_;
    
    // Pin memory system (PERFORMANCE CRITICAL)
    std::unique_ptr<PinnedMemoryManager> pinned_memory_manager_;
    
    // Advanced memory allocator (MEMORY CRITICAL)
    std::unique_ptr<ContiguousMemoryAllocator> contiguous_allocator_;
    std::atomic<size_t> fragmentation_events_;
    
    // Parameter persistence system (MEMORY CRITICAL)
    std::unique_ptr<ParameterPersistenceManager> persistence_manager_;
    std::unordered_set<size_t> always_resident_params_;  // Never partition these
    
    // Reuse distance tracking (PERFORMANCE CRITICAL)
    std::unique_ptr<ParameterReuseTracker> reuse_tracker_;
    std::atomic<size_t> reuse_prediction_hits_;
    std::atomic<size_t> reuse_prediction_misses_;
    
    // Mobile platfor optimizations (MOBILE CRITICAL)
    std::unique_ptr<MobilePlatforOptimizer> platfor_optimizer_;
    std::atomic<bool> thermal_monitoring_active_;
    std::atomic<size_t> thermal_throttle_count_;
    
    // Advanced prefetch system (PERFORMANCE CRITICAL)
    std::unique_ptr<AdvancedPrefetchSystem> advanced_prefetch_;
    std::atomic<size_t> prefetch_hit_count_;
    std::atomic<size_t> prefetch_miss_count_;
    
    // Async I/O system (PERFORMANCE CRITICAL)
    std::unique_ptr<AsyncIOManager> async_io_manager_;
    std::atomic<size_t> async_io_operations_;
    std::atomic<size_t> async_io_bytes_transferred_;
    
    // Memory management (ENHANCED)
    void* gpu_memory_pool_;
    void* cpu_memory_pool_;
    size_t gpu_memory_offset_;
    size_t cpu_memory_offset_;
    
    // Statistics and monitoring (ENHANCED)
    MemoryStats memory_stats_;
    mutable std::mutex stats_mutex_;
    mutable std::mutex manager_mutex_;
    
    // Enhanced prefetching
    std::vector<size_t> prefetch_queue_;
    std::thread prefetch_thread_;
    bool prefetch_enabled_;
    
    // Memory pressure monitoring
    std::atomic<double> current_memory_pressure_;
    std::chrono::steady_clock::time_point last_memory_check_;
    
    // ======= CRITICAL MOBILE-SPECIFIC MISSING MEMBERS =======
    
    // Mobile system state monitoring (MOBILE CRITICAL)
    std::atomic<MobileSystemState> current_system_state_;
    std::atomic<MemoryPressureLevel> current_memory_pressure_level_;
    MobileOptimizationStrategy current_optimization_strategy_;
    
    // Mobile hardware inforation (HARDWARE CRITICAL)
    MobileGPUVendor detected_gpu_vendor_;
    size_t detected_cpu_cores_;
    size_t detected_big_cores_;
    size_t detected_little_cores_;
    std::vector<size_t> cpu_cache_sizes_;           // L1, L2, L3 cache sizes
    size_t memory_bandwidth_mbps_;                  // Detected memory bandwidth
    
    // System integration monitoring (SYSTEM CRITICAL)
    std::thread system_monitor_thread_;
    std::atomic<bool> system_monitoring_active_;
    std::atomic<size_t> system_available_memory_mb_;
    std::atomic<float> system_memory_pressure_;
    std::atomic<bool> is_app_foreground_;
    std::atomic<bool> is_low_memory_warning_;
    
    // Mobile-specific perforance monitoring (PERFORMANCE CRITICAL)
    std::atomic<float> current_fps_impact_;         // Impact on UI frame rate
    std::atomic<size_t> ui_thread_blocked_count_;   // Times UI thread was blocked
    std::atomic<double> average_operation_latency_ms_;
    std::chrono::steady_clock::time_point last_ui_interaction_;
    
    // Power and thermal monitoring (POWER CRITICAL)
    std::atomic<float> current_cpu_utilization_;
    std::atomic<size_t> power_consumption_mw_;      // Estimated power consumption
    std::atomic<bool> is_charging_;
    std::atomic<size_t> battery_level_percent_;
    std::thread power_monitor_thread_;
    std::atomic<bool> power_monitoring_active_;
    
    // Mobile-specific caching and prefetching (CACHE CRITICAL)
    std::unordered_map<size_t, size_t> parameter_cache_affinity_; // param_id -> preferred cache level
    std::vector<size_t> cache_line_aligned_params_; // Parameters optimally aligned
    std::atomic<size_t> cache_miss_count_;
    std::atomic<size_t> cache_hit_count_;
    
    // Network and connectivity state (CONNECTIVITY CRITICAL)
    std::atomic<bool> is_on_cellular_;
    std::atomic<bool> is_network_metered_;
    std::atomic<size_t> data_usage_bytes_;
    
    // Mobile access pattern optimization (PATTERN CRITICAL)
    struct AccessPattern {
        size_t param_id;
        std::chrono::steady_clock::time_point access_time;
        MemoryTier access_tier;
        size_t access_size;
    };
    std::deque<AccessPattern> access_pattern_history_;
    mutable std::mutex access_pattern_mutex_;
    
    // Mobile-specific error handling (RELIABILITY CRITICAL)
    std::atomic<size_t> oom_avoidance_count_;       // Times we avoided OOM
    std::atomic<size_t> thermal_throttle_events_;
    std::atomic<size_t> system_kill_recoveries_;    // Recoveries from system kills
    std::function<void()> emergency_cleanup_callback_;
    
    // ======= MOBILE-SPECIFIC OPTIMIZATION COMPONENTS (FINAL ADDITIONS) =======
    
    // Comprehensive mobile optimization coordinator (COORDINATION CRITICAL)
    // Note: Using the full class definition from mobile_specific_optimizations.h
    std::unique_ptr<::ops::memory::MobileOptimizationCoordinator> mobile_coordinator_;
    std::atomic<bool> mobile_optimization_active_;
    std::atomic<double> mobile_optimization_effectiveness_;

public:
    explicit MobileParameterManager(const MobileParamConfig& config);
    ~MobileParameterManager();
    
    /**
     * @brief Register a model parameter for management
     * @param name Parameter name (e.g., "transforer.layer.0.weight")
     * @param tensor The parameter tensor
     * @param is_trainable Whether this parameter requires gradients
     * @return Parameter ID for future reference
     */
    size_t register_parameter(const std::string& name, const TensorPtr& tensor, bool is_trainable = true);
    
    /**
     * @brief Get a parameter tensor for computation
     * @param param_id Parameter ID
     * @param hint Access pattern hint for prefetching
     * @return Parameter tensor (automatically loaded if needed)
     */
    TensorPtr get_parameter(size_t param_id, const std::string& hint = "");
    TensorPtr get_parameter(const std::string& name, const std::string& hint = "");
    
    /**
     * @brief Release a parameter after computation
     * @param param_id Parameter ID
     * @param mark_dirty Whether the parameter was modified
     */
    void release_parameter(size_t param_id, bool mark_dirty = false);
    
    /**
     * @brief Force load parameters into specified memory tier
     * @param param_ids List of parameter IDs to load
     * @param tier Target memory tier
     */
    void prefetch_parameters(const std::vector<size_t>& param_ids, MemoryTier tier = MemoryTier::CPU_MEMORY);
    
    /**
     * @brief Evict parameters to make room for new ones
     * @param bytes_needed Amount of memory needed
     * @param preferred_tier Preferred memory tier to free
     */
    void evict_parameters(size_t bytes_needed, MemoryTier preferred_tier = MemoryTier::CPU_MEMORY);
    
    /**
     * @brief Perfor memory defragmentation
     * @param tier Memory tier to defragment
     */
    void defragment_memory(MemoryTier tier);
    
    /**
     * @brief Save all parameters to persistent storage
     * @param checkpoint_path Path to save checkpoint
     */
    void save_checkpoint(const std::string& checkpoint_path);
    
    /**
     * @brief Load parameters from persistent storage
     * @param checkpoint_path Path to load checkpoint from
     */
    void load_checkpoint(const std::string& checkpoint_path);
    
    /**
     * @brief Get current memory statistics
     */
    MemoryStats get_memory_stats() const;
    
    /**
     * @brief Get current configuration
     */
    const MobileParamConfig& get_config() const;
    
    /**
     * @brief Optimize memory layout for inference
     */
    void optimize_for_inference();
    
    /**
     * @brief Optimize memory layout for training
     */
    void optimize_for_training();
    
    /**
     * @brief Set memory pressure callback
     * @param callback Function to call when memory pressure is high
     */
    void set_memory_pressure_callback(const std::function<void(double)>& callback);
    
    // ======= CRITICAL MISSING METHODS =======
    
    /**
     * @brief Enable/disable quantization for specific parameter
     * @param param_id Parameter ID
     * @param mode Quantization mode to use
     * @return Estimated memory savings in bytes
     */
    size_t enable_parameter_quantization(size_t param_id, QuantizationMode mode = QuantizationMode::INT8);
    
    /**
     * @brief Get quantized parameter with automatic dequantization
     * @param param_id Parameter ID
     * @param auto_dequantize If true, automatically dequantize for computation
     * @return Parameter tensor (dequantized if requested)
     */
    TensorPtr get_quantized_parameter(size_t param_id, bool auto_dequantize = true);
    
    /**
     * @brief Check if parameter should be persistent (never partitioned)
     * @param param_id Parameter ID
     * @return True if parameter should remain always resident
     */
    bool should_be_persistent_parameter(size_t param_id) const;
    
    /**
     * @brief Force parameter to be persistent (always resident)
     * @param param_id Parameter ID
     * @param persistent True to make persistent, false to allow partitioning
     */
    void set_parameter_persistence(size_t param_id, bool persistent);
    
    /**
     * @brief Predict next parameters to be accessed
     * @param current_param_id Currently accessed parameter
     * @param max_predictions Maximum number of predictions
     * @return Predicted parameter IDs in order of likelihood
     */
    std::vector<size_t> predict_next_parameter_accesses(size_t current_param_id, size_t max_predictions = 3);
    
    /**
     * @brief Enable thermal throttling when device gets hot
     * @param enable True to enable thermal monitoring
     */
    void enable_thermal_throttling(bool enable = true);
    
    /**
     * @brief Get current device thermal state
     * @return Temperature in Celsius, -1 if unavailable
     */
    float get_current_temperature() const;
    
    /**
     * @brief Check if system is currently thermally throttling
     */
    bool is_thermal_throttling() const;
    
    /**
     * @brief Enable low power mode for battery conservation
     * @param enable True to enable low power optimizations
     */
    void enable_low_power_mode(bool enable = true);
    
    /**
     * @brief Trigger memory defragmentation
     * @param min_savings_mb Minimum memory to recover to proceed with defrag
     * @return Amount of memory recovered in bytes
     */
    size_t trigger_memory_defragmentation(size_t min_savings_mb = 64);
    
    /**
     * @brief Get comprehensive memory optimization statistics
     */
    struct OptimizationStats {
        // Quantization stats
        size_t quantized_parameters;
        size_t quantization_memory_saved;
        double average_compression_ratio;
        
        // Reuse tracking stats
        double reuse_prediction_accuracy;
        size_t parameters_released_early;
        
        // Thermal stats
        size_t thermal_throttle_events;
        float peak_temperature;
        
        // Prefetch stats
        double prefetch_hit_rate;
        size_t prefetch_memory_saved;
        
        // I/O stats
        double average_io_speed_mbps;
        size_t total_io_operations;
        
        // Fragmentation stats
        size_t defragmentation_events;
        size_t fragmentation_memory_recovered;
    };
    OptimizationStats get_optimization_stats() const;
    
    /**
     * @brief Export optimization report for analysis
     * @param report_path Path to save detailed optimization report
     */
    void export_optimization_report(const std::string& report_path) const;
    
    // ======= CRITICAL MOBILE-SPECIFIC MISSING METHODS =======
    
    /**
     * @brief Set mobile system state (foreground/background/etc.)
     * @param state Current system state
     * @param trigger_optimization Whether to immediately optimize for new state
     */
    void set_system_state(MobileSystemState state, bool trigger_optimization = true);
    
    /**
     * @brief Get current mobile system state
     */
    MobileSystemState get_current_system_state() const;
    
    /**
     * @brief Set optimization strategy based on current conditions
     * @param strategy Optimization strategy to use
     */
    void set_optimization_strategy(MobileOptimizationStrategy strategy);
    
    /**
     * @brief Auto-detect mobile hardware capabilities
     * @return True if detection successful
     */
    bool auto_detect_mobile_hardware();
    
    /**
     * @brief Get detected mobile GPU vendor
     */
    MobileGPUVendor get_detected_gpu_vendor() const;
    
    /**
     * @brief Get CPU architecture inforation
     */
    struct CPUArchInfo {
        size_t total_cores;
        size_t big_cores;
        size_t little_cores;
        std::vector<size_t> cache_sizes_kb;
        size_t memory_bandwidth_mbps;
    };
    CPUArchInfo get_cpu_architecture_info() const;
    
    /**
     * @brief Monitor system memory pressure and respond
     * @param enable True to enable continuous monitoring
     */
    void enable_system_memory_monitoring(bool enable = true);
    
    /**
     * @brief Get current system memory pressure level
     */
    MemoryPressureLevel get_current_memory_pressure_level() const;
    
    /**
     * @brief Force immediate memory pressure check and optimization
     */
    void check_and_optimize_memory_pressure();
    
    /**
     * @brief Set app lifecycle state (foreground/background)
     * @param is_foreground True if app is currently foreground
     */
    void set_app_lifecycle_state(bool is_foreground);
    
    /**
     * @brief Handle Android Low Memory Warning
     */
    void handle_low_memory_warning();
    
    /**
     * @brief Handle system OOM situation
     * @param aggressive_cleanup True to perfor aggressive cleanup
     */
    void handle_oom_pressure(bool aggressive_cleanup = true);
    
    /**
     * @brief Optimize for UI responsiveness
     * @param target_fps Target frame rate to maintain (default: 60)
     */
    void optimize_for_ui_responsiveness(float target_fps = 60.0f);
    
    /**
     * @brief Check if operation would block UI thread
     * @param estimated_duration_ms Estimated operation duration
     * @return True if operation would cause UI blocking
     */
    bool would_block_ui_thread(size_t estimated_duration_ms) const;
    
    /**
     * @brief Enable power-aware optimizations
     * @param enable True to enable power optimization
     */
    void enable_power_optimization(bool enable = true);
    
    /**
     * @brief Set battery state inforation
     * @param level_percent Battery level (0-100)
     * @param is_charging Whether device is charging
     */
    void set_battery_state(size_t level_percent, bool is_charging);
    
    /**
     * @brief Get estimated power consumption
     * @return Power consumption in milliwatts
     */
    size_t get_estimated_power_consumption() const;
    
    /**
     * @brief Enable cache-conscious parameter allocation
     * @param enable True to enable cache optimization
     */
    void enable_cache_conscious_allocation(bool enable = true);
    
    /**
     * @brief Optimize parameter for specific cache level
     * @param param_id Parameter ID
     * @param target_cache_level Target cache level (1=L1, 2=L2, 3=L3)
     */
    void optimize_parameter_for_cache(size_t param_id, size_t target_cache_level);
    
    /**
     * @brief Set network connectivity state
     * @param is_cellular True if on cellular connection
     * @param is_metered True if connection is metered
     */
    void set_network_state(bool is_cellular, bool is_metered);
    
    /**
     * @brief Get mobile-specific perforance metrics
     */
    struct MobilePerforanceMetrics {
        // UI impact metrics
        float average_fps_impact;
        size_t ui_thread_blocked_count;
        double average_latency_ms;
        
        // System integration metrics
        size_t oom_avoidance_count;
        size_t thermal_throttle_events;
        MemoryPressureLevel current_pressure_level;
        
        // Power metrics
        size_t estimated_power_consumption_mw;
        float cpu_utilization;
        bool is_thermally_throttling;
        
        // Cache metrics
        double cache_hit_rate;
        size_t cache_optimized_parameters;
        
        // Network metrics
        size_t data_usage_bytes;
        bool is_network_optimized;
    };
    MobilePerforanceMetrics get_mobile_perforance_metrics() const;
    
    /**
     * @brief Perfor emergency cleanup to avoid system kill
     * @param preserve_critical_params True to preserve critical parameters
     */
    void perfor_emergency_cleanup(bool preserve_critical_params = true);
    
    /**
     * @brief Set emergency cleanup callback
     * @param callback Function to call during emergency cleanup
     */
    void set_emergency_cleanup_callback(const std::function<void()>& callback);
    
    /**
     * @brief Enable detailed mobile profiling
     * @param enable True to enable detailed profiling
     * @param output_path Path to save profiling data
     */
    void enable_mobile_profiling(bool enable = true, const std::string& output_path = "");
    
    /**
     * @brief Adapt to current mobile conditions automatically
     * This method analyzes current system state and automatically adjusts
     * optimization strategy, memory usage, and perforance parameters
     */
    void adapt_to_mobile_conditions();
    
    /**
     * @brief Verify mobile optimization effectiveness
     * @return Score from 0.0 (poor) to 1.0 (excellent) indicating optimization effectiveness
     */
    double verify_mobile_optimization_effectiveness() const;

private:
    // Internal implementsation methods
    size_t calculate_partition_size(const TensorPtr& tensor);
    void load_parameter_sync(size_t param_id);
    void unload_parameter_sync(size_t param_id);
    void load_parameter_async(size_t param_id);
    void unload_parameter_async(size_t param_id);
    void update_access_pattern(size_t param_id);
    void trigger_eviction_if_needed();
    void allocate_memory_pools();
    void cleanup_memory_pools();
    
    // Storage I/O operations
    void save_parameter_to_storage(size_t param_id);
    void load_parameter_from_storage(size_t param_id);
    
    // Memory management utilities
    void* allocate_from_pool(MemoryTier tier, size_t size);
    void deallocate_from_pool(MemoryTier tier, void* ptr, size_t size);
    size_t get_available_memory(MemoryTier tier) const;
    double calculate_memory_pressure() const;
    
    // Prefetching logic
    void prefetch_worker_loop();
    void predict_next_parameters(size_t current_param_id, std::vector<size_t>& predictions);
    
    // Statistics update
    void update_memory_stats();
    void log_memory_usage();
    
    // ======= CRITICAL MOBILE-SPECIFIC MISSING PRIVATE METHODS =======
    
    // Mobile system monitoring
    void system_monitor_worker_loop();
    void power_monitor_worker_loop();
    void detect_mobile_hardware_capabilities();
    void update_system_memory_pressure();
    void handle_system_state_change(MobileSystemState new_state);
    
    // Mobile hardware optimization
    void optimize_for_detected_gpu_vendor();
    void configure_big_little_cpu_scheduling();
    void detect_and_configure_cache_hierarchy();
    size_t detect_memory_bandwidth();
    
    // Mobile-specific memory management
    void check_oom_killer_risk();
    void implements_zram_optimization();
    void optimize_for_memory_pressure_level(MemoryPressureLevel level);
    void trigger_aggressive_cleanup();
    
    // Mobile perforance optimization
    void monitor_ui_thread_impact();
    void adjust_operation_scheduling_for_ui();
    void implements_fps_aware_scheduling();
    void optimize_cpu_cache_usage();
    
    // Mobile power management
    void monitor_thermal_state();
    void implements_dvfs_optimizations();
    void adjust_for_battery_level();
    size_t estimate_power_consumption();
    
    // Mobile cache optimization
    void implements_cache_line_optimization();
    void analyze_access_patterns_for_cache();
    void prefetch_cache_lines();
    void align_parameters_to_cache_boundaries();
    
    // Mobile network awareness
    void monitor_network_state();
    void optimize_for_cellular_connection();
    void implements_data_usage_limits();
    
    // Mobile-specific access pattern analysis
    void record_mobile_access_pattern(size_t param_id, MemoryTier tier, size_t size);
    void analyze_mobile_access_patterns();
    void predict_mobile_access_patterns();
    void optimize_based_on_mobile_patterns();
    
    // Mobile error handling and recovery
    void setup_crash_safe_operation();
    void handle_system_kill_recovery();
    void implements_graceful_degradation();
    
    // Mobile profiling and debugging
    void log_mobile_perforance_event(const std::string& event, double duration_ms);
    void export_mobile_profiling_data(const std::string& output_path);
    void validate_mobile_optimization_configuration();
    
    // Mobile adaptive optimization
    void adapt_quantization_for_mobile();
    void adapt_prefetch_for_mobile();
    void adapt_caching_for_mobile();
    void adapt_scheduling_for_mobile();
    
    // Mobile hardware-specific optimizations
    void optimize_for_adreno_gpu();
    void optimize_for_mali_gpu();
    void optimize_for_apple_gpu();
    void implements_arm_neon_optimizations();
    
    // Mobile system integration
    void register_low_memory_callback();
    void register_thermal_callback();
    void register_lifecycle_callback();
    void handle_android_low_memory_killer();
    
    // Mobile memory pattern optimization
    void implements_sequential_access_optimization();
    void implements_spatial_locality_optimization();
    void implements_temporal_locality_optimization();
    
    // Mobile-specific utility methods
    bool is_mobile_gpu_memory_constrained() const;
    bool should_use_aggressive_quantization() const;
    bool should_prioritize_battery_over_perforance() const;
    size_t get_optimal_partition_size_for_mobile(const TensorPtr& tensor) const;
    MemoryTier get_optimal_tier_for_mobile_conditions(size_t param_id) const;
};

/**
 * @brief RAII helper for automatic parameter lifecycle management
 */
class ScopedParameterAccess {
private:
    MobileParameterManager* manager_;
    size_t param_id_;
    TensorPtr tensor_;
    bool is_dirty_;

public:
    ScopedParameterAccess(MobileParameterManager* manager, size_t param_id, const std::string& hint = "");
    ScopedParameterAccess(MobileParameterManager* manager, const std::string& name, const std::string& hint = "");
    ~ScopedParameterAccess();
    
    TensorPtr get() const { return tensor_; }
    void mark_dirty() { is_dirty_ = true; }
    
    // Prevent copying
    ScopedParameterAccess(const ScopedParameterAccess&) = delete;
    ScopedParameterAccess& operator=(const ScopedParameterAccess&) = delete;
    
    // Allow moving
    ScopedParameterAccess(ScopedParameterAccess&& other) noexcept;
    ScopedParameterAccess& operator=(ScopedParameterAccess&& other) noexcept;
};

/**
 * @brief Factory function to create mobile parameter manager with sensible defaults
 */
std::unique_ptr<MobileParameterManager> create_mobile_param_manager(
    size_t available_memory_mb = 1024,
    bool enable_storage_offload = true,
    const std::string& cache_dir = "./param_cache"
);

} // namespace memory
} // namespace ops
