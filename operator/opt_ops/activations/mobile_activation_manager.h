/**
 * @file mobile_activation_manager.h
 * @brief Mobile-optimized activation management system inspired by DeepSpeed
 * 
 * This system implementss comprehensive activation memory optimization for mobile training:
 * 1. Gradient Checkpointing with mobile-aware scheduling
 * 2. Activation Compression using quantization and sparsification  
 * 3. Hierarchical Storage (GPU -> CPU -> Compressed -> Storage)
 * 4. Mobile-specific optimizations (battery, thermal, memory pressure aware)
 * 
 * Key Innovation: Unlike DeepSpeed which targets data center GPUs, this system
 * is specifically designed for mobile constraints with advanced power and thermal
 * management integration.
 */

#pragma once

#include "../core/tensor.h"
#include "../core/device.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>
#include <deque>

namespace ops {
namespace memory {

// Import tensor types
using ops::TensorPtr;
using ops::Tensor;
using ops::Device;
using ops::DType;

// Forward declarations for activation management components
class ActivationCheckpointer;
class ActivationCompressor; 
class ActivationStorage;
class ActivationRecomputer;
class MobileActivationOptimizer;

// [Translated comment removed - see documentation]
class ZeROffloadActivationManager;
class ConstantBufferOptimizer;
class PinnedMemoryManager;
class ActivationBandwidthOptimizer;
class ActivationFusionEngine;

// [Translated comment removed - see documentation]
class MobileSystemIntegrationManager;
class MobileEfficientAttention;
class MobileActivationPartitioner;
class UMAMemoryOptimizer;
class LPDDRMemoryOptimizer;
class ANRProtectionManager;
class MobileDMAOptimizer;
class CacheLineOptimizer;
class DVFSAwareScheduler;
class BigLittleCPUScheduler;
class MobileGPUVendorOptimizer;

/**
 * @brief Activation storage tiers for mobile optimization
 */
enum class ActivationTier {
    GPU_FAST = 0,        // GPU/NPU fast memory (highest priority)
    CPU_MEMORY = 1,      // System RAM (medium priority)  
    COMPRESSED = 2,      // Compressed in-memory (low priority)
    PERSISTENT = 3       // File storage (lowest priority)
};

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
 * @brief Mobile system state for activation management
 */
enum class MobileActivationState {
    NORMAL = 0,           // Normal operation
    MEMORY_PRESSURE = 1,  // High memory usage
    BATTERY_LOW = 2,      // Low battery state
    THERMAL_WARNING = 3,  // Device overheating
    BACKGROUND = 4,       // App in background
    CRITICAL = 5          // Critical resource state
};

/**
 * @brief Activation checkpoint strategy
 */
enum class CheckpointStrategy {
    UNIFORM = 0,          // Unifor checkpointing interval
    ADAPTIVE = 1,         // Adaptive based on memory pressure
    LAYER_WISE = 2,       // Layer-wise selective checkpointing
    MOBILE_SMART = 3      // Mobile-optimized smart checkpointing
};

/**
 * @brief Activation metadata for tracking and optimization
 */
struct ActivationMetadata {
    size_t activation_id;
    std::string layer_name;
    std::vector<int64_t> shape;
    DType dtype;
    ActivationTier current_tier;
    ActivationCompressionMode compression_mode;
    
    // Timing inforation
    std::chrono::steady_clock::time_point creation_time;
    std::chrono::steady_clock::time_point last_access_time;
    std::chrono::steady_clock::time_point expected_release_time;
    
    // Memory inforation
    size_t original_size_bytes;
    size_t compressed_size_bytes;
    size_t memory_footprint;
    float compression_ratio;
    
    // Mobile-specific metadata
    bool is_checkpoint;
    bool is_recomputable;
    bool is_critical_for_ui;
    int access_frequency;
    int recomputation_cost;  // Relative cost (1-10)
    
    // System integration metadata
    bool thermal_sensitive;     // Avoid accessing when hot
    bool battery_sensitive;     // Defer access when low battery
    int priority_level;         // 0=lowest, 10=highest
    
        // [Translated]
    bool thermal_throttle_paused;     // [Translated]
    
    ActivationMetadata(size_t id, const std::string& name, const std::vector<int64_t>& s, DType dt)
        : activation_id(id), layer_name(name), shape(s), dtype(dt),
          current_tier(ActivationTier::GPU_FAST), compression_mode(ActivationCompressionMode::NONE),
          creation_time(std::chrono::steady_clock::now()), last_access_time(creation_time),
          original_size_bytes(0), compressed_size_bytes(0), memory_footprint(0), compression_ratio(1.0f),
          is_checkpoint(false), is_recomputable(true), is_critical_for_ui(false),
          access_frequency(1), recomputation_cost(5), thermal_sensitive(false),
          battery_sensitive(false), priority_level(5), thermal_throttle_paused(false) {}
};

/**
 * @brief Configuration for mobile activation management
 */
struct MobileActivationConfig {
    // Memory management
    size_t max_gpu_activation_memory_mb = 512;    // Max GPU memory for activations (512MB)
    size_t max_cpu_activation_memory_mb = 1024;   // Max CPU memory for activations (1GB)
    size_t max_compressed_memory_mb = 2048;       // Max compressed memory (2GB)
    size_t activation_cache_size = 100;           // Max number of cached activations
    
    // Checkpointing configuration
    CheckpointStrategy checkpoint_strategy = CheckpointStrategy::MOBILE_SMART;
    int default_checkpoint_interval = 4;          // Checkpoint every 4 layers by default
    float memory_pressure_checkpoint_threshold = 0.7f; // Checkpoint when 70% memory used
    bool enable_adaptive_checkpointing = true;    // Enable dynamic checkpoint adjustment
    
    // Compression configuration
    bool enable_activation_compression = true;    // Enable activation compression
    ActivationCompressionMode default_compression = ActivationCompressionMode::QUANTIZE_INT8;
    float compression_memory_threshold = 0.8f;   // Compress when 80% memory used
    bool enable_lossy_compression = false;       // Enable lossy compression for non-critical activations
    
    // Mobile-specific configuration
    bool enable_mobile_optimizations = true;     // Enable mobile-specific optimizations
    bool enable_thermal_management = true;       // Enable thermal-aware management
    bool enable_battery_management = true;       // Enable battery-aware management
    bool enable_ui_responsiveness = true;        // Maintain UI responsiveness
    
    // Recomputation configuration
    bool enable_activation_recomputation = true; // Enable activation recomputation
    int max_recomputation_threads = 2;          // Max background recomputation threads
    float recomputation_memory_threshold = 0.9f; // Recompute when 90% memory used
    int max_recomputation_cost = 8;             // Max cost for recomputation (1-10 scale)
    
    // Perforance tuning
    size_t activation_prefetch_count = 3;        // Number of activations to prefetch
    bool enable_async_operations = true;        // Enable async load/store operations
    size_t io_buffer_size_kb = 1024;           // I/O buffer size (1MB)
    
    // Storage configuration
    std::string storage_path = "./activation_cache"; // Path for activation storage
    bool enable_persistent_storage = true;      // Enable persistent storage
    bool compress_storage_files = true;         // Compress storage files
    
    // Mobile system integration
    float memory_pressure_warning_threshold = 0.75f; // Warning at 75% memory usage
    float memory_pressure_critical_threshold = 0.9f;  // Critical at 90% memory usage
    size_t ui_max_blocking_time_ms = 16;       // Max UI blocking time (60 FPS)
    float thermal_throttle_temperature = 75.0f; // Throttle at 75°C
    int battery_critical_level = 20;           // Critical battery level (%)
    
    // System resource utilization thresholds for optimization decisions
    float gpu_utilization_aggressive_threshold = 0.9f;     // [Translated]
    float cpu_utilization_aggressive_threshold = 0.8f;     // [Translated]
    float gpu_utilization_thermal_threshold = 0.8f;        // [Translated]
    float memory_pressure_optimization_threshold = 0.8f;     // [Translated]
    
    // Advanced mobile optimizations
    bool enable_power_aware_compression = true;  // Use more compression on battery
    bool enable_network_aware_storage = true;   // Avoid network storage on cellular
    bool enable_app_lifecycle_management = true; // Respond to app lifecycle changes
    
    // 🚨 CRITICAL MISSING: DeepSpeed ZeRO-Offload Configuration
    bool enable_zero_offload_activations = true;    // Enable ZeRO-Offload for activations
    bool enable_nvme_offload = false;               // Enable NVMe offloading (requires NVMe)
    std::string nvme_offload_path = "/tmp/activation_nvme"; // NVMe offload path
    bool enable_cpu_offload = true;                 // Enable CPU offloading
    size_t offload_threshold_mb = 100;              // Offload threshold (MB)
    
    // Constant Buffer Optimization (CBO)
    bool enable_constant_buffer_optimization = true; // Enable CBO
    size_t constant_buffer_size_mb = 64;            // Constant buffer pool size
    int max_buffer_reuse_count = 8;                 // Max buffer reuse
    
    // Pin Memory Configuration
    bool enable_pinned_memory = true;               // Enable pinned memory
    size_t max_pinned_memory_mb = 256;             // Max pinned memory allocation
    bool enable_memory_pool = true;                // Enable memory pool management
    size_t memory_pool_size_mb = 512;              // Memory pool size
    
    // Activation Fusion Configuration
    bool enable_activation_fusion = true;          // Enable activation fusion
    size_t fusion_buffer_size_mb = 32;            // Fusion buffer size
    int max_fusion_operations = 4;                // Max operations to fuse
    
    // Bandwidth Optimization Configuration
    bool enable_bandwidth_optimization = true;     // Enable bandwidth optimization
    size_t optimal_transfer_size_kb = 64;         // Optimal transfer chunk size
    bool enable_async_memory_copy = true;         // Enable async memory copy
    
    // 🚨 CRITICAL MISSING: Mobile Hardware-Specific Configuration
    
    // UMA (Unified Memory Architecture) Configuration
    bool enable_uma_optimization = false;          // Enable UMA optimization (auto-detected)
    bool detect_uma_automatically = true;          // Auto-detect UMA support
    float uma_memory_efficiency_target = 0.95f;   // UMA memory efficiency target
    
    // LPDDR (Low Power DDR) Configuration
    bool enable_lpddr_optimization = true;         // Enable LPDDR optimization
    bool optimize_for_lpddr_bandwidth = true;     // Optimize for LPDDR bandwidth
    size_t lpddr_burst_size = 64;                 // LPDDR burst size (bytes)
    bool enable_lpddr_power_saving = true;        // Enable LPDDR power saving
    
    // ANR (Application Not Responding) Protection Configuration
    bool enable_anr_protection = true;             // Enable ANR protection
    size_t max_blocking_operation_ms = 8;         // Max operation time before yielding
    size_t anr_detection_threshold_ms = 100;      // ANR detection threshold
    bool enable_operation_yielding = true;        // Enable operation yielding
    
    // Mobile DMA Configuration  
    bool enable_mobile_dma = true;                 // Enable mobile DMA optimization
    size_t dma_transfer_threshold_kb = 16;        // Min size for DMA transfer
    bool enable_dma_coherency = true;             // Enable DMA coherency optimization
    
    // Cache Line Optimization Configuration
    bool enable_cache_line_optimization = true;   // Enable cache line optimization
    size_t l1_cache_line_size = 64;              // L1 cache line size
    size_t l2_cache_line_size = 64;              // L2 cache line size  
    size_t l3_cache_line_size = 64;              // L3 cache line size
    bool enable_cache_prefetching = true;        // Enable cache prefetching
    bool enable_data_cache_optimization = true;  // Enable data cache optimization
    
    // DVFS (Dynamic Voltage Frequency Scaling) Configuration
    bool enable_dvfs_awareness = true;            // Enable DVFS awareness
    bool adapt_to_frequency_scaling = true;      // Adapt to CPU frequency changes
    float perforance_scaling_factor = 1.2f;     // Perforance scaling factor
    bool monitor_cpu_frequency = true;           // Monitor CPU frequency changes
    
    // big.LITTLE CPU Scheduling Configuration
    bool enable_big_little_scheduling = true;    // Enable big.LITTLE scheduling
    bool prefer_little_cores_for_memory = true;  // Use LITTLE cores for memory ops
    bool prefer_big_cores_for_compute = true;    // Use big cores for compute ops
    int little_core_affinity_mask = 0x0F;       // LITTLE core affinity mask
    int big_core_affinity_mask = 0xF0;          // big core affinity mask
    
    // Mobile GPU Vendor Optimization Configuration
    bool enable_gpu_vendor_optimization = true;  // Enable GPU vendor optimization
    bool auto_detect_gpu_vendor = true;         // Auto-detect GPU vendor
    // Adreno GPU optimization
    bool enable_adreno_tiled_rendering = false; // Enable Adreno tiled rendering opt
    bool enable_adreno_bandwidth_opt = false;   // Enable Adreno bandwidth opt
    // Mali GPU optimization  
    bool enable_mali_bandwidth_opt = false;     // Enable Mali bandwidth opt
    bool enable_mali_cache_opt = false;         // Enable Mali cache opt
    // Apple GPU optimization
    bool enable_apple_unified_memory = false;   // Enable Apple unified memory opt
    bool enable_apple_neural_engine = false;    // Enable Apple Neural Engine
    
    // Memory Pressure API Integration Configuration
    bool enable_memory_pressure_api = true;     // Enable memory pressure API
    bool enable_android_ontrim_memory = true;   // Enable Android OnTrimMemory
    bool enable_ios_memory_warning = true;      // Enable iOS memory warning
    bool enable_system_memory_callbacks = true; // Enable system memory callbacks
    
    // Background App Optimization Configuration (iOS specific)
    bool enable_background_app_optimization = true; // Enable background app optimization
    size_t background_memory_limit_mb = 200;    // Memory limit when backgrounded
    bool enable_background_task_completion = true; // Enable background task completion
    
    // 🚀 Advanced Component Configuration
    bool enable_efficient_attention = false;    // Enable mobile efficient attention
    bool enable_activation_partitioning = false; // Enable activation partitioning
    bool enable_predictive_prefetch = false;    // Enable predictive prefetching
    size_t access_pattern_history_size = 1000;  // Access pattern history size
    
        // [Translated]
    bool enable_activation_caching = false;     // Enable activation caching
    bool enable_memory_alignment_optimization = false; // Memory alignment optimization
    bool enable_batch_processing = false;       // Enable batch processing
    size_t default_transfer_chunk_size = 64 * 1024; // Default transfer chunk size (64KB)
    bool enable_zero_copy_optimization = false; // Enable zero-copy optimization
    bool enable_burst_memory_access = false;    // Enable burst memory access
    size_t memory_access_alignment = 64;        // Memory access alignment
    // bool enable_operation_yielding = false;  // (Already declared above)
    size_t large_transfer_threshold_bytes = 16 * 1024; // Large transfer threshold
    bool enable_hardware_acceleration = false;  // Enable hardware acceleration
    bool enable_data_layout_optimization = false; // Data layout optimization
    size_t cache_line_alignment = 64;          // Cache line alignment
    bool enable_frequency_adaptive_scheduling = false; // Frequency adaptive scheduling
    bool enable_core_affinity_optimization = false; // Core affinity optimization
    bool enable_gpu_specific_optimization = false; // GPU specific optimization
    
    // Perforance monitoring and analytics
    bool enable_activation_profiling = false;   // Enable detailed profiling
    bool log_activation_events = false;         // Log activation management events
    std::string profiling_output_path = "./activation_profile.json";
};

/**
 * @brief Statistics for activation management monitoring
 */
struct ActivationStats {
    // Memory usage statistics
    size_t total_activations;
    size_t gpu_activations;
    size_t cpu_activations; 
    size_t compressed_activations;
    size_t storage_activations;
    
    size_t total_memory_used;
    size_t gpu_memory_used;
    size_t cpu_memory_used;
    size_t compressed_memory_used;
    size_t storage_memory_used;
    
    // Perforance statistics  
    size_t total_checkpoints;
    size_t total_recomputations;
    size_t cache_hits;
    size_t cache_misses;
    size_t prefetch_hits = 0;
    size_t prefetch_misses = 0;
    
    // Compression statistics
    size_t total_compressions;
    size_t total_decompressions;
    float average_compression_ratio;
    size_t compression_memory_saved;
    
    // Mobile-specific statistics
    size_t thermal_throttle_events;
    size_t battery_optimizations;
    size_t ui_responsiveness_protections;
    size_t memory_pressure_events;
    
    // Timing statistics
    double average_access_time_ms;
    double average_compression_time_ms;
    double average_recomputation_time_ms;
    
        // [Translated]
    size_t prefetch_operations = 0;
    size_t removed_activations = 0;
    size_t total_original_bytes = 0;
    size_t total_compressed_bytes = 0;
    float memory_efficiency = 0.0f;
    float cache_hit_ratio = 0.0f;
    float prefetch_hit_ratio = 0.0f;
};

/**
 * @brief Main mobile activation management class
 * 
 * This class providess comprehensive activation memory management specifically
 * optimized for mobile training scenarios. It integrates multiple optimization
 * strategies from DeepSpeed while adding mobile-specific enhancements.
 */
class MobileActivationManager {
private:
    MobileActivationConfig config_;
    std::unordered_map<size_t, std::unique_ptr<ActivationMetadata>> activation_metadata_;
    std::unordered_map<size_t, TensorPtr> active_activations_;
    
    // Core components
        // [Translated]
    std::unique_ptr<ActivationCheckpointer> checkpointer_;
    std::unique_ptr<ActivationCompressor> compressor_;
    std::unique_ptr<ActivationStorage> storage_;
    
    // [Translated comment removed - see documentation]
    std::unique_ptr<ZeROffloadActivationManager> zero_offload_manager_;
    std::unique_ptr<ConstantBufferOptimizer> constant_buffer_optimizer_;
    std::unique_ptr<PinnedMemoryManager> pinned_memory_manager_;
    std::unique_ptr<ActivationBandwidthOptimizer> bandwidth_optimizer_;
    std::unique_ptr<ActivationFusionEngine> fusion_engine_;
    
    // [Translated comment removed - see documentation]
    std::unique_ptr<MobileSystemIntegrationManager> system_integration_manager_;
    std::unique_ptr<MobileEfficientAttention> efficient_attention_;
    std::unique_ptr<MobileActivationPartitioner> partition_manager_;
    std::unique_ptr<UMAMemoryOptimizer> uma_optimizer_;
    std::unique_ptr<LPDDRMemoryOptimizer> lpddr_optimizer_;
    std::unique_ptr<ANRProtectionManager> anr_protection_;
    std::unique_ptr<MobileDMAOptimizer> dma_optimizer_;
    std::unique_ptr<CacheLineOptimizer> cache_optimizer_;
    std::unique_ptr<DVFSAwareScheduler> dvfs_scheduler_;
    std::unique_ptr<BigLittleCPUScheduler> cpu_scheduler_;
    std::unique_ptr<MobileGPUVendorOptimizer> gpu_vendor_optimizer_;
    
    // optimizationstaterecord
    bool deepspeed_optimizations_enabled_ = false;
    bool mobile_optimizations_enabled_ = false;
    
    // Memory management
    std::atomic<size_t> gpu_memory_used_;
    std::atomic<size_t> cpu_memory_used_;
    std::atomic<size_t> compressed_memory_used_;
    std::atomic<size_t> next_activation_id_;
    
    // Mobile system monitoring
    std::atomic<MobileActivationState> current_mobile_state_;
    std::atomic<float> current_memory_pressure_;
    std::atomic<float> current_temperature_;
    std::atomic<int> current_battery_level_;
    std::atomic<bool> is_app_foreground_;
    
    // Threading and synchronization
    mutable std::mutex manager_mutex_;
    mutable std::mutex stats_mutex_;
    std::thread background_worker_;
    std::atomic<bool> worker_running_;
    
    // Statistics
    ActivationStats stats_;
    
    // Cache and optimization structures
    std::deque<size_t> recent_access_queue_;
    std::unordered_map<std::string, size_t> layer_checkpoint_intervals_;
    std::unordered_map<size_t, std::function<TensorPtr()>> recomputation_functions_;
    
    // [Translated comment removed - see documentation]
    std::mutex metadata_mutex_;
    
public:
    explicit MobileActivationManager(const MobileActivationConfig& config);
    ~MobileActivationManager();
    
    /**
     * @brief Register a new activation tensor
     * @param layer_name Name of the layer producing this activation
     * @param activation The activation tensor to manage
     * @param is_checkpoint Whether this activation should be a checkpoint
     * @param recomputation_fn Function to recompute this activation if needed
     * @return Activation ID for future reference
     */
    size_t register_activation(
        const std::string& layer_name,
        const TensorPtr& activation,
        bool is_checkpoint = false,
        std::function<TensorPtr()> recomputation_fn = nullptr
    );
    
    /**
     * @brief Get an activation tensor (may trigger recomputation or decompression)
     * @param activation_id Activation ID
     * @param hint Access pattern hint for optimization
     * @return The activation tensor
     */
    TensorPtr get_activation(size_t activation_id, const std::string& hint = "");
    
    /**
     * @brief Release an activation after use
     * @param activation_id Activation ID
     * @param mark_dirty Whether the activation was modified
     */
    void release_activation(size_t activation_id, bool mark_dirty = false);
    
    /**
     * @brief Create a checkpoint at the current point
     * @param layer_name Layer name for the checkpoint
     * @param activation Activation to checkpoint
     * @return Checkpoint ID
     */
    size_t create_checkpoint(const std::string& layer_name, const TensorPtr& activation);
    
    /**
     * @brief Clear all activations before a given checkpoint
     * @param checkpoint_id Checkpoint ID to clear before
     */
    void clear_before_checkpoint(size_t checkpoint_id);
    
    /**
     * @brief Optimize activation memory based on current system state
     */
    void optimize_activation_memory();
    
    /**
     * @brief Force garbage collection of unused activations
     */
    void garbage_collect_activations();
    
    /**
     * @brief Update mobile system state
     * @param state New mobile system state
     */
    void update_mobile_state(MobileActivationState state);
    
    /**
     * @brief Update system metrics
     * @param memory_pressure Current memory pressure (0.0-1.0)
     * @param temperature Current device temperature in Celsius
     * @param battery_level Current battery level (0-100)
     * @param is_foreground Whether app is in foreground
     */
    void update_system_metrics(float memory_pressure, float temperature, 
                             int battery_level, bool is_foreground);
    
    /**
     * @brief Set compression mode for activations
     * @param compression_mode New compression mode
     */
    void set_compression_mode(ActivationCompressionMode compression_mode);
    
    /**
     * @brief Enable/disable specific mobile optimizations
     * @param thermal_management Enable thermal management
     * @param battery_management Enable battery management  
     * @param ui_responsiveness Maintain UI responsiveness
     */
    void configure_mobile_optimizations(bool thermal_management = true,
                                      bool battery_management = true, 
                                      bool ui_responsiveness = true);
    
    /**
     * @brief Get current activation statistics
     * @return Current statistics
     */
    ActivationStats get_activation_stats() const;
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const MobileActivationConfig& get_config() const { return config_; }
    
    /**
     * @brief Save activation checkpoint to persistent storage
     * @param checkpoint_path Path to save checkpoint
     */
    void save_activation_checkpoint(const std::string& checkpoint_path);
    
    /**
     * @brief Load activation checkpoint from persistent storage
     * @param checkpoint_path Path to load checkpoint from
     */
    void load_activation_checkpoint(const std::string& checkpoint_path);
    
    /**
     * @brief Export detailed profiling report
     * @param report_path Path to save profiling report
     */
    void export_profiling_report(const std::string& report_path) const;
    
    // CRITICAL MISSING: DeepSpeed ZeRO-Offload Methods
    
    /**
     * @brief Enable ZeRO-Offload for activations
     * @param enable_cpu Whether to enable CPU offloading
     * @param enable_nvme Whether to enable NVMe offloading
     * @param nvme_path Path to NVMe device (if available)
     */
    void enable_zero_offload(bool enable_cpu = true, bool enable_nvme = false, 
                           const std::string& nvme_path = "/tmp/activation_nvme");
    
    /**
     * @brief Configure constant buffer optimization
     * @param buffer_size_mb Constant buffer size in MB
     * @param max_reuse_count Maximum buffer reuse count
     */
    void configure_constant_buffer_optimization(size_t buffer_size_mb = 64, int max_reuse_count = 8);
    
    /**
     * @brief Enable pinned memory management
     * @param max_pinned_mb Maximum pinned memory in MB
     * @param enable_memory_pool Whether to enable memory pool
     */
    void enable_pinned_memory_management(size_t max_pinned_mb = 256, bool enable_memory_pool = true);
    
    /**
     * @brief Enable activation fusion optimization
     * @param fusion_buffer_mb Fusion buffer size in MB
     * @param max_operations Maximum operations to fuse
     */
    void enable_activation_fusion(size_t fusion_buffer_mb = 32, int max_operations = 4);
    
    /**
     * @brief Configure bandwidth optimization
     * @param optimal_chunk_kb Optimal transfer chunk size in KB
     * @param enable_async Whether to enable async memory copy
     */
    void configure_bandwidth_optimization(size_t optimal_chunk_kb = 64, bool enable_async = true);
    
    // CRITICAL MISSING: Mobile Hardware-Specific Methods
    
    /**
     * @brief Enable UMA (Unified Memory Architecture) optimization
     * @param auto_detect Whether to auto-detect UMA support
     * @param efficiency_target Memory efficiency target (0.0-1.0)
     */
    void enable_uma_optimization(bool auto_detect = true, float efficiency_target = 0.95f);
    
    /**
     * @brief Enable LPDDR (Low Power DDR) optimization
     * @param optimize_bandwidth Whether to optimize for LPDDR bandwidth
     * @param burst_size LPDDR burst size in bytes
     */
    void enable_lpddr_optimization(bool optimize_bandwidth = true, size_t burst_size = 64);
    
    /**
     * @brief Enable ANR (Application Not Responding) protection
     * @param max_blocking_ms Maximum operation time before yielding
     * @param anr_threshold_ms ANR detection threshold
     */
    void enable_anr_protection(size_t max_blocking_ms = 8, size_t anr_threshold_ms = 100);
    
    /**
     * @brief Enable mobile DMA optimization
     * @param transfer_threshold_kb Minimum size for DMA transfer
     * @param enable_coherency Whether to enable DMA coherency
     */
    void enable_mobile_dma(size_t transfer_threshold_kb = 16, bool enable_coherency = true);
    
    /**
     * @brief Configure cache line optimization
     * @param l1_size L1 cache line size
     * @param l2_size L2 cache line size
     * @param l3_size L3 cache line size
     * @param enable_prefetch Whether to enable cache prefetching
     */
    void configure_cache_line_optimization(size_t l1_size = 64, size_t l2_size = 64, 
                                         size_t l3_size = 64, bool enable_prefetch = true);
    
    /**
     * @brief Enable DVFS (Dynamic Voltage Frequency Scaling) awareness
     * @param adapt_to_scaling Whether to adapt to frequency changes
     * @param scaling_factor Perforance scaling factor
     */
    void enable_dvfs_awareness(bool adapt_to_scaling = true, float scaling_factor = 1.2f);
    
    /**
     * @brief Configure big.LITTLE CPU scheduling
     * @param little_for_memory Use LITTLE cores for memory operations
     * @param big_for_compute Use big cores for compute operations
     * @param little_mask LITTLE core affinity mask
     * @param big_mask big core affinity mask
     */
    void configure_big_little_scheduling(bool little_for_memory = true, bool big_for_compute = true,
                                       int little_mask = 0x0F, int big_mask = 0xF0);
    
    /**
     * @brief Enable mobile GPU vendor-specific optimizations
     * @param auto_detect Whether to auto-detect GPU vendor
     * @param enable_adreno Enable Adreno optimizations
     * @param enable_mali Enable Mali optimizations
     * @param enable_apple Enable Apple GPU optimizations
     */
    void enable_gpu_vendor_optimizations(bool auto_detect = true, bool enable_adreno = false,
                                       bool enable_mali = false, bool enable_apple = false);
    
    /**
     * @brief Enable memory pressure API integration
     * @param enable_android Enable Android OnTrimMemory
     * @param enable_ios Enable iOS memory warning
     * @param enable_callbacks Enable system memory callbacks
     */
    void enable_memory_pressure_api(bool enable_android = true, bool enable_ios = true,
                                  bool enable_callbacks = true);
    
    /**
     * @brief Configure background app optimization (iOS specific)
     * @param memory_limit_mb Memory limit when backgrounded
     * @param enable_task_completion Enable background task completion
     */
    void configure_background_app_optimization(size_t memory_limit_mb = 200,
                                             bool enable_task_completion = true);
    
    /**
     * @brief Perfor comprehensive mobile system detection and optimization
     * This method detects the mobile platfor, hardware capabilities, and
     * automatically configures optimal settings
     */
    void auto_configure_for_mobile_platfor();
    
    /**
     * @brief Perfor comprehensive missing component check and display results
     * This method checks all DeepSpeed and mobile-specific components and
     * providess a detailed report of what's implementsed vs missing
     */
    void perfor_comprehensive_missing_component_check();
    
    /**
     * @brief Get comprehensive mobile optimization status
     */
    struct MobileOptimizationStatus {
        // DeepSpeed integration status
        bool zero_offload_enabled;
        bool constant_buffer_enabled;
        bool pinned_memory_enabled;
        bool activation_fusion_enabled;
        bool bandwidth_optimization_enabled;
        
        // Mobile hardware optimization status
        bool uma_optimization_enabled;
        bool lpddr_optimization_enabled;
        bool anr_protection_enabled;
        bool mobile_dma_enabled;
        bool cache_line_optimization_enabled;
        bool dvfs_awareness_enabled;
        bool big_little_scheduling_enabled;
        bool gpu_vendor_optimization_enabled;
        
        // System integration status
        bool memory_pressure_api_enabled;
        bool background_app_optimization_enabled;
        
        // Hardware detection results
        std::string detected_platfor;        // "Android", "iOS", "macOS", etc.
        std::string detected_gpu_vendor;      // "Adreno", "Mali", "Apple", etc.
        std::string detected_cpu_architecture; // "ARM64", "x86_64", etc.
        bool has_uma_support;                 // UMA support detected
        bool has_lpddr;                       // LPDDR memory detected
        
        // Perforance metrics
        float memory_efficiency_score;        // 0.0-1.0
        float power_efficiency_score;         // 0.0-1.0
        float ui_responsiveness_score;        // 0.0-1.0
        
        // System integration metrics
        size_t anr_events_prevented;
        size_t memory_pressure_responses;
        size_t background_optimizations;
    };
    MobileOptimizationStatus get_mobile_optimization_status() const;

        // [Translated]
    void apply_aggressive_memory_optimization();
    void apply_battery_aware_optimization();
    void apply_thermal_aware_optimization(); 
    void apply_balanced_optimization();
    void update_optimization_statistics();
    
        // [Translated]
    void optimize_for_low_memory_pressure();
    void optimize_for_thermal_throttling(); 
    void optimize_for_battery_conservation();
    void perfor_emergency_memory_cleanup();

private:
    // Component management methods
    void initialize_components();
    void cleanup_components();
    void start_background_worker();
    void stop_background_worker();
    
    // Internal memory management methods
    void evict_activations_if_needed();
    void compress_activations_if_needed(); 
    void offload_activations_to_storage(const std::vector<size_t>& activation_ids);
    void prefetch_activations(const std::vector<size_t>& activation_ids);
    
    // Mobile optimization methods
    void apply_thermal_optimizations();
    void apply_battery_optimizations();
    void apply_ui_responsiveness_optimizations();
    void apply_memory_pressure_optimizations();
    
    // Background worker methods
    void background_worker_loop();
    void optimize_activation_layout();
    void cleanup_expired_activations();
    void update_access_patterns();
    
    // Utility methods
    ActivationTier select_optimal_tier(const ActivationMetadata& metadata);
    ActivationTier select_optimal_tier_for_activation(const ActivationMetadata& metadata);
    float calculate_recomputation_cost(size_t activation_id);
    bool should_compress_activation(const ActivationMetadata& metadata);
    std::vector<size_t> select_eviction_candidates(size_t count);
    size_t calculate_activation_memory_footprint(const TensorPtr& activation);
    
    // Memory and statistics tracking methods
    void update_tier_memory_usage(ActivationTier tier, size_t memory_size, bool add);
    void update_tier_statistics(ActivationTier tier, size_t count, size_t memory_size, bool add);
    void record_access_pattern(size_t activation_id);
    void update_cache_statistics(bool hit);
    TensorPtr recompute_activation(size_t activation_id);
    
    // [Translated comment removed - see documentation]
    void prefetch_related_activations(size_t activation_id, const std::string& hint);
    size_t generate_layer_id(const std::string& layer_name);
    void remove_activation(size_t activation_id);
    void remove_activation_internal(size_t activation_id);
    float calculate_migration_priority(const ActivationMetadata& metadata);
    void analyze_access_patterns();
    
        // [Translated]
    float calculate_memory_efficiency();
    
    // Statistics update methods
    void update_stats();
    void log_activation_event(const std::string& event, size_t activation_id, double duration_ms = 0.0);
};

/**
 * @brief RAII helper for automatic activation lifecycle management
 */
class ScopedActivationAccess {
private:
    MobileActivationManager* manager_;
    size_t activation_id_;
    TensorPtr activation_;
    bool is_dirty_;

public:
    ScopedActivationAccess(MobileActivationManager* manager, size_t activation_id, const std::string& hint = "");
    ~ScopedActivationAccess();
    
    TensorPtr get() const { return activation_; }
    void mark_dirty() { is_dirty_ = true; }
    
    // Prevent copying
    ScopedActivationAccess(const ScopedActivationAccess&) = delete;
    ScopedActivationAccess& operator=(const ScopedActivationAccess&) = delete;
    
    // Allow moving
    ScopedActivationAccess(ScopedActivationAccess&& other) noexcept;
    ScopedActivationAccess& operator=(ScopedActivationAccess&& other) noexcept;
};

/**
 * @brief Factory function to create mobile activation manager
 */
std::unique_ptr<MobileActivationManager> create_mobile_activation_manager(
    size_t gpu_memory_mb = 512,
    size_t cpu_memory_mb = 1024,
    const std::string& cache_dir = "./activation_cache"
);

} // namespace memory
} // namespace ops
