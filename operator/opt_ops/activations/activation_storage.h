/**
 * @file activation_storage.h
 * @brief Mobile-optimized hierarchical activation storage system
 * 
 * This component implementss a sophisticated multi-tier storage system for activations,
 * specifically designed for mobile constraints. It manages activation storage across
 * GPU memory, CPU memory, compressed memory, and persistent storage with intelligent
 * migration policies based on access patterns and system state.
 * 
 * Storage Hierarchy:
 * L0: GPU/NPU Fast Memory (fastest, most limited)
 * L1: CPU System Memory (fast, limited)
 * L2: Compressed Memory (medium speed, efficient)
 * L3: Persistent Storage (slowest, unlimited)
 */

#pragma once

#include "../core/tensor.h"
#include "mobile_activation_manager.h"
#include "activation_compressor.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <fstream>

namespace ops {
namespace memory {

using ops::TensorPtr;
using ops::Tensor;

/**
 * @brief Storage location metadata
 */
struct StorageLocation {
    ActivationTier tier;
    std::string file_path;        // For persistent storage
    size_t memory_offset;         // For memory tiers
    size_t size_bytes;
    bool is_compressed;
    ActivationCompressionMode compression_mode;
    
    // Access pattern inforation
    std::chrono::steady_clock::time_point last_access_time;
    std::chrono::steady_clock::time_point creation_time;
    int access_count;
    float access_frequency;
    
    // Mobile optimization metadata
    bool is_pinned_memory;        // For CPU-GPU fast transfers
    bool is_memory_mapped;        // For large file storage
    bool is_encrypted;            // For sensitive data
    int migration_priority;       // Priority for tier migration
    
    StorageLocation(ActivationTier t) 
        : tier(t), memory_offset(0), size_bytes(0), is_compressed(false),
          compression_mode(ActivationCompressionMode::NONE),
          last_access_time(std::chrono::steady_clock::now()),
          creation_time(last_access_time), access_count(0), access_frequency(0.0f),
          is_pinned_memory(false), is_memory_mapped(false), is_encrypted(false),
          migration_priority(0) {}
};

/**
 * @brief Storage tier configuration
 */
struct StorageTierConfig {
    size_t max_capacity_bytes;    // Maximum capacity for this tier
    size_t current_usage_bytes;   // Current usage
    float eviction_threshold;     // Start eviction at this threshold
    float migration_threshold;    // Start migration at this threshold
    
    // Perforance characteristics
    double read_bandwidth_mbps;   // Read bandwidth in MB/s
    double write_bandwidth_mbps;  // Write bandwidth in MB/s
    double access_latency_ms;     // Average access latency
    
    // Mobile characteristics
    double power_consumption_mw;  // Power consumption per MB/s
    bool is_battery_dependent;    // Affected by battery state
    bool is_thermal_sensitive;    // Affected by temperature
    
    StorageTierConfig(size_t capacity)
        : max_capacity_bytes(capacity), current_usage_bytes(0),
          eviction_threshold(0.8f), migration_threshold(0.9f),
          read_bandwidth_mbps(1000.0), write_bandwidth_mbps(1000.0),
          access_latency_ms(1.0), power_consumption_mw(100.0),
          is_battery_dependent(false), is_thermal_sensitive(false) {}
};

/**
 * @brief Configuration for activation storage system
 */
struct StorageConfig {
    // Tier capacities (in MB)
    size_t gpu_memory_capacity_mb = 512;      // 512MB GPU memory
    size_t cpu_memory_capacity_mb = 1024;     // 1GB CPU memory
    size_t compressed_memory_capacity_mb = 2048; // 2GB compressed memory
    size_t persistent_storage_capacity_mb = 10240; // 10GB persistent storage
    
    // Migration policies
    bool enable_automatic_migration = true;   // Enable automatic tier migration
    bool enable_predictive_prefetch = true;   // Enable predictive prefetching
    bool enable_compression_migration = true; // Use compression during migration
    float migration_hysteresis = 0.1f;       // Hysteresis for migration decisions
    
    // Access pattern optimization
    bool enable_access_pattern_learning = true; // Learn access patterns
    int access_pattern_history_size = 1000;   // Size of access pattern history
    bool enable_temporal_locality_optimization = true; // Optimize for temporal locality
    bool enable_spatial_locality_optimization = true;  // Optimize for spatial locality
    
    // Mobile-specific configuration
    bool enable_battery_aware_storage = true; // Adjust behavior based on battery
    bool enable_thermal_aware_storage = true; // Adjust behavior based on temperature
    bool enable_network_aware_storage = false; // Consider network state (future)
    
    // I/O optimization
    size_t io_buffer_size_kb = 1024;          // I/O buffer size (1MB)
    int max_concurrent_io_operations = 4;     // Max concurrent I/O ops
    bool enable_async_io = true;              // Enable async I/O operations
    bool enable_io_batching = true;           // Batch small I/O operations
    
    // Persistent storage configuration
    std::string storage_base_path = "./activation_storage";
    bool enable_storage_compression = true;   // Compress files on disk
    bool enable_storage_encryption = false;   // Encrypt sensitive data
    bool enable_memory_mapping = true;        // Use memory mapping for large files
    
    // Perforance tuning
    size_t prefetch_buffer_size_mb = 64;      // Prefetch buffer size
    int migration_worker_threads = 2;         // Migration worker threads
    float cache_locality_weight = 0.7f;       // Weight for cache locality in decisions
    
    // Perforance monitoring and analytics
    bool enable_storage_profiling = false;    // Enable detailed profiling
    bool log_storage_events = false;          // Log storage events
    std::string profiling_output_path = "./storage_profile.json";
};

/**
 * @brief Storage operation metadata
 */
struct StorageOperation {
    enum Type { LOAD, STORE, MIGRATE, PREFETCH };
    
    Type operation_type;
    size_t activation_id;
    ActivationTier source_tier;
    ActivationTier target_tier;
    size_t data_size;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    bool completed;
    std::string error_message;
    
    StorageOperation(Type type, size_t id, ActivationTier source, ActivationTier target, size_t size)
        : operation_type(type), activation_id(id), source_tier(source), target_tier(target),
          data_size(size), start_time(std::chrono::steady_clock::now()), completed(false) {}
};

/**
 * @brief Statistics for storage system monitoring
 */
struct StorageStats {
    // Capacity and usage statistics
    struct TierStats {
        size_t capacity_bytes;
        size_t used_bytes;
        size_t available_bytes;
        float utilization_ratio;
        
        size_t read_operations;
        size_t write_operations;
        double total_read_time_ms;
        double total_write_time_ms;
        double average_read_bandwidth_mbps;
        double average_write_bandwidth_mbps;
    };
    
    TierStats gpu_tier_stats;
    TierStats cpu_tier_stats;
    TierStats compressed_tier_stats;
    TierStats persistent_tier_stats;
    
    // Migration statistics
    size_t total_migrations;
    size_t upward_migrations;      // To faster tiers
    size_t downward_migrations;    // To slower tiers
    double total_migration_time_ms;
    double average_migration_time_ms;
    
    // Cache and prefetch statistics
    size_t cache_hits;
    size_t cache_misses;
    size_t prefetch_hits;
    size_t prefetch_misses;
    float cache_hit_ratio;
    float prefetch_hit_ratio;
    
    // Mobile-specific statistics
    size_t battery_optimizations;
    size_t thermal_optimizations;
    size_t network_optimizations;
    
    // Error statistics
    size_t storage_errors;
    size_t recovery_operations;
    size_t data_corruptions;
};

/**
 * @brief Mobile-optimized activation storage system
 */
class ActivationStorage {
private:
    StorageConfig config_;
    std::unordered_map<size_t, std::unique_ptr<StorageLocation>> activation_locations_;
    std::unordered_map<ActivationTier, StorageTierConfig> tier_configs_;
    
    // Memory pools for different tiers
    void* gpu_memory_pool_;
    void* cpu_memory_pool_;
    void* compressed_memory_pool_;
    std::atomic<size_t> gpu_memory_offset_;
    std::atomic<size_t> cpu_memory_offset_;
    std::atomic<size_t> compressed_memory_offset_;
    
    // Compression integration
    std::unique_ptr<ActivationCompressor> compressor_;
    
    // Threading and async operations
    std::vector<std::thread> migration_workers_;
    std::queue<std::unique_ptr<StorageOperation>> pending_operations_;
    std::mutex operation_queue_mutex_;
    std::condition_variable operation_cv_;
    std::atomic<bool> workers_running_;
    
    // Mobile state monitoring
    std::atomic<float> current_memory_pressure_;
    std::atomic<int> current_battery_level_;
    std::atomic<float> current_temperature_;
    std::atomic<bool> is_on_cellular_;
    
    // Access pattern tracking
    std::deque<std::pair<size_t, std::chrono::steady_clock::time_point>> access_history_;
    std::mutex access_history_mutex_;
    std::unordered_map<size_t, std::vector<size_t>> spatial_locality_map_;
    
    // Statistics
    StorageStats stats_;
    mutable std::mutex stats_mutex_;
    
    // File management
    std::unordered_map<size_t, std::string> activation_file_paths_;
    std::mutex file_system_mutex_;

public:
    explicit ActivationStorage(const StorageConfig& config);
    ~ActivationStorage();
    
    /**
     * @brief Store an activation in the storage system
     * @param activation_id Unique activation identifier
     * @param activation The activation tensor to store
     * @param preferred_tier Preferred storage tier (hint)
     * @return Storage location inforation
     */
    std::unique_ptr<StorageLocation> store_activation(
        size_t activation_id,
        const TensorPtr& activation,
        ActivationTier preferred_tier = ActivationTier::CPU_MEMORY
    );
    
    /**
     * @brief Load an activation from storage
     * @param activation_id Activation identifier
     * @param target_tier Target tier to load into (optional)
     * @return The loaded activation tensor
     */
    TensorPtr load_activation(size_t activation_id, ActivationTier target_tier = ActivationTier::GPU_FAST);
    
    /**
     * @brief Migrate an activation to a different storage tier
     * @param activation_id Activation identifier
     * @param target_tier Target storage tier
     * @param async_migration Whether to perfor migration asynchronously
     * @return True if migration was successful or scheduled
     */
    bool migrate_activation(size_t activation_id, ActivationTier target_tier, bool async_migration = true);
    
    /**
     * @brief Remove an activation from all storage tiers
     * @param activation_id Activation identifier
     * @return True if removal was successful
     */
    bool remove_activation(size_t activation_id);
    
    /**
     * @brief Prefetch activations based on predicted access patterns
     * @param activation_ids List of activation IDs to prefetch
     * @param target_tier Target tier for prefetching
     */
    void prefetch_activations(const std::vector<size_t>& activation_ids, 
                            ActivationTier target_tier = ActivationTier::CPU_MEMORY);
    
    /**
     * @brief Optimize storage based on access patterns and system state
     * This method analyzes current storage utilization and access patterns
     * to optimize data placement across storage tiers
     */
    void optimize_storage_layout();
    
    /**
     * @brief Update mobile system state for adaptive storage management
     * @param memory_pressure Current system memory pressure (0.0-1.0)
     * @param battery_level Current battery level (0-100)
     * @param temperature Current device temperature in Celsius
     * @param is_on_cellular Whether device is on cellular network
     */
    void update_mobile_state(float memory_pressure, int battery_level, 
                           float temperature, bool is_on_cellular);
    
    /**
     * @brief Get inforation about where an activation is stored
     * @param activation_id Activation identifier
     * @return Storage location inforation (null if not found)
     */
    const StorageLocation* get_storage_location(size_t activation_id) const;
    
    /**
     * @brief Get current storage statistics
     * @return Current storage system statistics
     */
    StorageStats get_storage_stats() const;
    
    /**
     * @brief Configure storage system parameters
     * @param config New storage configuration
     */
    void configure_storage(const StorageConfig& config);
    
    /**
     * @brief Perfor storage system garbage collection
     * Cleans up unused storage, defragments memory pools, and optimizes layout
     */
    void garbage_collect_storage();
    
    /**
     * @brief Export detailed profiling report
     * @param report_path Path to save profiling report
     */
    void export_profiling_report(const std::string& report_path) const;

private:
    // Storage tier management
    void initialize_storage_tiers();
    void cleanup_storage_tiers();
    bool allocate_in_tier(ActivationTier tier, size_t size, size_t& offset);
    void deallocate_in_tier(ActivationTier tier, size_t offset, size_t size);
    float get_tier_utilization(ActivationTier tier) const;
    
    // Migration algorithms
    void migration_worker_loop();
    std::vector<size_t> select_migration_candidates(ActivationTier from_tier, ActivationTier to_tier);
    bool should_migrate_activation(size_t activation_id, ActivationTier to_tier);
    float calculate_migration_benefit(size_t activation_id, ActivationTier to_tier);
    
    // Access pattern analysis
    void record_access_pattern(size_t activation_id);
    void analyze_access_patterns();
    std::vector<size_t> predict_next_accesses(size_t current_activation_id, int count = 3);
    void update_spatial_locality_map(size_t activation_id);
    
    // Mobile optimization methods
    void adapt_storage_for_battery_state();
    void adapt_storage_for_thermal_state();
    void adapt_storage_for_memory_pressure();
    void adapt_storage_for_network_state();
    
    // File system operations
    std::string generate_activation_file_path(size_t activation_id);
    bool write_activation_to_file(size_t activation_id, const TensorPtr& activation);
    TensorPtr read_activation_from_file(size_t activation_id);
    void cleanup_activation_files();
    
    // Perforance optimization
    void optimize_memory_layout();
    void defragment_memory_pools();
    void update_tier_perforance_metrics();
    
    // Error handling and recovery
    void handle_storage_error(const StorageOperation& operation, const std::string& error);
    bool attempt_storage_recovery(size_t activation_id);
    void verify_storage_integrity();
    
    // Utility methods
    size_t calculate_activation_size(const TensorPtr& activation);
    ActivationTier select_optimal_tier(size_t activation_id, const TensorPtr& activation);
    bool has_available_capacity(ActivationTier tier, size_t required_size);
    void update_storage_statistics();
    void log_storage_event(const std::string& event, size_t activation_id, double duration_ms = 0.0);
};

/**
 * @brief Storage optimization utilities for mobile
 */
namespace mobile_storage_utils {
    
    /**
     * @brief Calculate optimal storage tier configuration based on device capabilities
     * @param total_memory_mb Total available system memory
     * @param gpu_memory_mb Available GPU memory
     * @param storage_space_mb Available persistent storage space
     * @return Optimized storage configuration
     */
    StorageConfig calculate_optimal_storage_config(
        size_t total_memory_mb,
        size_t gpu_memory_mb,
        size_t storage_space_mb
    );
    
    /**
     * @brief Analyze activation access patterns to optimize storage layout
     * @param access_history History of activation accesses
     * @param time_window_ms Time window for analysis
     * @return Storage optimization recommendations
     */
    struct StorageOptimizationAdvice {
        std::vector<std::pair<size_t, ActivationTier>> recommended_placements;
        std::vector<size_t> candidates_for_prefetch;
        std::vector<size_t> candidates_for_eviction;
        float expected_perforance_improvement;
    };
    StorageOptimizationAdvice analyze_access_patterns_for_optimization(
        const std::vector<std::pair<size_t, std::chrono::steady_clock::time_point>>& access_history,
        std::chrono::milliseconds time_window_ms
    );
    
    /**
     * @brief Estimate storage perforance for different configurations
     * @param config Storage configuration to evaluate
     * @param workload_pattern Expected workload pattern
     * @return Perforance estimates
     */
    struct StoragePerforanceEstimate {
        double average_access_latency_ms;
        double sustained_bandwidth_mbps;
        double power_consumption_mw;
        float storage_efficiency_ratio;
    };
    StoragePerforanceEstimate estimate_storage_perforance(
        const StorageConfig& config,
        const std::string& workload_pattern
    );
}

} // namespace memory
} // namespace ops
