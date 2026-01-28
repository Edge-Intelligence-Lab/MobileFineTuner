/**
 * @file mobile_efficient_attention.h
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * 4. dynamicaccuracyadjustandcompression
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 */

#pragma once

#include "../core/tensor.h"
#include "../core/ops.h"
#include <memory>
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>
#include <cmath>

namespace ops {
namespace memory {

using ops::TensorPtr;
using ops::Tensor;

/**
 * [Documentation available in English]
 */
enum class AttentionStrategy {
    STANDARD = 0,           // standardattention（highmemory）
    FLASH_ATTENTION = 1,        // [Translated]
    MOBILE_OPTIMIZED = 2,   // mobileoptimizationversion
    MEMORY_FIRST = 3,           // [Translated]
    SPEED_FIRST = 4             // [Translated]
};

/**
 * @brief chunkedcomputestrategy
 */
enum class BlockStrategy {
    UNIFORM_BLOCKS = 0,         // [Translated]
    ADAPTIVE_BLOCKS = 1,    // adaptivechunked
    IMPORTANCE_BLOCKS = 2,      // [Translated]
    MOBILE_AWARE = 3            // [Translated]
};

/**
 * [Documentation available in English]
 */
enum class AttentionPrecision {
    FP32 = 0,                  // [Translated]
    FP16 = 1,                  // [Translated]
    BF16 = 2,              // BFloat16
    MIXED = 3,             // mixed precision
    DYNAMIC = 4            // dynamicaccuracyadjust
};

/**
 * [Documentation available in English]
 */
enum class HeadGroupStrategy {
    NO_GROUPING = 0,           // [Translated]
    UNIFORM_GROUPING = 1,      // [Translated]
    ADAPTIVE_GROUPING = 2,     // [Translated]
    MOBILE_GROUPING = 3        // [Translated]
};

/**
 * @brief mobileattentionconfiguration
 */
struct MobileAttentionConfig {
    // basicparameter
    AttentionStrategy strategy = AttentionStrategy::MOBILE_OPTIMIZED;
    BlockStrategy block_strategy = BlockStrategy::MOBILE_AWARE;
    AttentionPrecision precision = AttentionPrecision::MIXED;
    HeadGroupStrategy head_grouping = HeadGroupStrategy::MOBILE_GROUPING;
    
    // chunkedparameter
    size_t block_size = 64;                    // defaultchunkedsize
    size_t min_block_size = 16;                    // [Translated]
    size_t max_block_size = 256;                   // [Translated]
    bool enable_adaptive_block_sizing = true;  // adaptivechunkedsize
    
    // memorymanageparameter
    size_t max_attention_memory_mb = 128;          // [Translated]
    float memory_pressure_threshold = 0.8f;        // [Translated]
    bool enable_attention_caching = true;      // enableattentioncache
    bool enable_kv_caching = true;             // enableKVcache
    
    // mobileoptimizationparameter
    bool enable_mobile_optimizations = true;   // enablemobileoptimization
    bool enable_neon_acceleration = true;          // [Translated]
    bool enable_gpu_acceleration = true;           // [Translated]
    bool optimize_for_power_efficiency = true; // optimizationpowerefficiency
    
    // accuracyandqualityparameter
    float attention_dropout = 0.0f;               // [Translated]
    float precision_threshold = 1e-4f;        // accuracythreshold
    bool enable_numerical_stability = true;       // [Translated]
    float temperature_scaling = 1.0f;         // temperaturescaleparameter
    
    // dynamicoptimizationparameter
    bool enable_dynamic_optimization = true;  // enabledynamicoptimization
    float battery_aware_scaling = 0.8f;           // [Translated]
    float thermal_aware_scaling = 0.7f;           // [Translated]
    
    // Perforance analysis parameters
    bool enable_attention_profiling = false;  // enableattentionperforanceanalyze
    bool log_attention_events = false;        // recordattentionevent
    std::string profiling_output_path = "./attention_profile.json";
};

/**
 * @brief attentioncomputestatisticsinforation
 */
struct AttentionStats {
    // basicstatistics
    size_t total_attention_calls;
    size_t total_blocks_processed;
    size_t cache_hits;
    size_t cache_misses;
    
    // memorystatistics
    size_t peak_memory_usage_bytes;
    size_t average_memory_usage_bytes;
    size_t memory_saved_by_blocking;
    float average_memory_efficiency;
    
    // perforancestatistics
    double total_attention_time_ms;
    double average_attention_time_ms;
    double total_softmax_time_ms;
    double total_matmul_time_ms;
    
    // mobilestatistics
    size_t neon_accelerated_operations;
    size_t gpu_accelerated_operations;
    size_t battery_optimized_calls;
    size_t thermal_optimized_calls;
    
    // accuracystatistics
    double average_attention_entropy;
    size_t precision_downgrades;
    size_t precision_upgrades;
};

/**
 * [Documentation available in English]
 */
struct AttentionBlock {
    size_t block_id;
    size_t start_row;
    size_t end_row;
    size_t start_col;
    size_t end_col;
    
    // chunkeddata
    TensorPtr q_block;    // Querychunked
    TensorPtr k_block;    // Keychunked
    TensorPtr v_block;    // Valuechunked
    TensorPtr scores_block;     // [Translated]
    
        // [Translated]
    TensorPtr row_max;        // [Translated]
    TensorPtr row_sum;        // [Translated]
    TensorPtr output_block; // outputchunked
    
    // computestate
    bool is_computed;
    bool is_cached;
    std::chrono::steady_clock::time_point last_access_time;
    
    AttentionBlock(size_t id, size_t sr, size_t er, size_t sc, size_t ec)
        : block_id(id), start_row(sr), end_row(er), start_col(sc), end_col(ec),
          is_computed(false), is_cached(false) {
        last_access_time = std::chrono::steady_clock::now();
    }
};

/**
 * [Documentation available in English]
 */
struct KVCacheEntry {
    TensorPtr cached_k;
    TensorPtr cached_v;
    size_t sequence_length;
    std::vector<int64_t> original_shape;
    std::chrono::steady_clock::time_point creation_time;
    std::chrono::steady_clock::time_point last_access_time;
    size_t access_count;
    bool is_compressed;
    
    KVCacheEntry(const TensorPtr& k, const TensorPtr& v, size_t seq_len)
        : cached_k(k), cached_v(v), sequence_length(seq_len), access_count(0), is_compressed(false) {
        creation_time = std::chrono::steady_clock::now();
        last_access_time = creation_time;
        if (k) original_shape = k->shape();
    }
};

/**
 * [Documentation available in English]
 */
class MobileEfficientAttention {
private:
    MobileAttentionConfig config_;
    
    // chunkedmanage
    std::vector<std::unique_ptr<AttentionBlock>> current_blocks_;
    std::unordered_map<size_t, std::unique_ptr<AttentionBlock>> block_cache_;
    
    // KVcachemanage
    std::unordered_map<std::string, std::unique_ptr<KVCacheEntry>> kv_cache_;
    size_t max_kv_cache_size_;
    size_t current_kv_cache_size_;
    
    // mobilestatemonitor
    std::atomic<float> current_memory_pressure_;
    std::atomic<int> current_battery_level_;
    std::atomic<float> current_temperature_;
    std::atomic<bool> is_app_foreground_;
    
    // threadsafe
    mutable std::mutex attention_mutex_;
    mutable std::mutex cache_mutex_;
    mutable std::mutex stats_mutex_;
    
    // statisticsinfo
    AttentionStats stats_;
    
    // dynamicoptimizationstate
    std::atomic<AttentionPrecision> current_precision_;
    std::atomic<size_t> current_block_size_;
    std::deque<double> recent_computation_times_;

public:
    explicit MobileEfficientAttention(const MobileAttentionConfig& config);
    ~MobileEfficientAttention();
    
    /**
     * @brief computememoryefficientmulti-head attention
     * @param query Querytensor [batch, seq_len, num_heads, head_dim]
     * @param key Keytensor [batch, seq_len, num_heads, head_dim]
     * @param value Valuetensor [batch, seq_len, num_heads, head_dim]
     * [Documentation available in English]
     * @param cache_key KVcachekey（used forinferenceoptimization）
     * @return attentionoutputtensor
     */
    TensorPtr compute_attention(
        const TensorPtr& query,
        const TensorPtr& key,
        const TensorPtr& value,
        const TensorPtr& mask = nullptr,
        const std::string& cache_key = ""
    );
    
    /**
     * [Documentation available in English]
     * @param query Querytensor
     * @param key Keytensor  
     * @param value Valuetensor
     * @param cache_key KVcachekey
     * @return attentionoutputtensor
     */
    TensorPtr compute_causal_attention(
        const TensorPtr& query,
        const TensorPtr& key,
        const TensorPtr& value,
        const std::string& cache_key = ""
    );
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation available in English]
     * @return attentionoutputtensor
     */
    TensorPtr compute_cross_attention(
        const TensorPtr& query,
        const TensorPtr& key,
        const TensorPtr& value,
        const TensorPtr& mask = nullptr
    );
    
    /**
     * [Documentation available in English]
     * @param cache_key cachekey
     * @param new_key newKeytensor
     * @param new_value newValuetensor
     */
    void update_kv_cache(
        const std::string& cache_key,
        const TensorPtr& new_key,
        const TensorPtr& new_value
    );
    
    /**
     * @brief cleanupKVcache
     * [Documentation available in English]
     */
    void clear_kv_cache(const std::string& cache_key = "");
    
    /**
     * @brief updatemobilesystemstate
     * [Documentation available in English]
     * @param battery_level batterybattery level (0-100)  
     * @param temperature devicetemperature
     * @param is_foreground is notforegroundrunning
     */
    void update_mobile_state(float memory_pressure, int battery_level,
                            float temperature, bool is_foreground);
    
    /**
     * @brief configurationattentionparameter
     * @param config newattentionconfiguration
     */
    void configure_attention(const MobileAttentionConfig& config);
    
    /**
     * @brief acquireattentionstatisticsinforation
     * @return currentstatisticsinforation
     */
    AttentionStats get_attention_stats() const;
    
    /**
     * @brief optimizationattentioncomputeconfiguration
     * [Documentation available in English]
     */
    void optimize_attention_configuration();
    
    /**
     * [Documentation available in English]
     * @param query_shape Querytensorshape
     * @param key_shape Keytensorshape
     * @param strategy computestrategy
     * [Documentation available in English]
     */
    size_t estimate_memory_requirement(
        const std::vector<int64_t>& query_shape,
        const std::vector<int64_t>& key_shape,
        AttentionStrategy strategy = AttentionStrategy::MOBILE_OPTIMIZED
    );
    
    /**
     * @brief exportattentionperforanceanalyzereport
     * @param report_path reportsavepath
     */
    void export_profiling_report(const std::string& report_path) const;

private:
    // coreattentionalgorithm
    TensorPtr compute_flash_attention(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
        const TensorPtr& mask);
    TensorPtr compute_mobile_optimized_attention(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
        const TensorPtr& mask);
    TensorPtr compute_memory_first_attention(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
        const TensorPtr& mask);
    
    // chunkedalgorithm
    std::vector<std::unique_ptr<AttentionBlock>> create_attention_blocks(
        const std::vector<int64_t>& query_shape,
        const std::vector<int64_t>& key_shape);
    void compute_block_attention(AttentionBlock& block, const TensorPtr& mask);
    TensorPtr merge_attention_blocks(const std::vector<std::unique_ptr<AttentionBlock>>& blocks);
    
        // [Translated]
    void online_softmax_forward(
        const TensorPtr& scores, TensorPtr& row_max, TensorPtr& row_sum, TensorPtr& output);
    void online_softmax_update(
        const TensorPtr& new_scores, TensorPtr& row_max, TensorPtr& row_sum, TensorPtr& output);
    
        // [Translated]
    void apply_neon_acceleration(TensorPtr& tensor);
    void apply_gpu_acceleration(TensorPtr& tensor);
    bool should_use_neon_for_operation(size_t tensor_size);
    bool should_use_gpu_for_operation(size_t tensor_size);
    
    // dynamicaccuracymanage
    AttentionPrecision determine_optimal_precision(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value);
    TensorPtr convert_precision(const TensorPtr& tensor, AttentionPrecision target_precision);
    void adapt_precision_for_mobile_state();
    
    // KVcachemanage
    bool should_cache_kv(const std::string& cache_key, size_t sequence_length);
    void evict_old_kv_cache();
    void compress_kv_cache_if_needed();
    std::pair<TensorPtr, TensorPtr> get_cached_kv(const std::string& cache_key);
    
    // chunkedoptimizationalgorithm
    size_t calculate_optimal_block_size(
        const std::vector<int64_t>& query_shape, 
        const std::vector<int64_t>& key_shape);
    void adapt_block_size_for_mobile_state();
    bool should_use_adaptive_blocking();
    
    // mobileoptimizationstrategy
    void apply_battery_aware_optimizations();
    void apply_thermal_aware_optimizations();  
    void apply_memory_pressure_optimizations();
    void apply_ui_responsiveness_optimizations();
    
        // [Translated]
    TensorPtr apply_temperature_scaling(const TensorPtr& scores);
    TensorPtr apply_attention_dropout(const TensorPtr& attention_weights);
    void ensure_numerical_stability(TensorPtr& tensor);
    
    // perforancemonitorandstatistics
    void update_attention_stats(double computation_time, size_t memory_used);
    void record_mobile_optimization(const std::string& optimization_type);
    void analyze_attention_patterns();
    
    // toolmethod
    std::vector<int64_t> calculate_output_shape(
        const std::vector<int64_t>& query_shape,
        const std::vector<int64_t>& value_shape);
    bool is_causal_mask_needed(const std::vector<int64_t>& query_shape);
    TensorPtr create_causal_mask(size_t sequence_length);
    
    // memorymanagetool
    void cleanup_expired_blocks();
    void handle_attention_memory_pressure();
    size_t get_current_attention_memory_usage();
    
    // Event logging and analytics
    void log_attention_event(const std::string& event, const std::string& details = "");
    void validate_attention_inputs(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value);
};

/**
 * @brief attentiontoolfunctionnamespace
 */
namespace attention_utils {
    
    /**
     * @brief computeattentioncomputetheoreticalmemorycomplexity
     * @param sequence_length sequencelength
     * @param hidden_size hidden layersize
     * [Documentation available in English]
     * [Documentation available in English]
     */
    std::pair<size_t, size_t> calculate_memory_complexity(
        size_t sequence_length, size_t hidden_size, size_t num_heads);
    
    /**
     * [Documentation available in English]
     * @param attention_weights attentionweightmatrix
     * [Documentation available in English]
     * [Documentation available in English]
     */
    struct SparsityAnalysis {
        float sparsity_ratio;                // [Translated]
        std::vector<int> sparse_heads;         // [Translated]
        bool can_optimize;           // is notcanoptimization
    };
    SparsityAnalysis analyze_attention_sparsity(
        const TensorPtr& attention_weights, float sparsity_threshold = 0.1f);
    
    /**
     * [Documentation available in English]
     * [Documentation available in English]
     * @param available_memory availablememory
     * [Documentation available in English]
     * [Documentation available in English]
     */
    std::vector<std::vector<int>> optimize_head_grouping(
        int num_heads, size_t available_memory, int target_groups = -1);
    
    /**
     * [Documentation available in English]
     * @param sequence_length sequencelength
     * [Documentation available in English]
     * [Documentation available in English]
     * [Documentation available in English]
     */
    TensorPtr create_mobile_optimized_mask(
        size_t sequence_length, 
        const std::string& mask_type,
        const TensorPtr& custom_mask = nullptr);
}

} // namespace memory
} // namespace ops
