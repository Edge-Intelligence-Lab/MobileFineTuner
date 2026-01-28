/**
 * @file parameter_sharder.h
 * @brief ZeRO-inspired single-device parameter sharding/offload for mobile.
 *
 * Design goals:
 * - Persist parameter segments to disk, only load active segments back to memory during forward/backward passes.
 * - LRU eviction + resident memory limit (by bytes) control.
 * - Disk storage supports optional FP16 quantization (quantize on write, dequantize on read).
 * - Only depends on existing Tensor class, easy to integrate with current training pipeline.
 *
 * Usage example:
 *   ParameterSharder sharder({"/tmp/offload", 64 * 1024 * 1024, true});
 *   sharder.register_parameter("blocks.0.attn.qkv.weight", tensor_ptr);
 *   auto w = sharder.require("blocks.0.attn.qkv.weight");  // Ensure in memory
 *   ... // Forward/backward
 *   sharder.offload_all();  // Release at end of training
 *
 * Note: Current implementation doesn't automatically replace TensorPtr in existing models.
 * To truly release memory, model should access parameters through sharder-managed pointers
 * (recommended to register during model construction).
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <mutex>

#include "../../finetune_ops/core/tensor.h"

namespace ops {
namespace sharding {

struct ShardConfig {
    std::string offload_dir;        // Disk storage directory
    size_t max_resident_bytes = 256 * 1024 * 1024;  // Resident memory limit
    bool quantize_fp16_on_disk = true;              // Whether to quantize to FP16 when writing to disk
};

enum class ShardState {
    InMemory,
    Offloaded
};

struct ShardEntry {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype = kFloat32;
    std::string path;       // Disk file path
    bool quantized = false; // Whether disk uses FP16 storage
    bool dirty = false;     // Whether memory data hasn't been written to disk
    ShardState state = ShardState::Offloaded;
    TensorPtr tensor;       // Only valid when InMemory
    uint64_t last_used = 0; // LRU timestamp
    TensorPtr* owner_ptr = nullptr; // Optional: External owner (for easy null/refill)
    
    size_t num_bytes_fp32() const;
};

class ParameterSharder {
public:
    explicit ParameterSharder(const ShardConfig& cfg);
    
    // Register parameter: by default immediately write to disk and (optionally) keep in memory.
    // owner_ptr (optional) points to the real holder; will be synchronized (nulled/refilled) during offload/load.
    void register_parameter(const std::string& name, const TensorPtr& tensor, bool keep_in_memory = true, TensorPtr* owner_ptr = nullptr);
    
    // Ensure parameter is in memory (may trigger eviction of other segments).
    TensorPtr require(const std::string& name);
    
    // Mark parameter as modified (will be rewritten to disk on future offload).
    void mark_dirty(const std::string& name);
    
    // Force offload all (write to disk then release memory).
    void offload_all();
    
    // Debug information
    std::string debug_string() const;
    
private:
    ShardConfig cfg_;
    size_t resident_bytes_;
    uint64_t clock_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ShardEntry> entries_;
    
    void ensure_budget(size_t incoming_bytes, const std::string& keep_name);
    void offload_entry(ShardEntry& e);
    void load_entry(ShardEntry& e);
    static std::string sanitize_filename(const std::string& name);
};

} // namespace sharding
} // namespace ops
