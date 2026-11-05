#pragma once

#include "../../finetune_ops/core/tensor.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ops {
namespace sharding {

struct ShardConfig {
    std::string offload_dir;
    size_t max_resident_bytes = 256 * 1024 * 1024;
    // Kept for source compatibility; true is rejected (lossless FP32 only).
    bool quantize_fp16_on_disk = false;
};

struct ShardStats {
    // Logical managed residency, NOT RSS, live alias storage, or a process cap.
    size_t resident_bytes = 0;
    size_t entries = 0;
    uint64_t loads = 0;
    uint64_t evictions = 0;
    // Payload bytes in successful operations, NOT physical device I/O.
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    uint64_t writes = 0;
};

// Single-device frozen CPU FP32 storage, not distributed/optimizer sharding.
// Each instance owns a unique subdirectory; files are retained on destruction.
// TensorPtr aliases remain valid and can pin memory after logical offload. Tied
// weights should use one name and be rebound by the model after require().
class ParameterSharder {
public:
    explicit ParameterSharder(const ShardConfig& config);
    // owner_ptr must outlive this sharder and must not be rebound by the caller.
    // Registration persists current values, even when keeping the same tensor in RAM.
    void register_parameter(const std::string& name, const TensorPtr& tensor,
                            bool keep_in_memory = true, TensorPtr* owner_ptr = nullptr);
    TensorPtr require(const std::string& name);
    void evict(const std::string& name);
    // Call after any deliberate edit of a resident frozen tensor (including aliases).
    void mark_dirty(const std::string& name);
    void offload_all();
    ShardStats stats() const;
    std::string debug_string() const;
    const std::string& storage_directory() const { return storage_dir_; }

private:
    struct Entry {
        std::vector<int64_t> shape;
        std::string path;
        size_t bytes = 0;
        TensorPtr tensor;
        TensorPtr* owner = nullptr;
        bool dirty = true;
        uint64_t last_used = 0;
    };
    ShardConfig config_;
    std::string storage_dir_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
    ShardStats stats_;
    uint64_t clock_ = 0;

    void ensure_budget(size_t incoming);
    void write_entry(Entry& entry);
    void offload_entry(Entry& entry);
    TensorPtr read_entry(const Entry& entry);
    static size_t validate_tensor(const TensorPtr& tensor);
};

} // namespace sharding
} // namespace ops
