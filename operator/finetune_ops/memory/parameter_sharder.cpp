#include "parameter_sharder.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace fs = std::filesystem;

namespace ops {
namespace sharding {

size_t ParameterSharder::validate_tensor(const TensorPtr& tensor) {
    if (!tensor || tensor->requires_grad() || tensor->dtype() != kFloat32 ||
        !tensor->device().is_cpu())
        throw TensorError("ParameterSharder requires frozen CPU FP32 tensors");
    size_t bytes = sizeof(float);
    for (auto dimension : tensor->shape()) {
        if (dimension <= 0 || static_cast<uint64_t>(dimension) >
            std::numeric_limits<size_t>::max() / bytes)
            throw TensorError("ParameterSharder: invalid or overflowing shape");
        bytes *= static_cast<size_t>(dimension);
    }
    if (!tensor->data_ptr() || bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
        throw TensorError("ParameterSharder: invalid payload size");
    return bytes;
}

ParameterSharder::ParameterSharder(const ShardConfig& config) : config_(config) {
    if (config.offload_dir.empty() || !config.max_resident_bytes || config.quantize_fp16_on_disk)
        throw TensorError("ParameterSharder requires a directory, positive budget and lossless FP32 storage");
    fs::create_directories(config.offload_dir);
    static std::atomic<uint64_t> sequence{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (;;) {
        auto path = fs::path(config.offload_dir) /
            ("shards-" + std::to_string(stamp) + "-" + std::to_string(sequence++));
        if (fs::create_directory(path)) {
            storage_dir_ = path.string();
            break;
        }
    }
}

void ParameterSharder::register_parameter(const std::string& name, const TensorPtr& tensor,
                                         bool keep_in_memory, TensorPtr* owner_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t bytes = validate_tensor(tensor);
    if (name.empty() || entries_.count(name)) throw TensorError("ParameterSharder: empty or duplicate name");
    if (bytes > config_.max_resident_bytes) throw TensorError("ParameterSharder: tensor exceeds budget");
    if (owner_ptr && *owner_ptr != tensor) throw TensorError("ParameterSharder: owner does not hold tensor");
    for (const auto& item : entries_)
        if (owner_ptr && item.second.owner == owner_ptr)
            throw TensorError("ParameterSharder: owner already registered");
    Entry entry;
    entry.shape = tensor->shape();
    // Numeric per-instance filenames cannot collide through name sanitization.
    entry.path = (fs::path(storage_dir_) / (std::to_string(entries_.size()) + ".bin")).string();
    entry.bytes = bytes;
    entry.tensor = tensor;
    entry.owner = owner_ptr;
    entry.last_used = ++clock_;
    write_entry(entry); // Failed I/O must never clear the caller's only copy.
    if (keep_in_memory) ensure_budget(bytes);
    auto inserted = entries_.emplace(name, std::move(entry)).first;
    ++stats_.entries;
    if (keep_in_memory) stats_.resident_bytes += bytes;
    else {
        if (owner_ptr) owner_ptr->reset();
        inserted->second.tensor.reset();
    }
}

void ParameterSharder::write_entry(Entry& entry) {
    if (validate_tensor(entry.tensor) != entry.bytes || entry.tensor->shape() != entry.shape)
        throw TensorError("ParameterSharder: registered tensor shape changed");
    const std::string temporary = entry.path + ".tmp";
    entry.dirty = true;
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw TensorError("ParameterSharder: cannot open " + temporary);
        // Public Tensor stores contiguous values; no private strides API is assumed.
        output.write(static_cast<const char*>(entry.tensor->data_ptr()), entry.bytes);
        if (!output) throw TensorError("ParameterSharder: incomplete write " + temporary);
        output.close();
        if (!output) throw TensorError("ParameterSharder: close failed " + temporary);
        fs::rename(temporary, entry.path);
    } catch (...) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
    entry.dirty = false;
    ++stats_.writes;
    stats_.bytes_written += entry.bytes;
}

TensorPtr ParameterSharder::read_entry(const Entry& entry) {
    if (fs::file_size(entry.path) != entry.bytes)
        throw TensorError("ParameterSharder: wrong payload size " + entry.path);
    auto tensor = std::make_shared<Tensor>(entry.shape, kFloat32, kCPU);
    std::ifstream input(entry.path, std::ios::binary);
    if (!input) throw TensorError("ParameterSharder: cannot open " + entry.path);
    input.read(static_cast<char*>(tensor->data_ptr()), entry.bytes);
    if (!input) throw TensorError("ParameterSharder: incomplete read " + entry.path);
    input.close();
    if (!input) throw TensorError("ParameterSharder: close failed " + entry.path);
    return tensor;
}

void ParameterSharder::offload_entry(Entry& entry) {
    if (!entry.tensor) return;
    validate_tensor(entry.tensor);
    if (entry.owner && *entry.owner != entry.tensor)
        throw TensorError("ParameterSharder: owner was rebound");
    std::error_code error;
    const auto size = fs::file_size(entry.path, error);
    if (entry.dirty || error || size != entry.bytes) write_entry(entry);
    if (entry.owner) entry.owner->reset();
    entry.tensor.reset();
    stats_.resident_bytes -= entry.bytes;
    ++stats_.evictions;
}

void ParameterSharder::ensure_budget(size_t incoming) {
    while (stats_.resident_bytes > config_.max_resident_bytes - incoming) {
        Entry* victim = nullptr;
        for (auto& item : entries_) {
            auto& entry = item.second;
            if (entry.tensor && (!victim || entry.last_used < victim->last_used)) victim = &entry;
        }
        if (!victim) throw TensorError("ParameterSharder: no eviction candidate");
        offload_entry(*victim);
    }
}

TensorPtr ParameterSharder::require(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(name);
    if (found == entries_.end()) throw TensorError("ParameterSharder: unknown parameter " + name);
    auto& entry = found->second;
    if (entry.owner && *entry.owner != entry.tensor) throw TensorError("ParameterSharder: owner was rebound");
    if (!entry.tensor) {
        // Validate/read before eviction so a missing/truncated file preserves residents.
        auto tensor = read_entry(entry);
        ensure_budget(entry.bytes);
        entry.tensor = std::move(tensor);
        if (entry.owner) *entry.owner = entry.tensor;
        stats_.resident_bytes += entry.bytes;
        ++stats_.loads;
        stats_.bytes_read += entry.bytes;
    } else validate_tensor(entry.tensor);
    entry.last_used = ++clock_;
    return entry.tensor;
}

void ParameterSharder::mark_dirty(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(name);
    if (found == entries_.end() || !found->second.tensor)
        throw TensorError("ParameterSharder: mark_dirty requires a resident parameter");
    validate_tensor(found->second.tensor);
    found->second.dirty = true;
}

void ParameterSharder::offload_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : entries_) offload_entry(item.second);
}

void ParameterSharder::evict(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(name);
    if (found == entries_.end()) throw TensorError("ParameterSharder: unknown parameter " + name);
    offload_entry(found->second);
}

ShardStats ParameterSharder::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

std::string ParameterSharder::debug_string() const {
    const auto snapshot = stats();
    std::ostringstream output;
    output << "ParameterSharder{managed_resident_bytes=" << snapshot.resident_bytes
           << ", entries=" << snapshot.entries << ", loads=" << snapshot.loads
           << ", evictions=" << snapshot.evictions << "}";
    return output.str();
}

} // namespace sharding
} // namespace ops
