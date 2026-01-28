/**
 * @file parameter_sharder.cpp
 * @brief Implementation of ZeRO-inspired single-device parameter sharding/offload.
 */

#include "parameter_sharder.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace fs = std::filesystem;

namespace ops {
namespace sharding {

// Helper: float32 <-> fp16 (same codepath as core ops, kept local to avoid exposing internals)
static uint16_t float32_to_fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        uint32_t shifted = mantissa >> (1 - exponent + 13);
        return static_cast<uint16_t>(sign | shifted);
    }

    if (exponent >= 31) {
        uint16_t inf_nan = (mantissa == 0) ? 0x7C00u : static_cast<uint16_t>(0x7C00u | (mantissa >> 13));
        return static_cast<uint16_t>(sign | inf_nan);
    }

    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
    return half;
}

static float fp16_to_float32(uint16_t value) {
    uint32_t sign = (value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1Fu;
    uint32_t mantissa = value & 0x3FFu;

    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            exponent = exponent - 1 + 127 - 15;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        exponent = exponent - 15 + 127;
        bits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

size_t ShardEntry::num_bytes_fp32() const {
    size_t elems = 1;
    for (auto d : shape) elems *= static_cast<size_t>(d);
    size_t bytes = elems * sizeof(float);
    if (dtype == kFloat16) bytes = elems * sizeof(uint16_t);
    return bytes;
}

ParameterSharder::ParameterSharder(const ShardConfig& cfg)
    : cfg_(cfg), resident_bytes_(0), clock_(0) {
    if (cfg_.offload_dir.empty()) {
        throw std::runtime_error("ParameterSharder: offload_dir must not be empty");
    }
    fs::create_directories(cfg_.offload_dir);
}

void ParameterSharder::register_parameter(const std::string& name, const TensorPtr& tensor, bool keep_in_memory, TensorPtr* owner_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tensor) throw std::runtime_error("register_parameter: tensor is null");
    if (entries_.count(name)) throw std::runtime_error("register_parameter: duplicated name " + name);

    ShardEntry e;
    e.name = name;
    e.shape = tensor->shape();
    e.dtype = tensor->dtype();
    e.quantized = (cfg_.quantize_fp16_on_disk && e.dtype == kFloat32);
    e.path = fs::path(cfg_.offload_dir) / (sanitize_filename(name) + ".bin");
    e.tensor = tensor;
    e.owner_ptr = owner_ptr;
    e.state = ShardState::InMemory;
    e.last_used = ++clock_;

    // First write a copy to disk (ensure disk as primary storage)
    offload_entry(e);
    // Return to memory state (optionally keep, otherwise release)
    if (keep_in_memory) {
        load_entry(e);
    } else {
        if (e.owner_ptr) *e.owner_ptr = nullptr;
        e.tensor.reset();
        e.state = ShardState::Offloaded;
    }

    if (e.tensor) {
        resident_bytes_ += e.num_bytes_fp32();
    }
    entries_.emplace(name, std::move(e));
}

TensorPtr ParameterSharder::require(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw std::runtime_error("require: unknown parameter " + name);
    }
    ShardEntry& e = it->second;
    size_t need = e.num_bytes_fp32();
    ensure_budget(need, name);
    if (e.state == ShardState::Offloaded) {
        load_entry(e);
        resident_bytes_ += need;
    }
    e.last_used = ++clock_;
    return e.tensor;
}

void ParameterSharder::mark_dirty(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(name);
    if (it == entries_.end()) return;
    it->second.dirty = true;
}

void ParameterSharder::offload_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : entries_) {
        offload_entry(kv.second);
        kv.second.tensor.reset();
        kv.second.state = ShardState::Offloaded;
        kv.second.dirty = false;
    }
    resident_bytes_ = 0;
}

std::string ParameterSharder::debug_string() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "ParameterSharder{resident=" << resident_bytes_ / 1024.0 / 1024.0 << "MB, entries=" << entries_.size() << "}\n";
    for (const auto& kv : entries_) {
        const auto& e = kv.second;
        oss << "  [" << kv.first << "] state=" << (e.state == ShardState::InMemory ? "RAM" : "Disk")
            << ", bytes=" << e.num_bytes_fp32()
            << ", quant=" << (e.quantized ? "fp16" : "fp32")
            << ", dirty=" << (e.dirty ? "1" : "0")
            << "\n";
    }
    return oss.str();
}

void ParameterSharder::ensure_budget(size_t incoming_bytes, const std::string& keep_name) {
    if (incoming_bytes > cfg_.max_resident_bytes) {
        throw std::runtime_error("require: incoming parameter bigger than max_resident_bytes");
    }
    // If exceeds budget, evict by LRU (ignore currently needed keep_name)
    while (resident_bytes_ + incoming_bytes > cfg_.max_resident_bytes) {
        auto victim_it = std::min_element(entries_.begin(), entries_.end(),
            [&](const auto& a, const auto& b) {
                // Skip keep_name and those already not in memory
                const bool a_valid = (a.first != keep_name && a.second.state == ShardState::InMemory);
                const bool b_valid = (b.first != keep_name && b.second.state == ShardState::InMemory);
                if (a_valid != b_valid) return a_valid; // true < false
                if (!a_valid && !b_valid) return false;
                return a.second.last_used < b.second.last_used;
            });
        if (victim_it == entries_.end()) break;
        if (victim_it->first == keep_name || victim_it->second.state == ShardState::Offloaded) {
            break;
        }
        ShardEntry& victim = victim_it->second;
        offload_entry(victim);
        resident_bytes_ -= victim.num_bytes_fp32();
        victim.tensor.reset();
        victim.state = ShardState::Offloaded;
        victim.dirty = false;
    }
}

void ParameterSharder::offload_entry(ShardEntry& e) {
    if (e.tensor && !e.dirty && fs::exists(e.path)) {
        // Already has latest disk copy
    } else if (e.tensor) {
        // Write to disk
        std::ofstream out(e.path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("offload_entry: cannot open " + e.path);
        }
        const size_t elems = e.tensor ? static_cast<size_t>(e.tensor->numel()) : 0;
        if (e.quantized) {
            // Streaming quantization, avoid allocating equally-sized buffer for entire parameter
            const float* src = e.tensor->data<float>();
            const size_t chunk = 1 << 18; // 256K elements (~512KB)
            std::vector<uint16_t> tmp(std::min(chunk, elems));
            size_t offset = 0;
            while (offset < elems) {
                size_t cur = std::min(chunk, elems - offset);
                for (size_t i = 0; i < cur; ++i) {
                    tmp[i] = float32_to_fp16(src[offset + i]);
                }
                out.write(reinterpret_cast<const char*>(tmp.data()), cur * sizeof(uint16_t));
                offset += cur;
            }
        } else {
            const void* src = e.tensor->data_ptr();
            size_t bytes = elems * ((e.dtype == kFloat16) ? sizeof(uint16_t) : sizeof(float));
            out.write(reinterpret_cast<const char*>(src), bytes);
        }
        e.dirty = false;
    }
    // Release memory and sync external pointer
    if (e.owner_ptr) *e.owner_ptr = nullptr;
    e.tensor.reset();
}

void ParameterSharder::load_entry(ShardEntry& e) {
    // If already has memory tensor, return directly
    if (e.tensor) {
        e.state = ShardState::InMemory;
        return;
    }
    size_t elems = 1;
    for (auto d : e.shape) elems *= static_cast<size_t>(d);
    DType target_dtype = (e.quantized ? kFloat32 : e.dtype);
    TensorPtr t = std::make_shared<Tensor>(e.shape, target_dtype, kCPU);
    
    std::ifstream in(e.path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("load_entry: cannot open " + e.path);
    }
    if (e.quantized) {
        const size_t chunk = 1 << 18; // 256K elements
        std::vector<uint16_t> tmp(std::min(chunk, elems));
        float* dst = t->data<float>();
        size_t offset = 0;
        while (offset < elems) {
            size_t cur = std::min(chunk, elems - offset);
            in.read(reinterpret_cast<char*>(tmp.data()), cur * sizeof(uint16_t));
            for (size_t i = 0; i < cur; ++i) {
                dst[offset + i] = fp16_to_float32(tmp[i]);
            }
            offset += cur;
        }
    } else {
        size_t bytes = elems * ((e.dtype == kFloat16) ? sizeof(uint16_t) : sizeof(float));
        in.read(reinterpret_cast<char*>(t->data_ptr()), bytes);
    }
    e.tensor = t;
    if (e.owner_ptr) *e.owner_ptr = e.tensor;
    e.state = ShardState::InMemory;
}

std::string ParameterSharder::sanitize_filename(const std::string& name) {
    std::string out = name;
    for (auto& ch : out) {
        if (ch == '/' || ch == ' ') ch = '_';
    }
    return out;
}

} // namespace sharding
} // namespace ops
