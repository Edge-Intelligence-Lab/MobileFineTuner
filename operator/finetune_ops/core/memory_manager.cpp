/**
 * @file memory_manager.cpp
 * @brief Implementation of advanced memory management for the operators framework
 */

#include "memory_manager.h"
#include "tensor.h"
#include "logger.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstring>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <malloc/malloc.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <fstream>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

namespace ops {

// Static member initialization
std::unique_ptr<MemoryManager> MemoryManager::instance_;
std::mutex MemoryManager::instance_mutex_;
double MemoryMonitor::memory_threshold_ = 0.8; // 80% by default

// MemoryPool implementsation
void* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Align size to 8 bytes for better perforance
    size = (size + 7) & ~7;
    
    size_t bucket_idx = get_bucket_index(size);
    
    // Try to find a suitable block in the bucket
    for (auto* block : size_buckets_[bucket_idx]) {
        if (!block->in_use && block->size >= size) {
            block->in_use = true;
            total_in_use_ += block->size;
            return block->ptr;
        }
    }
    
    // No suitable block found, allocate new one
    void* ptr = std::malloc(size);
    if (!ptr) {
        OPS_LOG_ERROR_F("Failed to allocate memory: %zu bytes", size);
        return nullptr;
    }
    
    auto block = std::make_unique<MemoryBlock>(ptr, size);
    block->in_use = true;
    
    size_buckets_[bucket_idx].push_back(block.get());
    blocks_.push_back(std::move(block));
    
    total_allocated_ += size;
    total_in_use_ += size;
    total_peak_ = std::max(total_peak_, total_in_use_);
    
    // Only log large allocations (>10MB)
    if (size > 10 * 1024 * 1024) {
        OPS_LOG_DEBUG_F("Allocated %zu MB, total in use: %zu MB", size / (1024 * 1024), total_in_use_ / (1024 * 1024));
    }
    return ptr;
}

void MemoryPool::deallocate(void* ptr, size_t size [[maybe_unused]]) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find the block
    for (auto& block : blocks_) {
        if (block->ptr == ptr) {
            if (block->in_use) {
                block->in_use = false;
                total_in_use_ -= block->size;
                // Only log large deallocations (>10MB)
                if (block->size > 10 * 1024 * 1024) {
                    OPS_LOG_DEBUG_F("Deallocated %zu MB, total in use: %zu MB", 
                                 block->size / (1024 * 1024), total_in_use_ / (1024 * 1024));
                }
            }
            return;
        }
    }
    
    OPS_LOG_WARNING("Attempted to deallocate unknown pointer");
}

void MemoryPool::clear_unused() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t freed_count = 0;
    size_t freed_bytes = 0;
    
    // Clear bucket references first
    for (auto& bucket : size_buckets_) {
        bucket.clear();
    }
    
    // Remove unused blocks (actually free back to the OS)
    auto it = std::remove_if(blocks_.begin(), blocks_.end(),
        [&](const std::unique_ptr<MemoryBlock>& block) {
            if (!block->in_use) {
                freed_count++;
                freed_bytes += block->size;
                total_allocated_ -= block->size;
                // ✅ MemoryBlock destructor automatically calls std::free(ptr)
                return true;  // Remove unique_ptr to trigger destructor
            } else {
                // Re-add to appropriate bucket
                size_t bucket_idx = get_bucket_index(block->size);
                size_buckets_[bucket_idx].push_back(block.get());
                return false;
            }
        });
    
    blocks_.erase(it, blocks_.end());  // ✅ This invokes MemoryBlock::~MemoryBlock() → std::free()
    
    // Log cleanup info (debug)
    if (freed_bytes > 10 * 1024 * 1024) {  // Log when >10MB
        OPS_LOG_DEBUG_F("Freed %zu MB from %zu blocks back to OS", 
                         freed_bytes / (1024 * 1024), freed_count);
    }
    
    // Always print pool stats to monitor for net growth
    OPS_LOG_DEBUG_F("[MemPool] blocks=%zu, in_use=%zu MB, allocated=%zu MB",
                    blocks_.size(),
                    get_total_in_use() / (1024 * 1024),
                    get_total_allocated() / (1024 * 1024));
    
    // macOS: explicitly nudge OS to reclaim free pages alongside clear_unused frees
    #ifdef __APPLE__
    malloc_zone_pressure_relief(nullptr, 0);
    #endif
}

void MemoryPool::clear_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& bucket : size_buckets_) {
        bucket.clear();
    }
    
    blocks_.clear();
    total_allocated_ = 0;
    total_in_use_ = 0;
    
    OPS_LOG_INFO("Cleared all memory blocks");
}

size_t MemoryPool::get_available_memory() const {
    if (total_allocated_ <= total_in_use_) {
        return 0;
    }
    return total_allocated_ - total_in_use_;
}

void MemoryPool::print_stats() const {
    // Note: This method doesn't modify state, so we skip the lock for const method
    
    std::cout << "Memory Pool Statistics:\n";
    std::cout << "  Total allocated: " << total_allocated_ / (1024 * 1024) << " MB\n";
    std::cout << "  Currently in use: " << total_in_use_ / (1024 * 1024) << " MB\n";
    std::cout << "  Peak usage: " << total_peak_ / (1024 * 1024) << " MB\n";
    std::cout << "  Available: " << get_available_memory() / (1024 * 1024) << " MB\n";
    std::cout << "  Total blocks: " << blocks_.size() << "\n";
    
    // Bucket statistics
    for (size_t i = 0; i < size_buckets_.size(); ++i) {
        if (!size_buckets_[i].empty()) {
            std::cout << "  Bucket " << i << ": " << size_buckets_[i].size() << " blocks\n";
        }
    }
}

// TensorCache implementsation
std::shared_ptr<Tensor> TensorCache::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        if (auto tensor = it->second.lock()) {
            return tensor;
        } else {
            // Remove expired weak_ptr
            cache_.erase(it);
        }
    }
    
    return nullptr;
}

void TensorCache::put(const std::string& key, std::shared_ptr<Tensor> tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Remove expired entries if cache is full
    if (cache_.size() >= max_cache_size_) {
        auto it = cache_.begin();
        while (it != cache_.end()) {
            if (it->second.expired()) {
                it = cache_.erase(it);
            } else {
                ++it;
            }
        }
        
        // If still full, remove oldest entry (simple strategy)
        if (cache_.size() >= max_cache_size_) {
            cache_.erase(cache_.begin());
        }
    }
    
    cache_[key] = tensor;
}

void TensorCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

// MemoryManager implementsation
MemoryManager& MemoryManager::instance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::unique_ptr<MemoryManager>(new MemoryManager());
    }
    return *instance_;
}

void MemoryManager::register_tensor(std::shared_ptr<Tensor> tensor) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    computation_graph_.push_back(tensor);
}

void MemoryManager::clear_computation_graph() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    computation_graph_.clear();
    OPS_LOG_DEBUG("Cleared computation graph");
}

void MemoryManager::cleanup_dead_references() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    size_t initial_size = computation_graph_.size();
    
    auto it = std::remove_if(computation_graph_.begin(), computation_graph_.end(),
        [](const std::weak_ptr<Tensor>& weak_tensor) {
            return weak_tensor.expired();
        });
    
    computation_graph_.erase(it, computation_graph_.end());
    
    size_t removed = initial_size - computation_graph_.size();
    if (removed > 0) {
        // Only log significant cleanup (>100 references)
        if (removed > 100) {
            OPS_LOG_DEBUG_F("Cleaned up %zu dead tensor references", removed);
        }
    }
}

void MemoryManager::force_cleanup() {
    clear_cache();
    clear_unused_memory();
    clear_computation_graph();
    cleanup_dead_references();
    
    // macOS: pressure the OS to return free pages promptly
    // Call only after actual cleanup to avoid extra overhead
    #ifdef __APPLE__
    malloc_zone_pressure_relief(nullptr, 0);
    #endif
    
    // Silent force cleanup - this is normal operation
}

void MemoryManager::print_memory_stats() const {
    std::cout << "\n=== Memory Manager Statistics ===\n";
    memory_pool_.print_stats();
    std::cout << "Tensor cache size: " << tensor_cache_.size() << "\n";
    std::cout << "Computation graph size: " << computation_graph_.size() << "\n";
    
    // System memory info
    size_t system_usage = MemoryMonitor::get_system_memory_usage();
    size_t system_available = MemoryMonitor::get_system_available_memory();
    
    std::cout << "System memory usage: " << system_usage / (1024 * 1024) << " MB\n";
    std::cout << "System available memory: " << system_available / (1024 * 1024) << " MB\n";
    std::cout << "Memory pressure: " << (MemoryMonitor::is_memory_pressure() ? "YES" : "NO") << "\n";
    std::cout << "===============================\n\n";
}

// MemoryMonitor implementsation
size_t MemoryMonitor::get_system_memory_usage() {
#ifdef __APPLE__
    mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        return info.resident_size;
    }
#elif defined(__linux__)
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("VmRSS:") == 0) {
            size_t kb = std::stoul(line.substr(6));
            return kb * 1024;
        }
    }
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#endif
    return 0;
}

size_t MemoryMonitor::get_system_available_memory() {
#ifdef __APPLE__
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    size_t size = sizeof(uint64_t);
    uint64_t total_memory;
    sysctl(mib, 2, &total_memory, &size, NULL, 0);
    
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &count);
    
    uint64_t free_memory = (vm_stat.free_count + vm_stat.inactive_count) * vm_page_size;
    return free_memory;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return si.freeram * si.mem_unit;
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return status.ullAvailPhys;
    }
#endif
    return 0;
}

bool MemoryMonitor::is_memory_pressure() {
    size_t available = get_system_available_memory();
    size_t total = available + get_system_memory_usage();
    
    if (total == 0) return false;
    
    double usage_ratio = static_cast<double>(get_system_memory_usage()) / total;
    return usage_ratio > memory_threshold_;
}

void MemoryMonitor::log_memory_status(const std::string& context) {
    size_t usage = get_system_memory_usage();
    size_t available = get_system_available_memory();
    bool pressure = is_memory_pressure();
    
    // Only log if there's actual memory pressure
    if (pressure) {
        OPS_LOG_ERROR_F("%s - Memory pressure detected: %zu MB used, %zu MB available", 
                         context.c_str(),
                         usage / (1024 * 1024),
                         available / (1024 * 1024));
    }
}

void MemoryMonitor::set_memory_threshold(double percentage) {
    memory_threshold_ = std::clamp(percentage, 0.0, 1.0);
}

bool MemoryMonitor::check_memory_threshold() {
    if (is_memory_pressure()) {
        OPS_LOG_WARNING("Memory pressure detected, consider cleaning up");
        return false;
    }
    return true;
}

} // namespace ops
