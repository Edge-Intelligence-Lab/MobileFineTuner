/**
 * @file memory_manager.h
 * @brief Advanced memory management for the operators framework
 * 
 * This file providess memory pool, caching, and automatic cleanup mechanisms
 * to prevent memory leaks and improve perforance in deep learning training.
 */

#pragma once

#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <cstddef>
#include <algorithm>

namespace ops {

/**
 * @brief Memory block structure for pooled allocation
 */
struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    bool in_use = false;
    
    MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
    ~MemoryBlock() {
        if (ptr) {
            std::free(ptr);
        }
    }
};

/**
 * @brief Memory pool for efficient tensor allocation
 */
class MemoryPool {
private:
    std::vector<std::unique_ptr<MemoryBlock>> blocks_;
    std::mutex mutex_;
    size_t total_allocated_ = 0;
    size_t total_peak_ = 0;
    size_t total_in_use_ = 0;
    
    // Size buckets for different allocation sizes
    static constexpr size_t BUCKET_COUNT = 32;
    std::vector<std::vector<MemoryBlock*>> size_buckets_;
    
    size_t get_bucket_index(size_t size) const {
        if (size <= 256) return 0;
        size_t bucket = 0;
        size_t s = size;
        while (s > 256 && bucket < BUCKET_COUNT - 1) {
            s >>= 1;
            bucket++;
        }
        return bucket;
    }

public:
    MemoryPool() : size_buckets_(BUCKET_COUNT) {}
    
    ~MemoryPool() {
        clear_all();
    }
    
    void* allocate(size_t size);
    void deallocate(void* ptr, size_t size);
    void clear_unused();
    void clear_all();
    
    // Statistics
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_total_peak() const { return total_peak_; }
    size_t get_total_in_use() const { return total_in_use_; }
    size_t get_available_memory() const;
    
    void print_stats() const;
};

/**
 * @brief Tensor cache for reusing computation results
 */
class TensorCache {
private:
    std::unordered_map<std::string, std::weak_ptr<class Tensor>> cache_;
    std::mutex mutex_;
    size_t max_cache_size_ = 1000; // Maximum number of cached tensors
    
public:
    std::shared_ptr<class Tensor> get(const std::string& key);
    void put(const std::string& key, std::shared_ptr<class Tensor> tensor);
    void clear();
    void set_max_size(size_t max_size) { max_cache_size_ = max_size; }
    size_t size() const { return cache_.size(); }
};

/**
 * @brief Global memory manager singleton
 */
class MemoryManager {
private:
    static std::unique_ptr<MemoryManager> instance_;
    static std::mutex instance_mutex_;
    
    MemoryPool memory_pool_;
    TensorCache tensor_cache_;
    
    // Computational graph management
    std::vector<std::weak_ptr<class Tensor>> computation_graph_;
    std::mutex graph_mutex_;
    
    MemoryManager() = default;

public:
    static MemoryManager& instance();
    
    // Memory pool interface
    void* allocate(size_t size) { return memory_pool_.allocate(size); }
    void deallocate(void* ptr, size_t size) { memory_pool_.deallocate(ptr, size); }
    
    // Cache interface
    std::shared_ptr<class Tensor> get_cached_tensor(const std::string& key) {
        return tensor_cache_.get(key);
    }
    void cache_tensor(const std::string& key, std::shared_ptr<class Tensor> tensor) {
        tensor_cache_.put(key, tensor);
    }
    
    // Computation graph management
    void register_tensor(std::shared_ptr<class Tensor> tensor);
    void clear_computation_graph();
    void cleanup_dead_references();
    
    // Memory cleanup operations
    void clear_cache() { tensor_cache_.clear(); }
    void clear_unused_memory() { memory_pool_.clear_unused(); }
    void force_cleanup();
    
    // Statistics and monitoring
    void print_memory_stats() const;
    size_t get_memory_usage() const { return memory_pool_.get_total_in_use(); }
    size_t get_peak_memory() const { return memory_pool_.get_total_peak(); }
    
    // Configuration
    void set_cache_size(size_t max_size) { tensor_cache_.set_max_size(max_size); }
    
    // Disable copy constructor and assignment
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
};

/**
 * @brief RAII memory guard for automatic cleanup
 */
class MemoryGuard {
private:
    bool active_ = true;
    
public:
    MemoryGuard() = default;
    ~MemoryGuard() {
        if (active_) {
            MemoryManager::instance().clear_unused_memory();
            MemoryManager::instance().cleanup_dead_references();
        }
    }
    
    void release() { active_ = false; }
    
    // Disable copy constructor and assignment
    MemoryGuard(const MemoryGuard&) = delete;
    MemoryGuard& operator=(const MemoryGuard&) = delete;
};

/**
 * @brief Memory monitoring utilities
 */
class MemoryMonitor {
public:
    static size_t get_system_memory_usage();
    static size_t get_system_available_memory();
    static bool is_memory_pressure();
    static void log_memory_status(const std::string& context = "");
    
    // Threshold monitoring
    static void set_memory_threshold(double percentage); // 0.0 to 1.0
    static bool check_memory_threshold();
    
private:
    static double memory_threshold_;
};

} // namespace ops
