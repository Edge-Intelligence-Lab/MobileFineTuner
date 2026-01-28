/**
 * @file arena_allocator.cpp
 * @brief 分区内存管理系统实现
 */

#include "arena_allocator.h"
#include "../core/logger.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#ifdef __APPLE__
#include <sys/mman.h>
#include <unistd.h>
#elif defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ops {
namespace memory {

// ============================================================================
// StepScratchArena 实现
// ============================================================================

StepScratchArena::StepScratchArena(size_t capacity_mb) 
    : capacity_(capacity_mb * 1024 * 1024), offset_(0), peak_usage_(0), 
      num_allocations_(0), num_resets_(0) {
    
    #if defined(__APPLE__) || defined(__linux__)
    // 使用 mmap 预留地址空间（MAP_ANON + MAP_PRIVATE）
    base_ptr_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ptr_ == MAP_FAILED) {
        base_ptr_ = nullptr;
        throw std::bad_alloc();
    }
    
    // madvise 告知内核使用模式
    #ifdef __APPLE__
    madvise(base_ptr_, capacity_, MADV_SEQUENTIAL);  // 顺序访问
    #endif
    
    #else
    // Windows 或其他平台：直接 malloc
    base_ptr_ = std::malloc(capacity_);
    if (!base_ptr_) {
        throw std::bad_alloc();
    }
    #endif
    
    // quiet log: StepScratchArena initialized
}

StepScratchArena::~StepScratchArena() {
    if (base_ptr_) {
        #if defined(__APPLE__) || defined(__linux__)
        munmap(base_ptr_, capacity_);
        #else
        std::free(base_ptr_);
        #endif
    }
}

void* StepScratchArena::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    // 对齐
    size_t aligned_offset = (offset_ + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
    if (aligned_offset + size > capacity_) {
        // Arena 用尽 - 这是严重问题，说明预算不够
        OPS_LOG_ERROR_F("StepScratchArena exhausted: need %zu MB, used %zu MB / %zu MB",
                       size / (1024 * 1024), aligned_offset / (1024 * 1024), 
                       capacity_ / (1024 * 1024));
        throw std::bad_alloc();
    }
    
    void* ptr = static_cast<char*>(base_ptr_) + aligned_offset;
    offset_ = aligned_offset + size;
    num_allocations_++;
    
    peak_usage_ = std::max(peak_usage_, offset_);
    
    // 零初始化
    std::memset(ptr, 0, size);
    
    return ptr;
}

void StepScratchArena::reset() {
    #ifdef __APPLE__
    // macOS强制释放：MADV_FREE太懒惰，改用MADV_DONTNEED立即回收物理页
    // 这会导致下次访问缺页，但能确保物理足迹不累积
    if (offset_ > 0) {
        madvise(base_ptr_, offset_, MADV_DONTNEED);
    }
    #elif defined(__linux__)
    // Linux: MADV_DONTNEED 立即释放物理页面
    if (offset_ > 0) {
        madvise(base_ptr_, offset_, MADV_DONTNEED);
    }
    #endif
    
    offset_ = 0;
    num_resets_++;
}

void StepScratchArena::recreate() {
    // 🔥 分代Arena：完全重建，重置虚拟地址空间
    // 这是唯一能让macOS物理足迹真正下降的方法
    
    #if defined(__APPLE__) || defined(__linux__)
    // 1. munmap释放虚拟地址空间
    if (base_ptr_) {
        munmap(base_ptr_, capacity_);
        base_ptr_ = nullptr;
    }
    
    // 2. 重新mmap分配新的虚拟地址空间
    base_ptr_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ptr_ == MAP_FAILED) {
        base_ptr_ = nullptr;
        OPS_LOG_ERROR("Arena recreate failed: mmap failed");
        throw std::bad_alloc();
    }
    
    #ifdef __APPLE__
    madvise(base_ptr_, capacity_, MADV_SEQUENTIAL);
    #endif
    
    #else
    // Windows或其他：重新malloc
    if (base_ptr_) {
        std::free(base_ptr_);
    }
    base_ptr_ = std::malloc(capacity_);
    if (!base_ptr_) {
        throw std::bad_alloc();
    }
    #endif
    
    // 3. 重置状态
    offset_ = 0;
    peak_usage_ = 0;
    num_allocations_ = 0;
    // num_resets_不重置，用于统计
    
    // 静默重建（避免日志噪音）
}

void StepScratchArena::print_stats() const {
    std::cout << "StepScratchArena Stats:\n";
    std::cout << "  Capacity: " << capacity_ / (1024 * 1024) << " MB\n";
    std::cout << "  Current usage: " << offset_ / (1024 * 1024) << " MB\n";
    std::cout << "  Peak usage: " << peak_usage_ / (1024 * 1024) << " MB\n";
    std::cout << "  Utilization: " << (100.0 * peak_usage_ / capacity_) << "%\n";
    std::cout << "  Total allocations: " << num_allocations_ << "\n";
    std::cout << "  Total resets: " << num_resets_ << "\n";
}

// ============================================================================
// StaticWeightArena 实现
// ============================================================================

StaticWeightArena::~StaticWeightArena() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& block : blocks_) {
        if (block.ptr) {
            #if defined(__APPLE__) || defined(__linux__)
            munmap(block.ptr, block.size);
            #else
            std::free(block.ptr);
            #endif
        }
    }
}

void* StaticWeightArena::allocate_static(size_t size, const std::string& name) {
    if (size == 0) return nullptr;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    void* ptr = nullptr;
    
    #if defined(__APPLE__) || defined(__linux__)
    // 使用 mmap 分配（可以设置只读保护）
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        OPS_LOG_ERROR_F("Failed to allocate static weight: %zu MB", size / (1024 * 1024));
        return nullptr;
    }
    
    // 告知内核：随机访问模式
    madvise(ptr, size, MADV_RANDOM);
    
    #else
    ptr = std::malloc(size);
    if (!ptr) {
        return nullptr;
    }
    #endif
    
    // 零初始化
    std::memset(ptr, 0, size);
    
    // 记录
    blocks_.push_back({ptr, size, name});
    total_size_ += size;
    
    // 静默分配（只在达到重要里程碑时输出，避免日志刷屏）
    static size_t last_logged_mb = 0;
    size_t current_mb = total_size_ / (1024 * 1024);
    if (current_mb >= last_logged_mb + 500) {  // 每增长500MB才输出一次
        OPS_LOG_INFO_F("StaticWeightArena total: %zu MB", current_mb);
        last_logged_mb = current_mb;
    }
    
    return ptr;
}

void StaticWeightArena::print_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "StaticWeightArena Stats:\n";
    std::cout << "  Total blocks: " << blocks_.size() << "\n";
    std::cout << "  Total size: " << total_size_ / (1024 * 1024) << " MB\n";
    
    for (const auto& block : blocks_) {
        std::cout << "    - " << block.name << ": " 
                  << block.size / (1024 * 1024) << " MB\n";
    }
}

// ============================================================================
// DirectLargeAllocator 实现
// ============================================================================

DirectLargeAllocator::~DirectLargeAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : allocations_) {
        if (pair.second.ptr) {
            #if defined(__APPLE__) || defined(__linux__)
            munmap(pair.second.ptr, pair.second.size);
            #else
            std::free(pair.second.ptr);
            #endif
        }
    }
}

void* DirectLargeAllocator::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    void* ptr = nullptr;
    
    #if defined(__APPLE__) || defined(__linux__)
    // 大张量用 mmap
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        OPS_LOG_ERROR_F("Failed to allocate large tensor: %zu MB", size / (1024 * 1024));
        return nullptr;
    }
    #else
    ptr = std::malloc(size);
    if (!ptr) {
        return nullptr;
    }
    #endif
    
    // 零初始化
    std::memset(ptr, 0, size);
    
    allocations_[ptr] = {ptr, size};
    total_allocated_ += size;
    num_allocations_++;
    
    // quiet log for DirectLarge allocations
    
    return ptr;
}

void DirectLargeAllocator::free(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        OPS_LOG_WARNING("Attempted to free unknown large pointer");
        return;
    }
    
    auto& block = it->second;
    
    #if defined(__APPLE__) || defined(__linux__)
    // 先 madvise 释放物理页面（保留地址空间片刻）
    #ifdef __APPLE__
    madvise(block.ptr, block.size, MADV_FREE);
    #elif defined(__linux__)
    madvise(block.ptr, block.size, MADV_DONTNEED);
    #endif
    
    // 然后 munmap
    munmap(block.ptr, block.size);
    #else
    std::free(block.ptr);
    #endif
    
    total_allocated_ -= block.size;
    allocations_.erase(it);
}

void DirectLargeAllocator::print_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "DirectLargeAllocator Stats:\n";
    std::cout << "  Active allocations: " << allocations_.size() << "\n";
    std::cout << "  Total allocated: " << total_allocated_ / (1024 * 1024) << " MB\n";
    std::cout << "  Total count: " << num_allocations_ << "\n";
}

// ============================================================================
// ArenaManager 实现
// ============================================================================

thread_local StepScratchArena* ArenaManager::current_step_arena_ = nullptr;

ArenaManager::ArenaManager() {
    static_arena_ = std::make_unique<StaticWeightArena>();
    large_allocator_ = std::make_unique<DirectLargeAllocator>();
    
    OPS_LOG_INFO("ArenaManager initialized (StaticWeight + DirectLarge)");
}

ArenaManager::~ArenaManager() = default;

ArenaManager& ArenaManager::instance() {
    static ArenaManager instance;
    return instance;
}

void ArenaManager::set_current_step_arena(StepScratchArena* arena) {
    current_step_arena_ = arena;
}

StepScratchArena* ArenaManager::get_current_step_arena() {
    return current_step_arena_;
}

void ArenaManager::clear_current_step_arena() {
    current_step_arena_ = nullptr;
}

void* ArenaManager::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    // 路由策略：
    // 1. 大张量（≥8MB）-> DirectLarge
    // 2. 当前在步内 -> StepScratchArena
    // 3. 否则 -> 回退到 malloc（或 MemoryPool）
    
    if (DirectLargeAllocator::is_large(size)) {
        return large_allocator_->allocate(size);
    }
    
    if (current_step_arena_) {
        try {
            return current_step_arena_->allocate(size);
        } catch (const std::bad_alloc&) {
            // Arena 用尽，回退到直配
            OPS_LOG_WARNING("StepArena exhausted, fallback to malloc");
            void* ptr = std::malloc(size);
            if (ptr) std::memset(ptr, 0, size);
            return ptr;
        }
    }
    
    // 默认回退：malloc
    void* ptr = std::malloc(size);
    if (ptr) {
        std::memset(ptr, 0, size);
    }
    return ptr;
}

void ArenaManager::free(void* ptr, size_t size) {
    if (!ptr) return;
    
    // 判断是否是大张量
    if (DirectLargeAllocator::is_large(size)) {
        large_allocator_->free(ptr);
        return;
    }
    
    // 判断是否在当前步 Arena 中（Arena 的内存在 reset 时统一回收，无需单独 free）
    // StepArena 的内存会在 reset() 时统一回收，这里不需要单独释放
    if (current_step_arena_) {
        // 简单启发式：假设在训练步内分配的都来自 Arena
        // Arena reset 时会统一处理，这里直接返回
        return;
    }
    
    // 否则：普通 malloc 分配，直接 free
    std::free(ptr);
}

void ArenaManager::print_all_stats() const {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Arena Memory Management Statistics\n";
    std::cout << std::string(60, '=') << "\n";
    
    static_arena_->print_stats();
    std::cout << "\n";
    
    large_allocator_->print_stats();
    std::cout << "\n";
    
    if (current_step_arena_) {
        current_step_arena_->print_stats();
    } else {
        std::cout << "StepScratchArena: Not active\n";
    }
    
    std::cout << std::string(60, '=') << "\n";
}

} // namespace memory
} // namespace ops

