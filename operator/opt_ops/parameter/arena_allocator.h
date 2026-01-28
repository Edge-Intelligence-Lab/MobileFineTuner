/**
 * @file arena_allocator.h
 * @brief 分区内存管理系统 - 根治物理足迹线性增长
 * 
 * 核心思路：
 * 1. StepScratchArena: 每步开始reset，所有激活从此分配，步末一键回收
 * 2. StaticWeightArena: 静态权重一次映射只读，不参与缓存/trim
 * 3. DirectLargeAllocation: 大张量（≥8MB）直配+MADV_FREE，完全旁路
 * 
 * 目标：活动监视器 Memory/Footprint 不再随 step 线性增长
 */

#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>

#ifdef __APPLE__
#include <sys/mman.h>
#elif defined(__linux__)
#include <sys/mman.h>
#endif

namespace ops {
namespace memory {

// ============================================================================
// StepScratchArena - 步级暂存区（每步 reset）
// ============================================================================

class StepScratchArena {
public:
    void* base_ptr_ = nullptr;  // Public for ArenaManager access
    
private:
    size_t capacity_ = 0;
    size_t offset_ = 0;
    size_t peak_usage_ = 0;
    size_t num_allocations_ = 0;
    size_t num_resets_ = 0;
    
    static constexpr size_t ALIGNMENT = 64;
    
public:
    explicit StepScratchArena(size_t capacity_mb = 128);
    ~StepScratchArena();
    
    // 分配内存（对齐到 64 字节）
    void* allocate(size_t size);
    
    // 步结束：一键回收所有内存
    void reset();
    
    // 🔥 分代Arena：完全重建，重置虚拟地址空间（macOS物理足迹控制）
    void recreate();
    
    // 统计
    size_t current_usage() const { return offset_; }
    size_t peak_usage() const { return peak_usage_; }
    size_t capacity() const { return capacity_; }
    void print_stats() const;
    
    // 禁止拷贝
    StepScratchArena(const StepScratchArena&) = delete;
    StepScratchArena& operator=(const StepScratchArena&) = delete;
};

// ============================================================================
// StaticWeightArena - 静态权重区（只读，不参与缓存）
// ============================================================================

class StaticWeightArena {
private:
    struct WeightBlock {
        void* ptr = nullptr;
        size_t size = 0;
        std::string name;
    };
    
    std::vector<WeightBlock> blocks_;
    size_t total_size_ = 0;
    mutable std::mutex mutex_;
    
public:
    StaticWeightArena() = default;
    ~StaticWeightArena();
    
    // 分配静态权重（mmap 只读映射）
    void* allocate_static(size_t size, const std::string& name = "");
    
    // 统计
    size_t total_size() const { return total_size_; }
    void print_stats() const;
    
    // 禁止拷贝
    StaticWeightArena(const StaticWeightArena&) = delete;
    StaticWeightArena& operator=(const StaticWeightArena&) = delete;
};

// ============================================================================
// DirectLargeAllocator - 大张量直配（≥8MB，bypass cache）
// ============================================================================

class DirectLargeAllocator {
private:
    struct LargeBlock {
        void* ptr = nullptr;
        size_t size = 0;
    };
    
    std::unordered_map<void*, LargeBlock> allocations_;
    size_t total_allocated_ = 0;
    size_t num_allocations_ = 0;
    mutable std::mutex mutex_;
    
    static constexpr size_t LARGE_THRESHOLD = 16 * 1024 * 1024;  // 16MB（避免9MB的MLP权重走DirectLarge）
    
public:
    DirectLargeAllocator() = default;
    ~DirectLargeAllocator();
    
    // 判断是否应该走大张量直配
    static bool is_large(size_t size) { return size >= LARGE_THRESHOLD; }
    
    // 分配大张量（直接 mmap 或 malloc）
    void* allocate(size_t size);
    
    // 释放大张量（madvise + munmap）
    void free(void* ptr);
    
    // 统计
    size_t total_allocated() const { return total_allocated_; }
    void print_stats() const;
};

// ============================================================================
// ArenaManager - 统一管理器（线程本地 + 全局单例）
// ============================================================================

class ArenaManager {
private:
    // 全局单例
    std::unique_ptr<StaticWeightArena> static_arena_;
    std::unique_ptr<DirectLargeAllocator> large_allocator_;
    
    // 线程本地当前步 Arena（用指针，nullptr 表示不使用）
    static thread_local StepScratchArena* current_step_arena_;
    
    mutable std::mutex mutex_;
    
    ArenaManager();
    
public:
    ~ArenaManager();
    
    // 单例访问
    static ArenaManager& instance();
    
    // 步级 Arena 控制
    void set_current_step_arena(StepScratchArena* arena);
    StepScratchArena* get_current_step_arena();
    void clear_current_step_arena();
    
    // 静态权重区访问
    StaticWeightArena& static_weights() { return *static_arena_; }
    
    // 大张量直配访问
    DirectLargeAllocator& large_alloc() { return *large_allocator_; }
    
    // 统一分配入口（根据大小和当前上下文智能路由）
    void* allocate(size_t size);
    void free(void* ptr, size_t size);
    
    // 统计和诊断
    void print_all_stats() const;
    
    // 禁止拷贝
    ArenaManager(const ArenaManager&) = delete;
    ArenaManager& operator=(const ArenaManager&) = delete;
};

// ============================================================================
// RAII 辅助：自动管理步级 Arena 生命周期
// ============================================================================

class StepArenaGuard {
private:
    StepScratchArena arena_;
    
public:
    explicit StepArenaGuard(size_t capacity_mb = 128) 
        : arena_(capacity_mb) {
        ArenaManager::instance().set_current_step_arena(&arena_);
    }
    
    ~StepArenaGuard() {
        ArenaManager::instance().clear_current_step_arena();
        arena_.reset();  // 一键回收
    }
    
    StepScratchArena& get_arena() { return arena_; }
    
    // 🔥 分代Arena：主动重建，阻止macOS物理足迹累积
    void regenerate() {
        arena_.recreate();
    }
    
    // 禁止拷贝
    StepArenaGuard(const StepArenaGuard&) = delete;
    StepArenaGuard& operator=(const StepArenaGuard&) = delete;
};

} // namespace memory
} // namespace ops

