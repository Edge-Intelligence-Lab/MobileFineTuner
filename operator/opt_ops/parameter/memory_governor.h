/**
 * @file memory_governor.h
 * @brief 内存管控器 - 硬上限与自动降配
 * 
 * 核心功能：
 * 1. 实时监控 RSS/Footprint
 * 2. 超过阈值时自动降配（减 block_size、增 checkpoint 等）
 * 3. 达硬上限时强制早停
 * 4. 提供详细的内存报告
 */

#pragma once

#include <cstddef>
#include <string>
#include <functional>

namespace ops {
namespace memory {

struct MemoryBudget {
    size_t soft_limit_mb = 2048;  // 2GB 软限制：触发降配
    size_t hard_limit_mb = 4096;  // 4GB 硬限制：强制停止
    size_t warning_threshold_mb = 1536;  // 1.5GB 警告阈值
};

enum class MemoryPressureLevel {
    NORMAL,      // < warning_threshold
    WARNING,     // >= warning_threshold
    CRITICAL,    // >= soft_limit
    EMERGENCY    // >= hard_limit
};

struct MemoryStatus {
    size_t rss_mb = 0;
    size_t footprint_mb = 0;
    size_t vsz_mb = 0;
    MemoryPressureLevel pressure = MemoryPressureLevel::NORMAL;
    bool should_reduce_config = false;
    bool should_stop = false;
};

class MemoryGovernor {
private:
    MemoryBudget budget_;
    size_t peak_rss_mb_ = 0;
    size_t num_warnings_ = 0;
    size_t num_reductions_ = 0;
    
    // 降配回调
    using ReductionCallback = std::function<void(MemoryPressureLevel)>;
    ReductionCallback reduction_callback_;
    
public:
    explicit MemoryGovernor(const MemoryBudget& budget = MemoryBudget());
    
    // 设置降配回调
    void set_reduction_callback(ReductionCallback callback);
    
    // 监控检查点（每步调用）
    MemoryStatus check_and_act();
    
    // 获取当前内存状态
    MemoryStatus get_status() const;
    
    // 强制检查（可用于调试）
    void force_check();
    
    // 统计
    void print_report() const;
    size_t peak_rss() const { return peak_rss_mb_; }
    
private:
    size_t get_current_rss_mb() const;
    size_t get_current_footprint_mb() const;
    MemoryPressureLevel assess_pressure(size_t rss_mb) const;
};

} // namespace memory
} // namespace ops

