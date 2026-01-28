/**
 * @file memory_governor.h
 */

#pragma once

#include <cstddef>
#include <string>
#include <functional>

namespace ops {
namespace memory {

struct MemoryBudget {
    size_t soft_limit_mb = 2048;
    size_t hard_limit_mb = 4096;
    size_t warning_threshold_mb = 1536;
};

enum class MemoryPressureLevel { NORMAL, WARNING, CRITICAL, EMERGENCY };

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
    using ReductionCallback = std::function<void(MemoryPressureLevel)>;
    ReductionCallback reduction_callback_;

public:
    explicit MemoryGovernor(const MemoryBudget& budget = MemoryBudget());
    void set_reduction_callback(ReductionCallback callback);
    MemoryStatus check_and_act();
    MemoryStatus get_status() const;
    void force_check();
    void print_report() const;
    size_t peak_rss() const { return peak_rss_mb_; }

private:
    size_t get_current_rss_mb() const;
    size_t get_current_footprint_mb() const;
    MemoryPressureLevel assess_pressure(size_t rss_mb) const;
};

} // namespace memory
} // namespace ops


