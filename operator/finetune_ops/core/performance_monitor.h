/**
 * @file performance_monitor.h
 * @brief Performance and memory monitor (similar to PyTorch profiler)
 */

#pragma once

#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace ops {

/**
 * @brief Memory snapshot (similar to torch.cuda.memory_summary)
 */
struct MemorySnapshot {
    size_t allocated_mb = 0;      // Allocated (MB)
    size_t in_use_mb = 0;         // In use (MB)
    size_t peak_mb = 0;           // Peak (MB)
    size_t system_rss_mb = 0;     // Process RSS (MB)
    size_t system_available_mb = 0;  // System available (MB)
    
    void print() const {
        std::cout << "\n📊 Memory Snapshot:" << std::endl;
        std::cout << "  Allocated:  " << std::setw(8) << allocated_mb << " MB" << std::endl;
        std::cout << "  In use:     " << std::setw(8) << in_use_mb << " MB" << std::endl;
        std::cout << "  Peak:       " << std::setw(8) << peak_mb << " MB" << std::endl;
        std::cout << "  System RSS: " << std::setw(8) << system_rss_mb << " MB" << std::endl;
        std::cout << "  Available:  " << std::setw(8) << system_available_mb << " MB" << std::endl;
    }
};

/**
 * @brief Performance monitor (RAII)
 * 
 * Usage:
 * {
 *     PerformanceMonitor mon("forward pass");
 *     // ... code ...
 * } // automatically prints duration and memory delta
 */
class PerformanceMonitor {
public:
    explicit PerformanceMonitor(const std::string& name, bool print_memory = true);
    ~PerformanceMonitor();
    
    void checkpoint(const std::string& label);
    MemorySnapshot get_memory_snapshot() const;
    
private:
    std::string name_;
    bool print_memory_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    MemorySnapshot start_memory_;
};

/**
 * @brief Get current memory snapshot
 */
MemorySnapshot get_current_memory_snapshot();

/**
 * @brief Print memory usage tips (similar to PyTorch hints)
 */
void print_memory_optimization_tips(const MemorySnapshot& snapshot);

} // namespace ops

