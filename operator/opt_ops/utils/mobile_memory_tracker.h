/**
 * @file mobile_memory_tracker.h
 * [English documentation available]
 * 
 * ：
 * [English documentation available]
 * [English documentation available]
 * [English documentation available]
 * [English documentation available]
 */

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <mutex>
#include <cstdio>
#include <cmath>

#ifdef __APPLE__
#include <mach/mach.h>
#include <malloc/malloc.h>
#elif defined(__ANDROID__)
#include <malloc.h>
#elif defined(__linux__)
#include <malloc.h>
#endif

namespace mobile_training {

struct MemorySnapshot {
    int epoch = 0;
    int step = 0;
    size_t rss_bytes = 0;               // current RSS
    size_t peak_rss_bytes = 0;          // historical peak RSS
    size_t delta_rss_bytes = 0;         // change relative to previous step
    float gradient_norm = 0.0f;         // gradient norm
    float loss_value = 0.0f;            // 
    long step_time_ms = 0;              // step
    size_t malloc_allocated = 0;        // allocator statistics (optional)
    
    std::chrono::system_clock::time_point timestamp;
};

class MobileMemoryTracker {
private:
    std::vector<MemorySnapshot> history_;
    std::ofstream csv_file_;
    std::ofstream json_file_;
    std::string output_dir_;
    std::mutex tracker_mutex_;
    
    size_t peak_rss_ever_ = 0;
    size_t baseline_rss_ = 0;
    size_t prev_rss_ = 0;
    bool leak_detected_ = false;
    int consecutive_increases_ = 0;
    
public:
    explicit MobileMemoryTracker(const std::string& output_dir = "training_logs");
    ~MobileMemoryTracker();
    
    // recordtrainingstep
    void record_step(int epoch, int step, float loss, float grad_norm, long step_time_ms);
    
    // [English documentation available]
    static size_t get_current_rss();
    
    // [English documentation available]
    bool detect_memory_leak() const;
    
    // getmemory（MB/step）
    double get_memory_growth_rate() const;
    
    // generatereport
    void generate_report(const std::string& filepath);
    
    // getmemorystatistics
    struct MemorySummary {
        size_t initial_rss_mb;
        size_t current_rss_mb;
        size_t peak_rss_mb;
        double avg_growth_rate_mb_per_step;
        bool leak_detected;
        int total_steps;
    };
    MemorySummary get_summary() const;
    
    // [English documentation available]
    static void force_memory_release();
    
private:
    void write_csv_header();
    void flush_all();
};

// Android memoryoptimization
#ifdef __ANDROID__
class AndroidMemoryHelper {
public:
    // [English documentation available]
    static void configure_malloc_aggressive();
    
    // [English documentation available]
    static void trim_heap();
    
    // [English documentation available]
    static bool is_low_memory();
};
#endif

// iOS memoryoptimization
#ifdef __APPLE__
class iOSMemoryHelper {
public:
    // [English documentation available]
    static void release_memory_pressure();
    
    // [English documentation available]
    static bool is_app_in_background();
    
    // [English documentation available]
    static void notify_memory_warning();
};
#endif

} // namespace mobile_training

