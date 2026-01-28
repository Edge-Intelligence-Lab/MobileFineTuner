/**
 * @file mobile_memory_tracker.cpp
 * [English documentation available]
 */

#include "mobile_memory_tracker.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <filesystem>

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace mobile_training {

// ===============================
// MobileMemoryTracker Implementation
// ===============================

MobileMemoryTracker::MobileMemoryTracker(const std::string& output_dir) 
    : output_dir_(output_dir) {
    
    // [English documentation available]
    std::filesystem::create_directories(output_dir);
    
    // [English documentation available]
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t);
#else
    localtime_r(&time_t, &tm_now);
#endif
    
    std::ostringstream ts_stream;
    ts_stream << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
    std::string timestamp = ts_stream.str();
    
    // CSVfile
    std::string csv_path = output_dir + "/memory_trace_" + timestamp + ".csv";
    csv_file_.open(csv_path, std::ios::out | std::ios::trunc);
    if (csv_file_.is_open()) {
        write_csv_header();
        // quiet log: CSV path
    }
    
    // [English documentation available]
    std::string json_path = output_dir + "/memory_detail_" + timestamp + ".json";
    json_file_.open(json_path, std::ios::out | std::ios::trunc);
    if (json_file_.is_open()) {
        json_file_ << "{\n  \"memory_snapshots\": [\n";
        // quiet log: JSON path
    }
    
    // recordmemory
    baseline_rss_ = get_current_rss();
    prev_rss_ = baseline_rss_;
    peak_rss_ever_ = baseline_rss_;
    
    // quiet log: MemoryTracker init RSS
}

MobileMemoryTracker::~MobileMemoryTracker() {
    if (csv_file_.is_open()) {
        csv_file_.close();
    }
    
    if (json_file_.is_open()) {
        json_file_ << "\n  ]\n}\n";
        json_file_.close();
    }
}

void MobileMemoryTracker::record_step(int epoch, int step, float loss, float grad_norm, long step_time_ms) {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    MemorySnapshot snapshot;
    snapshot.epoch = epoch;
    snapshot.step = step;
    snapshot.loss_value = loss;
    snapshot.gradient_norm = grad_norm;
    snapshot.step_time_ms = step_time_ms;
    snapshot.timestamp = std::chrono::system_clock::now();
    
    // getcurrent RSS
    snapshot.rss_bytes = get_current_rss();
    snapshot.delta_rss_bytes = (snapshot.rss_bytes > prev_rss_) ? 
                                (snapshot.rss_bytes - prev_rss_) : 0;
    
    // [English documentation available]
    if (snapshot.rss_bytes > peak_rss_ever_) {
        peak_rss_ever_ = snapshot.rss_bytes;
    }
    snapshot.peak_rss_bytes = peak_rss_ever_;
    
    // [English documentation available]
    if (snapshot.rss_bytes > prev_rss_ + 10 * 1024 * 1024) { // growth exceeds 10MB
        consecutive_increases_++;
        if (consecutive_increases_ > 5) {
            leak_detected_ = true;
        }
    } else if (snapshot.rss_bytes < prev_rss_) {
        consecutive_increases_ = 0;
    }
    
    prev_rss_ = snapshot.rss_bytes;
    // 僅每 10 步保留一個快照，降低常駐記憶體與 I/O
    if (step % 10 == 0) {
        history_.push_back(snapshot);
    }
    
    // [English documentation available]
    if (csv_file_.is_open() && step % 10 == 0) {
        csv_file_ << epoch << ","
                  << step << ","
                  << std::fixed << std::setprecision(4) << loss << ","
                  << std::fixed << std::setprecision(6) << grad_norm << ","
                  << (snapshot.rss_bytes / 1024.0 / 1024.0) << ","
                  << (snapshot.peak_rss_bytes / 1024.0 / 1024.0) << ","
                  << (snapshot.delta_rss_bytes / 1024.0 / 1024.0) << ","
                  << step_time_ms << "\n";
        csv_file_.flush();
    }
    
    // [English documentation available]
    if (json_file_.is_open() && step % 10 == 0) {
        if (step > 0) json_file_ << ",\n";
        json_file_ << "    {\"epoch\":" << epoch 
                   << ", \"step\":" << step
                   << ", \"rss_mb\":" << std::fixed << std::setprecision(2) << (snapshot.rss_bytes / 1024.0 / 1024.0)
                   << ", \"delta_mb\":" << (snapshot.delta_rss_bytes / 1024.0 / 1024.0)
                   << ", \"loss\":" << std::setprecision(4) << loss
                   << ", \"grad_norm\":" << std::setprecision(6) << grad_norm
                   << "}";
        json_file_.flush();
    }
    
    // memorywarning
    // quiet leak warnings in console; still recorded in CSV/JSON
}

size_t MobileMemoryTracker::get_current_rss() {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kerr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                   (task_info_t)&info, &size);
    return (kerr == KERN_SUCCESS) ? info.resident_size : 0;
    
#elif defined(__linux__) || defined(__ANDROID__)
    // [English documentation available]
    long resident_pages = 0;
    FILE* fp = std::fopen("/proc/self/statm", "r");
    if (fp) {
        long total_pages = 0;
        if (std::fscanf(fp, "%ld %ld", &total_pages, &resident_pages) == 2) {
            std::fclose(fp);
            long page_size = sysconf(_SC_PAGESIZE);
            return static_cast<size_t>(resident_pages) * static_cast<size_t>(page_size);
        }
        std::fclose(fp);
    }
    
    // [English documentation available]
    fp = std::fopen("/proc/self/status", "r");
    if (fp) {
        char line[256];
        while (std::fgets(line, sizeof(line), fp)) {
            if (std::strncmp(line, "VmRSS:", 6) == 0) {
                long kb = 0;
                if (std::sscanf(line, "VmRSS:%ld", &kb) == 1) {
                    std::fclose(fp);
                    return static_cast<size_t>(kb) * 1024;
                }
            }
        }
        std::fclose(fp);
    }
    return 0;
    
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<size_t>(pmc.WorkingSetSize);
    }
    return 0;
#else
    return 0;
#endif
}

bool MobileMemoryTracker::detect_memory_leak() const {
    if (history_.size() < 20) return false;
    
    // [English documentation available]
    double growth_rate = get_memory_growth_rate();
    
    // [English documentation available]
    return (growth_rate > 5.0) || leak_detected_;
}

double MobileMemoryTracker::get_memory_growth_rate() const {
    if (history_.size() < 10) return 0.0;
    
    // [English documentation available]
    size_t window = std::min(size_t(50), history_.size());
    const auto& start_snapshot = history_[history_.size() - window];
    const auto& end_snapshot = history_.back();
    
    double rss_delta_mb = (end_snapshot.rss_bytes - start_snapshot.rss_bytes) / 1024.0 / 1024.0;
    int step_delta = end_snapshot.step - start_snapshot.step;
    
    return (step_delta > 0) ? (rss_delta_mb / step_delta) : 0.0;
}

void MobileMemoryTracker::generate_report(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    std::ofstream report(filepath);
    if (!report.is_open()) {
        std::cerr << "[MemoryTracker] [Output]report: " << filepath << std::endl;
        return;
    }
    
    auto summary = get_summary();
    
    report << "==================================================\n";
    report << "Mobile training memory diagnostic report\n";
    report << "==================================================\n\n";
    
    report << "📊 memoryusage\n";
    report << "   RSS: " << summary.initial_rss_mb << " MB\n";
    report << "  current RSS: " << summary.current_rss_mb << " MB\n";
    report << "   RSS: " << summary.peak_rss_mb << " MB\n";
    report << "  trainingstep: " << summary.total_steps << "\n";
    report << "  Average growth rate: " << std::fixed << std::setprecision(3) 
           << summary.avg_growth_rate_mb_per_step << " MB/step\n\n";
    
    if (summary.leak_detected) {
        report << "⚠️  Memory leak detected: Yes\n";
        report << "  Suggestion: Check reset/cleanup calls in each step\n\n";
    } else {
        report << "✅ memory: （memory）\n\n";
    }
    
    // [English documentation available]
    report << "📈 memoryusage（10step）:\n";
    report << "  Step | RSS (MB) | Delta (MB) | Loss | Grad Norm | Time (ms)\n";
    report << "  -----|----------|------------|------|-----------|----------\n";
    
    for (size_t i = 0; i < history_.size(); i += 10) {
        const auto& snap = history_[i];
        report << "  " << std::setw(4) << snap.step << " | "
               << std::setw(8) << std::fixed << std::setprecision(2) << (snap.rss_bytes / 1024.0 / 1024.0) << " | "
               << std::setw(10) << (snap.delta_rss_bytes / 1024.0 / 1024.0) << " | "
               << std::setw(4) << std::setprecision(2) << snap.loss_value << " | "
               << std::setw(9) << std::scientific << std::setprecision(2) << snap.gradient_norm << " | "
               << std::setw(8) << snap.step_time_ms << "\n";
    }
    
    report << "\n==================================================\n";
    report.close();
    
    std::cout << "[MemoryTracker] reportgenerate: " << filepath << std::endl;
}

MobileMemoryTracker::MemorySummary MobileMemoryTracker::get_summary() const {
    MemorySummary summary;
    
    if (history_.empty()) {
        return summary;
    }
    
    summary.initial_rss_mb = baseline_rss_ / 1024 / 1024;
    summary.current_rss_mb = history_.back().rss_bytes / 1024 / 1024;
    summary.peak_rss_mb = peak_rss_ever_ / 1024 / 1024;
    summary.total_steps = static_cast<int>(history_.size());
    summary.avg_growth_rate_mb_per_step = get_memory_growth_rate();
    summary.leak_detected = detect_memory_leak();
    
    return summary;
}

void MobileMemoryTracker::force_memory_release() {
#ifdef __APPLE__
    // [English documentation available]
    malloc_zone_pressure_relief(nullptr, 0);
    
#elif defined(__ANDROID__)
    // Android: force
    mallopt(M_PURGE, 0);
    
#elif defined(__linux__)
    // Linux: glibc 
    malloc_trim(0);
#endif
}

void MobileMemoryTracker::write_csv_header() {
    csv_file_ << "epoch,step,loss,grad_norm,rss_mb,peak_rss_mb,delta_mb,step_ms\n";
    csv_file_.flush();
}

void MobileMemoryTracker::flush_all() {
    if (csv_file_.is_open()) csv_file_.flush();
    if (json_file_.is_open()) json_file_.flush();
}

// ===============================
// Android Helper Implementation
// ===============================

#ifdef __ANDROID__
void AndroidMemoryHelper::configure_malloc_aggressive() {
    // [English documentation available]
    mallopt(M_TRIM_THRESHOLD, 128 * 1024);      // [Comment in English]
    mallopt(M_MMAP_THRESHOLD, 128 * 1024);      // 128KBmmap
    mallopt(M_MMAP_MAX, 512);                   // [Comment in English]
    mallopt(M_TOP_PAD, 0);                      // top pad
    
    std::cout << "[AndroidMemory] [Output]configuration[Output]" << std::endl;
}

void AndroidMemoryHelper::trim_heap() {
    mallopt(M_PURGE, 0);  // Android bionic 
    malloc_trim(0);       // standard glibc
}

bool AndroidMemoryHelper::is_low_memory() {
    // [English documentation available]
    FILE* fp = std::fopen("/proc/meminfo", "r");
    if (!fp) return false;
    
    long mem_available_kb = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::sscanf(line, "MemAvailable: %ld", &mem_available_kb) == 1) {
            break;
        }
    }
    std::fclose(fp);
    
    // [English documentation available]
    return (mem_available_kb < 500 * 1024);
}
#endif

// ===============================
// iOS Helper Implementation
// ===============================

#ifdef __APPLE__
void iOSMemoryHelper::release_memory_pressure() {
    malloc_zone_pressure_relief(nullptr, 0);
}

bool iOSMemoryHelper::is_app_in_background() {
    // [English documentation available]
    // [English documentation available]
    return false;
}

void iOSMemoryHelper::notify_memory_warning() {
    // [English documentation available]
    // [English documentation available]
    std::cout << "[iOS] memorywarning" << std::endl;
}
#endif

} // namespace mobile_training

