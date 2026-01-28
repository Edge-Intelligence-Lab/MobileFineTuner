/**
 * @file power_monitor.cpp
 * @brief Implementation of the lightweight energy-aware scheduler.
 */

#include "power_monitor.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <thread>

#include <iostream>

namespace ops {
namespace energy {

PowerMonitor::PowerMonitor(const PowerConfig& cfg) : cfg_(cfg) {}

void PowerMonitor::set_manual_readings(float battery_percent, float cpu_temp_c) {
    battery_percent_ = battery_percent;
    cpu_temp_c_ = cpu_temp_c;
}

void PowerMonitor::set_step_schedule(const std::vector<StepSleep>& schedule) {
    schedule_ = schedule;
}

std::vector<StepSleep> PowerMonitor::parse_schedule(const std::string& spec) {
    // Accept tokens: "start-end:ms" or "start-:ms". Comma separated.
    std::vector<StepSleep> out;
    size_t pos = 0;
    while (pos < spec.size()) {
        // Find next comma
        size_t comma = spec.find(',', pos);
        std::string token = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = (comma == std::string::npos) ? spec.size() : comma + 1;
        if (token.empty()) continue;
        StepSleep ss;
        ss.sleep_ms = 0;

        size_t colon = token.find(':');
        if (colon == std::string::npos) continue;
        std::string range = token.substr(0, colon);
        std::string sleep_str = token.substr(colon + 1);
        try {
            ss.sleep_ms = std::max(0, std::stoi(sleep_str));
        } catch (...) { continue; }

        size_t dash = range.find('-');
        if (dash == std::string::npos) continue;
        std::string start_str = range.substr(0, dash);
        std::string end_str = range.substr(dash + 1);
        try {
            ss.start_step = std::stoll(start_str);
        } catch (...) { continue; }
        if (end_str.empty()) {
            ss.end_step = -1;
        } else {
            try { ss.end_step = std::stoll(end_str); } catch (...) { ss.end_step = -1; }
        }
        out.push_back(ss);
    }
    // Sort by start for deterministic behavior
    std::sort(out.begin(), out.end(), [](const StepSleep& a, const StepSleep& b) {
        return a.start_step < b.start_step;
    });
    return out;
}

int PowerMonitor::freq_to_sleep_ms(float freq_hz) {
    if (freq_hz <= 0.0f) return 0;
    float period_ms = 1000.0f / freq_hz;
    // Clamp to a reasonable upper bound to avoid pathological sleeps
    period_ms = std::min(period_ms, 5000.0f);
    return static_cast<int>(std::round(period_ms));
}

int PowerMonitor::recompute_from_policy() {
    float freq_b = cfg_.freq_b_high;
    if (cfg_.enable_battery && battery_percent_ < cfg_.battery_threshold) {
        freq_b = cfg_.freq_b_low;
    }

    float freq_t = cfg_.freq_t_high;
    if (cfg_.enable_temp && cpu_temp_c_ > cfg_.temp_threshold) {
        freq_t = cfg_.freq_t_low;
    }

    float target_freq = std::min(freq_b, freq_t);
    last_computed_sleep_ms_ = freq_to_sleep_ms(target_freq);
    return last_computed_sleep_ms_;
}

int PowerMonitor::suggest_sleep_ms(int64_t global_step) {
    // Manual schedule override
    for (const auto& ss : schedule_) {
        if (global_step < ss.start_step) break;
        if (ss.end_step < 0 || global_step <= ss.end_step) {
            // std::cout << "DEBUG: Schedule hit step=" << global_step << " start=" << ss.start_step << " end=" << ss.end_step << " sleep=" << ss.sleep_ms << "\n";
            return ss.sleep_ms;
        }
    }

    if (cfg_.check_interval_steps <= 0) {
        return last_computed_sleep_ms_;
    }

    if (global_step % cfg_.check_interval_steps == 0) {
        return recompute_from_policy();
    }
    return last_computed_sleep_ms_;
}

std::string PowerMonitor::debug_state() const {
    std::ostringstream oss;
    oss << "PowerMonitor{battery=" << battery_percent_ << "%, temp=" << cpu_temp_c_
        << "C, last_sleep_ms=" << last_computed_sleep_ms_ << "}";
    return oss.str();
}

}  // namespace energy
}  // namespace ops
