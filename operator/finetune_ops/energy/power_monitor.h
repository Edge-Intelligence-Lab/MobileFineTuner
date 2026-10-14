#pragma once

// Public, model-independent update throttling.

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ops {
namespace energy {

struct PowerConfig {
    int check_interval_steps = 0;  // <= 0 disables the sensor policy, not schedules.
    float battery_threshold = 20.0f;
    float freq_b_high = 2.0f;       // Target steps/second above battery threshold.
    float freq_b_low = 0.5f;
    bool enable_battery = true;
    float temp_threshold = 42.0f;
    float freq_t_high = 2.0f;       // Target steps/second at/below temperature threshold.
    float freq_t_low = 0.5f;
    bool enable_temp = true;
};

struct StepSleep {
    int64_t start_step = 0;  // Inclusive, zero-based optimizer update index.
    int64_t end_step = -1;  // Inclusive; -1 means unbounded.
    int sleep_ms = 0;
};

// Suggests delays only: does not sleep, poll sensors, change CPU clocks, or measure energy.
// Call from one training thread; applications provide their own platform readings.
class PowerMonitor {
public:
    explicit PowerMonitor(const PowerConfig& config = PowerConfig());

    // NaN marks a missing reading. Missing/disabled sources impose no constraint.
    void set_manual_readings(float battery_percent, float cpu_temp_c);
    // Sorted internally; malformed or overlapping ranges throw invalid_argument.
    void set_step_schedule(const std::vector<StepSleep>& schedule);
    static std::vector<StepSleep> parse_schedule(const std::string& spec);

    // Schedules specify an additional delay. Policy subtracts the completed update's
    // compute time from the target period (capped at 5000 ms). One-argument calls
    // retain the historical zero-compute-time convention.
    int suggest_sleep_ms(int64_t global_step, double compute_time_ms = 0.0);
    std::string debug_state() const;

private:
    PowerConfig config_;
    float battery_percent_ = std::numeric_limits<float>::quiet_NaN();
    float cpu_temp_c_ = std::numeric_limits<float>::quiet_NaN();
    std::vector<StepSleep> schedule_;
    double target_period_ms_ = 0.0;
    bool policy_initialized_ = false;
    int last_computed_sleep_ms_ = 0;

    void recompute_policy();
};

}  // namespace energy
}  // namespace ops
