/**
 * @file power_monitor.h
 * @brief Lightweight energy-aware scheduler used to throttle fine-tuning steps.
 *
 * The design follows the MobileFineTuner idea: we monitor battery / temperature
 * (optionally mocked via manual inputs) every K steps, derive a target
 * computation frequency, and map it to a per-step sleep interval. A manual
 * step-sleep schedule can override sensor checks for deterministic throttling.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ops {
namespace energy {

struct PowerConfig {
    // Check interval. If <=0, monitoring is disabled (sleep suggestions are 0 unless schedule overrides).
    int check_interval_steps = 0;

    // Battery policy
    float battery_threshold = 20.0f;   // mu_b
    float freq_b_high = 2.0f;          // f_b^high (steps per second)
    float freq_b_low  = 0.5f;          // f_b^low
    bool enable_battery = true;

    // Temperature policy
    float temp_threshold = 42.0f;      // mu_t (Celsius)
    float freq_t_high = 2.0f;          // f_t^high (steps per second)
    float freq_t_low  = 0.5f;          // f_t^low
    bool enable_temp = true;
};

struct StepSleep {
    int64_t start_step = 0;  // inclusive
    int64_t end_step   = -1; // inclusive; -1 means infinity
    int sleep_ms = 0;
};

class PowerMonitor {
public:
    explicit PowerMonitor(const PowerConfig& cfg = PowerConfig());

    // Update manual readings (for platforms where real telemetry is unavailable).
    void set_manual_readings(float battery_percent, float cpu_temp_c);

    // Optional deterministic throttling.
    void set_step_schedule(const std::vector<StepSleep>& schedule);

    // Parse a schedule string like "0-99:300,100-199:150,200-:50".
    static std::vector<StepSleep> parse_schedule(const std::string& spec);

    // Return the sleep time (ms) to apply at a given global step.
    // This reads schedule first; otherwise, on check steps it recomputes from telemetry.
    int suggest_sleep_ms(int64_t global_step);

    std::string debug_state() const;

private:
    PowerConfig cfg_;
    float battery_percent_ = 100.0f;
    float cpu_temp_c_ = 30.0f;

    std::vector<StepSleep> schedule_;
    int last_computed_sleep_ms_ = 0;

    int recompute_from_policy();
    static int freq_to_sleep_ms(float freq_hz);
};

}  // namespace energy
}  // namespace ops
