#include "power_monitor.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ops {
namespace energy {
namespace {

void require_frequency(float frequency) {
    if (!std::isfinite(frequency) || frequency <= 0.0f) {
        throw std::invalid_argument("PowerMonitor frequencies must be finite and positive");
    }
}

std::vector<StepSleep> validated_schedule(std::vector<StepSleep> schedule) {
    std::sort(schedule.begin(), schedule.end(), [](const StepSleep& a, const StepSleep& b) {
        return a.start_step < b.start_step;
    });
    for (size_t i = 0; i < schedule.size(); ++i) {
        const auto& range = schedule[i];
        if (range.start_step < 0 || range.end_step < -1 || range.sleep_ms < 0 ||
            (range.end_step != -1 && range.end_step < range.start_step)) {
            throw std::invalid_argument("Invalid PowerMonitor schedule range");
        }
        if (i && (schedule[i - 1].end_step == -1 ||
                  schedule[i - 1].end_step >= range.start_step)) {
            throw std::invalid_argument("Overlapping PowerMonitor schedule ranges");
        }
    }
    return schedule;
}

int64_t parse_nonnegative(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Schedule requires nonnegative integers");
    }
    try {
        return std::stoll(value);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("Schedule integer is out of range");
    }
}

}  // namespace

PowerMonitor::PowerMonitor(const PowerConfig& config) : config_(config) {
    if (config_.enable_battery) {
        require_frequency(config_.freq_b_high);
        require_frequency(config_.freq_b_low);
        if (!std::isfinite(config_.battery_threshold) || config_.battery_threshold < 0 ||
            config_.battery_threshold > 100) {
            throw std::invalid_argument("Battery threshold must be in [0, 100]");
        }
    }
    if (config_.enable_temp) {
        require_frequency(config_.freq_t_high);
        require_frequency(config_.freq_t_low);
        if (!std::isfinite(config_.temp_threshold)) {
            throw std::invalid_argument("Temperature threshold must be finite");
        }
    }
}

void PowerMonitor::set_manual_readings(float battery_percent, float cpu_temp_c) {
    if ((!std::isnan(battery_percent) && (!std::isfinite(battery_percent) ||
         battery_percent < 0 || battery_percent > 100)) ||
        (!std::isnan(cpu_temp_c) && !std::isfinite(cpu_temp_c))) {
        throw std::invalid_argument("Invalid readings: use NaN for unavailable sensors");
    }
    battery_percent_ = battery_percent;
    cpu_temp_c_ = cpu_temp_c;
    // The next scheduled policy check consumes these readings.
}

void PowerMonitor::set_step_schedule(const std::vector<StepSleep>& schedule) {
    schedule_ = validated_schedule(schedule);
}

std::vector<StepSleep> PowerMonitor::parse_schedule(const std::string& spec) {
    std::vector<StepSleep> schedule;
    if (spec.empty()) return schedule;
    size_t pos = 0;
    while (pos < spec.size()) {
        const size_t comma = spec.find(',', pos);
        const std::string token = spec.substr(pos, comma == std::string::npos ? comma : comma - pos);
        const size_t colon = token.find(':');
        const size_t dash = token.find('-');
        if (dash == std::string::npos || colon == std::string::npos || dash >= colon) {
            throw std::invalid_argument("Schedule format: start-end:ms,start-:ms");
        }
        StepSleep range;
        range.start_step = parse_nonnegative(token.substr(0, dash));
        range.end_step = colon == dash + 1 ? -1 :
            parse_nonnegative(token.substr(dash + 1, colon - dash - 1));
        const int64_t delay = parse_nonnegative(token.substr(colon + 1));
        if (delay > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Schedule sleep_ms exceeds INT_MAX");
        }
        range.sleep_ms = static_cast<int>(delay);
        schedule.push_back(range);
        if (comma == std::string::npos) break;
        pos = comma + 1;
        if (pos == spec.size()) throw std::invalid_argument("Empty schedule entry");
    }
    return validated_schedule(schedule);
}

void PowerMonitor::recompute_policy() {
    double frequency = std::numeric_limits<double>::infinity();
    if (config_.enable_battery && std::isfinite(battery_percent_)) {
        frequency = battery_percent_ < config_.battery_threshold ?
            config_.freq_b_low : config_.freq_b_high;
    }
    if (config_.enable_temp && std::isfinite(cpu_temp_c_)) {
        frequency = std::min(frequency, static_cast<double>(cpu_temp_c_ > config_.temp_threshold ?
            config_.freq_t_low : config_.freq_t_high));
    }
    target_period_ms_ = std::isfinite(frequency) ? std::min(5000.0, 1000.0 / frequency) : 0.0;
    policy_initialized_ = true;
}

int PowerMonitor::suggest_sleep_ms(int64_t global_step, double compute_time_ms) {
    if (global_step < 0 || !std::isfinite(compute_time_ms) || compute_time_ms < 0) {
        throw std::invalid_argument("Step and finite compute_time_ms must be nonnegative");
    }
    // Refresh even while an explicit schedule overrides output, so a schedule
    // ending between check steps does not reveal stale pre-schedule readings.
    if (config_.check_interval_steps > 0 && (!policy_initialized_ ||
        global_step % config_.check_interval_steps == 0)) {
        recompute_policy();
    }
    for (const auto& range : schedule_) {
        if (global_step < range.start_step) break;
        if (range.end_step == -1 || global_step <= range.end_step) {
            last_computed_sleep_ms_ = range.sleep_ms;
            return last_computed_sleep_ms_;
        }
    }
    last_computed_sleep_ms_ = config_.check_interval_steps <= 0 ? 0 :
        static_cast<int>(std::ceil(std::max(0.0, target_period_ms_ - compute_time_ms)));
    return last_computed_sleep_ms_;
}

std::string PowerMonitor::debug_state() const {
    std::ostringstream out;
    out << "PowerMonitor{battery=";
    if (std::isfinite(battery_percent_)) out << battery_percent_ << '%';
    else out << "unavailable";
    out << ", temp=";
    if (std::isfinite(cpu_temp_c_)) out << cpu_temp_c_ << 'C';
    else out << "unavailable";
    out << ", last_sleep_ms=" << last_computed_sleep_ms_ << '}';
    return out.str();
}

}  // namespace energy
}  // namespace ops
