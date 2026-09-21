#include "mobile_finetuner/mobile_finetuner.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using ops::energy::PowerConfig;
using ops::energy::PowerMonitor;

static int checks = 0;

static void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}

template <typename F>
static void rejects(F action) {
    bool rejected = false;
    try { action(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Expected invalid_argument");
}

int main() {
    try {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        PowerMonitor off;
        off.set_manual_readings(1, 90);
        require(off.suggest_sleep_ms(0) == 0, "Default must not throttle");
        require(off.suggest_sleep_ms(42, 123) == 0, "Default remains off");
        off.set_step_schedule(PowerMonitor::parse_schedule("4-:30,0-1:20,3-3:0"));
        require(off.suggest_sleep_ms(0, 10000) == 20, "Schedule is additional sleep");
        require(off.suggest_sleep_ms(1) == 20, "Inclusive end");
        require(off.suggest_sleep_ms(2) == 0, "Gap with policy disabled");
        require(off.suggest_sleep_ms(3) == 0, "Explicit zero schedule");
        require(off.suggest_sleep_ms(4) == 30, "Unbounded schedule start");
        require(off.suggest_sleep_ms(std::numeric_limits<int64_t>::max()) == 30, "Unbounded end");
        off.set_step_schedule({});
        require(off.suggest_sleep_ms(5) == 0, "Clearing schedule does not leak cached delay");

        PowerConfig cfg;
        cfg.check_interval_steps = 3;
        PowerMonitor monitor(cfg);
        require(monitor.suggest_sleep_ms(1) == 0, "No invented sensor readings");
        require(monitor.debug_state().find("unavailable") != std::string::npos, "Missing reading diagnostic");
        monitor.set_manual_readings(80, 30);
        require(monitor.suggest_sleep_ms(3, 125) == 375, "Subtract compute time");
        require(monitor.suggest_sleep_ms(4, 600) == 0, "Slow update requires no additional idle");
        monitor.set_manual_readings(5, 70);
        require(monitor.suggest_sleep_ms(5, 200) == 300, "Policy only refreshes on check steps");
        require(monitor.suggest_sleep_ms(6, 200) == 1800, "Battery/temperature low-frequency policy");
        monitor.set_step_schedule({{7, 9, 0}});
        monitor.set_manual_readings(90, 30);
        require(monitor.suggest_sleep_ms(9) == 0, "Zero schedule overrides policy");
        require(monitor.suggest_sleep_ms(10) == 500, "Policy refreshed under schedule");
        monitor.set_manual_readings(nan, nan);
        require(monitor.suggest_sleep_ms(12) == 0, "Missing readings clear policy at next check");

        cfg.enable_temp = false;
        cfg.freq_t_high = 0.01f;
        cfg.freq_b_high = 4;
        PowerMonitor battery(cfg);
        battery.set_manual_readings(80, 99);
        require(battery.suggest_sleep_ms(2) == 250, "First non-multiple step initializes policy");
        cfg.enable_temp = true;
        cfg.enable_battery = false;
        cfg.freq_b_high = 0.01f;
        cfg.freq_t_high = 4;
        PowerMonitor temperature(cfg);
        temperature.set_manual_readings(1, 30);
        require(temperature.suggest_sleep_ms(0) == 250, "Disabled battery cannot constrain policy");
        cfg.enable_temp = false;
        PowerMonitor neither(cfg);
        neither.set_manual_readings(1, 99);
        require(neither.suggest_sleep_ms(0) == 0, "Both sources disabled");

        cfg = PowerConfig{};
        cfg.check_interval_steps = 1;
        PowerMonitor boundary(cfg);
        boundary.set_manual_readings(20, 42);
        require(boundary.suggest_sleep_ms(0) == 500, "Threshold equality uses high frequency");
        boundary.set_manual_readings(nan, 43);
        require(boundary.suggest_sleep_ms(1) == 2000, "Available temperature alone");
        boundary.set_manual_readings(19, nan);
        require(boundary.suggest_sleep_ms(2) == 2000, "Available battery alone");
        cfg.freq_b_low = std::numeric_limits<float>::min();
        PowerMonitor capped(cfg);
        capped.set_manual_readings(1, nan);
        require(capped.suggest_sleep_ms(0, 250) == 4750, "Cap target period before subtracting compute");
        require(capped.suggest_sleep_ms(1, 6000) == 0, "Nonnegative sleep after cap");

        require(PowerMonitor::parse_schedule("").empty(), "Empty schedule disables schedule");
        for (const auto* spec : {"bad", "0-2:x", "0-2:10x", "0-x:1", "3-1:2",
                                 "0-1:-1", "-1-2:3", "0-:1,1-:2", "0-1:2,1-2:3",
                                 "0-0:1,", ",0-:1", "0-:1,,2-:3", "0-:2147483648",
                                 "9223372036854775808-:1", "0--1:2"}) {
            rejects([&] { PowerMonitor::parse_schedule(spec); });
        }
        rejects([&] { off.set_step_schedule({{5, 4, 0}}); });
        rejects([&] { off.set_step_schedule({{0, -2, 0}}); });
        rejects([&] { off.set_step_schedule({{0, 1, -1}}); });
        rejects([&] { off.set_step_schedule({{0, 1, 0}, {1, 2, 0}}); });
        rejects([&] { off.suggest_sleep_ms(-1); });
        rejects([&] { off.suggest_sleep_ms(0, -1); });
        rejects([&] { off.suggest_sleep_ms(0, nan); });
        rejects([&] { off.suggest_sleep_ms(0, inf); });
        rejects([&] { off.set_manual_readings(101, 30); });
        rejects([&] { off.set_manual_readings(-1, 30); });
        rejects([&] { off.set_manual_readings(inf, 30); });
        rejects([&] { off.set_manual_readings(50, inf); });
        for (float bad : {0.0f, -1.0f, nan, inf}) {
            cfg = PowerConfig{};
            cfg.freq_b_high = bad;
            rejects([&] { PowerMonitor invalid(cfg); });
            cfg = PowerConfig{};
            cfg.freq_t_low = bad;
            rejects([&] { PowerMonitor invalid(cfg); });
        }
        cfg = PowerConfig{};
        cfg.battery_threshold = 101;
        rejects([&] { PowerMonitor invalid(cfg); });
        cfg = PowerConfig{};
        cfg.temp_threshold = nan;
        rejects([&] { PowerMonitor invalid(cfg); });
        std::cout << "PowerMonitor: " << checks << " checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
