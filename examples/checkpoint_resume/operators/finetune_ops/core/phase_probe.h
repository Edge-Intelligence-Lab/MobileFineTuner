#pragma once
#include "../../third_party/nlohmann/json.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace ops::resume {
struct IoCounts { uint64_t read_bytes=0, write_bytes=0, reads=0, writes=0, fsyncs=0; };
IoCounts& logical_io();
class PhaseProbe {
public:
    PhaseProbe();
    ~PhaseProbe();
    nlohmann::json finish();
private:
    void sample();
    void record_sample(uint64_t value);
    void stop();
    std::chrono::steady_clock::time_point started_;
    IoCounts start_io_;
    nlohmann::json os_start_, result_;
    uint64_t rss_start_=0, hwm_start_=0;
    std::atomic<uint64_t> peak_{0}, samples_{0};
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopped_=false;
    std::thread sampler_;
};
}
