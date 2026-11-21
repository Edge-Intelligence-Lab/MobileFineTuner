#include "phase_probe.h"
#include <fstream>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <libproc.h>
#endif

namespace ops::resume {
IoCounts& logical_io() { static thread_local IoCounts counts; return counts; }
namespace {
uint64_t rss() {
#ifdef __APPLE__
    mach_task_basic_info info{}; mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
    if(task_info(mach_task_self(),MACH_TASK_BASIC_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS)
        return info.resident_size;
#else
    std::ifstream in("/proc/self/statm"); uint64_t total=0,resident=0;
    if(in>>total>>resident) return resident*static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
#endif
    return 0;
}
uint64_t hwm() {
    rusage r{}; if(getrusage(RUSAGE_SELF,&r)!=0) return 0;
#ifdef __APPLE__
    return r.ru_maxrss;
#else
    return static_cast<uint64_t>(r.ru_maxrss)*1024;
#endif
}
nlohmann::json os_io() {
#ifdef __APPLE__
    rusage_info_v4 info{};
    if(proc_pid_rusage(getpid(),RUSAGE_INFO_V4,reinterpret_cast<rusage_info_t*>(&info))==0)
        return {{"read_bytes",info.ri_diskio_bytesread},{"write_bytes",info.ri_diskio_byteswritten}};
#else
    std::ifstream in("/proc/self/io"); std::string key; uint64_t value;
    nlohmann::json j=nlohmann::json::object();
    while(in>>key>>value) {
        if(key=="read_bytes:") j["read_bytes"]=value;
        if(key=="write_bytes:") j["write_bytes"]=value;
    }
    if(j.size()==2) return j;
#endif
    return nullptr;
}
}
PhaseProbe::PhaseProbe(): started_(std::chrono::steady_clock::now()),start_io_(logical_io()),
    os_start_(os_io()),rss_start_(rss()),hwm_start_(hwm()) {
    record_sample(rss_start_);
    sampler_=std::thread([this]{
        std::unique_lock<std::mutex> lock(mutex_);
        while(!wake_.wait_for(lock,std::chrono::milliseconds(2),[this]{return stopped_;})) {
            lock.unlock(); sample(); lock.lock();
        }
    });
}
void PhaseProbe::sample() {
    record_sample(rss());
}
void PhaseProbe::record_sample(uint64_t r) {
    auto old=peak_.load();
    while(r>old && !peak_.compare_exchange_weak(old,r)) {}
    ++samples_;
}
void PhaseProbe::stop() {
    { std::lock_guard<std::mutex> lock(mutex_); stopped_=true; }
    wake_.notify_all(); if(sampler_.joinable()) sampler_.join();
}
PhaseProbe::~PhaseProbe() { stop(); }
nlohmann::json PhaseProbe::finish() {
    if(!result_.is_null()) return result_;
    stop(); const auto now=logical_io(); const auto disk=os_io();
    const auto rss_end=rss();
    record_sample(rss_end);
    auto available=[](uint64_t x)->nlohmann::json{return x?nlohmann::json(x):nlohmann::json(nullptr);};
    result_={{"wall_sec",std::chrono::duration<double>(std::chrono::steady_clock::now()-started_).count()},
             {"logical_read_bytes",now.read_bytes-start_io_.read_bytes},
             {"logical_write_bytes",now.write_bytes-start_io_.write_bytes},
             {"read_calls",now.reads-start_io_.reads},{"write_calls",now.writes-start_io_.writes},
             {"fsync_calls",now.fsyncs-start_io_.fsyncs},
             {"rss_start_bytes",available(rss_start_)},{"rss_end_bytes",available(rss_end)},
             {"peak_sampled_rss_bytes",available(peak_.load())},{"rss_sample_count",samples_.load()},
             {"sample_interval_ms",2},{"process_hwm_start_bytes",available(hwm_start_)},
             {"process_hwm_end_bytes",available(hwm())},{"os_io_bytes",nullptr}};
    if(!disk.is_null()&&!os_start_.is_null()) {
        nlohmann::json delta=nlohmann::json::object();
        for(const auto* k:{"read_bytes","write_bytes"}) {
            auto a=os_start_.at(k).get<uint64_t>(),b=disk.at(k).get<uint64_t>();
            delta[k]=b>=a?nlohmann::json(b-a):nlohmann::json(nullptr);
        }
        result_["os_io_bytes"]=delta;
    }
    return result_;
}
}
