#pragma once

#include <chrono>
#include <cstdint>
#include <sys/resource.h>
#ifdef __APPLE__
#include <libproc.h>
#include <unistd.h>
#endif

namespace measurement {
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
inline double unix_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
struct Usage {
    bool io_available = false;
    uint64_t os_read_bytes = 0, os_write_bytes = 0;
    uint64_t rss_bytes = 0, footprint_bytes = 0, peak_so_far_bytes = 0;
};
inline Usage usage() {
    Usage out;
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
#ifdef __APPLE__
        out.peak_so_far_bytes = ru.ru_maxrss;
#else
        out.peak_so_far_bytes = ru.ru_maxrss * 1024ull;
#endif
    }
#ifdef __APPLE__
    rusage_info_v2 info{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2, reinterpret_cast<rusage_info_t*>(&info)) == 0) {
        out.io_available = true;
        out.os_read_bytes = info.ri_diskio_bytesread;
        out.os_write_bytes = info.ri_diskio_byteswritten;
        out.rss_bytes = info.ri_resident_size;
        out.footprint_bytes = info.ri_phys_footprint;
    }
#endif
    return out;
}
} // namespace measurement
