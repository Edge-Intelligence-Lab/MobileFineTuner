#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

const std::vector<std::string> kKeys = {
    "VmPeak",
    "VmSize",
    "VmHWM",
    "VmRSS",
    "RssAnon",
    "RssFile",
    "RssShmem",
    "VmData",
    "VmStk",
    "VmExe",
    "VmLib",
    "VmSwap",
};

struct Args {
    int pid = -1;
    double interval_ms = 5.0;
    std::string csv_path;
    std::string summary_path;
};

bool file_exists(const std::string& path) {
    std::ifstream in(path);
    return in.good();
}

std::string proc_status_path(int pid) {
    return "/proc/" + std::to_string(pid) + "/status";
}

bool read_status_kb(int pid, std::unordered_map<std::string, int64_t>& values) {
    for (const auto& key : kKeys) values[key] = 0;
    std::ifstream in(proc_status_path(pid));
    if (!in) return false;

    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = line.substr(0, colon);
        if (values.find(key) == values.end()) continue;

        const char* cursor = line.c_str() + colon + 1;
        while (*cursor == ' ' || *cursor == '\t') ++cursor;
        char* end = nullptr;
        const long parsed = std::strtol(cursor, &end, 10);
        if (end != cursor) values[key] = parsed;
    }
    return true;
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

void write_summary(
    const Args& args,
    int64_t samples,
    double elapsed_s,
    const std::unordered_map<std::string, int64_t>& peaks,
    const std::string& error
) {
    std::ofstream out(args.summary_path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"csv\": \"" << args.csv_path << "\",\n";
    out << "  \"effective_interval_ms\": ";
    if (samples > 0) {
        out << (elapsed_s * 1000.0 / static_cast<double>(samples));
    } else {
        out << "null";
    }
    out << ",\n";
    out << "  \"elapsed_s\": " << elapsed_s << ",\n";
    if (!error.empty()) {
        out << "  \"error\": \"" << error << "\",\n";
    }
    out << "  \"interval_ms_requested\": " << args.interval_ms << ",\n";
    out << "  \"note\": \"VmRSS is resident set size; VmHWM is the kernel high-water RSS for the process.\",\n";
    out << "  \"peak_kb\": {\n";
    for (size_t i = 0; i < kKeys.size(); ++i) {
        const auto& key = kKeys[i];
        out << "    \"" << key << "\": " << peaks.at(key);
        out << (i + 1 == kKeys.size() ? "\n" : ",\n");
    }
    out << "  },\n";
    out << "  \"peak_mb\": {\n";
    out << std::setprecision(9);
    for (size_t i = 0; i < kKeys.size(); ++i) {
        const auto& key = kKeys[i];
        out << "    \"" << key << "\": " << (static_cast<double>(peaks.at(key)) / 1024.0);
        out << (i + 1 == kKeys.size() ? "\n" : ",\n");
    }
    out << "  },\n";
    out << "  \"pid\": " << args.pid << ",\n";
    out << "  \"sampler\": \"native_procfs_status\",\n";
    out << "  \"samples\": " << samples << "\n";
    out << "}\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (key == "--pid") {
            args.pid = std::stoi(require_value(key));
        } else if (key == "--interval_ms") {
            args.interval_ms = std::stod(require_value(key));
        } else if (key == "--csv") {
            args.csv_path = require_value(key);
        } else if (key == "--summary") {
            args.summary_path = require_value(key);
        } else {
            std::cerr << "unknown argument: " << key << "\n";
            std::exit(2);
        }
    }
    if (args.pid <= 0 || args.csv_path.empty() || args.summary_path.empty()) {
        std::cerr << "usage: proc_mem_monitor --pid PID --interval_ms 5 --csv PATH --summary PATH\n";
        std::exit(2);
    }
    if (args.interval_ms < 1.0) args.interval_ms = 1.0;
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    std::ofstream csv(args.csv_path);
    if (!csv) {
        std::cerr << "failed to open csv: " << args.csv_path << "\n";
        return 1;
    }

    csv << "sample,t_ms";
    for (const auto& key : kKeys) csv << "," << key << "_kB";
    csv << "\n";

    std::unordered_map<std::string, int64_t> values;
    std::unordered_map<std::string, int64_t> peaks;
    for (const auto& key : kKeys) peaks[key] = 0;

    int64_t samples = 0;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> interval(args.interval_ms);

    while (file_exists(proc_status_path(args.pid))) {
        if (!read_status_kb(args.pid, values)) break;
        ++samples;

        for (const auto& key : kKeys) {
            if (values[key] > peaks[key]) peaks[key] = values[key];
        }

        csv << samples << "," << std::fixed << std::setprecision(3) << elapsed_ms(start);
        for (const auto& key : kKeys) csv << "," << values[key];
        csv << "\n";
        if ((samples % 200) == 0) csv.flush();
        std::this_thread::sleep_for(interval);
    }

    csv.flush();
    const double elapsed_s = elapsed_ms(start) / 1000.0;
    if (samples == 0) error = "no samples captured";
    write_summary(args, samples, elapsed_s, peaks, error);
    return samples > 0 ? 0 : 1;
}
