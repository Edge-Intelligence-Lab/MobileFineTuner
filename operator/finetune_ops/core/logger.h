#ifndef OPERATORS_LOGGER_H
#define OPERATORS_LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <memory>

namespace ops {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

class Logger {
private:
    std::ofstream log_file_;
    LogLevel min_level_;
    bool console_output_;
    std::string log_dir_;

    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    std::string level_to_string(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

public:
    Logger(const std::string& log_dir = "logs", 
           const std::string& filename = "training.log",
           LogLevel min_level = LogLevel::INFO,
           bool console_output = true) 
        : min_level_(min_level), console_output_(console_output), log_dir_(log_dir) {
        
        // Create log directory
        std::string mkdir_cmd = "mkdir -p " + log_dir;
        system(mkdir_cmd.c_str());
        
        // Open log file
        std::string full_path = log_dir + "/" + filename;
        log_file_.open(full_path, std::ios::app);
        
        if (!log_file_.is_open()) {
            std::cerr << "Warning: Cannot open log file " << full_path << std::endl;
        }
    }

    ~Logger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

    void log(LogLevel level, const std::string& message) {
        if (level < min_level_) return;

        std::string timestamp = get_timestamp();
        std::string level_str = level_to_string(level);
        std::string foratted_msg = "[" + timestamp + "] [" + level_str + "] " + message;

        // Output to console
        if (console_output_) {
            if (level >= LogLevel::ERROR) {
                std::cerr << foratted_msg << std::endl;
            } else {
                std::cout << foratted_msg << std::endl;
            }
        }

        // Write to log file
        if (log_file_.is_open()) {
            log_file_ << foratted_msg << std::endl;
            log_file_.flush();
        }
    }

    void debug(const std::string& message) { log(LogLevel::DEBUG, message); }
    void info(const std::string& message) { log(LogLevel::INFO, message); }
    void warning(const std::string& message) { log(LogLevel::WARNING, message); }
    void error(const std::string& message) { log(LogLevel::ERROR, message); }

    // Formatted logging
    template<typename... Args>
    void info_f(const std::string& forat, Args... args) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), forat.c_str(), args...);
        info(std::string(buffer));
    }

    template<typename... Args>
    void error_f(const std::string& forat, Args... args) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), forat.c_str(), args...);
        error(std::string(buffer));
    }

    template<typename... Args>
    void debug_f(const std::string& forat, Args... args) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), forat.c_str(), args...);
        debug(std::string(buffer));
    }

    void set_level(LogLevel level) { min_level_ = level; }
    void enable_console(bool enable) { console_output_ = enable; }
};

// Training metrics logger
class MetricsLogger {
private:
    std::ofstream metrics_file_;
    std::string metrics_dir_;

public:
    MetricsLogger(const std::string& log_dir = "logs", const std::string& filename = "metrics.csv") 
        : metrics_dir_(log_dir) {
        
        std::string mkdir_cmd = "mkdir -p " + log_dir;
        system(mkdir_cmd.c_str());
        
        std::string full_path = log_dir + "/" + filename;
        metrics_file_.open(full_path, std::ios::app);
        
        if (metrics_file_.is_open()) {
            // Write CSV header
            metrics_file_ << "timestamp,epoch,step,loss,avg_loss,learning_rate,step_time_ms" << std::endl;
        }
    }

    ~MetricsLogger() {
        if (metrics_file_.is_open()) {
            metrics_file_.close();
        }
    }

    void log_training_step(int epoch, int step, float loss, float avg_loss, 
                          float learning_rate, int step_time_ms) {
        if (!metrics_file_.is_open()) return;

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        metrics_file_ << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                      << "," << epoch 
                      << "," << step 
                      << "," << std::fixed << std::setprecision(6) << loss
                      << "," << std::fixed << std::setprecision(6) << avg_loss
                      << "," << std::scientific << learning_rate
                      << "," << step_time_ms << std::endl;
        metrics_file_.flush();
    }

    void log_epoch_summary(int epoch, float avg_loss, int total_steps [[maybe_unused]], int epoch_time_sec) {
        if (!metrics_file_.is_open()) return;

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        metrics_file_ << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                      << "," << epoch 
                      << "," << "EPOCH_END"
                      << "," << std::fixed << std::setprecision(6) << avg_loss
                      << "," << std::fixed << std::setprecision(6) << avg_loss
                      << "," << "0"
                      << "," << (epoch_time_sec * 1000) << std::endl;
        metrics_file_.flush();
    }
};

// Global log manager
class LogManager {
private:
    static std::unique_ptr<Logger> instance_;
    
public:
    static Logger* get_logger() {
        return instance_.get();
    }
    
    static void init_logger(const std::string& log_dir = "logs", 
                           const std::string& filename = "training.log",
                           LogLevel min_level = LogLevel::INFO,
                           bool console_output = true) {
        instance_ = std::make_unique<Logger>(log_dir, filename, min_level, console_output);
    }
    
    static void shutdown() {
        instance_.reset();
    }
};

} // namespace ops

// Convenience macros
#define OPS_LOG_DEBUG(msg) if(ops::LogManager::get_logger()) ops::LogManager::get_logger()->debug(msg)
#define OPS_LOG_INFO(msg) if(ops::LogManager::get_logger()) ops::LogManager::get_logger()->info(msg)
#define OPS_LOG_WARNING(msg) if(ops::LogManager::get_logger()) ops::LogManager::get_logger()->warning(msg)
#define OPS_LOG_ERROR(msg) if(ops::LogManager::get_logger()) ops::LogManager::get_logger()->error(msg)

#define OPS_LOG_DEBUG_F(fmt, ...) if(ops::LogManager::get_logger()) ops::LogManager::get_logger()->debug_f(fmt, __VA_ARGS__)
#define OPS_LOG_INFO_F(fmt, ...) if(ops::LogManager::get_logger()) ops::LogManager::get_logger()->info_f(fmt, __VA_ARGS__)
#define OPS_LOG_ERROR_F(fmt, ...) if(ops::LogManager::get_logger()) ops::LogManager::get_logger()->error_f(fmt, __VA_ARGS__)

#endif // OPERATORS_LOGGER_H
