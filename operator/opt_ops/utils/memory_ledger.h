/**
 * @file memory_ledger.h
 * @brief Memory accounting and profiling for training
 * 
 * Provides detailed memory breakdown by category for optimization
 */

#pragma once

#include "tensor.h"
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace ops {

/**
 * @brief Memory breakdown by category
 */
struct MemoryLedger {
    size_t weights_bytes = 0;       // Model parameters (frozen + trainable)
    size_t gradients_bytes = 0;     // Gradients for trainable parameters
    size_t optimizer_bytes = 0;     // Optimizer state (Adam m/v)
    size_t activations_bytes = 0;   // Forward pass activations (estimated)
    size_t temp_buffers_bytes = 0;  // Temporary computation buffers
    size_t total_rss_bytes = 0;     // Total RSS from system
    
    /**
     * @brief Compute memory usage from parameter list
     */
    static MemoryLedger compute(const std::vector<TensorPtr>& all_params,
                                const std::vector<TensorPtr>& trainable_params,
                                size_t system_rss) {
        MemoryLedger ledger;
        ledger.total_rss_bytes = system_rss;
        
        // Compute weights
        for (const auto& param : all_params) {
            if (param) {
                size_t param_bytes = param->numel() * DTypeUtils::size_of(param->dtype());
                ledger.weights_bytes += param_bytes;
            }
        }
        
        // Compute gradients
        for (const auto& param : trainable_params) {
            if (param && param->grad()) {
                auto grad = param->grad();
                size_t grad_bytes = grad->numel() * DTypeUtils::size_of(grad->dtype());
                ledger.gradients_bytes += grad_bytes;
            }
        }
        
        // Optimizer state (Adam: 2x parameters for m and v in FP32)
        for (const auto& param : trainable_params) {
            if (param) {
                // Each param has m and v in FP32
                size_t state_bytes = param->numel() * sizeof(float) * 2;
                ledger.optimizer_bytes += state_bytes;
            }
        }
        
        // Activations (rough estimate: assume 10% of RSS)
        ledger.activations_bytes = system_rss / 10;
        
        // Temp buffers (remainder)
        size_t accounted = ledger.weights_bytes + ledger.gradients_bytes + 
                          ledger.optimizer_bytes + ledger.activations_bytes;
        if (system_rss > accounted) {
            ledger.temp_buffers_bytes = system_rss - accounted;
        }
        
        return ledger;
    }
    
    /**
     * @brief Format memory size for display
     */
    static std::string forat_bytes(size_t bytes) {
        const double KB = 1024.0;
        const double MB = KB * 1024.0;
        const double GB = MB * 1024.0;
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        
        if (bytes >= GB) {
            oss << (bytes / GB) << " GB";
        } else if (bytes >= MB) {
            oss << (bytes / MB) << " MB";
        } else if (bytes >= KB) {
            oss << (bytes / KB) << " KB";
        } else {
            oss << bytes << " B";
        }
        
        return oss.str();
    }
    
    /**
     * @brief Print detailed memory breakdown
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "\n  📊 memoryledger:\n";
        oss << "    weight:     " << std::setw(10) << forat_bytes(weights_bytes) << "\n";
        oss << "    gradient:     " << std::setw(10) << forat_bytes(gradients_bytes) << "\n";
        oss << "    optimizer:   " << std::setw(10) << forat_bytes(optimizer_bytes) << "\n";
        oss << "    activation:     " << std::setw(10) << forat_bytes(activations_bytes) << " ()\n";
        oss << "    temporarycache: " << std::setw(10) << forat_bytes(temp_buffers_bytes) << "\n";
        oss << "    Total RSS:    " << std::setw(10) << forat_bytes(total_rss_bytes);
        return oss.str();
    }
    
    /**
     * @brief Print compact one-line summary
     */
    std::string to_compact_string() const {
        std::ostringstream oss;
        oss << "W:" << forat_bytes(weights_bytes) 
            << " G:" << forat_bytes(gradients_bytes)
            << " O:" << forat_bytes(optimizer_bytes)
            << " A:" << forat_bytes(activations_bytes)
            << " T:" << forat_bytes(temp_buffers_bytes)
            << " RSS:" << forat_bytes(total_rss_bytes);
        return oss.str();
    }
};

} // namespace ops
