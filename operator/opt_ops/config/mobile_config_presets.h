/**
 * @file mobile_config_presets.h
 * [English documentation available]
 * 
 * [English documentation available]
 */

#pragma once

#include <string>

namespace mobile_training {

/**
 * [English documentation available]
 */
enum class MobileDeviceType {
    ANDROID_HIGH_END,      // Pixel 7 Pro, Galaxy S23（8GB+ RAM）
    ANDROID_MID_RANGE,     // Pixel 6a（6GB RAM）
    ANDROID_LOW_END,       // entry-level devices (4GB RAM)
    IOS_HIGH_END,          // iPhone 14 Pro+（6GB+ RAM）
    IOS_MID_RANGE,         // iPhone 13/14（4-6GB RAM）
    EMULATOR_X86           // Android
};

/**
 * @brief training
 */
enum class TrainingScenario {
    QUICK_VALIDATION,      // quick validation (20-50 steps)
    DEMO_SHOWCASE,         // demo showcase (100-200 steps)
    PRODUCTION_TRAINING,   // production training (full epochs)
    MEMORY_CONSTRAINED     // extreme memory-saving mode
};

/**
 * @brief configurationgenerate
 */
class ConfigPresets {
public:
    /**
     * @brief getrecommendedconfiguration
     * @param device devices
     * @param scenario training
     * [English documentation available]
     */
    static std::string get_config(MobileDeviceType device, TrainingScenario scenario);
    
    /**
     * @brief getmemory
     * @param device devices
     * @param scenario training
     * @return memoryusage（MB）
     */
    static std::pair<size_t, size_t> estimate_memory(MobileDeviceType device, TrainingScenario scenario);
    
    /**
     * @brief printrecommendedconfiguration
     */
    static void print_recommended_config(MobileDeviceType device, TrainingScenario scenario);
};

// ===============================
// [English documentation available]
// ===============================

/**
 * @brief Pixel 7 Pro configuration（recommended）
 */
struct Pixel7ProConfig {
    static constexpr int block_size = 32;
    static constexpr int batch_size = 1;
    static constexpr int grad_accum_steps = 2;
    static constexpr int lora_rank = 4;
    static constexpr int lora_layers = 4;
    static constexpr int chunk_size = 1024;
    static constexpr int max_train_steps = 200;
    static constexpr float lr = 3e-4f;
    static constexpr bool use_chunked_ce = true;
    static constexpr bool enable_checkpointing = true;
    
    // memory：900 MB - 1.2 GB
    // [English documentation available]
    // [English documentation available]
};

/**
 * @brief iPhone 14 Pro configuration（recommended）
 */
struct iPhone14ProConfig {
    static constexpr int block_size = 32;
    static constexpr int batch_size = 1;
    static constexpr int grad_accum_steps = 2;
    static constexpr int lora_rank = 4;
    static constexpr int lora_layers = 4;
    static constexpr int chunk_size = 1024;
    static constexpr int max_train_steps = 200;
    static constexpr float lr = 3e-4f;
    static constexpr bool use_chunked_ce = true;
    static constexpr bool enable_checkpointing = true;
    
    // memory：700 MB - 1.0 GB
    // [English documentation available]
    // [English documentation available]
};

/**
 * [English documentation available]
 */
struct QuickValidationConfig {
    static constexpr int block_size = 24;
    static constexpr int batch_size = 1;
    static constexpr int grad_accum_steps = 1;
    static constexpr int lora_rank = 2;
    static constexpr int lora_layers = 2;
    static constexpr int chunk_size = 512;
    static constexpr int max_train_steps = 20;
    static constexpr float lr = 3e-4f;
    static constexpr bool use_chunked_ce = true;
    static constexpr bool enable_checkpointing = true;
    
    // memory：600 MB - 800 MB
    // [English documentation available]
    // [English documentation available]
};

/**
 * [English documentation available]
 */
struct MemoryConstrainedConfig {
    static constexpr int block_size = 16;
    static constexpr int batch_size = 1;
    static constexpr int grad_accum_steps = 1;
    static constexpr int lora_rank = 2;
    static constexpr int lora_layers = 2;
    static constexpr int chunk_size = 256;
    static constexpr int max_train_steps = 100;
    static constexpr float lr = 3e-4f;
    static constexpr bool use_chunked_ce = true;
    static constexpr bool enable_checkpointing = true;
    static constexpr int n_layer = 6;  // [Comment in English]
    
    // memory：400 MB - 600 MB
    // [English documentation available]
    // devices：<4GB RAM
};

/**
 * @brief Android configuration（x86_64）
 */
struct EmulatorConfig {
    static constexpr int block_size = 24;
    static constexpr int batch_size = 1;
    static constexpr int grad_accum_steps = 1;
    static constexpr int lora_rank = 4;
    static constexpr int lora_layers = 2;
    static constexpr int chunk_size = 512;
    static constexpr int max_train_steps = 50;
    static constexpr float lr = 3e-4f;
    static constexpr bool use_chunked_ce = true;
    static constexpr bool enable_checkpointing = true;
    
    // memory：700 MB - 900 MB
    // [English documentation available]
    // [English documentation available]
};

} // namespace mobile_training

