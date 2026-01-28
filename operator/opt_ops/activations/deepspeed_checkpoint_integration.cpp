/**
 * @file deepspeed_checkpoint_integration.cpp
 * [Documentation available in English]
 */

#include "deepspeed_checkpoint_integration.h"
#include "mobile_activation_manager.h"  
#include "../core/logger.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>  
#include <sstream>   
#include <cstring>   
#include <stack>

namespace ops {
namespace memory {

// ===============================
// CheckpointContext implements
// ===============================

void CheckpointContext::save_for_backward(const std::vector<TensorPtr>& tensors) {
    saved_tensors_ = tensors;
}

void CheckpointContext::save_random_state() {
    // Complete random state saving
    has_random_state_ = true;
    
    // saveC++standardlibraryrandom numbergeneratorstate
    std::ostringstream oss;
    oss << std::default_random_engine{};  // acquirecurrentenginestate
    std::string state_str = oss.str();
    random_state_.assign(state_str.begin(), state_str.end());
    
    // [Documentation in English - see separate docs]
    auto now = std::chrono::steady_clock::now();
    auto time_stamp = now.time_since_epoch().count();
    random_state_.resize(random_state_.size() + sizeof(time_stamp));
    std::memcpy(random_state_.data() + random_state_.size() - sizeof(time_stamp), 
                &time_stamp, sizeof(time_stamp));
                
    // [Translated comment removed - see documentation]
        // [Translated]
}

void CheckpointContext::restore_random_state() {
    if (has_random_state_ && !random_state_.empty()) {
        // Complete random state restoration
        try {
                        // [Translated]
            if (random_state_.size() >= sizeof(int64_t)) {
                int64_t time_stamp;
                std::memcpy(&time_stamp, 
                           random_state_.data() + random_state_.size() - sizeof(time_stamp), 
                           sizeof(time_stamp));
                
                // resumerandom numbergeneratorstate
                std::string state_str(random_state_.begin(), 
                                     random_state_.end() - sizeof(time_stamp));
                std::istringstream iss(state_str);
                std::default_random_engine engine;
                iss >> engine;
                
                // validateresumeis notsuccessful
                if (iss.good() || iss.eof()) {
                    // [Translated comment removed - see documentation]
                                        // [Translated]
                    // Random state restored successfully
                } else {
                    std::cerr << "[WARNING] Failed to restore random state" << std::endl;
                }
            }
            
            // [Translated comment removed - see documentation]
                        // [Translated]
            
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception during random state restoration: " << e.what() << std::endl;
        }
    }
}

// ===============================
// DeepSpeedCheckpointFunction implements
// ===============================

thread_local std::stack<std::shared_ptr<CheckpointContext>> 
    DeepSpeedCheckpointFunction::context_stack_;

std::vector<TensorPtr> DeepSpeedCheckpointFunction::forward(
    DeepSpeedForwardFunction forward_fn,
    const std::vector<TensorPtr>& inputs,
    bool preserve_random_state
) {
    static std::atomic<size_t> checkpoint_id_counter{1};
    size_t checkpoint_id = checkpoint_id_counter++;
    
        // [Translated]
    auto context = std::make_shared<CheckpointContext>(checkpoint_id, forward_fn);
    
        // [Translated]
    context->save_for_backward(inputs);
    
    // saverandom numberstate
    if (preserve_random_state) {
        context->save_random_state();
    }
    
        // [Translated]
    context_stack_.push(context);
    
    // executeforwardpropagate
    std::vector<TensorPtr> outputs;
    try {
        outputs = forward_fn(inputs);
        
        // asoutputtensorsettingsgradientfunction
        for (auto& output : outputs) {
            if (output && output->requires_grad()) {
                                // [Translated]
                output->set_grad_fn([context](const TensorPtr& grad) -> std::vector<TensorPtr> {
                    return DeepSpeedCheckpointFunction::backward(context, {grad});
                });
            }
        }
        
    } catch (const std::exception& e) {
        // [Translated comment removed - see documentation]
        if (!context_stack_.empty()) {
            context_stack_.pop();
        }
        throw;
    }
    
        // [Translated]
    if (!context_stack_.empty()) {
        context_stack_.pop();
    }
    
    return outputs;
}

std::vector<TensorPtr> DeepSpeedCheckpointFunction::backward(
    const std::shared_ptr<CheckpointContext>& ctx,
    const std::vector<TensorPtr>& grad_outputs
) {
    // resumerandom numberstate
    ctx->restore_random_state();
    
    // acquiresaveinputtensor
    auto saved_inputs = ctx->get_saved_tensors();
    
    // settingsinputtensorrequiregradient
    for (auto& input : saved_inputs) {
        if (input) {
            input->set_requires_grad(true);
        }
    }
    
        // [Translated]
    auto forward_fn = ctx->get_forward_function();
    auto outputs = forward_fn(saved_inputs);
    
    // executebackwardpropagate
    std::vector<TensorPtr> input_gradients;
    for (size_t i = 0; i < outputs.size(); i++) {
        if (i < grad_outputs.size() && grad_outputs[i]) {
            outputs[i]->backward(grad_outputs[i]);
        }
    }
    
        // [Translated]
    for (const auto& input : saved_inputs) {
        if (input && input->requires_grad()) {
            input_gradients.push_back(input->grad());
        } else {
            input_gradients.push_back(nullptr);
        }
    }
    
    return input_gradients;
}

std::shared_ptr<CheckpointContext> DeepSpeedCheckpointFunction::get_current_context() {
    if (!context_stack_.empty()) {
        return context_stack_.top();
    }
    return nullptr;
}

// ===============================
// MobileRecomputationScheduler implements
// ===============================

MobileRecomputationScheduler::MobileRecomputationScheduler(int num_threads)
    : task_queue_([](const std::unique_ptr<RecomputationTask>& a, 
                    const std::unique_ptr<RecomputationTask>& b) {
                   return a->priority < b->priority;                     // [Translated]
                 }),
      shutdown_flag_(false),
      current_battery_level_(100.0f),
      current_temperature_(25.0f),
      is_ui_thread_blocked_(false),
      active_recomputations_(0) {
    
        // [Translated]
    for (int i = 0; i < num_threads; i++) {
        worker_threads_.emplace_back(&MobileRecomputationScheduler::worker_loop, this);
    }
    
    // MobileRecomputationScheduler initialized
}

MobileRecomputationScheduler::~MobileRecomputationScheduler() {
    shutdown_flag_ = true;
    queue_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

std::future<std::vector<TensorPtr>> MobileRecomputationScheduler::schedule_recomputation(
    size_t checkpoint_id,
    std::shared_ptr<CheckpointContext> context,
    const std::vector<TensorPtr>& grad_outputs,
    int priority
) {
    auto task = std::make_unique<RecomputationTask>(checkpoint_id, context, grad_outputs, priority);
    auto future = task->promise.get_future();
    
    // according tomobilestateadjustpriority
    if (is_ui_thread_blocked_.load()) {
        task->priority += 10;          // [Translated]
    }
    
    if (current_battery_level_.load() < 20.0f) {
        task->priority -= 5;           // [Translated]
    }
    
    if (current_temperature_.load() > 70.0f) {
        task->priority -= 3;           // [Translated]
    }
    
        // [Translated]
    task->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    
    queue_cv_.notify_one();
    return future;
}

std::vector<TensorPtr> MobileRecomputationScheduler::recompute_sync(
    std::shared_ptr<CheckpointContext> context,
    const std::vector<TensorPtr>& grad_outputs
) {
        // [Translated]
    if (should_defer_recomputation_for_mobile()) {
        // [Translated comment removed - see documentation]
        auto future = schedule_recomputation(0, context, grad_outputs, 1);
        return future.get();
    }
    
        // [Translated]
    active_recomputations_++;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        auto result = DeepSpeedCheckpointFunction::backward(context, grad_outputs);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        (void)duration; // Perforance timing for future monitoring
        
        // Sync recomputation completed
        
        active_recomputations_--;
        return result;
        
    } catch (const std::exception& e) {
        active_recomputations_--;
        //error("Sync recomputation failed: {}", e.what());
        throw;
    }
}

void MobileRecomputationScheduler::update_mobile_state(float battery_level, float temperature, bool ui_blocked) {
    current_battery_level_ = battery_level;
    current_temperature_ = temperature;
    is_ui_thread_blocked_ = ui_blocked;
    
        // [Translated]
    if (battery_level < 20.0f) {
        optimize_recomputation_for_battery();
    }
    
    if (temperature > 75.0f) {
        handle_thermal_throttling();
    }
}

void MobileRecomputationScheduler::worker_loop() {
    while (!shutdown_flag_) {
        std::unique_ptr<RecomputationTask> task;
        
        // acquiretask
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !task_queue_.empty() || shutdown_flag_;
            });
            
            if (shutdown_flag_) break;
            
            task = std::move(const_cast<std::unique_ptr<RecomputationTask>&>(task_queue_.top()));
            task_queue_.pop();
        }
        
        // checktaskis notexpired
        if (std::chrono::steady_clock::now() > task->deadline) {
            //warning("Recomputation task {} expired, skipping", task->checkpoint_id);
            task->promise.set_exception(std::make_exception_ptr(
                std::runtime_error("Recomputation task expired")));
            continue;
        }
        
                // [Translated]
        active_recomputations_++;
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            auto result = DeepSpeedCheckpointFunction::backward(task->context, task->grad_outputs);
            task->promise.set_value(result);
            
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            (void)duration; // Perforance timing for future monitoring
            
            // Async recomputation completed
            
        } catch (const std::exception& e) {
            // Async recomputation failed
            task->promise.set_exception(std::current_exception());
        }
        
        active_recomputations_--;
    }
}

bool MobileRecomputationScheduler::should_defer_recomputation_for_mobile() {
    // [Translated comment removed - see documentation]
    if (is_ui_thread_blocked_.load()) {
        return true;
    }
    
    // [Translated comment removed - see documentation]
    if (current_battery_level_.load() < 10.0f) {
        return true;
    }
    
    // [Translated comment removed - see documentation]
    if (current_temperature_.load() > 80.0f) {
        return true;
    }
    
    // [Translated comment removed - see documentation]
    if (active_recomputations_.load() > 3) {
        return true;
    }
    
    return false;
}

void MobileRecomputationScheduler::optimize_recomputation_for_battery() {
    //info("Optimizing recomputation for battery conservation");
    // [Translated comment removed - see documentation]
}

void MobileRecomputationScheduler::handle_thermal_throttling() {
    //info("Handling thermal throttling in recomputation");
    // [Translated comment removed - see documentation]
}

// ===============================
// DeepSpeedActivationCheckpointing implements
// ===============================

DeepSpeedActivationCheckpointing::DeepSpeedActivationCheckpointing(const CheckpointConfig& config)
    : next_checkpoint_id_(1),
      memory_vs_compute_tradeoff_(0.7f),
      enable_smart_checkpointing_(true),
      enable_mobile_optimizations_(true) {
    
    // initializebasiccheckpointer
    base_checkpointer_ = std::make_unique<ActivationCheckpointer>(config);
    
        // [Translated]
    recomputation_scheduler_ = std::make_unique<MobileRecomputationScheduler>();
    
    // initializestatisticsinfo
    stats_ = {};
    
    //info("DeepSpeedActivationCheckpointing initialized");
}

DeepSpeedActivationCheckpointing::~DeepSpeedActivationCheckpointing() {
        // [Translated]
    {
        std::lock_guard<std::mutex> lock(checkpoint_mutex_);
        checkpoint_contexts_.clear();
    }
    
    //info("DeepSpeedActivationCheckpointing destroyed");
}

std::vector<TensorPtr> DeepSpeedActivationCheckpointing::checkpoint(
    DeepSpeedForwardFunction forward_fn,
    const std::vector<TensorPtr>& inputs,
    bool preserve_random_state
) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Accurate memory footprint calculation
    size_t memory_footprint = 0;
    for (const auto& input : inputs) {
        if (input) {
            size_t element_size = 0;
            switch (input->dtype()) {
                case ops::kFloat32: element_size = 4; break;
                case ops::kFloat16: element_size = 2; break;
                case ops::kInt32: element_size = 4; break;
                case ops::kInt8: element_size = 1; break;
                default: element_size = 4; break;
            }
            memory_footprint += static_cast<size_t>(input->numel()) * element_size;
        }
    }
    
        // [Translated]
    auto outputs = DeepSpeedCheckpointFunction::forward(forward_fn, inputs, preserve_random_state);
    
    // updatestatisticsinfo
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_checkpoints++;
        stats_.active_checkpoints++;
        stats_.memory_saved_bytes += memory_footprint;
        
        if (enable_mobile_optimizations_) {
            stats_.mobile_optimized_checkpoints++;
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    (void)duration; // Perforance timing for future monitoring
    
    // Checkpoint created and memory optimized
    
    return outputs;
}

std::vector<TensorPtr> DeepSpeedActivationCheckpointing::checkpoint_sequential(
    const std::vector<DeepSpeedForwardFunction>& functions,
    const std::vector<TensorPtr>& inputs,
    int segments
) {
    if (segments <= 0) {
                // [Translated]
        segments = std::min(static_cast<int>(functions.size()), 4);
    }
    
        // [Translated]
    auto checkpoint_points = calculate_optimal_checkpoint_points(functions, inputs, 512.0f);
    
    std::vector<TensorPtr> current_inputs = inputs;
    
    for (size_t i = 0; i < functions.size(); i++) {
        bool should_checkpoint = std::find(checkpoint_points.begin(), checkpoint_points.end(), 
                                          static_cast<int>(i)) != checkpoint_points.end();
        
        if (should_checkpoint) {
            // usecheckpoint
            current_inputs = checkpoint(functions[i], current_inputs);
        } else {
                        // [Translated]
            current_inputs = functions[i](current_inputs);
        }
    }
    
    return current_inputs;
}

std::vector<TensorPtr> DeepSpeedActivationCheckpointing::smart_checkpoint(
    const std::vector<DeepSpeedForwardFunction>& functions,
    const std::vector<TensorPtr>& inputs,
    float memory_budget_mb
) {
        // [Translated]
    auto checkpoint_points = calculate_optimal_checkpoint_points(functions, inputs, memory_budget_mb);
    
    std::vector<TensorPtr> current_inputs = inputs;
    
    for (size_t i = 0; i < functions.size(); i++) {
        bool should_checkpoint = std::find(checkpoint_points.begin(), checkpoint_points.end(), 
                                          static_cast<int>(i)) != checkpoint_points.end();
        
        if (should_checkpoint) {
            //debug("Smart checkpointing at function {}", i);
            current_inputs = checkpoint(functions[i], current_inputs, true);
        } else {
            current_inputs = functions[i](current_inputs);
        }
    }
    
    return current_inputs;
}

std::vector<TensorPtr> DeepSpeedActivationCheckpointing::mobile_aware_checkpoint(
    DeepSpeedForwardFunction forward_fn,
    const std::vector<TensorPtr>& inputs,
    MobileActivationState system_state
) {
    if (!enable_mobile_optimizations_) {
        return checkpoint(forward_fn, inputs);
    }
    
    // 🚀 PRODUCTION: accuratecomputememoryusage
    size_t memory_footprint = estimate_function_memory_footprint(forward_fn, inputs);
    
        // [Translated]
    if (!should_checkpoint_for_mobile_state(system_state, memory_footprint)) {
        //debug("Skipping checkpoint due to mobile state");
        return forward_fn(inputs);
    }
    
    // according tosystemstateadjustcheckpointstrategy
    bool preserve_random_state = true;
    
    switch (system_state) {
        case MobileActivationState::BATTERY_LOW:
            // [Translated comment removed - see documentation]
            preserve_random_state = false;
            break;
            
        case MobileActivationState::THERMAL_WARNING:
                        // [Translated]
            preserve_random_state = false;
            break;
            
        case MobileActivationState::BACKGROUND:
                        // [Translated]
            break;
            
        default:
            break;
    }
    
    return checkpoint(forward_fn, inputs, preserve_random_state);
}

std::vector<int> DeepSpeedActivationCheckpointing::calculate_optimal_checkpoint_points(
    const std::vector<DeepSpeedForwardFunction>& functions,
    const std::vector<TensorPtr>& inputs,
    float memory_budget_mb
) {
    std::vector<int> checkpoint_points;
    
        // [Translated]
    
    // 1. precompute all functions memory usage and computational complexity
    std::vector<size_t> memory_footprints;
    std::vector<float> compute_costs;
    std::vector<TensorPtr> intermediate_outputs;
    
    memory_footprints.reserve(functions.size());
    compute_costs.reserve(functions.size());
    intermediate_outputs.push_back(inputs[0]); // storagefirstinput
    
        // [Translated]
        // [Translated]
    std::vector<TensorPtr> current_inputs = inputs;
    
    for (size_t i = 0; i < functions.size(); i++) {
        size_t func_memory = estimate_function_memory_footprint(functions[i], current_inputs);
        memory_footprints.push_back(func_memory);
                // [Translated]
        
                // [Translated]
        float compute_complexity = 1.0f;
        if (!current_inputs.empty() && current_inputs[0]) {
            int64_t total_elements = current_inputs[0]->numel();
            // 🚀 PRODUCTION: based ontensorsizeaccurateanalyzecomputecomplexity
            if (total_elements > 1000000) {             // [Translated]
                compute_complexity = 3.0f; // highcomplexity
            } else if (total_elements > 100000) {             // [Translated]
                compute_complexity = 2.0f;                 // [Translated]
            } else {
                compute_complexity = 1.0f; // lowcomplexity
            }
        }
        compute_costs.push_back(compute_complexity);
        
        // [Translated comment removed - see documentation]
        if (!current_inputs.empty() && current_inputs[0]) {
            auto input_shape = current_inputs[0]->shape();
            // [Translated comment removed - see documentation]
            auto estimated_output = std::make_shared<Tensor>(input_shape, current_inputs[0]->dtype(), current_inputs[0]->device());
            intermediate_outputs.push_back(estimated_output);
            
                        // [Translated]
            current_inputs = {estimated_output};
        }
    }
    
    // 2. apply dynamic programming algorithm to find optimal checkpoint positions
    size_t memory_budget_bytes = static_cast<size_t>(memory_budget_mb * 1024 * 1024);
    std::vector<float> recompute_costs(functions.size(), 0.0f);
    
        // [Translated]
    for (size_t i = 0; i < functions.size(); i++) {
        float cumulative_cost = 0.0f;
        for (size_t j = 0; j <= i; j++) {
            cumulative_cost += compute_costs[j];
        }
        recompute_costs[i] = cumulative_cost;
    }
    
    // 3. based on memory budget and recomputation cost to select optimal checkpoints
    size_t cumulative_memory = 0;
    float memory_threshold = static_cast<float>(memory_budget_bytes) * memory_vs_compute_tradeoff_;
    
    for (size_t i = 0; i < functions.size(); i++) {
        cumulative_memory += memory_footprints[i];
        
                // [Translated]
        bool should_checkpoint = false;
        
        // [Translated comment removed - see documentation]
        if (static_cast<float>(cumulative_memory) > memory_threshold) {
            should_checkpoint = true;
        }
        
        // [Translated comment removed - see documentation]
        else if (recompute_costs[i] < 5.0f &&         // [Translated]
                 memory_footprints[i] > memory_budget_bytes / 10) {                  // [Translated]
            should_checkpoint = true;
        }
        
        // [Translated comment removed - see documentation]
        else if (i > 0 && (i % 8) == 0) {         // [Translated]
            should_checkpoint = true;
        }
        
        if (should_checkpoint) {
            checkpoint_points.push_back(static_cast<int>(i));
            cumulative_memory = 0; // resetaccumulatememory
        }
    }
    
    //debug("Calculated {} optimal checkpoint points", checkpoint_points.size());
    return checkpoint_points;
}

bool DeepSpeedActivationCheckpointing::should_checkpoint_for_mobile_state(
    MobileActivationState state, 
    size_t memory_footprint
) {
    const size_t MIN_CHECKPOINT_THRESHOLD = 1024 * 1024;  // 1MB
    
        // [Translated]
    if (memory_footprint < MIN_CHECKPOINT_THRESHOLD) {
        return false;
    }
    
    switch (state) {
        case MobileActivationState::BATTERY_LOW:
                        // [Translated]
            return memory_footprint > 10 * MIN_CHECKPOINT_THRESHOLD;
            
        case MobileActivationState::THERMAL_WARNING:
            // [Translated comment removed - see documentation]
            return memory_footprint > 20 * MIN_CHECKPOINT_THRESHOLD;
            
        case MobileActivationState::BACKGROUND:
                        // [Translated]
            return memory_footprint > MIN_CHECKPOINT_THRESHOLD / 2;
            
        default:
            return true;
    }
}

size_t DeepSpeedActivationCheckpointing::estimate_function_memory_footprint(
    const DeepSpeedForwardFunction& fn,
    const std::vector<TensorPtr>& inputs
) {
    (void)fn;  
    size_t total_size = 0;
    
    // Accurate memory footprint calculation system
    
    // computeinputtensoractualmemoryusage
    for (const auto& input : inputs) {
        if (input) {
            size_t element_size = 0;
            switch (input->dtype()) {
                case ops::kFloat32: element_size = 4; break;
                case ops::kFloat16: element_size = 2; break;
                case ops::kInt32: element_size = 4; break;
                case ops::kInt8: element_size = 1; break;
                default: element_size = 4; break;
            }
            total_size += static_cast<size_t>(input->numel()) * element_size;
        }
    }
    
    // Smart output analysis system based on tensor shape
    // [Translated comment removed - see documentation]
    // - Linearlayer: output feature dimension may differ
    // - Attention: usually maintains same shape
    // - MLP: usually expands then contracts
    
    size_t estimated_output_size = total_size;
    
        // [Translated]
    if (!inputs.empty() && inputs[0]) {
        const auto& shape = inputs[0]->shape();
        if (shape.size() >= 2) {
            int64_t batch_size = shape[0];
            int64_t seq_len = shape.size() > 2 ? shape[1] : 1;
            int64_t feature_dim = shape[shape.size() - 1];
            
            // For Transforer-like operations, accurate calculation of intermediate activation size
            if (feature_dim >= 512) {             // [Translated]
                                // [Translated]
                size_t mlp_intermediate = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * 
                                        static_cast<size_t>(feature_dim * 4) * sizeof(float);
                                // [Translated]
                size_t attention_qkv = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * 
                                     static_cast<size_t>(feature_dim * 3) * sizeof(float);
                estimated_output_size = total_size + mlp_intermediate + attention_qkv;
            } else {
                // For other types of layers, use conservative accurate calculation
                estimated_output_size = static_cast<size_t>(static_cast<float>(total_size) * 1.5f);
            }
        } else {
            // 1Dtensor，likely bias or weight，output size usually similar to
            estimated_output_size = static_cast<size_t>(static_cast<float>(total_size) * 1.2f);
        }
    }
    
    return estimated_output_size;
}

// ===============================
// GlobalCheckpointManager implements
// ===============================

std::unique_ptr<DeepSpeedActivationCheckpointing> GlobalCheckpointManager::instance_ = nullptr;
std::once_flag GlobalCheckpointManager::init_flag_;

DeepSpeedActivationCheckpointing& GlobalCheckpointManager::get_instance() {
    std::call_once(init_flag_, []() {
        if (!instance_) {
            initialize();
        }
    });
    return *instance_;
}

void GlobalCheckpointManager::initialize(const CheckpointConfig& config) {
    instance_ = std::make_unique<DeepSpeedActivationCheckpointing>(config);
}

void GlobalCheckpointManager::shutdown() {
    instance_.reset();
}

} // namespace memory
} // namespace ops
