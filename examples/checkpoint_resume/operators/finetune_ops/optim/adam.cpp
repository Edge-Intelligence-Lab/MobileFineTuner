/**
 * @file adam.cpp
 * @brief Implementation of Adam optimizer
 * 
 * This file providess the implementsation of the Adam optimizer algorithm
 * with support for adaptive learning rates, momentum, and optional AMSGrad variant.
 */

#include "adam.h"
#include "../memory/arena_allocator.h"
#include "../core/logger.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <cstdint>

namespace ops {

namespace {
using Json = nlohmann::json;

void state_require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("Adam named state: " + message);
}

void state_keys(const Json& value, std::initializer_list<const char*> keys) {
    state_require(value.is_object() && value.size() == keys.size(), "unexpected object fields");
    for (const char* key : keys) state_require(value.contains(key), std::string("missing field: ") + key);
}

size_t state_size(const Json& value) {
    state_require(value.is_number_unsigned() ||
                  (value.is_number_integer() && value.get<int64_t>() >= 0), "expected nonnegative integer");
    const auto n = value.get<uint64_t>();
    state_require(n <= std::numeric_limits<size_t>::max(), "integer overflow");
    return static_cast<size_t>(n);
}

float state_float(const Json& value) {
    state_require(value.is_number(), "expected finite number");
    double v = value.get<double>();
    state_require(std::isfinite(v) && std::abs(v) <= std::numeric_limits<float>::max(), "nonfinite/out-of-range config");
    return static_cast<float>(v);
}

Json config_json(const AdamConfig& c) {
    return {{"learning_rate", c.learning_rate}, {"beta1", c.beta1}, {"beta2", c.beta2},
            {"epsilon", c.epsilon}, {"weight_decay", c.weight_decay},
            {"clip_grad_norm", c.clip_grad_norm}, {"amsgrad", c.amsgrad},
            {"decoupled_weight_decay", c.decoupled_weight_decay}};
}

AdamConfig read_config(const Json& j) {
    state_keys(j, {"learning_rate", "beta1", "beta2", "epsilon", "weight_decay",
                   "clip_grad_norm", "amsgrad", "decoupled_weight_decay"});
    state_require(j.at("amsgrad").is_boolean() && j.at("decoupled_weight_decay").is_boolean(), "expected boolean config");
    AdamConfig c(state_float(j.at("learning_rate")), state_float(j.at("beta1")),
                 state_float(j.at("beta2")), state_float(j.at("epsilon")),
                 state_float(j.at("weight_decay")), state_float(j.at("clip_grad_norm")),
                 j.at("amsgrad").get<bool>(), j.at("decoupled_weight_decay").get<bool>());
    state_require(c.learning_rate >= 0 && c.beta1 >= 0 && c.beta1 < 1 &&
                  c.beta2 >= 0 && c.beta2 < 1 && c.epsilon > 0 && c.weight_decay >= 0,
                  "invalid Adam configuration");
    // Nonpositive clip thresholds are preserved; this API does not add clipping.
    return c;
}

// Ordered address intervals catch different Tensor wrappers and overlapping views,
// without an O(parameter_count^2) alias scan.
using StorageRanges = std::map<uintptr_t, uintptr_t>;
void check_tensor(const TensorPtr& t, StorageRanges& ranges) {
    state_require(t && t->dtype() == kFloat32 && t->device() == kCPU && t->is_contiguous(),
                  "only contiguous CPU FP32 tensors are supported");
    size_t count = 1;
    for (int64_t dim : t->shape()) {
        state_require(dim > 0 && static_cast<uint64_t>(dim) <=
                      std::numeric_limits<size_t>::max() / sizeof(float) / count, "invalid tensor shape");
        count *= static_cast<size_t>(dim);
    }
    state_require(t->numel() > 0 && static_cast<uint64_t>(t->numel()) == count && t->data_ptr(), "invalid tensor storage");
    uintptr_t start = reinterpret_cast<uintptr_t>(t->data_ptr());
    state_require(count <= (std::numeric_limits<uintptr_t>::max() - start) / sizeof(float), "storage overflow");
    uintptr_t end = start + count * sizeof(float);
    auto next = ranges.lower_bound(start);
    state_require((next == ranges.end() || end <= next->first) &&
                  (next == ranges.begin() || std::prev(next)->second <= start), "aliased tensor storage");
    ranges.emplace(start, end);
    for (size_t i = 0; i < count; ++i) state_require(std::isfinite(t->data<float>()[i]), "nonfinite tensor");
}

std::map<std::string, TensorPtr> check_parameters(const NamedParameters& parameters, StorageRanges& ranges) {
    std::map<std::string, TensorPtr> named;
    for (const auto& [name, param] : parameters) {
        state_require(!name.empty() && named.emplace(name, param).second, "empty/duplicate parameter name");
        check_tensor(param, ranges);
    }
    return named;
}

std::string moment_key(const std::string& name, const char* moment) {
    return std::string(moment) + "/" + name;
}
}  // namespace

AdamSnapshot Adam::export_named_state(const NamedParameters& parameters) const {
    StorageRanges ranges;
    check_parameters(parameters, ranges);
    read_config(config_json(adam_config_));
    AdamSnapshot result;
    result.metadata = {{"schema", "mobilefinetuner.adam"}, {"version", 1},
                       {"config", config_json(adam_config_)}, {"parameters", Json::array()}};
    size_t initialized_count = 0;
    for (const auto& parameter : parameters) {
        const auto& name = parameter.first;
        const auto& param = parameter.second;
        auto it = states_.find(param);
        const bool initialized = it != states_.end();
        const size_t step = initialized ? it->second.step : 0;
        result.metadata["parameters"].push_back({{"name", name}, {"shape", param->shape()},
            {"dtype", "float32"}, {"initialized", initialized}, {"step", step}});
        if (!initialized) continue;
        ++initialized_count;
        const auto& state = it->second;
        state_require(step > 0 && step < std::numeric_limits<size_t>::max() &&
                      state.m.size() == 1 && state.v.size() == 1 &&
                      state.v_hat.size() == (adam_config_.amsgrad ? 1u : 0u), "invalid initialized state");
        auto add = [&](const char* kind, const TensorPtr& tensor) {
            check_tensor(tensor, ranges);
            state_require(tensor->shape() == param->shape(), "moment shape mismatch");
            if (std::string(kind) != "m") {
                for (int64_t i = 0; i < tensor->numel(); ++i)
                    state_require(tensor->data<float>()[i] >= 0, "negative second moment");
            }
            result.tensors.emplace(moment_key(name, kind), tensor);
        };
        add("m", state.m[0]);
        add("v", state.v[0]);
        if (adam_config_.amsgrad) add("vhat", state.v_hat[0]);
    }
    state_require(initialized_count == states_.size(), "parameter list omits optimizer state");
    return result;
}

void Adam::import_named_state(const NamedParameters& parameters, const AdamSnapshot& snapshot) {
    StorageRanges ranges;
    const auto named = check_parameters(parameters, ranges);
    const auto& j = snapshot.metadata;
    state_keys(j, {"schema", "version", "config", "parameters"});
    state_require(j.at("schema") == "mobilefinetuner.adam" && state_size(j.at("version")) == 1,
                  "unsupported schema/version");
    AdamConfig config = read_config(j.at("config"));
    const auto& entries = j.at("parameters");
    state_require(entries.is_array() && entries.size() == named.size(), "parameter count mismatch");
    std::set<std::string> seen;
    std::set<std::string> expected_tensors;
    std::unordered_map<TensorPtr, AdamState> restored;
    for (const auto& entry : entries) {
        state_keys(entry, {"name", "shape", "dtype", "initialized", "step"});
        state_require(entry.at("name").is_string() && entry.at("initialized").is_boolean(), "invalid parameter record");
        const auto name = entry.at("name").get<std::string>();
        auto param_it = named.find(name);
        state_require(param_it != named.end() && seen.insert(name).second, "unknown/duplicate parameter name");
        const auto& param = param_it->second;
        const auto& shape = entry.at("shape");
        state_require(shape.is_array() && shape.size() == param->shape().size() &&
                      entry.at("dtype") == "float32", "parameter shape/dtype mismatch");
        for (size_t d = 0; d < shape.size(); ++d)
            state_require(state_size(shape[d]) == static_cast<size_t>(param->shape()[d]), "parameter shape mismatch");
        AdamState state;
        state.step = state_size(entry.at("step"));
        const bool initialized = entry.at("initialized").get<bool>();
        state_require(initialized ? (state.step > 0 && state.step < std::numeric_limits<size_t>::max()) : state.step == 0,
                      "invalid initialization/step");
        if (!initialized) continue;
        auto take = [&](const char* kind) {
            auto key = moment_key(name, kind);
            auto it = snapshot.tensors.find(key);
            state_require(it != snapshot.tensors.end(), "missing tensor: " + key);
            const auto& tensor = it->second;
            check_tensor(tensor, ranges);
            state_require(tensor->shape() == param->shape(), "moment shape mismatch");
            if (std::string(kind) != "m") {
                for (int64_t i = 0; i < tensor->numel(); ++i)
                    state_require(tensor->data<float>()[i] >= 0, "negative second moment");
            }
            expected_tensors.insert(key);
            return tensor;
        };
        state.m.push_back(take("m"));
        state.v.push_back(take("v"));
        if (config.amsgrad) state.v_hat.push_back(take("vhat"));
        restored.emplace(param, std::move(state));
    }
    state_require(expected_tensors.size() == snapshot.tensors.size(), "unexpected tensor");
    // No observable mutation until all metadata, buffers and allocations pass.
    states_.swap(restored);
    adam_config_ = config;
    config_ = config;
}

Adam::Adam(const AdamConfig& config) 
    : Optimizer(config), adam_config_(config) {
}

// [Translated comment removed - see documentation]

void Adam::step(const std::vector<TensorPtr>& parameters,
               const std::vector<TensorPtr>& gradients) {
    if (parameters.size() != gradients.size()) {
        throw std::runtime_error("Parameters and gradients size mismatch");
    }
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        auto& param = parameters[i];
        auto& grad = gradients[i];
        
        if (!grad) continue;  // Skip if no gradient
        
        // Initialize state if needed
        if (states_.find(param) == states_.end()) {
            init_state(param);
        }
        
        auto& state = states_[param];
        state.step++;
        
        // Get current parameter and gradient data
        const float* grad_data = grad->data<float>();
        float* param_data = param->data<float>();
        float* m_data = state.m[0]->data<float>();
        float* v_data = state.v[0]->data<float>();
        float* v_hat_data = nullptr;
        
        if (adam_config_.amsgrad && !state.v_hat.empty()) {
            v_hat_data = state.v_hat[0]->data<float>();
        }
        
        // Compute bias correction factors
        float bias_correction1 = compute_bias_correction1(state.step);
        float bias_correction2 = compute_bias_correction2(state.step);
        
        // Update parameters. AdamW follows the PyTorch reference update:
        // param *= (1 - lr * weight_decay) before moment updates.
        for (int64_t j = 0; j < param->numel(); ++j) {
            if (adam_config_.decoupled_weight_decay && adam_config_.weight_decay > 0.0f) {
                param_data[j] *= (1.0f - adam_config_.learning_rate * adam_config_.weight_decay);
            }

            float grad_val = grad_data[j];
            
            // Classic Adam applies L2 regularization by folding weight decay into the gradient.
            if (!adam_config_.decoupled_weight_decay && adam_config_.weight_decay > 0.0f) {
                grad_val += adam_config_.weight_decay * param_data[j];
            }
            
            // Update biased first moment estimate
            m_data[j] = adam_config_.beta1 * m_data[j] + (1.0f - adam_config_.beta1) * grad_val;
            
            // Update biased second raw moment estimate
            v_data[j] = adam_config_.beta2 * v_data[j] + (1.0f - adam_config_.beta2) * grad_val * grad_val;
            
            float v_corrected = v_data[j] / bias_correction2;
            
            // AMSGrad variant
            if (adam_config_.amsgrad && v_hat_data) {
                v_hat_data[j] = std::max(v_hat_data[j], v_corrected);
                v_corrected = v_hat_data[j];
            }
            
            // Compute bias-corrected first moment estimate
            float m_corrected = m_data[j] / bias_correction1;
            
            // Update parameters
            param_data[j] -= adam_config_.learning_rate * m_corrected / (std::sqrt(v_corrected) + adam_config_.epsilon);
        }
    }
}

void Adam::save_state(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving Adam state: " + path);
    }
    
    // Save configuration
    file.write(reinterpret_cast<const char*>(&adam_config_.learning_rate), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.beta1), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.beta2), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.epsilon), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.weight_decay), sizeof(float));
    file.write(reinterpret_cast<const char*>(&adam_config_.amsgrad), sizeof(bool));
    file.write(reinterpret_cast<const char*>(&adam_config_.decoupled_weight_decay), sizeof(bool));
    
    // Save number of parameters
    size_t num_params = states_.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(size_t));
    
    // Save states for each parameter
    for (const auto& [param, state] : states_) {
        file.write(reinterpret_cast<const char*>(&state.step), sizeof(size_t));
        
        // Save first moment
        if (!state.m.empty()) {
            auto& m_tensor = state.m[0];
            int64_t m_size = m_tensor->numel();
            file.write(reinterpret_cast<const char*>(&m_size), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(m_tensor->data<float>()), m_size * sizeof(float));
        }
        
        // Save second moment
        if (!state.v.empty()) {
            auto& v_tensor = state.v[0];
            int64_t v_size = v_tensor->numel();
            file.write(reinterpret_cast<const char*>(&v_size), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(v_tensor->data<float>()), v_size * sizeof(float));
        }
        
        // Save v_hat for AMSGrad
        if (adam_config_.amsgrad && !state.v_hat.empty()) {
            auto& v_hat_tensor = state.v_hat[0];
            int64_t v_hat_size = v_hat_tensor->numel();
            file.write(reinterpret_cast<const char*>(&v_hat_size), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(v_hat_tensor->data<float>()), v_hat_size * sizeof(float));
        }
    }
    
    file.close();
}

void Adam::load_state(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading Adam state: " + path);
    }
    
    // Load configuration
    file.read(reinterpret_cast<char*>(&adam_config_.learning_rate), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.beta1), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.beta2), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.epsilon), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.weight_decay), sizeof(float));
    file.read(reinterpret_cast<char*>(&adam_config_.amsgrad), sizeof(bool));
    file.read(reinterpret_cast<char*>(&adam_config_.decoupled_weight_decay), sizeof(bool));
    
    // Load number of parameters
    size_t num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(size_t));
    
    // Note: This is a simplified implementsation
    // In a complete implementsation, we would need to match parameters by name or index
    
    file.close();
}

void Adam::update_param(TensorPtr /* param */, const TensorPtr& /* grad */, OptimizerState& /* state */) {
    // This method is called by the base class
    // The main logic is in the step() method above
}

void Adam::init_state(const TensorPtr& param) {
    AdamState state;
    state.step = 0;
    
    // Critical fix: Use StaticWeightArena to allocate optimizer state (long-term hold, not step-level Arena)
    #ifdef USE_ARENA_ALLOCATOR
    size_t state_bytes = param->numel() * sizeof(float);
    auto& static_arena = memory::ArenaManager::instance().static_weights();
    
    // Allocate m buffer (momentum)
    float* m_buffer = static_cast<float*>(static_arena.allocate_static(state_bytes, "adam_m"));
    std::memset(m_buffer, 0, state_bytes);
    auto m = std::make_shared<Tensor>(param->shape(), m_buffer, kFloat32, param->device(), true);
    state.m.push_back(m);
    
    // Allocate v buffer (variance)
    float* v_buffer = static_cast<float*>(static_arena.allocate_static(state_bytes, "adam_v"));
    std::memset(v_buffer, 0, state_bytes);
    auto v = std::make_shared<Tensor>(param->shape(), v_buffer, kFloat32, param->device(), true);
    state.v.push_back(v);
    
    // Allocate v_hat for AMSGrad if needed
    if (adam_config_.amsgrad) {
        float* v_hat_buffer = static_cast<float*>(static_arena.allocate_static(state_bytes, "adam_v_hat"));
        std::memset(v_hat_buffer, 0, state_bytes);
        auto v_hat = std::make_shared<Tensor>(param->shape(), v_hat_buffer, kFloat32, param->device(), true);
        state.v_hat.push_back(v_hat);
    }
    #else
    // Fallback: Traditional allocation (may enter step-level Arena, risky)
    auto m = std::make_shared<Tensor>(param->shape(), kFloat32, param->device());
    std::memset(m->data<float>(), 0, m->numel() * sizeof(float));
    state.m.push_back(m);
    
    auto v = std::make_shared<Tensor>(param->shape(), kFloat32, param->device());
    std::memset(v->data<float>(), 0, v->numel() * sizeof(float));
    state.v.push_back(v);
    
    if (adam_config_.amsgrad) {
        auto v_hat = std::make_shared<Tensor>(param->shape(), kFloat32, param->device());
        std::memset(v_hat->data<float>(), 0, v_hat->numel() * sizeof(float));
        state.v_hat.push_back(v_hat);
    }
    #endif
    
    states_[param] = std::move(state);
    
    // Silent initialization (avoid log noise)
    // OPS_LOG_DEBUG_F("Adam state initialized: %zu bytes (m+v)", state_bytes * 2);
}

float Adam::compute_bias_correction1(size_t step) const {
    return 1.0f - std::pow(adam_config_.beta1, static_cast<float>(step));
}

float Adam::compute_bias_correction2(size_t step) const {
    return 1.0f - std::pow(adam_config_.beta2, static_cast<float>(step));
}

void AdamState::to_file(const std::string& path) const {
        // [Translated]
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&step), sizeof(step));
        file.close();
    }
}

void AdamState::from_file(const std::string& path) {
        // [Translated]
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&step), sizeof(step));
        file.close();
    }
}

} // namespace ops
