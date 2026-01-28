/**
 * @file module.h
 * @brief Neural network module base class and parameter management
 * 
 * This file providess the base Module class for building neural networks
 * in a PyTorch-like style. It includes parameter management, forward/backward
 * propagation, and state serialization capabilities.
 */

#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace ops {
/**
 * @brief Neural network module namespace
 * 
 * Contains classes and utilities for building neural networks
 * with automatic parameter management and gradient computation.
 */
namespace nn {

class Module;
using ModulePtr = std::shared_ptr<Module>;

/**
 * @brief Parameter wrapper class for neural network parameters
 * 
 * Wraps a tensor to mark it as a trainable parameter.
 * Parameters automatically have gradients enabled and are
 * tracked by parent modules.
 */
class Parameter {
public:
    /**
     * @brief Construct a parameter from a tensor
     * @param data The tensor data for this parameter
     */
    Parameter(const TensorPtr& data) : data_(data) {
        data_->set_requires_grad(true);
    }

    TensorPtr data() const { return data_; }
    TensorPtr grad() const { return data_->grad(); }

    void zero_grad() { data_->zero_grad(); }

private:
    TensorPtr data_;
};

using ParameterPtr = std::shared_ptr<Parameter>;

class Module {
public:
    Module() = default;
    virtual ~Module() = default;

    virtual TensorPtr forward(const TensorPtr& input) = 0;

    virtual TensorPtr forward(const std::vector<TensorPtr>& inputs) {
        throw std::runtime_error("Multi-input forward not implementsed");
    }

    TensorPtr operator()(const TensorPtr& input) {
        return forward(input);
    }

    TensorPtr operator()(const std::vector<TensorPtr>& inputs) {
        return forward(inputs);
    }

    virtual void train(bool mode = true) {
        training_ = mode;
        for (auto& [name, module] : children_) {
            module->train(mode);
        }
    }

    void eval() { train(false); }
    bool training() const { return training_; }

    virtual std::vector<ParameterPtr> parameters(bool recurse = true) {
        std::vector<ParameterPtr> params = parameters_;

        if (recurse) {
            for (auto& [name, module] : children_) {
                auto child_params = module->parameters(true);
                params.insert(params.end(), child_params.begin(), child_params.end());
            }
        }

        return params;
    }

    virtual std::vector<ParameterPtr> named_parameters(bool recurse = true) {
        return parameters(recurse);
    }

    virtual void zero_grad() {
        for (auto& param : parameters()) {
            param->zero_grad();
        }
    }

    virtual void to(const Device& device) {
        device_ = device;
        for (auto& param : parameters_) {
            param->data() = param->data()->to(device);
        }
        for (auto& [name, module] : children_) {
            module->to(device);
        }
    }

    void cpu() { to(kCPU); }
    void cuda() { to(kCUDA); }

    Device device() const { return device_; }

    virtual void type(DType dtype) {
        dtype_ = dtype;

    }

    DType dtype() const { return dtype_; }

    void register_module(const std::string& name, const ModulePtr& module) {
        children_[name] = module;
    }

    void register_parameter(const std::string& name, const ParameterPtr& param) {
        parameters_.push_back(param);
        named_parameters_[name] = param;
    }

    ModulePtr get_submodule(const std::string& name) {
        auto it = children_.find(name);
        return (it != children_.end()) ? it->second : nullptr;
    }

    virtual std::unordered_map<std::string, TensorPtr> state_dict() {
        std::unordered_map<std::string, TensorPtr> state;

        for (auto& [name, param] : named_parameters_) {
            state[name] = param->data();
        }

        for (auto& [name, module] : children_) {
            auto child_state = module->state_dict();
            for (auto& [key, value] : child_state) {
                state[name + "." + key] = value;
            }
        }

        return state;
    }

    virtual void load_state_dict(const std::unordered_map<std::string, TensorPtr>& state_dict) {

        for (auto& [name, param] : named_parameters_) {
            auto it = state_dict.find(name);
            if (it != state_dict.end()) {

            }
        }

        for (auto& [name, module] : children_) {
            std::unordered_map<std::string, TensorPtr> child_state;
            std::string prefix = name + ".";

            for (auto& [key, value] : state_dict) {
                if (key.substr(0, prefix.length()) == prefix) {
                    child_state[key.substr(prefix.length())] = value;
                }
            }

            if (!child_state.empty()) {
                module->load_state_dict(child_state);
            }
        }
    }

    virtual void save(const std::string& path) {
        auto state = state_dict();

    }

    virtual void load(const std::string& path) {

    }

    virtual size_t num_parameters() {
        size_t count = 0;
        for (auto& param : parameters()) {
            count += param->data()->numel();
        }
        return count;
    }

    virtual void print_structure(int indent = 0) {
        std::string spaces(indent * 2, ' ');
        std::cout << spaces << get_module_name() << std::endl;

        for (auto& [name, module] : children_) {
            std::cout << spaces << "(" << name << "): ";
            module->print_structure(indent + 1);
        }
    }

protected:

    virtual std::string get_module_name() const {
        return "Module";
    }

    ParameterPtr create_parameter(const std::vector<int64_t>& shape,
                                 bool requires_grad = true,
                                 const std::string& init_method = "normal") {
        TensorPtr tensor;

        if (init_method == "normal") {
            tensor = randn(shape, dtype_, device_);
        } else if (init_method == "zeros") {
            tensor = zeros(shape, dtype_, device_);
        } else if (init_method == "ones") {
            tensor = ones(shape, dtype_, device_);
        } else {
            tensor = randn(shape, dtype_, device_);
        }

        tensor->set_requires_grad(requires_grad);
        return std::make_shared<Parameter>(tensor);
    }

private:
    bool training_ = true;
    Device device_ = kCPU;
    DType dtype_ = kFloat32;

    std::unordered_map<std::string, ModulePtr> children_;

    std::vector<ParameterPtr> parameters_;
    std::unordered_map<std::string, ParameterPtr> named_parameters_;
};

class Sequential : public Module {
public:
    Sequential() = default;

    Sequential(std::initializer_list<ModulePtr> modules) {
        for (size_t i = 0; i < modules.size(); ++i) {
            add_module(std::to_string(i), *(modules.begin() + i));
        }
    }

    void add_module(const std::string& name, const ModulePtr& module) {
        register_module(name, module);
        modules_.push_back(module);
    }

    void append(const ModulePtr& module) {
        add_module(std::to_string(modules_.size()), module);
    }

    TensorPtr forward(const TensorPtr& input) override {
        TensorPtr x = input;
        for (auto& module : modules_) {
            x = module->forward(x);
        }
        return x;
    }

protected:
    std::string get_module_name() const override {
        return "Sequential";
    }

private:
    std::vector<ModulePtr> modules_;
};

class ModuleList : public Module {
public:
    ModuleList() = default;

    ModuleList(std::initializer_list<ModulePtr> modules) {
        for (size_t i = 0; i < modules.size(); ++i) {
            append(*(modules.begin() + i));
        }
    }

    void append(const ModulePtr& module) {
        register_module(std::to_string(modules_.size()), module);
        modules_.push_back(module);
    }

    ModulePtr operator[](size_t index) {
        return modules_[index];
    }

    size_t size() const { return modules_.size(); }

    TensorPtr forward(const TensorPtr& input) override {
        throw std::runtime_error("ModuleList has no forward method. Use indexing to access modules.");
    }

protected:
    std::string get_module_name() const override {
        return "ModuleList";
    }

private:
    std::vector<ModulePtr> modules_;
};

}
}
