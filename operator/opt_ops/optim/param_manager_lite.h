/**
 * @file param_manager_lite.h  
 * @brief lightweightparametermanager - as MobileOptimizer providesparameterID/namemapping
 * 
 * [Documentation in English - see separate docs]
 * - parameterregisterwithIDallocate
 * - parameternamewithsizerecord
 * [Documentation available in English]
 * 
 * [Documentation in English - see separate docs]
 */

#pragma once

#include "../core/tensor.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

namespace ops {
namespace optim {

/**
 * [Documentation available in English]
 */
struct ParameterMetadata {
    size_t param_id;
    std::string param_name;
    size_t param_size;      // [Translated]
    std::string group_name;
    bool requires_grad;
    
    // [Translated comment removed - see documentation]
    TensorPtr param_tensor;      // [Translated]
    
        // [Translated]
    ParameterMetadata() 
        : param_id(0), param_name(""), param_size(0),
          group_name("default"), requires_grad(false), param_tensor(nullptr) {}
    
    ParameterMetadata(size_t id, const std::string& name, size_t size, 
                     const std::string& group = "default")
        : param_id(id), param_name(name), param_size(size),
          group_name(group), requires_grad(true), param_tensor(nullptr) {}
};

/**
 * @brief lightweightparametermanager
 * 
 * [Documentation available in English]
 * 1. asparameterallocateuniqueID
 * [Documentation available in English]
 * 3. providesparameterqueryinterface
 * [Documentation available in English]
 */
class ParameterManagerLite {
private:
    std::unordered_map<size_t, ParameterMetadata> param_registry_;
    std::unordered_map<std::string, size_t> name_to_id_;
    std::unordered_map<std::string, std::vector<size_t>> groups_;
    
    size_t next_param_id_ = 0;

public:
    ParameterManagerLite() = default;
    
    /**
     * @brief registerparameter
     * @param param parametertensor
     * @param name parametername
     * [Documentation available in English]
     * @return allocateparameterID
     */
    size_t register_parameter(const TensorPtr& param, 
                             const std::string& name,
                             const std::string& group_name = "default") {
        size_t param_id = next_param_id_++;
        size_t param_size = param ? param->numel() : 0;
        
        ParameterMetadata metadata(param_id, name, param_size, group_name);
        metadata.param_tensor = param;
        metadata.requires_grad = param ? param->requires_grad() : false;
        
        param_registry_[param_id] = metadata;
        name_to_id_[name] = param_id;
        groups_[group_name].push_back(param_id);
        
        return param_id;
    }
    
    /**
     * [Documentation available in English]
     * @param params parameterlist
     * [Documentation available in English]
     * [Documentation available in English]
     * @return parameterIDlist
     */
    std::vector<size_t> register_parameters(const std::vector<TensorPtr>& params,
                                           const std::string& name_prefix = "param",
                                           const std::string& group_name = "default") {
        std::vector<size_t> ids;
        ids.reserve(params.size());
        
        for (size_t i = 0; i < params.size(); ++i) {
            std::string name = name_prefix + "_" + std::to_string(i);
            ids.push_back(register_parameter(params[i], name, group_name));
        }
        
        return ids;
    }
    
    /**
     * [Documentation available in English]
     */
    const ParameterMetadata* get_metadata(size_t param_id) const {
        auto it = param_registry_.find(param_id);
        return (it != param_registry_.end()) ? &it->second : nullptr;
    }
    
    /**
     * @brief according tonameacquireparameterID
     */
    size_t get_param_id(const std::string& name) const {
        auto it = name_to_id_.find(name);
        if (it == name_to_id_.end()) {
            throw std::runtime_error("Parameter not found: " + name);
        }
        return it->second;
    }
    
    /**
     * [Documentation available in English]
     */
    std::vector<size_t> get_group_params(const std::string& group_name) const {
        auto it = groups_.find(group_name);
        return (it != groups_.end()) ? it->second : std::vector<size_t>();
    }
    
    /**
     * @brief acquireparametertensor
     */
    TensorPtr get_parameter(size_t param_id) const {
        auto metadata = get_metadata(param_id);
        return metadata ? metadata->param_tensor : nullptr;
    }
    
    /**
     * [Documentation available in English]
     */
    size_t num_parameters() const { return param_registry_.size(); }
    
    /**
     * [Documentation available in English]
     */
    size_t total_parameter_count() const {
        size_t total = 0;
        for (const auto& [id, meta] : param_registry_) {
            total += meta.param_size;
        }
        return total;
    }
    
    /**
     * @brief clearallregister
     */
    void clear() {
        param_registry_.clear();
        name_to_id_.clear();
        groups_.clear();
        next_param_id_ = 0;
    }
};

// [Translated comment removed - see documentation]
// [Translated]
// [Translated comment removed - see documentation]
class MobileParameterManager : public ParameterManagerLite {
public:
    using ParameterManagerLite::ParameterManagerLite;
};

} // namespace optim
} // namespace ops

