#pragma once
#include "../../opt_ops/sharding/parameter_sharder.h"
#include "../nn/lora_linear.h"

namespace ops {

// Owns backing files and model TensorPtr slots, not arbitrary external aliases.
class FrozenModelOffload {
public:
    using Group = std::vector<TensorPtr*>;
    FrozenModelOffload(const sharding::ShardConfig& config, const std::vector<Group>& groups)
        : groups_(groups) {
        for (const auto& group : groups_) {
            size_t bytes = 0;
            for (auto slot : group) {
                if (!slot || !*slot || (*slot)->requires_grad() || (*slot)->dtype()!=kFloat32 || !(*slot)->device().is_cpu())
                    throw std::invalid_argument("model offload requires fully loaded, frozen CPU FP32 weights");
                bytes += static_cast<size_t>((*slot)->numel()) * sizeof(float);
            }
            if (bytes>config.max_resident_bytes)
                throw std::invalid_argument("offload budget must hold the largest complete model block or head group");
        }
        sharder_ = std::make_unique<sharding::ParameterSharder>(config);
        std::vector<TensorPtr> original;
        for(const auto& group:groups_) for(auto slot:group) original.push_back(*slot);
        try {
            for (size_t g=0;g<groups_.size();++g) for(size_t w=0;w<groups_[g].size();++w)
                sharder_->register_parameter(key(g,w),*groups_[g][w],true,groups_[g][w]);
        } catch (...) {
            size_t n=0;
            for(const auto& group:groups_) for(auto slot:group) *slot=original[n++];
            throw;
        }
    }
    void load(size_t group) {
        for(size_t w=0;w<groups_.at(group).size();++w) sharder_->require(key(group,w));
    }
    void offload_all() { sharder_->offload_all(); }
    sharding::ShardStats stats() const { return sharder_->stats(); }
private:
    static std::string key(size_t g,size_t w) { return "group_"+std::to_string(g)+"_weight_"+std::to_string(w); }
    std::vector<Group> groups_;
    std::unique_ptr<sharding::ParameterSharder> sharder_;
};

// LoRALinear normally owns a shared base-weight handle. Drop that extra handle
// after a managed block so disk offload can actually release the model's storage.
class OffloadLinearLease {
public:
    explicit OffloadLinearLease(bool enabled) : enabled_(enabled) {}
    void bind(LoRALinear* linear,const TensorPtr& weight,const TensorPtr& bias=nullptr) {
        if(enabled_ && linear) { linear->set_base_weights(weight,bias); linears_.push_back(linear); }
    }
    ~OffloadLinearLease() { for(auto* linear:linears_) linear->set_base_weights(nullptr); }
private:
    bool enabled_;
    std::vector<LoRALinear*> linears_;
};
} // namespace ops
