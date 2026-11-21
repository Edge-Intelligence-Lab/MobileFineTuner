/**
 * @file lora_linear.h
 * @brief LoRA-enhanced linear layer (modular encapsulation)
 */

#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>

namespace ops {

/**
 * @brief LoRA slice (supports sub-matrix injection, e.g., separate injection for q/k/v in QKV)
 */
struct LoRASlice {
    TensorPtr A;     // [in_dim, rank]
    TensorPtr B;     // [rank, out_slice]
    float scale;     // alpha / rank
    int col0;        // Starting column in base W (q=0, k=C, v=2C; 0 for non-qkv)
    int cols;        // Number of slice columns
    
    LoRASlice(const TensorPtr& a, const TensorPtr& b, float s, int c0 = 0, int c = -1)
        : A(a), B(b), scale(s), col0(c0), cols(c) {}
};

/**
 * @brief LoRA-enhanced linear layer
 * 
 * Features:
 * - Training mode: y = x@W + b + Σ scale_i * (x @ A_i @ B_i)
 * - Inference mode: can merge LoRA into base weights
 * - Parameter management: only A/B require gradients, W/b are frozen
 */
class LoRALinear {
public:
    /**
     * @brief Constructor (references base weights, no copying)
     */
    LoRALinear(TensorPtr* W_base_ref, TensorPtr* b_base_ref = nullptr)
        : W_ref_(W_base_ref), b_ref_(b_base_ref), merged_(false) {}
    
    /**
     * @brief Attach a LoRA slice (can be called multiple times, e.g., 3 times for qkv)
     */
    void attach_lora(const TensorPtr& A, const TensorPtr& B, 
                     float scale, int col0 = 0, int cols = -1);
    
    /**
     * @brief Clear all LoRA slices (removes slices only, does not modify base)
     */
    void clear_lora();
    
    /**
     * @brief Export/Inference: merge ΔW into base (add to sub-matrix range)
     */
    void merge_to_base();
    
    /**
     * @brief Restore: subtract ΔW from base
     */
    void unmerge_from_base();
    
    /**
     * @brief Forward: y = x@W + b + Σ scale*(x@A@B)
     */
    TensorPtr forward(const TensorPtr& x) const;
    
    // Debug helper: name and export LoRA A/B
    void set_debug_name(const std::string& name) { debug_name_ = name; }
    std::vector<std::pair<std::string, TensorPtr>> debug_params() const;

    /**
     * @brief Enumerate trainable parameters (returns A/B only)
     */
    std::vector<TensorPtr> trainable_parameters() const;
    
    /**
     * @brief Read-only access
     */
    const TensorPtr& W() const { return base_W(); }
    const TensorPtr& b() const { return base_b(); }
    const std::vector<LoRASlice>& slices() const { return slices_; }
    bool is_merged() const { return merged_; }

private:
    TensorPtr* W_ref_ = nullptr;  // Pointer to externally held base weights
    TensorPtr* b_ref_ = nullptr;  // Pointer to externally held bias
    TensorPtr b_cache_;           // Bias cache for null-safe access
    std::vector<LoRASlice> slices_;
    bool merged_;
    std::string debug_name_;

    const TensorPtr& base_W() const;
    const TensorPtr& base_b() const;
};

}  // namespace ops
