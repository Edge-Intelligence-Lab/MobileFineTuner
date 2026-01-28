/**
 * @file lora_linear.h
 * @brief LoRA-augmented linear layer (modular wrapper)
 */

#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>

namespace ops {

/**
 * @brief LoRA slice (supports submatrix injection, e.g., q/k/v separately)
 */
struct LoRASlice {
    TensorPtr A;     // [in_dim, rank]
    TensorPtr B;     // [rank, out_slice]
    float scale;     // alpha / rank
    int col0;        // Starting column in base W (q=0, k=C, v=2C; otherwise 0)
    int cols;        // Number of columns in slice
    
    LoRASlice(const TensorPtr& a, const TensorPtr& b, float s, int c0 = 0, int c = -1)
        : A(a), B(b), scale(s), col0(c0), cols(c) {}
};

/**
 * @brief LoRA-augmented linear layer
 * 
 * Features:
 * - Training: y = x@W + b + Σ scale_i * (x @ A_i @ B_i)
 * - Inference: LoRA can be merged into base weights
 * - Parameter management: only A/B need gradients; W/b are frozen
 */
class LoRALinear {
public:
    /**
     * @brief Constructor (references base weights, no copy)
     */
    LoRALinear(const TensorPtr& W_base, const TensorPtr& b_base = nullptr)
        : W_(W_base), b_(b_base), merged_(false) {}
    
    /**
     * @brief Attach one LoRA slice (call multiple times, e.g., qkv has 3)
     */
    void attach_lora(const TensorPtr& A, const TensorPtr& B, 
                     float scale, int col0 = 0, int cols = -1);
    
    /**
     * @brief Clear all LoRA slices (remove slices only, keep base)
     */
    void clear_lora();
    
    /**
     * @brief Export/infer: bake ΔW into base (apply over submatrix range)
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
    
    // Debug helper: assign a name so exported A/B carry it
    void set_debug_name(const std::string& name) { debug_name_ = name; }
    std::vector<std::pair<std::string, TensorPtr>> debug_params() const;
    
    /**
     * @brief Enumerate trainable params (returns only A/B)
     */
    std::vector<TensorPtr> trainable_parameters() const;
    
    /**
     * @brief Read-only accessors
     */
    const TensorPtr& W() const { return W_; }
    const TensorPtr& b() const { return b_; }
    const std::vector<LoRASlice>& slices() const { return slices_; }
    bool is_merged() const { return merged_; }

private:
    TensorPtr W_;  // [in_dim, out_dim] (references base, not owning)
    TensorPtr b_;  // [out_dim]
    std::vector<LoRASlice> slices_;
    bool merged_;
    std::string debug_name_;
};

}  // namespace ops
