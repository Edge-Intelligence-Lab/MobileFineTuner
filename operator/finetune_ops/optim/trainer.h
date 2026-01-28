/**
 * @file trainer.h
 * @brief LoRA fine-tuning trainer (minimal loop)
 */

#pragma once

#include "../core/tensor.h"
#include "../graph/gpt2_model.h"
#include "../graph/lora_injector.h"
#include "adam.h"
#include <string>
#include <vector>
#include <memory>

namespace ops {
    // Forward declarations
    class WikiText2Dataset;
    struct Batch;
}

namespace ops {

struct TrainerConfig {
    // Optimizer
    float learning_rate = 2e-4f;
    float weight_decay = 0.01f;      // AdamW decoupled
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-8f;
    
    // Training
    int num_epochs = 3;
    int gradient_accumulation_steps = 1;
    float max_grad_norm = 1.0f;      // clip_grad_norm
    
    // Learning rate schedule
    std::string lr_scheduler = "linear";  // "linear" or "cosine"
    float warmup_ratio = 0.03f;
    
    // Logging
    int logging_steps = 10;
    int eval_steps = 100;
    std::string output_dir = "./lora_checkpoints";
};

class LoRATrainer {
public:
    LoRATrainer(GPT2Model& model, 
                LoraInjector& lora,
                WikiText2Dataset& train_data,
                WikiText2Dataset& eval_data,
                const TrainerConfig& config);
    
    /**
     * @brief Run full training
     */
    void train();
    
    /**
     * @brief Single training step (for debugging)
     */
    float train_step(const Batch& batch);
    
    /**
     * @brief Evaluation
     */
    float evaluate();
    
    /**
     * @brief Save LoRA weights
     */
    void save_lora(const std::string& path);
    
private:
    GPT2Model& model_;
    LoraInjector& lora_;
    WikiText2Dataset& train_data_;
    WikiText2Dataset& eval_data_;
    TrainerConfig config_;
    
    std::unique_ptr<Adam> optimizer_;
    int global_step_;
    
    float get_lr(int step);  // LR scheduler
    void clip_gradients();
};

}  // namespace ops

