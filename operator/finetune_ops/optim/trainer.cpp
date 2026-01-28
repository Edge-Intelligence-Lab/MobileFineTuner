/**
 * @file trainer.cpp
 * @brief LoRA fine-tuning trainer implementation
 */

#include "trainer.h"
#include "../data/wikitext2_dataset.h"
#include "../core/lm_loss.h"
#include "../core/logger.h"
#include "../core/performance_monitor.h"
#include "../core/memory_manager.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace ops {

LoRATrainer::LoRATrainer(GPT2Model& model,
                         LoraInjector& lora,
                         WikiText2Dataset& train_data,
                         WikiText2Dataset& eval_data,
                         const TrainerConfig& config)
    : model_(model), lora_(lora), train_data_(train_data), eval_data_(eval_data),
      config_(config), global_step_(0) {
    
    // Initialize optimizer (optimize LoRA parameters only)
    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.beta1 = config_.adam_beta1;
    adam_cfg.beta2 = config_.adam_beta2;
    adam_cfg.epsilon = config_.adam_eps;
    adam_cfg.weight_decay = 0.0f;  // LoRA params usually skip weight decay
    adam_cfg.clip_grad_norm = config_.max_grad_norm;
    
    optimizer_ = std::make_unique<Adam>(adam_cfg);
    
    std::cout << "[Trainer] Initialized with:" << std::endl;
    std::cout << "  LR: " << config_.learning_rate << std::endl;
    std::cout << "  Epochs: " << config_.num_epochs << std::endl;
    std::cout << "  Grad accum steps: " << config_.gradient_accumulation_steps << std::endl;
    std::cout << "  Max grad norm: " << config_.max_grad_norm << std::endl;
}

float LoRATrainer::get_lr(int step) {
    // Compute total steps
    int64_t total_steps = train_data_.num_sequences() / config_.gradient_accumulation_steps * config_.num_epochs;
    int warmup_steps = static_cast<int>(total_steps * config_.warmup_ratio);
    
    if (step < warmup_steps) {
        // Linear warmup
        return config_.learning_rate * (static_cast<float>(step) / warmup_steps);
    } else {
        // Linear decay or cosine
        if (config_.lr_scheduler == "linear") {
            float progress = static_cast<float>(step - warmup_steps) / (total_steps - warmup_steps);
            return config_.learning_rate * (1.0f - progress);
        } else if (config_.lr_scheduler == "cosine") {
            float progress = static_cast<float>(step - warmup_steps) / (total_steps - warmup_steps);
            return config_.learning_rate * 0.5f * (1.0f + std::cos(3.14159265f * progress));
        } else {
            return config_.learning_rate;
        }
    }
}

void LoRATrainer::clip_gradients() {
    // Collect all LoRA parameter gradients
    auto lora_params = lora_.get_trainable_params();
    
    // Compute global gradient norm
    float total_norm = 0.0f;
    for (const auto& param : lora_params) {
        if (!param->grad()) continue;
        const float* grad_data = param->grad()->data<float>();
        for (int64_t i = 0; i < param->grad()->numel(); ++i) {
            total_norm += grad_data[i] * grad_data[i];
        }
    }
    total_norm = std::sqrt(total_norm);
    
    // Scale all gradients if above threshold
    if (total_norm > config_.max_grad_norm) {
        float scale = config_.max_grad_norm / (total_norm + 1e-6f);
        for (const auto& param : lora_params) {
            if (!param->grad()) continue;
            float* grad_data = param->grad()->data<float>();
            for (int64_t i = 0; i < param->grad()->numel(); ++i) {
                grad_data[i] *= scale;
            }
        }
    }
}

float LoRATrainer::train_step(const Batch& batch) {
    // 1. Forward
    auto logits = model_.forward(batch.input_ids, batch.attention_mask);
    
    // 2. Loss
    auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
    float loss_val = loss->data<float>()[0];
    
    // 3. Backward
    loss->backward();
    
    // 4. Clip gradients
    clip_gradients();
    
    // 5. Optimizer step
    auto lora_params = lora_.get_trainable_params();
    std::vector<TensorPtr> grads;
    for (const auto& param : lora_params) {
        grads.push_back(param->grad());
    }
    optimizer_->step(lora_params, grads);
    
    // 6. Zero grad
    for (const auto& param : lora_params) {
        param->zero_grad();
    }
    
    global_step_++;
    
    // 7. Force memory cleanup each step to avoid long-term RSS growth
    MemoryManager::instance().force_cleanup();
    
    return loss_val;
}

float LoRATrainer::evaluate() {
    std::cout << "\n[Eval] Running evaluation..." << std::endl;
    
    eval_data_.reset_cursor();
    float total_loss = 0.0f;
    int num_batches = 0;
    
    // Disable dropout during evaluation (TODO: add mode switch)
    while (true) {
        auto batch = eval_data_.next_batch(config_.gradient_accumulation_steps, false);
        if (!batch.input_ids) break;  // no more data
        
        // Forward only
        auto logits = model_.forward(batch.input_ids, batch.attention_mask);
        auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
        
        total_loss += loss->data<float>()[0];
        num_batches++;
        
        // 🔧 During eval, clean memory after each batch to avoid buildup
        if (num_batches % 5 == 0) {
            MemoryManager::instance().cleanup_dead_references();
            MemoryManager::instance().clear_unused_memory();
        }
    }
    
    // Force cleanup after evaluation
    MemoryManager::instance().force_cleanup();
    
    float mean_loss = (num_batches > 0) ? (total_loss / num_batches) : 0.0f;
    float ppl = perplexity_from_loss(mean_loss);
    
    std::cout << "  Eval Loss: " << mean_loss << std::endl;
    std::cout << "  Perplexity: " << ppl << std::endl;
    
    return mean_loss;
}

void LoRATrainer::train() {
    std::cout << "\n========== Training Start ==========" << std::endl;
    
    // Print initial memory state
    auto initial_memory = get_current_memory_snapshot();
    std::cout << "\n📊 Initial Memory State:" << std::endl;
    initial_memory.print();
    
    for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << config_.num_epochs << " ---" << std::endl;
        
        train_data_.reset_cursor();
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        while (true) {
            auto batch = train_data_.next_batch(config_.gradient_accumulation_steps, false);
            if (!batch.input_ids) break;  // end of epoch
            
            float loss = train_step(batch);
            epoch_loss += loss;
            num_batches++;
            
            // Logging
            if (global_step_ % config_.logging_steps == 0) {
                float current_lr = get_lr(global_step_);
                float ppl = perplexity_from_loss(loss);
                std::cout << "[Step " << global_step_ << "] "
                          << "Loss: " << loss << ", "
                          << "PPL: " << ppl << ", "
                          << "LR: " << current_lr << std::endl;
            }
            
            // Evaluation
            if (global_step_ % config_.eval_steps == 0) {
                evaluate();  // Run evaluation
                train_data_.reset_cursor();  // Reset training data iterator
            }
            
            // Update learning rate
            float new_lr = get_lr(global_step_);
            optimizer_->set_learning_rate(new_lr);
        }
        
        float mean_loss = (num_batches > 0) ? (epoch_loss / num_batches) : 0.0f;
        std::cout << "Epoch " << (epoch + 1) << " finished, Mean Loss: " << mean_loss << std::endl;
        
        // Force cleanup at end of epoch
        MemoryManager::instance().force_cleanup();
        
        // Print current memory state
        auto current_memory = get_current_memory_snapshot();
        std::cout << "\n📊 Memory after Epoch " << (epoch + 1) << ":" << std::endl;
        current_memory.print();
    }
    
    std::cout << "\n========== Training Finished ==========" << std::endl;
    
    // Print final memory stats and optimization tips
    auto final_memory = get_current_memory_snapshot();
    final_memory.print();
    print_memory_optimization_tips(final_memory);
}

void LoRATrainer::save_lora(const std::string& path) {
    std::cout << "[Trainer] Saving LoRA weights to: " << path << std::endl;
    // TODO: implement safetensors save
    // LoraSaver::save(lora_, path);
}

}  // namespace ops

