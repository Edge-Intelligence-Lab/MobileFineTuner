#include "gemma_trainer.h"
#include "../data/wikitext2_dataset.h"
#include "../core/lm_loss.h"
#include "../core/ops.h"
#include "../core/logger.h"
#include "../core/memory_manager.h"
#include "../core/performance_monitor.h"
#include <filesystem>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

namespace ops {

GemmaLoRATrainer::GemmaLoRATrainer(GemmaModel& model,
                                   GemmaLoraInjector& injector,
                                   WikiText2Dataset& train_data,
                                   WikiText2Dataset& eval_data,
                                   const GemmaTrainerConfig& config)
    : model_(model),
      injector_(injector),
      train_data_(train_data),
      eval_data_(eval_data),
      config_(config),
      global_step_(0) {

    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.beta1 = config_.adam_beta1;
    adam_cfg.beta2 = config_.adam_beta2;
    adam_cfg.epsilon = config_.adam_eps;
    adam_cfg.weight_decay = config_.weight_decay;
    adam_cfg.clip_grad_norm = config_.max_grad_norm;

    optimizer_ = std::make_unique<Adam>(adam_cfg);

    std::cout << "[GemmaTrainer] LR=" << config_.learning_rate
              << ", epochs=" << config_.num_epochs
              << ", grad_accum=" << config_.grad_accum_steps << std::endl;
}

float GemmaLoRATrainer::get_lr(int step) {
    int64_t micro_per_epoch = (train_data_.num_sequences() + config_.micro_batch_size - 1) / config_.micro_batch_size;
    int64_t total_updates = (micro_per_epoch * config_.num_epochs + config_.grad_accum_steps - 1) / config_.grad_accum_steps;
    total_updates = std::max<int64_t>(1, total_updates);

    int warmup_steps = static_cast<int>(total_updates * config_.warmup_ratio);

    if (warmup_steps > 0 && step <= warmup_steps) {
        return config_.learning_rate * (static_cast<float>(step) / warmup_steps);
    }

    float progress = static_cast<float>(step - warmup_steps) / std::max<int64_t>(1, total_updates - warmup_steps);
    progress = std::clamp(progress, 0.0f, 1.0f);

    if (config_.lr_scheduler == "cosine") {
        return config_.learning_rate * 0.5f * (1.0f + std::cos(3.14159265f * progress));
    }

    return config_.learning_rate * (1.0f - progress);
}

void GemmaLoRATrainer::clip_gradients() {
    auto params = injector_.get_trainable_params();
    float total_norm = 0.0f;
    for (const auto& param : params) {
        if (!param->grad()) continue;
        const float* g = param->grad()->data<float>();
        for (int64_t i = 0; i < param->grad()->numel(); ++i) {
            total_norm += g[i] * g[i];
        }
    }
    total_norm = std::sqrt(total_norm);
    if (total_norm <= config_.max_grad_norm) return;
    float scale = config_.max_grad_norm / (total_norm + 1e-6f);
    for (const auto& param : params) {
        if (!param->grad()) continue;
        float* g = param->grad()->data<float>();
        for (int64_t i = 0; i < param->grad()->numel(); ++i) g[i] *= scale;
    }
}

float GemmaLoRATrainer::train_step(const Batch& batch) {
    micro_step_counter_++;
    // Reset per-step RSS trackers at the beginning of a new optimizer step window
    if (accum_counter_ == 0) {
        rss_pre_max_ = rss_fwd_max_ = rss_bwd_max_ = rss_opt_max_ = rss_post_max_ = rss_step_max_ = 0.0;
    }
    auto rss_to_mb = []() -> double {
        size_t bytes = MemoryMonitor::get_system_memory_usage();
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };
    // RSS before forward
    rss_pre_max_ = std::max(rss_pre_max_, rss_to_mb());
    if (config_.dump_embedding && !embedding_dump_scheduled_ &&
        micro_step_counter_ == config_.dump_embedding_step) {
        model_.request_embedding_dump(config_.dump_embedding_step, config_.dump_embedding_dir);
        embedding_dump_scheduled_ = true;
    }

    auto logits = model_.forward(batch.input_ids, batch.attention_mask);
    rss_fwd_max_ = std::max(rss_fwd_max_, rss_to_mb());
    auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
    float loss_val = loss->data<float>()[0];

    float scale = 1.0f / static_cast<float>(config_.grad_accum_steps);
    auto scaled_loss = mul(loss, scale);
    scaled_loss->backward();
    rss_bwd_max_ = std::max(rss_bwd_max_, rss_to_mb());

    accum_counter_++;
    accum_loss_ += loss_val;

    if (accum_counter_ < config_.grad_accum_steps) {
        return -1.0f;
    }

    clip_gradients();

    auto params = injector_.get_trainable_params();
    std::vector<TensorPtr> grads;
    grads.reserve(params.size());
    for (const auto& param : params) {
        grads.push_back(param->grad());
    }

    global_step_++;
    float current_lr = get_lr(global_step_);
    optimizer_->set_learning_rate(current_lr);
    optimizer_->step(params, grads);
    rss_opt_max_ = std::max(rss_opt_max_, rss_to_mb());
    for (const auto& param : params) {
        if (param->grad()) param->zero_grad();
    }
    MemoryManager::instance().force_cleanup();
    rss_post_max_ = std::max(rss_post_max_, rss_to_mb());

    accum_counter_ = 0;
    float avg_loss = accum_loss_ / static_cast<float>(config_.grad_accum_steps);
    accum_loss_ = 0.0f;
    // Aggregate step max and peak
    rss_step_max_ = std::max({rss_pre_max_, rss_fwd_max_, rss_bwd_max_, rss_opt_max_, rss_post_max_});
    rss_peak_ = std::max(rss_peak_, rss_step_max_);
    return avg_loss;
}

float GemmaLoRATrainer::evaluate() {
    std::cout << "[GemmaTrainer] Eval started..." << std::endl;
    eval_data_.reset_cursor();
    float total_loss = 0.0f;
    int batches = 0;

    while (true) {
        auto batch = eval_data_.next_batch(config_.micro_batch_size, false);
        if (!batch.input_ids) break;
        auto logits = model_.forward(batch.input_ids, batch.attention_mask);
        auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
        total_loss += loss->data<float>()[0];
        batches++;
    }

    float mean_loss = (batches > 0) ? total_loss / batches : 0.0f;
    float ppl = perplexity_from_loss(mean_loss);
    std::cout << "  Eval Loss: " << mean_loss << " PPL: " << ppl << std::endl;
    MemoryManager::instance().force_cleanup();
    return mean_loss;
}

void GemmaLoRATrainer::train() {
    if (!timing_initialized_) {
        training_start_time_ = std::chrono::steady_clock::now();
        timing_initialized_ = true;
    }
    for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
        std::cout << "\n=== Gemma Epoch " << (epoch + 1) << "/" << config_.num_epochs << " ===" << std::endl;
        train_data_.reset_cursor();
        float epoch_loss = 0.0f;
        int steps = 0;
        bool stop_early = false;

        while (true) {
            auto batch = train_data_.next_batch(config_.micro_batch_size, false);
            if (!batch.input_ids) break;

            float loss = train_step(batch);
            if (loss < 0.0f) {
                continue;
            }

            epoch_loss += loss;
            steps++;

            if (global_step_ % config_.logging_steps == 0) {
                float ppl = perplexity_from_loss(loss);
                auto now_tp = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - training_start_time_).count();
                double elapsed_s = static_cast<double>(elapsed_ms) / 1000.0;
                std::cout << "[Step " << global_step_ << "] Loss=" << loss
                          << " PPL=" << ppl << " LR=" << get_lr(global_step_)
                          << " RSS(pre/fwd/bwd/opt/post)="
                          << rss_pre_max_ << "/" << rss_fwd_max_ << "/"
                          << rss_bwd_max_ << "/" << rss_opt_max_ << "/"
                          << rss_post_max_
                          << " MB | step_max=" << rss_step_max_
                          << " MB | peak=" << rss_peak_ << " MB"
                          << " | time_cum=" << std::fixed << std::setprecision(3) << elapsed_s << "s"
                          << std::endl;
            }

            // Periodic save of LoRA adapter
            if (config_.save_every > 0 && (global_step_ % config_.save_every == 0)) {
                try {
                    std::filesystem::create_directories(config_.output_dir);
                    std::string ckpt = config_.output_dir + "/gemma_lora_step" + std::to_string(global_step_) + ".safetensors";
                    save_lora(ckpt);
                    std::cout << "[Checkpoint] Saved LoRA at step " << global_step_ << " → " << ckpt << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "[Checkpoint] Failed to save at step " << global_step_ << ": " << e.what() << std::endl;
                }
            }

            if (config_.eval_steps > 0 && global_step_ % config_.eval_steps == 0) {
                evaluate();
            }

            if (config_.max_steps > 0 && global_step_ >= config_.max_steps) {
                stop_early = true;
                break;
            }
        }

        float avg_loss = (steps > 0) ? epoch_loss / steps : 0.0f;
        std::cout << "Epoch " << (epoch + 1) << " avg loss: " << avg_loss << std::endl;
        MemoryManager::instance().cleanup_dead_references();
        MemoryManager::instance().clear_unused_memory();

        if (stop_early) {
            std::cout << "[GemmaTrainer] Reached max_steps=" << config_.max_steps << ", stopping early." << std::endl;
            break;
        }
    }

    MemoryManager::instance().force_cleanup();
}

void GemmaLoRATrainer::save_lora(const std::string& path) {
    injector_.save_lora_safetensors(path);
}

}  // namespace ops
