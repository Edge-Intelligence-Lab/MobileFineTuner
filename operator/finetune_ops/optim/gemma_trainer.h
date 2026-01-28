#pragma once

#include "../core/tensor.h"
#include "../graph/gemma_model.h"
#include "../graph/gemma_lora_injector.h"
#include "adam.h"
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace ops {
class WikiText2Dataset;
struct Batch;
}

namespace ops {

struct GemmaTrainerConfig {
    float learning_rate = 2e-4f;
    float weight_decay = 0.0f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-8f;

    int num_epochs = 3;
    float max_grad_norm = 1.0f;

    std::string lr_scheduler = "linear";
    float warmup_ratio = 0.03f;

    int logging_steps = 10;
    int eval_steps = 100;
    std::string output_dir = "./gemma_lora_ckpt";
    int max_steps = -1;
    int micro_batch_size = 1;
    int grad_accum_steps = 1;
    int save_every = 0;          // 0=only final save; >0 save LoRA every N optimizer steps
    bool dump_embedding = false;
    int dump_embedding_step = 1;
    std::string dump_embedding_dir = "./debug";
};

class GemmaLoRATrainer {
public:
    GemmaLoRATrainer(GemmaModel& model,
                     GemmaLoraInjector& injector,
                     WikiText2Dataset& train_data,
                     WikiText2Dataset& eval_data,
                     const GemmaTrainerConfig& config);

    void train();
    float train_step(const Batch& batch);
    float evaluate();
    void save_lora(const std::string& path);

private:
    GemmaModel& model_;
    GemmaLoraInjector& injector_;
    WikiText2Dataset& train_data_;
    WikiText2Dataset& eval_data_;
    GemmaTrainerConfig config_;

    std::unique_ptr<Adam> optimizer_;
    int global_step_;
    int accum_counter_ = 0;
    float accum_loss_ = 0.0f;
    int64_t micro_step_counter_ = 0;
    bool embedding_dump_scheduled_ = false;
    // RSS tracking (per optimizer step)
    double rss_pre_max_ = 0.0;
    double rss_fwd_max_ = 0.0;
    double rss_bwd_max_ = 0.0;
    double rss_opt_max_ = 0.0;
    double rss_post_max_ = 0.0;
    double rss_step_max_ = 0.0;
    double rss_peak_ = 0.0;

    // Cumulative timing
    std::chrono::time_point<std::chrono::steady_clock> training_start_time_;
    bool timing_initialized_ = false;

    float get_lr(int step);
    void clip_gradients();
};

}  // namespace ops
