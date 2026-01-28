#pragma once

#include "../core/tensor.h"
#include "../core/ops.h"
#include "../core/tokenizer_bpe.h"
#include "../optim/optimizer.h"
#include "../optim/adam.h"
#include "gpt2_model.h"
#include <vector>
#include <memory>
#include <string>

namespace ops {

struct GPT2FineTuneConfig {

    GPT2Config model_config;

    bool freeze_embeddings = true;
    bool freeze_lm_head = true;
    bool only_attention = true;

    float learning_rate = 3e-5f;
    int batch_size = 4;
    int num_epochs = 3;
    int max_seq_length = 512;

    bool use_lora = false;
    int lora_rank = 16;
    float lora_alpha = 32.0f;

    GPT2FineTuneConfig() {
        model_config = GPT2Config::GPT2_117M();
        model_config.max_seq_length = max_seq_length;
    }
};

struct TrainingSample {
    std::vector<int> input_ids;
    std::vector<int> labels;
    std::vector<int> attention_mask;

    TrainingSample() = default;
    TrainingSample(const std::vector<int>& inputs, const std::vector<int>& targets)
        : input_ids(inputs), labels(targets) {}
};

class GPT2FineTuneModel {
public:
    GPT2FineTuneModel(const GPT2FineTuneConfig& config);
    ~GPT2FineTuneModel() = default;

    void load_pretrained_weights(const std::string& model_path);
    void save_finetuned_weights(const std::string& output_path) const;

    GradTensorPtr forward(const std::vector<int>& input_ids, bool training = false);
    GradTensorPtr forward_batch(const std::vector<std::vector<int>>& batch_input_ids,
                               bool training = false);

    GradTensorPtr compute_loss(const GradTensorPtr& logits,
                              const std::vector<int>& labels);
    GradTensorPtr compute_batch_loss(const GradTensorPtr& logits,
                                    const std::vector<std::vector<int>>& batch_labels);

    void train(const std::vector<TrainingSample>& train_data,
               const std::vector<TrainingSample>& val_data = {});

    float evaluate(const std::vector<TrainingSample>& eval_data);

    std::vector<int> generate(const std::vector<int>& input_ids,
                             int max_length = 50,
                             float temperature = 1.0f,
                             int top_k = 50);

    void set_tokenizer(std::shared_ptr<Tokenizer> tokenizer) { tokenizer_ = tokenizer; }
    std::vector<int> encode_text(const std::string& text);
    std::string decode_tokens(const std::vector<int>& tokens);

    void print_model_info() const;
    size_t get_total_params() const;
    size_t get_trainable_params() const;

    std::vector<TensorPtr> get_trainable_parameters();
    std::vector<TensorPtr> get_trainable_gradients();

private:
    GPT2FineTuneConfig config_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::shared_ptr<Adam> optimizer_;

    std::shared_ptr<EmbeddingLayer> token_embedding_;
    std::shared_ptr<PositionalEmbeddingLayer> position_embedding_;
    std::vector<std::shared_ptr<AttentionLayer>> attention_layers_;
    std::vector<std::shared_ptr<MLPLayer>> mlp_layers_;

    std::vector<TensorPtr> ln1_weights_, ln1_biases_;
    std::vector<TensorPtr> ln2_weights_, ln2_biases_;
    TensorPtr ln_f_weight_, ln_f_bias_;

    TensorPtr lm_head_weight_;
    TensorPtr lm_head_bias_;

    std::vector<TensorPtr> grad_ln1_weights_, grad_ln1_biases_;
    std::vector<TensorPtr> grad_ln2_weights_, grad_ln2_biases_;
    TensorPtr grad_ln_f_weight_, grad_ln_f_bias_;
    TensorPtr grad_lm_head_weight_, grad_lm_head_bias_;

    void init_model();
    void init_optimizer();
    void freeze_parameters();
    void zero_gradients();

    GradTensorPtr apply_layer_norm(const GradTensorPtr& x,
                                  const TensorPtr& weight,
                                  const TensorPtr& bias);

    GradTensorPtr apply_linear(const GradTensorPtr& x,
                              const TensorPtr& weight,
                              const TensorPtr& bias = nullptr);

    void train_epoch(const std::vector<TrainingSample>& train_data, int epoch);
    float validate_epoch(const std::vector<TrainingSample>& val_data);

    std::vector<std::vector<TrainingSample>> create_batches(
        const std::vector<TrainingSample>& data) const;
    void pad_batch(std::vector<TrainingSample>& batch) const;
};

}
