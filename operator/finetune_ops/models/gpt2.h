/**
 * @file gpt2.h
 * @brief GPT-2 model implementsation using nn modules
 * 
 * This file providess a complete GPT-2 implementsation built on top
 * of the nn module system. It includes the full transforer architecture
 * with embeddings, attention blocks, and language model head.
 */

#pragma once

#include "../nn/module.h"
#include "../nn/layers.h"
#include "../functional/functional.h"
#include <memory>
#include <string>
#include <fstream>

namespace ops {
/**
 * @brief Model implementsations namespace
 * 
 * Contains complete model implementsations built using the nn module system.
 * Models here provides high-level interfaces for training and inference.
 */
namespace models {

/**
 * @brief Configuration structure for GPT-2 model (nn version)
 * 
 * Contains hyperparameters for the GPT-2 model built with nn modules.
 * This version is separate from the transforer/gpt2_components.h version
 * and is designed for use with the nn module system.
 */
struct GPT2Config {
    int vocab_size = 50257;   /**< Vocabulary size */
    int n_positions = 1024;   /**< Maximum sequence length */
    int n_embd = 768;         /**< Embedding dimension */
    int n_layer = 12;         /**< Number of transforer layers */
    int n_head = 12;          /**< Number of attention heads */
    int n_inner = 3072;       /**< MLP intermediate dimension */
    float dropout = 0.1;
    float layer_norm_epsilon = 1e-5;
    std::string activation_function = "gelu";
    bool use_cache = true;
    int pad_token_id = 50256;
    int eos_token_id = 50256;
    int bos_token_id = 50256;

    static GPT2Config from_pretrained(const std::string& model_name) {
        GPT2Config config;

        if (model_name == "gpt2" || model_name == "gpt2-small") {
            config.n_embd = 768;
            config.n_layer = 12;
            config.n_head = 12;
            config.n_inner = 3072;
        } else if (model_name == "gpt2-medium") {
            config.n_embd = 1024;
            config.n_layer = 24;
            config.n_head = 16;
            config.n_inner = 4096;
        } else if (model_name == "gpt2-large") {
            config.n_embd = 1280;
            config.n_layer = 36;
            config.n_head = 20;
            config.n_inner = 5120;
        } else if (model_name == "gpt2-xl") {
            config.n_embd = 1600;
            config.n_layer = 48;
            config.n_head = 25;
            config.n_inner = 6400;
        }

        return config;
    }

    void save_json(const std::string& path) const {
        std::ofstream file(path);
        file << "{\n";
        file << "  \"vocab_size\": " << vocab_size << ",\n";
        file << "  \"n_positions\": " << n_positions << ",\n";
        file << "  \"n_embd\": " << n_embd << ",\n";
        file << "  \"n_layer\": " << n_layer << ",\n";
        file << "  \"n_head\": " << n_head << ",\n";
        file << "  \"n_inner\": " << n_inner << ",\n";
        file << "  \"dropout\": " << dropout << ",\n";
        file << "  \"layer_norm_epsilon\": " << layer_norm_epsilon << ",\n";
        file << "  \"activation_function\": \"" << activation_function << "\",\n";
        file << "  \"use_cache\": " << (use_cache ? "true" : "false") << ",\n";
        file << "  \"pad_token_id\": " << pad_token_id << ",\n";
        file << "  \"eos_token_id\": " << eos_token_id << ",\n";
        file << "  \"bos_token_id\": " << bos_token_id << "\n";
        file << "}\n";
        file.close();
    }
};

class GPT2LMHead : public nn::Module {
public:
    GPT2LMHead(const GPT2Config& config) {

        lm_head_ = std::make_shared<nn::Linear>(config.n_embd, config.vocab_size, false);
        register_module("lm_head", lm_head_);
    }

    TensorPtr forward(const TensorPtr& hidden_states) override {
        return lm_head_->forward(hidden_states);
    }

protected:
    std::string get_module_name() const override {
        return "GPT2LMHead";
    }

private:
    std::shared_ptr<nn::Linear> lm_head_;
};

class GPT2Model : public nn::Module {
public:
    GPT2Model(const GPT2Config& config) : config_(config) {

        wte_ = std::make_shared<nn::Embedding>(config.vocab_size, config.n_embd);
        register_module("wte", wte_);

        wpe_ = std::make_shared<nn::Embedding>(config.n_positions, config.n_embd);
        register_module("wpe", wpe_);

        drop_ = std::make_shared<nn::Dropout>(config.dropout);
        register_module("drop", drop_);

        h_ = std::make_shared<nn::ModuleList>();
        for (int i = 0; i < config.n_layer; ++i) {
            auto block = std::make_shared<nn::TransforerBlock>(
                config.n_embd, config.n_head, config.n_inner,
                config.dropout, config.activation_function, true);
            h_->append(block);
        }
        register_module("h", h_);

        ln_f_ = std::make_shared<nn::LayerNorm>(config.n_embd, config.layer_norm_epsilon);
        register_module("ln_f", ln_f_);
    }

    TensorPtr forward(const TensorPtr& input_ids,
                     const TensorPtr& position_ids = nullptr,
                     const TensorPtr& attention_mask = nullptr) override {

        auto batch_size = input_ids->size(0);
        auto seq_length = input_ids->size(1);

        TensorPtr pos_ids = position_ids;
        if (!pos_ids) {

            std::vector<int64_t> pos_data;
            for (int64_t i = 0; i < seq_length; ++i) {
                pos_data.push_back(i);
            }

            pos_ids = tensor(std::vector<float>(pos_data.begin(), pos_data.end()));
            pos_ids = pos_ids->unsqueeze(0)->expand({batch_size, seq_length});
        }

        auto inputs_embeds = wte_->forward(input_ids);
        auto position_embeds = wpe_->forward(pos_ids);
        auto hidden_states = functional::add(inputs_embeds, position_embeds);

        hidden_states = drop_->forward(hidden_states);

        TensorPtr causal_mask = nullptr;
        if (!attention_mask) {
            causal_mask = functional::causal_mask(seq_length, hidden_states->dtype(), hidden_states->device());
        }

        for (int i = 0; i < config_.n_layer; ++i) {
            auto block = std::dynamic_pointer_cast<nn::TransforerBlock>((*h_)[i]);
            hidden_states = block->forward(hidden_states, causal_mask);
        }

        hidden_states = ln_f_->forward(hidden_states);

        return hidden_states;
    }

    const GPT2Config& config() const { return config_; }

protected:
    std::string get_module_name() const override {
        return "GPT2Model";
    }

private:
    GPT2Config config_;
    std::shared_ptr<nn::Embedding> wte_;
    std::shared_ptr<nn::Embedding> wpe_;
    std::shared_ptr<nn::Dropout> drop_;
    std::shared_ptr<nn::ModuleList> h_;
    std::shared_ptr<nn::LayerNorm> ln_f_;
};

class GPT2LMHeadModel : public nn::Module {
public:
    GPT2LMHeadModel(const GPT2Config& config) : config_(config) {

        transforer_ = std::make_shared<GPT2Model>(config);
        register_module("transforer", transforer_);

        lm_head_ = std::make_shared<GPT2LMHead>(config);
        register_module("lm_head", lm_head_);
    }

    TensorPtr forward(const TensorPtr& input_ids,
                     const TensorPtr& position_ids = nullptr,
                     const TensorPtr& attention_mask = nullptr,
                     const TensorPtr& labels = nullptr) override {

        auto hidden_states = transforer_->forward(input_ids, position_ids, attention_mask);

        auto logits = lm_head_->forward(hidden_states);

        if (labels) {

            auto shift_logits = logits->slice(1, 0, -1);
            auto shift_labels = labels->slice(1, 1, -1);

            auto loss = functional::cross_entropy_loss(
                shift_logits->reshape({-1, config_.vocab_size}),
                shift_labels->reshape({-1}),
                nullptr, -100, "mean"
            );

            return logits;
        }

        return logits;
    }

    TensorPtr generate(const TensorPtr& input_ids,
                      int max_length = 50,
                      float temperature = 1.0,
                      int top_k = 50,
                      float top_p = 0.9,
                      bool do_sample = true) {

        eval();

        auto current_ids = input_ids->clone();
        auto batch_size = input_ids->size(0);
        auto current_length = input_ids->size(1);

        for (int i = current_length; i < max_length; ++i) {

            auto logits = forward(current_ids);

            auto next_token_logits = logits->slice(1, -1, -1)->squeeze(1);

            if (temperature != 1.0) {
                next_token_logits = functional::div(next_token_logits, temperature);
            }

            TensorPtr next_tokens;
            if (do_sample) {

                auto probs = functional::softmax(next_token_logits, -1);

                next_tokens = functional::max(probs, -1);
            } else {

                next_tokens = functional::max(next_token_logits, -1);
            }

            current_ids = functional::cat({current_ids, next_tokens->unsqueeze(1)}, 1);

        }

        return current_ids;
    }

    static std::shared_ptr<GPT2LMHeadModel> from_pretrained(const std::string& model_name) {
        auto config = GPT2Config::from_pretrained(model_name);
        auto model = std::make_shared<GPT2LMHeadModel>(config);

        return model;
    }

    void save_pretrained(const std::string& save_directory) {

        config_.save_json(save_directory + "/config.json");

        save(save_directory + "/pytorch_model.bin");
    }

    const GPT2Config& config() const { return config_; }

protected:
    std::string get_module_name() const override {
        return "GPT2LMHeadModel";
    }

private:
    GPT2Config config_;
    std::shared_ptr<GPT2Model> transforer_;
    std::shared_ptr<GPT2LMHead> lm_head_;

    void load_pretrained_weights(const std::string& model_name) {

    }
};

}
}
