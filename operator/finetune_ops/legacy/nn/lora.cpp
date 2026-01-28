#include "lora.h"
#include "lora_ops.h"
#include "../core/tensor.h"
#include "../core/ops.h"
#include <fstream>
#include <stdexcept>

namespace ops {

LoRALayer::LoRALayer(const LoRAConfig& config)
    : BaseLayer(config), lora_config_(config) {
    init_parameters();
}

void LoRALayer::init_parameters() {

    std::vector<int> lora_A_shape = {lora_config_.rank, lora_config_.in_features};
    std::vector<int> lora_B_shape = {lora_config_.out_features, lora_config_.rank};

    lora_A_ = std::make_shared<Tensor>(lora_A_shape);
    lora_B_ = std::make_shared<Tensor>(lora_B_shape);

    lora_ops::init_lora_weights(*lora_A_, *lora_B_,
                                lora_config_.in_features,
                                lora_config_.out_features,
                                lora_config_.rank);

    if (lora_config_.use_bias) {
        std::vector<int> bias_shape = {lora_config_.out_features};
        bias_ = std::make_shared<Tensor>(bias_shape);
        bias_->zero_();
    }

    grad_lora_A_ = std::make_shared<Tensor>(lora_A_shape);
    grad_lora_B_ = std::make_shared<Tensor>(lora_B_shape);
    grad_lora_A_->zero_();
    grad_lora_B_->zero_();

    if (lora_config_.use_bias) {
        std::vector<int> bias_shape = {lora_config_.out_features};
        grad_bias_ = std::make_shared<Tensor>(bias_shape);
        grad_bias_->zero_();
    }
}

TensorPtr LoRALayer::forward(const Tensor& input, bool training) {

    last_input_ = std::make_shared<Tensor>(input);

    Tensor default_bias({lora_config_.out_features});
    default_bias.zero_();

    auto output = lora_ops::forward(input, *lora_A_, *lora_B_,
                                   bias_ ? *bias_ : default_bias,
                                   lora_config_.to_ops_config(),
                                   training);

    auto input_shape = input.shape();
    int batch_size = input_shape[0];
    int in_features = input_shape[1];

    auto lora_A_T = lora_ops::transpose(*lora_A_, 0, 1);
    last_hidden_ = lora_ops::batch_matmul(input, *lora_A_T,
                                         batch_size, in_features,
                                         lora_config_.rank, in_features);

    return output;
}

TensorPtr LoRALayer::backward(const Tensor& grad_output) {
    if (!last_input_ || !last_hidden_) {
        throw std::runtime_error("Forward pass must be called before backward pass");
    }

    auto grads = lora_ops::backward(grad_output, *last_input_,
                                   *lora_A_, *lora_B_, *last_hidden_,
                                   lora_config_.to_ops_config());

    if (grads.size() >= 4) {

        if (grads[1]) {
            grad_lora_A_->data() = grads[1]->data();
        }
        if (grads[2]) {
            grad_lora_B_->data() = grads[2]->data();
        }
        if (grads[3] && grad_bias_) {
            grad_bias_->data() = grads[3]->data();
        }

        return grads[0];
    }

    return nullptr;
}

std::vector<TensorPtr> LoRALayer::get_parameters() {
    std::vector<TensorPtr> params;
    params.push_back(lora_A_);
    params.push_back(lora_B_);
    if (bias_) {
        params.push_back(bias_);
    }
    return params;
}

std::vector<TensorPtr> LoRALayer::get_gradients() {
    std::vector<TensorPtr> grads;
    grads.push_back(grad_lora_A_);
    grads.push_back(grad_lora_B_);
    if (grad_bias_) {
        grads.push_back(grad_bias_);
    }
    return grads;
}

void LoRALayer::update_parameters(const std::vector<TensorPtr>& updates) {
    if (updates.size() >= 2) {
        lora_A_->data() = updates[0]->data();
        lora_B_->data() = updates[1]->data();

        if (updates.size() >= 3 && bias_) {
            bias_->data() = updates[2]->data();
        }
    }
}

void LoRALayer::save_weights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    int rank = lora_config_.rank;
    int in_features = lora_config_.in_features;
    int out_features = lora_config_.out_features;
    bool use_bias = lora_config_.use_bias;

    file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    file.write(reinterpret_cast<const char*>(&in_features), sizeof(in_features));
    file.write(reinterpret_cast<const char*>(&out_features), sizeof(out_features));
    file.write(reinterpret_cast<const char*>(&use_bias), sizeof(use_bias));

    lora_A_->save(path + ".lora_A");
    lora_B_->save(path + ".lora_B");
    if (bias_) {
        bias_->save(path + ".bias");
    }
}

void LoRALayer::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }

    int rank, in_features, out_features;
    bool use_bias;

    file.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    file.read(reinterpret_cast<char*>(&in_features), sizeof(in_features));
    file.read(reinterpret_cast<char*>(&out_features), sizeof(out_features));
    file.read(reinterpret_cast<char*>(&use_bias), sizeof(use_bias));

    if (rank != lora_config_.rank ||
        in_features != lora_config_.in_features ||
        out_features != lora_config_.out_features ||
        use_bias != lora_config_.use_bias) {
        throw std::runtime_error("Configuration mismatch when loading weights");
    }

    *lora_A_ = Tensor::load(path + ".lora_A");
    *lora_B_ = Tensor::load(path + ".lora_B");
    if (bias_) {
        *bias_ = Tensor::load(path + ".bias");
    }
}

size_t LoRALayer::get_param_count() const {
    size_t count = lora_A_->size() + lora_B_->size();
    if (bias_) {
        count += bias_->size();
    }
    return count;
}

TensorPtr LoRALayer::forward_lora(const Tensor& input, bool training) {
    return forward(input, training);
}

TensorPtr LoRALayer::backward_lora(const Tensor& grad_output) {
    return backward(grad_output);
}

void LoRALayer::merge_with_base_weights(const Tensor& base_weights) {
    (void)base_weights;

    if (!lora_A_ || !lora_B_) {
        throw std::runtime_error("LoRA weights not initialized");
    }

    auto lora_A_T = lora_ops::transpose(*lora_A_, 0, 1);
    auto lora_contribution = lora_ops::batch_matmul(*lora_B_, *lora_A_T,
                                                   1, lora_B_->shape()[0],
                                                   lora_A_->shape()[1],
                                                   lora_config_.rank);

    float scale = get_scale_factor();
    auto scaled_contribution = ops::scale(lora_contribution, scale);

}

void LoRALayer::save_lora_weights(const std::string& path) const {

    if (!lora_A_ || !lora_B_) {
        throw std::runtime_error("LoRA weights not initialized");
    }

    std::ofstream config_file(path + ".config");
    if (config_file.is_open()) {
        config_file << "{\n";
        config_file << "  \"rank\": " << lora_config_.rank << ",\n";
        config_file << "  \"alpha\": " << lora_config_.alpha << ",\n";
        config_file << "  \"dropout\": " << lora_config_.dropout << ",\n";
        config_file << "  \"use_bias\": " << (lora_config_.use_bias ? "true" : "false") << ",\n";
        config_file << "  \"residual\": " << (lora_config_.residual ? "true" : "false") << "\n";
        config_file << "}";
        config_file.close();
    }

    lora_A_->save(path + ".lora_A");
    lora_B_->save(path + ".lora_B");

    if (bias_) {
        bias_->save(path + ".bias");
    }
}

void LoRALayer::load_lora_weights(const std::string& path) {

    try {

        std::ifstream config_file(path + ".config");
        if (config_file.is_open()) {

            config_file.close();
        }

        *lora_A_ = Tensor::load(path + ".lora_A");
        *lora_B_ = Tensor::load(path + ".lora_B");

        if (bias_) {
            *bias_ = Tensor::load(path + ".bias");
        }

        grad_lora_A_->zero_();
        grad_lora_B_->zero_();
        if (grad_bias_) {
            grad_bias_->zero_();
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load LoRA weights: " + std::string(e.what()));
    }
}

TensorPtr LoRALayer::apply_dropout(const Tensor& x, bool training) {
    return lora_ops::apply_dropout(x, lora_config_.dropout, training);
}

}
