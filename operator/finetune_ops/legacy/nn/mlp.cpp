#include "mlp.h"
#include <random>
#include <iostream>

namespace ops {

FeedForward::FeedForward(const MLPConfig& config) : config_(config) {
    int n_embd = config_.input_size;
    int intermediate_size = 4 * n_embd;

    fc1_weight_ = std::make_unique<Tensor>(std::vector<int>{n_embd, intermediate_size});
    fc1_bias_ = std::make_unique<Tensor>(std::vector<int>{intermediate_size});
    fc2_weight_ = std::make_unique<Tensor>(std::vector<int>{intermediate_size, n_embd});
    fc2_bias_ = std::make_unique<Tensor>(std::vector<int>{n_embd});

    initialize_weights();
}

void FeedForward::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (float& val : fc1_weight_->data()) val = dist(gen);
    for (float& val : fc2_weight_->data()) val = dist(gen);

    for (float& val : fc1_bias_->data()) val = 0.0f;
    for (float& val : fc2_bias_->data()) val = 0.0f;
}

std::unique_ptr<Tensor> FeedForward::forward(const Tensor& input) {

    auto hidden = ops::linear(input, *fc1_weight_, *fc1_bias_);
    auto activated = ops::gelu(*hidden);

    auto output = ops::linear(*activated, *fc2_weight_, *fc2_bias_);

    return std::make_unique<Tensor>(*output);
}

void FeedForward::print_info() const {
    std::cout << "FeedForward Info:" << std::endl;
    std::cout << "  [Output]: " << config_.input_size << std::endl;
    std::cout << "  [Output]: " << 4 * config_.input_size << std::endl;
    std::cout << "  [Output]: " << config_.input_size << std::endl;
    std::cout << "  activation[Output]: GELU" << std::endl;
}

SimpleMLP::SimpleMLP(const MLPConfig& config) : config_(config) {
    weight1_ = std::make_unique<Tensor>(std::vector<int>{config_.hidden_size, config_.input_size});
    bias1_ = std::make_unique<Tensor>(std::vector<int>{config_.hidden_size});
    weight2_ = std::make_unique<Tensor>(std::vector<int>{config_.output_size, config_.hidden_size});
    bias2_ = std::make_unique<Tensor>(std::vector<int>{config_.output_size});

    initialize_weights();
}

void SimpleMLP::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (float& val : weight1_->data()) val = dist(gen);
    for (float& val : weight2_->data()) val = dist(gen);

    for (float& val : bias1_->data()) val = 0.0f;
    for (float& val : bias2_->data()) val = 0.0f;
}

std::unique_ptr<Tensor> SimpleMLP::forward(const Tensor& input) {

    auto weight1_t = ops::transpose(*weight1_, 0, 1);
    auto hidden = ops::linear(input, *weight1_t, *bias1_);

    std::unique_ptr<Tensor> activated;
    if (config_.activation == "relu") {
        auto result = ops::relu(*hidden);
        activated = std::make_unique<Tensor>(*result);
    } else if (config_.activation == "sigmoid") {
        auto result = ops::sigmoid(*hidden);
        activated = std::make_unique<Tensor>(*result);
    } else if (config_.activation == "tanh") {
        auto result = ops::tanh(*hidden);
        activated = std::make_unique<Tensor>(*result);
    } else if (config_.activation == "gelu") {
        auto result = ops::gelu(*hidden);
        activated = std::make_unique<Tensor>(*result);
    } else {

        auto result = ops::relu(*hidden);
        activated = std::make_unique<Tensor>(*result);
    }

    auto weight2_t = ops::transpose(*weight2_, 0, 1);
    auto output = ops::linear(*activated, *weight2_t, *bias2_);

    return std::make_unique<Tensor>(*output);
}

void SimpleMLP::print_info() const {
    std::cout << "SimpleMLP Info:" << std::endl;
    std::cout << "  [Output]: " << config_.input_size << std::endl;
    std::cout << "  [Output]: " << config_.hidden_size << std::endl;
    std::cout << "  [Output]: " << config_.output_size << std::endl;
    std::cout << "  activation[Output]: " << config_.activation << std::endl;
    std::cout << "  [Output]1[Output]: " << config_.hidden_size << " x " << config_.input_size << std::endl;
    std::cout << "  [Output]2[Output]: " << config_.output_size << " x " << config_.hidden_size << std::endl;
}

}