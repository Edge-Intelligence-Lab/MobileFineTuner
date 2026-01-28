/**
 * @file mlp.h
 * @brief Multi-Layer Perceptron (MLP) implementsation
 * 
 * This file providess the FeedForward class which implementss
 * a multi-layer perceptron with configurable activation functions.
 * It's commonly used as the feed-forward network in transforer blocks.
 */

#pragma once

#include "../core/tensor.h"
#include "../core/ops.h"
#include <memory>
#include <vector>

namespace ops {

/**
 * @brief Configuration structure for MLP layers
 * 
 * Contains the architectural parameters for a multi-layer perceptron
 * including input/output dimensions and activation function.
 */
struct MLPConfig {
    int input_size;     /**< Input dimension */
    int hidden_size;    /**< Hidden layer dimension */
    int output_size;    /**< Output dimension */
    std::string activation; /**< Activation function name */

    /**
     * @brief Construct MLP configuration
     * @param in Input dimension
     * @param hidden Hidden layer dimension
     * @param out Output dimension
     * @param act Activation function (default: "relu")
     */
    MLPConfig(int in, int hidden, int out, const std::string& act = "relu")
        : input_size(in), hidden_size(hidden), output_size(out), activation(act) {}
};

class FeedForward {
private:
    MLPConfig config_;

    std::unique_ptr<Tensor> fc1_weight_;
    std::unique_ptr<Tensor> fc1_bias_;
    std::unique_ptr<Tensor> fc2_weight_;
    std::unique_ptr<Tensor> fc2_bias_;

    void initialize_weights();

public:
    explicit FeedForward(const MLPConfig& config);
    ~FeedForward() = default;

    std::unique_ptr<Tensor> forward(const Tensor& input);

    void print_info() const;
};

class SimpleMLP {
private:
    MLPConfig config_;

    std::unique_ptr<Tensor> weight1_;
    std::unique_ptr<Tensor> bias1_;
    std::unique_ptr<Tensor> weight2_;
    std::unique_ptr<Tensor> bias2_;

    void initialize_weights();

public:
    explicit SimpleMLP(const MLPConfig& config);
    ~SimpleMLP() = default;

    std::unique_ptr<Tensor> forward(const Tensor& input);

    const Tensor& get_weight1() const { return *weight1_; }
    const Tensor& get_weight2() const { return *weight2_; }

    void print_info() const;
};

inline std::unique_ptr<Tensor> fully_connected(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    auto result = ops::linear(input, weight, bias);
    return std::make_unique<Tensor>(*result);
}

inline std::unique_ptr<Tensor> mlp_forward(const Tensor& input,
                                          const std::vector<std::unique_ptr<Tensor>>& weights,
                                          const std::vector<std::unique_ptr<Tensor>>& biases,
                                          const std::vector<std::string>& activations) {
    assert(weights.size() == biases.size() && weights.size() == activations.size());

    auto current = std::make_unique<Tensor>(input);

    for (size_t i = 0; i < weights.size(); ++i) {

        auto linear_result = ops::linear(*current, *weights[i], *biases[i]);
        current = std::make_unique<Tensor>(*linear_result);

        if (activations[i] == "relu") {
            auto activated = ops::relu(*current);
            current = std::make_unique<Tensor>(*activated);
        } else if (activations[i] == "sigmoid") {
            auto activated = ops::sigmoid(*current);
            current = std::make_unique<Tensor>(*activated);
        } else if (activations[i] == "tanh") {
            auto activated = ops::tanh(*current);
            current = std::make_unique<Tensor>(*activated);
        } else if (activations[i] == "gelu") {
            auto activated = ops::gelu(*current);
            current = std::make_unique<Tensor>(*activated);
        }

    }

    return current;
}

inline std::vector<std::unique_ptr<Tensor>> create_mlp_weights(const std::vector<int>& layer_sizes) {
    std::vector<std::unique_ptr<Tensor>> weights;

    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        weights.push_back(std::make_unique<Tensor>(std::vector<int>{layer_sizes[i + 1], layer_sizes[i]}));
    }

    return weights;
}

inline std::vector<std::unique_ptr<Tensor>> create_mlp_biases(const std::vector<int>& layer_sizes) {
    std::vector<std::unique_ptr<Tensor>> biases;

    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        biases.push_back(std::make_unique<Tensor>(std::vector<int>{layer_sizes[i]}));
    }

    return biases;
}

}