#include "lora_ops.h"
#include "../core/tensor.h"
#include "../core/ops.h"
#include "../core/utils.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace ops {

namespace lora_ops {

TensorPtr forward(const Tensor& input,
                  const Tensor& lora_A,
                  const Tensor& lora_B,
                  const Tensor& bias,
                  const LoRAOpsConfig& config,
                  bool training) {

    auto input_shape = input.shape();
    int batch_size = input_shape[0];
    int in_features = input_shape[1];

    auto dropped_input = apply_dropout(input, config.dropout, training);

        // [Translated]
    // input:   [batch_size, in_features]
    // lora_A:  [in_features, rank]
    // hidden:  [batch_size, rank]
    // lora_B:  [rank, out_features]
    // output:  [batch_size, out_features]
    // Note: batch_matmul(A, B, batch, M, N, K) require B shapeas [M, N]
    auto hidden = batch_matmul(*dropped_input, lora_A, batch_size, in_features, config.rank, in_features);

    int out_features = lora_B.shape()[0];
    auto output = batch_matmul(*hidden, lora_B, batch_size, config.rank, out_features, config.rank);

    float scale = get_scale_factor(config);
    auto scaled_output = ops::scale(*output, scale);

    if (config.use_bias) {
        scaled_output = ops::add_bias(*scaled_output, bias);
    }

        // [Translated]
    if (config.residual && in_features == out_features) {
        scaled_output = ops::add(*scaled_output, input);
    }

    return scaled_output;
}

std::vector<TensorPtr> backward(const Tensor& grad_output,
                                const Tensor& input,
                                const Tensor& lora_A,
                                const Tensor& lora_B,
                                const Tensor& hidden_state,
                                const LoRAOpsConfig& config) {

    auto input_shape = input.shape();
    int batch_size = input_shape[0];
    int in_features = input_shape[1];
    int out_features = lora_B.shape()[0];

    float scale = get_scale_factor(config);

    auto grad_output_T = ops::transpose(grad_output, 0, 1);

    auto grad_lora_B = batch_matmul(*grad_output_T, hidden_state, out_features, batch_size, config.rank, batch_size);

    auto grad_hidden = batch_matmul(grad_output, lora_B, batch_size, out_features, config.rank, out_features);

    auto scaled_grad_hidden = ops::scale(*grad_hidden, scale);

    auto input_T = ops::transpose(input, 0, 1);

    auto grad_lora_A = batch_matmul(*input_T, *scaled_grad_hidden, in_features, batch_size, config.rank, batch_size);

    auto grad_input = batch_matmul(*scaled_grad_hidden, lora_A, batch_size, config.rank, in_features, config.rank);

    if (config.residual && in_features == out_features) {

        grad_input = ops::add(*grad_input, grad_output);
    }

    TensorPtr grad_bias = nullptr;
    if (config.use_bias) {

        grad_bias = ops::sum(grad_output, 0);
    }

    return {grad_input, grad_lora_A, grad_lora_B, grad_bias};
}

TensorPtr apply_dropout(const Tensor& x, float dropout_rate, bool training) {
    if (!training || dropout_rate <= 0.0f) {
        return std::make_shared<Tensor>(x);
    }

    auto output = std::make_shared<Tensor>(x);
    auto& output_data = output->data();
    float scale = 1.0f / (1.0f - dropout_rate);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0f - dropout_rate);

    for (auto& val : output_data) {
        if (dist(gen)) {
            val *= scale;
        } else {
            val = 0.0f;
        }
    }

    return output;
}

float get_scale_factor(const LoRAOpsConfig& config) {
    if (config.rank <= 0) {
        std::cerr << "ERROR: Invalid LoRA rank " << config.rank << " in get_scale_factor, using 0.0f" << std::endl;
        return 0.0f;
    }
    return config.alpha / config.rank;
}

void init_lora_weights(Tensor& lora_A, Tensor& lora_B,
                      int in_features, int out_features, int rank) {

    std::random_device rd;
    std::mt19937 gen(rd());

    float scale_A = std::sqrt(2.0f / (in_features + rank));
    float scale_B = std::sqrt(2.0f / (rank + out_features));

    std::normal_distribution<float> dist_A(0.0f, scale_A);
    std::normal_distribution<float> dist_B(0.0f, scale_B);

    auto& lora_A_data = lora_A.data();
    for (auto& w : lora_A_data) {
        w = dist_A(gen);
    }

    auto& lora_B_data = lora_B.data();
    for (auto& w : lora_B_data) {
        w = dist_B(gen);
    }
}

TensorPtr transpose(const Tensor& input, int dim1, int dim2) {
    auto input_shape = input.shape();
    std::vector<int> new_shape = input_shape;
    std::swap(new_shape[dim1], new_shape[dim2]);

    auto output = std::make_shared<Tensor>(new_shape);
    auto& input_data = input.data();
    auto& output_data = output->data();

    if (input_shape.size() == 2) {
        int rows = input_shape[0];
        int cols = input_shape[1];

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output_data[j * rows + i] = input_data[i * cols + j];
            }
        }
    }

    return output;
}

TensorPtr batch_matmul(const Tensor& A, const Tensor& B,
                      int batch_size, int M, int N, int K) {
    (void)K;

    std::vector<int> output_shape = {batch_size, N};
    auto output = std::make_shared<Tensor>(output_shape);
    auto& output_data = output->data();

    const auto& A_data = A.data();
    const auto& B_data = B.data();

    auto A_shape = A.shape();
    auto B_shape = B.shape();

    if (A_shape.size() == 2 && A_shape[0] == batch_size && A_shape[1] == M) {

        if (B_shape.size() == 2 && B_shape[0] == M && B_shape[1] == N) {

            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < M; ++j) {

                        size_t a_idx = static_cast<size_t>(b) * M + j;
                        size_t b_idx = static_cast<size_t>(j) * N + i;

                        if (a_idx < A_data.size() && b_idx < B_data.size()) {
                            sum += A_data[a_idx] * B_data[b_idx];
                        }
                    }

                    size_t out_idx = static_cast<size_t>(b) * N + i;
                    if (out_idx < output_data.size()) {
                        output_data[out_idx] = sum;
                    }
                }
            }
        } else {
            throw std::runtime_error("batch_matmul: B matrix dimensions mismatch");
        }
    } else {
        throw std::runtime_error("batch_matmul: A matrix dimensions mismatch");
    }

    return output;
}

}

}
