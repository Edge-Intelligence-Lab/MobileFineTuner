/**
 * @file memory_first_mlp.cpp
 * @brief Memory-optimized MLP implementsation
 */

#include "memory_first_mlp.h"
#include "backward_functions.h"
#include "autograd_engine.h"
#include <algorithm>
#include <iostream>

namespace ops {
namespace memory_first {

TensorPtr memory_first_mlp_forward(
    const TensorPtr& input,
    const TensorPtr& fc_weight,
    const TensorPtr& fc_bias,
    const TensorPtr& proj_weight,
    const TensorPtr& proj_bias,
    int chunk_size
) {
    // input: [batch*seq_len, n_embd]
    // fc_weight: [n_inner, n_embd]
    // proj_weight: [n_embd, n_inner]
    
    auto input_shape = input->shape();
    if (input_shape.size() != 2) {
        throw TensorError("memory_first_mlp: input must be 2D [batch*seq, n_embd]");
    }
    
    int64_t batch_seq = input_shape[0];
    int64_t n_embd = input_shape[1];
    int64_t n_inner = fc_weight->shape()[0];
    
        // [Translated]
    auto output = zeros({batch_seq, n_embd}, input->dtype(), input->device());
    
    const float* input_data = input->data<float>();
    const float* fc_w_data = fc_weight->data<float>();
    const float* fc_b_data = fc_bias->data<float>();
    const float* proj_w_data = proj_weight->data<float>();
    const float* proj_b_data = proj_bias->data<float>();
    float* output_data = output->data<float>();
    
    // byintermediatelayerdimensionchunkedprocess
    for (int64_t chunk_start = 0; chunk_start < n_inner; chunk_start += chunk_size) {
        int64_t chunk_end = std::min(chunk_start + chunk_size, n_inner);
        int64_t actual_chunk_size = chunk_end - chunk_start;
        
        // allocatecurrentblocktemporarystorage [batch_seq, actual_chunk_size]
        std::vector<float> hidden_chunk(batch_seq * actual_chunk_size);
        std::vector<float> activated_chunk(batch_seq * actual_chunk_size);
        
        // === first layer：Compute hidden_chunk = input @ fc_weight[:,chunk_start:chunk_end].T + fc_bias ===
        
                // [Translated]
        for (int64_t i = 0; i < batch_seq; ++i) {
            for (int64_t j = 0; j < actual_chunk_size; ++j) {
                hidden_chunk[i * actual_chunk_size + j] = fc_b_data[chunk_start + j];
            }
        }
        
                // [Translated]
        // => hidden_chunk [batch_seq, actual_chunk_size]
        for (int64_t i = 0; i < batch_seq; ++i) {
            for (int64_t j = 0; j < actual_chunk_size; ++j) {
                int64_t fc_row = chunk_start + j;
                float sum = 0.0f;
                
                for (int64_t k = 0; k < n_embd; ++k) {
                    sum += input_data[i * n_embd + k] * fc_w_data[fc_row * n_embd + k];
                }
                
                hidden_chunk[i * actual_chunk_size + j] += sum;
            }
        }
        
        // === activation function GELU ===
        for (int64_t i = 0; i < batch_seq * actual_chunk_size; ++i) {
            float x = hidden_chunk[i];
            float tanh_input = 0.7978845608f * (x + 0.044715f * x * x * x);
            activated_chunk[i] = 0.5f * x * (1.0f + std::tanh(tanh_input));
        }
        
        // === second layer：output += activated_chunk @ proj_weight[chunk_start:chunk_end,:].T ===
        // activated_chunk [batch_seq, actual_chunk_size]
                // [Translated]
        // => output [batch_seq, n_embd]
        
        for (int64_t i = 0; i < batch_seq; ++i) {
            for (int64_t j = 0; j < n_embd; ++j) {
                float sum = 0.0f;
                
                for (int64_t k = 0; k < actual_chunk_size; ++k) {
                    int64_t proj_row = j;
                    int64_t proj_col = chunk_start + k;
                    sum += activated_chunk[i * actual_chunk_size + k] * 
                           proj_w_data[proj_row * n_inner + proj_col];
                }
                
                output_data[i * n_embd + j] += sum;
            }
        }
        
                // [Translated]
    }
    
        // [Translated]
    for (int64_t i = 0; i < batch_seq; ++i) {
        for (int64_t j = 0; j < n_embd; ++j) {
            output_data[i * n_embd + j] += proj_b_data[j];
        }
    }
    
    // Set gradient computation: Full recomputation backward (required for mobile training)
    if (input->requires_grad() || fc_weight->requires_grad() || proj_weight->requires_grad()) {
        output->set_requires_grad(true);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<MemoryFirstMLPBackward>(
            input, fc_weight, fc_bias, proj_weight, proj_bias, 
            chunk_size, batch_seq, n_embd, n_inner);
        ops::autograd::Engine::instance().register_node(output, {input}, backward_fn);
        #else
        // Old engine: Use grad_fn (recomputation-based, avoids saving intermediate activations)
        output->set_grad_fn([input, fc_weight, fc_bias, proj_weight, proj_bias, chunk_size, batch_seq, n_embd, n_inner]
                           (const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // Recompute forward to get intermediate activations (chunked recomputation, memory-controlled)
            // Initialize gradients
            auto grad_input = zeros(input->shape(), input->dtype(), input->device());
            
            const float* input_data = input->data<float>();
            const float* fc_w_data = fc_weight->data<float>();
            const float* proj_w_data = proj_weight->data<float>();
            const float* grad_out_data = grad_output->data<float>();
            float* grad_in_data = grad_input->data<float>();
            
            // Backward propagation chunk by chunk（withforwardcorresponding）
            for (int64_t chunk_start = 0; chunk_start < n_inner; chunk_start += chunk_size) {
                int64_t chunk_end = std::min(chunk_start + chunk_size, n_inner);
                int64_t actual_chunk_size = chunk_end - chunk_start;
                
                // Recompute forward intermediate values
                std::vector<float> hidden_chunk(batch_seq * actual_chunk_size);
                std::vector<float> activated_chunk(batch_seq * actual_chunk_size);
                
                // Recompute hidden_chunk（fclayer）
                for (int64_t i = 0; i < batch_seq; ++i) {
                    for (int64_t j = 0; j < actual_chunk_size; ++j) {
                        int64_t fc_row = chunk_start + j;
                        float sum = fc_bias->data<float>()[fc_row];
                        for (int64_t k = 0; k < n_embd; ++k) {
                            sum += input_data[i * n_embd + k] * fc_w_data[fc_row * n_embd + k];
                        }
                        hidden_chunk[i * actual_chunk_size + j] = sum;
                    }
                }
                
                // Recompute activated_chunk（GELU）
                std::vector<float> gelu_grad(batch_seq * actual_chunk_size);
                for (int64_t i = 0; i < batch_seq * actual_chunk_size; ++i) {
                    float x = hidden_chunk[i];
                    float tanh_input = 0.7978845608f * (x + 0.044715f * x * x * x);
                    float tanh_val = std::tanh(tanh_input);
                    activated_chunk[i] = 0.5f * x * (1.0f + tanh_val);
                    
                    // GELU derivative（used forbackward）
                    float sech2 = 1.0f - tanh_val * tanh_val;
                    float tanh_grad = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
                    gelu_grad[i] = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * tanh_grad;
                }
                
                                // [Translated]
                // grad_activated_chunk = grad_output @ proj_weight[chunk,:].T
                std::vector<float> grad_activated_chunk(batch_seq * actual_chunk_size, 0.0f);
                for (int64_t i = 0; i < batch_seq; ++i) {
                    for (int64_t k = 0; k < actual_chunk_size; ++k) {
                        float sum = 0.0f;
                        for (int64_t j = 0; j < n_embd; ++j) {
                            int64_t proj_col = chunk_start + k;
                            sum += grad_out_data[i * n_embd + j] * proj_w_data[j * n_inner + proj_col];
                        }
                        grad_activated_chunk[i * actual_chunk_size + k] = sum;
                    }
                }
                
                // via GELU derivative
                std::vector<float> grad_hidden_chunk(batch_seq * actual_chunk_size);
                for (int64_t i = 0; i < batch_seq * actual_chunk_size; ++i) {
                    grad_hidden_chunk[i] = grad_activated_chunk[i] * gelu_grad[i];
                }
                
                // Propagate to input：grad_input += grad_hidden_chunk @ fc_weight[chunk,:].T
                for (int64_t i = 0; i < batch_seq; ++i) {
                    for (int64_t j = 0; j < n_embd; ++j) {
                        float sum = 0.0f;
                        for (int64_t k = 0; k < actual_chunk_size; ++k) {
                            int64_t fc_row = chunk_start + k;
                            sum += grad_hidden_chunk[i * actual_chunk_size + k] * fc_w_data[fc_row * n_embd + j];
                        }
                        grad_in_data[i * n_embd + j] += sum;
                    }
                }
            }
            
            // Only propagate gradient to input (MLP weights frozen)
            if (input->requires_grad()) {
                accumulate_gradient(input, grad_input);
            }
            
            return {grad_input};
        });
        #endif
    }
    
    return output;
}

} // namespace memory_first
} // namespace ops

