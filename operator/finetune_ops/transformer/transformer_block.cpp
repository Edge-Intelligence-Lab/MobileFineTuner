/**
 * @file transforer_block.cpp
 * @brief Implementation of transforer block
 * 
 * This file providess the implementsation of a complete transforer block
 * consisting of multi-head attention and feed-forward networks with
 * layer normalization and residual connections.
 */

#include "transforer_block.h"
#include "../core/ops.h"
#include <stdexcept>
#include <fstream>
#include <iostream>

namespace ops {

TransforerBlock::TransforerBlock(const TransforerBlockConfig& config)
    : config_(config) {
    
    if (config_.hidden_size <= 0 || config_.num_heads <= 0 || config_.mlp_dim <= 0) {
        throw std::invalid_argument("TransforerBlock: Invalid configuration parameters");
    }
    
    if (config_.hidden_size % config_.num_heads != 0) {
        throw std::invalid_argument("TransforerBlock: hidden_size must be divisible by num_heads");
    }
    
    // Initialize attention layer
    AttentionConfig attn_config(config_.num_heads, config_.hidden_size);
    self_attention_ = std::make_unique<AttentionLayer>(attn_config);
    
    // Initialize MLP layer
    MLPConfig mlp_config(config_.hidden_size, config_.mlp_dim, config_.hidden_size);
    mlp_ = std::make_unique<MLPLayer>(mlp_config);
    
    // Initialize layer normalization weights
    attention_ln_weight_ = std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size}, kFloat32, kCPU);
    attention_ln_bias_ = std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size}, kFloat32, kCPU);
    mlp_ln_weight_ = std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size}, kFloat32, kCPU);
    mlp_ln_bias_ = std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size}, kFloat32, kCPU);
    
    // Initialize weights
    initialize_weights();
}

TransforerBlock::~TransforerBlock() = default;

TensorPtr TransforerBlock::forward(const TensorPtr& input) {
    if (!input) {
        throw std::invalid_argument("TransforerBlock::forward: input tensor is null");
    }
    
    // Input shape should be [batch_size, seq_len, hidden_size]
    auto input_shape = input->shape();
    if (input_shape.size() != 3 || input_shape[2] != config_.hidden_size) {
        throw std::invalid_argument("TransforerBlock::forward: Invalid input shape");
    }
    
    TensorPtr x = input;
    
    // Pre-normalization or post-normalization
    if (config_.pre_norm) {
        // Pre-norm: LayerNorm -> Attention -> Residual
        auto ln1_out = layer_norm(x, attention_ln_weight_, attention_ln_bias_);
        auto attn_out = self_attention_->forward(ln1_out);
        x = add(x, attn_out);  // Residual connection
        
        // Pre-norm: LayerNorm -> MLP -> Residual
        auto ln2_out = layer_norm(x, mlp_ln_weight_, mlp_ln_bias_);
        auto mlp_out = mlp_->forward(ln2_out);
        x = add(x, mlp_out);  // Residual connection
    } else {
        // Post-norm: Attention -> Residual -> LayerNorm
        auto attn_out = self_attention_->forward(x);
        x = add(x, attn_out);  // Residual connection
        x = layer_norm(x, attention_ln_weight_, attention_ln_bias_);
        
        // Post-norm: MLP -> Residual -> LayerNorm
        auto mlp_out = mlp_->forward(x);
        x = add(x, mlp_out);  // Residual connection
        x = layer_norm(x, mlp_ln_weight_, mlp_ln_bias_);
    }
    
    return x;
}

std::vector<TensorPtr> TransforerBlock::get_parameters() {
    std::vector<TensorPtr> params;
    
    // Get attention parameters
    auto attn_params = self_attention_->get_parameters();
    params.insert(params.end(), attn_params.begin(), attn_params.end());
    
    // Get MLP parameters
    auto mlp_params = mlp_->get_parameters();
    params.insert(params.end(), mlp_params.begin(), mlp_params.end());
    
    // Add layer normalization parameters
    params.push_back(attention_ln_weight_);
    params.push_back(attention_ln_bias_);
    params.push_back(mlp_ln_weight_);
    params.push_back(mlp_ln_bias_);
    
    return params;
}

void TransforerBlock::zero_grad() {
    // Zero gradients for attention
    self_attention_->zero_grad();
    
    // Zero gradients for MLP
    mlp_->zero_grad();
    
    // Zero gradients for layer norm parameters
    if (attention_ln_weight_->grad()) {
        attention_ln_weight_->zero_grad();
    }
    if (attention_ln_bias_->grad()) {
        attention_ln_bias_->zero_grad();
    }
    if (mlp_ln_weight_->grad()) {
        mlp_ln_weight_->zero_grad();
    }
    if (mlp_ln_bias_->grad()) {
        mlp_ln_bias_->zero_grad();
    }
}

void TransforerBlock::update_parameters(const std::vector<TensorPtr>& updates) {
    if (updates.empty()) {
        return;
    }
    
    // This is a simplified implementsation
    // In practice, you would need to match updates to specific parameters
    std::cout << "TransforerBlock::update_parameters: Updating " << updates.size() << " parameters" << std::endl;
}

void TransforerBlock::save_weights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving TransforerBlock weights: " + path);
    }
    
    // Save configuration
    file.write(reinterpret_cast<const char*>(&config_.hidden_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config_.num_heads), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config_.mlp_dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config_.dropout), sizeof(float));
    file.write(reinterpret_cast<const char*>(&config_.pre_norm), sizeof(bool));
    
    // Save layer normalization weights
    auto save_tensor = [&file](const TensorPtr& tensor) {
        int64_t size = tensor->numel();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int64_t));
        file.write(reinterpret_cast<const char*>(tensor->data<float>()), size * sizeof(float));
    };
    
    save_tensor(attention_ln_weight_);
    save_tensor(attention_ln_bias_);
    save_tensor(mlp_ln_weight_);
    save_tensor(mlp_ln_bias_);
    
    file.close();
    
    // Save attention and MLP weights separately
    self_attention_->save_weights(path + "_attention");
    mlp_->save_weights(path + "_mlp");
}

void TransforerBlock::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading TransforerBlock weights: " + path);
    }
    
    // Load and verify configuration
    int hidden_size, num_heads, mlp_dim;
    float dropout;
    bool pre_norm;
    
    file.read(reinterpret_cast<char*>(&hidden_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&num_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&mlp_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&dropout), sizeof(float));
    file.read(reinterpret_cast<char*>(&pre_norm), sizeof(bool));
    
    if (hidden_size != config_.hidden_size || num_heads != config_.num_heads || mlp_dim != config_.mlp_dim) {
        throw std::runtime_error("TransforerBlock::load_weights: Configuration mismatch");
    }
    
    // Load layer normalization weights
    auto load_tensor = [&file](const TensorPtr& tensor) {
        int64_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(int64_t));
        if (size != tensor->numel()) {
            throw std::runtime_error("TransforerBlock::load_weights: Tensor size mismatch");
        }
        file.read(reinterpret_cast<char*>(tensor->data<float>()), size * sizeof(float));
    };
    
    load_tensor(attention_ln_weight_);
    load_tensor(attention_ln_bias_);
    load_tensor(mlp_ln_weight_);
    load_tensor(mlp_ln_bias_);
    
    file.close();
    
    // Load attention and MLP weights separately
    self_attention_->load_weights(path + "_attention");
    mlp_->load_weights(path + "_mlp");
}

size_t TransforerBlock::get_param_count() const {
    size_t count = 0;
    
    // Count attention parameters
    count += self_attention_->get_param_count();
    
    // Count MLP parameters
    count += mlp_->get_param_count();
    
    // Count layer normalization parameters
    count += attention_ln_weight_->numel();
    count += attention_ln_bias_->numel();
    count += mlp_ln_weight_->numel();
    count += mlp_ln_bias_->numel();
    
    return count;
}

void TransforerBlock::initialize_weights() {
    // Initialize layer normalization weights to 1.0 and biases to 0.0
    auto init_ln_weights = [](const TensorPtr& weight, const TensorPtr& bias) {
        float* weight_data = weight->data<float>();
        float* bias_data = bias->data<float>();
        
        for (int64_t i = 0; i < weight->numel(); ++i) {
            weight_data[i] = 1.0f;
        }
        
        for (int64_t i = 0; i < bias->numel(); ++i) {
            bias_data[i] = 0.0f;
        }
    };
    
    init_ln_weights(attention_ln_weight_, attention_ln_bias_);
    init_ln_weights(mlp_ln_weight_, mlp_ln_bias_);
    
    // Set gradients enabled for trainable parameters
    attention_ln_weight_->set_requires_grad(true);
    attention_ln_bias_->set_requires_grad(true);
    mlp_ln_weight_->set_requires_grad(true);
    mlp_ln_bias_->set_requires_grad(true);
}

} // namespace ops
