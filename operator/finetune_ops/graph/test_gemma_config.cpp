#include "gemma_model.h"
#include "safetensors_loader.h"

#include <cassert>
#include <iostream>

using namespace ops;

int main(int argc, char** argv) {
    std::string model_dir = "../../../gemma-3-270m";
    if (argc > 1) {
        model_dir = argv[1];
    }

    std::cout << "[Test] GemmaTextConfig::from_pretrained\n";
    auto cfg = GemmaTextConfig::from_pretrained(model_dir);

    assert(cfg.vocab_size == 262144);
    assert(cfg.hidden_size == 640);
    assert(cfg.intermediate_size == 2048);
    assert(cfg.num_hidden_layers == 18);
    assert(cfg.num_attention_heads == 4);
    assert(cfg.num_key_value_heads == 1);
    assert(cfg.head_dim == 256);
    assert(cfg.sliding_window == 512);
    assert(cfg.layer_types.size() == static_cast<size_t>(cfg.num_hidden_layers));
    assert(cfg.layer_types[0] == "sliding_attention");

    std::cout << "  ✓ core fields match Gemma-3-270M config\n";

    std::cout << "\n[Test] GemmaKeyMapper::generate_gemma_mapping\n";
    auto mapping = GemmaKeyMapper::generate_gemma_mapping(cfg.num_hidden_layers);

    size_t expected_per_layer = 13;  // 4 norms + 4 attn proj + 2 attn norms + 3 mlp
    size_t expected_total = 3 + cfg.num_hidden_layers * expected_per_layer;  // top embed/norm/lm_head
    assert(mapping.size() == expected_total);

    auto it = mapping.find("layers.0.self_attn.q_proj.weight");
    assert(it != mapping.end());
    assert(it->second == "model.layers.0.self_attn.q_proj.weight");

    std::cout << "  ✓ mapping size = " << mapping.size() << ", sample entry verified\n";
    std::cout << "\nAll Gemma config/mapping tests passed ✅\n";
    return 0;
}

