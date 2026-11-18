#include "../../third_party/nlohmann/json.hpp"
#include "../core/autograd_engine.h"
#include "../core/ops.h"
#include "auto_model.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace ops;
namespace fs = std::filesystem;
namespace {
void check(bool ok, const char* why) {
    if (!ok)
        throw std::runtime_error(why);
}
template <class F> void rejects(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, "Invalid input was accepted");
}
void config(const fs::path& path, const std::string& text) { std::ofstream(path) << text; }
void run(const fs::path& root) {
    const QwenConfig aggregate{32, 8, 12, 1, 2, 1, 16, 1e-6f, 10000.0f, "silu", -1, -1, -1};
    check(aggregate.model_type == "qwen2" && aggregate.hidden_size == 8,
          "Legacy QwenConfig aggregate initialization changed");
    auto cfgfile = root / "config.json";
    config(
        cfgfile,
        R"({"model_type":"qwen3","hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":6,"num_hidden_layers":1,"vocab_size":32,"intermediate_size":12,"tie_word_embeddings":false})");
    auto c = QwenConfig::from_pretrained(cfgfile.string());
    check(c.head_dim == 6 && !c.attention_bias && c.uses_qk_norm() && !c.tie_word_embeddings,
          "Qwen3 config dispatch");
    check(ModelRegistry::infer_family(root.string()) == ModelFamily::Qwen, "Qwen3 family dispatch");
    QwenModel deferred(c, QwenModel::WeightInitialization::Deferred);
    check(deferred.embedding_weight()->data_ptr() == nullptr,
          "Deferred model allocated embedding storage");
    rejects([&] { deferred.forward(zeros({1, 3}, kInt32)); });
    rejects([&] { deferred.init_lora(); });
    rejects([&] { deferred.load_lora_adapter(root.string()); });
    rejects([&] { deferred.output_weight(); });
    QwenModel m(c);
    for (auto& p : m.parameters())
        for (int64_t i = 0; i < p->numel(); ++i)
            p->data<float>()[i] = p->ndim() == 1 ? 1.0f : 0.02f * std::sin(float(i + 1));
    m.init_lora(2, 4, 0, std::vector<std::string>{"k_proj"});
    check(m.get_lora_parameters().size() == 2, "Exact target selection");
    auto named = m.named_lora_parameters();
    check(named.begin()->first.find("k_proj") != std::string::npos, "Selected wrong LoRA module");
    auto ids = zeros({1, 3}, kInt32);
    ids->data<int32_t>()[0] = 1;
    ids->data<int32_t>()[1] = 2;
    ids->data<int32_t>()[2] = 3;
    auto y = m.forward(ids);
    check(y->shape() == std::vector<int64_t>({1, 3, 32}), "Non-square Qwen3 projection shape");
    y->backward(ones(y->shape()));
    auto b = m.get_lora_parameters()[1];
    check(bool(b->grad()), "Custom target gradient missing");
    double norm = 0;
    for (int64_t i = 0; i < b->numel(); ++i)
        norm += std::abs(b->grad()->data<float>()[i]);
    check(norm > 0, "Custom target gradient is zero");
    rejects([&] { m.init_lora(); });
    ids->data<int32_t>()[0] = 32;
    rejects([&] { m.forward(ids); });
    ids->data<int32_t>()[0] = 1;
    rejects([&] { m.forward(ids, ones({2, 3})); });
    QwenModel clean(c);
    rejects([&] { clean.init_lora(0, 4, 0, false); });
    rejects([&] { clean.init_lora(2, 4, 0, std::vector<std::string>{"typo"}); });
    rejects([&] { clean.init_lora(2, 4, 0, std::vector<std::string>{"q_proj", "q_proj"}); });
    check(clean.get_lora_parameters().empty(), "Invalid target request modified model");
    clean.init_lora(2, 4, 0, std::vector<std::string>{"all-linear"});
    check(clean.get_lora_parameters().size() == 14, "all-linear target count");
    auto incompatible = c;
    incompatible.num_key_value_heads = 3;
    rejects([&] { QwenModel bad(incompatible); });
    incompatible = c;
    incompatible.head_dim = 5;
    rejects([&] { QwenModel bad(incompatible); });
    config(cfgfile, R"({"model_type":"qwen3","rope_scaling":{"rope_type":"yarn"}})");
    rejects([&] { QwenConfig::from_pretrained(cfgfile.string()); });
    config(cfgfile, R"({"model_type":"qwen3_5","text_config":{"model_type":"qwen3"}})");
    rejects([&] { QwenConfig::from_pretrained(cfgfile.string()); });
    config(cfgfile, "{broken}");
    rejects([&] { QwenConfig::from_pretrained(cfgfile.string()); });
    config(cfgfile, R"({"model_type":"qwen3","head_dim":6.5})");
    rejects([&] { QwenConfig::from_pretrained(cfgfile.string()); });
    // A normal safetensors metadata object must never become a tensor entry.
    const std::string header =
        R"({"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})";
    auto weightfile = root / "model.safetensors";
    {
        std::ofstream out(weightfile, std::ios::binary);
        uint64_t size = header.size();
        float values[] = {1, 2, 3, 4};
        out.write(reinterpret_cast<const char*>(&size), 8);
        out << header;
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    }
    SafeTensorsReader reader(weightfile.string());
    reader.parse_header();
    check(reader.get_tensor_names() == std::vector<std::string>{"weight"},
          "Safetensors metadata was treated as a weight");
    check(reader.load_tensor("weight")->data<float>()[3] == 4, "Safetensors value changed");
    QwenModel partial(c);
    auto original = partial.parameters().front();
    SafeTensorsLoadOptions options;
    options.verbose = false;
    rejects([&] { partial.load_pretrained(root.string(), options); });
    check(partial.parameters().front() == original, "Failed checkpoint load modified the model");
    rejects([&] { partial.assign_weight("layers.0.self_attn.q_proj.weight", zeros({8, 8})); });
    fs::resize_file(weightfile, 8 + header.size() + 4);
    rejects([&] {
        SafeTensorsReader corrupt(weightfile.string());
        corrupt.parse_header();
    });
    const std::string half_header =
        R"({"weight":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}})";
    {
        std::ofstream out(weightfile, std::ios::binary);
        const uint64_t size = half_header.size();
        const uint16_t values[] = {0x0001, 0x8001, 0x0400, 0x7bff};
        out.write(reinterpret_cast<const char*>(&size), 8);
        out << half_header;
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    }
    SafeTensorsReader half_reader(weightfile.string());
    half_reader.parse_header();
    auto promoted = half_reader.load_tensor("weight");
    check(promoted->data<float>()[0] == std::ldexp(1.0f, -24) &&
              promoted->data<float>()[1] == -std::ldexp(1.0f, -24) &&
              promoted->data<float>()[2] == std::ldexp(1.0f, -14) &&
              promoted->data<float>()[3] == 65504.0f,
          "FP16 normal/subnormal checkpoint promotion failed");
    const auto indexfile = root / "model.safetensors.index.json";
    fs::rename(weightfile, root / "shard.safetensors");
    config(
        indexfile,
        R"({"metadata":{"note":"not-a-shard.safetensors"},"weight_map":{"weight":"shard.safetensors"}})");
    SafeTensorsModelReader sharded(root.string());
    sharded.parse_headers();
    check(sharded.get_tensor_names() == std::vector<std::string>{"weight"},
          "Shard index metadata was treated as a filename");
    config(indexfile, R"({"weight_map":{"weight":"../outside.safetensors"}})");
    rejects([&] {
        SafeTensorsModelReader bad(root.string());
        bad.parse_headers();
    });
    config(indexfile, R"({"weight_map":{"weight":12}})");
    rejects([&] {
        SafeTensorsModelReader bad(root.string());
        bad.parse_headers();
    });
    const std::string fractional =
        R"({"weight":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}})";
    {
        std::ofstream file(weightfile, std::ios::binary);
        const uint64_t n = fractional.size();
        const float zero = 0;
        file.write(reinterpret_cast<const char*>(&n), 8);
        file << fractional;
        file.write(reinterpret_cast<const char*>(&zero), 4);
    }
    rejects([&] {
        SafeTensorsReader bad(weightfile.string());
        bad.parse_header();
    });
    autograd::Engine::instance().clear_graph();

    // Dropout affects only the adapter path, advances its RNG, and is disabled
    // by eval mode. The same seed produces the same sequence in a fresh layer.
    auto w = zeros({8, 6});
    auto a = ones({8, 2});
    auto bb = ones({2, 6});
    auto x = ones({1, 64, 8});
    LoRALinear l(w), same(w);
    l.attach_lora(a, bb, 1);
    same.attach_lora(a, bb, 1);
    l.set_dropout(0.5f, 17);
    same.set_dropout(0.5f, 17);
    auto first = l.forward(x), second = l.forward(x), seeded = same.forward(x);
    bool different = false;
    for (int64_t i = 0; i < first->numel(); ++i) {
        check(first->data<float>()[i] == seeded->data<float>()[i], "Dropout seed mismatch");
        different |= first->data<float>()[i] != second->data<float>()[i];
    }
    check(different, "Dropout mask did not advance");
    l.set_training(false);
    auto eval = l.forward(x);
    for (int64_t i = 0; i < eval->numel(); ++i)
        check(eval->data<float>()[i] == 16.0f, "Dropout changed eval output");
    autograd::Engine::instance().clear_graph();
    auto identity = zeros({8, 8});
    for (int i = 0; i < 8; ++i)
        identity->data<float>()[i * 8 + i] = 1;
    LoRALinear derivative(zeros({8, 8}));
    derivative.attach_lora(identity->clone(), identity->clone(), 1);
    derivative.set_dropout(0.5f, 19);
    auto input = ones({1, 4, 8});
    input->set_requires_grad(true);
    auto output = derivative.forward(input);
    output->backward(ones(output->shape()));
    const auto trainable = derivative.trainable_parameters();
    for (int i = 0; i < 32; ++i)
        check(input->grad()->data<float>()[i] == output->data<float>()[i],
              "Dropout input derivative mismatch");
    for (int row = 0; row < 8; ++row) {
        float total = 0;
        for (int t = 0; t < 4; ++t)
            total += output->data<float>()[t * 8 + row];
        for (int col = 0; col < 8; ++col) {
            check(trainable[0]->grad()->data<float>()[row * 8 + col] == total,
                  "Dropout A derivative mismatch");
            check(trainable[1]->grad()->data<float>()[row * 8 + col] == total,
                  "Dropout B derivative mismatch");
        }
    }
    autograd::Engine::instance().clear_graph();
}
void hybrid(const fs::path& root) {
    config(
        root / "config.json",
        R"({"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","vocab_size":32,"hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"tie_word_embeddings":true,"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":3,"linear_value_head_dim":4,"linear_conv_kernel_dim":3,"layer_types":["linear_attention","full_attention"],"rope_parameters":{"rope_type":"default","partial_rotary_factor":0.5,"rope_theta":10000}}})");
    const auto c = QwenConfig::from_pretrained(root.string());
    check(c.hybrid() && c.linear_layer(0) && !c.linear_layer(1), "Hybrid config dispatch");
    check(ModelRegistry::infer_family(root.string()) == ModelFamily::Qwen,
          "Hybrid family dispatch");
    const auto spec = ModelRegistry::inspect_pretrained(root.string());
    check(spec.tie_word_embeddings == c.tie_word_embeddings,
          "Hybrid tied-head metadata disagrees with model");
    check(spec.default_lora_targets == c.default_lora_targets(),
          "Hybrid default targets metadata disagrees with model");
    std::ifstream cfg_stream(root / "config.json");
    auto defaults = nlohmann::json::parse(cfg_stream).at("text_config");
    defaults.erase("head_dim");
    defaults.erase("linear_num_value_heads");
    defaults.erase("rope_parameters");
    config(root / "config.json", defaults.dump());
    const auto default_cfg = QwenConfig::from_pretrained(root.string());
    check(default_cfg.attention_head_dim() == 256 && default_cfg.linear_num_value_heads == 32 &&
              default_cfg.rope_theta == 10000.0f,
          "Qwen3.5 defaults differ from the pinned reference");
    defaults["rope_scaling"] = {{"rope_type", "yarn"}, {"factor", 2}};
    config(root / "config.json", defaults.dump());
    rejects([&] { QwenConfig::from_pretrained(root.string()); });
    QwenModel model(c);
    for (auto& p : model.parameters())
        for (int64_t i = 0; i < p->numel(); ++i)
            p->data<float>()[i] = p->ndim() == 1 ? 0.1f : 0.05f * std::sin(float(i + 1));
    model.init_lora(2, 4, 0, false);
    model.freeze_base();
    check(model.get_lora_parameters().size() == 18, "Hybrid default must cover both mixer types");
    auto ids = zeros({1, 4}, kInt32);
    for (int i = 0; i < 4; ++i)
        ids->data<int32_t>()[i] = i + 1;
    auto logits = model.forward(ids);
    logits->backward(ones(logits->shape()));
    for (const auto& p : model.named_lora_parameters())
        if (p.first.find("lora_B") != std::string::npos) {
            check(bool(p.second->grad()), "Hybrid LoRA gradient missing");
            double norm = 0;
            for (int64_t i = 0; i < p.second->numel(); ++i)
                norm += std::abs(p.second->grad()->data<float>()[i]);
            check(norm > 0, "Hybrid mixer LoRA B gradient is zero");
        }
    // Offload preflight must be atomic and include convolution/state weights.
    QwenModel budget(c);
    for (auto& p : budget.parameters())
        for (int64_t i = 0; i < p->numel(); ++i)
            p->data<float>()[i] = p->ndim() == 1 ? 0.1f : 0.05f * std::sin(float(i + 1));
    budget.init_lora(2, 4, 0, std::vector<std::string>{"all-linear"});
    budget.freeze_base();
    auto reference = budget.forward(ids);
    autograd::Engine::instance().clear_graph();
    auto embedding = budget.embedding_weight();
    sharding::ShardConfig too_small;
    too_small.offload_dir = (root / "invalid_hybrid_offload").string();
    too_small.max_resident_bytes = 1;
    rejects([&] { budget.enable_parameter_sharding(too_small); });
    check(budget.embedding_weight() == embedding && budget.get_lora_parameters().size() == 30,
          "Failed hybrid offload changed active weights or adapters");
    auto restored = budget.forward(ids);
    for (int64_t i = 0; i < restored->numel(); ++i)
        check(restored->data<float>()[i] == reference->data<float>()[i],
              "Failed hybrid offload modified output");
    autograd::Engine::instance().clear_graph();
    model.set_activation_checkpointing(true);
    QwenModel stochastic(c);
    stochastic.init_lora(2, 4, 0.1f, false);
    rejects([&] { stochastic.set_activation_checkpointing(true); });
    QwenModel checkpointed(c);
    checkpointed.set_activation_checkpointing(true);
    rejects([&] { checkpointed.init_lora(2, 4, 0.1f, false); });
    auto invalid = c;
    invalid.linear_num_value_heads = 3;
    invalid.linear_num_key_heads = 2;
    rejects([&] { QwenModel bad(invalid); });
    invalid = c;
    invalid.layer_types[0] = "unsupported";
    rejects([&] { QwenModel bad(invalid); });
    invalid = c;
    invalid.partial_rotary_factor = 0.125f;
    rejects([&] { QwenModel bad(invalid); });
    autograd::Engine::instance().clear_graph();
}
} // namespace
int main() try {
    const auto root = fs::temp_directory_path() / "mft_qwen_architecture_test";
    fs::create_directories(root);
    run(root);
    hybrid(root);
    fs::remove_all(root);
    std::cout << "PASS: Qwen config, exact targets, validation and dropout\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
