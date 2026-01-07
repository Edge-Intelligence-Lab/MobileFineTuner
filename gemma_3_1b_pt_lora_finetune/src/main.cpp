#include "finetune_ops/optim/gemma_trainer.h"
#include "finetune_ops/graph/gemma_model.h"
#include "finetune_ops/graph/gemma_lora_injector.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/core/tokenizer_gemma.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include <iostream>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <algorithm>

using namespace ops;

namespace {

std::vector<int> parse_layers(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            out.push_back(std::stoi(token));
        }
    }
    return out;
}

enum class DumpDType { kFloat32, kInt32 };

bool save_npy(const std::string& path,
              const void* data,
              const std::vector<int64_t>& shape,
              DumpDType dtype) {
    std::string descr = (dtype == DumpDType::kFloat32) ? "<f4" : "<i4";
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (shape.size() == 1) shape_str += ",";
        if (i + 1 < shape.size()) shape_str += ", ";
    }
    shape_str += ")";
    std::string header_dict = "{'descr': '" + descr +
        "', 'fortran_order': False, 'shape': " + shape_str + ", }";

    std::string magic = "\x93NUMPY";
    uint8_t ver_major = 1, ver_minor = 0;
    size_t header_len = header_dict.size() + 1;
    size_t preamble = magic.size() + 2 + 2;
    size_t padding = 16 - ((preamble + header_len) % 16);
    if (padding == 16) padding = 0;
    header_dict += std::string(padding, ' ');
    header_dict.push_back('\n');
    uint16_t header_size_le = static_cast<uint16_t>(header_dict.size());

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(magic.data(), magic.size());
    out.put(static_cast<char>(ver_major));
    out.put(static_cast<char>(ver_minor));
    out.write(reinterpret_cast<const char*>(&header_size_le), sizeof(header_size_le));
    out.write(header_dict.data(), header_dict.size());

    size_t count = 1;
    for (auto d : shape) count *= static_cast<size_t>(d);
    size_t elem_size = 4;
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * elem_size));
    return true;
}

void save_with_transpose_if_needed(const std::string& path,
                                   const TensorPtr& t,
                                   bool transpose2d = false) {
    auto shape = t->shape();
    std::vector<int64_t> shp(shape.begin(), shape.end());
    const void* data_ptr = t->data<void>();
    std::vector<float> temp;
    DumpDType dt = (t->dtype() == DType::kFloat32) ? DumpDType::kFloat32 : DumpDType::kInt32;
    if (transpose2d && shp.size() == 2 && dt == DumpDType::kFloat32) {
        int64_t rows = shp[0], cols = shp[1];
        temp.resize(static_cast<size_t>(rows * cols));
        const float* src = t->data<float>();
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                temp[static_cast<size_t>(c * rows + r)] = src[static_cast<size_t>(r * cols + c)];
            }
        }
        data_ptr = temp.data();
        shp = {cols, rows};
    }
    save_npy(path, data_ptr, shp, dt);
}

std::vector<float> compute_per_token_nll(const TensorPtr& logits,
                                         const TensorPtr& labels,
                                         int ignore_index) {
    auto shape = logits->shape();
    if (shape.size() != 3) throw std::runtime_error("logits rank must be 3");
    int64_t B = shape[0], S = shape[1], V = shape[2];
    const float* logit_data = logits->data<float>();
    const int32_t* label_data = labels->data<int32_t>();
    std::vector<float> output(static_cast<size_t>(B * (S - 1)), 0.0f);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S - 1; ++s) {
            int64_t label_idx = b * S + (s + 1);
            int label = label_data[label_idx];
            if (label == ignore_index) {
                output[static_cast<size_t>(b * (S - 1) + s)] = 0.0f;
                continue;
            }
            const float* row = logit_data + (b * S + s) * V;
            float maxv = row[0];
            for (int64_t j = 1; j < V; ++j) {
                if (row[j] > maxv) maxv = row[j];
            }
            double sum = 0.0;
            for (int64_t j = 0; j < V; ++j) {
                sum += std::exp(static_cast<double>(row[j] - maxv));
            }
            double logsum = std::log(sum) + static_cast<double>(maxv);
            double nll = logsum - static_cast<double>(row[label]);
            output[static_cast<size_t>(b * (S - 1) + s)] = static_cast<float>(nll);
        }
    }
    return output;
}

// Lightweight npy float32 loader for alignment debugging
bool load_npy_float32(const std::string& path,
                      std::vector<int64_t>& shape_out,
                      std::vector<float>& data_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[7] = {0};
    in.read(magic, 6);
    if (std::string_view(magic, 6) != "\x93NUMPY") return false;
    char ver[2];
    in.read(ver, 2);
    uint16_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    if (header.find("<f4") == std::string::npos) return false;
    auto l = header.find('(');
    auto r = header.find(')', l);
    if (l == std::string::npos || r == std::string::npos) return false;
    std::string shape_str = header.substr(l + 1, r - l - 1);
    shape_out.clear();
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        shape_out.push_back(std::stoll(tok));
    }
    size_t count = 1;
    for (auto d : shape_out) count *= static_cast<size_t>(d);
    data_out.resize(count);
    in.read(reinterpret_cast<char*>(data_out.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return in.good();
}

}  // namespace

struct CliOptions {
    std::string model_dir = "gemma-3-1b-pt";
    std::string data_dir = "data/wikitext2/wikitext-2-raw";
    std::string output_dir = "./gemma_lora";
    std::string jsonl_train;
    std::string jsonl_valid;
    std::string jsonl_test;
    std::string pretokenized_path;
    std::string pretokenized_meta;
    std::string align_dump_dir;
    std::string align_layers = "0,1,17";
    bool align_dump_grads = true;
    bool align_do_step = true;
    bool align_disable_debug = false;
    bool align_no_retain_grad = false;
    std::string align_pt_weights_dir;
    bool align_numeric_attn = false;
    float align_numeric_eps = 1e-3f;
    int align_numeric_count = 4;
    std::string align_numeric_targets = "attn_out_raw_l0";
    std::string target_mode = "full";
    std::string lora_targets_override;
    int epochs = 1;
    int max_steps = -1;
    int seq_len = 256;
    int batch = 4;
    int grad_accum = 1;
    float learning_rate = 2e-4f;
    float warmup_ratio = 0.03f;
    float max_grad_norm = 1.0f;
    float data_fraction = 1.0f;
    int save_every = 0;
    float weight_decay = 0.0f;
    std::string loss_reduction = "mean";
    bool dump_embedding = false;
    int dump_embedding_step = 1;
    std::string dump_embedding_dir = "./debug";
    int preview_tokens = 0;
};

CliOptions parse_cli(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto get_val = [&](const std::string& key) -> std::string {
            auto pos = arg.find('=');
            if (pos != std::string::npos) return arg.substr(pos + 1);
            if (i + 1 < argc) return argv[++i];
            return "";
        };
        if (arg.rfind("--model_dir", 0) == 0) {
            opts.model_dir = get_val("--model_dir");
        } else if (arg.rfind("--data_dir", 0) == 0) {
            opts.data_dir = get_val("--data_dir");
        } else if (arg.rfind("--jsonl_train", 0) == 0) {
            opts.jsonl_train = get_val("--jsonl_train");
        } else if (arg.rfind("--jsonl_valid", 0) == 0) {
            opts.jsonl_valid = get_val("--jsonl_valid");
        } else if (arg.rfind("--jsonl_test", 0) == 0) {
            opts.jsonl_test = get_val("--jsonl_test");
        } else if (arg.rfind("--pretokenized_path", 0) == 0) {
            opts.pretokenized_path = get_val("--pretokenized_path");
        } else if (arg.rfind("--pretokenized_meta", 0) == 0) {
            opts.pretokenized_meta = get_val("--pretokenized_meta");
        } else if (arg.rfind("--align_dump_dir", 0) == 0) {
            opts.align_dump_dir = get_val("--align_dump_dir");
        } else if (arg.rfind("--align_layers", 0) == 0) {
            opts.align_layers = get_val("--align_layers");
        } else if (arg.rfind("--align_dump_grads", 0) == 0) {
            auto v = get_val("--align_dump_grads");
            opts.align_dump_grads = (v != "0" && v != "false" && v != "False");
        } else if (arg.rfind("--align_do_step", 0) == 0) {
            auto v = get_val("--align_do_step");
            opts.align_do_step = (v != "0" && v != "false" && v != "False");
        } else if (arg.rfind("--align_disable_debug", 0) == 0) {
            auto v = get_val("--align_disable_debug");
            opts.align_disable_debug = (v != "0" && v != "false" && v != "False");
        } else if (arg.rfind("--align_no_retain_grad", 0) == 0) {
            auto v = get_val("--align_no_retain_grad");
            opts.align_no_retain_grad = (v != "0" && v != "false" && v != "False");
        } else if (arg.rfind("--align_pt_weights_dir", 0) == 0) {
            opts.align_pt_weights_dir = get_val("--align_pt_weights_dir");
        } else if (arg.rfind("--output_dir", 0) == 0) {
            opts.output_dir = get_val("--output_dir");
        } else if (arg.rfind("--targets", 0) == 0) {
            opts.target_mode = get_val("--targets");
        } else if (arg.rfind("--lora_targets", 0) == 0) {
            opts.lora_targets_override = get_val("--lora_targets");
        } else if (arg.rfind("--epochs", 0) == 0) {
            opts.epochs = std::stoi(get_val("--epochs"));
        } else if (arg.rfind("--max_steps", 0) == 0) {
            opts.max_steps = std::stoi(get_val("--max_steps"));
        } else if (arg.rfind("--seq_len", 0) == 0) {
            opts.seq_len = std::stoi(get_val("--seq_len"));
        } else if (arg.rfind("--batch", 0) == 0) {
            opts.batch = std::stoi(get_val("--batch"));
        } else if (arg.rfind("--grad_accum", 0) == 0) {
            opts.grad_accum = std::stoi(get_val("--grad_accum"));
        } else if (arg.rfind("--lr", 0) == 0) {
            opts.learning_rate = std::stof(get_val("--lr"));
        } else if (arg.rfind("--warmup_ratio", 0) == 0) {
            opts.warmup_ratio = std::stof(get_val("--warmup_ratio"));
        } else if (arg.rfind("--max_grad_norm", 0) == 0) {
            opts.max_grad_norm = std::stof(get_val("--max_grad_norm"));
        } else if (arg.rfind("--save_every", 0) == 0) {
            opts.save_every = std::stoi(get_val("--save_every"));
        } else if (arg.rfind("--weight_decay", 0) == 0) {
            opts.weight_decay = std::stof(get_val("--weight_decay"));
        } else if (arg.rfind("--loss_reduction", 0) == 0) {
            opts.loss_reduction = get_val("--loss_reduction");
        } else if (arg.rfind("--data_fraction", 0) == 0) {
            opts.data_fraction = std::stof(get_val("--data_fraction"));
        } else if (arg.rfind("--dump_embedding_step", 0) == 0) {
            opts.dump_embedding_step = std::stoi(get_val("--dump_embedding_step"));
        } else if (arg.rfind("--dump_embedding_dir", 0) == 0) {
            opts.dump_embedding_dir = get_val("--dump_embedding_dir");
        } else if (arg.rfind("--dump_embedding", 0) == 0) {
            auto val = get_val("--dump_embedding");
            opts.dump_embedding = (val == "1" || val == "true" || val == "True");
        } else if (arg.rfind("--preview_tokens", 0) == 0) {
            opts.preview_tokens = std::stoi(get_val("--preview_tokens"));
        } else if (arg.rfind("--align_numeric_attn", 0) == 0) {
            auto v = get_val("--align_numeric_attn");
            opts.align_numeric_attn = (v != "0" && v != "false" && v != "False");
        } else if (arg.rfind("--align_numeric_eps", 0) == 0) {
            opts.align_numeric_eps = std::stof(get_val("--align_numeric_eps"));
        } else if (arg.rfind("--align_numeric_count", 0) == 0) {
            opts.align_numeric_count = std::stoi(get_val("--align_numeric_count"));
        } else if (arg.rfind("--align_numeric_targets", 0) == 0) {
            opts.align_numeric_targets = get_val("--align_numeric_targets");
        }
    }
    return opts;
}

int main(int argc, char** argv) {
    try {
        auto cli = parse_cli(argc, argv);

        std::cout << "========== Gemma LoRA Finetune ==========\n" << std::endl;
        std::cout << "[INFO] model_dir=" << cli.model_dir << std::endl;
        if (!cli.pretokenized_path.empty()) {
            std::cout << "[INFO] pretokenized_path=" << cli.pretokenized_path << std::endl;
            if (!cli.pretokenized_meta.empty()) {
                std::cout << "[INFO] pretokenized_meta=" << cli.pretokenized_meta << std::endl;
            }
        } else if (!cli.jsonl_train.empty() || !cli.jsonl_valid.empty() || !cli.jsonl_test.empty()) {
            std::cout << "[INFO] data_source=JSONL(masked)" << std::endl;
            if (!cli.jsonl_train.empty()) std::cout << "[INFO] jsonl_train=" << cli.jsonl_train << std::endl;
            if (!cli.jsonl_valid.empty()) std::cout << "[INFO] jsonl_valid=" << cli.jsonl_valid << std::endl;
            if (!cli.jsonl_test.empty())  std::cout << "[INFO] jsonl_test="  << cli.jsonl_test  << std::endl;
        } else {
            std::cout << "[INFO] data_dir=" << cli.data_dir << std::endl;
        }

        auto cfg = GemmaTextConfig::from_pretrained(cli.model_dir);
        GemmaModel model(cfg);
        if (cli.align_no_retain_grad) {
            model.set_debug_retain_grads(false);
        }

        SafeTensorsReader reader(cli.model_dir + "/model.safetensors");
        reader.parse_header();
        auto mapping = GemmaKeyMapper::generate_gemma_mapping(cfg.num_hidden_layers);
        SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;
        auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (auto& kv : tensors) {
            model.assign_weight(kv.first, kv.second);
        }
        std::cout << "✅ Gemma weights loaded\n";

        std::unique_ptr<GemmaTokenizer> tokenizer;
        std::function<std::vector<int32_t>(const std::string&)> encode_fn;
        if (cli.pretokenized_path.empty()) {
            auto tok_cfg = GemmaTokenizerConfig::from_pretrained(cli.model_dir);
            tokenizer = std::make_unique<GemmaTokenizer>(tok_cfg);
            tokenizer->load();

            encode_fn = [tok = tokenizer.get()](const std::string& text) {
                auto ids = tok->encode(text, false, 0, false);
                return std::vector<int32_t>(ids.begin(), ids.end());
            };
        } else {
            encode_fn = [](const std::string&) {
                return std::vector<int32_t>{};
            };
        }

        GemmaLoraSpec lora_spec = GemmaLoraSpec::full_attn_mlp();
        if (cli.target_mode == "light") {
            lora_spec = GemmaLoraSpec::attention_light();
        } else if (cli.target_mode == "attn") {
            lora_spec = GemmaLoraSpec::attention_only();
        }
        if (!cli.lora_targets_override.empty()) {
            lora_spec.target_modules.clear();
            std::stringstream ss(cli.lora_targets_override);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) lora_spec.target_modules.push_back(tok);
            }
        }
        if (!cli.align_dump_dir.empty()) {
            lora_spec.dropout = 0.0f;
        }
        GemmaLoraInjector injector;
        injector.inject(model, lora_spec);
        injector.print_info();

        if (!cli.align_pt_weights_dir.empty()) {
            auto load_lora_a = [&](LoRALinear* lora) {
                if (!lora) return;
                for (auto& pair : lora->debug_params()) {
                    if (pair.first.find("lora_A") == std::string::npos) continue;
                    std::string fname = cli.align_pt_weights_dir + "/weights_after_step/base_model_model_model_" + pair.first + ".npy";
                    if (!std::filesystem::exists(fname)) continue;
                    std::vector<int64_t> shp;
                    std::vector<float> data;
                    if (!load_npy_float32(fname, shp, data)) {
                        std::cerr << "[Align] Failed to load " << fname << std::endl;
                        continue;
                    }
                    auto expect = pair.second->shape();
                    size_t expect_count = 1;
                    bool shape_ok = (shp.size() == expect.size());
                    for (size_t i = 0; i < expect.size(); ++i) {
                        expect_count *= static_cast<size_t>(expect[i]);
                        if (!shape_ok || shp[i] != expect[i]) shape_ok = false;
                    }
                    if (!shape_ok || expect_count != data.size()) {
                        std::cerr << "[Align] Shape mismatch for " << fname << std::endl;
                        continue;
                    }
                    std::memcpy(pair.second->data<float>(), data.data(), expect_count * sizeof(float));
                    std::cout << "[Align] Loaded PT LoRA A from " << fname << std::endl;
                }
            };
            for (int l = 0; l < cfg.num_hidden_layers; ++l) {
                auto& blk = model.get_block(l);
                load_lora_a(blk.q_proj_lora.get());
                load_lora_a(blk.k_proj_lora.get());
                load_lora_a(blk.v_proj_lora.get());
                load_lora_a(blk.o_proj_lora.get());
            }
        }

        WT2Config data_cfg;
        data_cfg.seq_len = cli.seq_len;
        data_cfg.data_fraction = cli.data_fraction;
        if (!cli.align_dump_dir.empty()) {
            data_cfg.shuffle_train = false;
        }
        if (!cli.jsonl_train.empty() || !cli.jsonl_valid.empty() || !cli.jsonl_test.empty()) {
            data_cfg.jsonl_train = cli.jsonl_train;
            data_cfg.jsonl_valid = cli.jsonl_valid;
            data_cfg.jsonl_test  = cli.jsonl_test;
            data_cfg.streaming_mode = false;
            data_cfg.shuffle_train = true;
            data_cfg.eos_id = -1;
            data_cfg.pad_id = 0;
        } else if (cli.pretokenized_path.empty()) {
            data_cfg.train_path = cli.data_dir + "/wiki.train.raw";
            data_cfg.valid_path = cli.data_dir + "/wiki.valid.raw";
            data_cfg.eos_id = tokenizer->get_eos_token_id();
            int pad_id = tokenizer->get_pad_token_id();
            data_cfg.pad_id = (pad_id >= 0) ? pad_id : data_cfg.eos_id;
        } else {
            data_cfg.pretokenized_path = cli.pretokenized_path;
            data_cfg.pretokenized_meta = cli.pretokenized_meta;
            data_cfg.eos_id = -1;
            data_cfg.pad_id = 0;
            data_cfg.streaming_mode = false;
            data_cfg.insert_eos_between_lines = true;
        }

        WikiText2Dataset train_data(data_cfg, encode_fn);
        train_data.load(Split::Train);
        WikiText2Dataset eval_data(data_cfg, encode_fn);
        eval_data.load(Split::Valid);

        std::cout << "Train sequences: " << train_data.num_sequences()
                  << ", Eval sequences: " << eval_data.num_sequences() << std::endl;

        if (!cli.align_dump_dir.empty()) {
            auto layers = parse_layers(cli.align_layers);
            bool dump_all_layers = layers.empty();
            std::vector<int> target_layers = dump_all_layers ? std::vector<int>{0, 1, 17} : layers;
            std::sort(target_layers.begin(), target_layers.end());
            std::cout << "[Align] align_layers="
                      << (cli.align_layers.empty() ? "(empty)" : cli.align_layers)
                      << " parsed=";
            if (dump_all_layers) {
                std::cout << "ALL";
            } else {
                for (size_t i = 0; i < target_layers.size(); ++i) {
                    if (i) std::cout << ",";
                    std::cout << target_layers[i];
                }
            }
            std::cout << std::endl;

            if (cli.align_disable_debug) {
                std::cout << "[Align] debug dump disabled (no intermediate hooks/retain)" << std::endl;
            } else {
                model.enable_debug_dump(cli.align_dump_dir, layers);
            }
            auto batch = train_data.next_batch(1, false);
            {
                if (batch.input_ids) {
                    save_npy(cli.align_dump_dir + "/input_ids.npy",
                             batch.input_ids->data<int32_t>(),
                             batch.input_ids->shape(),
                             DumpDType::kInt32);
                }
                if (batch.attention_mask) {
                    save_npy(cli.align_dump_dir + "/attention_mask.npy",
                             batch.attention_mask->data<float>(),
                             batch.attention_mask->shape(),
                             DumpDType::kFloat32);
                }
                if (batch.labels) {
                    save_npy(cli.align_dump_dir + "/labels.npy",
                             batch.labels->data<int32_t>(),
                             batch.labels->shape(),
                             DumpDType::kInt32);
                }
                for (int layer : target_layers) {
                    model.dump_layer_norm_weights(layer, cli.align_dump_dir);
                }
            }
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            logits->set_requires_grad(true);
            logits->retain_grad();
            auto loss = lm_cross_entropy(logits, batch.labels, -100, cli.loss_reduction);

            auto per_token = compute_per_token_nll(logits, batch.labels, -100);
            int64_t S = batch.labels->shape()[1];
            save_npy(cli.align_dump_dir + "/per_token_nll.npy",
                     per_token.data(),
                     {batch.labels->shape()[0], S - 1},
                     DumpDType::kFloat32);
            float loss_val = loss->data<float>()[0];
            save_npy(cli.align_dump_dir + "/loss_scalar.npy", &loss_val, {1}, DumpDType::kFloat32);

            if (cli.align_dump_grads) {
                loss->backward();
                float base_loss = loss_val;
                auto dump_lora = [&](const TensorPtr& t, const std::string& name) {
                    if (!t || !t->grad()) return;
                    auto grad = t->grad();
                    auto shape = grad->shape();
                    std::vector<int64_t> shp(shape.begin(), shape.end());
                    const float* data = grad->data<float>();
                    if (shp.size() == 3 && shp[0] == 1) {
                        shp.erase(shp.begin());
                    }
                    save_npy(cli.align_dump_dir + "/grads/" + name + ".npy",
                             data,
                             shp,
                             DumpDType::kFloat32);
                };

                for (int layer : target_layers) {
                    auto& blk = model.get_block(layer);
                    for (auto pair : blk.q_proj_lora->debug_params()) dump_lora(pair.second, "base_model_model_model_" + pair.first);
                    for (auto pair : blk.o_proj_lora->debug_params()) dump_lora(pair.second, "base_model_model_model_" + pair.first);
                }

                if (logits->grad()) {
                    save_npy(cli.align_dump_dir + "/dlogits.npy",
                             logits->grad()->data<float>(),
                             logits->grad()->shape(),
                             DumpDType::kFloat32);
                }
                if (!cli.align_disable_debug) {
                    auto debug_map = model.debug_tensors();
                    auto dump_grad = [&](const std::string& key) {
                        auto it = debug_map.find(key);
                        if (it == debug_map.end()) return;
                        auto t = it->second;
                        if (!t || !t->grad()) return;
                        std::vector<int64_t> shp = t->grad()->shape();
                        if (!shp.empty() && shp[0] == 1) {
                            shp.erase(shp.begin());
                        }
                        save_npy(cli.align_dump_dir + "/grads/" + key + ".npy",
                                 t->grad()->data<float>(),
                                 shp,
                                 DumpDType::kFloat32);
                    };
                    dump_grad("hidden_states_emb");
                    const std::vector<std::string> attn_names = {
                        "hidden_before_attn",
                        "hidden_after_attn",
                        "hidden_after_mlp",
                        "hidden_after_attn_norm",
                        "hidden_after_mlp_norm",
                        "q_proj_out",
                        "k_proj_out",
                        "v_proj_out",
                        "q_norm_out",
                        "k_norm_out",
                        "q_norm_out_pre_rope",
                        "k_norm_out_pre_rope",
                        "q_rotary_out",
                        "k_rotary_out",
                        "attn_context",
                        "attn_scores",
                        "attn_probs",
                        "attn_out_raw"};
                    const std::vector<std::string> mlp_names = {
                        "hidden_before_mlp_norm",
                        "gate_proj_out",
                        "gate_act",
                        "up_proj_out",
                        "mlp_prod",
                        "down_proj_out"};
                    for (int layer : target_layers) {
                        for (const auto& base : attn_names) {
                            dump_grad(base + "_l" + std::to_string(layer));
                        }
                        for (const auto& base : mlp_names) {
                            dump_grad(base + "_l" + std::to_string(layer));
                        }
                    }

                    {
                        auto debug_map_eval = model.debug_tensors();
                        auto fetch = [&](const std::string& key) -> TensorPtr {
                            auto it = debug_map_eval.find(key);
                            if (it == debug_map_eval.end()) return nullptr;
                            return it->second;
                        };
                        const float qk_scale = std::pow(model.config().query_pre_attn_scalar, -0.5f);
                        for (int layer : target_layers) {
                            const std::string L = std::to_string(layer);
                            auto scores = fetch("attn_scores_l" + L);
                            auto probs  = fetch("attn_probs_l"  + L);
                            auto q      = fetch("q_rotary_out_l" + L);
                            auto k_in   = fetch("k_rotary_out_l" + L);
                            auto ctx    = fetch("attn_context_l" + L);
                            if (!scores || !scores->grad() || !probs || !q || !k_in) continue;

                            auto dScores = scores->grad();
                            auto dMat = mul(dScores, qk_scale);
                            {
                                std::vector<int64_t> shp = dMat->shape();
                                if (!shp.empty() && shp[0] == 1) shp.erase(shp.begin());
                                save_npy(cli.align_dump_dir + "/grads/attn_dmat_l" + L + ".npy",
                                         dMat->data<float>(), shp, DumpDType::kFloat32);
                            }
                            int H = model.config().num_attention_heads;
                            int kvH = model.config().num_key_value_heads;
                            int repeat_factor = std::max(1, H / std::max(1, kvH));
                            auto k_full = repeat_kv_heads(k_in, repeat_factor);
                            auto dQ_heads = matmul(dMat, k_full);
                            {
                                std::vector<int64_t> shp = dQ_heads->shape();
                                if (!shp.empty() && shp[0] == 1) shp.erase(shp.begin());
                                save_npy(cli.align_dump_dir + "/grads/analytical_dq_heads_l" + L + ".npy",
                                         dQ_heads->data<float>(), shp, DumpDType::kFloat32);
                            }
                            auto dK_heads = matmul(transpose(dMat, -2, -1), q);
                            {
                                std::vector<int64_t> shp = dK_heads->shape();
                                if (!shp.empty() && shp[0] == 1) shp.erase(shp.begin());
                                save_npy(cli.align_dump_dir + "/grads/analytical_dk_heads_l" + L + ".npy",
                                         dK_heads->data<float>(), shp, DumpDType::kFloat32);
                            }
                            if (ctx && ctx->grad()) {
                                auto G = ctx->grad();
                                const auto& gshape = G->shape();
                                if (gshape.size() == 3) {
                                    int64_t B = gshape[0], S = gshape[1];
                                    int64_t hidden = gshape[2];
                                    int64_t D = model.config().head_dim;
                                    int64_t Hcfg = model.config().num_attention_heads;
                                    if (hidden == Hcfg * D) {
                                        auto G_heads = permute(reshape(G, {B, S, Hcfg, D}), {0, 2, 1, 3});
                                        auto dV_heads = matmul(transpose(probs, -2, -1), G_heads);
                                        std::vector<int64_t> shp = dV_heads->shape();
                                        if (!shp.empty() && shp[0] == 1) shp.erase(shp.begin());
                                        save_npy(cli.align_dump_dir + "/grads/analytical_dv_heads_l" + L + ".npy",
                                                 dV_heads->data<float>(), shp, DumpDType::kFloat32);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            std::cout << "[AlignDump] wrote activations and loss to " << cli.align_dump_dir << std::endl;
            return 0;
        }

        if (cli.preview_tokens > 0) {
            auto head = train_data.peek_tokens(static_cast<size_t>(cli.preview_tokens));
            std::cout << "First " << head.size() << " train tokens: [";
            for (size_t i = 0; i < head.size(); ++i) {
                std::cout << head[i];
                if (i + 1 < head.size()) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        GemmaTrainerConfig trainer_cfg;
        trainer_cfg.learning_rate = cli.learning_rate;
        trainer_cfg.num_epochs = cli.epochs;
        trainer_cfg.micro_batch_size = std::max(1, cli.batch);
        trainer_cfg.grad_accum_steps = std::max(1, cli.grad_accum);
        trainer_cfg.output_dir = cli.output_dir;
        trainer_cfg.max_steps = cli.max_steps;
        trainer_cfg.eval_steps = 0;
        trainer_cfg.logging_steps = 1;
        trainer_cfg.save_every = std::max(0, cli.save_every);
        trainer_cfg.warmup_ratio = cli.warmup_ratio;
        trainer_cfg.max_grad_norm = cli.max_grad_norm;
        trainer_cfg.weight_decay = cli.weight_decay;
        trainer_cfg.dump_embedding = cli.dump_embedding;
        trainer_cfg.dump_embedding_step = std::max(1, cli.dump_embedding_step);
        trainer_cfg.dump_embedding_dir = cli.dump_embedding_dir;

        GemmaLoRATrainer trainer(model, injector, train_data, eval_data, trainer_cfg);
        trainer.train();

        std::cout << "[Save] Writing LoRA adapter..." << std::endl;
        trainer.save_lora(trainer_cfg.output_dir + "/gemma_lora.safetensors");

        std::cout << "🎉 Gemma LoRA training finished" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}


