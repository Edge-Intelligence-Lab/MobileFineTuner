#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstdint>

#include "finetune_ops/core/tokenizer_bpe.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/optim/adam.h"
#include "finetune_ops/optim/smoke_utils.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/graph/qwen_model.h"

using namespace ops;

struct Args {
    std::string model_dir = "Qwen2.5-0.5B";
    std::string data_dir = "data/wikitext2/wikitext-2-raw";
    std::string jsonl_train;
    std::string jsonl_valid;
    std::string jsonl_test;
    int seq_len = 1024;
    int batch_size = 1;
    int grad_accum_steps = 1; // gradient accumulation steps
    int max_steps = -1; // -1 means auto-run one epoch
    float lr = 2e-4f;
    float max_grad_norm = 1.0f;
    int lora_r = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.05f;
    bool qv_only = false; // default q/k/v/o, matching QwenModel and formal QNLI experiments
    bool shuffle_train = true;
    uint64_t seed = 42;
    std::string loss_impl = "auto"; // auto: selected for masked JSONL, dense otherwise; full_dense/full_streaming: full-token JSONL labels
    bool align_mode = false; // alignment mode: single step with loss print
    bool synthetic_smoke = false;
    int smoke_steps = 2;
    std::string output_dir = "outputs/qwen_wikitext";
    std::string dump_first_batch_json;
    bool dump_only = false;
    std::string base_weight_storage = "auto"; // auto/native: preserve F16/BF16 frozen base weights; fp32: promote all
};

static void parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto get = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error("arg missing val");
            if (k != name) throw std::runtime_error("unexpected arg");
            return std::string(argv[++i]);
        };
        if (k == "--model_dir") args.model_dir = get("--model_dir");
        else if (k == "--data_dir") args.data_dir = get("--data_dir");
        else if (k == "--jsonl_train") args.jsonl_train = get("--jsonl_train");
        else if (k == "--jsonl_valid") args.jsonl_valid = get("--jsonl_valid");
        else if (k == "--jsonl_test") args.jsonl_test = get("--jsonl_test");
        else if (k == "--seq_len") args.seq_len = std::stoi(get("--seq_len"));
        else if (k == "--batch_size") args.batch_size = std::stoi(get("--batch_size"));
        else if (k == "--grad_accum_steps") args.grad_accum_steps = std::stoi(get("--grad_accum_steps"));
        else if (k == "--max_steps") args.max_steps = std::stoi(get("--max_steps"));
        else if (k == "--lr") args.lr = std::stof(get("--lr"));
        else if (k == "--max_grad_norm") args.max_grad_norm = std::stof(get("--max_grad_norm"));
        else if (k == "--lora_r") args.lora_r = std::stoi(get("--lora_r"));
        else if (k == "--lora_alpha") args.lora_alpha = std::stof(get("--lora_alpha"));
        else if (k == "--lora_dropout") args.lora_dropout = std::stof(get("--lora_dropout"));
        else if (k == "--qv_only") args.qv_only = true;
        else if (k == "--qkvo") args.qv_only = false;
        else if (k == "--seed") args.seed = static_cast<uint64_t>(std::stoull(get("--seed")));
        else if (k == "--no_shuffle") args.shuffle_train = false;
        else if (k == "--loss_impl") args.loss_impl = get("--loss_impl");
        else if (k == "--align_mode") args.align_mode = true;
        else if (k == "--synthetic_smoke") args.synthetic_smoke = true;
        else if (k == "--smoke_steps") args.smoke_steps = std::stoi(get("--smoke_steps"));
        else if (k == "--output_dir") args.output_dir = get("--output_dir");
        else if (k == "--dump_first_batch_json") args.dump_first_batch_json = get("--dump_first_batch_json");
        else if (k == "--dump_only") args.dump_only = true;
        else if (k == "--base_weight_storage") args.base_weight_storage = get("--base_weight_storage");
    }
}

static void require_file(const std::string& path, const std::string& desc) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(desc + " not found: " + path);
    }
}

static void configure_qwen_base_weight_storage(SafeTensorsLoadOptions& opts,
                                               const std::string& policy) {
    if (policy == "fp32") {
        return;
    }
    if (policy != "auto" && policy != "native") {
        throw std::runtime_error("--base_weight_storage must be one of: auto, native, fp32");
    }

    // Keep only large frozen matrices in their checkpoint dtype. Norms and
    // biases stay FP32 because their kernels intentionally remain FP32.
    opts.preserve_low_precision_key_substrings = {
        "embed_tokens.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight"
    };
}

static void save_lora_weights_bin(const std::string& output_dir, const std::vector<TensorPtr>& params) {
    std::filesystem::create_directories(output_dir);
    const std::string path = output_dir + "/lora_weights.bin";
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open LoRA output file: " + path);
    }

    int32_t num_params = static_cast<int32_t>(params.size());
    out.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
    for (const auto& param : params) {
        int32_t ndim = static_cast<int32_t>(param->shape().size());
        out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        for (int64_t dim : param->shape()) {
            out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }
        const int64_t numel = param->numel();
        out.write(reinterpret_cast<const char*>(param->data<float>()), numel * sizeof(float));
    }
}

static void dump_batch_json(const Batch& batch, const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open batch dump file: " + path);
    }

    const auto shape = batch.input_ids->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int32_t* ids = batch.input_ids->data<int32_t>();
    const float* attn = batch.attention_mask->data<float>();
    const int32_t* labels = batch.labels->data<int32_t>();

    out << "{\n";
    auto write_int_tensor = [&](const char* name, const int32_t* data) {
        out << "  \"" << name << "\": [\n";
        for (int64_t b = 0; b < B; ++b) {
            out << "    [";
            for (int64_t i = 0; i < S; ++i) {
                if (i) out << ", ";
                out << data[b * S + i];
            }
            out << "]";
            if (b + 1 < B) out << ",";
            out << "\n";
        }
        out << "  ]";
    };
    auto write_attn_tensor = [&](const char* name, const float* data) {
        out << "  \"" << name << "\": [\n";
        for (int64_t b = 0; b < B; ++b) {
            out << "    [";
            for (int64_t i = 0; i < S; ++i) {
                if (i) out << ", ";
                out << (data[b * S + i] > 0.5f ? 1 : 0);
            }
            out << "]";
            if (b + 1 < B) out << ",";
            out << "\n";
        }
        out << "  ]";
    };

    write_int_tensor("input_ids", ids);
    out << ",\n";
    write_attn_tensor("attention_mask", attn);
    out << ",\n";
    write_int_tensor("labels", labels);
    out << "\n}\n";
}

static int run_synthetic_smoke(const Args& args) {
    std::cout << "\n========== Qwen LoRA Synthetic Smoke ==========\n";

    QwenConfig qcfg;
    qcfg.vocab_size = 64;
    qcfg.hidden_size = 16;
    qcfg.intermediate_size = 32;
    qcfg.num_hidden_layers = 2;
    qcfg.num_attention_heads = 4;
    qcfg.num_key_value_heads = 2;
    qcfg.max_position_embeddings = std::max(32, args.seq_len + 4);
    qcfg.eos_token_id = 0;

    QwenModel model(qcfg);
    std::mt19937 rng(42);
    smoke::initialize_tiny_qwen(model, rng);

    model.init_lora(args.lora_r, args.lora_alpha, args.lora_dropout, args.qv_only, args.seed);
    model.freeze_base();
    auto lora_params = model.get_lora_parameters();
    if (lora_params.empty()) {
        throw std::runtime_error("synthetic smoke created no trainable Qwen LoRA parameters");
    }

    auto before = smoke::clone_tensors(lora_params);
    const int batch_size = std::max(1, std::min(args.batch_size, 2));
    const int seq_len = std::max(4, std::min(args.seq_len, 8));
    auto input_ids = smoke::make_input_ids(batch_size, seq_len, qcfg.vocab_size, 1);
    auto labels = smoke::make_shifted_labels(input_ids, qcfg.vocab_size);
    auto attention_mask = smoke::make_attention_mask(batch_size, seq_len);

    AdamConfig opt_cfg;
    opt_cfg.learning_rate = std::max(args.lr, 1e-2f);
    Adam opt(opt_cfg);

    const int smoke_steps = std::max(1, args.smoke_steps);
    for (int step = 0; step < smoke_steps; ++step) {
        auto hidden = model.forward_hidden(input_ids, attention_mask);
        auto loss = selected_token_lm_cross_entropy(hidden, model.embedding_weight(), labels, -100, "mean");
        if (!smoke::is_finite_scalar(loss)) {
            throw std::runtime_error("non-finite loss during Qwen synthetic smoke");
        }
        loss->backward();

        std::vector<TensorPtr> grads;
        grads.reserve(lora_params.size());
        for (const auto& p : lora_params) {
            grads.push_back(p->grad());
        }
        double grad_norm = smoke::grad_l2_norm(grads);
        if (!(grad_norm > 0.0) || !std::isfinite(grad_norm)) {
            throw std::runtime_error("invalid gradient norm during Qwen synthetic smoke");
        }

        opt.step(lora_params, grads);
        smoke::zero_grads(lora_params);
        smoke::cleanup_step_memory();

        std::cout << "[smoke step " << (step + 1) << "/" << smoke_steps
                  << "] loss=" << loss->data<float>()[0]
                  << " grad_norm=" << grad_norm << std::endl;
    }

    double param_delta = smoke::max_param_delta(before, lora_params);
    save_lora_weights_bin(args.output_dir, lora_params);
    std::cout << "[SyntheticSmoke] param_delta=" << param_delta << std::endl;
    const bool ok = std::isfinite(param_delta) && param_delta > 0.0;
    std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);
    try {
        if (args.synthetic_smoke) {
            return run_synthetic_smoke(args);
        }

        const bool use_jsonl =
            !args.jsonl_train.empty() || !args.jsonl_valid.empty() || !args.jsonl_test.empty();
        if (args.loss_impl != "auto" && args.loss_impl != "selected" &&
            args.loss_impl != "dense" && args.loss_impl != "full_dense" &&
            args.loss_impl != "full_streaming") {
            throw std::runtime_error("--loss_impl must be one of: auto, selected, dense, full_dense, full_streaming");
        }
        if (args.base_weight_storage != "auto" &&
            args.base_weight_storage != "native" &&
            args.base_weight_storage != "fp32") {
            throw std::runtime_error("--base_weight_storage must be one of: auto, native, fp32");
        }
        require_file(args.model_dir + "/config.json", "Qwen config");
        if (use_jsonl) {
            require_file(args.jsonl_train, "masked JSONL train split");
            if (!args.jsonl_valid.empty()) {
                require_file(args.jsonl_valid, "masked JSONL valid split");
            }
        } else {
            require_file(args.data_dir + "/wiki.train.raw", "WikiText-2 train split");
            require_file(args.data_dir + "/wiki.valid.raw", "WikiText-2 valid split");
        }

        std::filesystem::create_directories(args.output_dir);
        std::ofstream metrics(args.output_dir + "/step_metrics.csv");
        if (!metrics.is_open()) {
            throw std::runtime_error("failed to open metrics file in output_dir");
        }
        metrics << "step,loss,step_time_s,elapsed_s\n";

        std::cout << "\n========== Qwen2.5-0.5B LoRA Finetune ("
                  << (use_jsonl ? "masked JSONL" : "WikiText-2")
                  << ", C++) ==========\n";
        // 1) tokenizer
        auto tok_cfg = QwenTokenizerConfig::from_pretrained(args.model_dir);
        QwenBPETokenizer tokenizer(tok_cfg);
        tokenizer.load();

        // 2) config + model
        QwenConfig qcfg = QwenConfig::from_pretrained(args.model_dir + "/config.json");
        QwenModel model(qcfg);

        // 3) load weights
        SafeTensorsModelReader reader(args.model_dir);
        reader.parse_headers();
        auto mapping = QwenKeyMapper::generate_qwen_mapping(qcfg.num_hidden_layers);
        SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;           // disable per-tensor log to avoid flooding
        load_opts.transpose_linear = true;   // HF Linear [out,in] -> internal [in,out]
        configure_qwen_base_weight_storage(load_opts, args.base_weight_storage);
        auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (auto& kv : tensors) model.assign_weight(kv.first, kv.second);
        // QwenModel constructs placeholder tensors before real safetensors are
        // assigned. Release those replaced blocks before the first training step.
        MemoryManager::instance().clear_unused_memory();

        // 4) LoRA init
        model.init_lora(args.lora_r, args.lora_alpha, args.lora_dropout, args.qv_only, args.seed);
        model.freeze_base();
        auto lora_params = model.get_lora_parameters();

        // 5) dataset
        WT2Config dcfg;
        dcfg.seq_len = args.seq_len;
        dcfg.insert_eos_between_lines = true;
        dcfg.stride = -1;
        dcfg.eos_id = tok_cfg.eos_token_id;
        dcfg.pad_id = tok_cfg.eos_token_id >= 0 ? tok_cfg.eos_token_id : 0;
        dcfg.streaming_mode = false;
        dcfg.seed = args.seed;
        dcfg.shuffle_train = args.shuffle_train;
        dcfg.jsonl_full_token_labels =
            (args.loss_impl == "full_dense" || args.loss_impl == "full_streaming");
        if (use_jsonl) {
            dcfg.jsonl_train = args.jsonl_train;
            dcfg.jsonl_valid = args.jsonl_valid;
            dcfg.jsonl_test = args.jsonl_test;
        } else {
            dcfg.train_path = args.data_dir + "/wiki.train.raw";
            dcfg.valid_path = args.data_dir + "/wiki.valid.raw";
            dcfg.test_path  = args.data_dir + "/wiki.test.raw";
        }
        
        WikiText2Dataset ds(
            dcfg,
            [&](const std::string& text) -> std::vector<int32_t> {
                return tokenizer.encode(text);
            }
        );
        ds.load(Split::Train);
        
        // Print dataset info and derive steps per epoch
        size_t num_seqs = ds.num_sequences();
        size_t total_micro_batches = (num_seqs + args.batch_size - 1) / args.batch_size;
        size_t steps_per_epoch = (total_micro_batches + args.grad_accum_steps - 1) / args.grad_accum_steps;
        
        std::cout << "[Dataset] Total sequences: " << num_seqs << std::endl;
        std::cout << "[Config] Micro batch size: " << args.batch_size << std::endl;
        std::cout << "[Config] Grad accum steps: " << args.grad_accum_steps << std::endl;
        std::cout << "[Config] Effective batch size: " << args.batch_size * args.grad_accum_steps << std::endl;
        std::cout << "[Config] Max grad norm: " << args.max_grad_norm << std::endl;
        std::cout << "[Config] LoRA targets: " << (args.qv_only ? "q,v" : "q,k,v,o") << std::endl;
        std::cout << "[Config] Base weight storage: " << args.base_weight_storage << std::endl;
        std::cout << "[Config] Shuffle train: " << (args.shuffle_train ? "true" : "false") << std::endl;
        std::cout << "[Config] Seed: " << args.seed << std::endl;
        auto effective_loss_impl = [&]() {
            if (args.loss_impl == "full_streaming") return std::string("full_token_streaming");
            if (args.loss_impl == "selected") return std::string("selected_streaming");
            if (args.loss_impl == "auto") return use_jsonl ? std::string("selected_streaming") : std::string("dense");
            return args.loss_impl;
        };
        std::cout << "[Config] Loss impl: " << args.loss_impl
                  << " (effective=" << effective_loss_impl() << ")" << std::endl;
        std::cout << "[Dataset] Steps per epoch: " << steps_per_epoch << std::endl;

        if (!args.dump_first_batch_json.empty()) {
            auto first_batch = ds.next_batch(args.batch_size, false);
            dump_batch_json(first_batch, args.dump_first_batch_json);
            std::cout << "[Debug] First batch dumped to " << args.dump_first_batch_json << std::endl;
            ds.reset_cursor();
            if (args.dump_only) {
                return 0;
            }
        }
        
        // If max_steps <= 0, fall back to one epoch
        if (args.max_steps <= 0) {
            args.max_steps = steps_per_epoch;
            std::cout << "[Dataset] Auto-set max_steps to " << args.max_steps << " (1 epoch)" << std::endl;
        }

        // 6) optimizer
        AdamConfig opt_cfg;
        opt_cfg.learning_rate = args.lr;
        Adam opt(opt_cfg);

        // 7) training loop
        int step = 0;
        float accum_loss = 0.0f;
        int accum_counter = 0;
        auto train_start = std::chrono::steady_clock::now();
        auto step_start = train_start;
        
        auto batch = ds.next_batch(args.batch_size, false);
        
        while (step < args.max_steps && batch.input_ids) {
            if (accum_counter == 0) {
                step_start = std::chrono::steady_clock::now();
            }
            // Forward. The streaming loss is mathematically equivalent to
            // dense LM CE over all non-ignore shifted labels. In full_streaming
            // mode those labels cover every shifted token; in selected mode
            // they cover only the JSONL mask positions.
            TensorPtr loss;
            const bool use_streaming_lm_loss =
                (args.loss_impl == "selected") ||
                (args.loss_impl == "full_streaming") ||
                (args.loss_impl == "auto" && use_jsonl);
            if (use_streaming_lm_loss) {
                auto hidden = model.forward_hidden(batch.input_ids, batch.attention_mask);
                loss = streaming_lm_cross_entropy(
                    hidden, model.embedding_weight(), batch.labels, -100, "mean");
            } else {
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            }
            
            // accumulate loss
            accum_loss += loss->data<float>()[0];
            
            // Backward (with gradient accumulation)
            loss->backward();

            accum_counter++;
            
            // Update parameters once accumulation target is met
            if (accum_counter >= args.grad_accum_steps) {
                std::vector<TensorPtr> grads;
                grads.reserve(lora_params.size());

                if (args.grad_accum_steps > 1) {
                    float scale = 1.0f / static_cast<float>(args.grad_accum_steps);
                    for (auto& p : lora_params) {
                        auto g = p->grad();
                        if (g) {
                            float* d = g->data<float>();
                            int64_t n = g->numel();
                            for (int64_t i = 0; i < n; ++i) d[i] *= scale;
                        }
                    }
                }

                if (args.max_grad_norm > 0.0f) {
                    opt.clip_grad_norm(lora_params, args.max_grad_norm);
                }
                for (auto& p : lora_params) grads.push_back(p->grad());

                // Print gradient L2 on first step for observation
                if (step == 0) {
                    double gsum = 0.0;
                    int64_t gcount = 0;
                    for (auto& g : grads) {
                        if (!g) continue;
                        const float* d = g->data<float>();
                        for (int64_t i = 0; i < g->numel(); ++i) { gsum += d[i] * d[i]; gcount++; }
                    }
                    double gnorm = (gcount > 0) ? std::sqrt(gsum) : 0.0;
                    std::cout << "[debug] grad L2 (all LoRA params) = " << gnorm << std::endl;
                }

                opt.step(lora_params, grads);
                for (auto& p : lora_params) p->zero_grad();

                auto step_end = std::chrono::steady_clock::now();
                double step_time_s = std::chrono::duration<double>(step_end - step_start).count();
                double elapsed_s = std::chrono::duration<double>(step_end - train_start).count();
                float avg_loss = accum_loss / args.grad_accum_steps;

                metrics << (step + 1) << ","
                        << std::fixed << std::setprecision(6) << avg_loss << ","
                        << std::fixed << std::setprecision(6) << step_time_s << ","
                        << std::fixed << std::setprecision(6) << elapsed_s << "\n";
                metrics.flush();

                std::cout << "[step " << (step + 1) << "/" << args.max_steps
                          << "] loss=" << avg_loss
                          << " step_time_s=" << std::fixed << std::setprecision(4) << step_time_s
                          << " elapsed_s=" << std::fixed << std::setprecision(4) << elapsed_s
                          << std::endl;

                MemoryManager::instance().force_cleanup();
                if (((step + 1) % 20) == 0) MemoryManager::instance().clear_unused_memory();
                if ((step + 1) % 10 == 0)   MemoryManager::instance().print_memory_stats();
                
                accum_loss = 0.0f;
                accum_counter = 0;
                step++;
                
                if (args.align_mode) break; 
            } else {
                MemoryManager::instance().force_cleanup();
            }
            
            // Next batch
            batch = ds.next_batch(args.batch_size, false);
        }
        save_lora_weights_bin(args.output_dir, lora_params);
        std::cout << "[Metrics] Step metrics written to "
                  << args.output_dir + "/step_metrics.csv" << std::endl;
        std::cout << "✅ Finetune finished." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
