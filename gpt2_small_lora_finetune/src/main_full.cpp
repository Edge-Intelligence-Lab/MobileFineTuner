#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include "finetune_ops/graph/gpt2_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/core/tokenizer_bpe.h"
#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/optim/adam.h"
#include "finetune_ops/core/memory_manager.h"

using namespace std;
using namespace ops;

struct CmdArgs {
    string data_dir;
    string pretrained_dir;
    string output_path;
    string resume_from;
    string eval_out;

    int epochs = 0;
    int steps = 0;
    int batch_size = 1;
    int grad_accum_steps = 1;
    int seq_len = 128;
    float lr = 1e-4f;
    float weight_decay = 0.01f;
    int warmup_steps = 0;
    float clip_grad_norm = 1.0f;
    float data_fraction = 1.0f;

    int log_interval = 1;
    int eval_interval = 0;
    int eval_batches = 50;
    int eval_batch_size = 2;
    int save_every = 0;
    int seed = 42;
};

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------
static void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " [options]\n"
         << "Options:\n"
         << "  --data_dir PATH          Data directory\n"
         << "  --pretrained_dir PATH    Pretrained model directory\n"
         << "  --output_path PATH       Full parameter weights output (.safetensors)\n"
         << "  --resume_from PATH       Resume from safetensors (override pretrained weights)\n"
         << "  --eval_out PATH          Evaluation results output (JSONL)\n"
         << "  --epochs N               Training epochs (overrides steps)\n"
         << "  --steps N                Training steps\n"
         << "  --batch_size N           Batch size\n"
         << "  --grad_accum_steps N     Gradient accumulation steps\n"
         << "  --seq_len N              Sequence length\n"
         << "  --lr LR                  Learning rate\n"
         << "  --weight_decay WD        Weight decay\n"
         << "  --warmup_steps N         Warmup steps\n"
         << "  --clip_grad_norm F       Gradient clipping threshold\n"
         << "  --data_fraction F        Data fraction to use (0.0-1.0]\n"
         << "  --log_interval N         Logging interval\n"
         << "  --eval_interval N        Evaluation interval\n"
         << "  --eval_batches N         Evaluation batch count\n"
         << "  --eval_batch_size N      Evaluation batch size\n"
         << "  --save_every N           Checkpoint save interval\n"
         << "  --seed N                 Random seed\n";
}

static CmdArgs parse_args(int argc, char** argv) {
    CmdArgs args;
    args.data_dir = "/Users/tony/Documents/restart/data/wikitext2/wikitext-2-raw";
    args.pretrained_dir = "/Users/tony/Documents/restart/gpt2_lora_finetune/pretrained/gpt2";

    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto get_val = [&](const string& key)->string {
            if (i + 1 >= argc) { print_usage(argv[0]); exit(1); }
            if (k != key) { print_usage(argv[0]); exit(1); }
            return string(argv[++i]);
        };
        if (k == "--data_dir") args.data_dir = get_val("--data_dir");
        else if (k == "--pretrained_dir") args.pretrained_dir = get_val("--pretrained_dir");
        else if (k == "--output_path") args.output_path = get_val("--output_path");
        else if (k == "--resume_from") args.resume_from = get_val("--resume_from");
        else if (k == "--eval_out") args.eval_out = get_val("--eval_out");
        else if (k == "--epochs") args.epochs = stoi(get_val("--epochs"));
        else if (k == "--steps") args.steps = stoi(get_val("--steps"));
        else if (k == "--batch_size") args.batch_size = stoi(get_val("--batch_size"));
        else if (k == "--grad_accum_steps") args.grad_accum_steps = stoi(get_val("--grad_accum_steps"));
        else if (k == "--seq_len") args.seq_len = stoi(get_val("--seq_len"));
        else if (k == "--lr") args.lr = stof(get_val("--lr"));
        else if (k == "--weight_decay") args.weight_decay = stof(get_val("--weight_decay"));
        else if (k == "--warmup_steps") args.warmup_steps = stoi(get_val("--warmup_steps"));
        else if (k == "--clip_grad_norm") args.clip_grad_norm = stof(get_val("--clip_grad_norm"));
        else if (k == "--data_fraction") args.data_fraction = stof(get_val("--data_fraction"));
        else if (k == "--log_interval") args.log_interval = stoi(get_val("--log_interval"));
        else if (k == "--eval_interval") args.eval_interval = stoi(get_val("--eval_interval"));
        else if (k == "--eval_batches") args.eval_batches = stoi(get_val("--eval_batches"));
        else if (k == "--eval_batch_size") args.eval_batch_size = stoi(get_val("--eval_batch_size"));
        else if (k == "--save_every") args.save_every = stoi(get_val("--save_every"));
        else if (k == "--seed") args.seed = stoi(get_val("--seed"));
        else if (k == "--help" || k == "-h") { print_usage(argv[0]); exit(0); }
        else { cerr << "Unknown arg: " << k << endl; print_usage(argv[0]); exit(1); }
    }
    return args;
}

// -----------------------------------------------------------------------------
// Helper: gather named parameters for saving
// -----------------------------------------------------------------------------
static unordered_map<string, TensorPtr> collect_named_parameters(GPT2Model& model) {
    unordered_map<string, TensorPtr> named;
    auto params = model.parameters();

    if (params.size() < 2) return named;
    named["wte.weight"] = params[0];
    named["wpe.weight"] = params[1];

    const int layers = model.config().n_layer;
    for (int i = 0; i < layers; ++i) {
        auto& blk = model.get_block(i);
        const string prefix = "blocks." + to_string(i) + ".";
        named[prefix + "ln_1.weight"] = blk.ln_1_weight;
        named[prefix + "ln_1.bias"] = blk.ln_1_bias;
        named[prefix + "attn.qkv.weight"] = blk.attn_qkv_weight;
        named[prefix + "attn.qkv.bias"] = blk.attn_qkv_bias;
        named[prefix + "attn.proj.weight"] = blk.attn_proj_weight;
        named[prefix + "attn.proj.bias"] = blk.attn_proj_bias;
        named[prefix + "ln_2.weight"] = blk.ln_2_weight;
        named[prefix + "ln_2.bias"] = blk.ln_2_bias;
        named[prefix + "mlp.fc_in.weight"] = blk.mlp_fc_in_weight;
        named[prefix + "mlp.fc_in.bias"] = blk.mlp_fc_in_bias;
        named[prefix + "mlp.fc_out.weight"] = blk.mlp_fc_out_weight;
        named[prefix + "mlp.fc_out.bias"] = blk.mlp_fc_out_bias;
    }

    if (params.size() >= 2) {
        named["ln_f.weight"] = params[params.size() - 2];
        named["ln_f.bias"] = params[params.size() - 1];
    }
    return named;
}

// Minimal safetensors writer (float32 only, hf-key order)
static void save_full_safetensors(const string& path,
                                  GPT2Model& model,
                                  const unordered_map<string, string>& key_map) {
    auto named = collect_named_parameters(model);

    struct Entry {
        string name;
        TensorPtr tensor;
        size_t offset;
        size_t nbytes;
    };

    vector<Entry> entries;
    entries.reserve(key_map.size());

    for (const auto& kv : key_map) {
        const string& internal = kv.first;
        const string& hf = kv.second;
        auto it = named.find(internal);
        if (it == named.end()) {
            cerr << "[save] missing tensor for " << internal << endl;
            continue;
        }
        auto t = it->second;
        size_t bytes = static_cast<size_t>(t->numel()) * sizeof(float);
        entries.push_back({hf, t, 0, bytes});
    }

    auto build_header = [&](uint64_t base_offset) -> string {
        size_t off = base_offset;
        ostringstream header;
        header << "{";
        for (size_t i = 0; i < entries.size(); ++i) {
            auto& e = entries[i];
            e.offset = off;
            size_t end = off + e.nbytes;
            header << "\"" << e.name << "\":{";
            header << "\"dtype\":\"F32\",";
            header << "\"shape\":[";
            const auto& shp = e.tensor->shape();
            for (size_t j = 0; j < shp.size(); ++j) {
                if (j) header << ",";
                header << shp[j];
            }
            header << "],\"data_offsets\":[" << e.offset << "," << end << "]}";
            if (i + 1 < entries.size()) header << ",";
            off = end;
        }
        header << "}";
        return header.str();
    };

    string header_str;
    uint64_t base_offset = sizeof(uint64_t);  // 8 bytes header_len
    for (int iter = 0; iter < 3; ++iter) {
        header_str = build_header(base_offset);
        uint64_t new_base = sizeof(uint64_t) + static_cast<uint64_t>(header_str.size());
        if (new_base == base_offset) break;
        base_offset = new_base;
    }

    uint64_t header_len = static_cast<uint64_t>(header_str.size());

    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    ofstream ofs(path, ios::binary);
    if (!ofs.is_open()) {
        throw runtime_error("Failed to open " + path + " for writing");
    }

    ofs.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    ofs.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));

    for (const auto& e : entries) {
        const float* data = e.tensor->data<float>();
        ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(e.nbytes));
    }
    ofs.close();
    cout << "  [OK] Saved safetensors to " << path << endl;
}

static string make_checkpoint_path(const string& base, int step) {
    size_t dot_pos = base.find_last_of('.');
    string stem = (dot_pos == string::npos) ? base : base.substr(0, dot_pos);
    string ext = (dot_pos == string::npos) ? "" : base.substr(dot_pos);
    stringstream ss;
    ss << stem << "_step" << step << ext;
    return ss.str();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);

    try {
        cout << "\n========== GPT-2 Full Finetune ==========\n" << endl;
        cout << "[Configuration]" << endl;
#if defined(USE_BLAS)
        cout << "  BLAS          : ON (compiled)" << endl;
#else
        cout << "  BLAS          : OFF (compiled)" << endl;
#endif
        cout << "  data_dir       : " << args.data_dir << endl;
        cout << "  pretrained_dir : " << args.pretrained_dir << endl;
        cout << "  output_path    : " << args.output_path << endl;
        if (!args.resume_from.empty()) cout << "  resume_from    : " << args.resume_from << endl;
        cout << "  epochs         : " << args.epochs << endl;
        cout << "  batch_size     : " << args.batch_size << endl;
        cout << "  grad_accum     : " << args.grad_accum_steps << endl;
        cout << "  seq_len        : " << args.seq_len << endl;
        cout << "  lr/wd          : " << args.lr << "/" << args.weight_decay << endl;
        cout << "  warmup_steps   : " << args.warmup_steps << endl;
        cout << "  clip_grad_norm : " << args.clip_grad_norm << endl;
        cout << "  data_fraction  : " << args.data_fraction << endl;
        cout << "  log_interval   : " << args.log_interval << endl;
        cout << "  eval_interval  : " << args.eval_interval << endl;
        cout << "  save_every     : " << args.save_every << endl;
        cout << "  seed           : " << args.seed << endl;

        // 1) Load config & model
        cout << "\n[1/6] Loading pretrained model..." << endl;
        GPT2Config cfg;
        try {
            cfg = GPT2Config::from_pretrained(args.pretrained_dir);
            cout << "  [OK] Loaded config: layers=" << cfg.n_layer
                 << ", hidden=" << cfg.n_embd
                 << ", heads=" << cfg.n_head << endl;
        } catch (const std::exception& e) {
            cout << "  [WARN] Failed to read config, using built-in defaults: " << e.what() << endl;
        }
        if (args.seq_len > cfg.n_positions) {
            cout << "  [WARN] seq_len(" << args.seq_len << ") exceeds n_positions("
                 << cfg.n_positions << "), truncated" << endl;
            args.seq_len = cfg.n_positions;
        }

        GPT2Model model(cfg);
        model.tie_weights();

        const string weight_path = args.resume_from.empty()
            ? args.pretrained_dir + "/model.safetensors"
            : args.resume_from;

        SafeTensorsReader reader(weight_path);
        reader.parse_header();
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        SafeTensorsLoadOptions load_opts;
        load_opts.transpose_linear = false;  // HF GPT-2 weights already in [in,out] format
        load_opts.verbose = false;
        auto loaded = reader.load_tensors_mapped(key_map, load_opts);
        for (const auto& kv : loaded) {
            model.assign_weight(kv.first, kv.second);
        }
        cout << "  [OK] Weights loaded successfully" << endl;

        // Mark all parameters as trainable
        auto trainable = model.parameters();
        for (auto& p : trainable) {
            p->set_requires_grad(true);
            p->zero_grad();
        }

        // 2) Tokenizer & Data
        cout << "\n[2/6] Loading dataset..." << endl;
        auto tok_cfg = BPEConfig::from_pretrained(args.pretrained_dir);
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();

        WT2Config data_cfg;
        data_cfg.train_path = args.data_dir + "/wiki.train.raw";
        data_cfg.valid_path = args.data_dir + "/wiki.valid.raw";
        data_cfg.test_path  = args.data_dir + "/wiki.test.raw";
        data_cfg.seq_len = args.seq_len;
        data_cfg.stride  = -1;
        data_cfg.eos_id  = 50256;
        data_cfg.seed = args.seed;
        data_cfg.data_fraction = args.data_fraction;

        WikiText2Dataset train_dataset(data_cfg, &tokenizer);
        train_dataset.load(Split::Train);

        WT2Config valid_cfg = data_cfg;
        valid_cfg.drop_last = false;
        WikiText2Dataset valid_dataset(valid_cfg, &tokenizer);
        valid_dataset.load(Split::Valid);

        cout << "  [OK] Train set: " << train_dataset.num_sequences() << " sequences" << endl;
        cout << "  [OK] Valid set: " << valid_dataset.num_sequences() << " sequences" << endl;

        // Calculate training schedule
        const int64_t train_seqs = train_dataset.num_sequences();
        const int64_t micro_bs = args.batch_size;
        const int64_t accum = max(1, args.grad_accum_steps);
        const int64_t steps_per_epoch = (train_seqs + micro_bs * accum - 1) / (micro_bs * accum);
        if (args.epochs > 0) {
            args.steps = steps_per_epoch * args.epochs;
        } else if (args.steps <= 0) {
            args.steps = steps_per_epoch;  // Default: run 1 epoch
            cout << "  [WARN] steps/epochs not specified, defaulting steps = steps_per_epoch = "
                 << args.steps << endl;
        }

        cout << "\n[Training Schedule]" << endl;
        cout << "  epochs         : " << args.epochs << endl;
        cout << "  steps_per_epoch: " << steps_per_epoch << endl;
        cout << "  total_steps    : " << args.steps << endl;
        cout << "  effective_batch: " << (micro_bs * accum) << " (micro=" << micro_bs
             << " × accum=" << accum << ")" << endl;

        // 3) Optimizer
        cout << "\n[3/6] Initializing optimizer..." << endl;
        AdamConfig opt_cfg;
        opt_cfg.learning_rate = args.lr;
        opt_cfg.beta1 = 0.9f;
        opt_cfg.beta2 = 0.999f;
        opt_cfg.epsilon = 1e-8f;
        opt_cfg.weight_decay = args.weight_decay;
        opt_cfg.clip_grad_norm = args.clip_grad_norm;
        Adam optimizer(opt_cfg);
        cout << "  [OK] Adam optimizer ready" << endl;

        auto count_tokens = [](const TensorPtr& mask) -> int64_t {
            if (!mask) return 0;
            const float* ptr = mask->data<float>();
            int64_t total = 0;
            for (int64_t i = 0; i < mask->numel(); ++i) {
                if (ptr[i] > 0.5f) total++;
            }
            return total;
        };

        auto lr_schedule = [&](int step) -> float {
            const float base_lr = args.lr;
            const int T = max(1, args.steps);
            const int W = max(0, args.warmup_steps);
            if (step < W) {
                return base_lr * (float(step + 1) / float(max(1, W)));
            }
            const int s = step - W;
            const int d = max(1, T - W);
            const float min_lr = 0.1f * base_lr;
            const float PI = acosf(-1.0f);
            const float cosv = 0.5f * (1.0f + cos(PI * float(s) / float(d)));
            return min_lr + (base_lr - min_lr) * cosv;
        };

        auto clip_and_get_grad_norm = [&](float max_norm) -> float {
            double norm_sq = 0.0;
            for (const auto& p : trainable) {
                auto grad = p->grad();
                if (!grad) continue;
                const float* g = grad->data<float>();
                for (int64_t i = 0; i < grad->numel(); ++i) {
                    norm_sq += double(g[i]) * double(g[i]);
                }
            }
            float norm = float(sqrt(norm_sq));
            if (max_norm > 0.0f && norm > max_norm) {
                float scale = max_norm / (norm + 1e-6f);
                for (const auto& p : trainable) {
                    auto grad = p->grad();
                    if (!grad) continue;
                    float* g = grad->data<float>();
                    for (int64_t i = 0; i < grad->numel(); ++i) {
                        g[i] *= scale;
                    }
                }
                norm = max_norm;
            }
            return norm;
        };

        auto evaluate_valid = [&]() -> float {
            struct NoGradGuard {
                NoGradGuard() { ops::autograd::Engine::instance().set_enabled(false); }
                ~NoGradGuard() {
                    ops::autograd::Engine::instance().clear_graph();
                    ops::autograd::Engine::instance().set_enabled(true);
                }
            } _nograd_guard;

            valid_dataset.reset_cursor();
            double loss_sum = 0.0;
            int64_t token_sum = 0;
            int processed = 0;

            while (processed < args.eval_batches) {
                auto batch = valid_dataset.next_batch(args.eval_batch_size, false);
                if (!batch.input_ids) break;

                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                float loss_val = loss->data<float>()[0];
                int64_t tokens = count_tokens(batch.attention_mask);

                loss_sum += double(loss_val) * double(tokens);
                token_sum += tokens;
                processed++;
            }

            if (token_sum == 0) return 1e9f;
            float mean_loss = float(loss_sum / double(token_sum));
            return perplexity_from_loss(mean_loss);
        };

        // 4) Training
        cout << "\n[4/6] Starting training..." << endl;
        cout << "========================================\n" << endl;

        int64_t total_tokens = 0;

        for (int step = 0; step < args.steps; ++step) {
            const int cur_epoch = step / steps_per_epoch + 1;
            const int step_in_epoch = step % steps_per_epoch + 1;

            double accum_loss = 0.0;
            int64_t accum_tokens = 0;

            for (int acc = 0; acc < accum; ++acc) {
                auto batch = train_dataset.next_batch(micro_bs);
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                float loss_val = loss->data<float>()[0];

                accum_loss += loss_val;
                accum_tokens += count_tokens(batch.attention_mask);

                float inv_accum = 1.0f / float(accum);
                auto scaled_loss = mul(loss, inv_accum);
                scaled_loss->backward();
            }

            float grad_norm = clip_and_get_grad_norm(args.clip_grad_norm);
            float cur_lr = lr_schedule(step);
            optimizer.set_learning_rate(cur_lr);

            vector<TensorPtr> grads;
            grads.reserve(trainable.size());
            for (const auto& p : trainable) {
                grads.push_back(p->grad());
            }
            optimizer.step(trainable, grads);
            for (auto& p : trainable) {
                p->zero_grad();
            }

            if (((step + 1) % 50) == 0) {
                MemoryManager::instance().force_cleanup();
            }
            if (((step + 1) % 100) == 0) {
                MemoryManager::instance().clear_unused_memory();
            }

            float avg_loss = float(accum_loss / double(accum));
            float ppl = perplexity_from_loss(avg_loss);
            total_tokens += accum_tokens;

            if ((step + 1) % max(1, args.log_interval) == 0) {
                cout << "[Train] epoch " << cur_epoch << "/" << args.epochs
                     << " | step " << step_in_epoch << "/" << steps_per_epoch
                     << " (global " << (step + 1) << "/" << args.steps << ")"
                     << " | lr " << fixed << setprecision(6) << cur_lr
                     << " | loss " << setprecision(4) << avg_loss
                     << " | ppl " << setprecision(2) << ppl
                     << " | grad_norm " << setprecision(3) << grad_norm
                     << " | tokens " << accum_tokens
                     << endl;
            }

            if ((step + 1) % 100 == 0) {
                MemoryManager::instance().print_memory_stats();
            }

            if (args.eval_interval > 0 && (step + 1) % args.eval_interval == 0) {
                float valid_ppl = evaluate_valid();
                cout << "\n[Eval] epoch " << cur_epoch
                     << " | step " << (step + 1)
                     << " | valid_ppl " << fixed << setprecision(2) << valid_ppl
                     << " | total_tokens " << total_tokens
                     << "\n" << endl;

                if (!args.eval_out.empty()) {
                    stringstream ss;
                    ss << "{\"step\":" << (step + 1)
                       << ",\"epoch\":" << cur_epoch
                       << ",\"valid_ppl\":" << valid_ppl
                       << ",\"total_tokens\":" << total_tokens
                       << "}";
                    ofstream ofs(args.eval_out, ios::app);
                    if (ofs) ofs << ss.str() << "\n";
                }
            }

            if (args.save_every > 0 && (step + 1) % args.save_every == 0 && !args.output_path.empty()) {
                string ckpt_path = make_checkpoint_path(args.output_path, step + 1);
                cout << "\n[Checkpoint] Saving to " << ckpt_path << endl;
                save_full_safetensors(ckpt_path, model, key_map);
                cout << endl;
            }
        }

        // 5) Save final weights
        cout << "\n[5/6] Saving final weights..." << endl;
        if (!args.output_path.empty()) {
            save_full_safetensors(args.output_path, model, key_map);
        } else {
            cout << "  [WARN] output_path not specified, skipping save" << endl;
        }

        cout << "\n========================================" << endl;
        cout << "[DONE] Training completed!" << endl;
        cout << "  Total steps: " << args.steps << endl;
        cout << "  Total tokens: " << total_tokens << endl;
        cout << "========================================\n" << endl;
        return 0;

    } catch (const exception& e) {
        cerr << "\n[ERROR] Exception: " << e.what() << endl;
        return 1;
    }
}
