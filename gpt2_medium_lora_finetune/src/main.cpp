#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "finetune_ops/graph/gpt2_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/graph/lora_injector.h"
#include "finetune_ops/graph/lora_saver.h"
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
    string lora_out;
    string resume_from;
    string eval_out;
    // Optional: masked JSONL inputs for supervised LM (ids/mask)
    string jsonl_train;
    string jsonl_valid;
    string jsonl_test;
    
    int epochs = 0;
    int steps = 0;
    int batch_size = 1;
    int grad_accum_steps = 1;
    int seq_len = 128;
    int rank = 8;
    float alpha = 16.0f;
    float lr = 1e-4f;
    float weight_decay = 0.0f;
    int warmup_steps = 0;
    float clip_grad_norm = 1.0f;
    float lora_dropout = 0.0f;
    
    int log_interval = 1;
    int eval_interval = 0;
    int eval_batches = 50;
    int eval_batch_size = 2;
    int save_every = 0;
    float ema_beta = 0.9f;
    int seed = 42;
};

static void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " [options]\n"
         << "Options:\n"
         << "  --data_dir PATH          Data directory (WikiText-2 raw)\n"
         << "  --pretrained_dir PATH    Pretrained model directory\n"
         << "  --jsonl_train PATH       JSONL train (optional; contains ids/mask)\n"
         << "  --jsonl_valid PATH       JSONL valid (optional; contains ids/mask)\n"
         << "  --jsonl_test PATH        JSONL test  (optional; contains ids/mask)\n"
         << "  --lora_out PATH          Output path for LoRA adapter\n"
         << "  --resume_from PATH       Resume from LoRA checkpoint\n"
         << "  --eval_out PATH          Write eval metrics to JSONL\n"
         << "  --epochs N               Num epochs (overrides steps if >0)\n"
         << "  --steps N                Num training steps\n"
         << "  --batch_size N           Micro-batch size\n"
         << "  --grad_accum_steps N     Gradient accumulation steps\n"
         << "  --seq_len N              Sequence length\n"
         << "  --rank R                 LoRA rank\n"
         << "  --alpha A                LoRA alpha\n"
         << "  --lr LR                  Learning rate\n"
         << "  --weight_decay WD        Weight decay\n"
         << "  --warmup_steps N         Warmup steps\n"
         << "  --clip_grad_norm F       Gradient clipping threshold\n"
         << "  --lora_dropout F         LoRA dropout\n"
         << "  --log_interval N         Logging interval (steps)\n"
         << "  --eval_interval N        Eval interval (steps; 0=disable)\n"
         << "  --eval_batches N         Num eval batches\n"
         << "  --eval_batch_size N      Eval micro-batch size\n"
         << "  --save_every N           Checkpoint interval (steps)\n"
         << "  --ema_beta F             EMA smoothing factor\n"
         << "  --seed N                 Random seed\n";
}

static CmdArgs parse_args(int argc, char** argv) {
    CmdArgs args;
    args.data_dir = "/Users/tony/Documents/FT/data/wikitext2/wikitext-2-raw";
    args.pretrained_dir = "/Users/tony/Documents/FT/gpt2_lora_finetune/pretrained/gpt2";

    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto get_val = [&](const string& key)->string {
            if (i + 1 >= argc) { print_usage(argv[0]); exit(1); }
            if (k != key) { print_usage(argv[0]); exit(1); }
            return string(argv[++i]);
        };
        if (k == "--data_dir") args.data_dir = get_val("--data_dir");
        else if (k == "--pretrained_dir") args.pretrained_dir = get_val("--pretrained_dir");
        else if (k == "--jsonl_train") args.jsonl_train = get_val("--jsonl_train");
        else if (k == "--jsonl_valid") args.jsonl_valid = get_val("--jsonl_valid");
        else if (k == "--jsonl_test") args.jsonl_test = get_val("--jsonl_test");
        else if (k == "--lora_out") args.lora_out = get_val("--lora_out");
        else if (k == "--resume_from") args.resume_from = get_val("--resume_from");
        else if (k == "--eval_out") args.eval_out = get_val("--eval_out");
        else if (k == "--epochs") args.epochs = stoi(get_val("--epochs"));
        else if (k == "--steps") args.steps = stoi(get_val("--steps"));
        else if (k == "--batch_size") args.batch_size = stoi(get_val("--batch_size"));
        else if (k == "--grad_accum_steps") args.grad_accum_steps = stoi(get_val("--grad_accum_steps"));
        else if (k == "--seq_len") args.seq_len = stoi(get_val("--seq_len"));
        else if (k == "--rank") args.rank = stoi(get_val("--rank"));
        else if (k == "--alpha") args.alpha = stof(get_val("--alpha"));
        else if (k == "--lr") args.lr = stof(get_val("--lr"));
        else if (k == "--weight_decay") args.weight_decay = stof(get_val("--weight_decay"));
        else if (k == "--warmup_steps") args.warmup_steps = stoi(get_val("--warmup_steps"));
        else if (k == "--clip_grad_norm") args.clip_grad_norm = stof(get_val("--clip_grad_norm"));
        else if (k == "--lora_dropout") args.lora_dropout = stof(get_val("--lora_dropout"));
        else if (k == "--log_interval") args.log_interval = stoi(get_val("--log_interval"));
        else if (k == "--eval_interval") args.eval_interval = stoi(get_val("--eval_interval"));
        else if (k == "--eval_batches") args.eval_batches = stoi(get_val("--eval_batches"));
        else if (k == "--eval_batch_size") args.eval_batch_size = stoi(get_val("--eval_batch_size"));
        else if (k == "--save_every") args.save_every = stoi(get_val("--save_every"));
        else if (k == "--ema_beta") args.ema_beta = stof(get_val("--ema_beta"));
        else if (k == "--seed") args.seed = stoi(get_val("--seed"));
        else if (k == "--help" || k == "-h") { print_usage(argv[0]); exit(0); }
        else { cerr << "Unknown arg: " << k << endl; print_usage(argv[0]); exit(1); }
    }
    return args;
}

static void append_jsonl(const string& path, const string& json_str) {
    ofstream ofs(path, ios::app);
    if (ofs) {
        ofs << json_str << "\n";
    }
}

static string make_checkpoint_path(const string& base, int step) {
    size_t dot_pos = base.find_last_of('.');
    string stem = (dot_pos == string::npos) ? base : base.substr(0, dot_pos);
    string ext = (dot_pos == string::npos) ? "" : base.substr(dot_pos);
    stringstream ss;
    ss << stem << "_step" << step << ext;
    return ss.str();
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);

    try {
        cout << "\n========== GPT-2 LoRA Finetune (Pro) ==========\n" << endl;
        cout << "[Config]" << endl;
#if defined(USE_BLAS)
        cout << "  BLAS          : ON (compiled)" << endl;
#else
        cout << "  BLAS          : OFF (compiled)" << endl;
#endif
        if (!args.jsonl_train.empty() || !args.jsonl_valid.empty() || !args.jsonl_test.empty()) {
            cout << "  data_source    : JSONL (masked Causal LM)" << endl;
            if (!args.jsonl_train.empty()) cout << "  jsonl_train    : " << args.jsonl_train << endl;
            if (!args.jsonl_valid.empty()) cout << "  jsonl_valid    : " << args.jsonl_valid << endl;
            if (!args.jsonl_test.empty())  cout << "  jsonl_test     : " << args.jsonl_test  << endl;
        } else {
            cout << "  data_dir       : " << args.data_dir << endl;
        }
        cout << "  pretrained_dir : " << args.pretrained_dir << endl;
        cout << "  lora_out       : " << args.lora_out << endl;
        if (!args.resume_from.empty()) {
            cout << "  resume_from    : " << args.resume_from << endl;
        }
        if (!args.eval_out.empty()) {
            cout << "  eval_out       : " << args.eval_out << endl;
        }
        cout << "  epochs         : " << args.epochs << endl;
        cout << "  batch_size     : " << args.batch_size << endl;
        cout << "  grad_accum     : " << args.grad_accum_steps << endl;
        cout << "  seq_len        : " << args.seq_len << endl;
        cout << "  rank/alpha     : " << args.rank << "/" << args.alpha << endl;
        cout << "  lr/wd          : " << args.lr << "/" << args.weight_decay << endl;
        cout << "  warmup_steps   : " << args.warmup_steps << endl;
        cout << "  clip_grad_norm : " << args.clip_grad_norm << endl;
        cout << "  lora_dropout   : " << args.lora_dropout << endl;
        cout << "  log_interval   : " << args.log_interval << endl;
        cout << "  eval_interval  : " << args.eval_interval << endl;
        cout << "  save_every     : " << args.save_every << endl;
        cout << "  ema_beta       : " << args.ema_beta << endl;
        cout << "  seed           : " << args.seed << endl;

        // 1) Construct model and load pretrained weights
        cout << "\n[1/6] Loading pretrained model..." << endl;
        GPT2Config cfg = GPT2Config::from_pretrained(args.pretrained_dir + "/config.json");
        GPT2Model model(cfg);
        model.tie_weights();
        model.print_model_info();

        string st_path = args.pretrained_dir + "/model.safetensors";
        SafeTensorsReader reader(st_path);
        reader.parse_header();
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        SafeTensorsLoadOptions load_opts;
        load_opts.transpose_linear = false;
        load_opts.verbose = false;  // reduce loader verbosity during training
        auto loaded = reader.load_tensors_mapped(key_map, load_opts);
        for (const auto& kv : loaded) {
            model.assign_weight(kv.first, kv.second);
        }
        cout << "  ✓ Model loaded" << endl;

        // 2) Initialize LoRA and inject / resume
        cout << "\n[2/6] Initialize/Resume LoRA..." << endl;
        model.init_lora_modules();
        auto clear_all_lora = [&]() {
            const auto& cfg_local = model.config();
            for (int li = 0; li < cfg_local.n_layer; ++li) {
                auto& blk = model.get_block(li);
                if (blk.qkv_lin) blk.qkv_lin->clear_lora();
                if (blk.proj_lin) blk.proj_lin->clear_lora();
                if (blk.fc_in_lin) blk.fc_in_lin->clear_lora();
                if (blk.fc_out_lin) blk.fc_out_lin->clear_lora();
            }
        };

        LoraSpec active_lora_spec;
        bool has_active_lora_spec = false;

        int start_step = 0;
        std::vector<TensorPtr> trainable;
        if (!args.resume_from.empty()) {
            cout << "  ↪ Found resume file, loading checkpoint: " << args.resume_from << endl;
            clear_all_lora();
            auto lora_state = LoraSaver::load_safetensors(args.resume_from);
            LoraSaver::attach_from_state(model, lora_state);
            trainable = model.get_lora_parameters();
            for (auto& p : trainable) { p->set_requires_grad(true); p->zero_grad(); }
            cout << "  ✓ LoRA weights loaded and attached (trainable=" << trainable.size() << ")" << endl;

            active_lora_spec.rank = lora_state.rank;
            active_lora_spec.alpha = lora_state.alpha;
            active_lora_spec.dropout = lora_state.dropout;
            active_lora_spec.split_qkv = lora_state.split_qkv;
            if (!lora_state.targets.empty()) {
                active_lora_spec.targets = lora_state.targets;
            } else {
                active_lora_spec.targets = { LoraTarget::AttnQKV, LoraTarget::AttnProj };
            }
            has_active_lora_spec = true;
        } else {
            LoraSpec lora_spec;
            lora_spec.rank = args.rank;
            lora_spec.alpha = args.alpha;
            lora_spec.dropout = args.lora_dropout;
            lora_spec.split_qkv = false;
            lora_spec.targets = { LoraTarget::AttnQKV, LoraTarget::AttnProj };

            LoraInjector injector;
            injector.inject(model, lora_spec);
            trainable = model.get_lora_parameters();
            for (auto& p : trainable) p->zero_grad();
            cout << "  ✓ LoRA injected: " << trainable.size() << " trainable parameters" << endl;

            active_lora_spec = lora_spec;
            has_active_lora_spec = true;
        }

        // 3) Prepare tokenizer and dataset
        cout << "\n[3/6] Loading dataset..." << endl;
        auto tok_cfg = BPEConfig::from_pretrained(args.pretrained_dir);
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();

        WT2Config data_cfg;
        if (!args.jsonl_train.empty() || !args.jsonl_valid.empty() || !args.jsonl_test.empty()) {
            data_cfg.jsonl_train = args.jsonl_train;
            data_cfg.jsonl_valid = args.jsonl_valid;
            data_cfg.jsonl_test  = args.jsonl_test;
            data_cfg.streaming_mode = false;
            data_cfg.shuffle_train = true;
        } else {
            data_cfg.train_path = args.data_dir + "/wiki.train.raw";
            data_cfg.valid_path = args.data_dir + "/wiki.valid.raw";
            data_cfg.test_path  = args.data_dir + "/wiki.test.raw";
        }
        data_cfg.seq_len = args.seq_len;
        data_cfg.stride  = -1;
        data_cfg.eos_id  = 50256;
        data_cfg.seed = args.seed;
        
        WikiText2Dataset train_dataset(data_cfg, &tokenizer);
        train_dataset.load(Split::Train);
        
        WT2Config valid_cfg = data_cfg;
        valid_cfg.drop_last = false;
        WikiText2Dataset valid_dataset(valid_cfg, &tokenizer);
        valid_dataset.load(Split::Valid);
        
        cout << "  ✓ Train set: " << train_dataset.num_sequences() << " sequences" << endl;
        cout << "  ✓ Valid set: " << valid_dataset.num_sequences() << " sequences" << endl;

        const int64_t train_seqs = train_dataset.num_sequences();
        const int64_t micro_bs = args.batch_size;
        const int64_t accum = max(1, args.grad_accum_steps);
        const int64_t steps_per_epoch = (train_seqs + micro_bs * accum - 1) / (micro_bs * accum);
        
        if (args.epochs > 0) {
            args.steps = steps_per_epoch * args.epochs;
        }
        
        // Training schedule
        cout << "\n[Schedule]" << endl;
        cout << "  epochs         : " << args.epochs << endl;
        cout << "  steps_per_epoch: " << steps_per_epoch << endl;
        cout << "  total_steps    : " << args.steps << endl;
        cout << "  Effective batch: " << (micro_bs * accum) << " (micro=" << micro_bs 
             << " × accum=" << accum << ")" << endl;

        // 4) Optimizer
        cout << "\n[4/6] Initializing optimizer..." << endl;
        AdamConfig opt_cfg;
        opt_cfg.learning_rate = args.lr;
        opt_cfg.beta1 = 0.9f;
        opt_cfg.beta2 = 0.999f;
        opt_cfg.epsilon = 1e-8f;
        opt_cfg.weight_decay = args.weight_decay;
        opt_cfg.clip_grad_norm = args.clip_grad_norm;
        Adam optimizer(opt_cfg);
        cout << "  ✓ Adam optimizer ready" << endl;

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
            const int T = args.steps;
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

        cout << "\n[5/6] Start training..." << endl;
        cout << "========================================\n" << endl;
        double ema_loss = 0.0;
        bool ema_initialized = false;
        int64_t total_tokens = 0;
        double rss_peak_mb = 0.0;

        for (int step = start_step; step < args.steps; ++step) {
            const int cur_epoch = step / steps_per_epoch + 1;
            const int step_in_epoch = step % steps_per_epoch + 1;
            double accum_loss = 0.0;
            int64_t accum_tokens = 0;
            auto rss_to_mb = []() -> double {
                size_t bytes = MemoryMonitor::get_system_memory_usage();
                return static_cast<double>(bytes) / (1024.0 * 1024.0);
            };
            double rss_pre_max = 0.0, rss_fwd_max = 0.0, rss_bwd_max = 0.0, rss_opt_max = 0.0, rss_post_max = 0.0;
            rss_pre_max = std::max(rss_pre_max, rss_to_mb());
            for (int acc = 0; acc < accum; ++acc) {
                auto batch = train_dataset.next_batch(micro_bs);
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                rss_fwd_max = std::max(rss_fwd_max, rss_to_mb());
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                float loss_val = loss->data<float>()[0];
                accum_loss += loss_val;
                accum_tokens += count_tokens(batch.attention_mask);
                {
                    float inv_accum = 1.0f / float(accum);
                    auto scaled_loss = mul(loss, inv_accum);
                    scaled_loss->backward();
                }
                rss_bwd_max = std::max(rss_bwd_max, rss_to_mb());
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
            rss_opt_max = std::max(rss_opt_max, rss_to_mb());
            if (((step + 1) % 50) == 0) {
                MemoryManager::instance().force_cleanup();
            }
            if (((step + 1) % 100) == 0) {
                MemoryManager::instance().clear_unused_memory();
            }
            MemoryManager::instance().cleanup_dead_references();
            rss_post_max = std::max(rss_post_max, rss_to_mb());
            double rss_step_max = std::max({rss_pre_max, rss_fwd_max, rss_bwd_max, rss_opt_max, rss_post_max});
            rss_peak_mb = std::max(rss_peak_mb, rss_step_max);
            float avg_loss = float(accum_loss / double(accum));
            float ppl = perplexity_from_loss(avg_loss);
            total_tokens += accum_tokens;
            if (!ema_initialized) {
                ema_loss = avg_loss;
                ema_initialized = true;
            } else {
                double beta = max(0.0, min(0.9999, double(args.ema_beta)));
                ema_loss = beta * ema_loss + (1.0 - beta) * avg_loss;
            }
            if ((step + 1) % max(1, args.log_interval) == 0) {
                cout << "[Train] epoch " << cur_epoch << "/" << args.epochs
                     << " | step " << step_in_epoch << "/" << steps_per_epoch
                     << " (global " << (step + 1) << "/" << args.steps << ")"
                     << " | lr " << fixed << setprecision(6) << cur_lr
                     << " | loss " << setprecision(4) << avg_loss
                     << " | ppl " << setprecision(2) << ppl
                     << " | grad_norm " << setprecision(3) << grad_norm
                     << " | tokens " << accum_tokens
                     << " | RSS(pre/fwd/bwd/opt/post)="
                     << fixed << setprecision(1)
                     << rss_pre_max << "/" << rss_fwd_max << "/"
                     << rss_bwd_max << "/" << rss_opt_max << "/"
                     << rss_post_max << " MB"
                     << " | step_max=" << rss_step_max << " MB"
                     << " | peak=" << rss_peak_mb << " MB"
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
                     << " | ema_loss " << setprecision(4) << float(ema_loss)
                     << " | total_tokens " << total_tokens
                     << "\n" << endl;
                if (!args.eval_out.empty()) {
                    stringstream ss;
                    ss << "{\"step\":" << (step + 1)
                       << ",\"epoch\":" << cur_epoch
                       << ",\"valid_ppl\":" << valid_ppl
                       << ",\"ema_loss\":" << float(ema_loss)
                       << ",\"total_tokens\":" << total_tokens
                       << "}";
                    append_jsonl(args.eval_out, ss.str());
                }
            }
            if (args.save_every > 0 && (step + 1) % args.save_every == 0 && !args.lora_out.empty()) {
                string ckpt_path = make_checkpoint_path(args.lora_out, step + 1);
                cout << "\n[Checkpoint] saving to " << ckpt_path << endl;
                if (has_active_lora_spec) {
                    LoraSaver::save_safetensors(ckpt_path, model, active_lora_spec);
                    cout << "  ✓ Checkpoint saved\n" << endl;
                } else {
                    cout << "  ⚠️ No active LoRA config detected, skip saving" << endl;
                }
            }
        }

        cout << "\n[6/6] Save final LoRA weights..." << endl;
        if (!args.lora_out.empty()) {
            if (has_active_lora_spec) {
                LoraSaver::save_safetensors(args.lora_out, model, active_lora_spec);
                cout << "  ✓ LoRA saved to: " << args.lora_out << endl;
            } else {
                cout << "  ⚠️ No LoRA weights found to save, skipped" << endl;
            }
        }

        cout << "\n========================================" << endl;
        cout << "✅ Training complete!" << endl;
        cout << "  Total steps: " << args.steps << endl;
        cout << "  Total tokens: " << total_tokens << endl;
        cout << "  Final EMA loss: " << fixed << setprecision(4) << float(ema_loss) << endl;
        cout << "========================================\n" << endl;
        return 0;
    } catch (const exception& e) {
        cerr << "\n❌ Exception: " << e.what() << endl;
        return 1;
    }
}


