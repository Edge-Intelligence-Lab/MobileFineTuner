#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <regex>
#include <algorithm>
#include <filesystem>

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
#include "finetune_ops/optim/smoke_utils.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/core/quant_io.h"

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
    // Q8_0 quantization of the frozen base weights ("" = fp32)
    string quantize;
    string load_quantized;
    string save_quantized;
    int seed = 42;
    bool synthetic_smoke = false;
    int smoke_steps = 2;
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
         << "  --quantize q8_0          Quantize the frozen base weights to Q8_0\n"
        << "  --save_quantized PATH    Write the Q8_0 model (int8 linears + fp32 rest)\n"
        << "  --load_quantized PATH    Load a Q8_0 model directly (skips the fp32 file)\n"
         << "                           (LoRA params stay fp32; forward uses the\n"
         << "                           fused int8 kernel, backward reads int8 too)\n"
         << "  --seed N                 Random seed\n"
         << "  --synthetic_smoke        Run a self-contained 2-step smoke train\n"
         << "  --smoke_steps N          Override synthetic smoke steps\n";
}

static CmdArgs parse_args(int argc, char** argv) {
    CmdArgs args;
    args.data_dir = "data/wikitext2/wikitext-2-raw";

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
        else if (k == "--quantize") args.quantize = get_val("--quantize");
        else if (k == "--load_quantized") args.load_quantized = get_val("--load_quantized");
        else if (k == "--save_quantized") args.save_quantized = get_val("--save_quantized");
        else if (k == "--synthetic_smoke") args.synthetic_smoke = true;
        else if (k == "--smoke_steps") args.smoke_steps = stoi(get_val("--smoke_steps"));
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

static void require_file(const string& path, const string& desc) {
    if (!std::filesystem::exists(path)) {
        throw runtime_error(desc + " not found: " + path);
    }
}

static int run_synthetic_smoke(const CmdArgs& args) {
    cout << "\n========== GPT-2 LoRA Synthetic Smoke ==========\n" << endl;

    GPT2Config cfg;
    cfg.vocab_size = 64;
    cfg.n_positions = max(16, args.seq_len + 2);
    cfg.n_embd = 16;
    cfg.n_layer = 1;
    cfg.n_head = 4;
    cfg.use_memory_efficient_attention = true;

    GPT2Model model(cfg);
    std::mt19937 rng(static_cast<uint32_t>(args.seed));
    smoke::initialize_tiny_gpt2(model, rng);
    model.tie_weights();

    LoraSpec lora_spec;
    lora_spec.rank = args.rank;
    lora_spec.alpha = args.alpha;
    lora_spec.dropout = args.lora_dropout;
    lora_spec.split_qkv = false;
    lora_spec.targets = {LoraTarget::AttnQKV, LoraTarget::AttnProj};

    LoraInjector injector;
    injector.inject(model, lora_spec);
    auto trainable = injector.get_trainable_params();
    if (trainable.empty()) {
        throw runtime_error("synthetic smoke created no trainable LoRA parameters");
    }

    auto before = smoke::clone_tensors(trainable);
    const int batch_size = max(1, min(args.batch_size, 2));
    const int seq_len = max(4, min(args.seq_len, 8));
    auto input_ids = smoke::make_input_ids(batch_size, seq_len, cfg.vocab_size, 1);
    auto labels = smoke::make_shifted_labels(input_ids, cfg.vocab_size);
    auto attention_mask = smoke::make_attention_mask(batch_size, seq_len);

    AdamConfig opt_cfg;
    opt_cfg.learning_rate = max(args.lr, 1e-2f);
    opt_cfg.weight_decay = 0.0f;
    Adam optimizer(opt_cfg);

    vector<float> losses;
    const int smoke_steps = max(1, args.smoke_steps);
    for (int step = 0; step < smoke_steps; ++step) {
        auto logits = model.forward(input_ids, attention_mask);
        auto loss = lm_cross_entropy(logits, labels, -100, "mean");
        if (!smoke::is_finite_scalar(loss)) {
            throw runtime_error("non-finite loss during GPT-2 synthetic smoke");
        }
        losses.push_back(loss->data<float>()[0]);
        loss->backward();

        vector<TensorPtr> grads;
        grads.reserve(trainable.size());
        for (const auto& param : trainable) {
            grads.push_back(param->grad());
        }
        double grad_norm = smoke::grad_l2_norm(grads);
        if (!(grad_norm > 0.0) || !std::isfinite(grad_norm)) {
            throw runtime_error("invalid gradient norm during GPT-2 synthetic smoke");
        }

        optimizer.step(trainable, grads);
        smoke::zero_grads(trainable);
        smoke::cleanup_step_memory();

        cout << "[smoke step " << (step + 1) << "/" << smoke_steps
             << "] loss=" << losses.back()
             << " grad_norm=" << grad_norm << endl;
    }

    double param_delta = smoke::max_param_delta(before, trainable);
    if (!args.lora_out.empty()) {
        injector.save_lora_safetensors(args.lora_out);
    }

    cout << "[SyntheticSmoke] param_delta=" << param_delta << endl;
    const bool ok = std::isfinite(param_delta) && param_delta > 0.0;
    cout << (ok ? "[PASS]" : "[FAIL]") << endl;
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);
    if (args.synthetic_smoke) {
        try {
            return run_synthetic_smoke(args);
        } catch (const exception& e) {
            cerr << "\n❌ Exception: " << e.what() << endl;
            return 1;
        }
    }

    try {
        if (args.pretrained_dir.empty()) {
            throw runtime_error("--pretrained_dir is required unless --synthetic_smoke is used");
        }
        require_file(args.pretrained_dir + "/config.json", "GPT-2 config");
        if (args.epochs <= 0 && args.steps <= 0) {
            args.epochs = 1;
        }
        if (args.jsonl_train.empty() && args.jsonl_valid.empty() && args.jsonl_test.empty()) {
            require_file(args.data_dir + "/wiki.train.raw", "WikiText-2 train split");
            require_file(args.data_dir + "/wiki.valid.raw", "WikiText-2 valid split");
        }

        cout << "\n========== GPT-2 LoRA Finetune (Pro) ==========\n" << endl;
        cout << "[Config]" << endl;
        // Build-time BLAS toggle (informational only; actual kernels may choose paths at runtime)
#if defined(USE_BLAS)
        cout << "  BLAS          : ON (compiled)" << endl;
#else
        cout << "  BLAS          : OFF (compiled)" << endl;
#endif
        // Data source: prefer masked JSONL (MMLU-style samples); otherwise fall back to raw text under data_dir
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
        cout << "\n[1/6] Load pretrained model..." << endl;
        // Load size dynamically from HF config.json (supports gpt2 / gpt2-medium / gpt2-large / gpt2-xl)
        GPT2Config cfg = GPT2Config::from_pretrained(args.pretrained_dir + "/config.json");
        GPT2Model model(cfg);
        model.tie_weights();
        model.print_model_info();

        // Two ways in: the original fp32 safetensors, or a self-contained Q8_0
        // file produced by --save_quantized. The latter stores the 48 linear
        // weights as int8 blocks, so the fp32 originals are never read.
        vector<ops::QuantTensor> q8_pending;
        const bool from_q8 = !args.load_quantized.empty();
        if (!from_q8) {
            SafeTensorsModelReader reader(args.pretrained_dir);
            reader.parse_headers();
            auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
            // Linear weights in GPT-2 safetensors are stored as [in, out] (e.g., [768, 2304]); no transpose needed.
            // Explicitly disable load-time transpose to avoid runtime transposition overhead.
            SafeTensorsLoadOptions load_opts;
            load_opts.transpose_linear = false;
            load_opts.verbose = false;        // reduce loader verbosity during training
            auto loaded = reader.load_tensors_mapped(key_map, load_opts);
            for (const auto& kv : loaded) {
                model.assign_weight(kv.first, kv.second);
            }
            cout << "  [load] source: fp32 safetensors" << endl;
        } else {
            vector<ops::QuantTensor> tensors;
            string qerr;
            if (!ops::read_quant_model(args.load_quantized, tensors, &qerr)) {
                cerr << "[ERROR] --load_quantized: " << qerr << endl;
                return 1;
            }
            size_t n_f32 = 0;
            for (auto& t : tensors) {
                if (t.is_q8) {
                    q8_pending.push_back(std::move(t));
                    continue;
                }
                // 4-arg overload copies; the 5-arg one WRAPS the buffer, which
                // would dangle as soon as `tensors` goes out of scope.
                auto ten = std::make_shared<Tensor>(
                    t.shape, static_cast<const void*>(t.f32.data()), DType::kFloat32, kCPU);
                model.assign_weight(t.name, ten);
                ++n_f32;
            }
            cout << "  [load] source: Q8_0 checkpoint, fp32 tensors=" << n_f32
                 << " q8 tensors=" << q8_pending.size() << endl;
        }
        cout << "  ✓ Model loaded" << endl;

        // 1b) Optional Q8_0 quantization of the frozen base weights.
        //
        // Done after the weights are loaded and BEFORE any LoRA slices are
        // attached. Only the base weight is quantized; LoRA A/B stay fp32 so
        // they remain trainable. Requires init_lora_modules() first, because
        // forward() only routes through the LoRALinear wrappers.
        model.init_lora_modules();

        // Install int8 base weights that came from a Q8_0 checkpoint.
        if (!q8_pending.empty()) {
            std::regex q8_name(R"(blocks\.(\d+)\.(.+))");
            int adopted = 0;
            for (auto& t : q8_pending) {
                std::smatch m;
                if (!std::regex_match(t.name, m, q8_name)) {
                    cerr << "[WARN] --load_quantized: unexpected q8 tensor name " << t.name << "\n";
                    continue;
                }
                const int li = std::stoi(m[1].str());
                const string sub = m[2].str();
                if (li < 0 || li >= cfg.n_layer) continue;
                auto& blk = model.get_block(li);
                LoRALinear* lin = nullptr;
                if (sub == "attn.qkv.weight") lin = blk.qkv_lin.get();
                else if (sub == "attn.proj.weight") lin = blk.proj_lin.get();
                else if (sub == "mlp.fc_in.weight") lin = blk.fc_in_lin.get();
                else if (sub == "mlp.fc_out.weight") lin = blk.fc_out_lin.get();
                if (!lin) continue;
                lin->adopt_quantized_weight(t.k, t.m, std::move(t.q8));
                ++adopted;
            }
            cout << "  [load] int8 base weights adopted: " << adopted << endl;
            if (adopted == 0) {
                cerr << "[ERROR] --load_quantized: no layer accepted the int8 weights" << endl;
                return 1;
            }
        }

        if (!args.quantize.empty()) {
            if (from_q8) {
                cerr << "[ERROR] --quantize cannot be combined with --load_quantized" << endl;
                return 1;
            }
            const auto t_q0 = chrono::steady_clock::now();
            int quantized = 0, skipped = 0;
            for (int li = 0; li < cfg.n_layer; ++li) {
                auto& blk = model.get_block(li);
                LoRALinear* layers[4] = {blk.qkv_lin.get(), blk.proj_lin.get(),
                                         blk.fc_in_lin.get(), blk.fc_out_lin.get()};
                for (LoRALinear* lin : layers) {
                    if (!lin || !lin->W()) continue;
                    if (lin->quantize_base_weight_q8_0(/*allow_skip=*/true)) quantized++;
                    else skipped++;
                }
            }
            const double qsec =
                chrono::duration<double>(chrono::steady_clock::now() - t_q0).count();
            cout << "  [Quant] scheme=" << args.quantize
                 << " quantized=" << quantized << " skipped=" << skipped
                 << " (" << fixed << setprecision(2) << qsec << "s)" << endl;
            if (quantized == 0) {
                cerr << "[ERROR] --quantize requested but nothing could be quantized" << endl;
                return 1;
            }
        }

        // Write the current quantized model out (self-contained: int8 linears +
        // fp32 embeddings/norms/biases).
        if (!args.save_quantized.empty()) {
            vector<ops::QuantTensor> out;
            auto push_f32 = [&](const string& name, const TensorPtr& tt) {
                if (!tt) return;
                ops::QuantTensor e;
                e.name = name;
                e.shape = tt->shape();
                const float* p = tt->data<float>();
                e.f32.assign(p, p + tt->numel());
                out.push_back(std::move(e));
            };
            auto push_q8 = [&](const string& name, const LoRALinear* lin) {
                if (!lin || !lin->is_quantized()) return;
                ops::QuantTensor e;
                e.name = name;
                e.is_q8 = true;
                e.k = lin->quantized_k();
                e.m = lin->quantized_m();
                e.q8 = lin->quantized_weight();
                out.push_back(std::move(e));
            };
            for (int li = 0; li < cfg.n_layer; ++li) {
                auto& b = model.get_block(li);
                const string p = "blocks." + std::to_string(li) + ".";
                push_q8(p + "attn.qkv.weight", b.qkv_lin.get());
                push_q8(p + "attn.proj.weight", b.proj_lin.get());
                push_q8(p + "mlp.fc_in.weight", b.fc_in_lin.get());
                push_q8(p + "mlp.fc_out.weight", b.fc_out_lin.get());
                push_f32(p + "attn.qkv.bias", b.attn_qkv_bias);
                push_f32(p + "attn.proj.bias", b.attn_proj_bias);
                push_f32(p + "mlp.fc_in.bias", b.mlp_fc_in_bias);
                push_f32(p + "mlp.fc_out.bias", b.mlp_fc_out_bias);
                push_f32(p + "ln_1.weight", b.ln_1_weight);
                push_f32(p + "ln_1.bias", b.ln_1_bias);
                push_f32(p + "ln_2.weight", b.ln_2_weight);
                push_f32(p + "ln_2.bias", b.ln_2_bias);
            }
            push_f32("wte.weight", model.wte());
            push_f32("wpe.weight", model.wpe());
            push_f32("ln_f.weight", model.ln_f_weight());
            push_f32("ln_f.bias", model.ln_f_bias());

            string werr;
            if (!ops::write_quant_model(args.save_quantized, out, &werr)) {
                cerr << "[ERROR] --save_quantized: " << werr << endl;
                return 1;
            }
            size_t nq = 0;
            for (const auto& e : out) nq += e.is_q8 ? 1 : 0;
            cout << "  [save] wrote " << out.size() << " tensors (" << nq
                 << " int8) to " << args.save_quantized << endl;
        }

        // 2) Initialize LoRA and inject / resume
        cout << "\n[2/6] Inject/Resume LoRA..." << endl;

        // Utility: clear existing LoRA slices to avoid duplicate attachments
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
            // After resuming, refresh trainable parameters and ensure they require grad
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
            // Align with PyTorch/PEFT default topology:
            // - do not split Q/K/V for fused c_attn
            // - only target attention QKV (c_attn) and attention proj (c_proj)
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
        cout << "\n[3/6] Load dataset..." << endl;
        auto tok_cfg = BPEConfig::from_pretrained(args.pretrained_dir);
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();

        WT2Config data_cfg;
        // Prefer JSONL when provided (masked labels applied only on answer spans)
        if (!args.jsonl_train.empty() || !args.jsonl_valid.empty() || !args.jsonl_test.empty()) {
            data_cfg.jsonl_train = args.jsonl_train;
            data_cfg.jsonl_valid = args.jsonl_valid;
            data_cfg.jsonl_test  = args.jsonl_test;
            // Disable streaming scan in JSONL mode
            data_cfg.streaming_mode = false;
            // Enable shuffling for training
            data_cfg.shuffle_train = true;
            // Ignore other text paths when JSONL is present
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

        // Derive the training schedule
        const int64_t train_seqs = train_dataset.num_sequences();
        const int64_t micro_bs = args.batch_size;
        const int64_t accum = max(1, args.grad_accum_steps);
        const int64_t steps_per_epoch = (train_seqs + micro_bs * accum - 1) / (micro_bs * accum);
        
        if (args.epochs > 0) {
            args.steps = steps_per_epoch * args.epochs;
        }
        
        cout << "\n[Training plan]" << endl;
        cout << "  epochs         : " << args.epochs << endl;
        cout << "  steps_per_epoch: " << steps_per_epoch << endl;
        cout << "  total_steps    : " << args.steps << endl;
        cout << "  Effective batch: " << (micro_bs * accum) << " (micro=" << micro_bs 
             << " × accum=" << accum << ")" << endl;

        // 4) Optimizer
        cout << "\n[4/6] Init optimizer..." << endl;
        AdamConfig opt_cfg;
        opt_cfg.learning_rate = args.lr;
        opt_cfg.beta1 = 0.9f;
        opt_cfg.beta2 = 0.999f;
        opt_cfg.epsilon = 1e-8f;
        opt_cfg.weight_decay = args.weight_decay;
        opt_cfg.clip_grad_norm = args.clip_grad_norm;
        Adam optimizer(opt_cfg);
        cout << "  ✓ Adam optimizer ready" << endl;

        // Helper: count valid (masked) tokens in attention mask
        auto count_tokens = [](const TensorPtr& mask) -> int64_t {
            if (!mask) return 0;
            const float* ptr = mask->data<float>();
            int64_t total = 0;
            for (int64_t i = 0; i < mask->numel(); ++i) {
                if (ptr[i] > 0.5f) total++;
            }
            return total;
        };

        // Learning-rate schedule: linear warmup followed by cosine decay
        auto lr_schedule = [&](int step) -> float {
            const float base_lr = args.lr;
            const int T = args.steps;
            const int W = max(0, args.warmup_steps);
            
            if (step < W) {
                // Linear warmup
                return base_lr * (float(step + 1) / float(max(1, W)));
            }
            
            // Cosine decay down to 10% of the base LR
            const int s = step - W;
            const int d = max(1, T - W);
            const float min_lr = 0.1f * base_lr;
            // Avoid portability issues with non-standard M_PI macro
            const float PI = acosf(-1.0f);
            const float cosv = 0.5f * (1.0f + cos(PI * float(s) / float(d)));
            return min_lr + (base_lr - min_lr) * cosv;
        };

        // Gradient clipping and L2 norm computation
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

        // Validation pass
        auto evaluate_valid = [&]() -> float {
            // Disable autograd during evaluation; restore state and clear graph on exit
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

        // 5) Training loop
        cout << "\n[5/6] Start training..." << endl;
        cout << "========================================\n" << endl;
        
        double ema_loss = 0.0;
        bool ema_initialized = false;
        int64_t total_tokens = 0;
        // Track global RSS peak across steps
        double rss_peak_mb = 0.0;
        // Wall-clock accounting for the training run
        const auto run_t0 = chrono::steady_clock::now();
        double step_ms_best = 1e30;
        double step_ms_sum = 0.0;
        int64_t timed_steps = 0;
        // Phase attribution across the whole run, in milliseconds.
        double phase_fwd_ms = 0.0, phase_loss_ms = 0.0, phase_bwd_ms = 0.0, phase_opt_ms = 0.0;

        for (int step = start_step; step < args.steps; ++step) {
            const int cur_epoch = step / steps_per_epoch + 1;
            const int step_in_epoch = step % steps_per_epoch + 1;
            const auto step_t0 = chrono::steady_clock::now();
            
            // Gradient accumulation
            double accum_loss = 0.0;
            int64_t accum_tokens = 0;
            // Per-step RSS trackers
            auto rss_to_mb = []() -> double {
                size_t bytes = MemoryMonitor::get_system_memory_usage();
                return static_cast<double>(bytes) / (1024.0 * 1024.0);
            };
            double rss_pre_max = 0.0, rss_fwd_max = 0.0, rss_bwd_max = 0.0, rss_opt_max = 0.0, rss_post_max = 0.0;
            rss_pre_max = std::max(rss_pre_max, rss_to_mb());
            
            for (int acc = 0; acc < accum; ++acc) {
                auto batch = train_dataset.next_batch(micro_bs);
                const auto tp0 = chrono::steady_clock::now();
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                const auto tp1 = chrono::steady_clock::now();
                rss_fwd_max = std::max(rss_fwd_max, rss_to_mb());
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                float loss_val = loss->data<float>()[0];
                const auto tp2 = chrono::steady_clock::now();
                
                accum_loss += loss_val;
                accum_tokens += count_tokens(batch.attention_mask);
                // Strategy A: scale loss by 1/accum at each micro step, then backprop
                {
                    float inv_accum = 1.0f / float(accum);
                    auto scaled_loss = mul(loss, inv_accum);
                    scaled_loss->backward();
                }
                const auto tp3 = chrono::steady_clock::now();
                // Phase attribution: forward / loss / backward. These are the
                // three phases whose balance can shift under quantization.
                phase_fwd_ms += chrono::duration<double, milli>(tp1 - tp0).count();
                phase_loss_ms += chrono::duration<double, milli>(tp2 - tp1).count();
                phase_bwd_ms += chrono::duration<double, milli>(tp3 - tp2).count();
                rss_bwd_max = std::max(rss_bwd_max, rss_to_mb());
            }
            const auto to0 = chrono::steady_clock::now();
            
            // Gradient clipping
            float grad_norm = clip_and_get_grad_norm(args.clip_grad_norm);
            
            // Update learning rate
            float cur_lr = lr_schedule(step);
            optimizer.set_learning_rate(cur_lr);
            
            // Optimizer step and zero grads
            vector<TensorPtr> grads;
            grads.reserve(trainable.size());
            for (const auto& p : trainable) {
                grads.push_back(p->grad());
            }
            optimizer.step(trainable, grads);
            for (auto& p : trainable) {
                p->zero_grad();
            }
            phase_opt_ms += chrono::duration<double, milli>(chrono::steady_clock::now() - to0).count();
            rss_opt_max = std::max(rss_opt_max, rss_to_mb());
            
            // Periodic light cleanup of free blocks (default every 50 steps) to reduce overhead
            if (((step + 1) % 50) == 0) {
                MemoryManager::instance().force_cleanup();
            }
            // Aggressive release of free memory back to OS every 100 steps to prevent RSS drift
            if (((step + 1) % 100) == 0) {
                MemoryManager::instance().clear_unused_memory();
            }
            // Also do a light cleanup each step to stabilize post-step RSS observation
            MemoryManager::instance().cleanup_dead_references();
            rss_post_max = std::max(rss_post_max, rss_to_mb());
            double rss_step_max = std::max({rss_pre_max, rss_fwd_max, rss_bwd_max, rss_opt_max, rss_post_max});
            rss_peak_mb = std::max(rss_peak_mb, rss_step_max);
            
            // Update running statistics
            float avg_loss = float(accum_loss / double(accum));
            float ppl = perplexity_from_loss(avg_loss);
            total_tokens += accum_tokens;

            // Step timing (skip step 0: it includes lazy allocations)
            const double step_ms =
                chrono::duration<double, milli>(chrono::steady_clock::now() - step_t0).count();
            if (step > start_step) {
                step_ms_best = std::min(step_ms_best, step_ms);
                step_ms_sum += step_ms;
                timed_steps++;
            }
            
            // EMA
            if (!ema_initialized) {
                ema_loss = avg_loss;
                ema_initialized = true;
            } else {
                double beta = max(0.0, min(0.9999, double(args.ema_beta)));
                ema_loss = beta * ema_loss + (1.0 - beta) * avg_loss;
            }
            
            // Logging
            if ((step + 1) % max(1, args.log_interval) == 0) {
                cout << "[Train] epoch " << cur_epoch << "/" << args.epochs
                     << " | step " << step_in_epoch << "/" << steps_per_epoch
                     << " (global " << (step + 1) << "/" << args.steps << ")"
                     << " | lr " << fixed << setprecision(6) << cur_lr
                     << " | loss " << setprecision(4) << avg_loss
                     << " | ppl " << setprecision(2) << ppl
                     << " | grad_norm " << setprecision(3) << grad_norm
                     << " | tokens " << accum_tokens
                     << " | step_ms " << setprecision(1) << step_ms
                     << " | RSS(pre/fwd/bwd/opt/post)="
                     << fixed << setprecision(1)
                     << rss_pre_max << "/" << rss_fwd_max << "/"
                     << rss_bwd_max << "/" << rss_opt_max << "/"
                     << rss_post_max << " MB"
                     << " | step_max=" << rss_step_max << " MB"
                     << " | peak=" << rss_peak_mb << " MB"
                     << endl;
            }
            
            // Every 100 steps, print memory stats (monitor total_allocated / total_in_use)
            if ((step + 1) % 100 == 0) {
                MemoryManager::instance().print_memory_stats();
            }
            
            // Periodic evaluation
            if (args.eval_interval > 0 && (step + 1) % args.eval_interval == 0) {
                float valid_ppl = evaluate_valid();
                cout << "\n[Eval] epoch " << cur_epoch 
                     << " | step " << (step + 1)
                     << " | valid_ppl " << fixed << setprecision(2) << valid_ppl
                     << " | ema_loss " << setprecision(4) << float(ema_loss)
                     << " | total_tokens " << total_tokens
                     << "\n" << endl;
                
                // Append metrics to JSONL
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
            
            // Periodic checkpointing
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

        // 6) Save final LoRA adapter
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

        const double run_sec =
            chrono::duration<double>(chrono::steady_clock::now() - run_t0).count();
        cout << "  Wall time: " << setprecision(1) << run_sec << " s" << endl;
        if (timed_steps > 0) {
            const double avg_ms = step_ms_sum / double(timed_steps);
            cout << "  Step time: avg " << setprecision(1) << avg_ms
                 << " ms, best " << step_ms_best << " ms (n=" << timed_steps << ")" << endl;
            cout << "  Throughput: " << setprecision(1)
                 << (double(total_tokens) / run_sec) << " tokens/s" << endl;
            cout << "  Phase split: forward=" << setprecision(1) << phase_fwd_ms
                 << " ms  loss=" << phase_loss_ms
                 << " ms  backward=" << phase_bwd_ms
                 << " ms  optimizer=" << phase_opt_ms << " ms" << endl;
            const double phase_tot = phase_fwd_ms + phase_loss_ms + phase_bwd_ms + phase_opt_ms;
            if (phase_tot > 0) {
                cout << "  Phase share: fwd=" << setprecision(1)
                     << (100.0 * phase_fwd_ms / phase_tot) << "%  bwd="
                     << (100.0 * phase_bwd_ms / phase_tot) << "%  opt="
                     << (100.0 * phase_opt_ms / phase_tot) << "%" << endl;
            }
            cout << "  Peak RSS: " << setprecision(1) << rss_peak_mb << " MB" << endl;
            // Machine-readable summary for automated comparison.
            // Every float is formatted explicitly: the stream's precision flag
            // would otherwise leak from one field into the next.
            auto f1 = [](double v) { std::ostringstream s; s << std::fixed << std::setprecision(1) << v; return s.str(); };
            auto f2 = [](double v) { std::ostringstream s; s << std::fixed << std::setprecision(2) << v; return s.str(); };
            auto f6 = [](double v) { std::ostringstream s; s << std::fixed << std::setprecision(6) << v; return s.str(); };
            cout << "TRAIN_SUMMARY {\"quantize\":\"" << args.quantize
                 << "\",\"steps\":" << args.steps
                 << ",\"tokens\":" << total_tokens
                 << ",\"ema_loss\":" << f6(double(ema_loss))
                 << ",\"wall_s\":" << f2(run_sec)
                 << ",\"avg_step_ms\":" << f2(avg_ms)
                 << ",\"best_step_ms\":" << f2(step_ms_best)
                 << ",\"tokens_per_s\":" << f2(double(total_tokens) / run_sec)
                 << ",\"peak_rss_mb\":" << f1(rss_peak_mb)
                 << ",\"seq_len\":" << args.seq_len
                 << ",\"batch_size\":" << args.batch_size
                 << ",\"grad_accum\":" << args.grad_accum_steps
                 << ",\"rank\":" << args.rank << "}" << endl;
        }
        cout << "========================================\n" << endl;

        return 0;

    } catch (const exception& e) {
        cerr << "\n❌ Exception: " << e.what() << endl;
        return 1;
    }
}
