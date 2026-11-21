#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/checkpoint.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/tokenizer_hf.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/graph/qwen_lora_saver.h"
#include "finetune_ops/graph/qwen_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/optim/adam.h"
#include "finetune_ops/optim/adamw.h"
#include "finetune_ops/optim/gradient_accumulator.h"
#include "resume_helpers.h"
#include "microbatch.h"
#include "measurement.h"

using namespace ops;

namespace {

struct CliOptions {
    std::string model_dir = "Qwen2.5-0.5B";
    std::string data_dir = "data/wikitext2/wikitext-2-raw";
    std::string jsonl_train;
    std::string jsonl_valid;
    std::string pretokenized_path;
    std::string pretokenized_meta;
    std::string output_dir = "./qwen_lora";
    int epochs = 0;
    int steps = 0;
    int stop_after_steps = 0;
    bool save_initial_adapter = false;
    bool shard_enable = false;
    std::string shard_dir;
    size_t shard_budget_mb = 768;
    bool shard_fp16_disk = false;
    bool activation_checkpointing = false;
    bool memory_efficient_attention = false;
    std::string measurement_dir;
    int seq_len = 128;
    int batch = 8;
    int grad_accum = 1;
    float learning_rate = 2e-4f;
    int warmup_steps = 0;
    std::string lr_scheduler = "cosine";
    float max_grad_norm = 1.0f;
    float data_fraction = 1.0f;
    float weight_decay = 0.0f;
    std::string optimizer = "adamw";
    std::string target_mode = "qv";
    int lora_r = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.0f;
    int logging_steps = 1;
    int eval_steps = 0;
    int eval_batches = 50;
    int save_every = 0;
    uint64_t seed = 42;
    std::string resume_from;
    bool shuffle_train = false;
    bool dump_first_batch = false;
    int dump_first_tokens = 16;
    std::string dump_first_batch_path;
    bool exit_after_dump = false;
    std::string fixed_batch_bin;
    int fixed_batch_steps = 0;
    bool print_step0 = false;
    std::string tokenizer_json;
    std::string eval_out;
};

struct FixedBatch {
    int32_t B = 0;
    int32_t S = 0;
    std::vector<int32_t> ids;
    std::vector<float> attn;
};

void append_jsonl(const std::string& path, const std::string& json_line) {
    if (path.empty()) return;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out(path, std::ios::app);
    if (out) out << json_line << "\n";
}

std::string make_checkpoint_path(const std::string& output_dir, int step) {
    std::ostringstream ss;
    ss << output_dir << "/lora_step" << step << ".safetensors";
    return ss.str();
}

float make_scheduler(int step,
                     int total_steps,
                     int warmup_steps,
                     float base_lr,
                     const std::string& mode) {
    constexpr float kPi = 3.14159265358979323846f;
    const int step_1indexed = step + 1;
    if (warmup_steps > 0 && step_1indexed <= warmup_steps) {
        return base_lr * static_cast<float>(step_1indexed) /
               static_cast<float>(std::max(1, warmup_steps));
    }
    const int remain = std::max(1, total_steps - warmup_steps);
    float progress = static_cast<float>(step_1indexed - warmup_steps) /
                     static_cast<float>(remain);
    progress = std::min(std::max(progress, 0.0f), 1.0f);
    if (mode == "constant") {
        return base_lr;
    }
    if (mode == "linear") {
        return base_lr * (1.0f - progress);
    }
    return base_lr * 0.5f * (1.0f + std::cos(kPi * progress));
}

bool parse_bool_value(const std::string& v) {
    return !(v == "0" || v == "false" || v == "False");
}

FixedBatch load_fixed_batch_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open fixed_batch_bin: " + path);
    }
    FixedBatch fb;
    in.read(reinterpret_cast<char*>(&fb.B), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&fb.S), sizeof(int32_t));
    if (fb.B <= 0 || fb.S <= 0) {
        throw std::runtime_error("Invalid fixed batch shape in: " + path);
    }
    const size_t n = static_cast<size_t>(fb.B) * static_cast<size_t>(fb.S);
    fb.ids.resize(n);
    fb.attn.resize(n);
    in.read(reinterpret_cast<char*>(fb.ids.data()),
            static_cast<std::streamsize>(n * sizeof(int32_t)));
    in.read(reinterpret_cast<char*>(fb.attn.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) {
        throw std::runtime_error("Failed to read fixed batch payload from: " + path);
    }
    return fb;
}

uint64_t fnv1a_hash_int32(const int32_t* data, size_t n) {
    const uint64_t kOffset = 1469598103934665603ull;
    const uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t v = static_cast<uint32_t>(data[i]);
        for (int b = 0; b < 4; ++b) {
            const uint8_t byte = (v >> (8 * b)) & 0xFF;
            hash ^= byte;
            hash *= kPrime;
        }
    }
    return hash;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch; break;
        }
    }
    return out;
}

int64_t count_tokens(const TensorPtr& mask) {
    if (!mask) return 0;
    const float* ptr = mask->data<float>();
    int64_t total = 0;
    for (int64_t i = 0; i < mask->numel(); ++i) {
        if (ptr[i] > 0.5f) total++;
    }
    return total;
}

struct NoGradGuard {
    NoGradGuard() { ops::autograd::Engine::instance().set_enabled(false); }
    ~NoGradGuard() {
        ops::autograd::Engine::instance().clear_graph();
        ops::autograd::Engine::instance().set_enabled(true);
    }
};

CliOptions parse_cli(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto get_val = [&](const std::string& key) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + key);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--model_dir") opts.model_dir = get_val(arg);
        else if (arg == "--data_dir") opts.data_dir = get_val(arg);
        else if (arg == "--jsonl_train") opts.jsonl_train = get_val(arg);
        else if (arg == "--jsonl_valid") opts.jsonl_valid = get_val(arg);
        else if (arg == "--pretokenized_path") opts.pretokenized_path = get_val(arg);
        else if (arg == "--pretokenized_meta") opts.pretokenized_meta = get_val(arg);
        else if (arg == "--output_dir") opts.output_dir = get_val(arg);
        else if (arg == "--epochs") opts.epochs = std::stoi(get_val(arg));
        else if (arg == "--steps" || arg == "--max_steps") opts.steps = std::stoi(get_val(arg));
        else if (arg == "--stop_after_steps") opts.stop_after_steps = std::stoi(get_val(arg));
        else if (arg == "--save_initial_adapter") opts.save_initial_adapter = true;
        else if (arg == "--shard_enable") opts.shard_enable = true;
        else if (arg == "--shard_dir") opts.shard_dir = get_val(arg);
        else if (arg == "--shard_budget_mb") {
            const auto value = std::stoll(get_val(arg));
            if (value <= 0 || static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max() / (1024ull * 1024ull)) {
                throw std::runtime_error("--shard_budget_mb must be positive and fit size_t");
            }
            opts.shard_budget_mb = static_cast<size_t>(value);
        }
        else if (arg == "--shard_fp16_disk") opts.shard_fp16_disk = parse_bool_value(get_val(arg));
        else if (arg == "--activation_checkpointing") opts.activation_checkpointing = true;
        else if (arg == "--memory_efficient_attention") opts.memory_efficient_attention = true;
        else if (arg == "--measurement_dir") opts.measurement_dir = get_val(arg);
        else if (arg == "--seq_len") opts.seq_len = std::stoi(get_val(arg));
        else if (arg == "--batch" || arg == "--batch_size") opts.batch = std::stoi(get_val(arg));
        else if (arg == "--grad_accum" || arg == "--grad_accum_steps") opts.grad_accum = std::stoi(get_val(arg));
        else if (arg == "--learning_rate" || arg == "--lr") opts.learning_rate = std::stof(get_val(arg));
        else if (arg == "--warmup_steps") opts.warmup_steps = std::stoi(get_val(arg));
        else if (arg == "--lr_scheduler") opts.lr_scheduler = get_val(arg);
        else if (arg == "--max_grad_norm") opts.max_grad_norm = std::stof(get_val(arg));
        else if (arg == "--data_fraction") opts.data_fraction = std::stof(get_val(arg));
        else if (arg == "--weight_decay") opts.weight_decay = std::stof(get_val(arg));
        else if (arg == "--optimizer") opts.optimizer = get_val(arg);
        else if (arg == "--target_mode") opts.target_mode = get_val(arg);
        else if (arg == "--qv_only") opts.target_mode = "qv";
        else if (arg == "--qkvo" || arg == "--full") opts.target_mode = "full";
        else if (arg == "--lora_r") opts.lora_r = std::stoi(get_val(arg));
        else if (arg == "--lora_alpha") opts.lora_alpha = std::stof(get_val(arg));
        else if (arg == "--lora_dropout") opts.lora_dropout = std::stof(get_val(arg));
        else if (arg == "--logging_steps" || arg == "--log_interval") opts.logging_steps = std::stoi(get_val(arg));
        else if (arg == "--eval_steps" || arg == "--eval_interval") opts.eval_steps = std::stoi(get_val(arg));
        else if (arg == "--eval_batches") opts.eval_batches = std::stoi(get_val(arg));
        else if (arg == "--save_every") opts.save_every = std::stoi(get_val(arg));
        else if (arg == "--seed") opts.seed = static_cast<uint64_t>(std::stoull(get_val(arg)));
        else if (arg == "--resume_from") opts.resume_from = get_val(arg);
        else if (arg == "--shuffle") opts.shuffle_train = true;
        else if (arg == "--no_shuffle") opts.shuffle_train = false;
        else if (arg == "--shuffle_train") opts.shuffle_train = parse_bool_value(get_val(arg));
        else if (arg == "--dump_first_batch") opts.dump_first_batch = true;
        else if (arg == "--dump_first_tokens") opts.dump_first_tokens = std::stoi(get_val(arg));
        else if (arg == "--dump_first_batch_path") opts.dump_first_batch_path = get_val(arg);
        else if (arg == "--exit_after_dump") opts.exit_after_dump = true;
        else if (arg == "--fixed_batch_bin") opts.fixed_batch_bin = get_val(arg);
        else if (arg == "--fixed_batch_steps") opts.fixed_batch_steps = std::stoi(get_val(arg));
        else if (arg == "--print_step0") opts.print_step0 = parse_bool_value(get_val(arg));
        else if (arg == "--tokenizer_json") opts.tokenizer_json = get_val(arg);
        else if (arg == "--eval_out") opts.eval_out = get_val(arg);
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (opts.batch <= 0 || opts.grad_accum <= 0) {
        throw std::runtime_error("batch and grad_accum must be positive");
    }
    if (opts.stop_after_steps < 0) {
        throw std::runtime_error("--stop_after_steps must be nonnegative");
    }
    if (opts.steps <= 0 && opts.epochs <= 0) {
        opts.epochs = 1;
    }
    if (opts.optimizer != "adam" && opts.optimizer != "adamw") {
        throw std::runtime_error("Unsupported optimizer: " + opts.optimizer + " (expected adam or adamw)");
    }
    if (opts.eval_out.empty() && !opts.output_dir.empty()) {
        opts.eval_out = opts.output_dir + "/eval.jsonl";
    }
    return opts;
}

void dump_batch(const Batch& batch,
                int dump_first_tokens,
                const std::string& dump_path) {
    auto ids = batch.input_ids->data<int32_t>();
    const int64_t B = batch.input_ids->size(0);
    const int64_t S = batch.input_ids->size(1);
    const int tok_n = std::max(0, std::min(dump_first_tokens, static_cast<int>(S)));

    std::cout << "[Dump] first_batch shape=(" << B << "," << S << ")\n";
    std::cout << "[Dump] sample0 first" << tok_n << "=[";
    for (int i = 0; i < tok_n; ++i) {
        if (i) std::cout << ",";
        std::cout << ids[i];
    }
    std::cout << "]\n";
    if (B > 1) {
        std::cout << "[Dump] sample1 first" << tok_n << "=[";
        const int64_t base = S;
        for (int i = 0; i < tok_n; ++i) {
            if (i) std::cout << ",";
            std::cout << ids[base + i];
        }
        std::cout << "]\n";
    }
    const uint64_t hash = fnv1a_hash_int32(ids, static_cast<size_t>(B * S));
    std::cout << "[Dump] batch_hash_fnv1a64=" << hash << "\n";

    if (!dump_path.empty()) {
        const auto parent = std::filesystem::path(dump_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        auto attn = batch.attention_mask->data<float>();
        std::ofstream ofs(dump_path, std::ios::binary);
        if (!ofs) {
            std::cerr << "[Dump] Failed to open " << dump_path << std::endl;
            return;
        }
        const int32_t b32 = static_cast<int32_t>(B);
        const int32_t s32 = static_cast<int32_t>(S);
        ofs.write(reinterpret_cast<const char*>(&b32), sizeof(int32_t));
        ofs.write(reinterpret_cast<const char*>(&s32), sizeof(int32_t));
        ofs.write(reinterpret_cast<const char*>(ids), sizeof(int32_t) * static_cast<size_t>(B * S));
        ofs.write(reinterpret_cast<const char*>(attn), sizeof(float) * static_cast<size_t>(B * S));
        std::cout << "[Dump] Wrote first batch to " << dump_path << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto invocation = resume::prepare_invocation(argc, argv, "qwen", resume_cli::batch_validator(parse_cli));
        auto* session = invocation.session.get();
        std::vector<char*> cli_argv;
        for (auto& argument : invocation.arguments) cli_argv.push_back(argument.data());
        auto cli = parse_cli(static_cast<int>(cli_argv.size()), cli_argv.data());
        if (session) resume_cli::configure(cli, *session);
        const bool use_jsonl = !cli.jsonl_train.empty();
        const bool use_pretokenized = !cli.pretokenized_path.empty();
        const bool qv_only = (cli.target_mode == "qv");

        std::cout << "\n========== Qwen LoRA Finetune ==========\n";
        std::cout << "[Config]\n";
        std::cout << "  model_dir       : " << cli.model_dir << "\n";
        if (use_pretokenized) {
            std::cout << "  data_source     : pretokenized stream\n";
            std::cout << "  pretokenized    : " << cli.pretokenized_path << "\n";
            if (!cli.pretokenized_meta.empty()) {
                std::cout << "  pretokenized_meta: " << cli.pretokenized_meta << "\n";
            }
        } else if (use_jsonl) {
            std::cout << "  data_source     : JSONL(masked)\n";
            std::cout << "  jsonl_train     : " << cli.jsonl_train << "\n";
            std::cout << "  jsonl_valid     : " << (cli.jsonl_valid.empty() ? cli.jsonl_train : cli.jsonl_valid) << "\n";
        } else {
            std::cout << "  data_source     : WikiText-2 raw\n";
            std::cout << "  data_dir        : " << cli.data_dir << "\n";
        }
        std::cout << "  output_dir      : " << cli.output_dir << "\n";
        std::cout << "  epochs/steps    : " << cli.epochs << "/" << cli.steps << "\n";
        std::cout << "  stop_after_steps: " << cli.stop_after_steps << "\n";
        std::cout << "  sharding        : " << cli.shard_enable
                  << ", budget_MiB=" << cli.shard_budget_mb
                  << ", fp16_disk=" << cli.shard_fp16_disk << "\n";
        std::cout << "  batch/accum     : " << cli.batch << "/" << cli.grad_accum << "\n";
        std::cout << "  checkpointing   : " << cli.activation_checkpointing << "\n";
        std::cout << "  efficient_attn  : " << cli.memory_efficient_attention << "\n";

        std::ofstream telemetry;
        std::ofstream checkpoint_trace;
        if (!cli.measurement_dir.empty()) {
            std::filesystem::create_directories(cli.measurement_dir);
            telemetry.open(cli.measurement_dir + "/measurements.jsonl");
            if (!telemetry) throw std::runtime_error("Cannot open measurement output");
            telemetry.exceptions(std::ios::failbit | std::ios::badbit);
            telemetry << std::setprecision(17);
            checkpoint_trace.open(cli.measurement_dir + "/checkpoint_events.csv");
            if (!checkpoint_trace) throw std::runtime_error("Cannot open checkpoint profile output");
            checkpoint_trace.exceptions(std::ios::failbit | std::ios::badbit);
            checkpoint_trace << std::setprecision(17);
            checkpoint_trace << "step,call_id,phase,input_bytes,copy_payload_bytes,wall_sec,success,os_io_available,os_read_bytes,os_write_bytes\n";
            set_checkpoint_profiling(true);
        }
        std::cout << "  seq_len         : " << cli.seq_len << "\n";
        std::cout << "  lr/scheduler    : " << cli.learning_rate << "/" << cli.lr_scheduler << "\n";
        std::cout << "  warmup_steps    : " << cli.warmup_steps << "\n";
        std::cout << "  max_grad_norm   : " << cli.max_grad_norm << "\n";
        std::cout << "  optimizer       : " << cli.optimizer << "\n";
        std::cout << "  lora r/a/drop   : " << cli.lora_r << "/" << cli.lora_alpha << "/" << cli.lora_dropout << "\n";
        std::cout << "  target_mode     : " << cli.target_mode << "\n";
        std::cout << "  shuffle_train   : " << (cli.shuffle_train ? "true" : "false") << "\n";
        std::cout << "  logging/eval    : " << cli.logging_steps << "/" << cli.eval_steps << "\n";
        std::cout << "  save_every      : " << cli.save_every << "\n";
        std::cout << "  tokenizer       : " << ((use_jsonl || use_pretokenized) ? "pretokenized input" : "hf(native)") << "\n";

        std::filesystem::create_directories(cli.output_dir);
        if (!cli.eval_out.empty()) {
            std::filesystem::create_directories(std::filesystem::path(cli.eval_out).parent_path());
        }

        std::unique_ptr<HFTokenizer> hf_tok;
        std::function<std::vector<int32_t>(const std::string&)> encode_fn;

        QwenConfig qcfg = QwenConfig::from_pretrained(cli.model_dir + "/config.json");
        int eos_id = qcfg.eos_token_id;
        int pad_id = qcfg.pad_token_id;
        const bool use_model_dir = cli.tokenizer_json.empty();
        const std::string tok_path = use_model_dir ? cli.model_dir : cli.tokenizer_json;
        if (use_jsonl || use_pretokenized) {
            std::cout << "[Tokenizer] skipped for pre-tokenized JSONL input\n";
            encode_fn = [](const std::string&) -> std::vector<int32_t> {
                throw std::runtime_error("Tokenizer is unavailable for raw-text input in this build.");
            };
        } else {
            hf_tok = std::make_unique<HFTokenizer>(tok_path, use_model_dir);
            if (!hf_tok->load()) {
                throw std::runtime_error(
                    "Failed to load HF tokenizer natively. Native HF tokenizer is now the standard path; "
                    "rebuild with -DQWEN_USE_HF_TOKENIZERS=ON.");
            }
            std::cout << "[Tokenizer] HF native tokenizer loaded"
                      << (use_model_dir ? " from model_dir" : " from tokenizer_json")
                      << "\n";
            if (eos_id < 0) {
                for (auto t : {"<|endoftext|>", "<|eos|>", "</s>"}) {
                    int id = hf_tok->token_to_id(t);
                    if (id >= 0) {
                        eos_id = id;
                        break;
                    }
                }
            }
            encode_fn = [tok = hf_tok.get()](const std::string& text) {
                return tok->encode(text);
            };
        }
        if (pad_id < 0) pad_id = eos_id;
        QwenModel model(qcfg);

        SafeTensorsReader reader(cli.model_dir + "/model.safetensors");
        reader.parse_header();
        auto mapping = QwenKeyMapper::generate_qwen_mapping(qcfg.num_hidden_layers);
        SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;
        load_opts.transpose_linear = true;
        auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        if (session) {
            for (const auto& entry : mapping) {
                if (!tensors.count(entry.first)) {
                    throw std::runtime_error("Missing Qwen base tensor: " + entry.second);
                }
            }
        }
        for (auto& kv : tensors) {
            model.assign_weight(kv.first, kv.second);
        }

        QwenLoraMetadata lora_meta;
        lora_meta.rank = cli.lora_r;
        lora_meta.alpha = cli.lora_alpha;
        lora_meta.dropout = cli.lora_dropout;
        lora_meta.qv_only = qv_only;

        if (!cli.resume_from.empty()) {
            QwenLoraMetadata file_meta = lora_meta;
            if (QwenLoraSaver::parse_metadata(cli.resume_from, file_meta)) {
                lora_meta = file_meta;
            }
        }

        model.init_lora(lora_meta.rank, lora_meta.alpha, lora_meta.dropout, lora_meta.qv_only);
        if (session && lora_meta.dropout != 0) {
            throw std::runtime_error("Session mode does not support warm-start metadata with dropout");
        }
        model.freeze_base();
        if (!cli.resume_from.empty()) {
            std::cout << "[Resume] loading " << cli.resume_from << std::endl;
            QwenLoraSaver::load_safetensors(cli.resume_from, model, lora_meta);
        }
        auto trainable = model.get_lora_parameters();
        NamedParameters named;
        resume::Json adapter_layout = resume::Json::array();
        if (session) {
            for (int i = 0; i < model.config().num_hidden_layers; ++i) {
                const auto& block = model.get_block(i);
                // Match get_lora_parameters() order, not saver map iteration order.
                resume_cli::add_linear(named, adapter_layout, i, "attn.q", block.q_lin.get());
                resume_cli::add_linear(named, adapter_layout, i, "attn.v", block.v_lin.get());
                resume_cli::add_linear(named, adapter_layout, i, "attn.k", block.k_lin.get());
                resume_cli::add_linear(named, adapter_layout, i, "attn.proj", block.o_lin.get());
            }
            resume_cli::validate_parameters(named, trainable);
        }
        std::cout << "[LoRA] trainable tensors: " << trainable.size() << std::endl;
        // The loader map must not pin frozen weights after model ownership is established.
        tensors.clear();
        model.set_activation_checkpointing(cli.activation_checkpointing);
        model.set_memory_efficient_attention(cli.memory_efficient_attention);
        if (cli.shard_enable) {
            const auto init_started = measurement::Clock::now();
            const auto before = measurement::usage();
            sharding::ShardConfig config;
            config.offload_dir = cli.shard_dir.empty() ? cli.output_dir + "/shards" : cli.shard_dir;
            config.max_resident_bytes = cli.shard_budget_mb * 1024ull * 1024ull;
            config.quantize_fp16_on_disk = cli.shard_fp16_disk;
            if (telemetry.is_open()) config.io_trace_path = cli.measurement_dir + "/shard_io_events.csv";
            model.enable_parameter_sharding(config);
            const auto after = measurement::usage();
            const auto stats = model.sharding_stats();
            if (telemetry.is_open()) {
                telemetry << "{\"event\":\"shard_init\",\"wall_sec\":" << measurement::seconds(init_started)
                          << ",\"bytes_written\":" << stats.bytes_written << ",\"writes\":" << stats.writes
                          << ",\"write_io_sec\":" << stats.write_io_sec << ",\"write_wall_sec\":" << stats.write_wall_sec
                          << ",\"os_io_available\":" << (before.io_available && after.io_available ? "true" : "false")
                          << ",\"os_read_bytes\":" << (after.os_read_bytes - before.os_read_bytes)
                          << ",\"os_write_bytes\":" << (after.os_write_bytes - before.os_write_bytes) << "}\n";
                telemetry.flush();
            }
            std::cout << "[Sharding] local frozen-weight disk offload initialized; not a process RSS cap" << std::endl;
        }

        auto make_dataset = [&](Split split) -> std::unique_ptr<WikiText2Dataset> {
            WT2Config cfg;
            cfg.seq_len = cli.seq_len;
            cfg.stride = -1;
            cfg.eos_id = eos_id;
            cfg.pad_id = pad_id;
            cfg.insert_eos_between_lines = true;
            cfg.drop_last = (split == Split::Train);
            cfg.seed = cli.seed;
            cfg.shuffle_train = (split == Split::Train) ? cli.shuffle_train : false;
            cfg.streaming_mode = false;
            cfg.data_fraction = (split == Split::Train) ? cli.data_fraction : 1.0f;
            if (use_pretokenized) {
                cfg.pretokenized_path = cli.pretokenized_path;
                cfg.pretokenized_meta = cli.pretokenized_meta;
            } else if (use_jsonl) {
                if (session) cfg.jsonl_vocab_limit = qcfg.vocab_size;
                cfg.jsonl_train = cli.jsonl_train;
                cfg.jsonl_valid = cli.jsonl_valid.empty() ? cli.jsonl_train : cli.jsonl_valid;
                cfg.jsonl_test = cfg.jsonl_valid;
            } else {
                cfg.train_path = cli.data_dir + "/wiki.train.raw";
                cfg.valid_path = cli.data_dir + "/wiki.valid.raw";
                cfg.test_path = cli.data_dir + "/wiki.test.raw";
            }

            auto ds = std::make_unique<WikiText2Dataset>(cfg, encode_fn);
            ds->load(split);
            if (split == Split::Train) {
                if (cli.shuffle_train) ds->shuffle();
                else ds->reset_cursor();
            } else {
                ds->reset_cursor();
            }
            return ds;
        };

        auto train_dataset = make_dataset(Split::Train);
        auto valid_dataset = make_dataset(Split::Valid);
        if (session && train_dataset->num_sequences() < static_cast<size_t>(cli.batch)) {
            throw std::runtime_error("Training split is smaller than the original logical batch (drop_last)");
        }

        FixedBatch fixed_storage;
        Batch fixed_batch;
        bool use_fixed_batch = false;
        if (!cli.fixed_batch_bin.empty()) {
            fixed_storage = load_fixed_batch_bin(cli.fixed_batch_bin);
            use_fixed_batch = true;
            fixed_batch.input_ids = from_blob(
                fixed_storage.ids.data(),
                {static_cast<int64_t>(fixed_storage.B), static_cast<int64_t>(fixed_storage.S)},
                kInt32, kCPU);
            fixed_batch.attention_mask = from_blob(
                fixed_storage.attn.data(),
                {static_cast<int64_t>(fixed_storage.B), static_cast<int64_t>(fixed_storage.S)},
                kFloat32, kCPU);
            fixed_batch.labels = from_blob(
                fixed_storage.ids.data(),
                {static_cast<int64_t>(fixed_storage.B), static_cast<int64_t>(fixed_storage.S)},
                kInt32, kCPU);
            if (cli.fixed_batch_steps > 0) {
                cli.steps = cli.fixed_batch_steps;
            }
        }

        if (cli.dump_first_batch) {
            Batch batch0 = use_fixed_batch ? fixed_batch : train_dataset->get_batch(0, static_cast<size_t>(cli.batch));
            dump_batch(batch0, cli.dump_first_tokens, cli.dump_first_batch_path);
            if (cli.exit_after_dump) {
                std::cout << "[Dump] exit_after_dump=1, stopping before training." << std::endl;
                return 0;
            }
        }

        auto print_step0 = [&]() {
            NoGradGuard guard;
            Batch batch0 = use_fixed_batch ? fixed_batch : train_dataset->get_batch(0, static_cast<size_t>(cli.batch));
            auto logits0 = model.forward(batch0.input_ids, batch0.attention_mask);
            auto loss0 = lm_cross_entropy(logits0, batch0.labels, -100, "mean");
            const float loss_val = loss0->data<float>()[0];
            const float ppl = perplexity_from_loss(loss_val);
            std::cout << "[Step0] loss " << std::fixed << std::setprecision(6) << loss_val
                      << " ppl " << std::setprecision(2) << ppl
                      << " tokens " << count_tokens(batch0.attention_mask) << std::endl;
        };
        if (cli.print_step0 && !session) print_step0();

        const size_t num_seqs = train_dataset->num_sequences();
        const size_t total_micro_batches = use_fixed_batch
                                               ? 1
                                               : (num_seqs / static_cast<size_t>(std::max(1, cli.batch)));
        const size_t steps_per_epoch = (total_micro_batches + static_cast<size_t>(cli.grad_accum) - 1) / static_cast<size_t>(cli.grad_accum);

        int total_steps = cli.steps;
        if (total_steps <= 0) {
            total_steps = static_cast<int>(steps_per_epoch) * std::max(1, cli.epochs);
        }
        // A prefix check must not compress the original cosine schedule.
        const int run_steps = cli.stop_after_steps > 0
                                  ? std::min(total_steps, cli.stop_after_steps)
                                  : total_steps;
        std::cout << "[Dataset] train_sequences=" << num_seqs
                  << " valid_sequences=" << valid_dataset->num_sequences() << "\n";
        std::cout << "[Schedule] total_steps=" << total_steps
                  << " steps_per_epoch=" << steps_per_epoch
                  << " effective_batch=" << (cli.batch * cli.grad_accum) << std::endl;
        std::cout << "[Schedule] execution_steps=" << run_steps << std::endl;

        std::unique_ptr<Optimizer> optimizer;
        if (cli.optimizer == "adamw") {
            AdamWConfig opt_cfg;
            opt_cfg.learning_rate = cli.learning_rate;
            opt_cfg.beta1 = 0.9f;
            opt_cfg.beta2 = 0.999f;
            opt_cfg.epsilon = 1e-8f;
            opt_cfg.weight_decay = cli.weight_decay;
            opt_cfg.clip_grad_norm = cli.max_grad_norm;
            optimizer = std::make_unique<AdamW>(opt_cfg);
        } else {
            AdamConfig opt_cfg;
            opt_cfg.learning_rate = cli.learning_rate;
            opt_cfg.beta1 = 0.9f;
            opt_cfg.beta2 = 0.999f;
            opt_cfg.epsilon = 1e-8f;
            opt_cfg.weight_decay = cli.weight_decay;
            opt_cfg.clip_grad_norm = cli.max_grad_norm;
            optimizer = std::make_unique<Adam>(opt_cfg);
        }

        auto clip_and_get_grad_norm = [&](float max_norm) -> float {
            double norm_sq = 0.0;
            for (const auto& p : trainable) {
                auto g = p ? p->grad() : nullptr;
                if (!g) continue;
                const float* gd = g->data<float>();
                for (int64_t i = 0; i < g->numel(); ++i) {
                    norm_sq += static_cast<double>(gd[i]) * static_cast<double>(gd[i]);
                }
            }
            float total_norm = static_cast<float>(std::sqrt(norm_sq));
            if (max_norm > 0.0f && total_norm > max_norm) {
                const float scale = max_norm / (total_norm + 1e-6f);
                for (const auto& p : trainable) {
                    auto g = p ? p->grad() : nullptr;
                    if (!g) continue;
                    float* gd = g->data<float>();
                    for (int64_t i = 0; i < g->numel(); ++i) gd[i] *= scale;
                }
                total_norm = max_norm;
            }
            return total_norm;
        };

        resume_cli::TrainerState trainer;
        trainer.physical_batch = cli.batch;
        auto evaluate_valid = [&]() -> float {
            NoGradGuard guard;
            valid_dataset->reset_cursor();
            double loss_sum = 0.0;
            int batches = 0;
            const int max_batches = cli.eval_batches > 0
                                        ? cli.eval_batches
                                        : static_cast<int>((valid_dataset->num_sequences() + static_cast<size_t>(cli.batch) - 1) / static_cast<size_t>(cli.batch));
            while (batches < max_batches) {
                auto batch = valid_dataset->next_batch(static_cast<size_t>(cli.batch), false);
                if (!batch.input_ids) break;
                if (trainer.physical_batch < cli.batch) {
                    loss_sum += microbatch::logical_mean_loss(model, batch, trainer.physical_batch);
                    ++batches;
                    continue;
                }
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                loss_sum += static_cast<double>(loss->data<float>()[0]);
                batches++;
            }
            valid_dataset->reset_cursor();
            if (batches == 0) return std::numeric_limits<float>::infinity();
            return std::exp(static_cast<float>(loss_sum / static_cast<double>(batches)));
        };

        auto save_adapter = [&](const std::string& path) {
            auto write = [&]() {
                std::filesystem::create_directories(std::filesystem::path(path).parent_path());
                QwenLoraSaver::save_safetensors(path, model, lora_meta);
                if (session) resume_cli::validate_adapter_file(path, named);
            };
            if (session) resume_cli::saving(write);
            else write();
        };
        if (cli.save_initial_adapter && !session) {
            const auto path = cli.output_dir + "/lora_initial.safetensors";
            save_adapter(path);
            std::cout << "[Checkpoint] initial adapter saved " << path << std::endl;
        }

        auto& ema_loss = trainer.ema_loss;
        auto& ema_init = trainer.ema_init;
        auto& total_tokens = trainer.total_tokens;
        auto& best_valid_ppl = trainer.best_valid_ppl;
        auto& best_path = trainer.best_path;
        auto* adam = dynamic_cast<Adam*>(optimizer.get());
        if (session) {
            if (!adam || total_steps < 0) throw std::runtime_error("Invalid session optimizer or schedule");
            auto restored = session->restore(named, *adam, resume_cli::prepare_restore(
                trainer, total_steps, adapter_layout, *train_dataset, *valid_dataset, cli.batch));
            if (session->restoring() && !restored) throw std::runtime_error("No resumable training state");
            if (restored && invocation.resume_microbatch && trainer.completed_updates == total_steps) {
                throw std::runtime_error("--resume_microbatch cannot be applied to a completed checkpoint");
            }
            trainer.physical_batch = invocation.resume_microbatch.value_or(
                trainer.physical_batch == 0 ? cli.batch : trainer.physical_batch);
            if (restored) {
                session->resume_ready();
            }
            const auto effective_batch = static_cast<int64_t>(cli.batch) * cli.grad_accum;
            session->event({{"type", "batch_plan"}, {"logical_batch", cli.batch},
                {"physical_batch", trainer.physical_batch}, {"logical_accumulation", cli.grad_accum},
                {"physical_accumulation", resume_cli::physical_accumulation(cli.batch, trainer.physical_batch, cli.grad_accum)},
                {"effective_batch", effective_batch}});
        }
        const std::string final_path = cli.output_dir + "/lora.safetensors";
        bool final_exported = false;
        auto commit_boundary = [&](bool initial) -> bool {
            if (!session) return false;
            const int completed = trainer.completed_updates;
            const bool finished = completed == total_steps;
            const bool suspend = !finished && session->suspend_requested(completed);
            if (initial || finished || suspend || session->checkpoint_due(completed)) {
                resume_cli::saving([&]() {
                    if (finished) {
                        save_adapter(final_path);
                        final_exported = true;
                    }
                    session->commit(named, *adam,
                        resume_cli::dataset_state(*train_dataset, *valid_dataset),
                        trainer.json(total_steps, adapter_layout),
                        finished ? "completed" : (suspend ? "suspended" : "running"));
                });
            }
            if (suspend) {
                std::cout << "SUSPENDED completed_updates=" << completed
                          << " run_dir=" << session->run_dir() << std::endl;
            }
            return suspend;
        };
        if (session) {
            if (cli.save_initial_adapter && !session->restoring()) {
                save_adapter(cli.output_dir + "/lora_initial.safetensors");
            }
            // Never replace a restored generation with raw initialization.
            if ((!session->restoring() || trainer.completed_updates == total_steps ||
                 session->suspend_requested(trainer.completed_updates)) &&
                commit_boundary(!session->restoring())) return 0;
            if (cli.print_step0 && !session->restoring() && total_steps > 0) print_step0();
        }

        std::cout << "[Train] start" << std::endl;
        for (int step = trainer.completed_updates; step < run_steps; ++step) {
            const auto step_started = std::chrono::steady_clock::now();
            const double start_unix = measurement::unix_seconds();
            const auto usage_before = measurement::usage();
            const auto io_before = model.sharding_stats();
            const auto blocks_before = model.block_forward_calls();
            const auto attention_before = model.efficient_attention_calls();
            double forward_sec = 0, backward_sec = 0, loss_sec = 0;
            double accum_loss = 0.0;
            int64_t accum_tokens = 0;
            optimizer->zero_grad(trainable);
            std::vector<Batch> microbatches;
            std::vector<int64_t> label_counts;
            int64_t supervised_tokens = 0;
            for (int acc = 0; acc < cli.grad_accum; ++acc) {
                auto logical = use_fixed_batch ? fixed_batch
                    : train_dataset->next_batch(static_cast<size_t>(cli.batch), true);
                if (cli.grad_accum > 1) logical = own_microbatch(logical);
                if (trainer.physical_batch < cli.batch) {
                    microbatch::for_each_chunk(logical, trainer.physical_batch,
                        [&](Batch chunk) { microbatches.push_back(std::move(chunk)); });
                } else {
                    microbatches.push_back(std::move(logical));
                }
            }
            for (const auto& batch : microbatches) {
                label_counts.push_back(valid_lm_label_count(batch.labels, model.config().vocab_size));
                supervised_tokens += label_counts.back();
            }
            if (supervised_tokens == 0) throw std::runtime_error("Effective batch has no supervised tokens");
            GradientAccumulator gradient_accumulator(trainable.size());
            for (size_t acc = 0; acc < microbatches.size(); ++acc) {
                auto& batch = microbatches[acc];
                if (trainer.physical_batch < cli.batch && label_counts[acc] == 0) {
                    accum_tokens += count_tokens(batch.attention_mask);
                    continue;
                }
                const auto micro_started = measurement::Clock::now();
                auto section_started = measurement::Clock::now();
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                const double micro_forward = measurement::seconds(section_started);
                section_started = measurement::Clock::now();
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                const double micro_loss_time = measurement::seconds(section_started);
                const auto micro_loss = loss->data<float>()[0];
                accum_loss += static_cast<double>(micro_loss) * label_counts[acc];
                accum_tokens += count_tokens(batch.attention_mask);
                const float weight = static_cast<float>(label_counts[acc]) / static_cast<float>(supervised_tokens);
                section_started = measurement::Clock::now();
                auto scaled_loss = mul(loss, weight);
                scaled_loss->backward();
                if (microbatches.size() > 1) gradient_accumulator.collect(trainable);
                const double micro_backward = measurement::seconds(section_started);
                forward_sec += micro_forward;
                loss_sec += micro_loss_time;
                backward_sec += micro_backward;
                if (telemetry.is_open()) {
                    telemetry << "{\"event\":\"micro_step\",\"step\":" << step + 1 << ",\"micro\":" << acc + 1
                              << ",\"supervised_tokens\":" << label_counts[acc] << ",\"loss\":" << micro_loss
                              << ",\"forward_sec\":" << micro_forward << ",\"loss_sec\":" << micro_loss_time
                              << ",\"backward_sec\":" << micro_backward << ",\"wall_sec\":" << measurement::seconds(micro_started) << "}\n";
                }
            }
            if (microbatches.size() > 1) gradient_accumulator.assign(trainable);
            const auto optimizer_started = measurement::Clock::now();
            const float grad_norm = clip_and_get_grad_norm(cli.max_grad_norm);
            const float cur_lr = make_scheduler(step, total_steps, cli.warmup_steps, cli.learning_rate, cli.lr_scheduler);
            optimizer->set_learning_rate(cur_lr);

            std::vector<TensorPtr> grads;
            grads.reserve(trainable.size());
            for (const auto& p : trainable) grads.push_back(p->grad());
            optimizer->step(trainable, grads);
            for (auto& p : trainable) p->zero_grad();
            const double optimizer_sec = measurement::seconds(optimizer_started);
            const float avg_loss = static_cast<float>(accum_loss / static_cast<double>(supervised_tokens));
            if (!std::isfinite(avg_loss) || !std::isfinite(grad_norm)) {
                throw std::runtime_error("Non-finite training loss or gradient norm");
            }
            const double step_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - step_started).count();
            total_tokens += accum_tokens;
            if (!ema_init) {
                ema_loss = avg_loss;
                ema_init = true;
            } else {
                ema_loss = 0.9f * ema_loss + 0.1f * avg_loss;
            }

            if ((step + 1) % std::max(1, cli.logging_steps) == 0) {
                std::cout << "[Train] step " << (step + 1) << "/" << total_steps
                          << " lr " << std::fixed << std::setprecision(6) << cur_lr
                          << " loss " << std::setprecision(4) << avg_loss
                          << " ppl " << std::setprecision(2) << perplexity_from_loss(avg_loss)
                          << " ema_loss " << std::setprecision(4) << ema_loss
                          << " grad_norm " << std::setprecision(3) << grad_norm
                          << " tokens " << accum_tokens
                          << " step_time_sec " << std::setprecision(6) << step_seconds
                          << " rss_bytes " << MemoryMonitor::get_system_memory_usage()
                          << std::endl;
            }

            if (cli.eval_steps > 0 && (step + 1) % cli.eval_steps == 0) {
                const auto eval_started = measurement::Clock::now();
                const float valid_ppl = evaluate_valid();
                std::cout << "[Eval] step " << (step + 1) << "/" << total_steps
                          << " valid_ppl " << std::fixed << std::setprecision(2) << valid_ppl
                          << " ema_loss " << std::setprecision(4) << ema_loss
                          << " total_tokens " << total_tokens
                          << std::endl;
                std::ostringstream js;
                js << "{\"step\":" << (step + 1)
                   << ",\"valid_ppl\":" << valid_ppl
                   << ",\"ema_loss\":" << ema_loss
                   << ",\"total_tokens\":" << total_tokens
                   << "}";
                if (session) {
                    resume::Json evaluation = {{"event", "evaluation"}, {"type", "evaluation"},
                        {"step", step + 1}, {"completed_updates", step + 1},
                        {"valid_ppl", resume_cli::finite_or_null(valid_ppl)},
                        {"ema_loss", ema_loss}, {"total_tokens", total_tokens}};
                    append_jsonl(cli.eval_out, evaluation.dump());
                } else {
                    append_jsonl(cli.eval_out, js.str());
                }
                if (valid_ppl < best_valid_ppl) {
                    best_valid_ppl = valid_ppl;
                    best_path = session
                        ? cli.output_dir + "/lora_best_step" + std::to_string(step + 1) + ".safetensors"
                        : cli.output_dir + "/lora_best.safetensors";
                    save_adapter(best_path);
                    std::cout << "[Best] updated " << best_path << std::endl;
                }
                if (session) {
                    trainer.add_time("eval_sec", measurement::seconds(eval_started));
                    trainer.timing["evaluations"] = trainer.timing.at("evaluations").get<int>() + 1;
                    resume_cli::saving([&]() {
                        session->event({{"event", "evaluation"}, {"type", "evaluation"},
                            {"update_id", step + 1}, {"completed_updates", step + 1},
                            {"valid_ppl", resume_cli::finite_or_null(valid_ppl)}, {"best_path", best_path}});
                    });
                }
            }

            if (cli.save_every > 0 && (step + 1) % cli.save_every == 0) {
                const std::string ckpt_path = make_checkpoint_path(cli.output_dir, step + 1);
                save_adapter(ckpt_path);
                std::cout << "[Checkpoint] saved " << ckpt_path << std::endl;
            }

            const auto cleanup_started = measurement::Clock::now();
            if (session) {
                // No live graph, batch or accumulated gradients cross a safe point.
                microbatches.clear();
                grads.clear();
                gradient_accumulator = GradientAccumulator(0);
                ops::autograd::Engine::instance().clear_graph();
            }
            MemoryManager::instance().force_cleanup();
            if (cli.shard_enable) {
                // Backward has completed; only frozen base parameters are managed here.
                model.offload_base_parameters();
                const auto stats = model.sharding_stats();
                std::cout << "[Sharding] step=" << (step + 1)
                          << " entries=" << stats.entries
                          << " managed_resident_bytes=" << stats.resident_bytes
                          << " loads=" << stats.loads << " evictions=" << stats.evictions
                          << " bytes_read=" << stats.bytes_read
                          << " bytes_written=" << stats.bytes_written << std::endl;
            }
            if ((step + 1) % 50 == 0) {
                MemoryManager::instance().clear_unused_memory();
            }
            const double cleanup_sec = measurement::seconds(cleanup_started);
            const auto usage_after = measurement::usage();
            const auto io_after = model.sharding_stats();
            if (telemetry.is_open()) {
                telemetry << "{\"event\":\"step\",\"step\":" << step + 1
                          << ",\"start_unix_sec\":" << start_unix << ",\"end_unix_sec\":" << measurement::unix_seconds()
                          << ",\"full_step_sec\":" << measurement::seconds(step_started)
                          << ",\"core_step_sec\":" << step_seconds << ",\"forward_sec\":" << forward_sec
                          << ",\"loss_sec\":" << loss_sec << ",\"backward_sec\":" << backward_sec
                          << ",\"optimizer_sec\":" << optimizer_sec << ",\"cleanup_sec\":" << cleanup_sec
                          << ",\"loss\":" << avg_loss << ",\"ppl\":" << perplexity_from_loss(avg_loss)
                          << ",\"supervised_tokens\":" << supervised_tokens << ",\"input_tokens\":" << accum_tokens
                          << ",\"rss_start_bytes\":" << usage_before.rss_bytes << ",\"rss_end_bytes\":" << usage_after.rss_bytes
                          << ",\"footprint_end_bytes\":" << usage_after.footprint_bytes
                          << ",\"process_peak_so_far_bytes\":" << usage_after.peak_so_far_bytes
                          << ",\"os_io_available\":" << (usage_before.io_available && usage_after.io_available ? "true" : "false")
                          << ",\"os_read_bytes\":" << (usage_after.os_read_bytes - usage_before.os_read_bytes)
                          << ",\"os_write_bytes\":" << (usage_after.os_write_bytes - usage_before.os_write_bytes)
                          << ",\"shard_reads\":" << io_after.loads - io_before.loads
                          << ",\"shard_writes\":" << io_after.writes - io_before.writes
                          << ",\"shard_read_bytes\":" << io_after.bytes_read - io_before.bytes_read
                          << ",\"shard_write_bytes\":" << io_after.bytes_written - io_before.bytes_written
                          << ",\"shard_read_io_sec\":" << io_after.read_io_sec - io_before.read_io_sec
                          << ",\"shard_write_io_sec\":" << io_after.write_io_sec - io_before.write_io_sec
                          << ",\"shard_load_wall_sec\":" << io_after.load_wall_sec - io_before.load_wall_sec
                          << ",\"shard_write_wall_sec\":" << io_after.write_wall_sec - io_before.write_wall_sec
                          << ",\"shard_event_begin\":" << io_before.loads + io_before.writes + 1
                          << ",\"shard_event_end\":" << io_after.loads + io_after.writes
                          << ",\"block_forward_calls\":" << model.block_forward_calls() - blocks_before
                          << ",\"efficient_attention_calls\":" << model.efficient_attention_calls() - attention_before << "}\n";
                telemetry.flush();
                // Trace formatting/flush is outside the full-step timer snapshot.
                for (const auto& event : take_checkpoint_profile_events()) {
                    checkpoint_trace << step + 1 << ',' << event.call_id << ',' << event.phase << ','
                        << event.input_bytes << ',' << event.copy_payload_bytes << ',' << event.wall_sec << ','
                        << event.success << ',' << event.os_io_available << ',' << event.os_read_bytes << ','
                        << event.os_write_bytes << '\n';
                }
                checkpoint_trace.flush();
            }
            if (session) {
                // Never consume the next batch before committing this safe point.
                trainer.completed_updates = step + 1;
                trainer.timing["updates"] = step + 1;
                trainer.add_time("core_step_sec", step_seconds);
                trainer.add_time("full_step_sec", measurement::seconds(step_started));
                trainer.add_time("forward_sec", forward_sec);
                trainer.add_time("loss_sec", loss_sec);
                trainer.add_time("backward_sec", backward_sec);
                trainer.add_time("optimizer_sec", optimizer_sec);
                trainer.add_time("cleanup_sec", cleanup_sec);
                resume_cli::saving([&]() {
                    session->event({{"event", "update"}, {"type", "update"},
                        {"update_id", step + 1}, {"completed_updates", step + 1},
                        {"loss", avg_loss}, {"lr", cur_lr}, {"grad_norm", grad_norm},
                        {"total_tokens", total_tokens}, {"ema_loss", ema_loss}, {"ema_init", ema_init},
                        {"core_step_sec", step_seconds},
                        {"rss_bytes", MemoryMonitor::get_system_memory_usage()},
                        {"process_peak_so_far_bytes", usage_after.peak_so_far_bytes}});
                });
                if (commit_boundary(false)) return 0;
            }
        }

        if (!final_exported) save_adapter(final_path);
        std::cout << "\n🎉 Qwen LoRA training done. Saved adapter to " << final_path << std::endl;
        std::cout << "Total steps " << run_steps
                  << ", total tokens " << total_tokens
                  << ", final EMA loss " << std::fixed << std::setprecision(4) << ema_loss
                  << std::endl;
        return 0;
    } catch (const resume::CheckpointsExhausted& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return resume::CheckpointsExhausted::exit_code;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
