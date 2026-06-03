#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#endif

#include "finetune_ops/core/tokenizer_bpe.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/optim/adam.h"
#include "finetune_ops/data/mmlu_dataset.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/graph/qwen_model.h"
#include "finetune_ops/graph/lora_saver.h"

// 获取当前进程RSS（MB）
static double get_rss_mb() {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    // Linux: read from /proc/self/statm
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long pages;
        statm >> pages;  // total program size
        statm >> pages;  // RSS in pages
        long page_size = sysconf(_SC_PAGESIZE);
        return static_cast<double>(pages * page_size) / (1024.0 * 1024.0);
    }
    return 0.0;
#endif
}

using namespace ops;

struct Args {
    std::string model_dir = "Qwen2.5-0.5B";
    std::string data_dir = "data/mmlu/data";
    int seq_len = 128;
    int batch_size = 8;
    int grad_accum_steps = 1;
    int max_steps = 150;
    float lr = 2e-4f;
    int lora_r = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.0f;  // MMLU finetune typically keeps dropout off
    bool qv_only = true;        // attach LoRA on q and v only
    uint64_t seed = 42;
    int log_interval = 10;
    std::string output_dir = "outputs";
    std::string log_file = "";  // optional log file path
    bool save_weights = true;   // whether to save weights
    std::string base_weight_storage = "auto";
};

// Simple logger that mirrors stdout to file when enabled
class Logger {
public:
    Logger(const std::string& path) {
        if (!path.empty()) {
            file_.open(path);
            enabled_ = file_.is_open();
        }
    }
    
    ~Logger() {
        if (file_.is_open()) file_.close();
    }
    
    template<typename T>
    Logger& operator<<(const T& val) {
        std::cout << val;
        if (enabled_) file_ << val;
        return *this;
    }
    
    Logger& operator<<(std::ostream& (*pf)(std::ostream&)) {
        std::cout << pf;
        if (enabled_) file_ << pf;
        return *this;
    }
    
    void flush() {
        std::cout.flush();
        if (enabled_) file_.flush();
    }
    
private:
    std::ofstream file_;
    bool enabled_ = false;
};

static void parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto get_str = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("arg missing val: " + k);
            return std::string(argv[++i]);
        };
        auto get_int = [&]() -> int { return std::stoi(get_str()); };
        auto get_float = [&]() -> float { return std::stof(get_str()); };
        
        if (k == "--model_dir") args.model_dir = get_str();
        else if (k == "--data_dir") args.data_dir = get_str();
        else if (k == "--seq_len") args.seq_len = get_int();
        else if (k == "--batch_size") args.batch_size = get_int();
        else if (k == "--grad_accum_steps") args.grad_accum_steps = get_int();
        else if (k == "--max_steps") args.max_steps = get_int();
        else if (k == "--lr") args.lr = get_float();
        else if (k == "--lora_r") args.lora_r = get_int();
        else if (k == "--lora_alpha") args.lora_alpha = get_float();
        else if (k == "--lora_dropout") args.lora_dropout = get_float();
        else if (k == "--qv_only") args.qv_only = true;
        else if (k == "--qkvo") args.qv_only = false;
        else if (k == "--seed") args.seed = static_cast<uint64_t>(std::stoull(get_str()));
        else if (k == "--log_interval") args.log_interval = get_int();
        else if (k == "--output_dir") args.output_dir = get_str();
        else if (k == "--log_file") args.log_file = get_str();
        else if (k == "--no_save") args.save_weights = false;
        else if (k == "--base_weight_storage") args.base_weight_storage = get_str();
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

void print_config(const Args& args) {
    std::cout << "\n========== MMLU LoRA Finetune Config ==========\n";
    std::cout << "Model: " << args.model_dir << "\n";
    std::cout << "Data: " << args.data_dir << "\n";
    std::cout << "Seq length: " << args.seq_len << "\n";
    std::cout << "Batch size: " << args.batch_size << "\n";
    std::cout << "Grad accum: " << args.grad_accum_steps << "\n";
    std::cout << "Effective batch: " << args.batch_size * args.grad_accum_steps << "\n";
    std::cout << "Max steps: " << args.max_steps << "\n";
    std::cout << "Learning rate: " << args.lr << "\n";
    std::cout << "LoRA rank: " << args.lora_r << "\n";
    std::cout << "LoRA alpha: " << args.lora_alpha << "\n";
    std::cout << "LoRA targets: " << (args.qv_only ? "q, v only" : "q, k, v, o") << "\n";
    std::cout << "Seed: " << args.seed << "\n";
    std::cout << "Base weight storage: " << args.base_weight_storage << "\n";
    std::cout << "================================================\n\n";
}

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);
    
    // 设置默认日志文件
    if (args.log_file.empty()) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "logs/mmlu_cpp_" << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S") << ".log";
        args.log_file = ss.str();
    }
    
    // 创建输出目录
    std::filesystem::create_directories(args.output_dir);
    std::filesystem::create_directories("logs");
    
    Logger log(args.log_file);
    
    log << "\n========== MMLU LoRA Finetune Config ==========\n";
    log << "Model: " << args.model_dir << "\n";
    log << "Data: " << args.data_dir << "\n";
    log << "Seq length: " << args.seq_len << "\n";
    log << "Batch size: " << args.batch_size << "\n";
    log << "Grad accum: " << args.grad_accum_steps << "\n";
    log << "Effective batch: " << args.batch_size * args.grad_accum_steps << "\n";
    log << "Max steps: " << args.max_steps << "\n";
    log << "Learning rate: " << args.lr << "\n";
    log << "LoRA rank: " << args.lora_r << "\n";
    log << "LoRA alpha: " << args.lora_alpha << "\n";
    log << "LoRA targets: " << (args.qv_only ? "q, v only" : "q, k, v, o") << "\n";
    log << "Log file: " << args.log_file << "\n";
    log << "Output dir: " << args.output_dir << "\n";
    log << "================================================\n\n";
    
    try {
        double init_rss = get_rss_mb();
        log << "[Init] RSS: " << std::fixed << std::setprecision(1) << init_rss << " MB\n";
        
        // 1) Tokenizer
        log << "[1/6] Loading tokenizer...\n";
        auto tok_cfg = QwenTokenizerConfig::from_pretrained(args.model_dir);
        QwenBPETokenizer tokenizer(tok_cfg);
        tokenizer.load();
        
        // 2) Model config
        log << "[2/6] Loading model config...\n";
        QwenConfig qcfg = QwenConfig::from_pretrained(args.model_dir + "/config.json");
        QwenModel model(qcfg);
        
        // 3) Load weights
        log << "[3/6] Loading model weights...\n";
        SafeTensorsModelReader reader(args.model_dir);
        reader.parse_headers();
        auto mapping = QwenKeyMapper::generate_qwen_mapping(qcfg.num_hidden_layers);
        SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;
        load_opts.transpose_linear = true;
        configure_qwen_base_weight_storage(load_opts, args.base_weight_storage);
        auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (auto& kv : tensors) {
            model.assign_weight(kv.first, kv.second);
        }
        // Release constructor placeholder tensors that were replaced by real
        // safetensors before any training allocations begin.
        MemoryManager::instance().clear_unused_memory();
        
        double after_load_rss = get_rss_mb();
        log << "[After model load] RSS: " << std::fixed << std::setprecision(1) << after_load_rss << " MB\n";
        
        // 4) LoRA init (qv_only=true)
        log << "[4/6] Initializing LoRA (q/v only)...\n";
        model.init_lora(args.lora_r, args.lora_alpha, args.lora_dropout, args.qv_only, args.seed);
        model.freeze_base();
        auto lora_params = model.get_lora_parameters();
        log << "  Total LoRA parameters: " << lora_params.size() << "\n";
        
        // 计算LoRA参数量
        int64_t total_params = 0;
        for (const auto& p : lora_params) {
            total_params += p->numel();
        }
        log << "  Total LoRA param count: " << total_params << " (~" 
            << (total_params * 4 / 1024 / 1024) << " MB)\n";
        
        // 5) Dataset
        log << "[5/6] Loading MMLU dataset...\n";
        MMLUConfig dcfg;
        dcfg.data_dir = args.data_dir;
        dcfg.seq_len = args.seq_len;
        dcfg.batch_size = args.batch_size;
        
        MMLUDataset ds(dcfg, &tokenizer);
        ds.load_train();
        ds.shuffle();
        ds.print_stats();
        
        // 6) Optimizer
        log << "[6/6] Setting up optimizer...\n";
        AdamConfig opt_cfg;
        opt_cfg.learning_rate = args.lr;
        opt_cfg.beta1 = 0.9f;
        opt_cfg.beta2 = 0.999f;
        opt_cfg.epsilon = 1e-8f;
        opt_cfg.weight_decay = 0.0f;
        Adam opt(opt_cfg);
        
        // Training loop
        log << "\n========== Training Start ==========\n";
        log << "step,loss,tokens,rss_mb,time_s\n";  // CSV header
        
        int step = 0;
        float accum_loss = 0.0f;
        int accum_counter = 0;
        int total_tokens = 0;
        double max_rss = 0.0;
        
        auto train_start = std::chrono::high_resolution_clock::now();
        
        while (step < args.max_steps) {
            auto batch = ds.next_batch(true);  // loop=true
            
            if (batch.num_samples == 0) {
                ds.shuffle();
                ds.reset_cursor();
                continue;
            }
            
            // Forward
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            
            // Measure RSS right after forward (peak usage point)
            double rss_after_forward = get_rss_mb();
            if (rss_after_forward > max_rss) max_rss = rss_after_forward;
            
            // Masked cross-entropy loss
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            
            float loss_val = loss->data<float>()[0];
            accum_loss += loss_val;
            
            // Count valid tokens
            const int32_t* labels_ptr = batch.labels->data<int32_t>();
            int valid_tokens = 0;
            for (int64_t i = 0; i < batch.labels->numel(); ++i) {
                if (labels_ptr[i] != -100) valid_tokens++;
            }
            total_tokens += valid_tokens;
            
            // Backward
            loss->backward();
            
            // Measure RSS again after backward
            double rss_after_backward = get_rss_mb();
            if (rss_after_backward > max_rss) max_rss = rss_after_backward;
            
            accum_counter++;
            
            // Update parameters after reaching the grad accumulation span
            if (accum_counter >= args.grad_accum_steps) {
                // Average gradients when accumulating
                if (args.grad_accum_steps > 1) {
                    float scale = 1.0f / static_cast<float>(args.grad_accum_steps);
                    for (auto& p : lora_params) {
                        auto g = p->grad();
                        if (g) {
                            float* d = g->data<float>();
                            for (int64_t i = 0; i < g->numel(); ++i) {
                                d[i] *= scale;
                            }
                        }
                    }
                }
                
                // Collect grads and update
                std::vector<TensorPtr> grads;
                for (auto& p : lora_params) {
                    grads.push_back(p->grad());
                }
                
                opt.step(lora_params, grads);
                
                // Zero grad
                for (auto& p : lora_params) {
                    p->zero_grad();
                }
                
                step++;
                
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - train_start).count();
                
                // Log every step (CSV)
                float avg_loss = accum_loss / args.grad_accum_steps;
                log << step << "," << std::fixed << std::setprecision(4) << avg_loss 
                    << "," << total_tokens 
                    << "," << std::setprecision(1) << rss_after_backward
                    << "," << std::setprecision(2) << elapsed << "\n";
                
                // Periodically print human-readable summary
                if (step % args.log_interval == 0 || step == 1) {
                    double steps_per_sec = step / elapsed;
                    std::cout << "[step " << std::setw(4) << step << "/" << args.max_steps << "] "
                              << "loss=" << std::fixed << std::setprecision(4) << avg_loss
                              << " | tokens=" << total_tokens
                              << " | RSS=" << std::setprecision(1) << rss_after_backward << "MB"
                              << " | speed=" << std::setprecision(2) << steps_per_sec << " steps/s"
                              << std::endl;
                }
                
                // Memory cleanup
                MemoryManager::instance().force_cleanup();
                
                // Reset accumulators
                accum_loss = 0.0f;
                accum_counter = 0;
            }
        }
        
        auto train_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(train_end - train_start).count();
        
        log << "\n========== Training Complete ==========\n";
        log << "Total steps: " << step << "\n";
        log << "Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds\n";
        log << "Average speed: " << (step / total_time) << " steps/second\n";
        log << "Total tokens trained: " << total_tokens << "\n";
        log << "Max RSS: " << std::setprecision(1) << max_rss << " MB\n";
        log << "Final RSS: " << get_rss_mb() << " MB\n";
        
        // Save LoRA weights
        if (args.save_weights) {
            std::string weight_path = args.output_dir + "/lora_weights.bin";
            log << "\nSaving LoRA weights to: " << weight_path << "\n";
            
            std::ofstream wf(weight_path, std::ios::binary);
            if (wf.is_open()) {
                int32_t num_params = static_cast<int32_t>(lora_params.size());
                wf.write(reinterpret_cast<char*>(&num_params), sizeof(num_params));
                
                for (const auto& p : lora_params) {
                    // write shape
                    int32_t ndim = static_cast<int32_t>(p->shape().size());
                    wf.write(reinterpret_cast<char*>(&ndim), sizeof(ndim));
                    for (int64_t dim : p->shape()) {
                        int64_t d = dim;
                        wf.write(reinterpret_cast<char*>(&d), sizeof(d));
                    }
                    // write data
                    int64_t numel = p->numel();
                    wf.write(reinterpret_cast<const char*>(p->data<float>()), numel * sizeof(float));
                }
                wf.close();
                log << "✅ Weights saved successfully\n";
            } else {
                log << "⚠️ Failed to save weights\n";
            }
        }
        
        log << "\n✅ MMLU LoRA Finetune finished successfully!\n";
        log.flush();
        
    } catch (const std::exception& e) {
        log << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
