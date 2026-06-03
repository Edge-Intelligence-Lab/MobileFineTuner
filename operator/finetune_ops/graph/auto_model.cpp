#include "auto_model.h"

#include "../core/memory_manager.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ops {

namespace {

bool contains_module(const std::vector<std::string>& modules, const std::string& name) {
    return std::find(modules.begin(), modules.end(), name) != modules.end();
}

LoraSpec make_gpt2_lora_spec(const AutoLoraConfig& cfg) {
    LoraSpec spec;
    spec.rank = cfg.rank;
    spec.alpha = cfg.alpha;
    spec.dropout = cfg.dropout;
    spec.split_qkv = true;

    const auto& modules = cfg.target_modules;
    if (modules.empty()) {
        spec.targets = {LoraTarget::AttnQKV, LoraTarget::AttnProj};
        return spec;
    }

    spec.targets.clear();
    if (contains_module(modules, "attn_qkv") ||
        contains_module(modules, "c_attn") ||
        contains_module(modules, "q_proj") ||
        contains_module(modules, "k_proj") ||
        contains_module(modules, "v_proj")) {
        spec.targets.push_back(LoraTarget::AttnQKV);
    }
    if (contains_module(modules, "attn_proj") ||
        contains_module(modules, "c_proj") ||
        contains_module(modules, "o_proj")) {
        spec.targets.push_back(LoraTarget::AttnProj);
    }
    if (contains_module(modules, "mlp_fc_in") || contains_module(modules, "c_fc")) {
        spec.targets.push_back(LoraTarget::MlpFcIn);
    }
    if (contains_module(modules, "mlp_fc_out") || contains_module(modules, "mlp_c_proj")) {
        spec.targets.push_back(LoraTarget::MlpFcOut);
    }
    if (spec.targets.empty()) {
        throw std::runtime_error("AutoModelForCausalLM: no supported GPT-2 LoRA targets requested");
    }
    return spec;
}

GemmaLoraSpec make_gemma_lora_spec(const AutoLoraConfig& cfg) {
    GemmaLoraSpec spec;
    spec.rank = cfg.rank;
    spec.alpha = cfg.alpha;
    spec.dropout = cfg.dropout;
    if (!cfg.target_modules.empty()) {
        spec.target_modules = cfg.target_modules;
    }
    return spec;
}

bool qwen_qv_only(const AutoLoraConfig& cfg) {
    if (cfg.target_modules.empty()) {
        return false;
    }
    std::unordered_set<std::string> requested(cfg.target_modules.begin(), cfg.target_modules.end());
    return requested.count("q_proj") > 0 &&
           requested.count("v_proj") > 0 &&
           requested.count("k_proj") == 0 &&
           requested.count("o_proj") == 0;
}

SafeTensorsLoadOptions family_load_options(ModelFamily family,
                                           const SafeTensorsLoadOptions& base,
                                           bool verbose) {
    SafeTensorsLoadOptions opts = base;
    opts.verbose = verbose;
    switch (family) {
        case ModelFamily::GPT2:
            // GPT-2 safetensors are already stored in the internal [in, out] layout.
            opts.transpose_linear = false;
            break;
        case ModelFamily::Gemma:
        case ModelFamily::Qwen:
            // Gemma/Qwen HF linear weights are [out, in]; MF kernels expect [in, out].
            opts.transpose_linear = true;
            break;
        default:
            break;
    }
    return opts;
}

}  // namespace

AutoModelForCausalLM::AutoModelForCausalLM(std::string model_dir,
                                           ModelArchitectureSpec spec)
    : model_dir_(std::move(model_dir)), spec_(std::move(spec)) {}

std::unique_ptr<AutoModelForCausalLM>
AutoModelForCausalLM::from_pretrained(const std::string& model_dir,
                                      const AutoModelLoadOptions& options) {
    auto spec = ModelRegistry::inspect_pretrained(model_dir);
    auto model = std::unique_ptr<AutoModelForCausalLM>(
        new AutoModelForCausalLM(model_dir, spec));

    switch (model->spec_.family) {
        case ModelFamily::GPT2: {
            GPT2Config cfg = GPT2Config::from_pretrained(model_dir);
            model->gpt2_ = std::make_unique<GPT2Model>(cfg);
            if (cfg.tie_word_embeddings) {
                model->gpt2_->tie_weights();
            }
            break;
        }
        case ModelFamily::Gemma: {
            GemmaTextConfig cfg = GemmaTextConfig::from_pretrained(model_dir);
            model->gemma_ = std::make_unique<GemmaModel>(cfg);
            break;
        }
        case ModelFamily::Qwen: {
            QwenConfig cfg = QwenConfig::from_pretrained(model_dir + "/config.json");
            model->qwen_ = std::make_unique<QwenModel>(cfg);
            break;
        }
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }

    if (options.load_weights) {
        model->load_pretrained_weights(options);
    }
    return model;
}

void AutoModelForCausalLM::load_pretrained_weights(const AutoModelLoadOptions& options) {
    SafeTensorsModelReader reader(model_dir_);
    reader.parse_headers();

    const auto load_opts =
        family_load_options(spec_.family, options.safetensors_options, options.verbose);

    switch (spec_.family) {
        case ModelFamily::GPT2: {
            auto mapping = GPT2KeyMapper::generate_gpt2_mapping(gpt2_->config().n_layer);
            auto tensors = reader.load_tensors_mapped(mapping, load_opts);
            for (const auto& kv : tensors) {
                gpt2_->assign_weight(kv.first, kv.second);
            }
            break;
        }
        case ModelFamily::Gemma: {
            auto mapping = GemmaKeyMapper::generate_gemma_mapping(gemma_->config().num_hidden_layers);
            auto tensors = reader.load_tensors_mapped(mapping, load_opts);
            for (const auto& kv : tensors) {
                gemma_->assign_weight(kv.first, kv.second);
            }
            break;
        }
        case ModelFamily::Qwen: {
            auto mapping = QwenKeyMapper::generate_qwen_mapping(qwen_->config().num_hidden_layers);
            auto tensors = reader.load_tensors_mapped(mapping, load_opts);
            for (const auto& kv : tensors) {
                qwen_->assign_weight(kv.first, kv.second);
            }
            MemoryManager::instance().clear_unused_memory();
            break;
        }
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

TensorPtr AutoModelForCausalLM::forward(const TensorPtr& input_ids,
                                        const TensorPtr& attention_mask) {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->forward(input_ids, attention_mask);
        case ModelFamily::Gemma:
            return gemma_->forward(input_ids, attention_mask);
        case ModelFamily::Qwen:
            return qwen_->forward(input_ids, attention_mask);
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

void AutoModelForCausalLM::init_lora(const AutoLoraConfig& config) {
    switch (spec_.family) {
        case ModelFamily::GPT2: {
            gpt2_->init_lora_modules();
            gpt2_lora_ = std::make_unique<LoraInjector>();
            gpt2_lora_->inject(*gpt2_, make_gpt2_lora_spec(config));
            break;
        }
        case ModelFamily::Gemma: {
            gemma_lora_ = std::make_unique<GemmaLoraInjector>();
            gemma_lora_->inject(*gemma_, make_gemma_lora_spec(config));
            break;
        }
        case ModelFamily::Qwen:
            qwen_->init_lora(config.rank, config.alpha, config.dropout,
                             qwen_qv_only(config), config.seed);
            qwen_->freeze_base();
            break;
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

std::vector<TensorPtr> AutoModelForCausalLM::parameters() {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->parameters();
        case ModelFamily::Gemma:
            return gemma_->parameters();
        case ModelFamily::Qwen:
            return qwen_->parameters();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

std::vector<TensorPtr> AutoModelForCausalLM::trainable_parameters() {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->get_lora_parameters();
        case ModelFamily::Gemma:
            return gemma_->get_lora_parameters();
        case ModelFamily::Qwen:
            return qwen_->get_lora_parameters();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

}  // namespace ops
