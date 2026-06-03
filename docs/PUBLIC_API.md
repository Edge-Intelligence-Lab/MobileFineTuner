# MobileFineTuner Public API

MobileFineTuner exposes one supported C++ entrypoint:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"
```

Downstream projects should link the installed CMake target:

```cmake
find_package(MobileFineTuner REQUIRED)
target_link_libraries(your_target PRIVATE MobileFineTuner::operators)
```

The `finetune_ops/*` headers are still installed because the current examples
use them directly, but new applications should include the umbrella header
first. Historical or non-buildable APIs are excluded from the supported product
surface.

Model weights and datasets are external assets. The library accepts standard
HuggingFace-style model directories with config/tokenizer files and either a
single `model.safetensors` file or `model.safetensors.index.json` plus all
referenced shard files. See `docs/MODEL_ASSETS.md` for the asset contract.

## Stable Auto APIs

Applications should discover model and tokenizer assets through the public
registry APIs instead of hard-coding per-model training directories. For common
fine-tuning code, use `AutoModelForCausalLM` and `AutoTrainer` so the application
does not branch manually between GPT-2, Gemma, and Qwen graph classes.

```cpp
#include "mobile_finetuner/mobile_finetuner.h"

auto spec = ops::ModelRegistry::inspect_pretrained(model_dir);
auto tokenizer = ops::TokenizerFactory::from_pretrained(model_dir);

auto model = ops::AutoModelForCausalLM::from_pretrained(model_dir);
model->init_lora(ops::AutoLoraConfig::attention_qkvo());

ops::AutoTrainerConfig trainer_cfg;
trainer_cfg.learning_rate = 2e-4f;
ops::AutoTrainer trainer(*model, trainer_cfg);
auto result = trainer.train_step(input_ids, attention_mask, labels);
```

`ModelRegistry` is the C++ equivalent of a small `AutoConfig` layer: it reads
`config.json`, identifies the supported model family, records tokenizer and
SafeTensors asset paths, and exposes default LoRA target names. `TokenizerFactory`
is the matching `AutoTokenizer` layer: it keeps GPT-2, Qwen, and Gemma tokenizers
behind one `ops::Tokenizer` interface while still using the model-specific
tokenization algorithm required by each checkpoint.

`AutoModelForCausalLM` is the matching model dispatcher. It constructs the
concrete graph class, loads SafeTensors with family-correct transpose defaults,
exposes one `forward(input_ids, attention_mask)` call, initializes LoRA, and
returns trainable parameters. `AutoTrainer` provides the shared one-step LoRA
training core: forward, LM cross entropy, backward, gradient clipping, Adam
update, and zero-grad.

The current Auto APIs cover the shared causal-LM/LoRA training core. Specialized
dataset loops, logging, checkpoint schedules, and model-specific diagnostics can
still use the lower-level graph classes directly.
