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

For Qwen2/Qwen3 dense and Qwen3.5 hybrid text models, `AutoLoraConfig::target_modules` selects exact
linear module suffixes (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`) or `all-linear`. Shapes come from the actual model
matrices. Unknown or duplicate targets are rejected. The Qwen graph exposes
`save_lora_adapter(directory)`, `load_lora_adapter(directory)`, and
`set_training(bool)` through `model->qwen()`; see [Qwen support](QWEN_DENSE_SUPPORT.md)
for the standard PEFT checkpoint contract and supported options.
Qwen3.5 additionally exposes `in_proj_qkv`, `in_proj_z`, `in_proj_a`,
`in_proj_b`, and `out_proj` on linear-attention layers.

## Opt-in Energy-Aware Throttling

`ops::energy::PowerMonitor` is a model-independent scheduler exposed by the same
umbrella header. It suggests idle milliseconds after completed optimizer updates
using an explicit schedule or application-provided battery/temperature readings.
It is off by default and does not read sensors, sleep, or change model parameters
itself. See [ENERGY_SCHEDULER.md](ENERGY_SCHEDULER.md) for the API, Qwen example
integration, timing semantics, and tests. Throttling is not evidence of reduced
total energy; that requires measurements on the target device.

## Android SDK API

Android applications can consume the AAR built from `android-visualizer/mft-sdk`.
The Java entrypoint is:

```java
import com.mobilefinetuner.sdk.MobileFineTuner;
```

It wraps the same C++ `AutoModelForCausalLM` and `AutoTrainer` objects through
JNI. The Android SDK is a packaging layer, not a separate implementation. See
`docs/ANDROID_SDK.md` for build, ABI, asset layout, and Java call-order details.

The Android API also exposes `MobileFineTuner.selfTest(File workingDir)`, which
runs a tiny synthetic native LoRA training step on device. This is intended for
installation validation and CI/device smoke testing; real applications should
still pass their own HuggingFace-style `modelDir` to `MobileFineTuner.open`.

## Training lifetime and concurrency

Initialize LoRA before constructing `AutoTrainer`; repeated Auto LoRA
initialization is rejected to avoid stale optimizer parameter references. Keep
the model alive for the trainer's lifetime. Seed and adapter-branch dropout are
bound by the GPT-2, Gemma and Qwen injectors. Gemma/Qwen checkpointed training
requires zero LoRA dropout because stochastic recomputation is unsupported.
A zero `max_grad_norm` disables clipping; non-finite loss/gradients reject the
step and clear the graph and gradients. This is not a transactional rollback
of an optimizer failure after updates have begun.

The autograd engine is process-wide: C++ callers must serialize training and
other graph-affecting calls across model instances. The Android JNI bridge
serializes native entrypoints across sessions; Java instance operations also
synchronize close with active calls. Run training on a worker thread because
close can wait for an active step. Device concurrency tests are included in the
instrumentation suite and require a connected device to execute.
