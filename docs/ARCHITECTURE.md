# MobileFineTuner Architecture

MobileFineTuner uses a PyTorch/Transformers-like asset discovery split, but
keeps the training path in native C++:

```text
HuggingFace-style model_dir
  config.json
  tokenizer assets
  model.safetensors or model.safetensors.index.json
        |
        v
ModelRegistry      -> model family, asset paths, default LoRA targets
TokenizerFactory   -> model-specific tokenizer through one interface
AutoModelForCausalLM -> concrete graph dispatch + SafeTensors loading + LoRA
AutoTrainer       -> shared one-step causal-LM LoRA training core
Graph class        -> GPT-2 / Gemma / Qwen forward and backward math
SafeTensorsLoader  -> external checkpoint tensors into graph weights
LoRA modules       -> trainable adapters on named target projections
Optimizer          -> native C++ parameter update
Android SDK AAR    -> Java/JNI packaging over AutoModel + AutoTrainer
```

## Design Principles

- Model files are runtime assets, not source files.
- Each model family keeps its own tokenizer algorithm.
- Applications should depend on stable discovery APIs, not on directory-specific
  training scripts.
- The core library should expose small, auditable interfaces and keep research
  experiments outside the supported product surface.

## Public Discovery Layer

Use the umbrella header:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"
```

Inspect a model directory:

```cpp
auto spec = ops::ModelRegistry::inspect_pretrained(model_dir);
```

`ModelRegistry` reads `config.json` and returns:

- `family`: `GPT2`, `Gemma`, or `Qwen`;
- `model_type`: the raw HuggingFace `model_type`;
- tokenizer and SafeTensors asset paths;
- single-file vs sharded SafeTensors availability;
- default LoRA target names for that family;
- `tie_word_embeddings` behavior;
- engine-aware training capabilities, distinct from enabled execution switches.

Load a tokenizer:

```cpp
auto tokenizer = ops::TokenizerFactory::from_pretrained(model_dir);
auto encoded = tokenizer->encode_with_attention(prompt, 128);
```

`TokenizerFactory` is intentionally an `AutoTokenizer`-style dispatcher. It
standardizes the C++ call site but does not collapse different tokenizers into
one algorithm. GPT-2 byte-level BPE, Qwen byte-level BPE, and Gemma tokenizer
logic remain separate because their vocabularies, special tokens, and
pre-tokenization rules differ.

## Model Graph Layer

Each supported architecture has a graph class under `operator/finetune_ops/graph`:

```text
gpt2_model.{h,cpp}
gemma_model.{h,cpp}
qwen_model.{h,cpp}
```

The graph owns:

- architecture configuration;
- tensor allocation for model parameters;
- `forward(input_ids, attention_mask)`;
- HuggingFace weight-name mapping via `assign_weight`;
- LoRA target attachment;
- trainable parameter discovery.

The graph classes remain explicit instead of hidden behind one large
runtime-polymorphic base class. This keeps the math readable and makes it clear
where architecture-specific behavior lives. Generic application code should use
`AutoModelForCausalLM` when it only needs the common causal-LM LoRA surface, and
drop to concrete graph classes when it needs model-specific diagnostics or
alignment hooks.

`AutoModelForCausalLM` uses `ModelRegistry` to construct GPT-2, Gemma, or Qwen,
load SafeTensors with family-correct layout defaults, initialize LoRA, run
forward, and expose trainable parameters. `AutoTrainer` sits one layer above it
and implements the shared one-step training core:

```text
input_ids + attention_mask + labels
        |
        v
AutoModelForCausalLM::forward
        |
        v
lm_cross_entropy -> backward -> grad clip -> Adam -> zero_grad
```

## Tokenizer Extension Contract

When adding a model:

1. Add or reuse a tokenizer implementation that matches the upstream
   HuggingFace tokenizer behavior.
2. Wrap it in an `ops::Tokenizer` adapter.
3. Register it in `TokenizerFactory::from_pretrained`.
4. Add a factory test that verifies `config.json` inference and adapter
   construction without loading large model weights.

The factory must not infer tokenizer type from `tokenizer.json` alone. Many
model families use that filename with incompatible semantics. Use
`config.json.model_type` or an explicit `TokenizerLoadOptions::model_type`.

Tokenizer correctness is validated in two layers:

- `TokenizerFactory` uses tiny in-tree fixtures to verify loading, special
  tokens, padding masks, and basic encode/decode behavior without external
  assets.
- `TokenizerHFGolden` optionally compares C++ token IDs against HuggingFace
  `AutoTokenizer` standard-answer JSONL fixtures generated from local model
  snapshots. Generate them with:

```bash
python3 scripts/generate_tokenizer_hf_golden_fixtures.py \
  --output runs/tokenizer_golden/hf_tokenizer_golden.jsonl
MFT_TOKENIZER_GOLDEN_JSONL=runs/tokenizer_golden/hf_tokenizer_golden.jsonl \
  ctest --test-dir operator/build --output-on-failure -R TokenizerHFGolden
```

The golden fixture path is intentionally external to the source package because
it contains local model paths. The generated token ID sequences are small, but
they are tied to the exact tokenizer snapshot used to generate them.

## Model Extension Contract

When adding a model family:

1. Add a config parser aligned with the upstream HuggingFace config.
2. Add a graph class under `finetune_ops/graph`.
3. Map SafeTensors keys explicitly in `assign_weight`.
4. Define default LoRA targets using upstream module names.
5. Register the family in `ModelRegistry`.
6. Add small synthetic tests first, then real-asset smoke tests outside the
   source package.

This mirrors the discovery part of the PyTorch/Transformers pattern:

```text
AutoConfig      -> ModelRegistry
AutoTokenizer   -> TokenizerFactory
AutoModel class  -> AutoModelForCausalLM
Trainer core     -> AutoTrainer
PEFT targets    -> default_lora_targets + graph LoRA injection
```

## Why This Is Cleaner Than Per-Directory Hardcoding

The previous training directories are useful runnable examples, but they should
not be the long-term architecture boundary. A new model should not require
copying a full application directory just to change tokenizer or config parsing.

With the discovery layer:

- application code can validate a `model_dir` before training;
- tokenizer selection is centralized;
- LoRA target defaults are visible and testable;
- asset contracts stay consistent across desktop and Android;
- unsupported model families fail with explicit errors instead of silently using
  the wrong tokenizer.

## Android Packaging Layer

The Android SDK lives under `android-visualizer/mft-sdk` and builds an AAR. It
adds a Java class, `com.mobilefinetuner.sdk.MobileFineTuner`, plus a small JNI
bridge. The bridge owns a native `AutoModelForCausalLM` and `AutoTrainer` and
forwards training requests into the same C++ code path used by desktop and adb
experiments.

The Android layer deliberately does not own model-specific math, tokenizer
implementations, optimizers, or dataset policy. Those stay in the C++ core and
public asset contracts so Android remains a packaging target rather than a fork
of the framework.

## Qwen architecture dispatch and training limits

The common Qwen graph reads the checkpoint architecture rather than selecting a
model-size-specific implementation. `qwen2` and `qwen3` use dense GQA decoders;
`qwen3_5_text` uses the configured mixture of Gated DeltaNet and gated full
attention. Full `qwen3_5` snapshots supply their `text_config` and text weights.
The new conv/partial-RoPE/delta-recurrence forward and backward kernels live in
`core/qwen35_ops.*`. This path accepts text tokens only.

LoRA attachment derives matrix dimensions from the actual projections. The
registry and graph share architecture-dependent default targets; `all-linear`
also selects MLP projections. The Auto layer freezes base weights and the
trainer applies causal loss, finite checks, precise gradient clipping, Adam and
cleanup after both successful and failed steps. Exported Qwen adapters follow
PEFT's transposed layout; resume restores adapter weights/configuration with a
fresh optimizer.

The current autograd engine is process-wide. C++ applications must serialize
training operations and graph-affecting model calls. The Android bridge does
this across sessions and synchronizes Java instance lifecycle operations.
Qwen3.5 supports full-block checkpointing and frozen FP32 parameter offload
through the shared execution contract. It has no incremental cache, vision or MoE API. See [Qwen support](QWEN_DENSE_SUPPORT.md) for the precise contract.

- [System architecture](../figures/mf-system-architecture.md)
- [Qwen3.5 text architecture](../figures/mf-qwen35-text-architecture.md)

## Execution contracts and ownership

The discovery layer reports `ModelArchitectureSpec::capabilities`, backed by one
engine-aware registry. Qwen2/3/3.5 use stable architecture metadata to enumerate
modules even when weight handles have been evicted. The same inventory drives
exact LoRA injection, adapter names, checkpoint captures and base-weight leases;
the complete named weight inventory drives loading validation and offload groups.
This keeps architectural differences in the model graph rather than copying
feature-specific lists into each optimization.

Pretrained Qwen construction separates shape metadata from storage allocation.
Deferred models reject forward and adapter initialization until the complete
checkpoint validates and installs. The default constructor remains available for
explicit custom initialization. Streamed LM loss manages its output-head lease
inside the graph; dataset loops do not reach into offloaded weight ownership.

Both Qwen dataset CLIs share `QwenExecutionOptions`. Shared optimizer clipping
and `GradModeGuard` own numerical and evaluation policy. Gemma accumulation
flushes short final groups using the actual count and clears failed-step state.
Family graphs remain explicit and the native runtime remains serialized.

- [Execution architecture](../figures/mf-execution-contracts.md)
- [Model support matrix](MODEL_SUPPORT.md)
