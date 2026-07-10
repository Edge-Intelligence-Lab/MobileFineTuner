# MobileFineTuner

**A C++ Framework for Mobile-Native LLM Fine-Tuning**

## News

- **[2026-07-10]** We have open-sourced MobileFineTuner, a C++ framework for mobile-native LLM fine-tuning.
- **[2026-07-10]** [MobileRLHF](MobileRLHF) now supports reinforcement-learning-based preference post-training on smartphones in both standalone and federated settings.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-C%2B%2B%20Core%20%7C%20Android%20Native-green.svg)]()
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange.svg)]()

<img width="1508" height="916" alt="overview-git" src="https://github.com/user-attachments/assets/1043765f-5841-4317-b3f3-beaeefee2c4d" />

<img width="1992" height="1112" alt="image" src="https://github.com/user-attachments/assets/35cb5c4e-a3bb-4a21-aac2-d38d50899a17" />

https://github.com/user-attachments/assets/6ba63485-34ba-42b6-b9b1-ff980717c77b





## Paper

Our paper is available on arXiv:

**MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications**  
Jiaxiang Geng*, Lunyu Zhao*, Yiyi Lu*, and Bing Luo

[![arXiv](https://img.shields.io/badge/arXiv-2512.08211-b31b1b.svg)](https://arxiv.org/abs/2512.08211)

If you find this project useful, please consider citing our paper.

---

## Overview

MobileFineTuner is an open-source C++ framework for practical, privacy-preserving fine-tuning of Large Language Models (LLMs) on mobile-class devices. The current release candidate verifies the stable C++ core, package/export surface, local training entrypoints, and Android native synthetic smoke execution. Full-weight on-phone training requires externally supplied model weights and device-specific validation.

Unlike simulation-based or desktop-bound approaches, MobileFineTuner is built around a lean native C++ implementation that eliminates Python runtime overhead in the training path and supports both Full Fine-Tuning (Full-FT) and Parameter-Efficient Fine-Tuning (PEFT/LoRA) under tight resource constraints.

### Verified Scope

- Stable C++ operator/autograd/LoRA core with unit tests and installable CMake package.
- Five local training smoke entrypoints: GPT-2 small, GPT-2 medium, Gemma 270M, Gemma 1B-PT, and Qwen.
- Android native build/run path validated with a synthetic Qwen LoRA smoke test, including loss/backward/update and 5 ms RSS sampling.
- Full pretrained weights, benchmark datasets, adapters, run logs, and phone QA media are external runtime artifacts and are not bundled in the source release.

### Key Features

- **Efficiency**: Pure C++ implementation with modular operators, automatic differentiation, and full backpropagation鈥攏o Python runtime or external ML frameworks required
- **Scalability**: Supports multiple mainstream decoder-only LLM architectures (GPT-2, Gemma, Qwen) with reusable graph, tokenizer, dataset, and LoRA components
- **Usability**: Simple high-level APIs that abstract away system complexity, enabling rapid prototyping and practical deployment
- **Privacy-Preserving**: All training data remains on-device, complying with GDPR and user privacy expectations
- **Resource-Aware**: Keeps model weights, datasets, run logs, adapters, and profiling output outside the source package; runtime scripts resolve assets through explicit environment variables

---

## Table of Contents

- [Installation & Build](#installation--build)
- [Quick Start](#quick-start)
- [Android SDK / AAR](#android-sdk--aar)
- [Model and Dataset Assets](#model-and-dataset-assets)
- [Supported Models](#supported-models)
- [Core Components](#core-components)
- [Architecture](#architecture)
- [Evaluation](#evaluation)
- [Benchmarks](#benchmarks)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Installation & Build

### Prerequisites

- **Compiler**: C++17 or later
- **Build System**: CMake 鈮?3.10
- **Threading**: pthreads
- **BLAS** (optional): Apple Accelerate, OpenBLAS, or Intel MKL for accelerated matrix operations

### Build Instructions

```bash
cmake -S operator -B operator/build -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build operator/build -j4
ctest --test-dir operator/build --output-on-failure
cmake --install operator/build --prefix /tmp/mobilefinetuner-install
```

**Build Outputs:**
- `liboperators.a` - Core framework library
- self-contained operator tests such as `test_qkt_softmax_grad`, `test_repeat_kv_softmax_grad`, `test_lora_roundtrip`

The default `operator/` build produces the reusable library and unit tests.
Model-specific training/eval CLIs are examples, built from directories under
`examples/`, such as `examples/gpt2_small_lora_finetune/`,
`examples/gpt2_medium_lora_finetune/`, `examples/gemma_3_270m_lora_finetune/`,
`examples/gemma_3_1b_pt_lora_finetune/`, and
`examples/qwen_lora_finetune/`.

The install step exports CMake packages. New downstream projects should use the
`MobileFineTuner` package name:

```cmake
find_package(MobileFineTuner REQUIRED)
target_link_libraries(your_target PRIVATE MobileFineTuner::operators)
```

Use the stable umbrella header in new applications:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"
```

See [docs/PUBLIC_API.md](docs/PUBLIC_API.md) for the public API boundary.

The legacy package name is still available for existing consumers:
`find_package(Operators REQUIRED)` and `Operators::operators`.

### Android SDK / AAR

MobileFineTuner also provides an Android Library module for app integration:

```bash
bash scripts/android/build_mft_sdk_aar.sh
```

This produces `android-visualizer/mft-sdk/build/outputs/aar/mft-sdk-release.aar`.
The AAR packages the Java API and native JNI bridge, while model weights and
datasets remain external runtime assets. See [docs/ANDROID_SDK.md](docs/ANDROID_SDK.md)
for the Android SDK contract.

For consumer-style integration, publish the SDK to a local Maven repository:

```bash
bash scripts/android/publish_mft_sdk_local.sh
```

The standalone sample consumer app is available at `android-visualizer/sdk-sample`.

---

## Quick Start

### 1. Prepare Data and Model

MobileFineTuner does not bundle pretrained weights or benchmark datasets. This
matches the PyTorch/Transformers convention: the framework ships code, and the
application passes a local model/data path at runtime.

Use shared asset roots for reproducible local runs:

```bash
export MFT_MODEL_ROOT=/path/to/mft-models
export MFT_DATA_ROOT=/path/to/mft-data
```

Expected layout:

```text
$MFT_MODEL_ROOT/
  gpt2/
  gpt2-medium/
  gemma-3-270m/
  gemma-3-1b-pt/
  Qwen2.5-0.5B/

$MFT_DATA_ROOT/
  wikitext2/wikitext-2-raw/
  mmlu/data/
```

Each model directory is a HuggingFace-style snapshot containing `config.json`,
tokenizer files, and either `model.safetensors` or a HuggingFace sharded
SafeTensors layout with `model.safetensors.index.json`. Per-model overrides such
as `QWEN_MODEL_DIR`, `GPT2_SMALL_MODEL_DIR`, and `GEMMA_270M_MODEL_DIR` are also
supported.

To verify what this checkout will use on your machine:

```bash
bash scripts/check_local_assets.sh
```

See [docs/MODEL_ASSETS.md](docs/MODEL_ASSETS.md) for exact file requirements
and download examples.

### 2. Validate the Five Training Entrypoints

Run the repo-level smoke suite before using real assets:

```bash
bash scripts/run_training_smoke.sh
```

This performs a two-step synthetic training pass for GPT-2 small, GPT-2 medium, Gemma 270M, Gemma 1B-PT, and Qwen.

To run a one-step real-asset sanity pass across the five training entrypoints:

```bash
bash scripts/run_training_real_assets.sh
```

To verify that C++ tokenization matches HuggingFace token IDs for local
tokenizer snapshots:

```bash
python3 scripts/generate_tokenizer_hf_golden_fixtures.py
MFT_TOKENIZER_GOLDEN_JSONL=runs/tokenizer_golden/hf_tokenizer_golden.jsonl \
  ctest --test-dir operator/build --output-on-failure -R TokenizerHFGolden
```

See [docs/MODEL_ASSETS.md](docs/MODEL_ASSETS.md) for the tokenizer alignment
asset contract.

### 3. Use the Unified C++ API

Applications that do not need model-specific diagnostics can use the Auto API:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"

auto tokenizer = ops::TokenizerFactory::from_pretrained(model_dir);
auto model = ops::AutoModelForCausalLM::from_pretrained(model_dir);
model->init_lora(ops::AutoLoraConfig::attention_qkvo());

ops::AutoTrainerConfig trainer_cfg;
trainer_cfg.learning_rate = 2e-4f;
ops::AutoTrainer trainer(*model, trainer_cfg);
auto step = trainer.train_step(input_ids, attention_mask, labels);
```

The lower-level GPT-2, Gemma, and Qwen graph classes remain available for
alignment debugging and model-specific experiment runners.

### 4. Run LoRA Fine-Tuning

#### WikiText-2 LoRA
- **GPT-2 Small (124M):**
  ```bash
  (
    cd examples/gpt2_small_lora_finetune &&
    TRAIN_MODE=wt2 STEPS=200 BATCH_SIZE=4 GRAD_ACCUM_STEPS=2 ./run_train.sh
  )
  ```
- **GPT-2 Medium (355M):**
  ```bash
  (
    cd examples/gpt2_medium_lora_finetune &&
    TRAIN_MODE=wt2 STEPS=200 BATCH_SIZE=2 GRAD_ACCUM_STEPS=2 ./run_train.sh
  )
  ```
- **Gemma 270M:**
  ```bash
  (
    cd examples/gemma_3_270m_lora_finetune &&
    TRAIN_MODE=wt2 STEPS=200 BATCH_SIZE=4 GRAD_ACCUM_STEPS=1 ./run_train.sh
  )
  ```
- **Gemma 1B-PT:**
  ```bash
  (
    cd examples/gemma_3_1b_pt_lora_finetune &&
    TRAIN_MODE=wt2 STEPS=200 BATCH_SIZE=2 GRAD_ACCUM_STEPS=2 ./run_train.sh
  )
  ```
- **Qwen2.5-0.5B:**
  ```bash
  (
    cd examples/qwen_lora_finetune &&
    MAX_STEPS=200 BATCH_SIZE=1 GRAD_ACCUM_STEPS=1 ./run_wikitext.sh
  )
  ```

#### MMLU LoRA

GPT-2 and Gemma consume masked JSONL prepared by `run_prepare_data.sh`. Qwen consumes the raw CSV tree under `data/mmlu/data` directly and does not use JSONL.

- **GPT-2 Small / Medium:**
  ```bash
  (
    cd examples/gpt2_small_lora_finetune &&
    ./run_prepare_data.sh &&
    TRAIN_MODE=mmlu STEPS=200 ./run_train.sh
  )
  (
    cd examples/gpt2_medium_lora_finetune &&
    ./run_prepare_data.sh &&
    TRAIN_MODE=mmlu STEPS=200 ./run_train.sh
  )
  ```
- **Gemma 270M / 1B-PT:**
  ```bash
  (
    cd examples/gemma_3_270m_lora_finetune &&
    ./run_prepare_data.sh &&
    TRAIN_MODE=mmlu STEPS=200 ./run_train.sh
  )
  (
    cd examples/gemma_3_1b_pt_lora_finetune &&
    ./run_prepare_data.sh &&
    TRAIN_MODE=mmlu STEPS=200 ./run_train.sh
  )
  ```
- **Qwen2.5-0.5B:**
  ```bash
  (
    cd examples/qwen_lora_finetune &&
    MAX_STEPS=150 BATCH_SIZE=8 ./run_mmlu.sh
  )
  ```

#### GPT-2 Small Full Fine-Tune (WikiText-2)
```bash
export GPT2_SMALL_MODEL_DIR=$MFT_MODEL_ROOT/gpt2
export WT2_DATA_DIR=$MFT_DATA_ROOT/wikitext2/wikitext-2-raw

cmake -S examples/gpt2_small_lora_finetune -B examples/gpt2_small_lora_finetune/build
cmake --build examples/gpt2_small_lora_finetune/build --target train_full -j
examples/gpt2_small_lora_finetune/build/train_full \
  --data_dir "$WT2_DATA_DIR" \
  --pretrained_dir "$GPT2_SMALL_MODEL_DIR" \
  --output_path runs/gpt2_small_full_ft.safetensors \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --seq_len 128 \
  --lr 1e-4 --warmup_steps 100
```

## Model and Dataset Assets

The industrial package boundary is:

- `operator/` provides the reusable C++ training core.
- model app directories provide example CLIs for GPT-2, Gemma, and Qwen.
- pretrained weights, datasets, run logs, and generated adapters are runtime
  assets and are not part of the source distribution.

Resolution order for maintained scripts is:

1. explicit per-model/data environment variables, for example
   `QWEN_MODEL_DIR` or `QWEN_DATA_DIR`;
2. shared roots `MFT_MODEL_ROOT` and `MFT_DATA_ROOT`;
3. the repo-local fallback directories such as
   `examples/qwen_lora_finetune/pretrained/` and `data/wikitext2/wikitext-2-raw/`.

The fallback directories are useful for local research, but they are ignored by
Git and should not be used as a release mechanism. For a new machine or CI
worker, prefer explicit asset roots and validate them with:

```bash
bash scripts/check_local_assets.sh
```

Detailed layout and download examples are documented in
[docs/MODEL_ASSETS.md](docs/MODEL_ASSETS.md).

---

## Supported Models

### GPT-2 Family
- **GPT-2 Small**: 124M parameters, 12 layers, 768 hidden dimensions
- **GPT-2 Medium**: 355M parameters, 24 layers, 1024 hidden dimensions
- **GPT-2 Full FT (Small)**: full fine-tuning path for end-to-end reference

### Gemma Family (Google)
- **Gemma-3 270M**: Compact decoder-only transformer with Grouped Query Attention (GQA)
- **Gemma-3 1B**: Scaled version with 18 layers, 2048 hidden dimensions

### Qwen Family (Alibaba)
- **Qwen2.5-0.5B**: Lightweight decoder-only transformer with rotary embeddings and QKV sharing

### Adding New Models

MobileFineTuner follows the same high-level split used by PyTorch/Transformers:

- `ModelRegistry::inspect_pretrained(model_dir)` reads `config.json` and
  identifies the supported model family, assets, tied-embedding behavior, and
  default LoRA targets.
- `TokenizerFactory::from_pretrained(model_dir)` returns the correct
  model-specific tokenizer behind one `ops::Tokenizer` interface.
- `AutoModelForCausalLM::from_pretrained(model_dir)` constructs the supported
  GPT-2, Gemma, or Qwen graph and loads SafeTensors with correct layout defaults.
- `AutoTrainer` provides a shared one-step causal-LM LoRA training core.
- model graph classes under `finetune_ops/graph/` implement the architecture
  math and HuggingFace weight-name mapping.
- LoRA injection is defined by target module names, not by application
  directory names.

Different models should keep their own tokenizer algorithms. The framework
standardizes how applications request tokenization; it does not force GPT-2,
Gemma, and Qwen to share one vocabulary or pre-tokenizer.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the extension contract.

---

## Core Components

### Tensor & Autograd Engine

**Custom Tensor Implementation:**
- Pooled memory allocation for reduced malloc overhead
- Automatic gradient tracking with topological sort-based backward pass
- In-place operation support with copy-on-write semantics

```cpp
// Example: Forward and backward through custom ops
auto x = Tensor::randn({batch_size, seq_len, hidden_dim});
auto y = ops::linear(x, weight, bias);
auto loss = ops::mse_loss(y, target);
loss.backward();  // Automatic gradient computation
```

### Model Graphs

**GPT-2 Architecture:**
- Transformer decoder with fused QKV attention
- Causal attention masking for autoregressive generation
- Layer normalization and residual connections

**Gemma Architecture:**
- Grouped Query Attention (GQA) for reduced memory footprint
- RoPE (Rotary Position Embedding) for positional encoding
- GeGLU activation in feed-forward layers

### LoRA Injection

Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation:
- Inject trainable low-rank matrices into attention and MLP layers
- Freeze base model parameters to reduce memory and computation
- PEFT-compatible SafeTensors format for adapter persistence

```cpp
// LoRA injection targets
GPT-2:  Attn QKV + Attn Proj
Gemma:  Q/K/V/O projections + Gate/Up/Down MLP projections
```

### SafeTensors I/O

Fast and safe tensor serialization:
- Load pretrained weights from HuggingFace format
- Automatic key mapping for model compatibility
- Optional transpose for linear layer weights
- Save LoRA adapters in PEFT-compatible format

---

## Architecture

The reusable library is organized around HuggingFace-style runtime assets rather
than bundled checkpoints:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"

auto spec = ops::ModelRegistry::inspect_pretrained(model_dir);
auto tokenizer = ops::TokenizerFactory::from_pretrained(model_dir);
auto model = ops::AutoModelForCausalLM::from_pretrained(model_dir);
```

`spec.family` selects the graph implementation, `tokenizer` owns the
model-specific tokenization algorithm, and `AutoModelForCausalLM` maps external
checkpoint keys into the selected graph. `AutoTrainer` provides the shared
native C++ one-step LoRA training core. Concrete graph classes remain public for
specialized diagnostics and model-specific experiments.

Detailed design and extension rules are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Runtime Memory Controls

MobileFineTuner keeps the stable library focused on transparent C++ training
semantics. The default memory controls are explicit and easy to audit:

- model weights and datasets are loaded from runtime asset paths rather than
  bundled into the library;
- SafeTensors loading supports HuggingFace single-file and sharded checkpoints;
- the training CLIs expose batch size, sequence length, step count, learning
  rate, and gradient accumulation as runtime knobs;
- the core tensor runtime can use the step arena allocator for selected
  workloads through `-DUSE_ARENA_ALLOCATOR=ON`;
- Android helper scripts can sample RSS and process telemetry during native
  training runs.

### Gradient Accumulation

Divide large batches into micro-batches to reduce activation memory:

```bash
--batch_size 8              # Effective batch size
--grad_accum_steps 4        # Accumulate over 4 micro-batches
```

Result: Forward/backward runs on `batch_size / grad_accum_steps = 2` samples at a time, reducing peak activation memory by ~75% while maintaining gradient quality.

---

## Evaluation

### Perplexity (WikiText-2)

Measure language modeling quality:
```bash
cmake -S examples/gpt2_small_lora_finetune -B examples/gpt2_small_lora_finetune/build
cmake --build examples/gpt2_small_lora_finetune/build --target eval_ppl -j
examples/gpt2_small_lora_finetune/build/eval_ppl \
  --data_root "$MFT_DATA_ROOT/wikitext2/wikitext-2-raw" \
  --pretrained_dir "$MFT_MODEL_ROOT/gpt2" \
  --lora_path examples/gpt2_small_lora_finetune/outputs/lora_final.safetensors \
  --lora_merge 1
```

**Expected Results:**
- GPT-2 Small baseline: ~29.5 PPL
- GPT-2 Small + LoRA (1 epoch): ~26.8 PPL

### MMLU Benchmark

Multi-task language understanding:
```bash
(
  cd examples/gpt2_small_lora_finetune &&
  ./run_eval.sh
)
(
  cd examples/gpt2_medium_lora_finetune &&
  ./run_eval.sh
)
(
  cd examples/gemma_3_270m_lora_finetune &&
  ./run_eval.sh
)
(
  cd examples/gemma_3_1b_pt_lora_finetune &&
  ./run_eval.sh
)

# Direct GPT-2 eval binary invocation
examples/gpt2_small_lora_finetune/build/eval_mmlu \
  --mmlu_root "$MFT_DATA_ROOT/mmlu/data" \
  --split dev \
  --pretrained_dir "$MFT_MODEL_ROOT/gpt2" \
  --lora_path examples/gpt2_small_lora_finetune/outputs/lora_final.safetensors \
  --lora_merge 1 \
  --fewshot 0
```

---

## Benchmarks

Performance depends on device hardware, BLAS availability, model size, sequence
length, batch size, and thermal state. The repository keeps benchmark
collection scripts, not generated result files, in the release tree.

Recommended benchmark protocol:

```bash
bash scripts/run_training_smoke.sh
bash scripts/run_training_real_assets.sh
bash scripts/android/run_qwen_qnli_native_phone.sh
```

For Android runs, pair the native training command with
`scripts/android/adb_resource_monitor.sh` to collect RSS and system telemetry.
Store generated logs, adapters, plots, and spreadsheets under `runs/` or an
external artifact directory; these paths are intentionally ignored by Git.

---

## Project Structure

```
MobileFineTuner/
鈹溾攢鈹€ operator/                           # Core C++ framework
鈹?  鈹溾攢鈹€ finetune_ops/
鈹?  鈹?  鈹溾攢鈹€ core/                       # Tensor, autograd, memory manager
鈹?  鈹?  鈹溾攢鈹€ graph/                      # GPT-2, Gemma, Qwen graphs
鈹?  鈹?  鈹溾攢鈹€ nn/                         # LoRA layers
鈹?  鈹?  鈹溾攢鈹€ optim/                      # Optimizers and trainers
鈹?  鈹?  鈹斺攢鈹€ data/                       # WikiText-2, MMLU dataset loaders/tokenizers
鈹?  鈹溾攢鈹€ include/mobile_finetuner/       # Stable public umbrella header
鈹?  鈹溾攢鈹€ cmake/                          # CMake package config templates
鈹?  鈹斺攢鈹€ CMakeLists.txt
鈹溾攢鈹€ examples/                           # Runnable model-specific applications
鈹?  鈹溾攢鈹€ common/                         # Shared example-only evaluation helpers
鈹?  鈹溾攢鈹€ gpt2_small_lora_finetune/       # GPT-2 Small LoRA + full FT
鈹?  鈹溾攢鈹€ gpt2_medium_lora_finetune/      # GPT-2 Medium LoRA
鈹?  鈹溾攢鈹€ gemma_3_270m_lora_finetune/     # Gemma 270M LoRA
鈹?  鈹溾攢鈹€ gemma_3_1b_pt_lora_finetune/    # Gemma 1B-PT LoRA
鈹?  鈹斺攢鈹€ qwen_lora_finetune/             # Qwen2.5-0.5B LoRA
鈹溾攢鈹€ scripts/                            # Automation and orchestration
鈹?  鈹溾攢鈹€ android/                        # Native Android build/run helpers
鈹?  鈹溾攢鈹€ lib/                            # Shared shell helpers
鈹?  鈹溾攢鈹€ run_training_smoke.sh
鈹?  鈹斺攢鈹€ run_training_real_assets.sh
鈹斺攢鈹€ README.md
```

Local data, run-output, review, and archive directories may exist in developer
checkouts, but they are ignored and excluded from source releases.

---

## Contributing

We welcome contributions from the community! Areas of interest include:

- **New Model Architectures**: Llama, Mistral, Qwen, etc.
- **Mobile Platform Support**: iOS Metal acceleration, Android NNAPI integration
- **Optimization Techniques**: FlashAttention, quantization (INT8/INT4), model pruning
- **Federated Learning**: Distributed training protocols for privacy-preserving aggregation
- **Benchmarking**: Real-world mobile device experiments and profiling

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Code Style

- C++: Follow Google C++ Style Guide
- Python: Follow PEP 8 with Black formatter
- Documentation: Add inline comments for complex logic, update README for new features

---

## Contact

**Authors:**
- Jiaxiang Geng (Duke Kunshan University, The University of Hong Kong)
- Lunyu Zhao (Duke Kunshan University)
- Yiyi Lu (Duke Kunshan University)
- Bing Luo (Duke Kunshan University)

**Email:** {jg645, lz269, yl996, bl291}@duke.edu

---

## Acknowledgments

We thank the open-source community for foundational tools and datasets:
- HuggingFace Transformers for model implementations and pretrained weights
- Microsoft DeepSpeed for ZeRO optimizer inspiration
- WikiText-2 and MMLU benchmark creators
- Apple and Google for mobile hardware access and development tools

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 Mobile LLM Fine-Tuning Project Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

**Built with passion for privacy-preserving mobile AI**

<br>

<div align="center">
  <img src="logo.jpg" alt="Duke University & Duke Kunshan University" width="250"/>
</div>
