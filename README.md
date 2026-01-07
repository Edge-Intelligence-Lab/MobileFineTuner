# MobileFineTuner

**A Unified End-to-End Framework for Fine-Tuning LLMs on Mobile Phones**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-iOS%20%7C%20Android-green.svg)]()
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange.svg)]()

<img width="2250" height="1370" alt="framework" src="https://github.com/user-attachments/assets/b4a6768c-d644-4ec8-8d3c-10d6ada04c3e" />
---

## Overview

MobileFineTuner is an open-source framework that enables practical, privacy-preserving fine-tuning of Large Language Models (LLMs) directly on commodity mobile phones. By keeping sensitive user data on-device, MobileFineTuner addresses critical privacy concerns while unlocking vast amounts of valuable private-domain data for personalized model adaptation.

Unlike simulation-based or desktop-bound approaches, MobileFineTuner runs natively on mobile hardware through a lean C++ implementation, eliminating Python runtime overhead and enabling both Full Fine-Tuning (Full-FT) and Parameter-Efficient Fine-Tuning (PEFT/LoRA) under tight resource constraints.

### Key Features

- **Efficiency**: Pure C++ implementation with modular operators, automatic differentiation, and full backpropagation—no Python runtime or external ML frameworks required
- **Scalability**: Supports multiple mainstream LLM architectures (GPT-2, Gemma) with flexible interfaces for custom training strategies and federated learning integration
- **Usability**: Simple high-level APIs that abstract away system complexity, enabling rapid prototyping and practical deployment
- **Privacy-Preserving**: All training data remains on-device, complying with GDPR and user privacy expectations
- **Resource-Aware**: Built-in memory and energy optimizations designed specifically for mobile constraints

### System Optimizations

**Memory Optimization**
- ZeRO-inspired parameter sharding with LRU-based offloading to disk
- Optional FP16 quantization for disk-stored parameters
- Gradient accumulation for micro-batch training under tight memory budgets

**Energy Optimization**
- Energy-aware computation scheduler adapting to battery level and temperature
- Dynamic throttling to reduce power draw during sustained training

---

## Table of Contents

- [Installation & Build](#installation--build)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Core Components](#core-components)
- [Memory Optimization](#memory-optimization)
- [Energy-Aware Training](#energy-aware-training)
- [Evaluation](#evaluation)
- [PyTorch Alignment](#pytorch-alignment)
- [Benchmarks](#benchmarks)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Installation & Build

### Prerequisites

- **Compiler**: C++17 or later
- **Build System**: CMake ≥ 3.10
- **Threading**: pthreads
- **BLAS** (optional): Apple Accelerate, OpenBLAS, or Intel MKL for accelerated matrix operations

### Build Instructions

```bash
cd operators
mkdir build && cd build
cmake .. -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Build Outputs:**
- `liboperators.a` - Core framework library
- `gpt2_lora_finetune` - GPT-2 LoRA training CLI
- `train_lora_gemma` - Gemma LoRA training CLI
- `eval_ppl` - WikiText-2 perplexity evaluation
- `eval_mmlu` - MMLU benchmark evaluation

---

## Quick Start

### 1. Prepare Data and Model

Download WikiText-2 dataset and pretrained model:
```bash
# WikiText-2 raw text files
mkdir -p data/wikitext2/wikitext-2-raw
# Place wiki.train.raw, wiki.valid.raw, wiki.test.raw in the above directory

# GPT-2 pretrained weights (HuggingFace format)
# Place in gpt2_lora_finetune/pretrained/gpt2/
```

### 2. Run LoRA Fine-Tuning
#### WikiText-2 LoRA
- **GPT-2 Small (124M):**
  ```bash
  ./build/gpt2_lora_finetune \
    --data_dir data/wikitext2/wikitext-2-raw \
    --pretrained_dir gpt2_small_lora_finetune/pretrained/gpt2 \
    --lora_out runs/gpt2_small_wt2_lora.safetensors \
    --epochs 1 --batch_size 4 --grad_accum_steps 2 --seq_len 128 \
    --rank 8 --alpha 16 --lr 2e-4 --warmup_steps 100 \
    --eval_interval 200 --clip_grad_norm 1.0
  ```
- **GPT-2 Medium (355M):**
  ```bash
  ./build/gpt2_lora_finetune \
    --data_dir data/wikitext2/wikitext-2-raw \
    --pretrained_dir gpt2_medium_lora_finetune/pretrained/gpt2-medium \
    --lora_out runs/gpt2_medium_wt2_lora.safetensors \
    --epochs 1 --batch_size 2 --grad_accum_steps 2 --seq_len 128 \
    --rank 8 --alpha 16 --lr 2e-4 --warmup_steps 100
  ```
- **Gemma 270M:**
  ```bash
  ./build/train_lora_gemma \
    --model_dir gemma_3_270m_lora_finetune/pretrained \
    --data_dir data/wikitext2/wikitext-2-raw \
    --output_dir runs/gemma_270m_wt2_lora \
    --epochs 1 --batch 4 --grad_accum 1 --seq_len 256 \
    --learning_rate 2e-4 --warmup_ratio 0.03 \
    --lora_r 8 --lora_alpha 32 --targets full
  ```
- **Gemma 1B-PT:**
  ```bash
  ./build/train_lora_gemma \
    --model_dir gemma_3_1b_pt_lora_finetune/pretrained \
    --data_dir data/wikitext2/wikitext-2-raw \
    --output_dir runs/gemma_1b_wt2_lora \
    --epochs 1 --batch 2 --grad_accum 2 --seq_len 256 \
    --learning_rate 2e-4 --warmup_ratio 0.03 \
    --lora_r 8 --lora_alpha 32 --targets full
  ```
- **Qwen2.5-0.5B:**
  ```bash
  ./build/qwen_lora_finetune \
    --data_dir data/wikitext2/wikitext-2-raw \
    --pretrained_dir qwen_lora_finetune/pretrained \
    --lora_out runs/qwen_wt2_lora.safetensors \
    --seq_len 1024 --batch_size 1 --grad_accum_steps 1 \
    --rank 8 --alpha 16 --lr 2e-4
  ```

#### MMLU LoRA (masked JSONL)
Data prep: in each model directory run `run_prepare_data.sh` (requires `transformers`) to generate `runs/mmlu_jsonl_*/{train,valid}.jsonl` with fields `ids` and `mask`.

- **GPT-2 Small / Medium:**
  ```bash
  ./build/gpt2_lora_finetune \
    --jsonl_train runs/mmlu_jsonl_gpt2_s128/train.jsonl \
    --jsonl_valid runs/mmlu_jsonl_gpt2_s128/valid.jsonl \
    --pretrained_dir gpt2_small_lora_finetune/pretrained/gpt2 \   # or medium directory
    --lora_out runs/gpt2_mmlu_lora.safetensors \
    --seq_len 128 --batch_size 8 --grad_accum_steps 1 \
    --rank 8 --alpha 16 --lr 2e-4
  ```
- **Gemma 270M / 1B-PT:**
  ```bash
  ./build/train_lora_gemma \
    --model_dir gemma_3_1b_pt_lora_finetune/pretrained \   # or 270m directory
    --jsonl_train runs/mmlu_jsonl_gemma1b_s128/train.jsonl \
    --jsonl_valid runs/mmlu_jsonl_gemma1b_s128/valid.jsonl \
    --output_dir runs/gemma_mmlu_lora \
    --seq_len 128 --batch 8 --grad_accum 1 \
    --learning_rate 2e-4 --warmup_ratio 0.03 \
    --lora_r 8 --lora_alpha 32
  ```
- **Qwen2.5-0.5B:**
  ```bash
  ./build/qwen_lora_finetune \
    --jsonl_train runs/mmlu_jsonl_qwen/train.jsonl \
    --jsonl_valid runs/mmlu_jsonl_qwen/valid.jsonl \
    --pretrained_dir qwen_lora_finetune/pretrained \
    --lora_out runs/qwen_mmlu_lora.safetensors \
    --seq_len 128 --batch_size 8 --grad_accum_steps 1 \
    --rank 8 --alpha 16 --lr 2e-4 --qv_only
  ```

#### GPT-2 Small Full Fine-Tune (WikiText-2)
```bash
./build/gpt2_full_finetune \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_small_lora_finetune/pretrained/gpt2 \
  --output_path runs/gpt2_small_full_ft \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --seq_len 128 \
  --lr 1e-4 --warmup_steps 100
```

### 3. Enable Memory Optimization

Add parameter sharding to reduce peak memory usage:
```bash
./build/gpt2_lora_finetune \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --lora_out runs/gpt2_lora_shard.safetensors \
  --shard_enable \
  --shard_dir /tmp/gft_param_shard \
  --shard_budget_mb 512 \
  --shard_fp16_disk 1 \
  --epochs 1 --batch_size 4 --seq_len 128
```

### 4. Enable Energy-Aware Scheduling

Adapt computation to battery and temperature constraints:
```bash
./build/gpt2_lora_finetune \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --lora_out runs/gpt2_lora_energy.safetensors \
  --pm_interval 10 \
  --pm_batt_thresh 20 --pm_fb_high 2.0 --pm_fb_low 0.5 \
  --pm_temp_thresh 42 --pm_ft_high 2.0 --pm_ft_low 0.5 \
  --epochs 1 --batch_size 4 --seq_len 128
```

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

MobileFineTuner's modular architecture supports easy extension to new models. Key interfaces:
- `Tensor` class for multi-dimensional arrays
- `ops::` namespace for differentiable operations
- Model graph definition (see `finetune_ops/graph/gpt2_model.cpp` and `gemma_model.cpp`)

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

## Memory Optimization

### Parameter Sharding

Inspired by ZeRO (Zero Redundancy Optimizer), MobileFineTuner implements parameter offloading to overcome mobile memory constraints:

**How It Works:**
1. All model parameters are registered with the `ParameterSharder`
2. A resident memory budget (e.g., 512 MB) is enforced
3. Parameters are loaded on-demand via `require()` calls during forward/backward
4. LRU eviction policy offloads inactive parameters to disk
5. Optional FP16 quantization reduces disk storage by 50%

**Configuration:**
```bash
--shard_enable              # Enable parameter sharding
--shard_dir /tmp/shard      # Disk offload directory
--shard_budget_mb 512       # Resident RAM budget (MB)
--shard_fp16_disk 1         # Enable FP16 disk quantization
```

**Memory Savings:**
- GPT-2 Small: ~60% reduction (2.4 GB → 1.0 GB)
- GPT-2 Medium: ~55% reduction (3.8 GB → 1.7 GB)
- Gemma 270M: ~40% reduction (4.2 GB → 2.5 GB)

**Trade-offs:**
- Adds disk I/O overhead (~5-10% runtime increase)
- Most effective when parameter memory dominates activation memory
- Minimal overhead with SSD storage

### Gradient Accumulation

Divide large batches into micro-batches to reduce activation memory:

```bash
--batch_size 8              # Effective batch size
--grad_accum_steps 4        # Accumulate over 4 micro-batches
```

Result: Forward/backward runs on `batch_size / grad_accum_steps = 2` samples at a time, reducing peak activation memory by ~75% while maintaining gradient quality.

---

## Energy-Aware Training

MobileFineTuner includes a power monitor (`opt_ops/energy/power_monitor`) that adapts computation intensity to battery and temperature constraints, extending battery life during sustained training. The monitor uses frequency-based throttling (Hz) which internally converts to sleep duration (ms) between training steps.

### Configuration

**Battery-Based Throttling:**
```bash
--pm_interval 10            # Check battery every 10 steps
--pm_batt_thresh 20.0       # Throttle below 20% battery (default threshold)
--pm_fb_high 2.0            # Frequency high: 2 Hz (500ms sleep) when battery < threshold
--pm_fb_low 0.5             # Frequency low: 0.5 Hz (2000ms sleep) when battery ≥ threshold
```

**Temperature-Based Throttling:**
```bash
--pm_temp_thresh 42.0       # Throttle above 42°C (default threshold)
--pm_ft_high 2.0            # Frequency high: 2 Hz when temp > threshold
--pm_ft_low 0.5             # Frequency low: 0.5 Hz when temp ≤ threshold
```

**Manual Schedule Override:**
```bash
--pm_schedule "0-99:300,100-199:200,200-:100"
# Steps 0-99: 300ms sleep per step
# Steps 100-199: 200ms sleep per step
# Steps 200+: 100ms sleep per step
```

### Impact on Training

- **Energy Savings**: Adjustable throttling based on real-time battery/temperature telemetry
- **Thermal Management**: Prevents device overheating during extended training sessions
- **Flexible Control**: Supports manual telemetry simulation and deterministic schedule override
- **Minimal Accuracy Loss**: Training time increases with sleep duration, but final model quality is preserved

**Note:** Frequency parameters (`pm_fb_high`, `pm_ft_high`) are specified in Hz and internally converted to sleep milliseconds. For example, 2.0 Hz = 500ms sleep, 0.5 Hz = 2000ms sleep.

---

## Evaluation

### Perplexity (WikiText-2)

Measure language modeling quality:
```bash
./build/eval_ppl \
  --data_root data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --lora_path runs/gpt2_lora.safetensors \
  --lora_merge 1
```

**Expected Results:**
- GPT-2 Small baseline: ~29.5 PPL
- GPT-2 Small + LoRA (1 epoch): ~26.8 PPL

### MMLU Benchmark

Multi-task language understanding:
```bash
./build/eval_mmlu \
  --mmlu_root data/mmlu/data \
  --split dev \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --lora_path runs/gpt2_lora.safetensors \
  --lora_merge 1 \
  --fewshot 0
```

---

## PyTorch Alignment

For numerical validation and debugging, MobileFineTuner includes PyTorch reference implementations in `pytorch_alignment/`:

```bash
# GPT-2 LoRA (PyTorch baseline)
python pytorch_alignment/gpt2_lora_finetune.py \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --lora_out pytorch_runs/gpt2_lora \
  --epochs 1 --batch_size 8 --seq_len 128

# Gemma LoRA (PyTorch baseline)
python pytorch_alignment/gemma_lora_finetune.py \
  --model_dir gemma-3-270m \
  --data_dir data/wikitext2/wikitext-2-raw \
  --output_dir pytorch_runs/gemma_lora \
  --epochs 1 --batch 4 --seq_len 256
```

**Use Cases:**
- Verify loss curves match C++ implementation
- Compare final adapter weights for numerical parity
- Debug gradient flow and optimizer behavior

---

## Benchmarks

Performance benchmarks on commodity mobile devices:

### Memory Usage (Peak RSS)

| Model | Baseline | + Sharding (512MB) | Reduction |
|-------|----------|-------------------|-----------|
| GPT-2 Small | 2.4 GB | 1.0 GB | 58% |
| GPT-2 Medium | 3.8 GB | 1.7 GB | 55% |
| Gemma 270M | 4.2 GB | 2.5 GB | 40% |
| Gemma 1B | 8.5 GB | 4.8 GB | 44% |

### Training Speed

Training speed depends on device hardware, BLAS acceleration, and model size. Representative examples:

| Model | Configuration | Approximate Time/Epoch |
|-------|--------------|------------------------|
| GPT-2 Small | batch=4, seq_len=128, BLAS enabled | 4-6 hours on modern mobile SoC |
| GPT-2 Medium | batch=4, seq_len=128, BLAS enabled | 10-14 hours on modern mobile SoC |
| Gemma 270M | batch=4, seq_len=256, BLAS enabled | 8-12 hours on modern mobile SoC |

*Times vary significantly based on device specs, thermal throttling, and memory optimization settings*

### Energy Efficiency

| Configuration | Training Time | Energy Impact |
|---------------|--------------|---------------|
| No throttling | Baseline | High power draw, thermal throttling likely |
| Energy-aware throttling | 1.5-2× baseline | Reduced power, extended battery life |

*Note: Actual power savings depend on device hardware, battery level, and throttling aggressiveness*

---

## Project Structure

```
MobileFineTuner/
├── operators/                          # Core C++ framework
│   ├── finetune_ops/
│   │   ├── core/                       # Tensor, autograd, memory manager
│   │   ├── graph/                      # GPT-2, Gemma, Qwen graphs
│   │   ├── nn/                         # Layers (Linear, LoRA, embeddings, norms)
│   │   ├── optim/                      # Optimizers and trainers
│   │   └── data/                       # WikiText-2, MMLU dataset loaders/tokenizers
│   ├── opt_ops/
│   │   ├── energy/                     # Power monitor
│   │   └── sharding/                   # Parameter sharder
│   ├── MEMORY_OPTIMIZATION_GUIDE.md
│   └── CMakeLists.txt
├── gpt2_small_lora_finetune/           # GPT-2 Small LoRA + full FT (WikiText-2, MMLU)
│   ├── src/{main.cpp, main_full.cpp, eval_ppl.cpp, eval_mmlu.cpp}
│   ├── run_{train,eval,prepare_data}.sh
│   └── CMakeLists.txt
├── gpt2_medium_lora_finetune/          # GPT-2 Medium LoRA (WikiText-2, MMLU)
│   ├── src/{main.cpp, eval_ppl.cpp, eval_mmlu.cpp}
│   ├── run_{train,eval,prepare_data}.sh
│   └── CMakeLists.txt
├── gemma_3_270m_lora_finetune/         # Gemma 270M LoRA (WikiText-2, MMLU)
│   ├── src/{main.cpp, eval_mmlu_gemma.cpp}
│   ├── run_{train,eval,prepare_data}.sh
│   └── CMakeLists.txt
├── gemma_3_1b_pt_lora_finetune/        # Gemma 1B-PT LoRA (WikiText-2, MMLU)
│   ├── src/{main.cpp, eval_mmlu_gemma.cpp}
│   ├── run_{train,eval,prepare_data}.sh
│   └── CMakeLists.txt
├── qwen_lora_finetune/                 # Qwen2.5-0.5B LoRA (WikiText-2, MMLU)
│   ├── src/{wikitext_main.cpp, mmlu_main.cpp}
│   ├── run_{wikitext,mmlu}.sh
│   └── CMakeLists.txt
├── pytorch_alignment/                  # PyTorch reference (numerical parity)
│   ├── gpt2_{lora,full}_finetune.py
│   ├── gemma_lora_finetune.py
│   ├── qwen_lora_finetune.py
│   └── README.md
├── scripts/                            # Automation and orchestration
│   ├── benchmark/                      # Sharding/energy benchmarks
│   └── Finetune/                       # Batch runners (C++ + PyTorch, incl. Qwen)
├── data/                               # Expected data root
│   ├── wikitext2/                      # raw and pretokenized
│   └── mmlu/                           # CSV and JSONL
├── runs/                               # Training outputs (LoRA, JSONL)
└── README.md
```


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
