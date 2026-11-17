# Qwen text models

The native Qwen decoder supports **Qwen2/Qwen3 dense text and Qwen3.5 hybrid text architectures**
using the same public Auto API. Model dimensions and LoRA adapters are configured
from the checkpoint and actual model layers, rather than a model-size-specific
training implementation. Full-weight Mac checks cover Qwen2.5-0.5B and
Qwen3-0.6B and Qwen3.5-0.8B; support for other sizes is architectural, not a claim that every
checkpoint or device memory requirement has been tested.

## Architecture contract

`config.json` must declare `model_type` (`qwen2`, `qwen3`, or `qwen3_5_text`) and the architecture
dimensions: `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`,
`num_attention_heads`, and `num_key_value_heads`.

- A full `qwen3_5` snapshot is accepted by extracting its `text_config` and loading
  `model.language_model.*` tensors. Text-only snapshots with `model.*` keys are
  also supported; ambiguous backbones are rejected. Vision weights are unused.
- Qwen2 derives the attention head dimension unless `head_dim` is explicit.
  Qwen3 supports independent `head_dim` and applies Q/K RMS normalization.
- Q and KV projection widths are `heads * head_dim` and
  `kv_heads * head_dim`. The output projection maps the full Q width back to
  `hidden_size`; it need not be square.
- Qwen2 has Q/K/V biases. Qwen3 follows `attention_bias`, including the output
  projection bias when enabled.
- `tie_word_embeddings` selects a shared embedding/output matrix or a separately
  loaded `lm_head.weight`. Dense and streaming language-model losses use the
  correct output matrix.
- Every required base tensor must exist and have the expected shape. Qwen3
  normalization weights and untied output heads are required when applicable.
  A failed checkpoint validation does not install a partial set of weights.
- Norms/biases stay FP32. The default loader promotes base matrices to FP32;
  the example CLI's `auto`/`native` policy can retain F16/BF16 frozen matrices.

Qwen3.5 follows `layer_types`: linear layers use causal depthwise convolution,
L2-normalized Q/K, recurrent Gated DeltaNet, and gated RMS normalization; full
layers use doubled/gated Q projections, zero-centered Q/K RMS normalization,
partial default RoPE, GQA, and output gating. Decoder norms are zero-centered.
Both mixers have complete reverse-mode derivatives. `attention_bias=false` and
`attn_output_gate=true` are required for Qwen3.5.

This is full-sequence **pure-text LoRA training**. Sliding-window layers,
non-default RoPE scaling, nonzero attention dropout, MoE, and vision/image/video
inputs are unsupported. Qwen3.5 checkpointing and parameter offload requests are
explicitly rejected; ordinary/streamed full-attention kernels both work. There
is no incremental KV/conv/recurrent-state cache API. The reference recurrence
retains `B * H_v * (S+1) * K * V` FP32 state values per linear layer for backward;
long-sequence memory/performance has not been qualified on phones.

## Model-independent LoRA injection within this graph

Use exact module suffixes:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Qwen3.5 adds `in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj` in
`linear_attn` layers. An empty target list selects all mixer projections present
in the architecture (Q/K/V/O in dense models). `{"all-linear"}` selects every
available mixer and MLP linear module. Targets apply only where that module
exists; e.g. `q_proj` selects full-attention layers and `in_proj_qkv` selects linear
layers. `all-linear` is recommended to train both Qwen3.5 mixer types.
Unknown and duplicate targets are rejected. Each adapter derives its dimensions
from the selected layer's actual `[in, out]` matrix. Only the requested modules
receive adapters; choosing `k_proj` alone does not implicitly select Q/V/O.

```cpp
#include "mobile_finetuner/mobile_finetuner.h"

auto model = ops::AutoModelForCausalLM::from_pretrained(model_dir);
auto lora = ops::AutoLoraConfig::attention_qkvo();
lora.rank = 8;
lora.alpha = 16;
lora.dropout = 0;
lora.target_modules = {"all-linear"};
model->init_lora(lora);

ops::AutoTrainer trainer(*model);
auto step = trainer.train_step(input_ids, attention_mask, labels);
model->qwen()->save_lora_adapter(adapter_dir);
```

Base weights are frozen by Qwen Auto injection. A direct `QwenModel` caller
should call `freeze_base()` after injecting or loading adapters. LoRA dropout is
applied only on the adapter branch, uses a seeded RNG that advances between
calls, and is disabled by `set_training(false)` or a disabled autograd engine.
Checkpointed training requires zero LoRA dropout so recomputation is deterministic.

## Adapter files and interoperability

`save_lora_adapter(directory)` writes:

```text
adapter_config.json
adapter_model.safetensors
```

Tensor names follow PEFT, for example:

```text
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
```

The file uses PEFT's `[r, in]` / `[out, r]` layout; the native in-memory matrices
use `[in, r]` / `[r, out]`. Export/import transposes explicitly. Both formats are
checked for finite adapter values. `named_lora_parameters()` uses the PEFT names
but returns tensors in the **native** layout.

Load a standard PEFT adapter into a compatible pretrained base with:

```cpp
auto model = ops::AutoModelForCausalLM::from_pretrained(model_dir);
model->qwen()->load_lora_adapter(adapter_dir);
model->qwen()->freeze_base();
```

The loader validates config, keys, rank, shapes, and target coverage before
replacing the active adapter. PEFT's expanded `model.layers.N.*` target lists
are accepted when they cover every compatible layer for each selected suffix.
Qwen3.5 exports use the `Qwen3_5ForCausalLM` text-only PEFT namespace; adapter
names for a multimodal wrapper are not silently rewritten. Arbitrary target
regexes, per-layer ranks/scales/selection, trainable biases, saved extra modules,
and LoRA variants such as DoRA/rsLoRA are not implemented and are rejected.

This restores **adapter weights and configuration**, not optimizer moments,
training-step counters, or the dropout RNG state. Continuing training creates a
new optimizer unless the application separately implements optimizer persistence.
Adapters must be used with their intended base checkpoint.

## Example CLI

```bash
cmake -S examples/qwen_lora_finetune -B examples/qwen_lora_finetune/build \
  -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build examples/qwen_lora_finetune/build -j4

examples/qwen_lora_finetune/build/train_wikitext \
  --model_dir "$QWEN_MODEL_DIR" --data_dir "$WT2_DATA_DIR" \
  --seq_len 32 --batch_size 1 --max_steps 3 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0 \
  --lora_targets q_proj,k_proj,v_proj,o_proj --output_dir runs/qwen_adapter

examples/qwen_lora_finetune/build/train_wikitext \
  --model_dir "$QWEN_MODEL_DIR" --data_dir "$WT2_DATA_DIR" \
  --seq_len 32 --batch_size 1 --max_steps 3 \
  --resume_from runs/qwen_adapter --output_dir runs/qwen_continued
```

For native Mac stability runs, the WikiText executable also accepts
`--stability_checks` (checks loss, all adapter gradients and parameters each step;
writes `stability_metrics.csv`) and `--save_every N` (periodic adapter snapshots).
Snapshots contain adapter weights/configuration; optimizer state is not saved.

MMLU labels supervise answer tokens with the causal shift applied only by the
loss. Set `--seq_len` large enough to retain answers; batches with no retained
answer tokens fail explicitly.

Both Qwen training CLIs accept `--lora_targets` and `--resume_from`. They retain
the legacy binary adapter export and also write the standard PEFT files. Unknown
CLI flags are errors. The shell wrappers retain their existing default arguments;
pass the new flags directly to the compiled executable.

## Validation

Independent reference generation lives in
`scripts/generate_qwen_hf_fixtures.py`. It requires a host Python environment with
PyTorch, Transformers, PEFT and SafeTensors; **the C++ runtime does not require
Python or any of those libraries**.

```bash
python3 -m pip install -r scripts/requirements-qwen-validation.txt
python3 scripts/generate_qwen_hf_fixtures.py --output runs/qwen_hf_fixtures
MFT_QWEN_HF_FIXTURES="$PWD/runs/qwen_hf_fixtures" \
  ctest --test-dir operator/build --output-on-failure -R 'QwenArchitecture|QwenHFParity'
python3 scripts/generate_qwen_hf_fixtures.py \
  --output runs/qwen_hf_fixtures --verify-native
```

Six tiny reference models cover Qwen2/Qwen3 (including native BF16 storage), shared/independent output heads,
GQA with non-square projections, attention biases, single/exact target selection,
and all-linear LoRA. Every logits coordinate, adapter gradient, and one Adam
update is compared with Transformers/PEFT in both ordinary and streamed attention
modes. Nonzero initial A and B exercise both derivatives. Native adapter reload
and PEFT-to-native/native-to-PEFT interoperability are checked independently.
Loss uses the same causal shift on both sides: `logits[:, :-1]` vs `labels[:, 1:]`.

The HF parity test is explicitly marked skipped (return code 77) when references
are not supplied; a plain CTest run is not evidence of HF alignment. The
hermetic architecture test still runs without external assets.

Additional tests cover Qwen3 attention/checkpointing/offload combinations,
malformed/missing weights, failed adapter replacement, invalid dimensions and
targets, and dropout mode/seed behavior. JSON parsing and a vendored MIT-licensed
`nlohmann/json` header are internal implementation details; installed clients do
not require its header.

See [the Mac validation record](validation/QWEN_20261115.md) for measured results,
runtime versions and the remaining physical-device checks.
See [Qwen3.5 validation](validation/QWEN35_20261115.md) for Transformers 5.3.0
references, official-weight gradients, and the pure-text training results.

The reference CI matrix generates dense and hybrid fixtures in separate pinned
Transformers environments and checks both native exports with PEFT. Hybrid
fixtures include BF16 storage and both supported base tensor namespaces. See
[PR integration checks](validation/PR_INTEGRATION_20261117.md) and
[Mac stability runs](MAC_STABILITY.md) for final-candidate qualification.
