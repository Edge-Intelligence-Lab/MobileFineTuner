# Opt-in memory optimizations

The public Qwen and Gemma C++ models expose three independent switches. All are
off by default. These are CPU FP32 training paths, not CUDA FlashAttention,
distributed ZeRO, or optimizer-state sharding. GPT-2 gradient fixes are outside
this change; its existing smoke tests do not establish full gradient coverage.

| Feature | Entry point | What is saved / recomputed |
| --- | --- | --- |
| Activation checkpointing | `set_activation_checkpointing(true)` | Save block inputs and boundary dependencies; replay the deterministic block during backward. No activation files are written. |
| Frozen parameter disk offload | `enable_parameter_sharding(ShardConfig)` | Persist frozen backbone weights, reload whole blocks on demand, release managed owner handles. LoRA and optimizer state stay in RAM. |
| Streaming attention | `set_memory_efficient_attention(true)` | Recompute softmax rows in backward; no full `[B,H,S,S]` scores/probabilities are retained. |

## Build and validate

```bash
cmake -S operator -B build -DCMAKE_BUILD_TYPE=Release \
  -DUSE_NEW_AUTOGRAD_ENGINE=ON -DUSE_BLAS=ON -DBUILD_TESTS=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure \
  -R 'StreamingAttentionParity|activation_checkpoint|parameter_offload|model_memory_optimizations'
```

The tests check attention against the standard path and finite differences;
checkpoint gradients with shared/non-leaf dependencies and exception recovery;
disk round trips and failed/truncated I/O; and every LoRA gradient and parameter
on tiny Qwen/Gemma models for all eight combinations. Gradients are compared at
identical parameter snapshots (elementwise `atol=2e-5, rtol=1e-4`); a separate
three-update free-running trajectory checks loss and adapter differences below
`2e-5`. The attention primitive also has stricter direct and finite-difference
checks. These small-model tolerances are not a guarantee for every workload.
These are correctness tests, not task accuracy or performance benchmarks.
See [validation and numerical limitations](MEMORY_OPTIMIZATION_VALIDATION.md).

## Model API

Load all weights first, initialize LoRA, and freeze the backbone before offload.
For Qwen, call `model.freeze_base()`. For Gemma, loaded base tensors must have
`requires_grad=false`; LoRA tensors returned by the injector remain trainable.

```cpp
model.set_activation_checkpointing(true);
model.set_memory_efficient_attention(true);
ops::sharding::ShardConfig config;
config.offload_dir = "run/offload";
config.max_resident_bytes = size_t(1536) * 1024 * 1024;
model.enable_parameter_sharding(config);

auto trainable = model.get_lora_parameters();
auto logits = model.forward(input_ids, attention_mask);
auto loss = ops::lm_cross_entropy(logits, labels);
loss->backward();
// Apply the optimizer to trainable only, then clear the autograd graph as usual.
auto io = model.sharding_stats();
```

The model must outlive backward. Do not mutate weights, adapters, inputs, memory
switches, or model block handles between forward and backward. Once offload is
enabled, loading/replacing base weights, reinitializing adapters, enumerating
base `parameters()`, or merging adapters is unsupported. Save adapters without
merging. Recreate the model to disable offload. Release loader maps/external
base-weight aliases before training, otherwise they pin storage in RAM.

The budget must hold the largest **complete block or embedding/head group**.
Gemma has separate embedding and transposed head storage in the public tensor
implementation. The budget is logical managed residency, **not a peak RSS cap**:
saved backward tensors, aliases, activations, allocator caching, and model
loading can exceed it. Offload alone may not lower lifecycle peak RSS; combine
it with checkpointing when activation retention is the bottleneck and measure
the actual workload. This release makes no speedup or memory-saving percentage
claim.

`bytes_read`/`bytes_written` count successful tensor-payload bytes, not physical
disk traffic (OS caching applies). `loads`/`evictions` count storage operations.
Each sharder uses a unique directory; files remain after it is destroyed. Remove
only that run's offload directory after training/backward is finished. Do not
reuse files as portable model checkpoints. FP16-on-disk is rejected to preserve
the FP32 values. No swapping or privileged OS configuration is performed.

## Existing training CLIs

The following flags are available in the Qwen **`train_mmlu`** executable and
both Gemma `train` executables. They are not added to Qwen `train_wikitext` or
`AutoTrainer` in this change.

```text
--activation_checkpointing
--memory_efficient_attention
--offload_dir PATH
--offload_resident_mb 1536
```

Omit a switch to keep its baseline path. For Qwen offload also specify
`--base_weight_storage fp32`. Use LoRA dropout zero. Gemma alignment/debug and
synthetic-smoke modes reject these switches rather than silently ignoring them.

For existing model and dataset assets prepared by the respective example README:

```bash
cmake -S examples/qwen_lora_finetune -B build-qwen -DCMAKE_BUILD_TYPE=Release -DUSE_BLAS=ON
cmake --build build-qwen --target train_mmlu -j 4
./build-qwen/train_mmlu --model_dir "$QWEN_MODEL_DIR" --data_dir "$MMLU_DATA_DIR" \
  --seq_len 128 --batch_size 1 --max_steps 10 --lora_dropout 0 \
  --base_weight_storage fp32 --activation_checkpointing \
  --memory_efficient_attention --offload_dir run/qwen-offload --offload_resident_mb 1536

cmake -S examples/gemma_3_270m_lora_finetune -B build-gemma -DCMAKE_BUILD_TYPE=Release -DUSE_BLAS=ON
cmake --build build-gemma --target train -j 4
mkdir -p run/gemma-output
./build-gemma/train --model_dir "$GEMMA_MODEL_DIR" --data_dir "$WIKITEXT_DATA_DIR" \
  --seq_len 128 --batch 1 --max_steps 10 --lora_dropout 0 \
  --activation_checkpointing --memory_efficient_attention \
  --offload_dir run/gemma-offload --offload_resident_mb 1536 --output_dir run/gemma-output
```

The Gemma 1B CLI uses the same flags; build
`examples/gemma_3_1b_pt_lora_finetune` instead. Device/model-size feasibility is
not implied by API support. Start with a short run, check memory and disk space,
and validate the intended workload before a long experiment.

After building all three executables, check new CLI option validation without
loading assets:

```bash
python3 examples/tests/test_memory_options.py \
  build-qwen/train_mmlu build-gemma/train build-gemma1b/train
```

## Lower-level contracts

`ops::checkpoint(fn, input, captures)` requires the new autograd engine and
CPU FP32 tensors. Pass block trainables explicitly. Captured differentiable
non-leaf dependencies are supported. `fn` must be deterministic and use tracked
operators; RNG/dropout state is not replayed. First-order derivatives only.
The singleton engine is not safe for simultaneous training calls.

The streaming kernel takes equal positive `[B,H,S,D]` Q/K/V shapes and constant
additive FP32 masks: `[S,S]` plus optional `[B,S]` or `[B,1,1,S]` padding bias.
Models expand GQA heads before this call. No cross-attention, attention dropout,
softcapping, or mask gradients are implemented. Explicit masks override the
automatic causal mask consistently in forward and backward. All-negative-
infinity rows return zero context/gradient. Finite negative biases retain their
ordinary softmax semantics. The public tensor API materializes transpose/slice
operations as dense tensors; there is no strided-view API in this base version.

The old five-argument attention signature remains available. With the new
engine it uses the streaming backward; the legacy engine deliberately retains
the standard attention training fallback. Qwen/Gemma still create quadratic
`[S,S]` masks, so O(S) scratch in the kernel is not O(S) total model memory.
