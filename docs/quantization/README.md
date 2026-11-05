# Q8_0 Quantization for LoRA Fine-Tuning

An opt-in path that stores the **frozen base weights** of `LoRALinear` as 8-bit
blocks, so on-device fine-tuning keeps the base in int8 while the LoRA adapters
stay in fp32.

Nothing changes unless you pass `--quantize` or `--load_quantized`.

- [Results](RESULTS.md) — accuracy, size, speed, memory
- [Format](#checkpoint-format) — what a `.q8` file contains

---

## Quick start

### 1. Train with quantization

```bash
./train --pretrained_dir /path/to/gpt2_small \
        --data_dir       /path/to/wikitext-2-raw \
        --steps 200 --seq_len 128 --rank 4 --alpha 8 --lr 2e-4 \
        --quantize q8_0 \
        --lora_out adapter.safetensors
```

The base weights are quantized once at startup (48 layers, ~1.3 s), then forward
and backward read the int8 blocks. LoRA A/B and all ``bias``/LayerNorm tensors
stay fp32.

### 2. Export a quantized checkpoint

```bash
./train ... --quantize q8_0 --save_quantized gpt2-small-q8.q8
```

This writes a **self-contained** file: the 48 linear weights as int8 blocks, and
the embeddings / LayerNorm / biases as fp32. For GPT-2 small it is 236.8 MiB,
versus 522.7 MiB for the original `model.safetensors`.

### 3. Load it directly

```bash
./train --pretrained_dir /path/to/gpt2_small \
        --data_dir       /path/to/wikitext-2-raw \
        --load_quantized gpt2-small-q8.q8 \
        --steps 200 --rank 4 --alpha 8 --lr 2e-4
```

`--pretrained_dir` is still needed for `config.json` (and the tokenizer lives
there too), but the `.safetensors` weights are **not read**: the fp32 base
tensors are never created, and the per-layer transpose + quantization work is
skipped entirely.

---

## What is and is not quantized

| Component | Storage | Notes |
|---|---|---|
| 48 linear weights (`c_attn`, `c_proj`, `c_fc`, `mlp.c_proj`) | **Q8_0 int8** | 8.5 bit per weight |
| `wte` / `wpe` embeddings | fp32 | `wte` is 31% of the parameters; not quantized yet |
| LayerNorm weights/biases | fp32 | small |
| All linear biases | fp32 | small |
| LoRA `A` / `B` | fp32 | trainable |
| Gradients | fp32 | backward runs in fp32 against the dequantized blocks |

## Model input and output

### How the weights are read in

Both entry points live in `examples/gpt2_small_lora_finetune/src/main.cpp`.

**fp32 checkpoint (default)**

```cpp
GPT2Config cfg = GPT2Config::from_pretrained(dir + "/config.json");
SafeTensorsModelReader reader(dir);
auto loaded = reader.load_tensors_mapped(
    GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer), opts);
for (auto& kv : loaded) model.assign_weight(kv.first, kv.second);
model.init_lora_modules();          // wraps the 4 linear weights per block

// only with --quantize q8_0
for (int li = 0; li < cfg.n_layer; ++li) {
    auto& blk = model.get_block(li);
    for (LoRALinear* lin : {blk.qkv_lin.get(), blk.proj_lin.get(),
                            blk.fc_in_lin.get(), blk.fc_out_lin.get()}) {
        if (lin && lin->W()) lin->quantize_base_weight_q8_0(/*allow_skip=*/true);
    }
}
```

**Q8_0 checkpoint (`--load_quantized PATH`)**

```cpp
std::vector<ops::QuantTensor> tensors;
ops::read_quant_model(path, tensors, &err);
for (auto& t : tensors) {
    if (t.is_q8) { q8_pending.push_back(t); continue; }   // deferred
    model.assign_weight(t.name, make_tensor(t));          // fp32 entries
}
model.init_lora_modules();     // LoRALinear holds a null base weight for now

// Names look like "blocks.7.attn.qkv.weight"; the loop parses the block index
// and the sub-key, picks the matching LoRALinear, and installs the blocks:
for (auto& t : q8_pending) {
    LoRALinear* lin = /* blk.qkv_lin / proj_lin / fc_in_lin / fc_out_lin */;
    lin->adopt_quantized_weight(t.k, t.m, std::move(t.q8));
}
```

`--pretrained_dir` is still required in both cases: it provides `config.json`
and the tokenizer. With `--load_quantized` the `.safetensors` file is **not
opened** — the loader prints `[load] source: Q8_0 checkpoint` so you can confirm
which path ran.

Tensor names in a `.q8` file are the same keys `assign_weight()` accepts
(`blocks.0.attn.qkv.weight`, `wte.weight`, ...), so the two paths stay
interchangeable.

### What goes into forward()

| argument | shape | dtype | notes |
|---|---|---|---|
| `input_ids` | `[B, S]` | **int32** | values in `[0, vocab)` |
| `attention_mask` | `[B, S]` | **float32** | optional; `1` = keep, `0` = pad |

```cpp
TensorPtr logits = model.forward(input_ids);        // [B, S, vocab], float32
```

`S` may not exceed `n_positions` (1024 for GPT-2 small). The result holds
next-token logits at every position: `logits[b, S-1, :]` is the distribution for
the token after the last prompt token.

### What the CLIs consume and produce

| binary | input | output |
|---|---|---|
| `train` | token ids from WikiText-2, `[B, S]` int32 per step | per-step loss/PPL on stderr, LoRA adapter `.safetensors` |
| `eval_ppl` | a `max_tokens` window of the valid split | `ppl` / `nll` and a `RESULT_JSON` line |
| `--save_quantized` | the in-memory model after quantization | a `.q8` file (see [Checkpoint format](#checkpoint-format)) |

### Is the interface different for a quantized model?

No. `forward()` takes the same `input_ids` and returns the same
`[B, S, vocab]` float32 logits; only the storage of the base weights and the
arithmetic inside the linear layers change. Everything downstream — the loss,
the backward pass, the evaluation code — is untouched.

The one visible difference is the output projection: `logits = matmul(x, wteᵀ)`
transposes `wte` (147 MiB) on every forward. That cost exists in both paths and
is not affected by this change; quantizing `wte` would remove it as well.

## How the forward works

```
int8 x int8  ──►  accumulate into int32      (exact, no rounding)
int32 × (d_a · d_w)                          (per block, fp32)
fp32 accumulate across blocks
```

The dot product itself is integer: 32 values of a block are multiplied and
summed into an `int32` with no intermediate rounding, and only then is the
block's scale applied — in fp32, not fp16. The fp16 field in `block_q8_0` is
just how the scale is *stored*.

This is why quantized weights also reduce memory traffic: 34 bytes per 32
weights instead of 128.

### Kernel dispatch

The int8 dot product and the `grad_x` kernel have AVX2 implementations selected
at run time:

```cpp
__attribute__((target("avx2,fma"))) inline int32_t q8_dot_avx2(...);
if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) { ... }
```

The default build target stays baseline x86-64, so the artifact remains
portable; no `-march` flag is required. Both kernels fall back to scalar code.

AVX-512 VNNI is deliberately **not** used: `_mm512_dpbusd_epi32` is `u8 × s8`,
and moving the sign to the other operand overflows on the `-128` code that
Q8_0's `d = amax / -128` can produce.

## The backward pass

The frozen-weight branch computes `grad_x = grad_y @ W` by reading the int8
blocks and converting each element to float on the fly. LoRA gradients
(`grad_A`, `grad_B`) are plain fp32 matmuls.

One subtlety matters here: the fused branch builds its output with `zeros()`,
which has no input edges, so registering the backward with `set_grad_fn` makes
the engine take its legacy path and **discard the returned gradient**. The
frozen-weight contribution to `dL/dx` then silently disappears and training
degrades into "LoRA branch only" — while `loss` looks perfectly normal. The
backward must be attached with:

```cpp
Engine::instance().register_node(y2d, {x}, backward_fn);
```

This is covered by the per-layer gradient check described in
[RESULTS.md](RESULTS.md).

## Checkpoint format

`.q8` files produced by `--save_quantized` are little-endian:

```
magic      "MFTQ8\0\0\0"       8 bytes
version    u32                 currently 1
count      u32                 number of tensors
per tensor:
  name_len u32, name bytes     same keys as assign_weight(), e.g.
                               "blocks.0.attn.qkv.weight", "wte.weight"
  is_q8    u8
  if is_q8:  k u32, m u32, nblocks u64, nblocks * 34 bytes
  else:      ndim u32, shape[ndim] i64, count u64, count * 4 bytes (fp32)
```

A Q8_0 block is `{ fp16 d; int8 qs[32] }` = 34 bytes, row-major over `[M, K]`,
the same layout llama.cpp uses.

The reader validates the magic, the version and the payload length, so a
truncated or foreign file fails with a message instead of misbehaving.

## Limitations

- **`--quantize` and `--load_quantized` are mutually exclusive.** A loaded model
  is already quantized; re-quantizing is rejected.
- **Merging a quantized layer re-quantizes it.** `merge_to_base()` folds the
  LoRA delta into an fp32 view of the base and quantizes again, so the merged
  result differs from the un-merged one by the quantization error (measured at
  0.6-0.9% relative). `unmerge_from_base()` restores the exact pre-merge blocks.
- **`wte` is not quantized.** It is 31% of the parameters and also the `lm_head`
  projection, so it is the largest remaining opportunity (roughly 107 MiB of the
  236.8 MiB file).
- **`eval_ppl --quantize` crashes for `seq_len >= 1024`.** Training-side
  evaluation is unaffected.
- Layers whose `in_features` is not a multiple of 32 cannot be represented and
  are left in fp32 (`skipped` in the `[Quant]` log line).

## Tests

```bash
cmake -S operator -B build
cmake --build build -j
ctest --test-dir build
```

Beyond the unit tests, the change was validated with a per-layer gradient check:
one `LoRALinear` (`K=64, M=32, r=4`) is run through both the fp32 and the
quantized path with identical inputs, and `|y|`, `|grad_x|`, `|grad_A|`,
`|grad_B|` are compared (see [RESULTS.md](RESULTS.md#5-correctness-checks)).

See [RESULTS.md](RESULTS.md) for the numbers these produce.
