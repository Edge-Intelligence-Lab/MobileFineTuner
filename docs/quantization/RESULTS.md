# Q8_0 Quantization — Results

All numbers below were measured on this machine (AMD Ryzen 9 8945HX, x86-64,
single-threaded build) with **GPT-2 small (124M)**, LoRA `r=4`, `alpha=8`,
WikiText-2, `seq_len=128`, `batch=1`, `seed=1234`.

Two axes are compared throughout:

| | baseline | quantized |
|---|---|---|
| weights | fp32 (original checkpoint) | Q8_0 blocks, 8.5 bit/weight |
| activations | fp32 | quantized to Q8_0 per forward call |

"Baseline" means the same binary **without** `--quantize`; that path is
untouched by this feature.

---

## 1. Accuracy

### Inference (base model, no adapter)

| | PPL |
|---|---|
| fp32 | **48.7261** |
| Q8_0 | **48.6568** |
| difference | −0.0693 (−0.14%) |

Protocol: WikiText-2 `valid`, 16,384 tokens, `seq_len=512`, all 48 layers
quantized (`skipped=0`).

### Training (200 steps, LoRA)

| step | fp32 | Q8_0 | Δ |
|---|---|---|---|
| 20 | 84.10 | 83.74 | −0.36 |
| 60 | 71.34 | 71.00 | −0.34 |
| 100 | 50.05 | 50.03 | −0.02 |
| 120 | 47.93 | 47.93 | 0.00 |
| 160 | 45.10 | 45.07 | −0.03 |
| **200** | **44.24** | **44.31** | **+0.07 (0.16%)** |

The two runs are identical apart from `--quantize`: 200 steps, 25,600 tokens,
same seed and hyper-parameters. The quantized run is slightly *better* for the
first ~110 steps and ends 0.07 behind.

> Caveat: 200 steps × 128 tokens is 25,600 tokens — about 1% of a full epoch.
> This shows "no degradation at equal steps", not convergence behaviour.

### Per-layer gradient check

One `LoRALinear` (`K=64, M=32, r=4`), fp32 path vs quantized path:

| quantity | fp32 | Q8_0 | ratio |
|---|---|---|---|
| `|y|` (forward) | 16.799659 | 16.761439 | 0.9977 |
| `|grad_x|` | 13.891847 | 13.869336 | 0.9984 |
| `|grad_A|` | 4.229406 | 4.229406 | 1.0000 |
| `|grad_B|` | 1.619986 | 1.619986 | 1.0000 |

`grad_A`/`grad_B` are unchanged because the LoRA branch never touches the
quantized weights.

---

## 2. Size

| | bytes | MiB | ratio |
|---|---|---|---|
| original `model.safetensors` (fp32) | 548,105,171 | 522.7 | — |
| `.q8` checkpoint (this feature) | 248,270,500 | **236.8** | **2.21×** |
| 48 linear weights, fp32 | 339,738,624 | 324.0 | — |
| 48 linear weights, Q8_0 | 90,243,072 | 86.06 | **3.76×** |

The 48-layer figure is exact arithmetic: 84,934,656 parameters × 4 bytes vs
÷32 × 34 bytes.

The whole-file ratio (2.21×) is smaller than the 3.76× weight ratio because
`wte` (38.6M parameters, 147 MiB) is not quantized. It is also a little better
than a naive subtraction predicts, because the original safetensors carries
padding/tensors the model does not use, while the `.q8` file stores exactly the
148 tensors the loader needs.

---

## 3. Speed

Three alternating runs in one session, 8 steps each, no evaluation in the loop:

| | step time | ratio |
|---|---|---|
| fp32 | 25,953 – 26,283 ms | — |
| Q8_0 | 16,072 – 16,159 ms | **1.61 – 1.63×** |

Absolute times drift with machine state (an earlier session measured
21.9 s / 13.5 s for the same pair, also 1.62×), so only same-session ratios are
meaningful. Comparing runs from different days is invalid — one such comparison
gave 1.06× and was simply two different machine states.

Phase split (share of step time):

| | forward | backward |
|---|---|---|
| fp32 | 21.4% | 78.4% |
| Q8_0 | 35.5% | 64.2% |

The gain comes from the backward pass. Profiling the backward by operator type
shows where the time goes:

| backward operator | share |
|---|---|
| `MatmulBackward` (generic fp32) | 87.1% |
| `QuantBaseMatmulBackward` (int8 `grad_x`) | 10.3% |
| everything else (cross-entropy, GELU, Add, LayerNorm) | 2.6% |

So the int8 kernel is only ~10% of backward time; quantizing `grad_y` as well
would have a ceiling of roughly 10% and is not worth the accuracy risk yet. The
real remaining lever is the generic fp32 matmul (single-threaded).

---

## 4. Memory

**Weight storage** is unambiguous: 324.0 MiB → 86.06 MiB.

**Process RSS** depends on how the model is loaded:

| | RSS (pre-step) | note |
|---|---|---|
| fp32 weights + quantize at startup | 1055 MB | fp32 and int8 copies coexist |
| `--load_quantized` (this feature) | **730.6 MB** | no fp32 base at all |

Two caveats:

1. These were measured with the in-tree memory pool enabled (it does not return
   freed memory to the OS). With `-DDISABLE_MEMORY_POOL=ON` the fp32 model sits
   at ~512 MB instead of ~1228 MB, so the absolute numbers move a lot.
2. Without `--save_quantized`/`--load_quantized`, quantizing at startup still
   needs the fp32 file (522.7 MiB read) and keeps the fp32 tensors alive behind
   `GPT2Model`. That is why the plain `--quantize` path does **not** reduce RSS:
   it adds an 86 MiB int8 copy on top of the 324 MiB fp32 original.

---

## 5. Correctness checks

| check | result |
|---|---|
| unit tests (`ctest`) | 21/21 pass |
| round trip: fp32 → `--save_quantized` → `--load_quantized` | step-1 numerics identical: `loss 4.3130`, `ppl 74.67`, `grad_norm 0.174` |
| int8 dot product: AVX2 vs scalar | checksums equal |
| `grad_x` kernel: AVX2 vs scalar | max relative difference 2.1e-7 |
| merge/unmerge on a quantized layer | merge changes the output by 0.6-0.9% (requantization only); unmerge restores it exactly (0.0) |
| dequantize(quantize(x)) | relative error < 1e-6 |

The step-1 equivalence is the strongest single check: the same seed produces
bit-identical loss and gradient norm whether the weights came from the fp32
file or from a `.q8` checkpoint.

---

## 6. Reproducing

```bash
# baseline (fp32)
./train --pretrained_dir $MODEL --data_dir $DATA --steps 200 --seq_len 128 \
        --rank 4 --alpha 8 --lr 2e-4 --seed 1234 --eval_interval 10

# quantized
./train ... --quantize q8_0 --lora_out q8.safetensors

# export and reload
./train ... --quantize q8_0 --save_quantized model.q8
./train ... --load_quantized model.q8

# inference PPL of the base model
./eval_ppl --pretrained_dir $MODEL --data_dir $DATA --seq_len 512 \
           --max_tokens 16384 --quantize q8_0
```

---

## 7. What this does not claim

- It does not show that 4-bit works. Q8_0 has 256 levels; the error budget at
  4 bits is 16× tighter and this feature says nothing about it.
- It does not show behaviour at scale. GPT-2 small, 25,600 tokens, one seed.
- It is not an inference-speed result. The quantized path is slower than fp32
  for a single short forward pass; the win is in training, where the backward
  pass dominates.
