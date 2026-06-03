# Model And Dataset Assets

MobileFineTuner does not bundle pretrained model weights or benchmark datasets.
The library code, training CLIs, and tests are shipped separately from large
assets. Users provide local HuggingFace-style model snapshots at runtime, similar
to how PyTorch/Transformers accepts either a model identifier or a local
`from_pretrained(...)` path.

MobileFineTuner currently uses local paths only. It does not call
`from_pretrained(...)` and does not download model IDs at runtime. Download or
copy the model snapshot yourself, then pass that directory through an
environment variable or a CLI flag such as `--model_dir`.

## Recommended Layout

Use one shared model root:

```text
/path/to/mft-models/
  gpt2/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    vocab.json
    merges.txt
  gpt2-medium/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    vocab.json
    merges.txt
  gemma-3-270m/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    tokenizer.model
  gemma-3-1b-pt/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    tokenizer.model
  Qwen2.5-0.5B/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    vocab.json
    merges.txt
```

Then export:

```bash
export MFT_MODEL_ROOT=/path/to/mft-models
```

The runner scripts also accept per-model overrides:

```bash
export GPT2_SMALL_MODEL_DIR=/path/to/gpt2
export GPT2_MEDIUM_MODEL_DIR=/path/to/gpt2-medium
export GEMMA_270M_MODEL_DIR=/path/to/gemma-3-270m
export GEMMA_1B_PT_MODEL_DIR=/path/to/gemma-3-1b-pt
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B
```

For backwards-compatible local experiments, each model application also checks
its own `pretrained/` directory, for example:

```text
qwen_lora_finetune/pretrained/
gpt2_small_lora_finetune/pretrained/
```

Do not commit these directories. They are intentionally ignored by Git.

## Downloading Models

One practical workflow is to use HuggingFace CLI:

```bash
pip install -U huggingface_hub

huggingface-cli download gpt2 \
  --local-dir "$MFT_MODEL_ROOT/gpt2" \
  --local-dir-use-symlinks False

huggingface-cli download gpt2-medium \
  --local-dir "$MFT_MODEL_ROOT/gpt2-medium" \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir "$MFT_MODEL_ROOT/Qwen2.5-0.5B" \
  --local-dir-use-symlinks False
```

For gated models such as Gemma, accept the model license on HuggingFace first and
authenticate with `huggingface-cli login`.

## Weight File Contract

The stable C++ loader accepts either:

- one file named `model.safetensors`; or
- a HuggingFace sharded snapshot with `model.safetensors.index.json` and the
  referenced shard files.

Sharded examples look like:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

The loader reads the index, opens the listed shards, and resolves requested
tensor names across all shard files. If both `model.safetensors` and an index
are present, the single-file weight takes precedence.

## Dataset Layout

Use one shared data root:

```text
/path/to/mft-data/
  wikitext2/wikitext-2-raw/
    wiki.train.raw
    wiki.valid.raw
    wiki.test.raw
  mmlu/data/
    README.txt
    dev/abstract_algebra_dev.csv
    ...
```

Then export:

```bash
export MFT_DATA_ROOT=/path/to/mft-data
```

The default repository-local fallback is:

```text
data/wikitext2/wikitext-2-raw/
data/mmlu/data/
```

## Validation

Check all local assets before running real training:

```bash
bash scripts/check_local_assets.sh
```

The smoke suite does not require real model weights:

```bash
bash scripts/run_training_smoke.sh
```

The real-asset sanity suite requires valid model and dataset paths:

```bash
bash scripts/run_training_real_assets.sh
```

## Runtime Contract

The reusable C++ library does not download assets. Applications must pass asset
paths explicitly. The common patterns are:

```bash
QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B \
QWEN_DATA_DIR=/path/to/wikitext-2-raw \
./qwen_lora_finetune/run_wikitext.sh
```

or:

```bash
MFT_MODEL_ROOT=/path/to/mft-models \
MFT_DATA_ROOT=/path/to/mft-data \
bash scripts/run_training_real_assets.sh
```
