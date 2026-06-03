#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
source "$ROOT/scripts/lib/asset_paths.sh"

report() {
  local name="$1"
  shift
  local resolved
  if resolved="$("$@" 2>&1)"; then
    echo "[OK]      $name: $resolved"
  else
    echo "[MISSING] $name"
    echo "$resolved" | sed 's/^/          /'
    return 1
  fi
}

missing=0
report "WikiText-2" mf_resolve_wikitext_dir "$ROOT" || missing=1
report "MMLU" mf_resolve_mmlu_dir "$ROOT" || missing=1
report "GPT-2 Small" mf_resolve_model_dir gpt2_small "$ROOT/gpt2_small_lora_finetune/pretrained" || missing=1
report "GPT-2 Medium" mf_resolve_model_dir gpt2_medium "$ROOT/gpt2_medium_lora_finetune/pretrained" || missing=1
report "Gemma 270M" mf_resolve_model_dir gemma_270m "$ROOT/gemma_3_270m_lora_finetune/pretrained" || missing=1
report "Gemma 1B-PT" mf_resolve_model_dir gemma_1b_pt "$ROOT/gemma_3_1b_pt_lora_finetune/pretrained" || missing=1
report "Qwen2.5-0.5B" mf_resolve_model_dir qwen "$ROOT/qwen_lora_finetune/pretrained" || missing=1

if [ "$missing" -ne 0 ]; then
  cat <<EOF

Some assets are missing. MobileFineTuner does not ship pretrained model
weights or benchmark datasets. Provide them explicitly with:

  export MFT_MODEL_ROOT=/path/to/huggingface/snapshots
  export MFT_DATA_ROOT=/path/to/datasets

or set per-model variables such as GPT2_SMALL_MODEL_DIR, GEMMA_270M_MODEL_DIR,
GEMMA_1B_PT_MODEL_DIR, and QWEN_MODEL_DIR.

Model directories may contain either model.safetensors or a HuggingFace
model.safetensors.index.json plus shard files. See docs/MODEL_ASSETS.md for
the required directory layout.
EOF
  exit 1
fi
