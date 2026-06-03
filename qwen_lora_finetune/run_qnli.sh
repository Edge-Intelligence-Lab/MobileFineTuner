#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/.. && pwd)"
source "$REPO_ROOT/scripts/lib/asset_paths.sh"
BIN="$SCRIPT_DIR/build/train_wikitext"
MODEL_DIR="${QWEN_MODEL_DIR:-$(mf_resolve_model_dir qwen "$SCRIPT_DIR/pretrained")}"
QNLI_JSONL_DIR="${QNLI_JSONL_DIR:-$REPO_ROOT/data/qnli/qwen_qnli_s64}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/outputs/qnli}"

SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-${GRAD_ACCUM:-1}}"
MAX_STEPS="${MAX_STEPS:-3}"
LR="${LR:-2e-4}"
LOSS_IMPL="${LOSS_IMPL:-full_dense}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LORA_R="${LORA_R:-${RANK:-8}}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGETS="${LORA_TARGETS:-qkvo}"
SEED="${SEED:-42}"
NO_SHUFFLE="${NO_SHUFFLE:-0}"
DUMP_FIRST_BATCH_JSON="${DUMP_FIRST_BATCH_JSON:-}"
DUMP_ONLY="${DUMP_ONLY:-0}"
BASE_WEIGHT_STORAGE="${BASE_WEIGHT_STORAGE:-auto}"

TARGET_FLAG="--qv_only"
if [[ "$LORA_TARGETS" == "qkvo" ]]; then
  TARGET_FLAG="--qkvo"
fi

EXTRA_FLAGS=()
if [[ "$NO_SHUFFLE" == "1" ]]; then
  EXTRA_FLAGS+=("--no_shuffle")
fi
if [[ -n "$DUMP_FIRST_BATCH_JSON" ]]; then
  EXTRA_FLAGS+=("--dump_first_batch_json" "$DUMP_FIRST_BATCH_JSON")
fi
if [[ "$DUMP_ONLY" == "1" ]]; then
  EXTRA_FLAGS+=("--dump_only")
fi
PY="${PYTHON:-python3}"

if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$BIN" ]; then
  cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build"
  cmake --build "$SCRIPT_DIR/build" --target train_wikitext -j
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "找不到模型目录：$MODEL_DIR" >&2
  exit 1
fi

if [ ! -f "$QNLI_JSONL_DIR/train.jsonl" ] || [ ! -f "$QNLI_JSONL_DIR/valid.jsonl" ]; then
  "$PY" "$REPO_ROOT/scripts/prepare_qnli_jsonl.py" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$QNLI_JSONL_DIR" \
    --seq_len "$SEQ_LEN"
fi

mkdir -p "$OUT_DIR"

"$BIN" \
  --model_dir "$MODEL_DIR" \
  --jsonl_train "$QNLI_JSONL_DIR/train.jsonl" \
  --jsonl_valid "$QNLI_JSONL_DIR/valid.jsonl" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --max_steps "$MAX_STEPS" \
  --lr "$LR" \
  --loss_impl "$LOSS_IMPL" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --base_weight_storage "$BASE_WEIGHT_STORAGE" \
  --seed "$SEED" \
  "$TARGET_FLAG" \
  "${EXTRA_FLAGS[@]}" \
  --output_dir "$OUT_DIR"
