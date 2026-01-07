#!/bin/zsh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
PY="${PYTHON:-python3}"

JSONL="$REPO_ROOT/runs/mmlu_jsonl_gemma270m_s128/train.jsonl"
PRETRAINED="$(cd "$(dirname "$0")" && pwd)/pretrained"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)/outputs"
mkdir -p "$OUT_DIR"

STEPS="${STEPS:-200}"  # If >0, use max_steps; otherwise use EPOCHS
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-128}"
LR="${LR:-2e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
EPOCHS="${EPOCHS:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

echo "[Train] jsonl=$JSONL"
echo "[Train] pretrained=$PRETRAINED"
echo "[Train] out_dir=$OUT_DIR"

# Use local C++ binary built in this directory
LOCAL_BIN="$(cd "$(dirname "$0")" && pwd)/build/train"
if [ ! -x "$LOCAL_BIN" ]; then
  echo "[Build] 構建本地二進位..."
  "$(cd "$(dirname "$0")" && pwd)/build.sh"
fi

ARGS=( \
  --model_dir "$PRETRAINED" \
  --jsonl_train "$JSONL" \
  --seq_len "$SEQ_LEN" \
  --batch "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --learning_rate "$LR" \
  --output_dir "$OUT_DIR" \
)

if [ "${STEPS}" -gt 0 ]; then
  ARGS+=( --max_steps "$STEPS" )
else
  ARGS+=( --epochs "$EPOCHS" )
fi

"$LOCAL_BIN" "${ARGS[@]}"

echo "[Train] Done. LoRA artifact: $OUT_DIR/lora_adapter"


