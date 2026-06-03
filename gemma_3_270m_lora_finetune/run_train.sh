#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/.. && pwd)"
source "$REPO_ROOT/scripts/lib/asset_paths.sh"
PY="${PYTHON:-python3}"

TRAIN_MODE="${TRAIN_MODE:-mmlu}"
JSONL="${MMLU_JSONL:-$REPO_ROOT/runs/mmlu_jsonl_gemma270m_s128/train.jsonl}"
WT2_DATA_DIR="${WT2_DATA_DIR:-}"
PRETRAINED="${GEMMA_270M_MODEL_DIR:-}"
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

STEPS="${STEPS:-200}"  # If >0, use max_steps; otherwise use EPOCHS
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-128}"
LR="${LR:-2e-4}"
RANK="${RANK:-${LORA_R:-8}}"
ALPHA="${ALPHA:-16}"
EPOCHS="${EPOCHS:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-${GRAD_ACCUM_STEPS:-1}}"
SMOKE="${SMOKE:-0}"
SMOKE_STEPS="${SMOKE_STEPS:-2}"

echo "[Train] jsonl=$JSONL"
echo "[Train] mode=$TRAIN_MODE"
echo "[Train] out_dir=$OUT_DIR"

# Use local C++ binary built in this directory
LOCAL_BIN="$SCRIPT_DIR/build/train"
if [ ! -x "$LOCAL_BIN" ]; then
  echo "[Build] 構建本地二進位..."
  "$SCRIPT_DIR/build.sh"
fi

if [ "$SMOKE" = "1" ]; then
  ARGS=( \
    --synthetic_smoke \
    --smoke_steps "$SMOKE_STEPS" \
    --seq_len "$SEQ_LEN" \
    --batch "$BATCH_SIZE" \
    --lr "$LR" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --output_dir "$OUT_DIR" \
  )
else
  WT2_DATA_DIR="${WT2_DATA_DIR:-$(mf_resolve_wikitext_dir "$REPO_ROOT")}"
  PRETRAINED="${PRETRAINED:-$(mf_resolve_model_dir gemma_270m "$SCRIPT_DIR/pretrained")}"
  echo "[Train] wt2_data=$WT2_DATA_DIR"
  echo "[Train] pretrained=$PRETRAINED"
  ARGS=( \
    --model_dir "$PRETRAINED" \
    --seq_len "$SEQ_LEN" \
    --batch "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --output_dir "$OUT_DIR" \
  )
  case "$TRAIN_MODE" in
    mmlu)
      ARGS+=( --jsonl_train "$JSONL" )
      ;;
    wt2)
      ARGS+=( --data_dir "$WT2_DATA_DIR" )
      ;;
    *)
      echo "Unsupported TRAIN_MODE=$TRAIN_MODE (expected mmlu or wt2)" >&2
      exit 1
      ;;
  esac
  if [ "${STEPS}" -gt 0 ]; then
    ARGS+=( --max_steps "$STEPS" )
  else
    ARGS+=( --epochs "$EPOCHS" )
  fi
fi

"$LOCAL_BIN" "${ARGS[@]}"

echo "[Train] Done. LoRA artifact dir: $OUT_DIR"
