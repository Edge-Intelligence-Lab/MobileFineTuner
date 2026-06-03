#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
REAL_STEPS="${REAL_STEPS:-1}"
SEQ_LEN="${SEQ_LEN:-32}"

run_case() {
  local name="$1"
  local workdir="$2"
  shift 2
  echo
  echo "==== $name ===="
  (
    cd "$ROOT/$workdir"
    "$@"
  )
}

run_case "gpt2_small_wt2" "examples/gpt2_small_lora_finetune" \
  env TRAIN_MODE=wt2 STEPS="$REAL_STEPS" BATCH_SIZE=1 GRAD_ACCUM=1 SEQ_LEN="$SEQ_LEN" ./run_train.sh

run_case "gpt2_medium_wt2" "examples/gpt2_medium_lora_finetune" \
  env TRAIN_MODE=wt2 STEPS="$REAL_STEPS" BATCH_SIZE=1 GRAD_ACCUM=1 SEQ_LEN="$SEQ_LEN" ./run_train.sh

run_case "gemma_270m_wt2" "examples/gemma_3_270m_lora_finetune" \
  env TRAIN_MODE=wt2 STEPS="$REAL_STEPS" BATCH_SIZE=1 GRAD_ACCUM=1 SEQ_LEN="$SEQ_LEN" ./run_train.sh

run_case "gemma_1b_pt_wt2" "examples/gemma_3_1b_pt_lora_finetune" \
  env TRAIN_MODE=wt2 STEPS="$REAL_STEPS" BATCH_SIZE=1 GRAD_ACCUM=1 SEQ_LEN="$SEQ_LEN" ./run_train.sh

run_case "qwen_wikitext" "examples/qwen_lora_finetune" \
  env MAX_STEPS="$REAL_STEPS" BATCH_SIZE=1 GRAD_ACCUM_STEPS=1 SEQ_LEN="$SEQ_LEN" ./run_wikitext.sh

echo
echo "All five training runs completed with real local assets."
