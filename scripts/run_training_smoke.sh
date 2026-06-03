#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
SMOKE_STEPS="${SMOKE_STEPS:-2}"

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

run_case "gpt2_small" "examples/gpt2_small_lora_finetune" env SMOKE=1 SMOKE_STEPS="$SMOKE_STEPS" ./run_train.sh
run_case "gpt2_medium" "examples/gpt2_medium_lora_finetune" env SMOKE=1 SMOKE_STEPS="$SMOKE_STEPS" ./run_train.sh
run_case "gemma_270m" "examples/gemma_3_270m_lora_finetune" env SMOKE=1 SMOKE_STEPS="$SMOKE_STEPS" ./run_train.sh
run_case "gemma_1b_pt" "examples/gemma_3_1b_pt_lora_finetune" env SMOKE=1 SMOKE_STEPS="$SMOKE_STEPS" ./run_train.sh
run_case "qwen_wikitext" "examples/qwen_lora_finetune" env SMOKE=1 SMOKE_STEPS="$SMOKE_STEPS" ./run_wikitext.sh

echo
echo "All five training smoke runs passed."
