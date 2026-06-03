#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

fail=0

say() {
  printf '%s\n' "$*"
}

check_no_personal_paths() {
  local files=(
    README.md
    docs/MODEL_ASSETS.md
    scripts/README.md
    scripts/asset_paths.py
    scripts/lib/asset_paths.sh
    scripts/check_local_assets.sh
    scripts/run_training_smoke.sh
    scripts/run_training_real_assets.sh
    scripts/android/android_env.sh
    scripts/android/adb_resource_monitor.sh
    scripts/android/build_qwen_android.sh
    scripts/android/run_qwen_qnli_native_phone.sh
    scripts/android/stage_qwen_qnli_phone_assets.sh
    operator/CMakeLists.txt
    operator/cmake/OperatorsConfig.cmake.in
    operator/cmake/MobileFineTunerConfig.cmake.in
    operator/include/mobile_finetuner/mobile_finetuner.h
  )
  local matches
  matches="$(cd "$ROOT" && rg -n '/Users/|Documents/pretrained_models|Documents/MobileFineTuner|Documents/MobileFinetuner' "${files[@]}" || true)"
  if [ -n "$matches" ]; then
    say "[FAIL] Personal paths found in maintained release files:"
    say "$matches"
    fail=1
  else
    say "[OK]   No personal absolute paths in maintained release files"
  fi
}

check_no_archived_experiment_surface() {
  local matches
  matches="$(
    cd "$ROOT" && rg -n 'ONNX|Onnx|onnx|\bORT\b|OrtTraining|Termux|termux' \
      README.md docs scripts android-visualizer/app/src/main android-visualizer/app/build.gradle.kts operator .github \
      --glob '!Rubbish/**' \
      --glob '!scripts/check_release_tree.sh' \
      --glob '!operator/build*/**' \
      --glob '!**/build/**' \
      --glob '!**/build-android/**' \
      --glob '!review-stage/**' \
      || true
  )"
  if [ -n "$matches" ]; then
    say "[FAIL] Archived ONNX/Termux experiment surface leaked outside Rubbish:"
    say "$matches"
    fail=1
  else
    say "[OK]   ONNX/Termux experiment surface is isolated in Rubbish"
  fi

  matches="$(
    cd "$ROOT" && find . \
      -path './.git' -prune -o \
      -path './.agents' -prune -o \
      -path './.aris' -prune -o \
      -path './Rubbish' -prune -o \
      -path './operator/build*' -prune -o \
      -path './*/build' -prune -o \
      -path './*/build-android' -prune -o \
      \( -path './operator/opt_ops' \
         -o -path './pytorch_alignment' \
         -o -path './scripts/Finetune' \
         -o -path './scripts/onnx_lora' \
         -o -path './operator/finetune_ops/graph/test_safetensors_simple' \
         -o -path './operator/finetune_ops/graph/pt_last_logits.json' \
         -o -path './operator/finetune_ops/graph/layer0_activations.json' \
         -o -path './operator/finetune_ops/graph/save_pt_gold.py' \) \
      -print
  )"
  if [ -n "$matches" ]; then
    say "[FAIL] Archived experimental/generated files remain in the release tree:"
    say "$matches"
    fail=1
  else
    say "[OK]   Experimental/generated source artifacts are isolated in Rubbish"
  fi
}

check_large_source_files() {
  local matches
  matches="$(
    cd "$ROOT" && find . -type f -size +50M \
      -not -path './.git/*' \
      -not -path './.agents/*' \
      -not -path './.aris/*' \
      -not -path './.venv*/*' \
      -not -path './data/*' \
      -not -path './runs/*' \
      -not -path './pytorch_runs/*' \
      -not -path './review-stage/*' \
      -not -path './Rubbish/*' \
      -not -path './operator/build*/*' \
      -not -path './*/build/*' \
      -not -path './*/build-android/*' \
      -not -path './*/pretrained/*' \
      -not -path './*/outputs/*' \
      -print
  )"
  if [ -n "$matches" ]; then
    say "[FAIL] Large files outside ignored asset/build areas:"
    say "$matches"
    fail=1
  else
    say "[OK]   No >50MB source-tree files outside ignored asset/build areas"
  fi
}

check_package_files() {
  local required=(
    operator/cmake/OperatorsConfig.cmake.in
    operator/cmake/MobileFineTunerConfig.cmake.in
    operator/include/mobile_finetuner/mobile_finetuner.h
    docs/MODEL_ASSETS.md
    docs/PUBLIC_API.md
  )
  local path
  local missing=0
  for path in "${required[@]}"; do
    if [ ! -f "$ROOT/$path" ]; then
      say "[FAIL] Missing package/documentation file: $path"
      fail=1
      missing=1
    fi
  done
  if [ "$missing" -eq 0 ]; then
    say "[OK]   Package config templates and asset docs exist"
  fi
}

check_no_generated_caches() {
  local matches
  matches="$(
    cd "$ROOT" && find . \
      -path './.git' -prune -o \
      -path './.agents' -prune -o \
      -path './.aris' -prune -o \
      -path './Rubbish' -prune -o \
      -path './runs' -prune -o \
      -path './review-stage' -prune -o \
      -name __pycache__ -type d -print
  )"
  if [ -n "$matches" ]; then
    say "[FAIL] Python cache directories remain in the maintained tree:"
    say "$matches"
    fail=1
  else
    say "[OK]   No Python cache directories in maintained tree"
  fi
}

check_no_stale_operator_api() {
  local stale=(
    operator/finetune_ops/model
    operator/finetune_ops/nn/embedding.cpp
    operator/finetune_ops/nn/embedding.h
    operator/finetune_ops/optim/lora_optimizer.cpp
    operator/finetune_ops/optim/lora_optimizer.h
    operator/finetune_ops/optim/train_lora_gemma.cpp
  )
  local path
  local found=()
  for path in "${stale[@]}"; do
    if [ -e "$ROOT/$path" ]; then
      found+=("$path")
    fi
  done
  if [ "${#found[@]}" -gt 0 ]; then
    say "[FAIL] Stale operator API files remain in maintained tree:"
    printf '  %s\n' "${found[@]}"
    fail=1
  else
    say "[OK]   Stale operator API files are archived"
  fi
}

check_shell_syntax() {
  local scripts=(
    scripts/check_local_assets.sh
    scripts/run_training_smoke.sh
    scripts/run_training_real_assets.sh
    scripts/lib/asset_paths.sh
    scripts/android/adb_resource_monitor.sh
    scripts/android/android_env.sh
    scripts/android/build_qwen_android.sh
    scripts/android/run_qwen_qnli_native_phone.sh
    scripts/android/stage_qwen_qnli_phone_assets.sh
    examples/gpt2_small_lora_finetune/run_train.sh
    examples/gpt2_medium_lora_finetune/run_train.sh
    examples/gemma_3_270m_lora_finetune/run_train.sh
    examples/gemma_3_1b_pt_lora_finetune/run_train.sh
    examples/qwen_lora_finetune/run_wikitext.sh
    examples/qwen_lora_finetune/run_mmlu.sh
    examples/qwen_lora_finetune/run_qnli.sh
  )
  local path
  for path in "${scripts[@]}"; do
    bash -n "$ROOT/$path"
  done
  say "[OK]   Maintained shell entrypoints pass bash -n"
}

check_no_personal_paths
check_no_archived_experiment_surface
check_large_source_files
check_package_files
check_no_generated_caches
check_no_stale_operator_api
check_shell_syntax

if [ "$fail" -ne 0 ]; then
  exit 1
fi

say "[OK]   Release tree audit passed"
