# Scripts Directory

This directory keeps the maintained helper scripts that match the current
repository layout and entrypoints. Historical benchmark and experiment scripts
are archived under `Rubbish/`; the scripts documented here are the stable
interfaces expected by CI and release checks.

## Directory Structure

```text
scripts/
├── asset_paths.py
├── check_local_assets.sh
├── check_release_tree.sh
├── android/
│   ├── adb_resource_monitor.sh
│   ├── android_env.sh
│   ├── build_qwen_android.sh
│   ├── proc_mem_monitor.cpp
│   ├── run_qwen_qnli_native_phone.sh
│   └── stage_qwen_qnli_phone_assets.sh
├── lib/
│   └── asset_paths.sh
├── plot_loss_curve.py
├── pretokenize_wikitext2_gemma.py
├── run_training_real_assets.sh
└── run_training_smoke.sh
```

## Maintained Scripts

### `check_local_assets.sh`

Prints the actual model/data asset directories that the repo will use on the current machine.

```bash
export MFT_MODEL_ROOT=/path/to/mft-models
export MFT_DATA_ROOT=/path/to/mft-data
bash scripts/check_local_assets.sh
```

### `check_release_tree.sh`

Runs source-distribution checks for maintained entrypoints:

- no personal absolute paths in release-facing files;
- no large source files outside ignored asset/build areas;
- package config templates and asset docs are present;
- maintained shell scripts pass `bash -n`.

```bash
bash scripts/check_release_tree.sh
```

### `run_training_smoke.sh`

Runs the repo-level five-model synthetic smoke suite.

```bash
bash scripts/run_training_smoke.sh
```

Environment overrides:
- `SMOKE_STEPS` to control the number of synthetic update steps

### `run_training_real_assets.sh`

Runs a one-step real-asset validation for all five training entrypoints using auto-discovered local model/data directories.

```bash
bash scripts/run_training_real_assets.sh
```

Environment overrides:
- `REAL_STEPS` to control the number of optimizer steps
- `SEQ_LEN` to control the short sanity-check sequence length
- `MFT_MODEL_ROOT` and `MFT_DATA_ROOT` to provide external assets

### Android Native MF Helpers

The maintained Android helpers build and run the native MF Qwen/QNLI path:

```bash
bash scripts/android/build_qwen_android.sh
bash scripts/android/stage_qwen_qnli_phone_assets.sh
bash scripts/android/run_qwen_qnli_native_phone.sh
```

Environment overrides:
- `ANDROID_NDK_ROOT`, `ANDROID_NDK_HOME`, `ANDROID_HOME`, or `ANDROID_SDK_ROOT`
  to locate Android tooling
- `ADB` to override the adb binary
- `DEVICE_ROOT` and `DEVICE_OUT_DIR` to choose phone-side asset/output paths

### `pretokenize_wikitext2_gemma.py`

Offline-tokenizes WikiText-2 for Gemma-family experiments.

```bash
python3 scripts/pretokenize_wikitext2_gemma.py
```

Outputs:
- `data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin`
- `data/wikitext2/pretokenized_gemma/meta.json`

### `plot_loss_curve.py`

Plots a loss curve from a training log.

```bash
python3 scripts/plot_loss_curve.py path/to/train.log path/to/loss.png
```
