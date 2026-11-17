# Native Qwen stability runs on macOS

Build the candidate once. The controller copies that executable into the run
folder and records its SHA-256 before launching training. Later builds do not
change an already queued run. Model snapshots and WikiText data are external
assets; no Python ML library is used by the controller or native training.

```bash
cmake -S examples/qwen_lora_finetune -B runs/candidate-build \
  -DCMAKE_BUILD_TYPE=Release -DUSE_BLAS=ON
cmake --build runs/candidate-build -j2
python3 scripts/run_qwen_mac_stability.py \
  --binary runs/candidate-build/train_wikitext \
  --model-dir "$QWEN35_MODEL_DIR" \
  --model-dir "$QWEN3_MODEL_DIR" \
  --model-dir "$QWEN25_MODEL_DIR" \
  --data-dir "$WT2_DATA_DIR" --output runs/candidate-stability \
  --steps 300 --resume-steps 30 --stress-steps 5 --background
```

For each model, the serial queue runs three calibration steps, the long run,
and adapter resume with a fresh optimizer. The first model also gets a final
sequence-length-128 stress run. Defaults: sequence length 32, batch 1, rank 4,
alpha 8, dropout 0.05, all-linear targets, learning rate 2e-4, clipping norm 1,
native frozen matrix storage and a four-thread environment cap. Thread caps are
requests to the numerical runtime, not CPU affinity guarantees. Optional
`--calibration-data-dir` supplies a small excerpt; otherwise calibration uses the
main dataset. `--after-status runs/previous/status.json` waits for a prior queue
to finish successfully before starting, so queues need not train concurrently.

Inspect `status.json`, `REPORT.md`, `plan.json` and per-job logs/CSV files. The
controller checks exact step counts, finite loss/gradients/parameters, nonzero B
matrices, adapter tensor counts, resumed tensor changes, clipping bounds and
periodic/final snapshot agreement. Resume does not restore optimizer moments,
step counters or RNG state. Changing minibatches means loss need not decrease at
every step; this run does not measure downstream model quality.

External RSS samples are taken approximately every five seconds; macOS swap,
memory-pressure level and thermal-warning reports are collected as available.
Thermal-warning output is not a measured chip temperature. RSS trend and timing
flags are diagnostics, not automatic proof of a leak or speed regression. The
queue stops its own child process group on sustained critical pressure, RSS
above 8 GiB or swap growth above 3 GiB by default. These limits are configurable.
Caffeinate prevents idle sleep while the queue is active. Logs remain local.

To request a stop, write `{"stop_requested":true}` to the run folder's
`control.json`. Use a new output folder for a new run. Interrupted/failed queues
retain their evidence and are not silently counted as completed. `--background`
returns a PID; it does not promise a later chat notification.

A short hermetic integration test exercises detached execution, training/resume,
frozen executable hashing, report creation and stopping a queued run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_qwen_training_cli.py \
  --binary runs/candidate-build/train_wikitext \
  --mmlu-binary runs/candidate-build/train_mmlu
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_qwen_mac_stability.py \
  --binary runs/candidate-build/train_wikitext
```

Mac long-run completion and Android device acceptance are separate checks.
Neither a cross-compile nor a synthetic run qualifies every model size, context
length or target phone. See [integration evidence](validation/PR_INTEGRATION_20261117.md).
