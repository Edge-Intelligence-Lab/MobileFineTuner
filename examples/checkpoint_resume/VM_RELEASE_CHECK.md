# Single-emulator Suspend / Resume

Use the [Android guide](ANDROID_CONTINUITY.md) for the APK, input folder, and controls.
Do not update/reinstall the APK or move assets during an unfinished run.
The candidate restores at completed optimizer-update boundaries, not at an
arbitrary instruction or microbatch. Wait for a committed suspended state and
native process exit before treating a pause as successful. If killed, work
after the last committed checkpoint may be replayed.

## Fixes

- Qwen and Gemma validate the trainer and both data samplers *inside* candidate
  selection. Invalid current state can fall back to the recorded predecessor.
  No new checkpoint format or training mathematics. No search of unrelated runs.
- Session JSONL inputs are validated using the loaded model vocabulary, binary
  supervision mask and post-truncation length. A bad row reports its file/line
  instead of silently reducing the workload. This is not a tokenizer identity check.
- Historical system stops are checked under the task-control lock against the
  latest explicit resume. Manual start handoff is owned by the application
  process, not the lifetime of a Compose screen. This survives recreation,
  not arbitrary death before durable admission.
- Cold launch remembers the selected task. Repeated pause polls avoid rewriting
  an existing marker. Record export skips orphan frozen-weight shard caches.
  App import rejects a one-token sequence (no supervised next-token target).
- A delayed UI Start cannot re-enable training after a later Pause: admission
  uses boot-scoped monotonic intent order under the task lock. Slice progress
  is measured from the actual native restore event, including predecessor replay.
- Optional instrumentation fixtures require explicit selection by their host
  driver. Normal suite discovery must not schedule a reboot fixture.

## Reproduce

Use an idle disposable arm64 API 35 emulator with the APK and instrumentation
APK built from this tree installed. Run the commands from `examples/checkpoint_resume/`;
set ADB to the absolute Android SDK adb executable and SERIAL to the emulator.
Never use a personal phone for lifecycle tests.

```sh
python3 tests/android_reboot.py --adb "$ADB" --serial "$SERIAL" \
  --mode paused-doze --output "$OUT/doze"
python3 tests/android_saf_two_steps.py --adb "$ADB" --serial "$SERIAL" \
  --folder GemmaSuspendPilot --output "$OUT/app-pair"
```

For the second command, stage a prepared two-update folder under shared
Documents, e.g. via tools/prepare_android_task.py and adb push. Use
seq16, rank8, microbatch1, model-specific tokenized inputs and a measured
watchdog grace (420s used in the Gemma pilot). The test needs a slow enough
first update to press Pause; short tiny-model tests belong in instrumentation.
Use English DocumentsUI. The harness does not edit private task records.

For a native numeric pair (distinct from the Android user/worker path):

```sh
python3 tests/android_model_pair.py --adb "$ADB" --serial "$SERIAL" \
  --family qwen --model "$MODEL" --data "$TOKENIZED" --optimization combined \
  --steps 10 --stop 4 --seq 64 --timeout 3600 --output "$OUT/combined"
```

The model directory contains config.json and model.safetensors; the data directory
contains train.jsonl and valid.jsonl. Use a fresh output for each pair. Options
are baseline, accumulation, checkpointing, streaming, sharding and combined.
Each pair uses its own single-PID reference, shared initial adapter, fresh
processes and archived numerical comparison. Scratch is removed only on success.
For planning 200 steps use --steps 200 --stop 73 with a per-process timeout
based on the matching pilot; this is a command, not a claim those full-model
experiments have already run.

## Evidence boundary

The new trainers and independent Android app are same-build CPU FP32 frozen
LoRA with dropout zero. A short same-build match measures continuity, not
agreement with PyTorch, full task accuracy, hardware power, or performance
benefit. A system-picker flow, shell-only native training and host comparisons
are different evidence. Legacy SDK/JNI/MMLU, multi-device coordination,
cross-build resume, nonzero dropout, Full-FT/GPU and arbitrary mid-update
snapshots are not implemented by this package.
Background continuation uses WorkManager constraints and retry backoff; a
successful unfinished slice does not promise immediate next execution. Export
is a record for analysis, not portable resume after uninstall. The APKs here
are built locally for debugging, not distributed as store-signed releases.
