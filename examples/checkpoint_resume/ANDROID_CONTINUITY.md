# Android continuity

Independent research app: `com.mobilefinetuner.continuity`, Android 10+ (API 29),
arm64-v8a. It does not replace the existing visualizer. No model/data downloads,
analytics, network training or credential requirements are added.

## Build and install

Start in `examples/checkpoint_resume/`. Use JDK 17, Android SDK 35, NDK
27.0.12077973, CMake, Python 3 and a configured ANDROID_HOME.
The native PIE executables must be packaged/extracted by Android. Executing code
copied into a writable files directory is prohibited on modern Android.

```sh
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_HOME/ndk/27.0.12077973/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release -DUSE_BLAS=OFF -DBUILD_TESTS=OFF
cmake --build build-android --target mft_qwen_resume mft_gemma_resume -j4
python3 tools/make_android_fixture.py
cd android
./gradlew :app:assembleDebug :app:testDebugUnitTest
adb install app/build/outputs/apk/debug/app-debug.apk
```

Each trainer target also builds `libmft_owned_process.so` in the same ABI
directory. Package all three files. For an isolated cross-build, set
`-DMFT_ANDROID_LIB_DIR=/absolute/output/arm64-v8a` (do not use that override
for the normal APK build).

The runner forks via JNI and execs the trainer directly. Launch returns after
fork, not after model load, exec, or a child readiness message. The child arms
its parent-death signal before redirecting output, changing directory or exec.
Setup/exec failures are observed as exit 126/127; a forced kill is 137.
The owning runner must reap even on cancellation. No process-wide child reaper,
SIGCHLD auto-reap, or other waiter for these children is supported. JNI checks
the SIGCHLD disposition before fork and never uses a PID from a log or checkpoint.

The locally built debug APK is for testing, not a Play Store release. Android
FGS `specialUse` has an explicit subtype; Play distribution would require its
review. Native Android is CPU FP32, without BLAS in this build. Do not infer full
model phone speed/memory from the tiny emulator tests.

## Use

1. Open MFT Continuity. Create the tiny random Qwen test task, or import a folder.
2. Select whether to continue automatically in background and require charging.
3. Start / Resume. A visible foreground notification includes Pause; tapping its
   body selects that notification's task, including when the activity is reused.
   A manual start while another task is active is rejected, not queued. Pause the
   active task and explicitly retry. The initial
   job is persisted in the main app process before starting the service process.
   A newer Resume for the same active task is retained by the live foreground
   service while the old generation exits. It waits at most 550 seconds for that
   same-task lease, then reports failure instead of silently losing the request.
   Pause cancels this handoff. This is not a general queue or persistence of a
   pending manual request after the service itself is killed.
4. Pause disables automatic continuation and requests the next complete update.
   Wait for `suspended`. The model child exits, releasing its address space.
5. Start / Resume later uses the same original task and native `LATEST`; no
   re-entering training arguments. Export records only while all training is idle.
   For an unfinished checkpoint, choose **Resume physical batch** before
   **Start / Resume**. **Keep previous batch** preserves the last selection.
   Choose any cap from 1 through the original batch. For example, 8 rows with
   cap 3 are processed as 3+3+2; a later resume at cap 5 uses 5+3.
   This is a user choice, not an automatic RAM estimate; it does not override
   the memory-pressure hold or change examples per optimizer update.
   Slice seconds is a runtime control (1..120), adjustable without changing the
   saved training configuration. Scheduling completes before the FGS is started.
   Explicit Resume replaces the pending job so a changed charging constraint takes
   effect. Boot recovery keeps the existing job instead of replacing it repeatedly.
6. Status separates `historical_state`, current `control`, per-task
   `execution_active`, and checkpoint metadata. Another task's execution lock is
   not evidence that this task is alive. Controls refresh after notification Pause
   and external changes without resetting unsaved UI edits on every poll.
7. Failure shows a bounded error excerpt from the attempt log (up to 8KiB read,
   1200 characters displayed), with full logs retained under `attempts/`.
   `checkpoint_available` means readable current/recorded-predecessor metadata and
   matching payload lengths, not successful tensor/asset validation. Native Resume
   remains authoritative. If every committed candidate (including a sole first
   checkpoint) is definitively rejected by the
   native validator, or a stopped attempt never committed a checkpoint, use
   **Retry as new task (reinitialize)**: it copies inputs to a new task, leaves the
   original failure directory intact, and starts disabled until explicit Start.
   It is not exact resume and does not repair invalid inputs; missing/bad assets
   may require re-import. A temporarily inaccessible checkpoint is not proof
   of corruption; resolve the cause and try the same task again.

### Choosing how to stop

| Action | What happens to this update | How to resume |
| --- | --- | --- |
| Pause (including notification Pause) | Cooperative: finish the current optimizer update and commit before exiting normally. | Press Start / Resume; wait for the trainer to exit before switching physical batch. |
| Release resources now | Confirm to terminate this task's native trainer. A commit already published is retained; unfinished work is replayed. | Use the actual last committed update shown after exit. Before the first commit, reinitialization may be necessary. |
| Focus for one hour | Confirm to release the active trainer and disarm scheduled training for **all** tasks. New starts are blocked. | End focus early or wait for expiry. Neither restarts anything; explicitly resume the task you want. |
| Memory pressure / optional CPU contention | The supervisor asks the trainer to stop at a safe boundary. A system kill or slice timeout can still happen first. | Clear the pressure and explicitly resume. No game name or usage-access permission is needed. |

The status line distinguishes a pending stop from an exited trainer. It reports
the committed update, not the update that happened to be running at the tap.
Focus uses elapsed time while booted; after a reboot the remaining duration is
estimated once from the wall clock, then measured with elapsed time again.
No FPS guarantee or GPU-only detector is implied. Different physical batch
partitions can cause FP32 differences; changing a cap is not a bitwise or task
accuracy guarantee.

Imported folder:

```text
experiment.json
model/config.json
model/model.safetensors
train.jsonl
valid.jsonl
```

The offline `tools/prepare_android_task.py` helper prepares this SAF folder from
local weights and already-tokenized data; it does not download or tokenize.
Use the tested Qwen seq64 five-update example in `tools/README.md`, including its
explicit `--pause-grace 300`. The generic specification below illustrates fields;
it is not evidence that the shown 200-update workload has passed emulator tests.
Bundle generation alone is not evidence of a successful system-picker import.
The November 17 emulator audit separately records actual DocumentsUI import and
export. Consult its evidence matrix for the model, step count and APK tested;
do not infer every model/configuration was exercised through the picker.

Example experiment.json (only listed keys are accepted):

```json
{
  "model": "qwen", "steps": 200, "batch_size": 1, "seq_len": 32,
  "lora_r": 2, "grad_accum": 1, "lr": 0.0002,
  "threads": 4, "slice_seconds": 60, "pause_grace_seconds": 30, "min_available_mb": 1024,
  "activation_checkpointing": false,
  "cpu_contention_guard": false,
  "memory_efficient_attention": false,
  "sharding": false, "shard_budget_mb": 768
}
```

`model` is `qwen` or `gemma`. Gemma assets must match the existing trainer's
supported config/weight layout. Data is pre-tokenized JSONL: each record has
`ids` and equal-length `mask`; prepare it with the matching model tokenizer.
Fixed session settings: frozen backbone, dropout 0, Q/V LoRA, alpha=2*rank,
cosine schedule, warmup2, seed42, shuffle, evaluation/save every3 updates,
one validation batch. This is a continuity interface, not a full benchmark UI.
`batch_size` is the original logical loader batch; effective batch remains
`batch_size * grad_accum` regardless of the physical cap.

Assets are copied once to private internal storage (not cache). New experiments
can use different configs, but existing runs cannot change their math/config.
Do not move assets or update the APK during an unfinished run: native executable
identity checks intentionally reject a different build. Exports omit frozen
model weights and retain logs/state; they are not cross-device resume bundles.

## Lifecycle and limits

- One native training child per app, enforced by a process-shared execution lock.
  Each native run also retains its own writer lock.
- Automatic continuation uses normal finite WorkManager jobs, with charging,
  battery and storage constraints. It may be delayed by Android/Doze/quotas.
  The initial persisted job has a 5-second delay so the visible foreground slice
  can acquire the execution lease first. A normal retry has linear backoff.
- During a slice the supervisor checks system memory via Android
  [ActivityManager.MemoryInfo](https://developer.android.com/reference/android/app/ActivityManager.MemoryInfo).
  No foreground app identity, usage-access permission, CPU attribution or game
  detection is used. A CPU/GPU-heavy game with plenty of available memory need
  not trigger this policy; it also reacts if training itself consumes the reserve.
- Sizes are MiB. The runtime reserve is the largest of the task's
  `min_available_mb`, the Android low-memory threshold + 128, and one eighth
  of total RAM capped at 1024. A sample with `lowMemory` set or below either
  hard floor (task minimum or Android threshold) requests a hold immediately.
  Otherwise the available memory must stay below the reserve in successive
  polls spanning at least 1 second. Starts need a further 128 MiB of headroom.
  These are policy defaults, not measured thresholds for any particular game.
- A memory hold disables scheduled continuation and asks the native child to
  checkpoint and exit at a completed update. The UI distinguishes a pending
  stop from a released execution lease. After the other workload ends, press
  **Start / Resume** on the same task; select background continuation again if
  desired. Clearing pressure alone does not restart training. Starting with too
  little headroom retains the hold without launching another trainer.
- **Pause for CPU contention** is a per-task switch saved immediately for the
  next foreground or scheduled slice. It does not start training, clear a hold,
  change the active slice, or rewrite the experiment or its checkpoint. Without
  a saved choice, the imported `cpu_contention_guard` value applies (default
  false); switching it off also overrides an imported true value.
  This experimental guard samples only
  the live trainer leader's scheduler run time and runnable wait. With healthy
  RAM and other gates clear, sustained queue delay requests a cooperative
  checkpoint, disables automatic continuation, and requires explicit Resume.
  There is no CPU admission check before a child exists. Unreadable or missing
  scheduler stats do not block training. This measures contention experienced
  by the leader, not which app caused it, worker-wide load, or GPU contention.
  A bounded emulator CPU pilot is not a calibration for production phones or games.
- Charging, battery under 20%, severe thermal state and less than 128 MiB
  free storage retain the existing resource gates. None guarantees an orderly
  exit if Android kills the process before the next checkpoint.
- `slice_seconds` 1..120 includes model loading. `pause_grace_seconds` is an
  explicit runtime-only watchdog grace, 30..420 seconds, default **30**. Both can
  be set in the import spec and overridden in the UI for the next execution;
  changing them does not alter the saved training mathematics. Safe exit is
  cooperative; execution beyond slice+grace is killed and auto-run disabled,
  reported as failure rather than a successful pause. Maximum execution budget
  is 540s, with a bounded 550s wake lock including the 10s cleanup allowance,
  leaving headroom below a normal Worker budget; Android may still stop it sooner.
  Recovery requires a committed checkpoint; none is promised before initial save.
  Longer profiles require a full pilot including evaluation/save/cleanup before
  selecting explicit grace (for example 300s if the measured profile supports it).
  Default30 is not a claim that full models or every configuration fit. Increasing
  grace does not extend the native slice or remove its cooperative stop request.
  A normal slice with no new updates due to loading also disables automatic mode
  with an explicit error, rather than retrying forever. Resource/user pauses are
  not classified as this error.
- A bounded wake lock covers the slice, not indefinite phone keep-alive.
- Native child has Android parent-death SIGKILL: service/worker process death
  cannot leave orphan training. Crash recovery may replay uncommitted work.
- Android Task Manager Stop is detected via process exit history before background
  launch; automatic mode is disabled until explicit Resume. Force-stop requires
  reopening. Reboot may resume an opt-in constrained job after unlock.
  A private BOOT_COMPLETED receiver re-enqueues only enabled automatic tasks;
  it never starts a foreground service at boot or overrides a user pause.
- If killed before a readable initial checkpoint exists, the partial task fails
  closed; use the explicit new-task retry and retain the original diagnostics.
  Work done before a committed save is not represented as recoverable progress.
- Nonzero dropout, GPU, Full-FT, mid-update saves, Flower multi-client/server
  recovery and automatic external EdgeFlowerTune Logger integration are outside
  this implementation. Native per-session metrics are retained and exportable.

## Metrics

`checkpoint_committed.save_metrics`: serialization through durable LATEST
publication, before logging that event and pruning old payloads. Includes fsync.
`resume.state_restore_metrics`: read/validate/import LoRA and Adam state,
including attempted fallback; excludes initial backbone loading and final sampler
import. `resume_ready.metrics`: native session establishment through model/data
setup and all state/sampler restoration; excludes process launch and earlier CLI
parsing. It is not UI-click-to-ready latency.

Logical byte counts cover instrumented checkpoint helpers (JSON and tensors),
not arbitrary backbone reads; read/write calls are helper operations, not all
kernel syscalls. OS disk counters are process-wide deltas and can be zero with
cache or null when unavailable. RSS is sampled every2ms plus endpoints, a lower
bound on the instantaneous phase peak. Process HWM includes earlier phases.
Measurement overhead is included. No power measurement or zero-energy claim.

## Validation commands

Experiment JSON uses numbers for numeric fields, integral values for step/batch/
sequence/rank/accumulation/thread/budget fields, and real JSON `true`/`false` for
optimization flags. Strings such as `"on"`, integer flags such as `1`, fractional
steps and invalid provided values are rejected, not replaced with defaults.
An omitted optional field still uses its documented default. A Qwen session also
requires the complete mapped backbone tensor inventory; a partial weight export
fails before any resumable checkpoint is published.

```sh
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.mobilefinetuner.visualizer.continuity.ContinuityTest,com.mobilefinetuner.visualizer.continuity.UiContinuityTest,com.mobilefinetuner.visualizer.continuity.TaskFlowTest,com.mobilefinetuner.visualizer.continuity.SpecAdmissionTest
```

The class filter runs only a subset. For the controls suite, including the
opt-in pressure and CPU pilots, see [user controls validation](USER_CONTROLS_VALIDATION.md).
Full-model checks are separate; see the [single-emulator checks](VM_RELEASE_CHECK.md).

Use a dedicated emulator with no other active training tasks. Gradle's connected
test runner can uninstall test APKs on completion. For the standalone checks below,
install the app and `app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`
first. Do not run these lifecycle fixtures against a user's phone or valuable app data.

From the source root, dedicated emulator-only reboot test:

```sh
python3 tests/android_reboot.py --adb "$ANDROID_HOME/platform-tools/adb" \
  --serial emulator-5554 --output /absolute/new-reboot-evidence
```

Tests retain raw native sessions in app internal files. Android emulator results
do not establish Huawei/vivo OEM lifecycle behavior or physical power-cut safety.
`tests/android_cold_start.py` stages a full Qwen model explicitly, tests a one-second
load budget, then resumes with 120 seconds. `PretrainedParityCheck` is a separate
explicit instrumentation class requiring those staged assets; it checks 10=4+6
with full Qwen weights at batch1/seq4/rank2. Neither runs in the regular tiny suite.

Official lifecycle references:
[FGS types](https://developer.android.com/develop/background-work/services/fgs/service-types),
[user stopping](https://developer.android.com/develop/background-work/services/fgs/handle-user-stopping),
[WorkManager limits](https://developer.android.com/develop/background-work/background-tasks/persistent/how-to/long-running),
[native executable restrictions](https://developer.android.com/about/versions/10/behavior-changes-10#execute-permission).
