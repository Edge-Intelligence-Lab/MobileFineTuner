# User controls: single-emulator release checks

This is the independent continuity entry point, not an upgrade to every historical trainer or SDK API. The scope is same-build CPU FP32, frozen-backbone LoRA, dropout zero. We do not guarantee foreground FPS, realtime release deadlines, arbitrary-device behavior or unchanged task accuracy when the physical batch changes.

| Check | Observation |
| --- | --- |
| Graceful vs immediate | Pause requests a commit at an optimizer boundary. Release now and Focus for one hour can also stop a non-cooperating child. With a deliberately stopped tiny Qwen child, both recovered to 100 updates, with checkpoint tensors, optimizer metadata and sampler identical to continuous training. Work after the last published checkpoint can be lost. |
| Foreground memory pressure | On API 35 arm64, a separate foreground process held 320 MiB. With a test-adjusted reserve, tiny Qwen paused at update 23. The user then selected physical cap 3 (original batch 4) and completed 256 updates. Final trainable tensors, Adam state and sampler matched the same cap schedule without the competing process. Releasing RAM alone did not restart training. The exact stopping step depends on scheduling. |
| CPU-only pilot | Optional. With sufficient RAM, two competing workloads pinned to the trainer's vCPU increased the leader's runnable-wait share from 1.6% to 92.3% in the final run. Sustained contention requested a cooperative hold. This is scheduler evidence, not game detection or FPS data. |
| Recovery/UI | Bad current checkpoint falls back to its recorded predecessor. If neither is usable due to verified corruption, Retry as new preserves the old task. Transient read failures do not authorize a reset. Generation-specific controls prevent an old request from re-enabling paused work. |
| Physical batch | Any integer cap from 1 to the original logical loader batch, not a new optimizer batch size. Example: cap 3 partitions 8 rows as 3+3+2. Same-schedule full Gemma270M host control matched across an extra restart (216 trainable/Adam tensors); versus a different cap schedule, eight tensor-by-update comparisons missed the pre-existing gate (maximum absolute discrepancy 5.33e-7). No tolerance was relaxed. |

## Reproduce the Android controls

Build the trainers and fixture using [Android build instructions](ANDROID_CONTINUITY.md#build-and-install). Use a disposable arm64 Android 15 emulator with sufficient RAM. In `android/`:

```sh
./gradlew :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -r -e cpuPilot true -e pressureE2E true \
  com.mobilefinetuner.continuity.test/androidx.test.runner.AndroidJUnitRunner
adb logcat -d -s MftPressureE2E:I MftCpuPilot:I
```

Final suite: 80 executed passes, 4 dedicated setup/full-model tests skipped, 0 failures. Count instrumentation status code `0` for each pass, not the runner's total of 84. Local JVM tests: 28 passes. Neither set substitutes for testing real hardware. The installed APK is a debug build; do not distribute it as a production-signed release.
