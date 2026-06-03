# Android SDK / AAR Packaging

MobileFineTuner ships an Android Library module at:

```text
android-visualizer/mft-sdk
```

The module builds an AAR containing:

- `com.mobilefinetuner.sdk.MobileFineTuner`: Java API for Android apps.
- `libmobilefinetuner_jni.so`: JNI bridge into the native C++ core.
- The compiled MF operator/autograd/model/trainer implementation.

The AAR does not bundle pretrained weights, datasets, adapters, or run logs.
Applications provide HuggingFace-style model directories at runtime, matching
the desktop C++ asset contract in `docs/MODEL_ASSETS.md`.

## Build

Requirements:

- JDK 17+
- Android SDK API 35
- Android NDK `26.1.10909125`
- CMake `3.22.1`

Build the release AAR:

```bash
bash scripts/android/build_mft_sdk_aar.sh
```

Output:

```text
android-visualizer/mft-sdk/build/outputs/aar/mft-sdk-release.aar
```

You can also build from the Android project root:

```bash
cd android-visualizer
./gradlew :mft-sdk:assembleRelease
```

## Supported ABI

The SDK currently packages `arm64-v8a` only. This matches the phone-training
target and avoids shipping unvalidated ABI variants.

## Runtime Asset Layout

Push or download model assets into app-readable storage, for example:

```text
/sdcard/Android/data/<your.app.id>/files/models/Qwen2.5-0.5B/
  config.json
  tokenizer.json / tokenizer.model / vocab files
  model.safetensors
  or model.safetensors.index.json + shard files
```

The model directory must be readable by the app process. For production apps,
copy user-selected files into app-private storage before opening the model.

## Java API

```java
import com.mobilefinetuner.sdk.MobileFineTuner;

try (MobileFineTuner mf = MobileFineTuner.open(modelDir, true)) {
    mf.initLora(MobileFineTuner.LoraConfig.attentionQkvo());
    mf.createTrainer(MobileFineTuner.TrainerConfig.defaults());

    MobileFineTuner.TrainStepResult result = mf.trainStep(
            inputIds,
            attentionMask,
            labels,
            batchSize,
            sequenceLength
    );

    float loss = result.loss;
}
```

Call order:

1. `MobileFineTuner.open(modelDir, loadWeights)`
2. `initLora(...)`
3. `createTrainer(...)`
4. `trainStep(...)`
5. `close()`

`inputIds`, `attentionMask`, and `labels` are flattened row-major arrays with
length `batchSize * sequenceLength`. Labels use the same shifted causal-LM
contract as the C++ `AutoTrainer`: ignored positions should be `-100`.

## Native Boundary

The JNI bridge is intentionally thin:

```text
Java MobileFineTuner
        |
        v
libmobilefinetuner_jni.so
        |
        v
ops::AutoModelForCausalLM + ops::AutoTrainer
```

The Android layer does not duplicate model math. All forward, loss, backward,
gradient clipping, Adam update, and zero-grad behavior comes from the native C++
core.

## Current Limitations

- `arm64-v8a` only.
- Java API currently exposes the one-step LoRA training core, not a full
  dataset loop or checkpoint manager.
- Model files are loaded from normal filesystem paths; the SDK does not load
  directly from compressed APK assets.
- Full-weight phone training still depends on device memory limits and the
  model asset dtype/shape.

These are packaging boundaries, not algorithmic shortcuts. The training step is
the same native MF step used by desktop and adb-driven Android experiments.
