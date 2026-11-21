# Android checkpoint and resume example

This standalone arm64 app (`com.mobilefinetuner.continuity`, API 29+) starts and
pauses on-device Qwen and Gemma training. Follow the [build and install
instructions](../ANDROID_CONTINUITY.md#build-and-install) from the example root
before running Gradle. CMake produces the two trainers and their JNI process
owner; `python3 tools/make_android_fixture.py` produces a deterministic random
Qwen model and synthetic rows, not pretrained weights or benchmark data.
Neither group of generated files belongs in source control.

For JVM and emulator checks see [user controls validation](../USER_CONTROLS_VALIDATION.md).
The small sample logs and RSS CSV under `app/src/test/resources/samples/` are
JVM parser test inputs, not app assets. The optional in-app log viewer uses
the bundled fonts; their OFL notices ship under `app/src/main/assets/licenses/`.
