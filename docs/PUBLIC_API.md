# MobileFineTuner Public API

MobileFineTuner exposes one supported C++ entrypoint:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"
```

Downstream projects should link the installed CMake target:

```cmake
find_package(MobileFineTuner REQUIRED)
target_link_libraries(your_target PRIVATE MobileFineTuner::operators)
```

The `finetune_ops/*` headers are still installed because the current examples
and research CLIs use them directly, but new applications should include the
umbrella header first. Historical or non-buildable APIs have been archived under
`Rubbish/` and are not part of the supported product surface.

Model weights and datasets are external assets. The library accepts standard
HuggingFace-style model directories with config/tokenizer files and either a
single `model.safetensors` file or `model.safetensors.index.json` plus all
referenced shard files. See `docs/MODEL_ASSETS.md` for the asset contract.
