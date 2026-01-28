# Legacy NN Module Files

This directory keeps several early NN implementations (`lora.h/cpp`, `lora_ops.h/cpp`, `attention.h/cpp`, `mlp.h/cpp`, `module.h`, `layers.h`, etc.). They still depend on the old Tensor interface and missing types like `BaseLayer`/`LayerConfig`/`GradTensorPtr`, and are provided for reference only. Building them directly will fail due to interface mismatches.

## Current status
- ✅ `nn/lora_linear.h/cpp`: migrated to modern TensorPtr/ops API, in active use (under `finetune_ops/nn/`)
- ✅ `nn/embedding.h/cpp`: migrated to modern API, kept in `finetune_ops/nn/`
- ⚠️ `legacy/nn/lora.h/cpp`: depends on removed BaseLayer/LayerConfig
- ⚠️ `legacy/nn/lora_ops.h/cpp`: uses legacy `.data/.shape` Tensor API
- ⚠️ `legacy/nn/attention.h/cpp`, `legacy/nn/mlp.h/cpp`, `legacy/nn/module.h`, `legacy/nn/layers.h`: examples of the old interface
- ⚠️ `legacy/gpt2_finetune_model.h`: depends on GradTensorPtr/Tokenizer and other old interfaces

## Build configuration
These legacy files are excluded from FINETUNE_SOURCES in CMakeLists.txt and are not compiled.

## Current alternatives
- LoRA: use `nn/lora_linear.h` + `graph/lora_injector.h`
- Attention: inlined in `graph/gpt2_model.cpp`
- Embedding: use `nn/embedding.h` (migrated) or gpt2_model's embedding_lookup

## Enabling the old files
They must be migrated to the modern Tensor/ops interface or backed by the missing base-layer framework first.
