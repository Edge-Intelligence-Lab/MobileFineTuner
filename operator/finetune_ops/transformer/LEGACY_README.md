# Legacy Transformer Files

## File status
- ✅ `gpt2_model.h/cpp`: modern implementation, in use
- ✅ `lora_injector.h/cpp`: modern implementation, in use
- ✅ `lora_saver.h/cpp`: modern implementation, in use
- ✅ `safetensors_loader.h/cpp`: modern implementation, in use
- ⚠️ `legacy/gpt2_finetune_model.h`: legacy file depending on GradTensorPtr/grad_tensor.h (missing)

## Current approach
Use `gpt2_model.h` + `lora_injector.h` + `optim/trainer.h` to run the full training flow.

## Build configuration
Legacy headers have been moved to `finetune_ops/legacy/` and are not referenced by current targets.
