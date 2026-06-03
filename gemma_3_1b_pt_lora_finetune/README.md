以下脚本封装了从数据准备到开始 LoRA 微调与评测的最小可运行流程（基于 Gemma 3 1B PT 权重）。

要求：
- C++17、CMake，以及可用的本机或 Android NDK 编译环境
- Python 仅用于离线数据准备脚本；原生 LoRA 训练不依赖 PyTorch/Transformers/PEFT
- 数据集路径通过 `MFT_DATA_ROOT`、`MMLU_DATA_DIR`、`WT2_DATA_DIR` 或仓库本地 `data/` fallback 提供
- 预训练权重通过 `MFT_MODEL_ROOT`、`GEMMA_1B_MODEL_DIR` 或当前目录 `pretrained/` fallback 提供

使用顺序：
1) 生成 JSONL（masked 标签，仅答案位置参与损失）
   bash run_prepare_data.sh
   输出目录：仓库根目录 `runs/mmlu_jsonl_gemma1b_s128/`

2) 启动微调（LoRA）
   bash run_train.sh
   产物：当前目录 `outputs/`

3) 评测（MMLU MCQ）
   # DEV 验证
   bash run_eval.sh
   # TEST（可选 few-shot，默认 5-shot）
   FEWSHOT=5 SPLIT=test bash run_eval.sh

可调参数：
- 在 run_train.sh 中通过环境变量覆盖：TRAIN_MODE（`mmlu`/`wt2`）、MMLU_JSONL、WT2_DATA_DIR、STEPS、BATCH_SIZE、SEQ_LEN、LR、RANK、ALPHA、EPOCHS、LOG_EVERY、GRAD_ACCUM
- 在 run_eval.sh 中通过环境变量覆盖：SPLIT、FEWSHOT
