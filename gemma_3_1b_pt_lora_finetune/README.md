以下脚本封装了从数据准备到开始 LoRA 微调与评测的最小可运行流程（基于 Gemma 3 1B PT 权重）。

要求：
- 已安装 Python 依赖（pytorch、transformers、peft、safetensors 等）
- 本仓库根目录含有数据集 `data/mmlu/`
- 预训练权重：`gemma-3-1b-pt/`

使用顺序：
1) 生成 JSONL（masked 标签，仅答案位置参与损失）
   bash run_prepare_data.sh
   输出目录：仓库根目录 `runs/mmlu_jsonl_gemma1b_s128/`

2) 启动微调（LoRA）
   bash run_train.sh
   产物：当前目录 `outputs/lora_adapter/`

3) 评测（MMLU MCQ）
   # DEV 验证
   bash run_eval.sh
   # TEST（可选 few-shot，默认 5-shot）
   FEWSHOT=5 SPLIT=test bash run_eval.sh

可调参数：
- 在 run_train.sh 中通过环境变量覆盖：STEPS、BATCH_SIZE、SEQ_LEN、LR、RANK、ALPHA、EPOCHS、LOG_EVERY、GRAD_ACCUM
- 在 run_eval.sh 中通过环境变量覆盖：SPLIT、FEWSHOT、DEVICE


