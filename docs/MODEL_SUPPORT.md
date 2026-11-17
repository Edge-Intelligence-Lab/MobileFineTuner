# 模型与优化支持范围

本页描述当前维护的**纯文本 LoRA 训练**接口。模型家族按已实现的架构识别，
不能根据名称包含 GPT、Gemma 或 Qwen 就推断兼容。模型大小可由配置描述，
但没有逐个验收的 checkpoint 不列为已验证型号。

## 已覆盖的模型

| 模型 | 配置类型 | 图实现 |
| --- | --- | --- |
| GPT-2 small | `gpt2` | GPT2Model |
| GPT-2 medium | `gpt2` | GPT2Model |
| Gemma 3 270M | `gemma3_text` | GemmaModel |
| Gemma 3 1B-PT | `gemma3_text` | GemmaModel |
| Qwen2.5-0.5B | `qwen2` | QwenModel，dense GQA |
| Qwen3-0.6B | `qwen3` | QwenModel，独立 head_dim、Q/K norm |
| Qwen3.5-0.8B 文本部分 | `qwen3_5_text`，或 `qwen3_5.text_config` | QwenModel，Gated DeltaNet + gated GQA |

Gemma 2、Gemma 多模态 wrapper、未实现的其他 Gemma 架构及 Qwen MoE 等不会
静默落入现有 decoder。Qwen3.5 wrapper 只取语言模型配置与权重；不处理图片或视频。

## 优化能力矩阵

下表针对新 autograd 引擎的 C++ 模型图和原生训练入口。
“支持”表示已实现并有回归覆盖；实权重、长跑和真机的验收层级见后面的验证记录。
开关默认关闭或沿用对应示例的既有默认值，不自动根据模型名启用。

| 模型 | 流式 Attention | 激活重算 | 冻结参数磁盘卸载 | 原生 F16/BF16 基座矩阵 | 完整 Q8_0 CLI 工作流 |
| --- | --- | --- | --- | --- | --- |
| GPT-2 small | 支持 | 未接入 | 未接入 | 当前完整入口使用 FP32 | 支持 |
| GPT-2 medium | 支持 | 未接入 | 未接入 | 当前完整入口使用 FP32 | 未接入专用 CLI |
| Gemma 3 270M / 1B-PT | 支持 | 支持 | 支持，FP32 | 当前完整入口使用 FP32 | 未接入 |
| Qwen2.5-0.5B | 支持 | 支持 | 支持，FP32 | 支持 | 未接入 |
| Qwen3-0.6B | 支持 | 支持 | 支持，FP32 | 支持 | 未接入 |
| Qwen3.5-0.8B 文本 | 支持其 full-attention 层 | 支持完整 decoder block | 支持，包含两类 mixer 与 MLP | 支持 | 未接入 |

Qwen3.5 的流式 Attention 不改变 DeltaNet 递推算法。激活重算覆盖完整 block，
会在 backward 重建 DeltaNet 状态；它不是增量推理缓存。卸载清单来自与加载校验
相同的权重清单，覆盖卷积、衰减参数、门控 norm、投影和共享/独立输出头。

约束：

- 重算要求 LoRA dropout 为零；重算不会保存或回放随机数状态。
- 冻结参数卸载要求已加载、已冻结的 CPU FP32 权重，预算必须容纳最大的完整
  block 或 embedding/head 组。Qwen CLI 必须指定 `--base_weight_storage fp32`。
- 原生 F16/BF16 只保留 Qwen 的冻结矩阵；norm、bias、卷积、激活、LoRA 与梯度保持 FP32。
  这不是 BF16 全程训练，也不是 Q8/QLoRA。
- 卸载预算是模型管理的权重驻留量，不是进程 RSS 上限。保存的 backward 张量、
  外部别名与分配器缓存仍可能保留内存，不承诺固定的内存节省或加速比例。
- Q8_0 的完整量化、导出、重载、训练入口目前在 GPT-2 small 示例；共享量化算子
  可复用，不代表其他模型的专用 CLI 已完成量化接入。

## 其他能力的边界

| 能力 | 当前范围 |
| --- | --- |
| 梯度累积、FP32 Adam、梯度裁剪 | 七个型号的维护入口；Qwen/Gemma 尾组按实际批次数处理 |
| 标准 PEFT 两文件 adapter | Qwen2 / Qwen3 / Qwen3.5 文本；GPT-2/Gemma 使用 MF 自有 adapter 合约 |
| adapter 保存/恢复 | 保存权重与配置；不恢复 Adam moments、累计步数或 dropout RNG |
| 流式输出头交叉熵 | Qwen WikiText 的 `--loss_impl full_streaming`；覆盖与重算/卸载组合，其他入口未接入该开关 |
| 能耗调度 | 公共 PowerMonitor 可复用；训练 CLI 的延迟调度集成在 Qwen WikiText |
| 评估模式 | 统一 GradModeGuard，关闭梯度记录和 LoRA dropout，退出时恢复原状态 |
| Full-FT | GPT-2 small 有专用示例；本轮跨模型统一验收为 LoRA，其他型号不据此声明完整 Full-FT 工作流 |
| GPU/MPS/CUDA、分布式 ZeRO、优化器状态卸载 | 不属于本次已验收能力；当前执行为 CPU，可选 BLAS |
| 多模态、MoE、增量 KV/卷积/递推状态缓存 | 未实现 |

Android AAR 暴露基础 AutoModel/AutoTrainer。高级内存开关目前通过 C++ 或原生 CLI
配置，不能据此认为 Java 层已暴露同样的控制面。Android 编译与真机验收分开记录。

## 统一能力查询

```cpp
#include "mobile_finetuner/mobile_finetuner.h"
auto spec = ops::ModelRegistry::inspect_pretrained(model_dir);
auto capability = spec.capabilities;
// causal_lm_lora, streamed_attention, activation_checkpointing,
// frozen_parameter_offload, standard_peft_adapter_io
```

能力来自库的实际引擎构建，而不是客户端预处理宏。没有新引擎时，未经该引擎验收的
内存能力不会被声明为可用。能力描述不等同于当前已启用，也不是设备内存可行性保证。

## 验证证据

- [模型与优化组合验收](validation/OPTIMIZATIONS_20261117.md)
- [Gemma 实权重回归](validation/GEMMA_20261117.md)
- [Qwen 支持与 adapter 合约](QWEN_DENSE_SUPPORT.md)
- [内存优化约束](MEMORY_OPTIMIZATIONS.md)
- [Mac 长跑方法与状态范围](MAC_STABILITY.md)
