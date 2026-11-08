# 实验说明

本项目比较同一百炼模型 `qwen3.5-27b` 在材料科学数值题上的四种答题策略。所有正式结果均来自同一批 120 道题；每题保存原始回答和解析结果，可在课堂页按题号复核。

## 题目与评分

选题来自 MatSciBench 官方发布的 `test` 分区。项目以种子 6020 自行划出互不重叠的开发集和测试集。题目必须是无需图像的数值题、具有唯一可解析的数值答案；排除依赖外部图表、前题或补充材料的题目、已用过的演示题，并进行题干近重复筛查。开发集难度配额为易 8、中 7、难 5；测试集为易 48、中 42、难 30。这些结果只代表该数值子集。[MatSciBench 数据](https://huggingface.co/datasets/JunkaiZ/MatSciBench) · [上游评测仓库](https://github.com/Jun-Kai-Zhang/MatSciBench)

每题均使用指定单位。输出须正常结束，且末行符合 `FINAL = <number>`。允许有符号十进制数与科学计数法；参考值非零时相对误差容限为 5%，参考值为零时绝对容限为 1e-12。格式错误和截断均计错。

## 四组正式结果

Direct 要求只给最终答案。普通 CoT 要求写出关系式、带单位代入和计算过程。两者及计算器组的温度均为 0。所有组关闭原生 thinking，通过提示控制可见的解题说明。前三组每题共享 16,384 个输出 token；五票组为五份回答各设 16,384 上限。

| 与普通 CoT 配对 | 对照策略独对 | 普通 CoT 独对 | 都对 | 都错 |
|---|---:|---:|---:|---:|
| Direct Answer | 4 | 66 | 28 | 22 |
| CoT + 条件计算器 | 8 | 7 | 87 | 18 |
| CoT + Self-Consistency | 5 | 6 | 88 | 21 |

从 Direct 到普通 CoT 净增 62 题，是本次最大的变化。第 1 题中，Direct 答 0.0385 V，超出 0.0414 V 的容差；普通 CoT 写出浓差电池关系式后答对。

计算器只在模型给出明确公式、可追溯的输入，且确有多步计算或单位换算需要时执行。120 题中有 4 题实际执行；这 4 题的首轮与最终答案均正确。与普通 CoT 相比分歧的 15 题都没有执行计算器，因此多出的 1 题无法单独归因于计算工具。第 72 题可用于展示数值复核；第 23 和 94 题展示最终格式错误。

五票组在温度 0.6 下独立作答五次。相对差不超过 1% 的数值可归入同一组，优先票数最多的一组，再按跨度和原始票号打破平局，取该组实际出现过的中位数。没有一致对时采用首个有效答案。投票阶段不读取标准答案。在同温度的控制中，首票单独答对 95 题，投票后为 93 题；在本次样本中，五票改变了个别题的答案，总成绩没有超过普通 CoT。

## 微调训练与权重

我们使用同一基准的其他题目进行 QLoRA 训练，独立复算数值答案，检查教材母题、替换数值的相似题和解题家族。训练集含 120 条样本，内部验证集含 26 条样本，两者与课堂开发集和正式测试集的题号没有交集。[训练数据](../finetuning/train.jsonl) · [验证数据](../finetuning/val.jsonl)

底座为 `Qwen/Qwen3.5-27B`，训练使用的 ModelScope revision 为 `8ee5435f4c700c6d8d4bb69efbf2e60c46b3e23e`。我们在语言层上训练适配器，采用 NF4 双重量化、BF16 计算、LoRA rank 8、alpha 16、dropout 0.05 和学习率 3e-5。序列长度上限为 2048，单卡 batch 1，梯度累积 4 次。训练 1 个 epoch，共 30 次优化更新。[Qwen 官方模型](https://huggingface.co/Qwen/Qwen3.5-27B)

原训练环境使用 Linux、Python 3.11 和 PyTorch 2.8.0 + CUDA 12.8。在兼容的 GPU 环境中，从仓库根目录执行以下命令，并替换为底座的本地路径和一个新的输出目录。

```bash
python -m pip install -r finetuning/requirements.txt
bash finetuning/train.sh /path/to/Qwen3.5-27B /path/to/new-run
```

[权重目录](../models/lora-adapter/) 包含 PEFT 适配器权重和配置。先载入上述 revision 的底座，再从仓库根目录用 `PeftModel.from_pretrained(base_model, "models/lora-adapter")` 挂载适配器。权重的 SHA-256 为 `5c0b0d93497dabba06a2ae223baef95d984a2c603b3de0620ed178762a507d49`。克隆后可用 `git lfs pull` 获取权重。[Git LFS 说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)

微调模型尚无完整 120 道测试题的准确率。
