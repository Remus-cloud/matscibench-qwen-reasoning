# B 正式评测与项目合并记录

B 已完成 Qwen3.5-27B 底座和现有 V3 LoRA 的两组正式评测，每组覆盖固定 120 道测试题。本次合并接入实际运行代码、材料与环境记录及逐题输出。本地重新评分得到相同准确率。V3 在该轮 NF4 环境下的正确率降低，原因需要结合逐题回答继续分析。

## 结果

| 指标 | 底座 CoT | V3 LoRA CoT |
|---|---:|---:|
| 已回答 | 120 | 120 |
| 正确 | 87 | 44 |
| 准确率 | 72.50% | 36.67% |
| 数值错误 | 30 | 65 |
| 格式错误 | 0 | 5 |
| 截断 | 3 | 6 |
| 未知结束 | 0 | 0 |
| 输出 token | 220,410 | 113,465 |
| 生成总耗时 秒 | 15,659.20 | 14,291.46 |
| 每题中位耗时 秒 | 63.13 | 5.88 |

3 题从错变对，46 题从对变错，41 题均正确，30 题均错误。净减少 43 道正确题，准确率下降 35.83 个百分点。两组共享编译缓存，耗时差异不能全部解释为模型速度差异。

## 环境与方法

B 记录的实际环境为 Python 3.12.3、PyTorch 2.8.0+cu128、CUDA 12.8、Transformers 5.22.2、PEFT 0.19.1、bitsandbytes 0.50.2。硬件为单张 RTX PRO 6000 Blackwell Server Edition 96 GB，25 vCPU，120 GB 系统内存。

两组均采用 NF4 双重量化、BF16、SDPA、batch 1、seed 6020、greedy、关闭原生 thinking、每题最多 16,384 个输出 token。题集、CoT 提示和评分函数与本地原项目逐字节相同。微调组加载时校验了 992 个适配器张量。现有 V3 权重沿用原作者训练产物。

历史 API 成绩仍保存在原来的 results/summary.json 和 results/questions.jsonl。本轮对照单独保存在 results/b_eval。报告比较微调贡献时使用本轮底座 87/120。

## 文件与复算

- src/run_27b.py 接入 B 的实际执行版，并调整为从项目内读取材料清单，支持 --materials 指定其他清单。
- src/compare_b.py 检查两组配置、加载证据、题集和提示哈希，调用原评分函数，生成汇总与逐题 CSV。
- results/b_eval 保存两组 JSONL、manifest、events、B 原始汇总、逐题 CSV 和 pair_audit.json。
- provenance/b_eval 保留 B 交付的实际执行源码、材料核验清单、冻结参数和环境清单。

```sh
python src/compare_b.py results/b_eval/base_results.jsonl results/b_eval/lora_results.jsonl --output-dir results/b_eval/recomputed
```

离线复算只依赖 Python 标准库。输出 summary.json 和 paired_results.csv。B 原始文件与本地复算文件分目录保存。

## 来源和适配边界

说明来源为用户提供的 01_B说明.docx 和实际运行代码目录。逐题结果来源为同目录下、B 使用说明明确引用的 02_交给C.zip，仅接入与配对复算有关的结果文件。训练题集和其他组员交付文件未纳入本次合并。

B 原始 run_27b.py 依赖项目父目录 logs/materials_verified.json，并要求当前 Git HEAD 等于原固定提交。集成版将材料路径改为项目内 provenance/b_eval/provenance/materials_verified.json，记录当前提交及材料原始提交，允许后续项目提交后继续运行。逐题 manifest 仍保存数据、提示、运行脚本和材料的哈希，续跑要求相同配置。

集成版同时以 UTF-8 读写文本，并使用流式 SHA-256 读取模型权重以降低核验内存占用。这些适配会改变运行脚本哈希。历史结果的 manifest 保持原样，原始执行版副本与 frozen_protocol.json 可用于核对历史脚本哈希。

原始 environment.txt 包含云机本地路径，作为环境证据保留。新环境按 V3_GPU_RUN.md 配置。此次核查完成离线复算和命令入口验证，尚未在新云机重新生成模型回答。
