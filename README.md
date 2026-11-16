# MatSciBench · Qwen3.5-4B 材料科学推理评测

本项目以 Qwen3.5-4B 为底座，在 MatSciBench 固定的 120 道材料科学数值题上比较直接回答、分步解答、自一致性投票、领域微调与候选验证。我们用 1,072 道带解题步骤和题族权重的材料题训练了 D3 LoRA 适配器，并让同一底座担任匿名裁判。裁判结合两种顺序的盲评、物理公式和本地单位计算，在固定题集复测中答对 **85/120（70.83%）**。

下载项目后，用浏览器打开 [课堂演示页](demo/index.html)。点选任一救回题，即可并排阅读底座和 D3 的原回答、查看两次盲评记录与复算依据。逐题视图覆盖全部 120 题，五个组可以自由组合对照，Self-Consistency 保留了每题五次采样的全文。

## 五组实验

| 方法 | 正确数 | 正确率 | 配置 |
|---|---:|---:|---|
| Direct Answer | 16/120 | 13.33% | 底座直接提交最终数值 |
| 底座普通 CoT | 69/120 | 57.50% | 给出公式、代入过程与最终数值 |
| CoT + Self-Consistency | 77/120 | 64.17% | 温度 0.6，每题 5 次采样，按数值聚类投票 |
| D3 普通 CoT | 66/120 | 55.00% | checkpoint-134，LoRA scale 0.25 |
| **底座 + D3 盲评与物理验证** | **85/120** | **70.83%** | 从两份普通 CoT 原回答中选择一份 |

底座与 D3 共同答对 50 题，另有 19 题仅底座答对、16 题仅 D3 答对。最终选择器保住了底座的 69 道正确题，并选中了 D3 独自答对的全部 16 题。结果达到两份候选在该题集上的正确题并集。Self-Consistency 与最终系统也有互补性，15 题仅最终系统答对，7 题仅 Self-Consistency 答对。

## 最终选择器怎样工作

1. 裁判读取题干和匿名候选 A、B，交换次序再评一次。两次选中同一份回答时采纳该回答，分歧时以底座为默认候选。双顺序盲评选回 **7 题**。
2. 检查模型是否正常结束，以及最后一行是否提供可解析的数值。底座缺少有效提交、D3 提交有效时采用 D3。这条规则再选回 **1 题**。
3. 验证器从题干独立提取物理关系和条件，本地工具处理单位与计数、复合材料表格的行列、半径与晶胞边长，以及工程量与真量等定义。对于点估计，一个候选距工具复算值不超过 5%、另一个偏差超过 10% 时，选择得到支持的候选。这些证据再选回 **8 题**。例如 T075 的 317.9 GPa 是混合模量，除以混合比重 16.18 后得到 19.648，支持 D3 提交的 19.6。
4. 选答完成后用参考答案评分。最终行数值的相对容差为 5%，零参考值使用绝对容差 10⁻¹²。分数对应这次固定题集上的复测，工具设计参考了此前观察到的错误类型。

## 复现结果与课堂页面

在仓库根目录运行以下 PowerShell 命令。离线复核和 HTML 构建使用 Python 3.11 或更新版本的标准库。

```powershell
python src/replay.py
python src/build_demo.py
Start-Process .\demo\index.html
```

第一条命令逐题重新解析直接回答、CoT 和 D3 的最终行，重投五个采样，使用保存的匿名裁判意见和物理证据重放选择，再核对五组分数。第二条命令将同一份记录嵌入单文件 HTML。原始题目、五组逐题成绩和冻结协议分别见 [测试集](data/test120.json)、[结果表](results/scores.csv) 和 [评测参数](results/protocol.json)。

## 重新生成模型回答

正式推理使用 llama.cpp 提交 `47a39665e7081dc482feec169961acc09750a5c4`、BF16 文本底座、随库的聊天模板、每次最多 16,384 个输出 token。以下命令以 Windows PowerShell 为例。先拉取 LFS 适配器并安装物理验证依赖。

```powershell
git lfs install --local
git lfs pull
python -m pip install -r requirements.txt
```

准备好该提交编译的 `llama-server.exe` 和 BF16 GGUF 后，在第一个终端启动推理服务。将前两行改为实际文件位置。

```powershell
$serverExe = 'F:\llama.cpp\build\bin\Release\llama-server.exe'
$baseGGUF = 'F:\models\Qwen3.5-4B-text-bf16.gguf'
& $serverExe -m $baseGGUF --lora .\models\d3\d3-f32.gguf --lora-init-without-apply --jinja --chat-template-file .\models\chat_template.jinja --host 127.0.0.1 --port 8080 -ngl all -np 1 -c 32768 -fa on -ctk f16 -ctv f16 --no-context-shift
```

在第二个终端运行单题或全量评测。脚本核对底座与适配器 SHA-256，每个模型请求和每题成绩写入 `runs/evaluation/`；重复执行同一命令会复用已完成的请求。单次评测会生成两种普通 CoT 候选、直接回答和五个采样，并执行双顺序盲评和物理复算。

```powershell
python src/evaluate_local.py --question 75
python src/evaluate_local.py
```

从原始 Hugging Face 权重准备底座时，使用上述 llama.cpp 版本的转换器；`$llama` 是该源码目录，`$baseHF` 是下载后的 Qwen3.5-4B 模型目录。`results/protocol.json` 记录本次实验所用 GGUF 的完整 SHA-256。

```powershell
$llama = 'F:\llama.cpp'
$baseHF = 'F:\Qwen3.5-4B'
python -m pip install -r "$llama\requirements\requirements-convert_hf_to_gguf.txt"
python "$llama\convert_hf_to_gguf.py" $baseHF --outtype bf16 --no-mtp --outfile $baseGGUF
```

## 复现 D3 微调

从原始 Qwen3.5-4B 开始，用 NF4 双重量化、BF16 计算训练语言层 LoRA。训练用 1 个 epoch、峰值学习率 5e-5、rank 16、alpha 32、dropout 0.05、单卡 microbatch 1、梯度累积 8、最长 1,024 tokens。训练损失按题目权重归一化，权重作用于 assistant 答案 token。训练顺序、随机种子、数据哈希和软件版本保存在 [训练配置](finetuning/config.json)。外部 100 题用于数值 token 的验证损失。

在 Windows、支持 BF16 的 NVIDIA GPU 和独立 Python 环境中执行：

```powershell
python -m pip install -r finetuning/requirements.txt
python finetuning/train.py --base 'F:\Qwen3.5-4B'
```

训练在第 67、134 步保存检查点。课堂组使用第 134 步；权重提供 PEFT 和 llama.cpp GGUF 两种格式。要将新训练的第 134 步检查点转为 GGUF，可运行：

```powershell
python -m pip install -r "$llama\requirements\requirements-convert_lora_to_gguf.txt"
python finetuning/export_adapter.py --llama-cpp $llama --base $baseHF --checkpoint finetuning/runs/checkpoint-134 --out finetuning/runs/d3-f32.gguf
```

## 文件与来源

| 目录 | 内容 |
|---|---|
| `demo/` | 可离线打开的课堂演示页 |
| `data/` | 固定 120 题，含题目、单位、参考答案和来源 |
| `results/` | 逐题 CSV、冻结参数和压缩的完整回答与证据 |
| `src/` | 页面模板、重建脚本、评分、投票、盲评、物理计算与选答逻辑 |
| `finetuning/` | D3 训练与导出脚本、依赖、1,072 道训练题与 100 道外部验证题 |
| `models/` | D3 适配器、权重哈希、基础模型许可与聊天模板 |

题目来自 Zhang 等人的 MatSciBench。训练题的题目来源、答案来源、解题步骤和课程权重保存在训练 JSONL 的对应字段。[MatSciBench 论文](https://arxiv.org/abs/2510.12171) · [基准数据](https://huggingface.co/datasets/JunkaiZ/MatSciBench) · [Qwen3.5-4B 底座](https://huggingface.co/Qwen/Qwen3.5-4B) · [实验使用的 llama.cpp 提交](https://github.com/ggml-org/llama.cpp/commit/47a39665e7081dc482feec169961acc09750a5c4)

