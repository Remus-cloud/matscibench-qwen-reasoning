# V3 云机运行说明

B 已完成两组正式评测，已有记录可直接离线复算，见 [B 评测说明](B_EVALUATION.md)。下面用于重新生成回答。B 原始云机实际使用 Python 3.12.3，本说明提供 Python 3.11 的新环境配置。

在 Linux 云机上运行，使用 Python 3.11、支持 BF16 的 NVIDIA GPU 和带 CUDA 的 PyTorch。脚本把模型放在单张卡上，建议用一张 48 GB 显存的卡留出加载和生成余量。模型下载约 56 GB，建议预留至少 80 GB 磁盘空间。云机镜像可选已安装 CUDA 版 PyTorch 的 Python 3.11 镜像。

把项目放到云机，在项目根目录执行下面的命令。虚拟环境沿用云机镜像提供的 PyTorch。

```bash
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install 'transformers==5.22.2' 'huggingface-hub==1.33.0' 'peft==0.19.1' 'bitsandbytes==0.50.2' accelerate modelscope
python -m pip check
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

下载训练时使用的底座版本。已有同版本完整目录时，直接将 `BASE` 改为该路径。

```bash
export BASE="$HOME/model-weights/Qwen3.5-27B"
python -c 'import os; from modelscope import snapshot_download; snapshot_download("Qwen/Qwen3.5-27B", revision="8ee5435f4c700c6d8d4bb69efbf2e60c46b3e23e", local_dir=os.environ["BASE"])'
sha256sum models/v3-checkpoint-30/adapter_model.safetensors
```

适配器的 SHA-256 应为 `5c0b0d93497dabba06a2ae223baef95d984a2c603b3de0620ed178762a507d49`。如果克隆得到的是 Git LFS 指针，在项目根目录执行 `git lfs pull`，然后重新校验。

两组按顺序运行，每次只加载一个模型。试跑均使用开发集第一题，复用项目的 CoT 提示，关闭原生 thinking，采用相同的 NF4 量化和生成设置。下面用 Bash 执行。

```bash
set -o pipefail
python src/run_27b.py --model "$BASE" --split dev --limit 1 \
  --output results/dev_base_smoke.jsonl 2>&1 | tee results/dev_base_smoke.log
python src/run_27b.py --model "$BASE" --adapter models/v3-checkpoint-30 \
  --split dev --limit 1 --output results/dev_lora_smoke.jsonl 2>&1 | tee results/dev_lora_smoke.log

python src/compare.py results/dev_base_smoke.jsonl --split dev
python src/compare.py results/dev_lora_smoke.jsonl --split dev
python -m pip freeze > results/v3_environment.txt
```

验收时，两份结果都应显示 `answered: 1`，题号相同。微调组日志应显示 `MODEL VERIFIED lora_cot matched LoRA tensors: 992`。检查回答是否正常结束，并保留原始输出。单题试跑的 `accuracy: null` 表示开发集还未跑完。复现 B 的五道开发题试跑时，将 `--limit 1` 换为 `--smoke`。

把两份 JSONL、相应的 manifest.json 和 events.jsonl、两份日志和环境清单交给负责评分的组员。每行 JSONL 保留题号、原始回答、停止原因、提示、Token ID、配置指纹和生成耗时。正式评测时将两条推理命令改成 `--split test`，去掉 `--limit 1`，另存为两份测试集结果。用下面的命令完成本轮两组配对评分：

```bash
python src/compare_b.py results/new_run/base_results.jsonl results/new_run/lora_results.jsonl --output-dir results/new_run/scored
```

脚本拒绝覆盖已有输出。中途失败时，保持代码、模型路径、材料和参数完全一致，添加 `--resume` 续跑。若配置改变，使用新输出路径。默认材料清单位于项目内 `provenance/b_eval/provenance/materials_verified.json`，可用 `--materials` 指定其他已核验清单。
