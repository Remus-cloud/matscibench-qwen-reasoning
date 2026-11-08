#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash train.sh /path/to/pinned/Qwen3.5-27B /path/to/new/run" >&2
  exit 2
fi
model_dir=$1
output_dir=$2
data_dir=$(cd "$(dirname "$0")" && pwd)
if [[ ! -f "$model_dir/config.json" ]]; then
  echo "Missing base model config in $model_dir" >&2
  exit 2
fi
export HF_DEACTIVATE_ASYNC_LOAD=1

swift sft --model "$model_dir" --tuner_type lora --target_modules all-linear \
  --freeze_vit true --freeze_aligner true --quant_method bnb --quant_bits 4 \
  --bnb_4bit_quant_type nf4 --bnb_4bit_use_double_quant true \
  --torch_dtype bfloat16 --bnb_4bit_compute_dtype bfloat16 \
  --attn_impl sdpa --gradient_checkpointing true \
  --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 --learning_rate 3e-5 \
  --dataset "$data_dir/train.jsonl" --val_dataset "$data_dir/val.jsonl" --split_dataset_ratio 0 \
  --add_non_thinking_prefix true --loss_scale ignore_empty_think --max_length 2048 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 \
  --num_train_epochs 1 --lr_scheduler_type cosine --warmup_ratio 0.05 --seed 6020 --data_seed 6020 \
  --output_dir "$output_dir" --add_version false --logging_steps 1 --save_total_limit 2 \
  --save_steps 15 --eval_steps 15 --save_only_model false --report_to none
