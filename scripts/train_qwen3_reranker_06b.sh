#!/usr/bin/env bash
set -euo pipefail

TRAIN_FILE="${TRAIN_FILE:-data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3_reranker_06b_lora}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-Reranker-0.6B}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

python src/train_pointwise.py \
  --backend "${BACKEND:-auto}" \
  --train_file "${TRAIN_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --max_length "${MAX_LENGTH}" \
  --epochs "${EPOCHS:-3}" \
  --lr "${LR:-2e-5}" \
  --per_device_train_batch_size "${BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --use_lora \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  "$@"
