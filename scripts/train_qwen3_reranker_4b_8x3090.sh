#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

TRAIN_FILE="${TRAIN_FILE:-data/split_seed42/train.jsonl}"
DEV_FILE="${DEV_FILE:-data/split_seed42/dev.jsonl}"
TEST_FILE="${TEST_FILE:-data/split_seed42/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3_reranker_4b_8x3090_lora}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-Reranker-4B}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision fp16 \
  src/train_pointwise.py \
  --train_file "${TRAIN_FILE}" \
  --dev_file "${DEV_FILE}" \
  --test_file "${TEST_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --max_length "${MAX_LENGTH}" \
  --epochs "${EPOCHS:-3}" \
  --lr "${LR:-2e-5}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --use_lora \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --gradient_checkpointing \
  --fp16 \
  "$@"
