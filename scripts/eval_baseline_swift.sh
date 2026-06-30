#!/usr/bin/env bash
set -euo pipefail

TEST_FILE="${TEST_FILE:-data/split_seed42/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baseline_qwen3_reranker_06b_swift}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-Reranker-0.6B}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
SWIFT_ATTN_IMPL="${SWIFT_ATTN_IMPL:-flash_attention_2}"

python src/evaluate.py \
  --backend swift \
  --test_file "${TEST_FILE}" \
  --model_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --swift_attn_impl "${SWIFT_ATTN_IMPL}" \
  "$@"
