#!/usr/bin/env bash
set -euo pipefail

TEST_FILE="${TEST_FILE:-data/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baseline_qwen3_reranker_06b}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-Reranker-0.6B}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PRECISION="${PRECISION:-fp16}"

PRECISION_ARGS=()
if [[ "${PRECISION}" == "fp16" ]]; then
  PRECISION_ARGS+=(--fp16)
elif [[ "${PRECISION}" == "bf16" ]]; then
  PRECISION_ARGS+=(--bf16)
fi

python src/evaluate.py \
  --test_file "${TEST_FILE}" \
  --model_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  "${PRECISION_ARGS[@]}" \
  "$@"
