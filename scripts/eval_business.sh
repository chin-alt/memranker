#!/usr/bin/env bash
set -euo pipefail

GT_FILE="${GT_FILE:-data/business/ground_truth.xlsx}"
RECALL_FILE="${RECALL_FILE:-data/business/recall.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/business_eval}"
MODEL_PATH="${MODEL_PATH:-outputs/qwen3_reranker_06b_lora/best}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PRECISION="${PRECISION:-fp16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
INSTRUCTION="${INSTRUCTION:-Given a user query, retrieve relevant documents that answer the query.}"

PRECISION_ARGS=()
if [[ "${PRECISION}" == "fp16" ]]; then
  PRECISION_ARGS+=(--fp16)
elif [[ "${PRECISION}" == "bf16" ]]; then
  PRECISION_ARGS+=(--bf16)
fi

ATTN_ARGS=()
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARGS+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
fi

python src/evaluate_business.py \
  --gt_file "${GT_FILE}" \
  --recall_file "${RECALL_FILE}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --instruction "${INSTRUCTION}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  "${PRECISION_ARGS[@]}" \
  "${ATTN_ARGS[@]}" \
  "$@"
