#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-outputs/qwen3_reranker_06b_lora/best}"
DOCS_FILE="${DOCS_FILE:-data/docs.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-predictions_ranked.json}"
QUERY="${QUERY:-Which pocket camera ships faster?}"
INSTRUCTION="${INSTRUCTION:-Judge whether the document answers the query.}"
BACKEND="${BACKEND:-causal_lm}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

python src/predict.py \
  --backend "${BACKEND}" \
  --model_path "${MODEL_PATH}" \
  --instruction "${INSTRUCTION}" \
  --query "${QUERY}" \
  --docs_file "${DOCS_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --top_k "${TOP_K:-10}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  "$@"
