#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-outputs/qwen3_reranker_06b_lora/best}"
DOCS_FILE="${DOCS_FILE:-data/docs.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-predictions_ranked.json}"
QUERY="${QUERY:-我刚才看的口袋相机哪款配送速度更快}"
INSTRUCTION="${INSTRUCTION:-请判断文档是否能回答用户查询，并给出相关性分数。}"

python src/predict.py \
  --model_path "${MODEL_PATH}" \
  --instruction "${INSTRUCTION}" \
  --query "${QUERY}" \
  --docs_file "${DOCS_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --top_k "${TOP_K:-10}" \
  "$@"
