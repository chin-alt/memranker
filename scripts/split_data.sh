#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${INPUT_FILE:-data/all.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/split_seed42}"

python src/split_data.py \
  --input_file "${INPUT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED:-42}" \
  --eval_ratio "${EVAL_RATIO:-0.1}" \
  --test_ratio "${TEST_RATIO:-0.1}" \
  "$@"
