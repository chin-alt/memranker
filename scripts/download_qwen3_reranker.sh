#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-Reranker-0.6B}"
LOCAL_DIR="${LOCAL_DIR:-models/Qwen3-Reranker-0.6B}"

python src/download_model.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --local_dir "${LOCAL_DIR}" \
  "$@"
