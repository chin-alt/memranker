# MemReranker-style Pointwise Distillation

This repository is a small-scale, runnable reproduction of the MemReranker
Stage 2 idea: pointwise BCE distillation for domain reranking.

It does not train a large model from scratch. It starts from a Qwen3 reranker
checkpoint, formats each sample as instruction + query + document, and fits
soft teacher labels from your query-doc-score data.

Default base model:

```text
Qwen/Qwen3-Reranker-0.6B
```

The 8-GPU script targets:

```text
Qwen/Qwen3-Reranker-4B
```

## What Is Reproduced

This project focuses on MemReranker Stage 2:

- Student initialized from a Qwen3-Reranker checkpoint.
- Input text is instruction + query + document.
- The model predicts a query-document relevance score.
- Training uses pointwise BCE soft-label distillation.
- Original `labels` are treated as 0-10 scores and normalized as `labels / 10.0`.
- Labels are clipped to `[0, 1]`.
- `reason` is not used for training; it is kept for debugging and case study output.

Differences from the original paper:

- Stage 0 Rank-DistiLLM data training is not reproduced.
- Stage 1 GPT/Qwen ensemble pairwise label generation is not reproduced.
- Stage 3 InfoNCE is not fully reproduced.
- This project uses your existing query-doc-score data as teacher soft labels.

## Data Format

JSONL and JSON arrays are supported.

```json
{
  "instruction": "Score whether the document answers the query.",
  "query": "Which pocket camera I viewed ships faster?",
  "doc": "title: Pocket Camera A, type: product, abstract: Ships tomorrow.",
  "reason": "The document mentions delivery speed.",
  "labels": 9.475
}
```

If `query_id` exists, it is used as the group key. Otherwise, the raw `query`
string is used. Grouping matters for both split leakage prevention and ranking
metrics.

If `doc` is missing, the loader tries to build a document string from fields
such as `title`, `type`, `abstract`, `content`, `text`, and `memory`.

## Install

```bash
pip install -r requirements.txt
```

For QLoRA, use a Linux CUDA environment with `bitsandbytes`.

## Fixed Train/Dev/Test Split

For formal experiments, first export fixed split files with a fixed seed:

```bash
python src/split_data.py \
  --input_file data/all.jsonl \
  --output_dir data/split_seed42 \
  --seed 42 \
  --eval_ratio 0.1 \
  --test_ratio 0.1
```

Or use the helper script:

```bash
INPUT_FILE=data/all.jsonl OUTPUT_DIR=data/split_seed42 bash scripts/split_data.sh
```

This writes:

```text
data/split_seed42/train.jsonl
data/split_seed42/dev.jsonl
data/split_seed42/test.jsonl
data/split_seed42/split_info.json
data/split_seed42/splits.json
```

The split is by query group, so the same `query_id` or same `query` text will
not appear in multiple splits. The original JSON fields are preserved.

## Baseline Evaluation

Evaluate the unfinetuned 0.6B model on the fixed test split:

```bash
TEST_FILE=data/split_seed42/test.jsonl \
OUTPUT_DIR=outputs/baseline_qwen3_reranker_06b \
bash scripts/eval_baseline.sh
```

Outputs:

```text
overall_metrics.json
per_query_metrics.jsonl
predictions.jsonl
```

## Train 0.6B LoRA

```bash
python src/train_pointwise.py \
  --train_file data/split_seed42/train.jsonl \
  --dev_file data/split_seed42/dev.jsonl \
  --test_file data/split_seed42/test.jsonl \
  --output_dir outputs/qwen3_reranker_06b_lora \
  --model_name_or_path Qwen/Qwen3-Reranker-0.6B \
  --max_length 4096 \
  --epochs 3 \
  --lr 2e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --use_lora \
  --fp16
```

The script supports automatic model download from Hugging Face when the model
name is used. If your cluster is offline, download the model first and pass the
local path to `--model_name_or_path`.

## Train 4B on 8 RTX 3090 GPUs

Use the 8-GPU helper:

```bash
TRAIN_FILE=data/split_seed42/train.jsonl \
DEV_FILE=data/split_seed42/dev.jsonl \
TEST_FILE=data/split_seed42/test.jsonl \
OUTPUT_DIR=outputs/qwen3_reranker_4b_8x3090_lora \
bash scripts/train_qwen3_reranker_4b_8x3090.sh
```

The script runs:

```text
accelerate launch --num_processes 8 --mixed_precision fp16
```

Important defaults for RTX 3090:

- `--backend causal_lm`
- `--model_name_or_path Qwen/Qwen3-Reranker-4B`
- `--fp16`
- `--use_lora`
- `--gradient_checkpointing`
- `--per_device_train_batch_size 1`
- `--gradient_accumulation_steps 8`
- `--max_length 2048`

The effective batch size is:

```text
num_gpus * per_device_train_batch_size * gradient_accumulation_steps
```

With the defaults, that is `8 * 1 * 8 = 64`.

If memory is still tight, reduce `MAX_LENGTH` to 1024 or add `--load_in_4bit`
to the script command line:

```bash
bash scripts/train_qwen3_reranker_4b_8x3090.sh --load_in_4bit
```

## Finetuned Evaluation

Use the same fixed test split:

```bash
python src/evaluate.py \
  --model_path outputs/qwen3_reranker_4b_8x3090_lora/best \
  --test_file data/split_seed42/test.jsonl \
  --output_dir outputs/finetuned_eval \
  --max_length 2048 \
  --fp16
```

Compare:

```text
outputs/baseline_qwen3_reranker_06b/overall_metrics.json
outputs/finetuned_eval/overall_metrics.json
```

Main metrics:

- BCE
- MSE
- Pearson
- Spearman
- MAP
- MRR
- NDCG@1
- NDCG@3
- NDCG@10
- Recall@1
- Recall@3
- Recall@5

NDCG uses the normalized graded label in `[0, 1]`. MAP, MRR, and Recall use a
binary relevance threshold. The default is normalized label `>= 0.7`, equivalent
to original label `>= 7`.

## Prediction

Prepare `docs.jsonl`:

```json
{"doc": "title: PocketCam A, abstract: Ships tomorrow."}
{"title": "PocketCam B", "abstract": "Ships in two weeks."}
```

Run:

```bash
python src/predict.py \
  --model_path outputs/qwen3_reranker_4b_8x3090_lora/best \
  --instruction "Score whether the document answers the query." \
  --query "Which pocket camera ships faster?" \
  --docs_file docs.jsonl \
  --top_k 10 \
  --output_file predictions_ranked.json \
  --fp16
```

## Smoke Test

The repository supports a `--mock` scorer for local pipeline checks without a
GPU or downloaded model:

```bash
python src/evaluate.py --test_file tmp/smoke/toy.jsonl --output_dir tmp/smoke/eval --mock
python src/predict.py --instruction "rank" --query "fast pocket camera delivery" --docs_file tmp/smoke/docs.jsonl --output_file tmp/smoke/predictions_ranked.json --mock
```

The mock scorer is only for smoke tests. Do not use it for real experiments.
