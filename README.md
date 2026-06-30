# MemReranker-style Pointwise Distillation

This repository is a small-scale, runnable reproduction of the MemReranker
Stage 2 idea: pointwise BCE distillation for domain reranking.

It does not train a large model from scratch. It starts from a Qwen3 reranker
checkpoint, formats each sample as instruction + query + document, and fits
soft teacher labels from your query-doc-score data.

The paper-aligned causal backend does not generate text. It follows the
Qwen3-Reranker scoring path:

```text
score = softmax([logit_no, logit_yes])[yes]
```

Training optimizes the equivalent binary logit:

```text
BCEWithLogitsLoss(logit_yes - logit_no, labels / 10)
```

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
- The model predicts a continuous query-document relevance score in `[0, 1]`.
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

The default baseline uses the causal LM yes/no-logit backend. The
`sentence-transformers` package is kept for the optional CrossEncoder backend.

If logs say `Can't load the configuration of 'Qwen/Qwen3-Reranker-0.6B'` after
an `httpx.ProxyError` or `504 Gateway Time-out`, the model id is probably fine;
the machine failed to download from Hugging Face. Fix the proxy/mirror or use a
pre-downloaded local model directory.

Common model ids:

```text
Qwen/Qwen3-Reranker-0.6B
Qwen/Qwen3-Reranker-4B
```

The code also normalizes common typos such as `qwen/qwen3-reranker-0.6` to
`Qwen/Qwen3-Reranker-0.6B`.

## Offline Model Download

On a machine that can reach Hugging Face:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-Reranker-0.6B \
LOCAL_DIR=models/Qwen3-Reranker-0.6B \
bash scripts/download_qwen3_reranker.sh
```

For the 4B model:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-Reranker-4B \
LOCAL_DIR=models/Qwen3-Reranker-4B \
bash scripts/download_qwen3_reranker.sh
```

Then copy that directory to the cluster and pass it as a local path:

```bash
python src/evaluate.py \
  --backend causal_lm \
  --model_path /path/to/Qwen3-Reranker-0.6B \
  --test_file data/split_seed42/test.jsonl \
  --output_dir outputs/baseline_local_model \
  --attn_implementation flash_attention_2
```

All scoring and training loops use `tqdm` progress bars. For non-interactive
logs, add `--disable_tqdm` to `src/train_pointwise.py`.

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

If the model is already on the machine, set `MODEL_NAME_OR_PATH` to that local
directory:

```bash
TEST_FILE=data/split_seed42/test.jsonl \
MODEL_NAME_OR_PATH=/path/to/Qwen3-Reranker-0.6B \
OUTPUT_DIR=outputs/baseline_qwen3_reranker_06b \
bash scripts/eval_baseline.sh
```

The script defaults to `BACKEND=causal_lm`, `PRECISION=fp16`,
`BATCH_SIZE=16`, and `ATTN_IMPLEMENTATION=flash_attention_2`. If flash-attn is
not installed, set `ATTN_IMPLEMENTATION=eager`. For a quick throughput check,
inspect these fields in `overall_metrics.json`:

```text
score_time_seconds
seconds_per_example
examples_per_second
```

Common reasons for slow evaluation:

- The run is on CPU, or the model was loaded in fp32 instead of fp16.
- `batch_size=4` and `max_length=4096` are conservative and can underuse the GPU.
- Very long documents make reranking expensive because each query-document pair
  is a full forward pass.
- Hugging Face download/proxy stalls can look like model load latency.

For local 0.6B model evaluation on your Linux machine:

```bash
TEST_FILE=data/split_seed42/test.jsonl \
MODEL_NAME_OR_PATH=/home/c50061497/MemOS/reranker/memranker/models/Qwen3-Reranker-0.6B \
BACKEND=causal_lm \
BATCH_SIZE=16 \
MAX_LENGTH=2048 \
ATTN_IMPLEMENTATION=flash_attention_2 \
OUTPUT_DIR=outputs/baseline_qwen3_reranker_06b \
bash scripts/eval_baseline.sh
```

## Why the old ms-swift script is disabled

The previous ms-swift baseline used a generative reranker interface and parsed
returned text into a numeric score. That is not equivalent to the MemReranker
Stage 2 objective, because BCE distillation needs the continuous probability
derived from final-token `yes/no` logits. Text parsing can turn many rows into
hard `0/1` or `0.0`, which explains large drops in MSE, NDCG, MAP, and MRR.

The repository keeps `scripts/eval_baseline_swift.sh` only as a guardrail; it
exits with an explanation. A future ms-swift integration should expose logits
and compute exactly:

```text
score = softmax([logit_no, logit_yes])[yes]
```

Evaluation outputs:

```text
overall_metrics.json
per_query_metrics.jsonl
predictions.jsonl
```

## Train 0.6B LoRA

```bash
python src/train_pointwise.py \
  --backend causal_lm \
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
  --attn_implementation flash_attention_2 \
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
  --backend causal_lm \
  --model_path outputs/qwen3_reranker_4b_8x3090_lora/best \
  --test_file data/split_seed42/test.jsonl \
  --output_dir outputs/finetuned_eval \
  --max_length 2048 \
  --attn_implementation flash_attention_2 \
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
  --backend causal_lm \
  --model_path outputs/qwen3_reranker_4b_8x3090_lora/best \
  --instruction "Score whether the document answers the query." \
  --query "Which pocket camera ships faster?" \
  --docs_file docs.jsonl \
  --top_k 10 \
  --output_file predictions_ranked.json \
  --attn_implementation flash_attention_2 \
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
