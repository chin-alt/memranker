# MemReranker-style Pointwise Distillation

这是一个小规模的 **MemReranker Stage 2 pointwise BCE distillation** 领域复现项目。它不从零训练大模型，而是基于 `Qwen/Qwen3-Reranker-0.6B`，使用已有的 query-doc-score 数据做 soft-label 蒸馏。

输入格式是：

```text
<Instruct>: {instruction}
<Query>: {query}
<Document>: {doc}
```

训练标签来自数据中的 `labels` 字段，原始范围按 0-10 处理，训练时转换为 `labels / 10.0` 并 clip 到 `[0, 1]`。`reason` 字段默认不参与训练，只保留在 debug、case study 和预测输出里。

## 和原论文的差异

本项目复现的是论文方法里最适合小数据落地的 Stage 2：

- 不复现 Stage 0 Rank-DistiLLM 数据训练。
- 不复现 Stage 1 GPT/Qwen ensemble pairwise label generation。
- 不完整复现 Stage 3 InfoNCE，对比学习可作为后续扩展。
- 使用你已有的约 14150 条 query-doc-score 数据作为 teacher soft labels。

论文原始规模约为 1M+50K pairs，且公开材料不包含完整训练数据与训练脚本，所以这里提供的是可运行、可对比 baseline 的领域微调复现。

## 项目结构

```text
requirements.txt
src/data.py
src/modeling.py
src/train_pointwise.py
src/evaluate.py
src/predict.py
src/metrics.py
scripts/train_qwen3_reranker_06b.sh
scripts/eval_baseline.sh
scripts/predict_example.sh
```

## 数据格式

支持 `.jsonl` 和 JSON 数组：

```json
{
  "instruction": "请根据用户查询判断文档相关性，分数越高越相关。",
  "query": "我刚才看的口袋相机哪款配送速度更快",
  "doc": "title: Pocket Camera A, type: product, abstract: 次日达，轻便相机。",
  "reason": "文档包含配送速度。",
  "labels": 9.475
}
```

如果数据中有 `query_id`，训练/评估会优先用 `query_id` 分组；没有时使用 query 文本本身作为 group key。默认切分 train/dev/test 时按 query 分组，避免同一个 query 泄漏到不同 split。

`doc` 缺失时，代码会尝试用 `title`、`type`、`abstract`、`content`、`text`、`memory` 拼出文档文本。

## 安装

```bash
pip install -r requirements.txt
```

如果使用 QLoRA，需要 Linux CUDA 环境下的 `bitsandbytes`。Windows 本地更适合做数据和脚本 smoke test，真实训练建议放到集群 GPU。

## Baseline 评估

直接用未微调的 `Qwen/Qwen3-Reranker-0.6B` 在测试集上预测：

```bash
TEST_FILE=/path/to/test.jsonl bash scripts/eval_baseline.sh
```

输出目录默认是 `outputs/baseline_qwen3_reranker_06b`，包含：

- `overall_metrics.json`
- `per_query_metrics.jsonl`
- `predictions.jsonl`

## 训练

```bash
TRAIN_FILE=/path/to/all_data.jsonl OUTPUT_DIR=outputs/qwen3_reranker_06b_lora bash scripts/train_qwen3_reranker_06b.sh
```

等价 Python 命令：

```bash
python src/train_pointwise.py \
  --train_file /path/to/all_data.jsonl \
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
  --bf16
```

自动模式会先尝试 `sentence_transformers.CrossEncoder` 路径；如果 Qwen3 checkpoint 与 CrossEncoder 训练接口不兼容，会回退到 `transformers AutoModelForCausalLM` 的 yes/no logits 路径：

```text
score = sigmoid(logit_yes - logit_no)
loss = BCEWithLogitsLoss(score_logit, soft_label)
```

每个 epoch 后在 dev set 上评估 `BCE`、`MSE`、`Pearson`、`Spearman`、`NDCG@1/3/10`、`MAP`、`MRR`、`Recall@1/3/5`。best checkpoint 保存到 `OUTPUT_DIR/best`，选择标准为优先 `NDCG@3`，并用 `BCE` 作为并列时的 tie-breaker。

也可以显式传入 split：

```bash
python src/train_pointwise.py \
  --train_file /path/to/all_data.jsonl \
  --split_file /path/to/splits.json \
  --output_dir outputs/run
```

`splits.json` 可包含：

```json
{
  "train": ["query-or-query-id-1"],
  "dev": ["query-or-query-id-2"],
  "test": ["query-or-query-id-3"]
}
```

## 微调模型评估

```bash
python src/evaluate.py \
  --model_path outputs/qwen3_reranker_06b_lora/best \
  --test_file /path/to/test.jsonl \
  --output_dir outputs/finetuned_eval \
  --max_length 4096
```

对比 baseline 和 finetuned 时，比较两个目录下的 `overall_metrics.json` 即可。建议重点看 `NDCG@3`、`MRR`、`MAP`，同时关注 `BCE/MSE` 是否下降。

NDCG 使用归一化后的 graded label，也就是 `[0, 1]` 的 soft label。MAP/MRR/Recall 需要二值 relevant，默认阈值是 normalized label `>= 0.7`，等价于原始 label `>= 7`；可通过 `--relevance_threshold` 修改。

## 单条推理

`docs_file` 支持每行 `{"doc": "..."}`，也支持 `{"title": "...", "abstract": "..."}` 自动拼接。

```bash
python src/predict.py \
  --model_path outputs/qwen3_reranker_06b_lora/best \
  --instruction "请判断文档是否能回答用户查询，并给出相关性分数。" \
  --query "我刚才看的口袋相机哪款配送速度更快" \
  --docs_file docs.jsonl \
  --top_k 10
```

输出默认写入 `predictions_ranked.json`。

## 显存建议

- `Qwen3-Reranker-0.6B + LoRA` 建议 24GB GPU 起步。
- `max_length=8192` 显存压力明显更高，建议先用 2048 或 4096 跑通。
- 显存不足时优先降低 `--max_length`、`--per_device_train_batch_size`，提高 `--gradient_accumulation_steps`。
- 需要进一步省显存时启用 `--load_in_4bit` 走 QLoRA 路径。
- 单卡训练优先使用 `--use_lora --bf16`；如果 GPU 不支持 bf16，改用 `--fp16`。

## 常见问题

**labels 不是整数怎么办？**  
保留 soft label。比如 `9.475` 会变成 `0.9475`，直接参与 BCE soft-label distillation。

**reason 是否参与训练？**  
默认不参与。`reason` 只写入预测结果，便于 debug 和 case study。

**query 没有 query_id 怎么办？**  
使用 query 文本作为 group key，并按 group key 做切分和排序指标。

**为什么不是完整复现论文？**  
因为公开材料没有完整训练数据和训练脚本，且论文原始训练规模约为 1M+50K pairs。本项目聚焦 Stage 2 pointwise BCE distillation，适合用现有 14150 条领域数据快速验证微调收益。

## Smoke Test

本仓库提供 `--mock` 词面重叠 scorer，便于在没有 GPU、没有下载模型时检查数据读取、指标和推理脚本：

```bash
python src/evaluate.py --test_file tmp/smoke/toy.jsonl --output_dir tmp/smoke/eval --mock
python src/predict.py --instruction "rank" --query "fast pocket camera delivery" --docs_file tmp/smoke/docs.jsonl --output_file tmp/smoke/predictions_ranked.json --mock
```

真实模型训练不使用 `--mock`。

本地 smoke test 已在 8 条 toy JSONL 上跑通：

```text
data split: train=5, dev=2, test=1
mock eval: BCE=0.6777, MSE=0.0786, Pearson=0.7472, Spearman=0.4910, MAP=0.8750, MRR=0.8750, NDCG@3=0.9741
mock predict: wrote tmp/smoke/predictions_ranked.json
```
