from __future__ import annotations

import argparse
import json
import logging
import time

from pathlib import Path

from data import load_examples, write_jsonl
from metrics import add_group_ranks, compute_all_metrics
from modeling import DEFAULT_MODEL_NAME, load_scorer


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3/MemReranker-style reranker.")
    parser.add_argument("--test_file", required=True, help="JSON/JSONL file with query-doc-label rows.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_NAME, help="Base model or finetuned checkpoint.")
    parser.add_argument("--output_dir", default="outputs/eval", help="Directory for metric and prediction files.")
    parser.add_argument("--backend", default="causal_lm", choices=["auto", "cross_encoder", "causal_lm"])
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--default_instruction", default="")
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional transformers attention backend, for example flash_attention_2 or eager.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use lexical mock scorer for smoke tests.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")

    examples = load_examples(args.test_file, default_instruction=args.default_instruction)
    scorer = load_scorer(
        args.model_path,
        backend=args.backend,
        max_length=args.max_length,
        bf16=args.bf16,
        fp16=args.fp16,
        mock=args.mock,
        attn_implementation=args.attn_implementation,
    )

    input_texts = [ex.input_text for ex in examples]
    logger.info(
        "Scoring %d examples with backend=%s batch_size=%d max_length=%d",
        len(input_texts),
        args.backend,
        args.batch_size,
        args.max_length,
    )
    start_time = time.perf_counter()
    scores = scorer.predict(input_texts, batch_size=args.batch_size)
    score_time = time.perf_counter() - start_time
    sec_per_example = score_time / max(1, len(input_texts))
    examples_per_sec = len(input_texts) / score_time if score_time > 0 else 0.0
    logger.info(
        "Scoring finished in %.2fs: %.4fs/example, %.2f examples/s",
        score_time,
        sec_per_example,
        examples_per_sec,
    )

    rows = []
    for ex, score in zip(examples, scores, strict=False):
        row = {
            "group_key": ex.group_key,
            "query": ex.query,
            "query_id": ex.query_id,
            "doc": ex.doc,
            "label": ex.label,
            "raw_label": ex.raw_label,
            "score": float(score),
            "reason": ex.reason,
        }
        rows.append(row)
    rows = add_group_ranks(rows, query_key="group_key")
    overall, per_query = compute_all_metrics(
        rows,
        query_key="group_key",
        relevance_threshold=args.relevance_threshold,
    )
    overall["score_time_seconds"] = float(score_time)
    overall["seconds_per_example"] = float(sec_per_example)
    overall["examples_per_second"] = float(examples_per_sec)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "per_query_metrics.jsonl", per_query)
    write_jsonl(output_dir / "predictions.jsonl", rows)

    logger.info("Wrote evaluation outputs to %s", output_dir)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
