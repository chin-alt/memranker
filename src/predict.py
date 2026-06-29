from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path
from typing import Any

from data import format_input_text, read_json_records, record_to_doc
from modeling import DEFAULT_MODEL_NAME, load_scorer


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank documents for one query with a finetuned reranker.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--docs_file", required=True, help="JSONL/JSON docs. Each row may contain doc or title/abstract.")
    parser.add_argument("--output_file", default="predictions_ranked.json")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--backend", default="auto", choices=["auto", "cross_encoder", "causal_lm"])
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use lexical mock scorer for smoke tests.")
    return parser.parse_args()


def _load_docs(path: str | Path) -> list[dict[str, Any]]:
    rows = read_json_records(path)
    docs = []
    skipped = 0
    for idx, row in enumerate(rows):
        doc = record_to_doc(row)
        if not doc:
            skipped += 1
            continue
        docs.append({"doc": doc, "source_index": idx, "raw": row})
    if skipped:
        logger.warning("Skipped %d docs without usable text", skipped)
    return docs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")

    docs = _load_docs(args.docs_file)
    if not docs:
        raise ValueError(f"No usable docs found in {args.docs_file}")

    input_texts = [format_input_text(args.instruction, args.query, row["doc"]) for row in docs]
    scorer = load_scorer(
        args.model_path,
        backend=args.backend,
        max_length=args.max_length,
        batch_query=args.query,
        bf16=args.bf16,
        fp16=args.fp16,
        mock=args.mock,
    )
    scores = scorer.predict(input_texts, batch_size=args.batch_size)

    ranked = []
    for row, score in zip(docs, scores, strict=False):
        ranked.append(
            {
                "doc": row["doc"],
                "score": float(score),
                "source_index": row["source_index"],
                "raw": row["raw"],
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx

    top_k = ranked[: max(0, args.top_k)]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(top_k, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote top-%d predictions to %s", len(top_k), output_path)
    print(json.dumps(top_k, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
