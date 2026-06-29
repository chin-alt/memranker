from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path
from typing import Any

from data import read_json_records, record_to_example, split_by_query, write_jsonl


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split reranker JSON/JSONL data into train/dev/test by query group with a fixed seed."
    )
    parser.add_argument("--input_file", required=True, help="Input JSONL or JSON array file.")
    parser.add_argument("--output_dir", required=True, help="Directory to write train/dev/test files.")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_instruction", default="")
    parser.add_argument("--train_name", default="train.jsonl")
    parser.add_argument("--dev_name", default="dev.jsonl")
    parser.add_argument("--test_name", default="test.jsonl")
    parser.add_argument("--split_info_name", default="split_info.json")
    parser.add_argument("--split_keys_name", default="splits.json")
    return parser.parse_args()


def _records_for_examples(
    records: list[dict[str, Any]],
    examples: list[Any],
) -> list[dict[str, Any]]:
    rows = []
    for ex in examples:
        if ex.source_index is None:
            continue
        rows.append(dict(records[ex.source_index]))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    records = read_json_records(args.input_file)
    examples = []
    skipped = 0
    for idx, record in enumerate(records):
        ex = record_to_example(
            record,
            source_index=idx,
            default_instruction=args.default_instruction,
            require_label=True,
        )
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)
    if not examples:
        raise ValueError("No valid examples found; nothing to split.")
    if skipped:
        logger.warning("Skipped %d malformed rows while splitting.", skipped)

    splits, split_info = split_by_query(
        examples,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "train": output_dir / args.train_name,
        "dev": output_dir / args.dev_name,
        "test": output_dir / args.test_name,
    }
    for split, path in output_files.items():
        rows = _records_for_examples(records, splits[split])
        write_jsonl(path, rows)
        logger.info("Wrote %d %s rows to %s", len(rows), split, path)

    split_info = {
        **split_info,
        "input_file": str(args.input_file),
        "output_files": {name: str(path) for name, path in output_files.items()},
        "num_input_records": len(records),
        "num_valid_records": len(examples),
        "num_skipped_records": skipped,
    }
    (output_dir / args.split_info_name).write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    split_keys = {
        name: split_info["splits"][name].get("group_keys", [])
        for name in ("train", "dev", "test")
    }
    (output_dir / args.split_keys_name).write_text(
        json.dumps(split_keys, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(split_info["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
