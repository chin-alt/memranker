from __future__ import annotations

import argparse
import csv
import json
import logging

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

BASE_COLUMNS = [
    "dataset",
    "model",
    "run_name",
    "output_dir",
    "model_path",
    "gt_file",
    "recall_file",
    "gt_doc_id_col",
    "max_length",
    "batch_size",
    "precision",
    "attn_implementation",
]
METRIC_COLUMNS = [
    "num_gt_queries",
    "num_scored_queries",
    "num_scored_pairs",
    "Accuracy@GTCount",
    "MicroAccuracy@GTCount",
    "total_gt_docs",
    "total_hits_at_gt_count",
    "MRR",
    "Precision@1",
    "Recall@1",
    "F1@1",
    "HitRate@1",
    "Precision@3",
    "Recall@3",
    "F1@3",
    "HitRate@3",
    "Precision@5",
    "Recall@5",
    "F1@5",
    "HitRate@5",
    "Precision@10",
    "Recall@10",
    "F1@10",
    "HitRate@10",
    "score_time_seconds",
    "seconds_per_example",
    "examples_per_second",
    "skipped_recall_queries_without_gt",
    "summary_csv",
    "summary_xlsx",
]
SUMMARY_COLUMNS = BASE_COLUMNS + METRIC_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize business reranker matrix evaluation metrics."
    )
    parser.add_argument("--output_root", required=True, help="Directory containing per-run output folders.")
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--summary_xlsx", default=None)
    return parser.parse_args()


def split_run_name(run_name: str) -> tuple[str, str]:
    if "__" not in run_name:
        return "", run_name
    dataset, model = run_name.split("__", 1)
    return dataset, model


def jsonable_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def collect_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(output_root.glob("*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipped unreadable metrics file %s: %s", metrics_path, exc)
            continue

        run_name = metrics_path.parent.name
        dataset, model = split_run_name(run_name)
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "run_name": run_name,
            "output_dir": str(metrics_path.parent),
        }
        for key in SUMMARY_COLUMNS:
            if key in row:
                continue
            row[key] = jsonable_cell(metrics.get(key, ""))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_xlsx(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        from openpyxl import Workbook
    except ImportError:
        logger.warning("openpyxl is not installed; skipped xlsx matrix summary output.")
        return False

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "summary"
    sheet.append(SUMMARY_COLUMNS)
    for row in rows:
        sheet.append([row.get(col, "") for col in SUMMARY_COLUMNS])
    for column_cells in sheet.columns:
        max_length = max(len(str(cell.value or "")) for cell in column_cells)
        sheet.column_dimensions[column_cells[0].column_letter].width = min(max(12, max_length + 2), 72)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    output_root = Path(args.output_root)
    rows = collect_rows(output_root)
    if not rows:
        raise ValueError(f"No metrics.json files found under {output_root}")

    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "summary_metrics.csv"
    summary_json = Path(args.summary_json) if args.summary_json else output_root / "summary_metrics.json"
    summary_xlsx = Path(args.summary_xlsx) if args.summary_xlsx else output_root / "summary_metrics.xlsx"

    write_csv(summary_csv, rows)
    write_json(summary_json, rows)
    wrote_xlsx = write_xlsx(summary_xlsx, rows)

    logger.info("Wrote %d matrix summary rows to %s", len(rows), summary_csv)
    print(
        json.dumps(
            {
                "num_runs": len(rows),
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
                "summary_xlsx": str(summary_xlsx) if wrote_xlsx else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
