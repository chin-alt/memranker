from __future__ import annotations

import json
import logging
import random

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)


INPUT_TEMPLATE = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


@dataclass
class RerankerExample:
    instruction: str
    query: str
    doc: str
    label: float
    raw_label: float
    reason: str = ""
    query_id: str | None = None
    source_index: int | None = None

    @property
    def group_key(self) -> str:
        return str(self.query_id) if self.query_id not in (None, "") else self.query

    @property
    def input_text(self) -> str:
        return format_input_text(self.instruction, self.query, self.doc)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["group_key"] = self.group_key
        data["input_text"] = self.input_text
        return data


class RerankerDataset:
    """Small torch-compatible dataset that keeps metadata for grouped eval."""

    def __init__(self, examples: list[RerankerExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ex = self.examples[index]
        return {
            "input_text": ex.input_text,
            "label": ex.label,
            "query": ex.query,
            "doc": ex.doc,
            "reason": ex.reason,
            "raw_label": ex.raw_label,
            "query_id": ex.query_id,
            "group_key": ex.group_key,
        }


def format_input_text(instruction: str, query: str, doc: str) -> str:
    return INPUT_TEMPLATE.format(
        instruction=(instruction or "").strip(),
        query=(query or "").strip(),
        doc=(doc or "").strip(),
    )


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_no}")
                rows.append(obj)
        return rows

    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ("data", "records", "examples", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [obj]
    raise ValueError(f"Unsupported JSON root in {path}: {type(obj).__name__}")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def record_to_doc(record: dict[str, Any]) -> str:
    doc = record.get("doc")
    if doc not in (None, ""):
        return _stringify(doc).strip()

    fields = []
    for key in ("title", "type", "abstract", "content", "text", "memory"):
        value = record.get(key)
        if value not in (None, ""):
            fields.append(f"{key}: {_stringify(value).strip()}")
    return ", ".join(fields).strip()


def _coerce_label(record: dict[str, Any], require_label: bool) -> tuple[float, float] | None:
    raw = record.get("labels", record.get("label", record.get("score")))
    if raw is None:
        if require_label:
            return None
        raw = 0.0
    try:
        raw_label = float(raw)
    except (TypeError, ValueError):
        if require_label:
            return None
        raw_label = 0.0
    label = max(0.0, min(1.0, raw_label / 10.0))
    return raw_label, label


def record_to_example(
    record: dict[str, Any],
    source_index: int,
    default_instruction: str = "",
    require_label: bool = True,
) -> RerankerExample | None:
    label_pair = _coerce_label(record, require_label=require_label)
    if label_pair is None:
        return None
    raw_label, label = label_pair

    instruction = _stringify(record.get("instruction") or default_instruction).strip()
    query = _stringify(record.get("query", record.get("question", record.get("q", "")))).strip()
    doc = record_to_doc(record)
    if not query or not doc:
        return None

    query_id = record.get("query_id", record.get("qid"))
    if query_id is not None:
        query_id = str(query_id)

    return RerankerExample(
        instruction=instruction,
        query=query,
        doc=doc,
        label=label,
        raw_label=raw_label,
        reason=_stringify(record.get("reason", "")).strip(),
        query_id=query_id,
        source_index=source_index,
    )


def load_examples(
    path: str | Path,
    default_instruction: str = "",
    require_label: bool = True,
) -> list[RerankerExample]:
    records = read_json_records(path)
    examples: list[RerankerExample] = []
    skipped = 0
    for idx, record in enumerate(records):
        ex = record_to_example(
            record,
            source_index=idx,
            default_instruction=default_instruction,
            require_label=require_label,
        )
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)
    if skipped:
        logger.warning("Skipped %d malformed/empty records from %s", skipped, path)
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def _group_examples(examples: Iterable[RerankerExample]) -> dict[str, list[RerankerExample]]:
    groups: dict[str, list[RerankerExample]] = {}
    for ex in examples:
        groups.setdefault(ex.group_key, []).append(ex)
    return groups


def split_by_query(
    examples: list[RerankerExample],
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, list[RerankerExample]], dict[str, Any]]:
    if not examples:
        raise ValueError("No examples loaded; cannot split dataset.")

    groups = _group_examples(examples)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n_groups = len(keys)

    if n_groups == 1:
        train_keys, dev_keys, test_keys = keys, [], []
    else:
        n_test = int(round(n_groups * max(0.0, test_ratio)))
        if test_ratio > 0 and n_groups >= 3:
            n_test = max(1, n_test)
        n_test = min(n_test, n_groups - 1)

        remaining = n_groups - n_test
        n_dev = int(round(n_groups * max(0.0, eval_ratio)))
        if eval_ratio > 0 and remaining >= 2:
            n_dev = max(1, n_dev)
        n_dev = min(n_dev, remaining - 1)

        test_keys = keys[:n_test]
        dev_keys = keys[n_test : n_test + n_dev]
        train_keys = keys[n_test + n_dev :]

    split_keys = {"train": train_keys, "dev": dev_keys, "test": test_keys}
    splits = {
        name: [ex for key in split_keys[name] for ex in groups[key]]
        for name in ("train", "dev", "test")
    }
    split_info = {
        "strategy": "group_by_query",
        "seed": seed,
        "eval_ratio": eval_ratio,
        "test_ratio": test_ratio,
        "num_groups": n_groups,
        "splits": {
            name: {
                "num_groups": len(split_keys[name]),
                "num_examples": len(splits[name]),
                "group_keys": split_keys[name],
            }
            for name in ("train", "dev", "test")
        },
    }
    return splits, split_info


def _read_split_file(split_file: str | Path) -> dict[str, set[str] | set[int]]:
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if isinstance(data, list):
        result: dict[str, set[str] | set[int]] = {"train": set(), "dev": set(), "test": set()}
        for row in data:
            if not isinstance(row, dict):
                continue
            split = str(row.get("split", "")).lower()
            if split not in result:
                continue
            key = row.get("query_id", row.get("query", row.get("group_key", row.get("index"))))
            if isinstance(key, int):
                result[split].add(key)  # type: ignore[arg-type]
            elif key is not None:
                result[split].add(str(key))  # type: ignore[arg-type]
        return result

    if not isinstance(data, dict):
        raise ValueError("Split file must be a JSON object or a list of split rows.")

    result = {}
    for split in ("train", "dev", "test"):
        values = data.get(split, [])
        if not isinstance(values, list):
            raise ValueError(f"Split file key '{split}' must be a list.")
        if values and all(isinstance(v, int) for v in values):
            result[split] = set(values)
        else:
            result[split] = {str(v) for v in values}
    return result


def split_from_file(
    examples: list[RerankerExample],
    split_file: str | Path,
) -> tuple[dict[str, list[RerankerExample]], dict[str, Any]]:
    split_values = _read_split_file(split_file)
    splits: dict[str, list[RerankerExample]] = {"train": [], "dev": [], "test": []}

    uses_indices = any(
        values and all(isinstance(v, int) for v in values)
        for values in split_values.values()
    )
    for idx, ex in enumerate(examples):
        assigned = False
        for split, values in split_values.items():
            if uses_indices:
                key_matches = idx in values or (ex.source_index is not None and ex.source_index in values)
            else:
                key_matches = ex.group_key in values or ex.query in values
            if key_matches:
                splits[split].append(ex)
                assigned = True
                break
        if not assigned:
            splits["train"].append(ex)

    split_info = {
        "strategy": "split_file",
        "split_file": str(split_file),
        "splits": {
            name: {
                "num_examples": len(items),
                "num_groups": len({ex.group_key for ex in items}),
            }
            for name, items in splits.items()
        },
    }
    return splits, split_info


def load_dataset_splits(
    data_file: str | Path,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_file: str | Path | None = None,
    default_instruction: str = "",
) -> tuple[dict[str, list[RerankerExample]], dict[str, Any]]:
    examples = load_examples(data_file, default_instruction=default_instruction, require_label=True)
    if split_file:
        return split_from_file(examples, split_file)
    return split_by_query(examples, eval_ratio=eval_ratio, test_ratio=test_ratio, seed=seed)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
