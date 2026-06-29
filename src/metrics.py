from __future__ import annotations

import math

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(labels: Iterable[float], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.float64)
    p = np.asarray(list(scores), dtype=np.float64)
    if y.size == 0:
        return 0.0
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def mean_squared_error(labels: Iterable[float], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.float64)
    p = np.asarray(list(scores), dtype=np.float64)
    if y.size == 0:
        return 0.0
    return float(np.mean((y - p) ** 2))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(labels: Iterable[float], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.float64)
    p = np.asarray(list(scores), dtype=np.float64)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def spearman_corr(labels: Iterable[float], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.float64)
    p = np.asarray(list(scores), dtype=np.float64)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    return pearson_corr(_rankdata(y), _rankdata(p))


def compute_pointwise_metrics(labels: Iterable[float], scores: Iterable[float]) -> dict[str, float]:
    labels_list = list(labels)
    scores_list = list(scores)
    return {
        "MSE": mean_squared_error(labels_list, scores_list),
        "BCE": binary_cross_entropy(labels_list, scores_list),
        "Pearson": pearson_corr(labels_list, scores_list),
        "Spearman": spearman_corr(labels_list, scores_list),
    }


def _dcg(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    gains = labels[order[:k]]
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains * discounts))


def _average_precision(relevant: np.ndarray, order: np.ndarray) -> float:
    rel_ordered = relevant[order].astype(bool)
    total_relevant = int(np.sum(rel_ordered))
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, is_rel in enumerate(rel_ordered, start=1):
        if is_rel:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / total_relevant


def _mrr(relevant: np.ndarray, order: np.ndarray) -> float:
    rel_ordered = relevant[order].astype(bool)
    for idx, is_rel in enumerate(rel_ordered, start=1):
        if is_rel:
            return 1.0 / idx
    return 0.0


def _recall_at_k(relevant: np.ndarray, order: np.ndarray, k: int) -> float:
    total_relevant = int(np.sum(relevant))
    if total_relevant == 0:
        return 0.0
    return float(np.sum(relevant[order[:k]]) / total_relevant)


def group_records(records: Iterable[dict[str, Any]], query_key: str = "query") -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get(query_key, ""))].append(row)
    return dict(groups)


def compute_per_query_metrics(
    records: Iterable[dict[str, Any]],
    label_key: str = "label",
    score_key: str = "score",
    query_key: str = "query",
    relevance_threshold: float = 0.7,
    ndcg_ks: tuple[int, ...] = (1, 3, 10),
    recall_ks: tuple[int, ...] = (1, 3, 5),
) -> list[dict[str, Any]]:
    per_query = []
    for group_key, rows in group_records(records, query_key=query_key).items():
        labels = np.asarray([float(row.get(label_key, 0.0)) for row in rows], dtype=np.float64)
        scores = np.asarray([float(row.get(score_key, 0.0)) for row in rows], dtype=np.float64)
        order_score = np.argsort(-scores, kind="mergesort")
        order_label = np.argsort(-labels, kind="mergesort")
        relevant = labels >= relevance_threshold

        metrics: dict[str, Any] = {
            "query": group_key,
            "num_docs": int(len(rows)),
            "num_relevant": int(np.sum(relevant)),
            "AP": _average_precision(relevant, order_score),
            "MRR": _mrr(relevant, order_score),
        }
        for k in ndcg_ks:
            ideal = _dcg(labels, order_label, k)
            metrics[f"NDCG@{k}"] = 0.0 if ideal <= 0 else _dcg(labels, order_score, k) / ideal
        for k in recall_ks:
            metrics[f"Recall@{k}"] = _recall_at_k(relevant, order_score, k)
        per_query.append(metrics)
    return per_query


def aggregate_ranking_metrics(per_query: list[dict[str, Any]]) -> dict[str, float]:
    if not per_query:
        return {
            "MAP": 0.0,
            "MRR": 0.0,
            "NDCG@1": 0.0,
            "NDCG@3": 0.0,
            "NDCG@10": 0.0,
            "Recall@1": 0.0,
            "Recall@3": 0.0,
            "Recall@5": 0.0,
        }
    keys = [key for key in per_query[0] if key.startswith("NDCG@") or key.startswith("Recall@")]
    keys.extend(["MRR"])
    result = {"MAP": float(np.mean([row["AP"] for row in per_query]))}
    for key in keys:
        result[key] = float(np.mean([row.get(key, 0.0) for row in per_query]))
    return result


def compute_all_metrics(
    records: Iterable[dict[str, Any]],
    label_key: str = "label",
    score_key: str = "score",
    query_key: str = "query",
    relevance_threshold: float = 0.7,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = list(records)
    labels = [float(row.get(label_key, 0.0)) for row in rows]
    scores = [float(row.get(score_key, 0.0)) for row in rows]
    overall = compute_pointwise_metrics(labels, scores)
    per_query = compute_per_query_metrics(
        rows,
        label_key=label_key,
        score_key=score_key,
        query_key=query_key,
        relevance_threshold=relevance_threshold,
    )
    overall.update(aggregate_ranking_metrics(per_query))
    overall["num_examples"] = float(len(rows))
    overall["num_queries"] = float(len(per_query))
    return overall, per_query


def add_group_ranks(
    records: list[dict[str, Any]],
    label_key: str = "label",
    score_key: str = "score",
    query_key: str = "query",
) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in records]
    by_query: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(ranked):
        by_query[str(row.get(query_key, ""))].append(idx)

    for indices in by_query.values():
        by_label = sorted(indices, key=lambda i: (-float(ranked[i].get(label_key, 0.0)), i))
        by_score = sorted(indices, key=lambda i: (-float(ranked[i].get(score_key, 0.0)), i))
        for rank, idx in enumerate(by_label, start=1):
            ranked[idx]["rank_by_label"] = rank
        for rank, idx in enumerate(by_score, start=1):
            ranked[idx]["rank_by_score"] = rank
    return ranked


def is_better_metric(
    current: dict[str, float],
    best: dict[str, float] | None,
    primary: str = "NDCG@3",
    loss_key: str = "BCE",
) -> bool:
    if best is None:
        return True
    cur_primary = current.get(primary, -math.inf)
    best_primary = best.get(primary, -math.inf)
    if cur_primary > best_primary:
        return True
    if cur_primary < best_primary:
        return False
    return current.get(loss_key, math.inf) < best.get(loss_key, math.inf)
