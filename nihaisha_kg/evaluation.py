from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


SUPPORTED_EVALUATION_MODES = frozenset({"hybrid", "vector", "text", "knowledge"})


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    task_type: str
    relevant_paragraph_ids: tuple[str, ...]
    forbidden_paragraph_ids: tuple[str, ...] = ()


def load_eval_cases(path: Path) -> list[EvalCase]:
    required_fields = ("case_id", "query", "task_type", "relevant_paragraph_ids")
    cases: list[EvalCase] = []
    seen_case_ids: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}:"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location} invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{location} evaluation row must be a JSON object")
            missing_fields = [field for field in required_fields if field not in row]
            if missing_fields:
                missing = ", ".join(missing_fields)
                raise ValueError(f"{location} missing required field(s): {missing}")

            for field in ("case_id", "query", "task_type"):
                value = row[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{location} {field} must be a non-empty string")

            relevant_ids = row["relevant_paragraph_ids"]
            forbidden_ids = row.get("forbidden_paragraph_ids", [])
            for field, values in (
                ("relevant_paragraph_ids", relevant_ids),
                ("forbidden_paragraph_ids", forbidden_ids),
            ):
                if not isinstance(values, list):
                    raise ValueError(f"{location} {field} must be a JSON array")
                if any(not isinstance(value, str) or not value.strip() for value in values):
                    raise ValueError(f"{location} {field} must contain non-empty strings")
                if len(values) != len(set(values)):
                    raise ValueError(f"{location} {field} must not contain duplicate IDs")

            if not relevant_ids:
                raise ValueError(f"{location} relevant_paragraph_ids must not be empty")
            if set(relevant_ids) & set(forbidden_ids):
                raise ValueError(f"{location} relevant and forbidden paragraph IDs must not overlap")

            case_id = row["case_id"]
            if case_id in seen_case_ids:
                raise ValueError(f"{location} duplicate case_id: {case_id}")
            seen_case_ids.add(case_id)
            cases.append(
                EvalCase(
                    case_id=case_id,
                    query=row["query"],
                    task_type=row["task_type"],
                    relevant_paragraph_ids=tuple(relevant_ids),
                    forbidden_paragraph_ids=tuple(forbidden_ids),
                )
            )

    if not cases:
        raise ValueError(f"{path}:1: evaluation file is empty")
    return cases


def evaluate_ranked_ids(
    case: EvalCase,
    ranked_ids: Iterable[str],
    k_values: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    ranking = list(dict.fromkeys(str(paragraph_id) for paragraph_id in ranked_ids))
    relevant = set(case.relevant_paragraph_ids)
    forbidden = set(case.forbidden_paragraph_ids)
    metrics: dict[str, float] = {}

    for k in k_values:
        if k <= 0:
            raise ValueError("k values must be positive")
        top_k = ranking[:k]
        relevant_hits = sum(paragraph_id in relevant for paragraph_id in top_k)
        metrics[f"hit_at_{k}"] = float(relevant_hits > 0)
        metrics[f"recall_at_{k}"] = relevant_hits / len(relevant) if relevant else 1.0
        metrics[f"forbidden_hits_at_{k}"] = float(
            sum(paragraph_id in forbidden for paragraph_id in top_k)
        )

        precision_sum = 0.0
        hits_through_rank = 0
        for rank, paragraph_id in enumerate(top_k, start=1):
            if paragraph_id in relevant:
                hits_through_rank += 1
                precision_sum += hits_through_rank / rank
        precision_denominator = min(len(relevant), k)
        metrics[f"context_precision_at_{k}"] = (
            precision_sum / precision_denominator if precision_denominator else 1.0
        )

        discounted_gain = sum(
            1.0 / math.log2(rank + 1)
            for rank, paragraph_id in enumerate(top_k, start=1)
            if paragraph_id in relevant
        )
        ideal_gain = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(relevant), k) + 1)
        )
        metrics[f"ndcg_at_{k}"] = discounted_gain / ideal_gain if ideal_gain else 1.0

    metrics["reciprocal_rank"] = next(
        (
            1.0 / rank
            for rank, paragraph_id in enumerate(ranking, start=1)
            if paragraph_id in relevant
        ),
        0.0,
    )
    return metrics


def aggregate_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: totals[key] / len(rows) for key in sorted(totals)}


def evaluate_database(
    store: object,
    cases: Iterable[EvalCase],
    mode: str,
    limit: int = 10,
) -> dict[str, object]:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be a positive integer")
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not isinstance(mode, str):
        raise TypeError("mode must be a supported string")
    if mode not in SUPPORTED_EVALUATION_MODES:
        raise ValueError(f"unsupported evaluation mode: {mode}")

    case_list = list(cases)
    results: list[dict[str, object]] = []
    metric_rows: list[dict[str, float]] = []
    for case in case_list:
        search_results = store.search(case.query, limit=limit, mode=mode)  # type: ignore[attr-defined]
        ranked_ids = [
            str(row["paragraph_id"])
            for row in search_results
            if isinstance(row, Mapping) and row.get("paragraph_id") is not None
        ]
        metrics = evaluate_ranked_ids(case, ranked_ids, k_values=(1, 5, 10))
        metric_rows.append(metrics)
        results.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "task_type": case.task_type,
                "ranked_paragraph_ids": ranked_ids,
                "metrics": dict(metrics),
            }
        )
    return {
        "cases": len(case_list),
        "aggregate": aggregate_metrics(metric_rows),
        "results": results,
    }
