from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


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

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing_fields = [field for field in required_fields if field not in row]
            if missing_fields:
                missing = ", ".join(missing_fields)
                raise ValueError(f"{path}:{line_number}: missing required field(s): {missing}")
            cases.append(
                EvalCase(
                    case_id=str(row["case_id"]),
                    query=str(row["query"]),
                    task_type=str(row["task_type"]),
                    relevant_paragraph_ids=tuple(str(value) for value in row["relevant_paragraph_ids"]),
                    forbidden_paragraph_ids=tuple(
                        str(value) for value in row.get("forbidden_paragraph_ids", ())
                    ),
                )
            )

    if not cases:
        raise ValueError(f"{path}: evaluation file is empty")
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
        metrics[f"recall_at_{k}"] = relevant_hits / len(relevant) if relevant else 0.0
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
            precision_sum / precision_denominator if precision_denominator else 0.0
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
        metrics[f"ndcg_at_{k}"] = discounted_gain / ideal_gain if ideal_gain else 0.0

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
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in sorted(totals)}
