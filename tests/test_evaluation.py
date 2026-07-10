from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from nihaisha_kg.evaluation import (
    EvalCase,
    aggregate_metrics,
    evaluate_ranked_ids,
    load_eval_cases,
)


class EvaluationTests(unittest.TestCase):
    def assert_invalid_eval_jsonl(self, content: str, physical_line: int = 1) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.jsonl"
            path.write_text(content, encoding="utf-8")

            with self.assertRaises(ValueError) as caught:
                load_eval_cases(path)

            self.assertIn(f"{path}:{physical_line}:", str(caught.exception))

    def test_load_eval_cases_reads_jsonl_record(self) -> None:
        record = {
            "case_id": "case-1",
            "query": "query",
            "task_type": "source_lookup",
            "relevant_paragraph_ids": ["p2", "p4"],
            "forbidden_paragraph_ids": ["p8"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            cases = load_eval_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertIsInstance(cases[0], EvalCase)
        self.assertEqual(cases[0].relevant_paragraph_ids, ("p2", "p4"))
        self.assertEqual(cases[0].forbidden_paragraph_ids, ("p8",))

    def test_evaluate_ranked_ids_reports_retrieval_metrics(self) -> None:
        case = EvalCase(
            case_id="case-1",
            query="query",
            task_type="comparison",
            relevant_paragraph_ids=("p2", "p4"),
            forbidden_paragraph_ids=("p8",),
        )

        metrics = evaluate_ranked_ids(case, ["p8", "p2", "p3", "p4"], k_values=(1, 3, 5))

        self.assertEqual(metrics["hit_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 0.5)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["forbidden_hits_at_5"], 1.0)
        self.assertEqual(metrics["context_precision_at_5"], 0.5)
        expected_ndcg = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics["ndcg_at_5"], expected_ndcg)

    def test_load_eval_cases_rejects_malformed_or_invalid_records_with_location(self) -> None:
        valid = {
            "case_id": "case-1",
            "query": "query",
            "task_type": "source_lookup",
            "relevant_paragraph_ids": ["p1"],
            "forbidden_paragraph_ids": [],
        }
        invalid_records = {
            "non-object row": [],
            "missing field": {},
            "scalar relevant IDs": {**valid, "relevant_paragraph_ids": "p1"},
            "scalar forbidden IDs": {**valid, "forbidden_paragraph_ids": "p2"},
            "empty relevant IDs": {**valid, "relevant_paragraph_ids": []},
            "duplicate relevant IDs": {**valid, "relevant_paragraph_ids": ["p1", "p1"]},
            "duplicate forbidden IDs": {**valid, "forbidden_paragraph_ids": ["p2", "p2"]},
            "empty relevant ID": {**valid, "relevant_paragraph_ids": [""]},
            "empty forbidden ID": {**valid, "forbidden_paragraph_ids": [""]},
            "overlapping IDs": {
                **valid,
                "relevant_paragraph_ids": ["p1"],
                "forbidden_paragraph_ids": ["p1"],
            },
        }
        for label, record in invalid_records.items():
            with self.subTest(label=label):
                self.assert_invalid_eval_jsonl("\n" + json.dumps(record) + "\n", physical_line=2)

        self.assert_invalid_eval_jsonl('\n{"case_id":\n', physical_line=2)
        self.assert_invalid_eval_jsonl("\n\n", physical_line=1)

    def test_load_eval_cases_rejects_invalid_scalar_fields_with_location(self) -> None:
        valid = {
            "case_id": "case-1",
            "query": "query",
            "task_type": "source_lookup",
            "relevant_paragraph_ids": ["p1"],
        }
        for field in ("case_id", "query", "task_type"):
            for value in (None, 7, ""):
                with self.subTest(field=field, value=value):
                    record = {**valid, field: value}
                    self.assert_invalid_eval_jsonl(
                        "\n" + json.dumps(record) + "\n",
                        physical_line=2,
                    )

    def test_load_eval_cases_rejects_duplicate_case_id_with_location(self) -> None:
        record = {
            "case_id": "case-1",
            "query": "query",
            "task_type": "source_lookup",
            "relevant_paragraph_ids": ["p1"],
        }
        content = json.dumps(record) + "\n\n" + json.dumps(record) + "\n"

        self.assert_invalid_eval_jsonl(content, physical_line=3)

    def test_evaluate_ranked_ids_deduplicates_at_first_occurrence(self) -> None:
        case = EvalCase(
            case_id="case-1",
            query="query",
            task_type="source_lookup",
            relevant_paragraph_ids=("p2",),
        )

        metrics = evaluate_ranked_ids(case, ["p8", "p8", "p2"], k_values=(2,))

        self.assertEqual(metrics["hit_at_2"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)

    def test_evaluate_ranked_ids_treats_empty_relevance_as_vacuously_complete(self) -> None:
        case = EvalCase(
            case_id="case-1",
            query="query",
            task_type="source_lookup",
            relevant_paragraph_ids=(),
        )

        metrics = evaluate_ranked_ids(case, ["p1"], k_values=(5,))

        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertEqual(metrics["context_precision_at_5"], 1.0)
        self.assertEqual(metrics["ndcg_at_5"], 1.0)

    def test_aggregate_metrics_handles_empty_and_sparse_rows(self) -> None:
        self.assertEqual(aggregate_metrics([]), {})
        self.assertEqual(
            aggregate_metrics([{"hit_at_1": 1.0}, {"recall_at_5": 0.5}]),
            {"hit_at_1": 0.5, "recall_at_5": 0.25},
        )

    def test_golden_v1_loads_seven_unique_cases(self) -> None:
        path = Path(__file__).resolve().parents[1] / "evals" / "golden_v1.jsonl"

        cases = load_eval_cases(path)
        by_id = {case.case_id: case for case in cases}

        self.assertEqual(len(cases), 7)
        self.assertEqual(len(by_id), 7)
        self.assertEqual(
            by_id["acupuncture-four-gates"].relevant_paragraph_ids,
            ("fb1cc40bb9bbb8be",),
        )


if __name__ == "__main__":
    unittest.main()
