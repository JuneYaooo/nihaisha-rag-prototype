from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nihaisha_kg.evaluation import EvalCase, evaluate_ranked_ids, load_eval_cases


class EvaluationTests(unittest.TestCase):
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
        self.assertGreater(metrics["ndcg_at_5"], 0.5)


if __name__ == "__main__":
    unittest.main()
