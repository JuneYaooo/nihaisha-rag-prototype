from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from nihaisha_kg import pdf_vector


class RuntimeKnowledgeStructureTests(unittest.TestCase):
    def test_knowledge_relations_returns_grounded_non_rejected_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir) / "rag.sqlite"
            with sqlite3.connect(store) as conn:
                conn.executescript(
                    """
                    CREATE TABLE documents (
                      document_id TEXT PRIMARY KEY, canonical_title TEXT,
                      source_layer TEXT, logical_source_path TEXT
                    );
                    CREATE TABLE evidence_records (
                      evidence_id TEXT PRIMARY KEY, document_id TEXT,
                      paragraph_id TEXT, locator TEXT, original_text TEXT,
                      previous_evidence_id TEXT, next_evidence_id TEXT
                    );
                    CREATE TABLE entities (
                      entity_id TEXT PRIMARY KEY, entity_type TEXT,
                      canonical_name TEXT, normalized_key TEXT
                    );
                    CREATE TABLE relations (
                      relation_id TEXT PRIMARY KEY, subject_entity_id TEXT,
                      predicate TEXT, object_entity_id TEXT, literal_value TEXT,
                      evidence_id TEXT, source_layer TEXT, confidence REAL,
                      extraction_method TEXT, extractor_version TEXT,
                      review_status TEXT
                    );
                    """
                )
                conn.execute(
                    "INSERT INTO documents VALUES ('d1','课程讲义','course_primary','pdfs/课程讲义.pdf')"
                )
                conn.execute(
                    "INSERT INTO evidence_records VALUES ('ev1','d1','p1','p12','太阳中风，桂枝汤主之。',NULL,NULL)"
                )
                conn.execute(
                    "INSERT INTO entities VALUES ('e1','formula','桂枝汤','桂枝汤')"
                )
                conn.executemany(
                    "INSERT INTO relations VALUES (?, 'e1', ?, NULL, ?, 'ev1', 'course_primary', 0.8, 'legacy', 'v1', ?)",
                    [
                        ("r1", "indicates_pattern", "太阳中风", "needs_review"),
                        ("r2", "supported_by", "不应返回", "rejected"),
                    ],
                )

            relations = pdf_vector.knowledge_relations(store, "桂枝汤")

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["predicate"], "indicates_pattern")
        self.assertEqual(relations[0]["paragraph_text"], "太阳中风，桂枝汤主之。")
        self.assertEqual(relations[0]["source_layer"], "course_primary")
        self.assertEqual(relations[0]["source_path"], "pdfs/课程讲义.pdf")
        self.assertEqual(relations[0]["review_status"], "needs_review")

    def test_knowledge_relations_is_compatible_with_legacy_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir) / "rag.sqlite"
            with sqlite3.connect(store) as conn:
                conn.execute("CREATE TABLE paragraphs (paragraph_id TEXT)")
            self.assertEqual(pdf_vector.knowledge_relations(store, "桂枝汤"), [])

    def test_citations_include_complete_paragraph_and_context_ids(self) -> None:
        paragraph = "太阳中风，桂枝汤主之。其原始段落可完整展开。"
        citations = pdf_vector.build_citations(
            [
                {
                    "paragraph_id": "p1",
                    "source_path": "pdfs/课程讲义.pdf",
                    "title": "课程讲义",
                    "page_start": 12,
                    "page_end": 12,
                    "text": paragraph,
                    "previous_evidence_id": "ev0",
                    "next_evidence_id": "ev2",
                }
            ]
        )

        self.assertEqual(citations[0]["paragraph_text"], paragraph)
        self.assertEqual(citations[0]["previous_evidence_id"], "ev0")
        self.assertEqual(citations[0]["next_evidence_id"], "ev2")
        self.assertLessEqual(len(str(citations[0]["evidence_quote"])), 220)


if __name__ == "__main__":
    unittest.main()
