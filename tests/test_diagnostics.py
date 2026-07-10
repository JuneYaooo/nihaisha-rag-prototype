from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from nihaisha_kg import diagnostics
from nihaisha_kg.diagnostics import doctor
from nihaisha_kg.pdf_vector import DenseEmbeddingBackend, LocalVectorStore, RetrievalUnit


class ToyDenseBackend(DenseEmbeddingBackend):
    name = "toy_dense"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class DiagnosticsTests(unittest.TestCase):
    def _create_dense_store(self, base: Path) -> Path:
        db_path = base / "rag.sqlite"
        store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
        store.recreate()
        paragraph = diagnostics_test_paragraph()
        store.insert_paragraphs([paragraph])
        store.insert_units(
            [
                RetrievalUnit(
                    unit_id="u1", paragraph_id=paragraph.paragraph_id, doc_id=paragraph.doc_id,
                    unit_type="sentence", text="桂枝汤", text_for_embedding="桂枝汤",
                    sentence_start=0, sentence_end=0, weight=1.0,
                )
            ]
        )
        store.rebuild_text_index()
        store.rebuild_knowledge_units()
        return db_path

    def test_doctor_reports_dense_database_with_missing_faiss_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rag.sqlite"
            store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
            store.recreate()
            store.rebuild_text_index()
            store.rebuild_knowledge_units()

            report = doctor(db_path, faiss_loader=lambda: None)

        codes = {item["code"] for item in report["diagnoses"]}
        self.assertEqual(report["status"], "error")
        self.assertIn("faiss_files_missing", codes)

    def test_doctor_does_not_require_faiss_for_sparse_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rag.sqlite"
            store = LocalVectorStore(db_path)
            store.recreate()
            store.rebuild_text_index()
            store.rebuild_knowledge_units()

            report = doctor(db_path, faiss_loader=lambda: None)

        codes = {item["code"] for item in report["diagnoses"]}
        self.assertEqual(report["status"], "ok")
        self.assertNotIn("faiss_files_missing", codes)
        self.assertNotIn("faiss_module_missing", codes)

    def test_doctor_rejects_missing_empty_and_unknown_vector_kind(self) -> None:
        for vector_kind in (None, "", "bogus"):
            with self.subTest(vector_kind=vector_kind), tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "rag.sqlite"
                store = LocalVectorStore(db_path)
                store.recreate()
                store.rebuild_text_index()
                store.rebuild_knowledge_units()
                with closing(sqlite3.connect(db_path)) as conn:
                    if vector_kind is None:
                        conn.execute("DELETE FROM meta WHERE key = 'vector_kind'")
                    else:
                        conn.execute(
                            "UPDATE meta SET value = ? WHERE key = 'vector_kind'",
                            (vector_kind,),
                        )
                    conn.commit()

                report = doctor(db_path, faiss_loader=lambda: None)

            codes = {item["code"] for item in report["diagnoses"]}
            self.assertNotEqual(report["status"], "ok")
            self.assertIn("vector_metadata_invalid", codes)

    def test_doctor_rejects_missing_embedding_and_nonpositive_vector_dimension(self) -> None:
        for sql in (
            "DELETE FROM meta WHERE key = 'embedding'",
            "UPDATE meta SET value = '0' WHERE key = 'vector_dim'",
        ):
            with self.subTest(sql=sql), tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "rag.sqlite"
                store = LocalVectorStore(db_path)
                store.recreate()
                store.rebuild_text_index()
                store.rebuild_knowledge_units()
                with closing(sqlite3.connect(db_path)) as conn:
                    conn.execute(sql)
                    conn.commit()

                report = doctor(db_path, faiss_loader=lambda: None)

            codes = {item["code"] for item in report["diagnoses"]}
            self.assertNotEqual(report["status"], "ok")
            self.assertIn("vector_metadata_invalid", codes)

    def test_doctor_handles_missing_invalid_and_incomplete_databases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cases = [base / "missing.sqlite", base / "invalid.sqlite", base / "empty.sqlite"]
            cases[1].write_text("not sqlite", encoding="utf-8")
            sqlite3.connect(cases[2]).close()

            reports = [doctor(path, faiss_loader=lambda: None) for path in cases]

        self.assertTrue(all(report["status"] == "error" for report in reports))
        self.assertTrue(all(report["diagnoses"] for report in reports))

    def test_doctor_handles_missing_meta_and_malformed_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            db_path = base / "rag.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.executescript(
                    """
                    CREATE TABLE paragraphs (paragraph_id TEXT);
                    CREATE TABLE retrieval_units (unit_id TEXT);
                    CREATE TABLE paragraphs_fts (paragraph_id TEXT);
                    CREATE TABLE knowledge_units (knowledge_unit_id TEXT);
                    """
                )
            (base / "vectors.faiss").write_bytes(b"index")
            (base / "vector_ids.jsonl").write_text('{"unit_id": "u1"}\nnot-json\n', encoding="utf-8")

            report = doctor(db_path, faiss_loader=lambda: object())

        codes = {item["code"] for item in report["diagnoses"]}
        self.assertIn("meta_missing", codes)
        self.assertIn("faiss_mapping_invalid", codes)

    def test_doctor_reports_mapping_count_and_index_count_mismatch(self) -> None:
        class Index:
            ntotal = 1
            d = 2

        class FakeFaiss:
            def read_index(self, path: str) -> Index:
                return Index()

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            db_path = base / "rag.sqlite"
            store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
            store.recreate()
            store.rebuild_text_index()
            store.rebuild_knowledge_units()
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    """INSERT INTO retrieval_units
                    (unit_id, paragraph_id, doc_id, unit_type, text, sentence_start,
                     sentence_end, weight, vector_blob)
                    VALUES ('u1', 'p1', 'd', 'sentence', 'a', 0, 0, 1, X'00000000')"""
                )
            (base / "vectors.faiss").write_bytes(b"index")
            (base / "vector_ids.jsonl").write_text(
                "\n".join(json.dumps({"unit_id": value}) for value in ("u1", "u2")) + "\n",
                encoding="utf-8",
            )

            report = doctor(db_path, faiss_loader=lambda: FakeFaiss())

        codes = {item["code"] for item in report["diagnoses"]}
        self.assertIn("faiss_mapping_count_mismatch", codes)
        self.assertIn("faiss_index_count_mismatch", codes)

    def test_doctor_streams_deeply_nested_json_as_structured_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            db_path = self._create_dense_store(base)
            (base / "vectors.faiss").write_bytes(b"index")
            (base / "vector_ids.jsonl").write_text("[" * 10_000 + "0" + "]" * 10_000 + "\n")

            report = doctor(db_path, faiss_loader=lambda: None)

        codes = {item["code"] for item in report["diagnoses"]}
        self.assertIn("faiss_mapping_invalid", codes)

    def test_doctor_rejects_oversized_mapping_line_and_file(self) -> None:
        for limit_name in ("MAX_MAPPING_LINE_BYTES", "MAX_MAPPING_FILE_BYTES"):
            with self.subTest(limit_name=limit_name), tempfile.TemporaryDirectory() as tmpdir:
                base = Path(tmpdir)
                db_path = self._create_dense_store(base)
                (base / "vectors.faiss").write_bytes(b"index")
                (base / "vector_ids.jsonl").write_bytes(b"x" * 128)
                with patch.object(diagnostics, limit_name, 64):
                    report = doctor(db_path, faiss_loader=lambda: None)

            codes = {item["code"] for item in report["diagnoses"]}
            self.assertIn("faiss_mapping_limits_exceeded", codes)

    def test_doctor_honors_relative_custom_faiss_paths(self) -> None:
        class Index:
            ntotal = 1
            d = 2

        class FakeFaiss:
            def read_index(self, path: str) -> Index:
                return Index()

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            db_path = self._create_dense_store(base)
            custom = base / "custom"
            custom.mkdir()
            (custom / "index.faiss").write_bytes(b"index")
            (custom / "ids.jsonl").write_text('{"unit_id": "u1"}\n', encoding="utf-8")
            with closing(sqlite3.connect(db_path)) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    (("faiss_index", "custom/index.faiss"), ("faiss_ids", "custom/ids.jsonl")),
                )
                conn.commit()

            report = doctor(db_path, faiss_loader=lambda: FakeFaiss())

        self.assertEqual(report["status"], "ok")

    def test_doctor_honors_legacy_cwd_relative_paths_written_by_builder(self) -> None:
        class Index:
            ntotal = 1
            d = 2

        class FakeFaiss:
            def read_index(self, path: str) -> Index:
                return Index()

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            store_dir = base / "store"
            store_dir.mkdir()
            db_path = self._create_dense_store(store_dir)
            (store_dir / "vectors.faiss").write_bytes(b"index")
            (store_dir / "vector_ids.jsonl").write_text('{"unit_id": "u1"}\n', encoding="utf-8")
            with closing(sqlite3.connect(db_path)) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    (
                        ("faiss_index", "store/vectors.faiss"),
                        ("faiss_ids", "store/vector_ids.jsonl"),
                    ),
                )
                conn.commit()
            previous_cwd = Path.cwd()
            try:
                os.chdir(base)
                report = doctor(db_path, faiss_loader=lambda: FakeFaiss())
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(report["status"], "ok")

    def test_doctor_reports_faiss_dimension_mismatch(self) -> None:
        class Index:
            ntotal = 1
            d = 3

        class FakeFaiss:
            def read_index(self, path: str) -> Index:
                return Index()

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            db_path = self._create_dense_store(base)
            (base / "vectors.faiss").write_bytes(b"index")
            (base / "vector_ids.jsonl").write_text('{"unit_id": "u1"}\n', encoding="utf-8")
            report = doctor(db_path, faiss_loader=lambda: FakeFaiss())

        codes = {item["code"] for item in report["diagnoses"]}
        self.assertIn("faiss_index_dimension_mismatch", codes)

    def test_doctor_rejects_nonintegral_faiss_counts(self) -> None:
        for ntotal in (True, "1", 1.5, -1):
            with self.subTest(ntotal=ntotal), tempfile.TemporaryDirectory() as tmpdir:
                class Index:
                    d = 2

                index = Index()
                index.ntotal = ntotal

                class FakeFaiss:
                    def read_index(self, path: str) -> Index:
                        return index

                base = Path(tmpdir)
                db_path = self._create_dense_store(base)
                (base / "vectors.faiss").write_bytes(b"index")
                (base / "vector_ids.jsonl").write_text('{"unit_id": "u1"}\n', encoding="utf-8")
                report = doctor(db_path, faiss_loader=lambda: FakeFaiss())

            codes = {item["code"] for item in report["diagnoses"]}
            self.assertIn("faiss_index_invalid", codes)


def diagnostics_test_paragraph():
    from nihaisha_kg.pdf_vector import ParsedParagraph

    return ParsedParagraph(
        paragraph_id="p1", doc_id="doc", source_path="/tmp/doc.pdf", title="test",
        page_start=1, page_end=1, text="桂枝汤主之。",
    )


if __name__ == "__main__":
    unittest.main()
