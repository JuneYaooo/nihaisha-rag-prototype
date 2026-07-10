from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Callable


REQUIRED_TABLES = (
    "paragraphs",
    "retrieval_units",
    "paragraphs_fts",
    "knowledge_units",
)
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def doctor(db_path: Path, faiss_loader: Callable[[], object | None]) -> dict[str, object]:
    """Inspect a local retrieval database without trusting any of its artifacts."""
    db_path = Path(db_path)
    checks: list[dict[str, object]] = []
    diagnoses: list[dict[str, object]] = []

    def add(code: str, status: str, message: str, **details: object) -> None:
        item: dict[str, object] = {"code": code, "status": status, "message": message}
        if details:
            item["details"] = details
        checks.append(item)
        if status != "ok":
            diagnoses.append(item)

    def report() -> dict[str, object]:
        levels = {str(item["status"]) for item in diagnoses}
        status = "error" if "error" in levels else "warning" if "warning" in levels else "ok"
        return {
            "status": status,
            "db_path": str(db_path),
            "checks": checks,
            "diagnoses": diagnoses,
        }

    if not db_path.exists():
        add("db_missing", "error", "database file is missing")
        return report()
    if not db_path.is_file():
        add("db_not_file", "error", "database path is not a regular file")
        return report()

    try:
        size = db_path.stat().st_size
        with db_path.open("rb") as db_file:
            prefix = db_file.read(200)
    except OSError as exc:
        add("db_unreadable", "error", "database file could not be inspected", error=type(exc).__name__)
        return report()
    if prefix.startswith(LFS_POINTER_PREFIX):
        add("db_git_lfs_pointer", "error", "database is a Git LFS pointer, not SQLite data")
        return report()
    if size < 4096:
        add("db_suspiciously_small", "warning", "database file is suspiciously small", bytes=size)
    else:
        add("db_file", "ok", "database file is present", bytes=size)

    meta: dict[str, str] = {}
    unit_count: int | None = None
    sqlite_unit_ids: set[str] | None = None
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
            tables = {str(row[0]) for row in rows}
            missing_tables = [name for name in REQUIRED_TABLES if name not in tables]
            if missing_tables:
                add(
                    "schema_missing_tables",
                    "error",
                    "required SQLite tables are missing",
                    tables=missing_tables,
                )
            else:
                add("schema", "ok", "required SQLite tables are present")

            if "meta" not in tables:
                add("meta_missing", "warning", "metadata table is missing")
            else:
                try:
                    meta = {
                        str(row[0]): str(row[1])
                        for row in conn.execute("SELECT key, value FROM meta").fetchall()
                    }
                    add("meta", "ok", "metadata is readable")
                except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
                    add("meta_invalid", "error", "metadata could not be read", error=type(exc).__name__)

            if "retrieval_units" in tables:
                try:
                    unit_count = int(conn.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0])
                    sqlite_unit_ids = {
                        str(row[0])
                        for row in conn.execute("SELECT unit_id FROM retrieval_units").fetchall()
                    }
                    add("retrieval_units", "ok", "retrieval-unit count is readable", count=unit_count)
                except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
                    add(
                        "retrieval_units_invalid",
                        "error",
                        "retrieval units could not be inspected",
                        error=type(exc).__name__,
                    )
    except sqlite3.DatabaseError as exc:
        add("db_invalid_sqlite", "error", "database is not valid readable SQLite", error=type(exc).__name__)
        return report()

    invalid_metadata: list[str] = []
    vector_kind = meta.get("vector_kind", "").strip()
    if vector_kind not in {"sparse", "dense"}:
        invalid_metadata.append("vector_kind")
    if not meta.get("embedding", "").strip():
        invalid_metadata.append("embedding")
    try:
        vector_dim = int(meta.get("vector_dim", ""))
        if vector_dim <= 0:
            raise ValueError("vector dimension must be positive")
    except (TypeError, ValueError):
        vector_dim = 0
        invalid_metadata.append("vector_dim")
    if invalid_metadata:
        add(
            "vector_metadata_invalid",
            "error",
            "vector metadata is missing or invalid",
            fields=invalid_metadata,
        )
    metadata_valid = not invalid_metadata
    dense = metadata_valid and vector_kind == "dense"
    if metadata_valid:
        add("vector_metadata", "ok", "vector metadata is valid", vector_kind=vector_kind)
        if dense:
            add("dense_metadata", "ok", "dense-vector metadata is present", dimensions=vector_dim)
        else:
            return report()

    index_path = db_path.with_name("vectors.faiss")
    ids_path = db_path.with_name("vector_ids.jsonl")
    artifacts_present = index_path.is_file() and ids_path.is_file()
    if not metadata_valid:
        if not ids_path.is_file():
            return report()
    elif dense and not artifacts_present:
        add("faiss_files_missing", "error", "vectors.faiss or vector_ids.jsonl is missing")
    elif artifacts_present:
        add("faiss_files", "ok", "FAISS index and ID mapping are present")
    else:
        return report()

    mapping_ids: list[str] | None = None
    if ids_path.is_file():
        try:
            mapping_ids = []
            for line_number, line in enumerate(ids_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                unit_id = payload.get("unit_id") if isinstance(payload, dict) else None
                if not isinstance(unit_id, str) or not unit_id.strip():
                    raise ValueError(f"invalid unit_id at line {line_number}")
                mapping_ids.append(unit_id)
            if len(mapping_ids) != len(set(mapping_ids)):
                add("faiss_mapping_duplicate_ids", "error", "FAISS ID mapping contains duplicate IDs")
            else:
                add("faiss_mapping", "ok", "FAISS ID mapping is readable", count=len(mapping_ids))
            if unit_count is not None and len(mapping_ids) != unit_count:
                add(
                    "faiss_mapping_count_mismatch",
                    "error",
                    "SQLite and FAISS mapping counts differ",
                    sqlite=unit_count,
                    mapping=len(mapping_ids),
                )
            elif sqlite_unit_ids is not None and set(mapping_ids) != sqlite_unit_ids:
                add("faiss_mapping_ids_mismatch", "error", "SQLite and FAISS mapping IDs differ")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            mapping_ids = None
            add(
                "faiss_mapping_invalid",
                "error",
                "FAISS ID mapping could not be parsed",
                error=type(exc).__name__,
            )

    if not metadata_valid:
        return report()

    faiss: object | None = None
    if dense or artifacts_present:
        try:
            faiss = faiss_loader()
        except Exception as exc:  # third-party imports may fail in many ways
            add("faiss_module_error", "error", "FAISS module could not be loaded", error=type(exc).__name__)
        if faiss is None and not any(item["code"] == "faiss_module_error" for item in diagnoses):
            add("faiss_module_missing", "error", "FAISS Python module is unavailable")
        elif faiss is not None:
            add("faiss_module", "ok", "FAISS Python module is available")

    if faiss is not None and index_path.is_file():
        try:
            index = faiss.read_index(str(index_path))
            ntotal = int(index.ntotal)
            add("faiss_index", "ok", "FAISS index is readable", count=ntotal)
            if mapping_ids is not None and ntotal != len(mapping_ids):
                add(
                    "faiss_index_count_mismatch",
                    "error",
                    "FAISS index and ID mapping counts differ",
                    index=ntotal,
                    mapping=len(mapping_ids),
                )
            if unit_count is not None and ntotal != unit_count:
                add(
                    "faiss_sqlite_count_mismatch",
                    "error",
                    "FAISS index and SQLite retrieval-unit counts differ",
                    index=ntotal,
                    sqlite=unit_count,
                )
        except Exception as exc:
            add("faiss_index_invalid", "error", "FAISS index could not be read", error=type(exc).__name__)

    return report()
