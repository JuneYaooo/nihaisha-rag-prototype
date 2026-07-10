from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from numbers import Integral
from pathlib import Path
from typing import Callable


REQUIRED_TABLES = (
    "paragraphs",
    "retrieval_units",
    "paragraphs_fts",
    "knowledge_units",
)
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
MAX_MAPPING_FILE_BYTES = 64 * 1024 * 1024
MAX_MAPPING_LINE_BYTES = 1024 * 1024
MAX_MAPPING_RECORDS = 500_000
MAPPING_INSERT_BATCH_SIZE = 1_000


def _resolve_artifact_path(db_path: Path, configured: str, default_name: str) -> Path:
    candidate = Path(configured) if configured else Path(default_name)
    if candidate.is_absolute():
        return candidate
    db_relative = db_path.parent / candidate
    if configured and not db_relative.exists() and candidate.exists():
        # Older builds stored paths relative to the build process working directory.
        return candidate.resolve()
    return db_relative


def _strict_index_integer(value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("index metadata must be integral")
    parsed = int(value)
    if parsed < (1 if positive else 0):
        raise ValueError("index metadata is out of range")
    return parsed


def _inspect_mapping(db_path: Path, ids_path: Path) -> dict[str, object]:
    """Stream and exactly compare a mapping using a disk-backed temporary SQLite table."""
    try:
        file_size = ids_path.stat().st_size
        if file_size > MAX_MAPPING_FILE_BYTES:
            return {"error": "limits", "bytes": file_size}
        uri = db_path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("PRAGMA temp_store=FILE")
            conn.execute("CREATE TEMP TABLE doctor_mapping_ids(unit_id TEXT PRIMARY KEY)")
            batch: list[tuple[str]] = []
            count = 0
            processed_bytes = 0
            with ids_path.open("rb") as mapping_file:
                while True:
                    raw_line = mapping_file.readline(MAX_MAPPING_LINE_BYTES + 1)
                    if not raw_line:
                        break
                    processed_bytes += len(raw_line)
                    if processed_bytes > MAX_MAPPING_FILE_BYTES:
                        return {"error": "limits", "bytes": processed_bytes}
                    if len(raw_line) > MAX_MAPPING_LINE_BYTES:
                        return {"error": "limits", "line": count + 1}
                    if not raw_line.strip():
                        continue
                    if count >= MAX_MAPPING_RECORDS:
                        return {"error": "limits", "records": count}
                    payload = json.loads(raw_line.decode("utf-8"))
                    unit_id = payload.get("unit_id") if isinstance(payload, dict) else None
                    if not isinstance(unit_id, str) or not unit_id.strip():
                        raise ValueError("invalid unit ID")
                    batch.append((unit_id,))
                    count += 1
                    if len(batch) >= MAPPING_INSERT_BATCH_SIZE:
                        conn.executemany(
                            "INSERT INTO doctor_mapping_ids(unit_id) VALUES (?)",
                            batch,
                        )
                        batch.clear()
            if batch:
                conn.executemany("INSERT INTO doctor_mapping_ids(unit_id) VALUES (?)", batch)
            unknown = conn.execute(
                """
                SELECT 1 FROM doctor_mapping_ids AS m
                LEFT JOIN retrieval_units AS r ON r.unit_id = m.unit_id
                WHERE r.unit_id IS NULL LIMIT 1
                """
            ).fetchone()
            missing = conn.execute(
                """
                SELECT 1 FROM retrieval_units AS r
                LEFT JOIN doctor_mapping_ids AS m ON m.unit_id = r.unit_id
                WHERE m.unit_id IS NULL LIMIT 1
                """
            ).fetchone()
            return {"count": count, "ids_mismatch": unknown is not None or missing is not None}
    except sqlite3.IntegrityError:
        return {"error": "duplicate"}
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        MemoryError,
        OverflowError,
        RecursionError,
        sqlite3.DatabaseError,
        TypeError,
        ValueError,
    ) as exc:
        return {"error": "invalid", "exception": type(exc).__name__}


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

    try:
        index_path = _resolve_artifact_path(
            db_path,
            meta.get("faiss_index", ""),
            "vectors.faiss",
        )
        ids_path = _resolve_artifact_path(
            db_path,
            meta.get("faiss_ids", ""),
            "vector_ids.jsonl",
        )
        artifacts_present = index_path.is_file() and ids_path.is_file()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        add("faiss_path_invalid", "error", "FAISS artifact paths are invalid", error=type(exc).__name__)
        return report()
    if not metadata_valid:
        if not ids_path.is_file():
            return report()
    elif dense and not artifacts_present:
        add("faiss_files_missing", "error", "vectors.faiss or vector_ids.jsonl is missing")
    elif artifacts_present:
        add("faiss_files", "ok", "FAISS index and ID mapping are present")
    else:
        return report()

    mapping_count: int | None = None
    if ids_path.is_file():
        inspection = _inspect_mapping(db_path, ids_path)
        error = inspection.get("error")
        if error == "limits":
            add(
                "faiss_mapping_limits_exceeded",
                "error",
                "FAISS ID mapping exceeds diagnostic safety limits",
            )
        elif error == "duplicate":
            add("faiss_mapping_duplicate_ids", "error", "FAISS ID mapping contains duplicate IDs")
        elif error:
            add(
                "faiss_mapping_invalid",
                "error",
                "FAISS ID mapping could not be parsed",
                error=str(inspection.get("exception", "InvalidMapping")),
            )
        else:
            mapping_count = int(inspection["count"])
            add("faiss_mapping", "ok", "FAISS ID mapping is readable", count=mapping_count)
            if unit_count is not None and mapping_count != unit_count:
                add(
                    "faiss_mapping_count_mismatch",
                    "error",
                    "SQLite and FAISS mapping counts differ",
                    sqlite=unit_count,
                    mapping=mapping_count,
                )
            if bool(inspection.get("ids_mismatch")):
                add("faiss_mapping_ids_mismatch", "error", "SQLite and FAISS mapping IDs differ")

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
            ntotal = _strict_index_integer(index.ntotal)
            index_dim = _strict_index_integer(index.d, positive=True)
            add("faiss_index", "ok", "FAISS index is readable", count=ntotal)
            if mapping_count is not None and ntotal != mapping_count:
                add(
                    "faiss_index_count_mismatch",
                    "error",
                    "FAISS index and ID mapping counts differ",
                    index=ntotal,
                    mapping=mapping_count,
                )
            if unit_count is not None and ntotal != unit_count:
                add(
                    "faiss_sqlite_count_mismatch",
                    "error",
                    "FAISS index and SQLite retrieval-unit counts differ",
                    index=ntotal,
                    sqlite=unit_count,
                )
            if index_dim != vector_dim:
                add(
                    "faiss_index_dimension_mismatch",
                    "error",
                    "FAISS index and vector metadata dimensions differ",
                    index=index_dim,
                    metadata=vector_dim,
                )
        except Exception as exc:
            add("faiss_index_invalid", "error", "FAISS index could not be read", error=type(exc).__name__)

    return report()
