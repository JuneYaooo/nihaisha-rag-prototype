# Knowledge Structure Phase Two Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal evidence-first knowledge schema, migrate reliable existing data into it, expose full paragraph citations and graph relations, and replace fixed leakage fixtures with generated synthetic values.

**Architecture:** A builder-side staging migration copies the current portable asset generation, adds five normalized tables, validates provenance, and emits a compact quality report. Runtime remains read-only and gains small query helpers over the new tables plus full-paragraph citation fields; the existing search and answer pipeline stays compatible.

**Tech Stack:** Python 3.11+, SQLite/FTS5, unittest, existing FAISS assets, Git LFS.

---

## File map

- Builder create `scripts/migrate_knowledge_structure.py`: staging-only schema creation, migration, validation, report, and atomic publication.
- Builder create `tests/test_migrate_knowledge_structure.py`: isolated SQLite fixtures and migration tests.
- Builder modify `tests/test_migrate_portable_source_paths.py`: generate path and credential sentinels instead of fixed private narratives.
- Runtime modify `nihaisha_kg/pdf_vector.py`: read-only graph lookup and full citation paragraph/context fields.
- Runtime modify `nihaisha_kg/cli.py`: include the new fields in JSON without changing concise terminal rendering.
- Runtime modify `tests/test_pdf_vector.py`: graph/provenance and citation expansion tests; replace fixed leakage fixtures.
- Runtime modify `README.md` and `docs/BUILD_AND_UPDATE.md`: document the schema, quality report, and builder-only publication path.
- Runtime replace `data/pdf_rag_bge_m3/rag.sqlite` and update `data/pdf_rag_bge_m3/manifest.json`, `evals/baseline_v1.json`: publish the validated generation and provenance.

### Task 1: Synthetic security fixtures

**Files:**
- Modify: `tests/test_pdf_vector.py`
- Modify: `/Users/june/code/github/nihaisha-rag-builder/tests/test_migrate_portable_source_paths.py`

- [ ] **Step 1: Add generated fixture helpers**

Add this helper in both test modules and use it to construct POSIX, Windows and URI inputs:

```python
def synthetic_private_values() -> dict[str, str]:
    marker = uuid.uuid4().hex
    account = f"account-{marker}"
    directory = f"segment-{marker}"
    credential = f"value-{marker}"
    return {
        "marker": marker,
        "account": account,
        "directory": directory,
        "credential": credential,
        "posix": f"/mnt/{account}/{directory}/source.pdf",
        "windows": rf"Z:\{account}\{directory}\source.pdf",
        "uri": f"https://{account}:{credential}@example.test/{directory}/source.pdf?signature={credential}",
    }
```

- [ ] **Step 2: Replace fixed identity and credential literals**

Build all private path and credential assertions from the helper. Keep platform-shape constants such as `/mnt/`, `Z:\\`, `https://` and `example.test`; assert generated marker fragments are absent from public outputs. Remove fixed `alice`, `/Users/...`, `private.example`, and reusable credential values from active test fixtures.

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m unittest tests.test_pdf_vector -v
cd /Users/june/code/github/nihaisha-rag-builder
python -m unittest tests.test_migrate_portable_source_paths -v
```

Expected: both suites pass and `rg -n 'alice|private\.example|/Users/alice' tests` returns no matches in either repository.

- [ ] **Step 4: Commit each repository**

```bash
git add tests/test_pdf_vector.py
git commit -m "test: generate private boundary fixtures"
```

```bash
cd /Users/june/code/github/nihaisha-rag-builder
git add tests/test_migrate_portable_source_paths.py
git commit -m "test: generate migration boundary fixtures"
```

### Task 2: Builder knowledge schema migration

**Files:**
- Create: `/Users/june/code/github/nihaisha-rag-builder/scripts/migrate_knowledge_structure.py`
- Create: `/Users/june/code/github/nihaisha-rag-builder/tests/test_migrate_knowledge_structure.py`
- Modify: `/Users/june/code/github/nihaisha-rag-builder/README.md`

- [ ] **Step 1: Write a failing isolated migration test**

Create a temporary runtime asset with `paragraphs` and `knowledge_units`. Include one verified formula relation whose quote is a literal substring of its paragraph and one invalid relation whose quote is absent. Assert the migration creates `documents`, `evidence_records`, `entities`, `entity_aliases`, and `relations`; the valid relation is `needs_review`, the invalid relation is omitted and counted in the report, and the source database hash is unchanged.

```python
report = migrate_knowledge_structure(runtime, staging)
with sqlite3.connect(staging / "rag.sqlite") as conn:
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM evidence_records").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0] == 1
    assert conn.execute("SELECT review_status FROM relations").fetchone()[0] == "needs_review"
assert report["rejected_candidates"] == 1
assert sha256(source_db) == source_hash
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
cd /Users/june/code/github/nihaisha-rag-builder
python -m unittest tests.test_migrate_knowledge_structure -v
```

Expected: import failure because `scripts.migrate_knowledge_structure` does not exist.

- [ ] **Step 3: Implement the normalized schema**

Create the five tables with foreign keys and checks. Use deterministic SHA-256 prefixes for IDs and keep the accepted migration deliberately conservative:

```sql
CREATE TABLE documents (
  document_id TEXT PRIMARY KEY,
  canonical_title TEXT NOT NULL,
  source_layer TEXT NOT NULL CHECK(source_layer IN ('course_primary','classic_primary','reference_secondary','derived')),
  media_type TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  logical_source_path TEXT NOT NULL UNIQUE,
  ingestion_version TEXT NOT NULL
);
CREATE TABLE evidence_records (
  evidence_id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(document_id),
  paragraph_id TEXT NOT NULL UNIQUE REFERENCES paragraphs(paragraph_id),
  locator TEXT NOT NULL,
  original_text TEXT NOT NULL,
  normalized_text TEXT NOT NULL,
  previous_evidence_id TEXT,
  next_evidence_id TEXT,
  quality_flags TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE entities (
  entity_id TEXT PRIMARY KEY,
  entity_type TEXT NOT NULL,
  canonical_name TEXT NOT NULL,
  normalized_key TEXT NOT NULL,
  definition_status TEXT NOT NULL,
  created_by_version TEXT NOT NULL,
  UNIQUE(entity_type, normalized_key)
);
CREATE TABLE entity_aliases (
  alias_id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL REFERENCES entities(entity_id),
  surface_form TEXT NOT NULL,
  normalized_form TEXT NOT NULL,
  evidence_id TEXT NOT NULL REFERENCES evidence_records(evidence_id),
  confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
  review_status TEXT NOT NULL CHECK(review_status IN ('auto_accepted','needs_review','reviewed','rejected'))
);
CREATE TABLE relations (
  relation_id TEXT PRIMARY KEY,
  subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
  predicate TEXT NOT NULL,
  object_entity_id TEXT REFERENCES entities(entity_id),
  literal_value TEXT,
  evidence_id TEXT NOT NULL REFERENCES evidence_records(evidence_id),
  source_layer TEXT NOT NULL,
  confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
  extraction_method TEXT NOT NULL,
  extractor_version TEXT NOT NULL,
  review_status TEXT NOT NULL CHECK(review_status IN ('auto_accepted','needs_review','reviewed','rejected')),
  CHECK((object_entity_id IS NULL) != (literal_value IS NULL))
);
```

Document classification is explicit and minimal: `黄帝内经原文和翻译.pdf` is `classic_primary`; the other current documents are `course_primary`. Create an evidence row for every paragraph. Only migrate knowledge candidates when the nonempty `evidence_quote` occurs literally in the paragraph and the subject is a compact non-generic label; preserve the old predicate as a candidate relation mapped through a small controlled mapping. Mark all migrated relations `needs_review`, never `reviewed` or `auto_accepted`.

- [ ] **Step 4: Add validation and report**

Validate foreign keys, source layers, relationship evidence joins, quote containment, and absence of absolute paths. Emit `knowledge_structure_report.json` with counts for documents, evidence, entities, aliases, relations, rejected candidates, orphan entities, evidence coverage, relation types, review statuses, and migration version. No original private path or full paragraph text enters the report.

- [ ] **Step 5: Run builder tests and commit**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all builder tests pass.

```bash
git add scripts/migrate_knowledge_structure.py tests/test_migrate_knowledge_structure.py README.md
git commit -m "feat: migrate evidence-first knowledge structure"
```

### Task 3: Runtime graph and complete citation access

**Files:**
- Modify: `nihaisha_kg/pdf_vector.py`
- Modify: `nihaisha_kg/cli.py`
- Modify: `tests/test_pdf_vector.py`

- [ ] **Step 1: Write failing runtime tests**

Add an in-memory fixture with the new tables. Verify `knowledge_relations(store, "桂枝汤")` returns only non-rejected relations with evidence metadata. Verify answer citations contain the concise `evidence_quote`, complete `paragraph_text`, and optional `previous_paragraph_id`/`next_paragraph_id` without using derived text as original evidence.

```python
relations = pdf_vector.knowledge_relations(store, "桂枝汤")
self.assertEqual(relations[0]["predicate"], "treated_by")
self.assertEqual(relations[0]["source_layer"], "course_primary")
self.assertEqual(answer["citations"][0]["paragraph_text"], paragraph)
```

- [ ] **Step 2: Run focused tests to verify RED**

Run:

```bash
python -m unittest tests.test_pdf_vector.RuntimeKnowledgeStructureTests -v
```

Expected: failure because `knowledge_relations` and citation expansion fields do not exist.

- [ ] **Step 3: Implement read-only graph lookup**

Add `knowledge_relations(store_path, entity_name, limit=20)` with a hard maximum of 100. Join `entities`, `relations`, `evidence_records`, and `documents`; filter `review_status != 'rejected'`; return portable source, locator, original evidence, confidence, extractor version and review status. If the new schema is absent, return an empty list to preserve compatibility with older databases.

- [ ] **Step 4: Add complete paragraph citation fields**

When building citations, copy the selected result paragraph into `paragraph_text` and include stable paragraph/context IDs when available. Keep `evidence_quote` at its existing 220/520-character bounds and keep the terminal formatter concise; JSON callers receive the expandable fields.

- [ ] **Step 5: Run runtime tests and commit**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all runtime tests pass.

```bash
git add nihaisha_kg/pdf_vector.py nihaisha_kg/cli.py tests/test_pdf_vector.py
git commit -m "feat: expose evidence graph and full citations"
```

### Task 4: Stage, publish and verify the real asset

**Files:**
- Modify: `data/pdf_rag_bge_m3/rag.sqlite`
- Modify: `data/pdf_rag_bge_m3/manifest.json`
- Modify: `evals/baseline_v1.json`
- Modify: `README.md`
- Modify: `docs/BUILD_AND_UPDATE.md`

- [ ] **Step 1: Generate a staging candidate**

Run the builder migration against the runtime repository into a temporary directory outside runtime `data/`. Do not modify the source asset in place.

```bash
cd /Users/june/code/github/nihaisha-rag-builder
python scripts/migrate_knowledge_structure.py \
  --runtime /Users/june/code/github/nihaisha-rag-prototype \
  --staging /tmp/nihaisha-knowledge-generation
```

Expected: candidate contains all four runtime assets plus `knowledge_structure_report.json`; source database hash is unchanged.

- [ ] **Step 2: Validate candidate quality**

Run SQLite integrity and foreign-key checks, verify every relation joins to evidence and all report coverage values agree with SQL counts. Run runtime doctor and the stored evaluation against the candidate by using a temporary asset root or staged copy.

- [ ] **Step 3: Publish the complete generation**

Use the builder's directory-level publication operation. Update manifest schema/provenance with migration version and report summary; update baseline database LFS OID and hashes after Git LFS stages the new SQLite.

- [ ] **Step 4: Update documentation**

Document evidence layers, review-status semantics, graph limitations, complete paragraph citation fields, quality-report command, and builder-only staging/publish boundary. State that migrated relations are `needs_review` and must not be called complete or expert-verified.

- [ ] **Step 5: Run final gates**

Runtime:

```bash
python -m unittest discover -s tests -v
python -m nihaisha_kg.cli doctor --json
python -m nihaisha_kg.cli evaluate --golden evals/golden_v1.jsonl --mode hybrid --limit 10 --json
git lfs fsck
git diff --check
```

Builder:

```bash
cd /Users/june/code/github/nihaisha-rag-builder
python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests and integrity checks pass; live evaluation matches the stored baseline; no published path contains a machine directory.

- [ ] **Step 6: Commit runtime publication**

```bash
git add data/pdf_rag_bge_m3/rag.sqlite data/pdf_rag_bge_m3/manifest.json \
  evals/baseline_v1.json README.md docs/BUILD_AND_UPDATE.md
git commit -m "feat: publish evidence-first knowledge graph"
```
