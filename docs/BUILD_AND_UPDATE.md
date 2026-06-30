# Build And Update Guide

This document is for maintainers who rebuild or incrementally update the bundled Nihaisha RAG database. It is intentionally separate from `README.md`: the README is for users of the runtime Skill, while the dedicated builder Skill should live in a separate `nihaisha-rag-builder` repository.

## What Gets Committed

The repository should only commit the usable Skill and the final database assets:

- `SKILL.md`
- `README.md`
- `agents/openai.yaml`
- `nihaisha_kg/`
- `tests/`
- `data/pdf_rag_bge_m3/rag.sqlite`
- `data/pdf_rag_bge_m3/manifest.json`
- `data/pdf_rag_bge_m3/vectors.faiss`
- `data/pdf_rag_bge_m3/vector_ids.jsonl`

`rag.sqlite` is the source of truth. It stores original paragraphs, PDF source paths, page numbers, retrieval units, vectors, knowledge units, FTS indexes, and metadata. `vectors.faiss` is only the acceleration layer. `vector_ids.jsonl` maps FAISS rows back to `retrieval_units.unit_id`.

Intermediate artifacts should be kept locally for audit while building, but they should not be committed. The ignore rules cover local trace and staging directories such as `data/**/traces/`, `data/**/incoming/`, `data/**/incremental/`, `data/**/staging/`, `data/**/tmp/`, and SQLite WAL/SHM files.

## Current Build Method

The production database is built from PDFs using the formal CLI in `nihaisha_kg.cli`. The CLI wraps the mature functions in `nihaisha_kg.pdf_vector`:

- `build_pdf_vector_store`
- `augment_pdf_vector_store_questions`
- `LocalVectorStore.rebuild_text_index`
- `LocalVectorStore.rebuild_knowledge_units`
- `build_faiss_vector_index`

The full build flow is:

1. Parse PDFs into source-grounded paragraphs with document path and page metadata.
2. Build multi-level retrieval units: sentence, sentence window, paragraph, and generated question units.
3. Embed retrieval units with the same embedding family used by the target database. The current production database uses SiliconFlow `BAAI/bge-m3` dense embeddings with 1024 dimensions.
4. Store paragraphs, retrieval units, dense vectors, and metadata in `rag.sqlite`.
5. Rebuild the paragraph FTS index.
6. Extract local-rule knowledge units into `knowledge_units`.
7. Write build traces locally.
8. Write or update `manifest.json`.
9. Build `vectors.faiss` and `vector_ids.jsonl`.
10. Run verification before committing.

Recommended full rebuild command:

```bash
python3 -m nihaisha_kg build \
  --pdf-dir /Users/june/code/data/nihaisha_deeliu/nihaisha \
  --out data/pdf_rag_bge_m3 \
  --embedding siliconflow \
  --model BAAI/bge-m3 \
  --window-size 6 \
  --overlap 2 \
  --trace-dir data/pdf_rag_bge_m3/traces
```

After vectors change, rebuild FAISS:

```bash
python3 -m pip install -e ".[faiss]"
python3 -m nihaisha_kg build-faiss --db data/pdf_rag_bge_m3/rag.sqlite
```

`data/pdf_rag_bge_m3/traces/` is intentionally ignored. Keep it locally until the update has been reviewed. Do not commit it unless there is a deliberate reason to preserve a sampled audit artifact.

## Incremental Update Strategy

Do not directly mutate the production database as the first step. Use a staging build first.

For a small batch of new PDFs:

1. Put new source files under an ignored local working directory such as `data/pdf_rag_bge_m3/incoming/`.
2. Build a staging database under an ignored directory such as `data/pdf_rag_bge_m3/incremental/<date-or-topic>/`.
3. Inspect staging `manifest.json` counts, paragraph quality, source paths, page metadata, and sample retrieval results.
4. Decide whether to merge or do a clean full rebuild.

For substantial updates, parser changes, chunking changes, embedding changes, knowledge extraction changes, or any change that affects retrieval-unit identity, prefer a full rebuild. It is more reliable because sentence windows, generated questions, knowledge units, FTS indexes, and FAISS all depend on the same paragraph and retrieval-unit set.

For a narrow additive update where the parsing and embedding rules are unchanged, merging can be considered, but only after validating:

- no duplicate document source paths;
- stable paragraph IDs for existing documents;
- expected paragraph and retrieval-unit counts for new documents;
- no vector kind or vector dimension mismatch;
- knowledge unit extraction version is unchanged;
- FAISS is rebuilt after merge.

If any of these checks fail, discard the staging database and do a full rebuild.

## Maintenance Commands

The formal CLI exposes the database maintenance surface:

```bash
python3 -m nihaisha_kg build
python3 -m nihaisha_kg augment-questions
python3 -m nihaisha_kg rebuild-text-index
python3 -m nihaisha_kg rebuild-knowledge-units
python3 -m nihaisha_kg build-faiss
python3 -m nihaisha_kg stats
```

Use `python3 -m nihaisha_kg <command> --help` for exact flags. Keep this command surface stable so future maintainers do not depend on hidden module entrypoints.

## Trace Policy

Trace files are useful for local review and debugging. They should include:

- parsed paragraph rows;
- retrieval unit rows;
- build events;
- build configuration;
- knowledge unit rows.

Trace files must not contain API keys, authorization headers, or provider secrets. The code writes trace metadata with a secret policy note, and `.env` is ignored. Before publishing, still scan for secrets.

Because traces can become large and may duplicate source text, they are ignored by default. The committed `manifest.json` is the compact build summary that belongs in Git.

## Verification Checklist

Before committing a database update:

1. Run the unit test suite.
2. Compile the Python modules.
3. Run Skill validation.
4. Run database stats and check document, paragraph, retrieval-unit, knowledge-unit, embedding, vector dimension, and FAISS counts.
5. Run a FAISS self-check when `vectors.faiss` changes: read a stored dense vector from SQLite, query FAISS, and confirm the nearest row maps back correctly.
6. Run representative answer/search checks for dosage, source-location, and clinical-boundary questions.
7. Scan tracked text files for API keys and obsolete version labels.
8. Confirm large database assets are Git LFS pointers in Git, not raw blobs.
9. Confirm ignored trace and staging directories are not staged.

Useful verification commands:

```bash
python3 -m unittest tests.test_pdf_vector -v
python3 -m py_compile nihaisha_kg/pdf_vector.py nihaisha_kg/cli.py
python3 /Users/june/.codex/skills/.system/skill-creator/scripts/quick_validate.py .
python3 -m nihaisha_kg stats
git lfs ls-files
git cat-file -s :data/pdf_rag_bge_m3/rag.sqlite
git cat-file -s :data/pdf_rag_bge_m3/vectors.faiss
```

The `git cat-file -s` checks should show small pointer objects for the LFS-managed files, not hundreds of megabytes.

## Release Rules

Commit final data assets only after verification. The final pushed state should include updated `rag.sqlite`, `manifest.json`, `vectors.faiss`, and `vector_ids.jsonl` when the database changes.

Use Git LFS for `rag.sqlite` and `vectors.faiss`. If LFS upload fails because of quota or network issues, do not push a partial release. Resolve the LFS problem first or move the large assets to a deliberate external artifact store.

Never commit `.env` or real API keys.
