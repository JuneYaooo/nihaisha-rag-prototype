---
name: nihaisha-rag-builder
description: Use when rebuilding, incrementally updating, validating, or publishing the Nihaisha RAG database assets, including PDF ingestion, retrieval-unit generation, knowledge graph extraction, BAAI/bge-m3 embedding, FAISS indexing, manifest updates, trace handling, Git LFS checks, and database release safety.
---

# Nihaisha RAG Builder

Use this skill for database construction and maintenance. Do not use it for ordinary user questions about Nihaisha course content; use the root `nihaisha-rag-prototype` skill for retrieval and answers.

## Scope

This skill owns:

- full database rebuilds from PDFs;
- staged incremental updates;
- paragraph, retrieval-unit, question-unit, and knowledge-unit generation;
- BAAI/bge-m3 embedding via SiliconFlow or local model;
- FTS and FAISS rebuilds;
- manifest and local trace review;
- Git LFS and release verification.

The runtime skill owns search, answer synthesis, citations, and medical safety response behavior.

## Repo Assumptions

Run from the repository root containing:

```text
SKILL.md
docs/BUILD_AND_UPDATE.md
nihaisha_kg/
data/pdf_rag_bge_m3/
```

Before a real rebuild or update, read the full maintenance guide:

```text
../../docs/BUILD_AND_UPDATE.md
```

## Build Assets

Commit only final assets:

```text
data/pdf_rag_bge_m3/rag.sqlite
data/pdf_rag_bge_m3/manifest.json
data/pdf_rag_bge_m3/vectors.faiss
data/pdf_rag_bge_m3/vector_ids.jsonl
```

Do not commit local traces, incoming files, staging databases, incremental scratch directories, `.env`, SQLite WAL/SHM files, or API keys. These are ignored by `.gitignore`.

## Command Surface

Use the formal CLI, not hidden module entrypoints:

```bash
python3 -m nihaisha_kg build
python3 -m nihaisha_kg augment-questions
python3 -m nihaisha_kg rebuild-text-index
python3 -m nihaisha_kg rebuild-knowledge-units
python3 -m nihaisha_kg build-faiss
python3 -m nihaisha_kg stats
```

Run `python3 -m nihaisha_kg <command> --help` for exact flags.

## Update Policy

For new source material, build a staging database first under an ignored local directory. Inspect manifest counts, paragraph quality, page metadata, source paths, retrieval behavior, and knowledge-unit extraction before touching production assets.

Prefer a full rebuild when any of these change:

- parser behavior;
- chunking strategy;
- question generation;
- embedding backend, model, vector dimension, or vector kind;
- knowledge extraction rules;
- retrieval-unit identity;
- existing source PDFs.

Use additive merge only for narrow new-document updates where parsing, chunking, embedding, and extraction rules are unchanged. Rebuild FAISS after any vector change.

## Verification Gate

Before committing or pushing, run:

```bash
python3 -m unittest tests.test_pdf_vector -v
python3 -m py_compile nihaisha_kg/pdf_vector.py nihaisha_kg/cli.py
python3 /Users/june/.codex/skills/.system/skill-creator/scripts/quick_validate.py .
python3 -m nihaisha_kg stats
git lfs ls-files
git cat-file -s :data/pdf_rag_bge_m3/rag.sqlite
git cat-file -s :data/pdf_rag_bge_m3/vectors.faiss
```

Also scan tracked text files for real API keys and obsolete version labels. The LFS `cat-file` checks should show small pointer sizes, not raw database file sizes.

## Release Rule

If LFS upload fails, do not push a partial release. Resolve Git LFS first or move large assets to an intentional external artifact store.

