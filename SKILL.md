---
name: nihaisha-rag-prototype
description: Source-grounded RAG skill for 倪海厦 PDF course materials. Use when the user asks about Ni Haixia / 倪海厦 course content, source lookup, page citations, formula-pattern comparison, dosage-unit lookup, symptoms, methods, original paragraphs, knowledge graph triples, vector retrieval, or evidence-based answers from the bundled SQLite knowledge graph and BAAI/bge-m3 vector database.
---

# Nihaisha RAG Prototype

Use this skill to answer from the bundled local RAG database. Always ground factual claims in retrieved PDF evidence.

This skill is for course-material study, source lookup, and TCM theory organization only. Do not provide personal diagnosis, prescriptions, dosage decisions, herb purchasing advice, acupuncture/external-treatment instructions, or self-treatment plans.

For database rebuilds, incremental updates, FAISS regeneration, manifest maintenance, trace review, or Git LFS release checks, use the separate `nihaisha-rag-builder` Skill repository instead.

## Skill Root

Run commands from the directory containing this `SKILL.md`. If working from another directory, pass the absolute database path:

```text
data/pdf_rag_bge_m3/rag.sqlite
```

## Data Included

The skill includes both a lightweight knowledge graph and a vector database in SQLite:

```text
data/pdf_rag_bge_m3/rag.sqlite
data/pdf_rag_bge_m3/manifest.json
data/pdf_rag_bge_m3/vectors.faiss
data/pdf_rag_bge_m3/vector_ids.jsonl
```

Key tables:

```text
paragraphs          original PDF paragraphs with source path and page numbers
retrieval_units     sentence/window/paragraph/question units with BAAI/bge-m3 vectors
knowledge_units     subject-predicate-object knowledge graph units
paragraphs_fts      exact-text FTS index
knowledge_units_fts knowledge graph FTS index
```

The final evidence returned to the user should be original paragraphs with PDF names and page numbers, not only derived triples.

FAISS files accelerate vector search:

```text
vectors.faiss     nearest-neighbor index
vector_ids.jsonl  FAISS row -> retrieval_unit_id mapping
```

SQLite remains the source of truth for original paragraphs, knowledge graph units, FTS, and citations.

## Retrieval Policy

Prefer `hybrid` retrieval for normal questions because it combines:

```text
vector    semantic recall from BAAI/bge-m3 vectors
text      exact phrase/term recall from paragraphs_fts
knowledge knowledge graph recall from knowledge_units
```

Use `knowledge` for direct lookup of dose units, source locations, formula-pattern facts, method names, and structured comparisons.

Use `text` for exact quotes, rare terms, or when no embedding backend is available.

## Embedding Backend

Recommended path: SiliconFlow API with `BAAI/bge-m3`.

SiliconFlow embedding docs:

```text
https://api-docs.siliconflow.cn/docs/api/embeddings-post
```

Set:

```text
SILICONFLOW_API_KEY=...
```

Local `BAAI/bge-m3` is optional and must not be assumed installed. Mention it only when the user wants offline/no-API retrieval:

```bash
python3 -m pip install -e ".[local]"
```

Then use `--embedding local-bge-m3`.

## Commands

Stats:

```bash
python3 -m nihaisha_kg stats
```

Knowledge graph search:

```bash
python3 -m nihaisha_kg search "一钱是多少克" --mode knowledge --limit 5
python3 -m nihaisha_kg search "木香饼热熨法 出处" --mode knowledge --limit 5
```

Hybrid vector + text + knowledge search:

```bash
python3 -m nihaisha_kg search "下利 恶心 黄芩加半夏生姜汤" --mode hybrid --limit 8
```

Grounded answer draft with citations:

```bash
python3 -m nihaisha_kg answer "木香饼热熨法是来自哪一本书哪一段？" --mode hybrid --limit 8
```

Use `--json` when programmatic access to citations, scores, matched knowledge units, or raw result records is needed.

The command returns retrieved evidence, citations, and a conservative local draft. The agent should use its own reasoning and language ability to reorganize the evidence into a fluent final answer, but must stay strictly within the returned citations and original paragraphs.

## Answer Requirements

Use this final answer structure:

```text
1. 总结
2. 原文依据
3. 关联知识点与辨证线索
4. 安全边界
```

The summary comes first. Keep it concise and useful; every factual claim in the summary must carry citation markers such as `[1]`.

The original-evidence section must make citations traceable back to the PDF text. For each citation, include the PDF name, page number, and a short original excerpt from the retrieved paragraph. Do not cite derived knowledge graph triples unless they are backed by original paragraph evidence.

The related-knowledge section should use `related_knowledge_units` from `answer --json` when available. It may summarize formula-pattern, symptom, dosage, method, comparison, or caution units to help the user reason about辨证, but each point must remain tied to retrieved original evidence and must not become a personal diagnosis or prescription.

If evidence is insufficient, say so in the summary and show what was retrieved. Do not fill gaps with model memory or outside knowledge.

## Safety Requirements

- Frame answers as course-material study, source lookup, or TCM theory organization.
- Do not provide individualized diagnosis, prescription, dosage, decoction method, purchasing advice, or treatment plan.
- For clinical scenarios, list course evidence, possible 方证 clues, and missing differentiation information only.
- For all dose, formula, prescription, herb, external-treatment, or clinical answers, include this warning:

```text
涉及剂量、方药或处方线索时必须谨慎：不同人的体质不同，病情阶段、兼证、年龄、基础病和用药史都不同；现代药材来源、炮制、浓度和药效也和以前差很多。建议去线下正规中医渠道面诊辨证，不要私自购药有风险。
```
