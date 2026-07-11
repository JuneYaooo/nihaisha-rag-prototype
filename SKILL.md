---
name: nihaisha-rag-prototype
description: Use when answering questions about Ni Haixia / 倪海厦 course PDFs, source or page lookup, original paragraphs, formula-pattern comparisons, dosage units, symptoms, methods, citations, knowledge graph clues, or bundled RAG retrieval.
---

# Nihaisha RAG Prototype

Use the bundled database for course study and source lookup. Runtime factual evidence is the retrieved course PDF original paragraph with its PDF name and page; retrieval aids are not truth by themselves.

Run from the directory containing this file. Database: `data/pdf_rag_bge_m3/rag.sqlite`.

## First use

Before the first retrieval in a workspace, run:

```bash
git lfs pull
python3 -m pip install -e ".[runtime]"
python3 -m nihaisha_kg doctor
```

Proceed only when `doctor` reports `status: ok`. The 159,286-unit dense production database requires FAISS for `vector`/`hybrid`; there is no full-scan fallback. `text`/`knowledge` need no API key. `vector`/`hybrid` need FAISS plus a query embedding backend: SiliconFlow `BAAI/bge-m3` is recommended; local is optional with `.[local]` and `--embedding local-bge-m3`. The runtime parses the nearest `.env` without shell execution; an already-exported variable takes precedence.

## Quick reference

| Need | Command/policy |
| --- | --- |
| Exact text | `python3 -m nihaisha_kg search "原词" --mode text --limit 5` |
| Structured fact | `python3 -m nihaisha_kg search "一钱是多少克" --mode knowledge --limit 5` |
| Normal semantic query | `python3 -m nihaisha_kg search "桂枝汤和麻黄汤的方证如何鉴别？" --mode hybrid --limit 8` |
| Grounded draft | `python3 -m nihaisha_kg answer "木香饼热熨法来自哪一本书哪一段？" --mode hybrid --limit 8` |
| Questionable citation | rerun `search` or `answer` with `--json --trace`; inspect selected IDs, channels/ranks, reranker degradation, then verify citation PDF/page/quote |
| Maintainer regression | `python3 -m nihaisha_kg evaluate --cases evals/golden_v1.jsonl --mode hybrid --limit 10` |

CLI reranking defaults to `auto`: it uses SiliconFlow only when a key exists; `--reranker none` disables it. Degradation is sanitized in trace. Search trace is visible with `--json --trace`; answer trace is in answer JSON with those flags. Trace is diagnostic metadata, never evidence.

## Evidence policy

1. Cite `course_primary`: retrieved course PDF originals, including PDF/file name, page, and a short exact excerpt. If insufficient, say so; do not fill gaps from memory.
2. Treat `derived` knowledge triples, guide nodes, query expansions, flows, and follow-ups as navigation only. Never cite them as standalone truth; bind useful clues back to a retrieved original paragraph.
3. `classic_primary` authoritative classical originals and `reference_secondary` scholarship/web sources are future, separately versioned layers. Material not retrieved from this database must be labeled “external / not retrieved from bundled DB” and must not be presented with bundled-evidence authority.
4. Cite a lecture claim and a classical original separately. Never silently turn Ni Haixia’s paraphrase into an ancient quotation.
5. Generate follow-up questions only from the user query and retrieved evidence. Do not inject a fixed diagnostic checklist.

Answer with a concise summary, numbered original evidence, relevant evidence-bound clues, and safety boundary. Every factual summary claim needs a citation. For future ingestion/rebuild work use `/Users/june/code/github/nihaisha-rag-builder`; this runtime repository must not mutate the production DB directly.

## Common mistakes

- Missing FAISS is an error, not a slow fallback: run `doctor`.
- Plain `--trace` does not make search trace visible: pair it with `--json`.
- A graph/guide hit is a route to evidence, not a citable original.
- External memory is not bundled evidence; label and separate it.
- Seven evaluation cases are a regression seed, not proof of comprehensive accuracy.

## Medical safety

Frame output as course study or source lookup. Do not provide individualized diagnosis, prescription, dosage decision, purchasing advice, acupuncture/external-treatment instruction, or self-treatment plan.

涉及剂量、方药或处方线索时必须谨慎：不同人的体质不同，病情阶段、兼证、年龄、基础病和用药史都不同；现代药材来源、炮制、浓度和药效也和以前差很多。建议去线下正规中医渠道面诊辨证，不要私自购药有风险。
