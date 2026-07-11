---
name: nihaisha-rag-prototype
description: Use when answering questions about Ni Haixia / 倪海厦 course PDFs, source/page lookup, original paragraphs, formula-pattern comparisons, dosage units, symptoms, methods, citations, knowledge graph clues, or bundled RAG retrieval.
---

# Nihaisha RAG Prototype

Answer from the legacy bundled PDF corpus. Authoritative runtime evidence is a retrieved original PDF paragraph with portable filename, page, and excerpt; retrieval aids are navigation only. Run in this Skill directory; DB: `data/pdf_rag_bge_m3/rag.sqlite`.

## First use

```bash
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[runtime]"
python3 -m nihaisha_kg doctor
```

Retrieve only after `status: ok`. The 159,286-unit dense DB requires FAISS for `vector`/`hybrid`; no full-scan fallback exists. `text`/`knowledge` need no key. `vector`/`hybrid` need FAISS plus query embeddings: recommended SiliconFlow `BAAI/bge-m3`, or `.[local]` with `--embedding local-bge-m3`. The nearest `.env` is parsed without shell execution; exported values win.

## Commands

```bash
python3 -m nihaisha_kg search "原词" --mode text --limit 5
python3 -m nihaisha_kg search "一钱是多少克" --mode knowledge --limit 5
python3 -m nihaisha_kg search "桂枝汤和麻黄汤的方证如何鉴别？" --mode hybrid --limit 8
python3 -m nihaisha_kg answer "木香饼热熨法来自哪一本书哪一段？" --mode hybrid --limit 8
python3 -m nihaisha_kg evaluate --cases evals/golden_v1.jsonl --mode hybrid --limit 10
```

Hybrid helps semantic mismatch but does not beat text on the current seven-case seed; measure mode choice. Seven cases are regression seeds, not accuracy proof.

CLI reranking is `auto`: SiliconFlow only when a key exists; `--reranker none` disables it. For questionable citations rerun search/answer with `--json --trace`, inspect selected IDs, channels/ranks and degradation, then verify PDF/page/quote. Plain output never displays trace.

Trace allowlists diagnostics; it does not copy provider credentials, headers, env dumps, vectors, or full evidence, and recognized credential/error patterns are sanitized. `normalized_query` retains query text: never query with keys, passwords, patient/private text, or secrets. Trace is neither evidence nor a confidentiality boundary. Public `source_path` must be portable (`pdfs/<basename>`), never an absolute machine path.

## Evidence policy

1. Cite only retrieved PDF original paragraphs. If evidence is insufficient, say so; never fill from memory. The current SQLite stores document `source_layer`, paragraph-level evidence, candidate entities and candidate relations.
2. Ten course documents are classified `course_primary`; `黄帝内经原文和翻译.pdf` is classified `classic_primary` as a candidate document. This classification is structural provenance, not edition verification or expert review.
3. `derived` triples, guide nodes, expansions, flows, and follow-ups only navigate to originals; never cite them standalone or assert their subject/object as facts or formula names. Follow-ups must arise from the query and retrieved evidence, not a fixed checklist.
4. This runtime has no external retrieval. It may retrieve classical material already present in the legacy bundle, but a separately versioned and verified authoritative classic layer is not implemented. Never reconstruct an absent ancient quotation from model memory.
5. The current `classic_primary` candidate is not yet independently version-verified, and `reference_secondary` remains future work. User-supplied or explicitly authorized outside research must be labeled “external / not retrieved from bundled DB,” verified and cited separately, without bundled authority.
6. Cite lecture claims and classical originals separately; never turn a paraphrase into an ancient quote.

Use the separate `nihaisha-rag-builder` repository/Skill (often `../nihaisha-rag-builder`, or configured path) for incremental PDFs. Stage, audit, validate, and atomically publish the full asset set; runtime must not mutate production DB.

## Safety

Frame output as course study/source lookup. Never provide individualized diagnosis, prescription, dosage decision, purchasing advice, acupuncture/external-treatment instruction, or self-treatment plan.

涉及剂量、方药或处方线索时必须谨慎：不同人的体质不同，病情阶段、兼证、年龄、基础病和用药史都不同；现代药材来源、炮制、浓度和药效也和以前差很多。建议去线下正规中医渠道面诊辨证，不要私自购药有风险。
