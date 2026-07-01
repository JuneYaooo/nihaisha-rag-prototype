# Guide Flow Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight clinical-guide layer inspired by the external mind-map format, but generated from the bundled original PDF evidence so RAG answers can include friendly differentiation flowcharts and traceable guide nodes.

**Architecture:** Rebuild source-grounded guide nodes from `paragraphs` and `knowledge_units`, store/search them in SQLite, and fuse guide hits into `answer` output. Keep PDF citations as the source of truth; guide nodes are navigation aids for辨证 logic.

**Tech Stack:** SQLite FTS, existing deterministic knowledge extraction, `unittest` suite and CLI.

---

### Task 1: Build Guide Nodes From Original Evidence

**Files:**
- Modify: `nihaisha_kg/pdf_vector.py`
- Test: `tests/test_pdf_vector.py`

- [ ] Write a failing test that rebuilds guide nodes from `paragraphs` and `knowledge_units`, with source path, page number, paragraph id, and evidence quote.
- [ ] Run the targeted test and confirm it fails because source-grounded guide rebuilding does not exist.
- [ ] Implement `GuideNode` and `rebuild_guide_nodes`.
- [ ] Run the targeted test and confirm it passes.

### Task 2: Store And Search Guide Nodes

**Files:**
- Modify: `nihaisha_kg/pdf_vector.py`
- Test: `tests/test_pdf_vector.py`

- [ ] Write a failing test that rebuilds guide nodes into a temp SQLite database and searches by symptom/formula.
- [ ] Run the targeted test and confirm it fails because guide schema/search do not exist.
- [ ] Implement `ensure_guide_schema`, source metadata columns, and `search_guide_nodes`.
- [ ] Run the targeted test and confirm it passes.

### Task 3: Fuse Guide Nodes Into Answers

**Files:**
- Modify: `nihaisha_kg/pdf_vector.py`
- Modify: `nihaisha_kg/cli.py`
- Test: `tests/test_pdf_vector.py`

- [ ] Write a failing test that `answer_pdf_rag` returns `related_guide_nodes`, `differentiation_flow`, and `followup_questions` when source-grounded guide nodes are available.
- [ ] Run the targeted test and confirm it fails.
- [ ] Implement guide lookup from SQLite, plus deterministic flow generation from guide hits and related knowledge units.
- [ ] Print guide nodes and flow text in non-JSON CLI output.
- [ ] Run the targeted test and confirm it passes.

### Task 4: Document Skill Output

**Files:**
- Modify: `SKILL.md`
- Modify: `README.md`

- [ ] Update final answer requirements to include `辨证流程图`, `关键追问`, and `关联导图节点`.
- [ ] Run full tests and an example CLI query.
