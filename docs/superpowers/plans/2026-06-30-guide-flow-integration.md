# Guide Flow Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight clinical-guide layer from the existing HTML mind map so RAG answers can include friendly differentiation flowcharts and traceable guide nodes.

**Architecture:** Parse the standalone HTML into structured guide nodes, store/search them in SQLite, and fuse guide hits into `answer` output. Keep PDF citations as the source of truth; guide nodes are navigation aids for辨证 logic.

**Tech Stack:** Python stdlib `html.parser`, SQLite FTS, existing `unittest` suite and CLI.

---

### Task 1: Parse HTML Guide Nodes

**Files:**
- Modify: `nihaisha_kg/pdf_vector.py`
- Test: `tests/test_pdf_vector.py`

- [ ] Write a failing test that parses a tiny HTML tree and returns guide nodes with `node_id`, `label`, `node_type`, `badge`, `parent_id`, `path`, and `content`.
- [ ] Run the targeted test and confirm it fails because parsing does not exist.
- [ ] Implement `GuideNode` and `parse_guide_html`.
- [ ] Run the targeted test and confirm it passes.

### Task 2: Store And Search Guide Nodes

**Files:**
- Modify: `nihaisha_kg/pdf_vector.py`
- Test: `tests/test_pdf_vector.py`

- [ ] Write a failing test that imports guide nodes into a temp SQLite database and searches by symptom/formula.
- [ ] Run the targeted test and confirm it fails because guide schema/search do not exist.
- [ ] Implement `ensure_guide_schema`, `import_guide_html`, and `search_guide_nodes`.
- [ ] Run the targeted test and confirm it passes.

### Task 3: Fuse Guide Nodes Into Answers

**Files:**
- Modify: `nihaisha_kg/pdf_vector.py`
- Modify: `nihaisha_kg/cli.py`
- Test: `tests/test_pdf_vector.py`

- [ ] Write a failing test that `answer_pdf_rag` returns `related_guide_nodes`, `differentiation_flow`, and `followup_questions` when a guide is available.
- [ ] Run the targeted test and confirm it fails.
- [ ] Implement guide lookup from the default HTML path when available, plus deterministic flow generation from guide hits and related knowledge units.
- [ ] Print guide nodes and flow text in non-JSON CLI output.
- [ ] Run the targeted test and confirm it passes.

### Task 4: Document Skill Output

**Files:**
- Modify: `SKILL.md`
- Modify: `README.md`

- [ ] Update final answer requirements to include `辨证流程图`, `关键追问`, and `关联导图节点`.
- [ ] Run full tests and an example CLI query.
