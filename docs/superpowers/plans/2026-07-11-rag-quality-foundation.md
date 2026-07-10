# RAG Quality Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current 11-PDF runtime measurably more accurate, fast, explainable, and easy to operate before rebuilding the knowledge graph or adding classical texts.

**Architecture:** Preserve the existing SQLite/FAISS assets and public CLI while extracting focused runtime modules for normalization, rank fusion, reranking, evaluation, and diagnostics. Retrieval becomes lexical + dense + knowledge channel ranking followed by RRF and one optional SiliconFlow rerank; answer synthesis becomes evidence-driven, and production-sized dense search fails fast when FAISS is unavailable.

**Tech Stack:** Python 3.11+, SQLite FTS5, FAISS, SiliconFlow BAAI/bge-m3 embeddings, SiliconFlow BAAI/bge-reranker-v2-m3, `requests`, `unittest`, JSONL evaluation fixtures.

---

## Scope

This plan implements phase one of the approved design only:

- a versioned retrieval evaluation harness and baseline dataset;
- Chinese-aware lexical query terms;
- evidence-driven source and clinical answer drafts;
- channel-level RRF instead of raw score addition;
- one optional SiliconFlow rerank after candidate fusion;
- FAISS fail-fast behavior, cached index loading, and `doctor` diagnostics;
- query traces, CLI integration, and a five-minute README path.

The normalized entity/edge schema, builder-side graph extraction, current-database rebuild, and classical-text ingestion belong to separate phase-two and phase-three plans. They must consume the quality gates created here.

Phase one measures retrieval with ID-based recall, MRR, nDCG, and context precision, while deterministic answer regressions cover citation-topic correctness and forbidden template leakage. Claim-level automated faithfulness scoring is added after phase two provides stable evidence-edge IDs; introducing an LLM judge before that schema would create an unstable release gate.

## File Map

**Create:**

- `nihaisha_kg/normalization.py` — traditional/simplified normalization and Chinese lexical query terms.
- `nihaisha_kg/fusion.py` — channel-level reciprocal rank fusion and field merging.
- `nihaisha_kg/rerank.py` — provider-neutral outcome plus SiliconFlow reranker.
- `nihaisha_kg/evaluation.py` — golden-case loading and deterministic ranking metrics.
- `nihaisha_kg/diagnostics.py` — database, FTS, vector, FAISS, and mapping health checks.
- `evals/golden_v1.jsonl` — initial production-database retrieval cases.
- `tests/test_normalization.py` — lexical regression tests.
- `tests/test_fusion.py` — channel fusion tests.
- `tests/test_rerank.py` — SiliconFlow request/response and fallback tests.
- `tests/test_evaluation.py` — ranking metric tests.
- `tests/test_diagnostics.py` — doctor and production-size fail-fast tests.
- `tests/test_answer_quality.py` — removal of hard-coded source/clinical answers.

**Modify:**

- `nihaisha_kg/pdf_vector.py` — delegate focused behavior, use RRF, rerank once, cache FAISS, expose traces, and remove fixed answer text.
- `nihaisha_kg/cli.py` — add `doctor`, `evaluate`, `--reranker`, `--rerank-model`, and `--trace`.
- `pyproject.toml` — expose a recommended runtime extra containing FAISS.
- `.env.example` — document reranker configuration without secrets.
- `README.md` — install, doctor, search, answer, trace, performance, and troubleshooting.
- `SKILL.md` — use the new runtime commands and state the evidence/trace contract.
- `tests/test_pdf_vector.py` — preserve compatibility while updating assertions that encoded the old fixed templates.

## Task 1: Add the Retrieval Evaluation Contract

**Files:**

- Create: `nihaisha_kg/evaluation.py`
- Create: `tests/test_evaluation.py`
- Create: `evals/golden_v1.jsonl`

- [ ] **Step 1: Write failing metric and fixture-loading tests**

Create `tests/test_evaluation.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nihaisha_kg.evaluation import EvalCase, evaluate_ranked_ids, load_eval_cases


class EvaluationTests(unittest.TestCase):
    def test_load_eval_cases_reads_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "case_id": "source-1",
                        "query": "木香饼出处",
                        "task_type": "source_lookup",
                        "relevant_paragraph_ids": ["p1", "p2"],
                        "forbidden_paragraph_ids": ["p9"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            cases = load_eval_cases(path)

        self.assertEqual(cases[0].case_id, "source-1")
        self.assertEqual(cases[0].relevant_paragraph_ids, ("p1", "p2"))
        self.assertEqual(cases[0].forbidden_paragraph_ids, ("p9",))

    def test_evaluate_ranked_ids_reports_recall_mrr_and_forbidden_hits(self) -> None:
        case = EvalCase(
            case_id="compare-1",
            query="桂枝汤和麻黄汤如何鉴别",
            task_type="comparison",
            relevant_paragraph_ids=("p2", "p4"),
            forbidden_paragraph_ids=("p8",),
        )

        metrics = evaluate_ranked_ids(case, ["p8", "p2", "p3", "p4"], k_values=(1, 3, 5))

        self.assertEqual(metrics["hit_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 0.5)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["forbidden_hits_at_5"], 1.0)
        self.assertAlmostEqual(metrics["context_precision_at_5"], 0.5)
        self.assertGreater(metrics["ndcg_at_5"], 0.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
python3 -m unittest tests.test_evaluation -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'nihaisha_kg.evaluation'`.

- [ ] **Step 3: Implement deterministic evaluation primitives**

Create `nihaisha_kg/evaluation.py`:

```python
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    task_type: str
    relevant_paragraph_ids: tuple[str, ...]
    forbidden_paragraph_ids: tuple[str, ...] = ()


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            required = {"case_id", "query", "task_type", "relevant_paragraph_ids"}
            missing = required - payload.keys()
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"{path}:{line_number} missing fields: {names}")
            cases.append(
                EvalCase(
                    case_id=str(payload["case_id"]),
                    query=str(payload["query"]),
                    task_type=str(payload["task_type"]),
                    relevant_paragraph_ids=tuple(str(value) for value in payload["relevant_paragraph_ids"]),
                    forbidden_paragraph_ids=tuple(
                        str(value) for value in payload.get("forbidden_paragraph_ids", [])
                    ),
                )
            )
    if not cases:
        raise ValueError(f"evaluation file is empty: {path}")
    return cases


def evaluate_ranked_ids(
    case: EvalCase,
    ranked_ids: Iterable[str],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    ranking = list(dict.fromkeys(str(value) for value in ranked_ids))
    relevant = set(case.relevant_paragraph_ids)
    forbidden = set(case.forbidden_paragraph_ids)
    metrics: dict[str, float] = {}
    for k in k_values:
        prefix = ranking[:k]
        hits = relevant.intersection(prefix)
        metrics[f"hit_at_{k}"] = float(bool(hits))
        metrics[f"recall_at_{k}"] = len(hits) / len(relevant) if relevant else 1.0
        metrics[f"forbidden_hits_at_{k}"] = float(len(forbidden.intersection(prefix)))
        precision_sum = 0.0
        dcg = 0.0
        relevant_seen = 0
        for rank, paragraph_id in enumerate(prefix, start=1):
            if paragraph_id not in relevant:
                continue
            relevant_seen += 1
            precision_sum += relevant_seen / rank
            dcg += 1.0 / math.log2(rank + 1)
        relevant_denominator = min(len(relevant), k)
        metrics[f"context_precision_at_{k}"] = (
            precision_sum / relevant_denominator if relevant_denominator else 1.0
        )
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1) for rank in range(1, relevant_denominator + 1)
        )
        metrics[f"ndcg_at_{k}"] = dcg / ideal_dcg if ideal_dcg else 1.0
    first_relevant_rank = next(
        (index for index, paragraph_id in enumerate(ranking, start=1) if paragraph_id in relevant),
        None,
    )
    metrics["reciprocal_rank"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    return metrics


def aggregate_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    items = list(rows)
    if not items:
        return {}
    keys = sorted({key for row in items for key in row})
    return {key: sum(row.get(key, 0.0) for row in items) / len(items) for key in keys}
```

- [ ] **Step 4: Add initial production-database cases**

Create `evals/golden_v1.jsonl` with one JSON object per line:

```jsonl
{"case_id":"dose-one-qian","query":"古时候的一钱，是现代的多少克？","task_type":"dosage","relevant_paragraph_ids":["9e061d6932a26775","929e53c511a408c8","f3eebf353ec1ee28"],"forbidden_paragraph_ids":[]}
{"case_id":"source-muxiang","query":"木香饼热熨法来自哪一本书哪一段？","task_type":"source_lookup","relevant_paragraph_ids":["4f29767e17dfa956"],"forbidden_paragraph_ids":[]}
{"case_id":"compare-guizhi-mahuang","query":"桂枝汤和麻黄汤的方证如何鉴别？","task_type":"comparison","relevant_paragraph_ids":["a56d3146d7453891"],"forbidden_paragraph_ids":[]}
{"case_id":"formula-huangqin-banxia","query":"黄芩加半夏生姜汤对应哪些原文证候？","task_type":"entity_fact","relevant_paragraph_ids":["8d8dce6bfeae1f8b"],"forbidden_paragraph_ids":[]}
{"case_id":"acupuncture-four-gates","query":"针灸课程里合谷和太冲合称什么？","task_type":"entity_fact","relevant_paragraph_ids":["fb1cc40bb9bbb8be"],"forbidden_paragraph_ids":[]}
{"case_id":"formula-danggui-sini","query":"手足厥寒、脉细欲绝对应哪段原文？","task_type":"source_lookup","relevant_paragraph_ids":["30c26f1fd51a4ff8"],"forbidden_paragraph_ids":[]}
{"case_id":"source-guizhi-not-muxiang","query":"桂枝汤的出处在哪本书哪一页？","task_type":"source_lookup","relevant_paragraph_ids":["a56d3146d7453891"],"forbidden_paragraph_ids":["4f29767e17dfa956"]}
```

- [ ] **Step 5: Run the focused tests**

Run:

```bash
python3 -m unittest tests.test_evaluation -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add nihaisha_kg/evaluation.py tests/test_evaluation.py evals/golden_v1.jsonl
git commit -m "test: add retrieval quality evaluation contract"
```

## Task 2: Fix Chinese Lexical Query Terms

**Files:**

- Create: `nihaisha_kg/normalization.py`
- Create: `tests/test_normalization.py`
- Modify: `nihaisha_kg/pdf_vector.py:461-483,2603-2630`
- Modify: `tests/test_pdf_vector.py`

- [ ] **Step 1: Write failing Chinese query and real retrieval tests**

Create `tests/test_normalization.py` and cover these exact helper behaviors:

```python
cases = (
    ("调和营卫的方证有哪些？", ("方证",), ["方证", "方證", "调和营卫", "調和營衛"]),
    ("和解少阳如何理解？", (), ["和解少阳", "和解少陽", "理解"]),
    ("从容脉是什么脉象？", (), ["从容脉", "從容脈", "脉象", "脈象"]),
    ("手足厥冷应该用什么方？", (), ["手足厥冷"]),
)
```

Also verify:

- `太陽病的脈證有哪些？` emits both `太阳病` / `脉证` and `太陽病` / `脈證`.
- `太阳病` suppresses a nested-only `太阳`, while an independent `太阳` occurrence is retained.
- no fallback contains `什么/哪些/这个/那个/问题/资料/内容/时候`.
- a zero fallback budget retains domain and script variants but no fallback terms.
- simplified/traditional pairs for `手足厥冷应该用什么方？`, generic-only question glue, and
  `课程里太阳病的原文？` produce equivalent content terms without scaffold leakage.
- every traditional character used by scaffold, generic-fragment, and trailing-particle vocabulary
  normalizes completely.
- more than 1,000 ASCII tokens produce at most `max_terms`, with domain/script anchors first.
- reverse equivalences retain every ordered traditional form for a simplified character; in particular,
  `经方里面辨证` emits both `經方裡面辨證` and `經方裏面辨證`, with at most eight variants per term.
- `text_search_terms` and `knowledge_search_terms` expose the same boundary-preserving behavior.
- a real temporary `LocalVectorStore` retrieves a traditional-only `太陽病/脈證` paragraph from a simplified query and a simplified-only `少阴病/脉证` paragraph from a traditional query.
- live SQLite text search handles more than 1,000 ASCII query tokens without `OperationalError`, and
  `手足厥冷應該用什麼方？` retrieves a simplified-only content paragraph.

- [ ] **Step 2: Run the tests and verify the rejected behavior fails**

Run the helper, delegation, and real SQLite retrieval tests before changing production code. Expected
failures must show the root causes: normalized-only terms, incomplete traditional scaffold coverage,
concatenation after global replacement, generic-glue trigrams, redundant substring domain anchors, and
an unbounded complete term list that exceeds SQLite expression limits.

- [ ] **Step 3: Implement dual-script, boundary-preserving lexical terms**

Create `nihaisha_kg/normalization.py` with these invariants:

```python
_TRADITIONAL_TO_SIMPLIFIED = {
    "錢": "钱", "證": "证", "發": "发", "燒": "烧", "噁": "恶",
    "瀉": "泻", "黃": "黄", "餅": "饼", "熱": "热", "藥": "药",
    "處": "处", "裡": "里", "來": "来", "頭": "头", "頸": "颈",
    "痠": "酸", "湯": "汤", "鑒": "鉴", "別": "别", "與": "与",
    "脈": "脉", "隨": "随", "兩": "两", "陽": "阳", "陰": "阴",
    "時": "时", "從": "从", "麼": "么", "書": "书", "頁": "页",
    "經": "经", "衛": "卫", "營": "营", "氣": "气", "虛": "虚",
    "實": "实", "風": "风", "濕": "湿", "為": "为", "歸": "归",
    "屬": "属", "開": "开", "關": "关", "體": "体", "現": "现",
    "歲": "岁", "調": "调", "較": "较", "區": "区", "於": "于",
    "應": "应", "請": "请", "問": "问", "訴": "诉", "對": "对",
    "該": "该", "課": "课", "這": "这", "個": "个", "題": "题",
    "資": "资", "內": "内", "嗎": "吗", "裏": "里",
}
TRADITIONAL_QUERY_TRANSLATION = str.maketrans(_TRADITIONAL_TO_SIMPLIFIED)

def build_simplified_to_traditional_equivalents() -> dict[str, tuple[str, ...]]:
    equivalents: dict[str, list[str]] = {}
    for traditional, simplified in _TRADITIONAL_TO_SIMPLIFIED.items():
        equivalents.setdefault(simplified, []).append(traditional)
    return {simplified: tuple(values) for simplified, values in equivalents.items()}

SIMPLIFIED_TO_TRADITIONAL_EQUIVALENTS = build_simplified_to_traditional_equivalents()
MAX_TERM_VARIANTS = 8

QUESTION_SCAFFOLD = (
    "从什么时候到什么时候", "什么时候", "是什么", "有哪些", "如何", "怎么",
    "哪一本书", "哪本书", "哪一页", "哪一段", "对应", "相关", "原文", "出处",
    "课程里", "课程中", "应该用", "请问", "能不能", "可以", "告诉我",
)
GENERIC_FRAGMENTS = {"这个", "那个", "问题", "资料", "内容", "时候", "什么", "哪些"}
```

The implementation must then:

1. normalize the query character-for-character but retain the original query at identical offsets;
2. sort domain candidates longest-first, select occurrence spans, and emit a shorter candidate only if
   at least one occurrence is outside spans already claimed by longer candidates;
3. emit normalized recognized terms first, followed by original-surface and every ordered traditional
   equivalent, using bounded Cartesian expansion (`MAX_TERM_VARIANTS=8`) and stable deduplication;
4. replace recognized, measure, ASCII, full question-phrase, and generic-fragment spans with whitespace
   boundaries in both aligned query variants, then normalize whitespace;
5. never globally delete `和/与/从/到/的`; only trim trailing `的/了/呢/吗` from a standalone fallback;
6. emit CJK chunks of at most eight characters or overlapping trigrams, interleaving script variants,
   rejecting any candidate containing generic glue, with `max_fallback_terms=12` as the fallback budget;
7. deduplicate the complete priority-ordered sequence—domain/script anchors, measures, ASCII, then
   fallback—and cap the final return value with `max_terms=64`.

- [ ] **Step 4: Delegate existing search-term functions**

In `nihaisha_kg/pdf_vector.py`, import:

```python
from .normalization import lexical_query_terms, normalize_query_text
```

Replace `text_search_terms` and `knowledge_search_terms` with:

```python
def query_domain_terms(query: str) -> list[str]:
    normalized = normalize_query_text(query)
    fixed_terms = [
        "一钱", "黄金比例", "木香饼", "热熨", "主之", "方证", "鉴别",
        "出处", "太阳病", "欲解", *SYMPTOM_TERMS, *SIX_CHANNEL_TERMS,
    ]
    return dedupe_keep_order(
        [*extract_formula_terms(normalized), *(term for term in fixed_terms if term in normalized)]
    )


def text_search_terms(query: str) -> list[str]:
    return lexical_query_terms(query, domain_terms=query_domain_terms(query))


def knowledge_search_terms(query: str) -> list[str]:
    normalized = normalize_query_text(query)
    extra = ["禁忌", "误用", "比较", "区别"]
    return dedupe_keep_order(
        [*text_search_terms(query), *(term for term in extra if term in normalized)]
    )
```

Remove the old `TRADITIONAL_QUERY_TRANSLATION` constant and local `normalize_query_text` definition so there is one normalization source.

- [ ] **Step 5: Run focused and compatibility tests**

Run:

```bash
python3 -m unittest tests.test_normalization tests.test_pdf_vector.PdfVectorTests.test_search_term_apis_delegate_to_boundary_preserving_normalization tests.test_pdf_vector.PdfVectorTests.test_text_search_retrieves_across_traditional_and_simplified_scripts tests.test_pdf_vector.PdfVectorTests.test_text_search_bounds_large_ascii_query_without_sqlite_error tests.test_pdf_vector.PdfVectorTests.test_traditional_question_scaffold_retrieves_simplified_paragraph tests.test_pdf_vector.PdfVectorTests.test_simplified_query_retrieves_each_traditional_li_form tests.test_pdf_vector.PdfVectorTests.test_text_search_returns_exact_original_paragraph_without_embedding -v
```

Expected: all selected helper, delegation, cross-script SQLite retrieval, and compatibility tests pass.

- [ ] **Step 6: Commit**

```bash
git add nihaisha_kg/normalization.py nihaisha_kg/pdf_vector.py tests/test_normalization.py tests/test_pdf_vector.py
git commit -m "fix: retain all traditional query variants"
```

## Task 3: Remove Hard-Coded Answer Content

**Files:**

- Create: `tests/test_answer_quality.py`
- Modify: `nihaisha_kg/pdf_vector.py:2633-2673,3375-3408,3493-3567`
- Modify: `tests/test_pdf_vector.py:563-607,1015-1051`

- [ ] **Step 1: Write failing source and clinical answer regressions**

Create `tests/test_answer_quality.py`:

```python
from __future__ import annotations

import unittest

from nihaisha_kg.pdf_vector import synthesize_pdf_rag_answer


def result(text: str, source: str = "伤寒论.pdf", page: int = 10) -> dict[str, object]:
    return {
        "paragraph_id": f"p-{page}",
        "source_path": source,
        "page_start": page,
        "page_end": page,
        "text": text,
        "score": 1.0,
        "matched_knowledge_units": [],
    }


class AnswerQualityTests(unittest.TestCase):
    def test_source_lookup_uses_query_anchor_instead_of_muxiang_template(self) -> None:
        payload = synthesize_pdf_rag_answer(
            "桂枝汤的出处在哪本书哪一页？",
            [result("太阳中风，汗出恶风，桂枝汤主之。", page=68)],
        )

        self.assertIn("桂枝汤", payload["answer"])
        self.assertIn("伤寒论.pdf p68", payload["answer"])
        self.assertNotIn("木香饼", payload["answer"])
        self.assertNotIn("热熨", payload["answer"])

    def test_clinical_answer_does_not_inject_diarrhea_questions_into_cough_case(self) -> None:
        payload = synthesize_pdf_rag_answer(
            "患者咳嗽、怕冷、无汗，课程有哪些相关线索？",
            [result("恶寒无汗而喘，麻黄汤主之。", page=69)],
        )

        self.assertIn("咳嗽", payload["answer"])
        self.assertNotIn("下利性质", payload["answer"])
        self.assertNotIn("心下痞满", payload["answer"])

    def test_no_results_never_emits_empty_citation_marker(self) -> None:
        payload = synthesize_pdf_rag_answer("桂枝汤出处", [])
        self.assertNotIn("证据见 。", payload["answer"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm the two fixed-template failures**

Run:

```bash
python3 -m unittest tests.test_answer_quality -v
```

Expected: the source lookup and clinical tests fail because the current answer contains fixed 木香饼 and 下利 text.

- [ ] **Step 3: Generalize intent anchors and answer synthesis**

In `nihaisha_kg/pdf_vector.py`, add:

```python
SOURCE_LOOKUP_MARKERS = ("出处", "哪本书", "哪一本书", "哪一页", "哪一段", "原文")


def answer_anchor_terms(query: str) -> list[str]:
    normalized = normalize_query_text(query)
    terms = query_domain_terms(normalized)
    generic = {
        "出处", "原文", "方证", "鉴别", "主之", "太阳", "少阳", "阳明",
        "太阴", "少阴", "厥阴",
    }
    return [term for term in terms if term not in generic][:6]


def reliable_source_anchors(query: str) -> list[str]:
    """Return only explicit formula names and allowlisted distinctive named entities.

    Unlike display-oriented answer_anchor_terms, this list is safe to use as a
    hard evidence filter. Generic references, symptoms, and fallback phrases
    must never enter it.
    """
    normalized = normalize_query_text(query)
    named_terms = ("木香饼", "一钱", "太阳病", "黄金比例")
    formulas = direct_present_terms(normalized, KNOWN_FORMULA_ANCHORS)
    return dedupe_keep_order([*formulas, *(term for term in named_terms if term in normalized)])[:8]


def detect_answer_intent(query: str) -> str:
    normalized = normalize_query_text(query)
    if "一钱" in normalized and any(marker in normalized for marker in ("克", "多少", "几")):
        return "dosage"
    if any(marker in normalized for marker in SOURCE_LOOKUP_MARKERS):
        return "source_lookup"
    if any(marker in normalized for marker in ("鉴别", "比较", "区别")):
        return "comparison"
    clinical_markers = (
        "病人", "患者", "发烧", "下利", "拉肚子", "恶心", "咳嗽", "怕冷",
        "建议开", "开什么方", "处方", "男性", "女性", "男患者", "女患者",
    )
    has_age = bool(re.search(r"\d+\s*岁(?:\s*[男女])?", normalized))
    return "clinical" if has_age or any(marker in normalized for marker in clinical_markers) else "general"
```

Replace the `method` branch in `synthesize_pdf_rag_answer` with:

```python
    elif intent == "source_lookup":
        anchors = answer_anchor_terms(query)
        topic = "、".join(anchors) if anchors else normalize_query_text(query).rstrip("？?")
        locations = "；".join(
            f"{citation['label']}[{citation['index']}]" for citation in citations[:4]
        )
        answer = (
            f"关于“{topic}”，当前检索到的主要原文出处是：{locations}。"
            "请以下列原文摘录为准；知识单元和导图只用于定位，不单独作为事实依据。"
        )
        safety_notice = (
            with_formula_dosage_safety("这是课程原文定位，不是个人诊断或处方建议。")
            if collect_formula_names(relevant_results)
            else ""
        )
```

Replace the `clinical` branch with:

```python
    elif intent == "clinical":
        formulas = collect_formula_names(relevant_results)[:8]
        formula_text = "、".join(formulas) if formulas else "未稳定抽取到方名"
        observed_clues = clinical_evidence_clues(normalize_query_text(query))
        clue_text = "、".join(observed_clues) if observed_clues else "用户描述中的症状"
        evidence_tail = f"可先核对原文证据 {citation_refs}。" if citation_refs else "当前没有足够直接原文证据。"
        answer = (
            "这个问题含有具体病人信息，我不直接给个人处方。"
            f"围绕“{clue_text}”检索到的课程方证线索包括：{formula_text}。"
            f"{evidence_tail}真实病情仍需由合格医师面诊辨证。"
        )
        safety_notice = with_formula_dosage_safety(
            "课程资料不能替代诊断；这里不直接给个人处方、剂量或治疗建议。"
        )
```

Keep display topic extraction (`answer_anchor_terms`) separate from hard-filter anchors (`reliable_source_anchors`). Display terms must occur directly in the normalized query and must avoid cross-boundary manufactured terms. Reliable formula anchors use exact normalized membership in the immutable phase-one `KNOWN_FORMULA_ANCHORS` lexicon; no character heuristic is permitted. A shared bounded query scanner validates both formula and named anchors, with term-specific task suffixes for dosage and method queries. Embedded product phrases such as 桂枝汤圆、木香饼干、太阳病人、一钱包 therefore yield no hard anchor or formula safety warning. This conservative runtime lexicon will be replaced or expanded by phase-two normalized entity data; omission means “do not hard-filter,” not “do not retrieve.” Change `filter_results_for_intent` from `(intent, results)` to `(query, intent, results)` and update both call sites in `synthesize_pdf_rag_answer` and `answer_pdf_rag`. For `source_lookup`, use one evidence-aware matcher for both filtering and citation selection: `validated_source_anchor_spans` returns only product-safe formula and named-anchor occurrences, and both filtering and excerpt construction consume those spans rather than raw substring positions. Named anchors reject only narrow term-specific product continuations, allowing ordinary prose such as 木香饼外敷、太阳病发热、一钱折合、黄金比例达到. Source grounding inputs are limited to paragraph text and nonempty knowledge-unit evidence quotes; derived subjects, predicates, objects, labels, and paths never ground a filter or citation. If no reliable query anchor exists, do not over-filter. For clinical intent, centrally canonicalize bidirectional groups for cold, diarrhea, nausea, fever, throat pain, and appetite clues, then retain only quality evidence whose formula or canonical clue overlaps the query. Source citations are query-aware: prefer a non-empty unit quote containing all reliable anchors, then use complete validated anchor-bearing paragraph sentences or bounded validated-span windows that visibly retain every anchor, before unrelated fallbacks. Multi-anchor fallback reserves the text and separator cost for every unique present anchor first, then deterministically divides the remaining context budget; it never applies a blind final truncation. Evidence-free citations are omitted. Evidence-prose formula safety uses the same primary evidence texts and a separate exact-lexicon detector with explicit product-continuation exclusions. Filtered-empty clinical, dosage, and reliable-formula source answers preserve the exact medical safety notice. Include only actual citation locations and excerpts in the synthesized source answer. `build_followup_questions` may use only actual query clues and explicit guide-node or knowledge-unit fields, wrapping derived items in a neutral question. It must not contain fixed diagnostic, differentiation, or medication-risk templates; if no item is derivable, return an empty list. Remove the unused `build_followup_query` helper.

- [ ] **Step 4: Delete dead fixed query expansion**

Remove `expand_answer_query`, which is not called by runtime code. Replace its isolated unit test with assertions against `build_query_plan` and `answer_anchor_terms` so tests cover the actual runtime path.

- [ ] **Step 5: Run answer and existing safety tests**

Run:

```bash
python3 -m unittest \
  tests.test_answer_quality \
  tests.test_pdf_vector.PdfVectorTests.test_answer_pdf_rag_returns_clinical_safety_boundary \
  tests.test_pdf_vector.PdfVectorTests.test_synthesize_answer_aggregates_dosage_evidence_with_citations \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add nihaisha_kg/pdf_vector.py tests/test_answer_quality.py tests/test_pdf_vector.py
git commit -m "fix: synthesize answers only from retrieved evidence"
```

## Task 4: Replace Raw Hybrid Score Addition with Channel RRF

**Files:**

- Create: `nihaisha_kg/fusion.py`
- Create: `tests/test_fusion.py`
- Modify: `nihaisha_kg/pdf_vector.py` hybrid search and query-rewrite fusion
- Modify: `tests/test_pdf_vector.py`

- [ ] **Step 1: Write failing fusion tests**

Create `tests/test_fusion.py`:

```python
from __future__ import annotations

import unittest

from nihaisha_kg.fusion import fuse_ranked_channels


def hit(paragraph_id: str, score: float, source: str) -> dict[str, object]:
    return {
        "paragraph_id": paragraph_id,
        "score": score,
        "retrieval_sources": [source],
        "matched_units": [],
        "matched_knowledge_units": [],
        "matched_text_terms": [],
        "unit_types": [],
    }


class FusionTests(unittest.TestCase):
    def test_result_seen_in_two_channels_beats_incomparable_large_raw_score(self) -> None:
        fused = fuse_ranked_channels(
            {
                "vector": [hit("shared", 0.61, "vector"), hit("vector-only", 0.60, "vector")],
                "text": [hit("shared", 0.72, "text"), hit("text-only", 99.0, "text")],
                "knowledge": [hit("knowledge-only", 500.0, "knowledge")],
            },
            limit=5,
        )

        self.assertEqual(fused[0]["paragraph_id"], "shared")
        self.assertEqual(fused[0]["channel_ranks"], {"text": 1, "vector": 1})
        self.assertGreater(fused[0]["fusion_score"], fused[1]["fusion_score"])

    def test_fusion_merges_sources_and_knowledge_units(self) -> None:
        knowledge = hit("p1", 0.9, "knowledge")
        knowledge["matched_knowledge_units"] = [{"knowledge_unit_id": "k1"}]
        fused = fuse_ranked_channels(
            {"text": [hit("p1", 0.8, "text")], "knowledge": [knowledge]},
            limit=1,
        )
        self.assertEqual(fused[0]["retrieval_sources"], ["knowledge", "text"])
        self.assertEqual(fused[0]["matched_knowledge_units"], [{"knowledge_unit_id": "k1"}])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run:

```bash
python3 -m unittest tests.test_fusion -v
```

Expected: `ERROR` with `ModuleNotFoundError`.

- [ ] **Step 3: Implement channel-level RRF**

Create `nihaisha_kg/fusion.py`:

```python
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence


def merge_unique(left: list[object], right: Sequence[object]) -> list[object]:
    seen = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in left}
    merged = list(left)
    for value in right:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            merged.append(value)
    return merged


def fuse_ranked_channels(
    channels: Mapping[str, Sequence[dict[str, object]]],
    limit: int,
    rank_constant: int = 60,
    channel_weights: Mapping[str, float] | None = None,
) -> list[dict[str, object]]:
    if limit <= 0 or not channels:
        return []
    weights = {"vector": 1.0, "text": 1.0, "knowledge": 1.0, **(channel_weights or {})}
    fused: dict[str, dict[str, object]] = {}
    for channel in sorted(channels):
        seen_paragraph_ids: set[str] = set()
        for rank, result in enumerate(channels[channel], start=1):
            paragraph_id = str(result.get("paragraph_id", ""))
            if not paragraph_id or paragraph_id in seen_paragraph_ids:
                continue
            seen_paragraph_ids.add(paragraph_id)
            contribution = weights.get(channel, 1.0) / (rank_constant + rank)
            current = fused.get(paragraph_id)
            if current is None:
                current = copy.deepcopy(result)
                current["raw_channel_scores"] = {}
                current["channel_ranks"] = {}
                current["fusion_score"] = 0.0
                current["retrieval_sources"] = []
                current["matched_units"] = []
                current["matched_knowledge_units"] = []
                current["matched_text_terms"] = []
                current["unit_types"] = []
                fused[paragraph_id] = current
            current["raw_channel_scores"][channel] = float(result.get("score", 0.0))
            current["channel_ranks"][channel] = rank
            current["fusion_score"] = float(current["fusion_score"]) + contribution
            current["score"] = current["fusion_score"]
            current["retrieval_sources"] = sorted(
                set(current["retrieval_sources"]) | set(result.get("retrieval_sources", [channel]))
            )
            for field in ("matched_units", "matched_knowledge_units", "matched_text_terms", "unit_types"):
                current[field] = merge_unique(list(current[field]), list(result.get(field, [])))
            for score_field in ("vector_score", "text_score", "knowledge_score"):
                if score_field in result:
                    current[score_field] = max(
                        float(current.get(score_field, 0.0)),
                        float(result.get(score_field, 0.0)),
                    )
    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["fusion_score"]),
            -len(item["channel_ranks"]),
            -max(item["raw_channel_scores"].values(), default=0.0),
            str(item["paragraph_id"]),
        ),
    )
    return ranked[:limit]
```

- [ ] **Step 4: Replace `search_hybrid` score addition**

Import `fuse_ranked_channels` in `pdf_vector.py`. Replace the current `combined` dictionary and `vector_score + text_score + knowledge_score` block with:

```python
        return fuse_ranked_channels(
            {
                "vector": vector_results,
                "text": text_results,
                "knowledge": knowledge_results,
            },
            limit=limit,
        )
```

Rename the separate cross-query fusion layer to `fuse_query_rewrites` and store its score in `query_fusion_score`, while preserving each channel-level `fusion_score` and `channel_ranks`. Each paragraph contributes at most once per rewrite, using its first/best rank, while later duplicate occurrences are still inspected for representative selection and merged diagnostics. Select one representative occurrence by highest input `score`, then earliest rewrite/rank, and copy its channel diagnostics atomically; aggregate only query RRF and stable-deduplicated list diagnostics. Add a bounded `rewrite_observations` trace, return empty for nonpositive limits, and sort ties explicitly by paragraph ID. For hybrid mode, do not perform a second knowledge lookup in `run_query_plan_search`, because `search_hybrid` already includes that channel.

- [ ] **Step 5: Run fusion and hybrid regression tests**

Run:

```bash
python3 -m unittest \
  tests.test_fusion \
  tests.test_pdf_vector.PdfVectorTests.test_hybrid_search_combines_vector_and_text_sources \
  tests.test_pdf_vector.PdfVectorTests.test_hybrid_search_includes_grounded_knowledge_units \
  tests.test_pdf_vector.PdfVectorTests.test_fuse_query_rewrites_selects_one_atomic_representative_observation \
  -v
```

Expected: all selected tests pass after updating the renamed rewrite-fusion import in the existing test.

- [ ] **Step 6: Commit**

```bash
git add nihaisha_kg/fusion.py nihaisha_kg/pdf_vector.py tests/test_fusion.py tests/test_pdf_vector.py
git commit -m "fix: fuse retrieval channels by reciprocal rank"
```

## Task 5: Add One Optional SiliconFlow Rerank

**Files:**

- Create: `nihaisha_kg/rerank.py`
- Create: `tests/test_rerank.py`
- Modify: `nihaisha_kg/pdf_vector.py:3570-3637`
- Modify: `tests/test_pdf_vector.py`
- Modify: `tests/test_answer_quality.py`
- Modify: `.env.example`

- [x] **Step 1: Write failing reranker request and fallback tests**

Create `tests/test_rerank.py` with a fake response/session:

```python
from __future__ import annotations

import unittest

from nihaisha_kg.rerank import SiliconFlowReranker


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class FakeSession:
    def __init__(self, payload: dict[str, object] | None = None, error: Exception | None = None) -> None:
        self.payload = payload or {}
        self.error = error
        self.calls: list[tuple[str, dict[str, str], dict[str, object], int]] = []

    def post(self, url: str, headers: dict[str, str], json: dict[str, object], timeout: int) -> FakeResponse:
        self.calls.append((url, headers, json, timeout))
        if self.error:
            raise self.error
        return FakeResponse(self.payload)


class RerankTests(unittest.TestCase):
    def test_siliconflow_reranker_posts_documents_and_maps_indices(self) -> None:
        session = FakeSession(
            {"results": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.21}]}
        )
        backend = SiliconFlowReranker(api_key="secret", session=session, max_retries=1)
        candidates = [
            {"paragraph_id": "p1", "title": "甲", "text": "无关"},
            {"paragraph_id": "p2", "title": "乙", "text": "桂枝汤主之"},
        ]

        outcome = backend.rerank("桂枝汤", candidates, limit=2)

        self.assertEqual([row["paragraph_id"] for row in outcome.results], ["p2", "p1"])
        self.assertEqual(outcome.results[0]["rerank_score"], 0.91)
        url, headers, payload, timeout = session.calls[0]
        self.assertEqual(url, "https://api.siliconflow.cn/v1/rerank")
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(payload["model"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(payload["top_n"], 2)
        self.assertFalse(payload["return_documents"])
        self.assertEqual(timeout, 60)

    def test_non_strict_failure_returns_original_order_and_degradation(self) -> None:
        backend = SiliconFlowReranker(
            api_key="secret",
            session=FakeSession(error=RuntimeError("offline")),
            max_retries=1,
            strict=False,
        )
        candidates = [{"paragraph_id": "p1", "text": "甲"}, {"paragraph_id": "p2", "text": "乙"}]
        outcome = backend.rerank("问题", candidates, limit=2)
        self.assertEqual([row["paragraph_id"] for row in outcome.results], ["p1", "p2"])
        self.assertEqual(outcome.degraded_feature, "siliconflow_rerank")
        self.assertIn("offline", outcome.error)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run tests and verify the module is missing**

Run:

```bash
python3 -m unittest tests.test_rerank -v
```

Expected: `ERROR` with `ModuleNotFoundError`.

- [x] **Step 3: Implement the provider adapter**

Create `nihaisha_kg/rerank.py`:

```python
from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RerankOutcome:
    results: list[dict[str, object]]
    model: str
    degraded_feature: str = ""
    error: str = ""


class SiliconFlowReranker:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: int = 60,
        max_retries: int = 2,
        strict: bool = False,
        session: object | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY", "")
        self.model = model or os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.strict = strict
        self.session = session
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for SiliconFlow reranking")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        limit: int,
    ) -> RerankOutcome:
        if not candidates or limit <= 0:
            return RerankOutcome(results=[], model=self.model)
        session = self.session
        if session is None:
            import requests

            session = requests.Session()
        documents = [
            "\n".join(
                value for value in (str(row.get("title", "")).strip(), str(row.get("text", "")).strip()) if value
            )
            for row in candidates
        ]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
            "top_n": min(limit, len(documents)),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = session.post(
                    f"{self.base_url}/rerank",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                ranked: list[dict[str, object]] = []
                for item in response.json()["results"]:
                    candidate = dict(candidates[int(item["index"])])
                    candidate["rerank_score"] = float(item["relevance_score"])
                    ranked.append(candidate)
                return RerankOutcome(results=ranked, model=self.model)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        if self.strict:
            raise RuntimeError(f"SiliconFlow rerank request failed: {last_error}") from last_error
        return RerankOutcome(
            results=[dict(row) for row in candidates[:limit]],
            model=self.model,
            degraded_feature="siliconflow_rerank",
            error=str(last_error),
        )
```

- [x] **Step 4: Rerank once after all query rewrites**

Add `reranker_backend: object | None = None`, `reranker: str = "none"`, and `rerank_model: str | None = None` to `answer_pdf_rag`. The library default is deliberately network-free; Task 7 gives the user-facing CLI its `auto` default. Validate `reranker` as one of `auto`, `none`, or `siliconflow`. After rewrite fusion and intent filtering, but before `select_diverse_results`, add:

```python
    rerank_outcome = None
    if reranker_backend is not None:
        rerank_outcome = reranker_backend.rerank(
            query,
            intent_results,
            limit=min(len(intent_results), max(limit * 3, 12)),
        )
    elif reranker == "siliconflow" or (reranker == "auto" and siliconflow_api_key_available()):
        from .rerank import SiliconFlowReranker

        rerank_outcome = SiliconFlowReranker(model=rerank_model).rerank(
            query,
            intent_results,
            limit=min(len(intent_results), max(limit * 3, 12)),
        )
    if rerank_outcome is not None:
        intent_results = rerank_outcome.results
```

Attach only bounded, serializable rerank metadata to the answer as `{"model": ..., "degraded_feature": ..., "error": ...}`; never attach candidate documents or credentials. Task 7 can consume this metadata in its trace. Do not rerank inside each query rewrite; the API call count for one `answer` command must be at most one. Existing direct answer tests pass `reranker="none"`, and orchestration tests inject a fake backend so tests never reach the network.

- [x] **Step 5: Document environment settings**

Append to `.env.example`:

```dotenv
# Optional answer-stage reranker. Library callers opt in; the CLI may select it in auto mode.
SILICONFLOW_RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

- [x] **Step 6: Run rerank and answer tests**

Run:

```bash
python3 -m unittest tests.test_rerank tests.test_answer_quality -v
```

Expected: all tests pass and the fake session receives one request.

- [x] **Step 7: Commit**

```bash
git add nihaisha_kg/rerank.py nihaisha_kg/pdf_vector.py tests/test_rerank.py .env.example
git commit -m "feat: add optional SiliconFlow evidence reranking"
```

## Task 6: Fail Fast without FAISS and Add Runtime Diagnostics

**Files:**

- Create: `nihaisha_kg/diagnostics.py`
- Create: `tests/test_diagnostics.py`
- Modify: `nihaisha_kg/pdf_vector.py:1183-1295,1303-1313,2070-2169`
- Modify: `pyproject.toml:11-17`

- [ ] **Step 1: Write failing production-size guard and doctor tests**

Create `tests/test_diagnostics.py`:

```python
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from nihaisha_kg.diagnostics import doctor
from nihaisha_kg.pdf_vector import LocalVectorStore


class DiagnosticsTests(unittest.TestCase):
    def test_doctor_reports_missing_faiss_module_for_dense_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "rag.sqlite"
            store = LocalVectorStore(db)
            store.recreate()
            with store.connect() as conn:
                conn.execute("UPDATE meta SET value = 'dense' WHERE key = 'vector_kind'")
                conn.execute("UPDATE meta SET value = '10' WHERE key = 'retrieval_units'")

            report = doctor(db, faiss_loader=lambda: None)

        codes = {check["code"] for check in report["checks"] if check["status"] == "error"}
        self.assertIn("faiss_files_missing", codes)
        self.assertEqual(report["status"], "error")

    def test_dense_search_refuses_large_bruteforce_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "rag.sqlite"
            store = LocalVectorStore(db, brute_force_limit=2)
            store.recreate()
            with store.connect() as conn:
                conn.executemany(
                    "INSERT INTO retrieval_units "
                    "(unit_id, paragraph_id, doc_id, unit_type, text, text_for_embedding, "
                    "sentence_start, sentence_end, weight, vector_kind, vector_blob) "
                    "VALUES (?, 'p', 'd', 'sentence', 'x', 'x', 0, 0, 1, 'dense', ?)",
                    [(f"u{index}", sqlite3.Binary(b"\x00\x00\x00\x00")) for index in range(3)],
                )
                conn.execute("UPDATE meta SET value = 'dense' WHERE key = 'vector_kind'")
            store.embedding_backend.vector_kind = "dense"
            store.embedding_backend.embed_texts = lambda texts: [[1.0]]

            with self.assertRaisesRegex(RuntimeError, "faiss"):
                store.search_vector("query", faiss_module=False)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm diagnostics are missing**

Run:

```bash
python3 -m unittest tests.test_diagnostics -v
```

Expected: `ERROR` with `ModuleNotFoundError`.

- [ ] **Step 3: Implement doctor checks**

Create `nihaisha_kg/diagnostics.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable


def doctor(db_path: Path, faiss_loader: Callable[[], object | None]) -> dict[str, object]:
    db_path = db_path.expanduser().resolve()
    checks: list[dict[str, str]] = []

    def add(code: str, status: str, message: str) -> None:
        checks.append({"code": code, "status": status, "message": message})

    if not db_path.exists():
        add("database_missing", "error", f"database does not exist: {db_path}")
        return {"status": "error", "db_path": str(db_path), "checks": checks}
    if db_path.stat().st_size < 1024:
        add("database_lfs_pointer", "error", "database is too small; run git lfs pull")
        return {"status": "error", "db_path": str(db_path), "checks": checks}

    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"paragraphs", "retrieval_units", "paragraphs_fts", "knowledge_units"}
        missing = required - tables
        add(
            "sqlite_schema",
            "error" if missing else "ok",
            f"missing tables: {', '.join(sorted(missing))}" if missing else "required tables exist",
        )
        meta = dict(conn.execute("SELECT key, value FROM meta")) if "meta" in tables else {}
        vector_count = conn.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0]

    if meta.get("vector_kind") == "dense":
        index_path = db_path.with_name("vectors.faiss")
        ids_path = db_path.with_name("vector_ids.jsonl")
        if not index_path.exists() or not ids_path.exists():
            add("faiss_files_missing", "error", "vectors.faiss or vector_ids.jsonl is missing")
        elif faiss_loader() is None:
            add("faiss_module_missing", "error", "install with: python3 -m pip install -e '.[faiss]' ")
        else:
            add("faiss_runtime", "ok", "FAISS files and Python module are available")
        add("vector_count", "ok", f"SQLite retrieval units: {vector_count}")

    status = "error" if any(check["status"] == "error" for check in checks) else "ok"
    return {"status": status, "db_path": str(db_path), "checks": checks}
```

When wiring this module, pass `load_faiss_module` from `pdf_vector.py`; keeping it injected avoids a circular import.

- [ ] **Step 4: Add a bounded brute-force guard**

Add `brute_force_limit: int = 10_000` to `LocalVectorStore.__init__` and store it. In `search_vector`, after `_search_vector_faiss` returns `None`, count retrieval units:

```python
            unit_count = int(conn.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0])
            if dense_query_vector is not None and unit_count > self.brute_force_limit:
                raise RuntimeError(
                    f"dense search requires FAISS for {unit_count} retrieval units; "
                    "install with `python3 -m pip install -e \".[faiss]\"` and run `nihaisha-rag doctor`"
                )
```

Interpret `faiss_module=False` as an explicit unavailable module in `_search_vector_faiss`, so the test cannot accidentally import an installed FAISS package. Small temporary test stores remain allowed to use brute-force search.

- [ ] **Step 5: Cache production FAISS bundles**

Add a module-level cache keyed by resolved paths and modification times:

```python
_FAISS_BUNDLE_CACHE: dict[tuple[str, int, str, int], tuple[object, list[str]]] = {}


def load_faiss_bundle(faiss: object, index_path: Path, ids_path: Path) -> tuple[object, list[str]]:
    key = (
        str(index_path.resolve()),
        index_path.stat().st_mtime_ns,
        str(ids_path.resolve()),
        ids_path.stat().st_mtime_ns,
    )
    cached = _FAISS_BUNDLE_CACHE.get(key)
    if cached is not None:
        return cached
    bundle = (faiss.read_index(str(index_path)), read_faiss_unit_ids(ids_path))
    _FAISS_BUNDLE_CACHE.clear()
    _FAISS_BUNDLE_CACHE[key] = bundle
    return bundle
```

Use the bundle in `_search_vector_faiss`. Extend the fake-FAISS test to call twice and assert `read_index` runs once.

- [ ] **Step 6: Add the recommended runtime extra**

In `pyproject.toml`, preserve `faiss` and add:

```toml
runtime = [
  "faiss-cpu>=1.8.0",
]
```

This keeps text-only installation possible while giving README a single recommended `.[runtime]` install path.

- [ ] **Step 7: Run diagnostics and vector tests**

Run:

```bash
python3 -m unittest \
  tests.test_diagnostics \
  tests.test_pdf_vector.PdfVectorTests.test_vector_search_uses_faiss_index_when_available \
  tests.test_pdf_vector.PdfVectorTests.test_vector_store_search_returns_paragraphs_deduplicated_by_score \
  -v
```

Expected: all selected tests pass; small stores scan, production-sized stores fail fast.

- [ ] **Step 8: Commit**

```bash
git add nihaisha_kg/diagnostics.py nihaisha_kg/pdf_vector.py tests/test_diagnostics.py tests/test_pdf_vector.py pyproject.toml
git commit -m "feat: add FAISS health checks and scan guard"
```

## Task 7: Wire Doctor, Evaluate, Reranker, and Query Trace into the CLI

**Files:**

- Modify: `nihaisha_kg/cli.py:20-264`
- Modify: `nihaisha_kg/evaluation.py`
- Modify: `nihaisha_kg/pdf_vector.py:3058-3077,3570-3637`
- Modify: `tests/test_pdf_vector.py`

- [ ] **Step 1: Write failing CLI dispatch and trace tests**

Add these methods to the existing CLI test class in `tests/test_pdf_vector.py`:

```python
    def test_cli_doctor_returns_nonzero_for_error_report(self) -> None:
        with mock.patch("nihaisha_kg.cli.run_doctor") as run_doctor:
            run_doctor.return_value = {"status": "error", "checks": []}
            exit_code = rag_cli.main(["doctor", "--db", "missing.sqlite"])
        self.assertEqual(exit_code, 1)

    def test_cli_evaluate_prints_aggregate_json(self) -> None:
        payload = {"cases": 2, "aggregate": {"recall_at_5": 1.0}}
        with mock.patch("nihaisha_kg.cli.evaluate_database", return_value=payload):
            with mock.patch("builtins.print") as print_mock:
                exit_code = rag_cli.main(
                    ["evaluate", "--db", "rag.sqlite", "--cases", "evals/golden_v1.jsonl", "--mode", "text"]
                )
        self.assertEqual(exit_code, 0)
        self.assertIn('"recall_at_5": 1.0', print_mock.call_args.args[0])

    def test_answer_trace_contains_plan_channels_and_latency(self) -> None:
        store = mock.Mock()
        store.search.return_value = []
        store.search_knowledge_units.return_value = []
        store.search_guide_nodes.return_value = []
        with mock.patch("nihaisha_kg.pdf_vector.LocalVectorStore", return_value=store):
            payload = answer_pdf_rag(
                "桂枝汤出处",
                db_path=Path("rag.sqlite"),
                mode="text",
                trace_enabled=True,
                reranker="none",
            )
        self.assertIn("query_plan", payload["trace"])
        self.assertIn("retrieval_channels", payload["trace"])
        self.assertIn("latency_ms", payload["trace"])

    def test_cli_search_calls_reranker_once_after_retrieval(self) -> None:
        store = mock.Mock()
        candidates = [{"paragraph_id": "p1", "title": "伤寒", "text": "桂枝汤主之"}]
        store.search.return_value = candidates
        reranker = mock.Mock()
        reranker.rerank.return_value = mock.Mock(
            results=candidates,
            model="BAAI/bge-reranker-v2-m3",
            degraded_feature="",
        )
        with mock.patch("nihaisha_kg.cli.create_embedding_backend_for_db", return_value=mock.Mock()):
            with mock.patch("nihaisha_kg.cli.LocalVectorStore", return_value=store):
                with mock.patch("nihaisha_kg.rerank.SiliconFlowReranker", return_value=reranker):
                    with mock.patch("builtins.print"):
                        exit_code = rag_cli.main(
                            ["search", "桂枝汤", "--reranker", "siliconflow", "--json"]
                        )
        self.assertEqual(exit_code, 0)
        store.search.assert_called_once()
        reranker.rerank.assert_called_once()
```

- [ ] **Step 2: Run the selected tests and confirm missing CLI symbols/options**

Run:

```bash
python3 -m unittest \
  tests.test_pdf_vector.PdfVectorTests.test_cli_doctor_returns_nonzero_for_error_report \
  tests.test_pdf_vector.PdfVectorTests.test_cli_evaluate_prints_aggregate_json \
  tests.test_pdf_vector.PdfVectorTests.test_answer_trace_contains_plan_channels_and_latency \
  -v
```

Expected: failures for missing `run_doctor`, `evaluate_database`, and trace parameters.

- [ ] **Step 3: Add database evaluation orchestration**

Append to `nihaisha_kg/evaluation.py`:

```python
def evaluate_database(
    store: object,
    cases: list[EvalCase],
    mode: str,
    limit: int = 10,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, float]] = []
    for case in cases:
        results = store.search(case.query, limit=limit, mode=mode)
        ranked_ids = [str(result.get("paragraph_id", "")) for result in results]
        metrics = evaluate_ranked_ids(case, ranked_ids, k_values=(1, 5, 10))
        metric_rows.append(metrics)
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "ranked_paragraph_ids": ranked_ids,
                "metrics": metrics,
            }
        )
    return {"cases": len(rows), "aggregate": aggregate_metrics(metric_rows), "results": rows}
```

- [ ] **Step 4: Add a bounded trace to `answer_pdf_rag`**

Add `trace_enabled: bool = False`, record `time.perf_counter()` around planning, retrieval, rerank, guide lookup, and synthesis, then attach:

```python
    if trace_enabled:
        answer["trace"] = {
            "normalized_query": normalize_query_text(query),
            "intent": intent,
            "query_plan": initial_plan,
            "followup_plan": followup_plan,
            "retrieval_channels": sorted(
                {source for result in results for source in result.get("retrieval_sources", [])}
            ),
            "selected_paragraph_ids": [str(result.get("paragraph_id", "")) for result in results],
            "reranker": rerank_outcome.model if rerank_outcome is not None else "none",
            "degraded_features": (
                [rerank_outcome.degraded_feature]
                if rerank_outcome is not None and rerank_outcome.degraded_feature
                else []
            ),
            "latency_ms": {key: round(value * 1000, 2) for key, value in timings.items()},
        }
```

Do not include API keys, request headers, environment dumps, full vectors, or unrestricted candidate text in the trace.

- [ ] **Step 5: Add CLI parsers and dispatch**

In `nihaisha_kg/cli.py`, import diagnostics and evaluation, aliasing diagnostics as:

```python
from .diagnostics import doctor as run_doctor
from .evaluation import evaluate_database, load_eval_cases
from .pdf_vector import load_faiss_module
```

Add parsers:

```python
    doctor_parser = sub.add_parser("doctor", help="check database and retrieval runtime health")
    doctor_parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    evaluate_parser = sub.add_parser("evaluate", help="run versioned retrieval evaluation cases")
    evaluate_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    evaluate_parser.add_argument("--cases", type=Path, default=Path("evals/golden_v1.jsonl"))
    evaluate_parser.add_argument("--mode", choices=["hybrid", "vector", "text", "knowledge"], default="hybrid")
    evaluate_parser.add_argument("--limit", type=int, default=10)
```

After both `search` and `answer` parsers have been created, add:

```python
    for command_parser in (search, answer):
        command_parser.add_argument(
            "--reranker",
            choices=["auto", "none", "siliconflow"],
            default="auto",
        )
        command_parser.add_argument(
            "--rerank-model",
            default=None,
            help="SiliconFlow reranker model; defaults to SILICONFLOW_RERANK_MODEL or BAAI/bge-reranker-v2-m3",
        )
        command_parser.add_argument("--trace", action="store_true")
```

Dispatch doctor and evaluate before build/search handling:

```python
    if args.command == "doctor":
        payload = run_doctor(args.db, faiss_loader=load_faiss_module)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["status"] == "ok" else 1

    if args.command == "evaluate":
        if args.mode in {"text", "knowledge"}:
            store = LocalVectorStore(args.db)
        else:
            backend = create_embedding_backend_for_db(args.db)
            store = LocalVectorStore(args.db, embedding_backend=backend)
        payload = evaluate_database(store, load_eval_cases(args.cases), args.mode, args.limit)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
```

Also import `siliconflow_api_key_available` from `pdf_vector.py`. In the existing `search` dispatch, replace the direct `store.search(..., limit=args.limit)` call and JSON branch with:

```python
        candidate_limit = max(args.limit * 3, 12) if args.reranker != "none" else args.limit
        results = store.search(args.query, limit=candidate_limit, mode=args.mode)
        rerank_outcome = None
        if args.reranker == "siliconflow" or (
            args.reranker == "auto" and siliconflow_api_key_available()
        ):
            from .rerank import SiliconFlowReranker

            rerank_outcome = SiliconFlowReranker(model=args.rerank_model).rerank(
                args.query,
                results,
                limit=args.limit,
            )
            results = rerank_outcome.results
        else:
            results = results[: args.limit]
        if args.json:
            if args.trace:
                payload = {
                    "results": results,
                    "trace": {
                        "normalized_query": normalize_query_text(args.query),
                        "retrieval_channels": sorted(
                            {source for row in results for source in row.get("retrieval_sources", [])}
                        ),
                        "channel_ranks": {
                            str(row.get("paragraph_id", "")): row.get("channel_ranks", {})
                            for row in results
                        },
                        "reranker": rerank_outcome.model if rerank_outcome is not None else "none",
                        "degraded_features": (
                            [rerank_outcome.degraded_feature]
                            if rerank_outcome is not None and rerank_outcome.degraded_feature
                            else []
                        ),
                    },
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0
```

Import `normalize_query_text` from `normalization.py`. Leave the existing plain-text result loop immediately after this replacement so non-JSON output stays backward compatible.

In the existing `answer_pdf_rag` call, pass the new options exactly once:

```python
        payload = answer_pdf_rag(
            args.query,
            db_path=args.db,
            limit=args.limit,
            mode=args.mode,
            embedding=args.embedding,
            model=args.model,
            batch_size=args.batch_size,
            reranker=args.reranker,
            rerank_model=args.rerank_model,
            trace_enabled=args.trace,
        )
```

- [ ] **Step 6: Run CLI tests and local health commands**

Run:

```bash
python3 -m unittest tests.test_pdf_vector -v
python3 -m nihaisha_kg doctor
python3 -m nihaisha_kg evaluate --mode text --limit 10
```

Expected:

- all existing and new CLI tests pass;
- `doctor` reports the missing local FAISS module with the exact install command until `.[runtime]` is installed;
- text evaluation emits per-case metrics and an aggregate without API access.

- [ ] **Step 7: Commit**

```bash
git add nihaisha_kg/cli.py nihaisha_kg/evaluation.py nihaisha_kg/pdf_vector.py tests/test_pdf_vector.py
git commit -m "feat: expose RAG diagnostics evaluation and traces"
```

## Task 8: Rewrite Runtime Documentation and Run the Quality Gate

**Files:**

- Modify: `README.md`
- Modify: `SKILL.md`
- Modify: `.env.example`
- Create: `evals/baseline_v1.json`

- [ ] **Step 1: Update README with a copyable five-minute path**

Replace the configuration/use sections with commands that include:

```bash
git lfs pull
python3 -m pip install -e ".[runtime]"
cp .env.example .env
python3 -m nihaisha_kg doctor
python3 -m nihaisha_kg search "桂枝汤和麻黄汤的方证如何鉴别？" --mode hybrid --limit 8
python3 -m nihaisha_kg answer "木香饼热熨法来自哪一本书哪一段？" --trace
python3 -m nihaisha_kg evaluate --cases evals/golden_v1.jsonl --mode hybrid
```

Document these exact runtime rules:

- hybrid/vector requires FAISS for the bundled production database;
- text/knowledge works without the embedding API;
- `auto` rerank uses SiliconFlow when the key is configured and degrades visibly on failure;
- trace records plans, channel ranks, selected paragraph IDs, degradation, and latency but never secrets;
- runtime answers cite the current course PDFs; classical-text evidence is introduced only after phase three.

- [ ] **Step 2: Update SKILL command and answer guidance**

Add `doctor` before first use, add `evaluate` for maintainers, state that retrieval traces are preferred when reviewing questionable citations, and replace any wording that implies derived guide nodes are authoritative. Keep the existing medical safety warning exactly unchanged.

- [ ] **Step 3: Install the recommended runtime extra and run verification**

Run:

```bash
python3 -m pip install -e ".[runtime]"
python3 -m unittest discover -s tests -v
python3 -m py_compile nihaisha_kg/*.py
python3 /Users/june/.codex/skills/.system/skill-creator/scripts/quick_validate.py .
python3 -m nihaisha_kg doctor
python3 -m nihaisha_kg evaluate --cases evals/golden_v1.jsonl --mode hybrid --limit 10 > evals/baseline_v1.json
git diff --check
git lfs ls-files
git cat-file -s :data/pdf_rag_bge_m3/rag.sqlite
git cat-file -s :data/pdf_rag_bge_m3/vectors.faiss
```

Expected:

- all tests pass;
- all modules compile;
- Skill validation prints `Skill is valid!`;
- doctor status is `ok`;
- baseline JSON contains seven cases with aggregate Recall@5, Recall@10, MRR, nDCG@10, Context Precision@10, and forbidden-hit metrics;
- `git diff --check` has no output;
- SQLite and FAISS staged Git objects remain small LFS pointers.

- [ ] **Step 4: Manually inspect the two former critical failures**

Run:

```bash
python3 -m nihaisha_kg answer "桂枝汤的出处在哪本书哪一页？" --limit 6 --trace
python3 -m nihaisha_kg answer "患者咳嗽、怕冷、无汗，课程有哪些相关线索？" --limit 6 --trace
```

Expected:

- the 桂枝汤 answer contains no 木香饼/热熨 fixed text and no 木香饼 paragraph among primary citations;
- the cough answer contains no injected 下利/恶心/心下痞 checklist;
- both outputs retain the medical safety boundary when formula or clinical content is present;
- trace shows at most one reranker call through a single reranker model field.

- [ ] **Step 5: Commit documentation and the measured baseline**

```bash
git add README.md SKILL.md .env.example evals/baseline_v1.json
git commit -m "docs: publish measured RAG runtime workflow"
```

- [ ] **Step 6: Run the final phase-one gate from a clean status**

Run:

```bash
git status --short
python3 -m unittest discover -s tests -v
python3 -m nihaisha_kg doctor
```

Expected: clean status, all tests pass, and doctor reports `ok`.

## Phase-One Completion Criteria

Phase one is complete only when all of the following are true:

- the two fixed-template correctness failures are covered by tests and removed;
- natural Chinese questions produce bounded lexical terms rather than one full-sentence term;
- hybrid fusion uses channel ranks, not raw cross-channel score addition;
- optional rerank runs no more than once per final search/answer request;
- a production-sized dense store cannot silently brute-force scan without FAISS;
- `doctor`, `evaluate`, and `--trace` are documented and tested;
- the seven-case baseline is committed with aggregate recall, MRR, nDCG, context-precision, and forbidden-hit metrics;
- the full test suite, compile check, Skill validation, doctor, and LFS checks pass.
