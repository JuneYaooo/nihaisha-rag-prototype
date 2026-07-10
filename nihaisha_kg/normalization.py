from __future__ import annotations

import re
from collections.abc import Iterable


TRADITIONAL_QUERY_TRANSLATION = str.maketrans(
    {
        "錢": "钱",
        "證": "证",
        "發": "发",
        "燒": "烧",
        "噁": "恶",
        "瀉": "泻",
        "黃": "黄",
        "餅": "饼",
        "熱": "热",
        "藥": "药",
        "處": "处",
        "裡": "里",
        "來": "来",
        "頭": "头",
        "頸": "颈",
        "痠": "酸",
        "湯": "汤",
        "鑒": "鉴",
        "別": "别",
        "與": "与",
        "兩": "两",
        "現": "现",
        "歲": "岁",
        "開": "开",
    }
)

CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
ASCII_TERM_RE = re.compile(r"[A-Za-z0-9_+.-]{2,}")
MEASURE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:克|钱|两|分|升|斗|斤|铢)")

QUESTION_SCAFFOLD = (
    "哪一本书",
    "什么时候",
    "告诉我",
    "是什么",
    "有哪些",
    "哪本书",
    "哪一页",
    "哪一段",
    "课程里",
    "课程中",
    "能不能",
    "可以",
    "请问",
    "如何",
    "怎么",
    "对应",
    "相关",
    "原文",
    "出处",
    "从",
    "到",
    "的",
    "和",
    "与",
)

GENERIC_FRAGMENTS = {
    "这个",
    "那个",
    "问题",
    "资料",
    "内容",
    "时候",
    "什么",
    "哪些",
}


def normalize_query_text(query: str) -> str:
    return query.translate(TRADITIONAL_QUERY_TRANSLATION)


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        term = value.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        unique.append(term)
    return unique


def lexical_query_terms(
    query: str,
    domain_terms: Iterable[str] = (),
    max_fallback_terms: int = 12,
) -> list[str]:
    normalized = normalize_query_text(query)
    normalized_domain_terms = _dedupe_keep_order(
        normalize_query_text(term) for term in domain_terms
    )
    recognized_terms = sorted(
        (term for term in normalized_domain_terms if term in normalized),
        key=lambda term: (-len(term), term),
    )
    measure_terms = [
        match.group(0).replace(" ", "") for match in MEASURE_RE.finditer(normalized)
    ]
    ascii_terms = ASCII_TERM_RE.findall(normalized)

    remainder = normalized
    for term in recognized_terms:
        remainder = remainder.replace(term, "")
    remainder = MEASURE_RE.sub("", remainder)
    remainder = ASCII_TERM_RE.sub("", remainder)
    for phrase in sorted(QUESTION_SCAFFOLD, key=len, reverse=True):
        remainder = remainder.replace(phrase, "")

    fallback_terms: list[str] = []
    for chunk in CHINESE_RUN_RE.findall(remainder):
        candidates = (
            [chunk]
            if len(chunk) <= 8
            else [chunk[index : index + 3] for index in range(len(chunk) - 2)]
        )
        fallback_terms.extend(
            candidate for candidate in candidates if candidate not in GENERIC_FRAGMENTS
        )

    bounded_fallback_terms = _dedupe_keep_order(fallback_terms)[:max_fallback_terms]
    return _dedupe_keep_order(
        [*recognized_terms, *measure_terms, *ascii_terms, *bounded_fallback_terms]
    )
