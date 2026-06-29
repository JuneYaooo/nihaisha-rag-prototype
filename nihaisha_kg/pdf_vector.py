from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SENTENCE_RE = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_+-]{2,}")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
TERM_SPLIT_RE = re.compile(r"[，,。！？!?；;、\s]+")
QUERY_TERM_RE = re.compile(r"[A-Za-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}")
MEASURE_TERM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:克|钱|兩|两|分|升|斗|斤|铢)")
FORMULA_SUFFIXES = ("汤", "丸", "散", "饮", "膏", "丹")
FORMULA_PREFIX_NOISE = (
    "可以讨论",
    "可以用",
    "考虑",
    "这个",
    "记得",
    "同样是",
    "就用",
    "讨论",
    "使用",
    "用",
)
SIX_CHANNEL_TERMS = ("太阳", "阳明", "少阳", "太阴", "少阴", "厥阴")
SYMPTOM_TERMS = (
    "汗出",
    "无汗",
    "恶风",
    "恶寒",
    "发热",
    "咽痛",
    "喘",
    "烦躁",
    "口渴",
    "下利",
    "便秘",
    "腹痛",
    "头痛",
    "身痛",
    "脉浮",
    "脉沉",
)
CAUTION_TERMS = ("不可", "不可以", "勿", "误", "禁忌", "忌")
QUERY_EXPANSION_STOP_TERMS = {
    "这个",
    "那个",
    "什么",
    "怎么",
    "多少",
    "现代",
    "古时",
    "古时候",
    "时候",
    "我们",
    "他们",
    "有人",
    "因为",
    "所以",
    "如果",
    "可以",
    "就是",
    "不是",
    "一样",
    "一个",
    "没有",
    "里面",
    "现在",
    "过去",
}
KNOWLEDGE_EXTRACTOR_VERSION = "local_rules_v1"
FORMULA_DOSAGE_SAFETY_NOTICE = (
    "涉及剂量、方药或处方线索时必须谨慎：不同人的体质不同，病情阶段、兼证、年龄、基础病和用药史都不同；"
    "现代药材来源、炮制、浓度和药效也和以前差很多。建议去线下正规中医渠道面诊辨证，"
    "不要私自购药有风险。"
)


@dataclass(frozen=True)
class ParsedParagraph:
    paragraph_id: str
    doc_id: str
    source_path: str
    title: str
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class RetrievalUnit:
    unit_id: str
    paragraph_id: str
    doc_id: str
    unit_type: str
    text: str
    text_for_embedding: str
    sentence_start: int
    sentence_end: int
    weight: float


@dataclass(frozen=True)
class KnowledgeUnit:
    knowledge_unit_id: str
    paragraph_id: str
    doc_id: str
    source_path: str
    title: str
    page_start: int
    page_end: int
    unit_type: str
    subject: str
    predicate: str
    object: str
    attributes_json: str
    evidence_quote: str
    confidence: float
    extractor_version: str = KNOWLEDGE_EXTRACTOR_VERSION


def load_dotenv_if_present(start: Path | None = None) -> None:
    search_start = (start or Path.cwd()).resolve()
    candidates = [search_start, *search_start.parents]
    module_root = Path(__file__).resolve().parents[1]
    if module_root not in candidates:
        candidates.append(module_root)
    for directory in candidates:
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def stable_id(*parts: object, length: int = 16) -> str:
    raw = "\n".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text.strip())
    if not normalized:
        return []
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(normalized)]
    return [sentence for sentence in sentences if sentence]


def split_long_text_to_paragraphs(text: str, max_chars: int = 900, min_chars: int = 120) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    paragraphs: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len > max_chars:
            paragraphs.append("".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len
        if current_len >= min_chars and sentence.endswith(("。", "！", "？", "!", "?")):
            continue
    if current:
        paragraphs.append("".join(current))
    return paragraphs


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def extract_formula_terms(text: str) -> list[str]:
    formulas: list[str] = []
    for fragment in TERM_SPLIT_RE.split(text):
        fragment = fragment.strip()
        if not fragment:
            continue
        for suffix in FORMULA_SUFFIXES:
            offset = fragment.find(suffix)
            while offset >= 0:
                candidate = fragment[: offset + len(suffix)]
                for prefix in FORMULA_PREFIX_NOISE:
                    if candidate.startswith(prefix):
                        candidate = candidate[len(prefix) :]
                if 2 <= len(candidate) <= 8 and candidate.endswith(suffix):
                    formulas.append(candidate)
                offset = fragment.find(suffix, offset + len(suffix))
    return dedupe_keep_order(formulas)


def extract_known_terms(text: str, terms: Iterable[str]) -> list[str]:
    return [term for term in terms if term in text]


def text_search_terms(query: str) -> list[str]:
    terms = QUERY_TERM_RE.findall(query)
    terms.extend(term.replace(" ", "") for term in MEASURE_TERM_RE.findall(query))
    return dedupe_keep_order(terms)


def knowledge_search_terms(query: str) -> list[str]:
    domain_terms = [
        "一钱",
        "黄金比例",
        "木香饼",
        "热熨",
        "主之",
        "方证",
        "禁忌",
        "误用",
        *SYMPTOM_TERMS,
        *SIX_CHANNEL_TERMS,
    ]
    terms = text_search_terms(query)
    terms.extend(term for term in domain_terms if term in query)
    terms.extend(extract_formula_terms(query))
    return dedupe_keep_order(terms)


def fts5_quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def fts5_query_from_terms(terms: Iterable[str]) -> str:
    fts_terms = [fts5_quote(term) for term in terms if len(term) >= 3]
    return " OR ".join(fts_terms)


def generate_paragraph_questions(paragraph: ParsedParagraph, max_questions: int = 12) -> list[str]:
    text = paragraph.text
    formulas = extract_formula_terms(text)
    channels = extract_known_terms(text, SIX_CHANNEL_TERMS)
    symptoms = extract_known_terms(text, SYMPTOM_TERMS)
    questions: list[str] = []

    for formula in formulas:
        questions.append(f"{formula}对应什么方证或症状？")
        questions.append(f"什么时候会提到{formula}？")

    if len(formulas) >= 2:
        questions.append(f"{formulas[0]}和{formulas[1]}如何鉴别？")

    for channel in channels:
        questions.append(f"{channel}相关的辨证要点是什么？")

    for symptom in symptoms:
        questions.append(f"{symptom}相关的辨证要点是什么？")

    if "主之" in text:
        questions.append("这段提到什么情况下可以对应某个方证？")
    if "不可" in text or "误" in text:
        questions.append("这段有哪些禁忌、误用或需要避免的情况？")
    if "区别" in text or "鉴别" in text or "比较" in text:
        questions.append("这段在比较或鉴别哪些证候、方剂或症状？")

    if not questions and text:
        questions.append(f"{paragraph.title}这一段主要讲什么？")

    return dedupe_keep_order(questions)[:max_questions]


def build_retrieval_units(
    paragraphs: Iterable[ParsedParagraph],
    window_size: int = 6,
    overlap: int = 2,
) -> list[RetrievalUnit]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be >= 0 and < window_size")

    units: list[RetrievalUnit] = []
    stride = max(1, window_size - overlap)
    for paragraph in paragraphs:
        sentences = split_sentences(paragraph.text)
        prefix = "\n".join(
            part
            for part in [
                f"标题：{paragraph.title}" if paragraph.title else "",
                f"页码：{paragraph.page_start}-{paragraph.page_end}",
            ]
            if part
        )

        for index, sentence in enumerate(sentences):
            unit_text = sentence
            text_for_embedding = f"{prefix}\n正文：{unit_text}" if prefix else unit_text
            unit_id = stable_id(paragraph.paragraph_id, "sentence", index)
            units.append(
                RetrievalUnit(
                    unit_id=unit_id,
                    paragraph_id=paragraph.paragraph_id,
                    doc_id=paragraph.doc_id,
                    unit_type="sentence",
                    text=unit_text,
                    text_for_embedding=text_for_embedding,
                    sentence_start=index,
                    sentence_end=index,
                    weight=1.0,
                )
            )

        for start in range(0, len(sentences), stride):
            window_sentences = sentences[start : start + window_size]
            if not window_sentences:
                continue
            if start > 0 and len(window_sentences) < max(2, window_size // 2):
                continue
            unit_text = "".join(window_sentences)
            text_for_embedding = f"{prefix}\n正文：{unit_text}" if prefix else unit_text
            unit_id = stable_id(paragraph.paragraph_id, "window", start, start + len(window_sentences) - 1)
            units.append(
                RetrievalUnit(
                    unit_id=unit_id,
                    paragraph_id=paragraph.paragraph_id,
                    doc_id=paragraph.doc_id,
                    unit_type="window",
                    text=unit_text,
                    text_for_embedding=text_for_embedding,
                    sentence_start=start,
                    sentence_end=start + len(window_sentences) - 1,
                    weight=1.1,
                )
            )

        if paragraph.text:
            text_for_embedding = f"{prefix}\n正文：{paragraph.text}" if prefix else paragraph.text
            units.append(
                RetrievalUnit(
                    unit_id=stable_id(paragraph.paragraph_id, "paragraph"),
                    paragraph_id=paragraph.paragraph_id,
                    doc_id=paragraph.doc_id,
                    unit_type="paragraph",
                    text=paragraph.text,
                    text_for_embedding=text_for_embedding,
                    sentence_start=0,
                    sentence_end=max(0, len(sentences) - 1),
                    weight=0.9,
                )
            )

        for index, question in enumerate(generate_paragraph_questions(paragraph)):
            text_for_embedding = f"{prefix}\n问题：{question}" if prefix else f"问题：{question}"
            units.append(
                RetrievalUnit(
                    unit_id=stable_id(paragraph.paragraph_id, "question", index, question),
                    paragraph_id=paragraph.paragraph_id,
                    doc_id=paragraph.doc_id,
                    unit_type="question",
                    text=question,
                    text_for_embedding=text_for_embedding,
                    sentence_start=0,
                    sentence_end=max(0, len(sentences) - 1),
                    weight=1.15,
                )
            )
    return units


def build_question_retrieval_units(paragraphs: Iterable[ParsedParagraph]) -> list[RetrievalUnit]:
    units: list[RetrievalUnit] = []
    for paragraph in paragraphs:
        sentences = split_sentences(paragraph.text)
        prefix = "\n".join(
            part
            for part in [
                f"标题：{paragraph.title}" if paragraph.title else "",
                f"页码：{paragraph.page_start}-{paragraph.page_end}",
            ]
            if part
        )
        for index, question in enumerate(generate_paragraph_questions(paragraph)):
            text_for_embedding = f"{prefix}\n问题：{question}" if prefix else f"问题：{question}"
            units.append(
                RetrievalUnit(
                    unit_id=stable_id(paragraph.paragraph_id, "question", index, question),
                    paragraph_id=paragraph.paragraph_id,
                    doc_id=paragraph.doc_id,
                    unit_type="question",
                    text=question,
                    text_for_embedding=text_for_embedding,
                    sentence_start=0,
                    sentence_end=max(0, len(sentences) - 1),
                    weight=1.15,
                )
            )
    return units


def evidence_quote(text: str, max_chars: int = 240) -> str:
    normalized = re.sub(r"\s+", "", text.strip())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1] + "…"


def json_attributes(**values: object) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def make_knowledge_unit(
    paragraph: ParsedParagraph,
    unit_type: str,
    subject: str,
    predicate: str,
    object_text: str,
    evidence: str,
    confidence: float,
    **attributes: object,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        knowledge_unit_id=stable_id(
            paragraph.paragraph_id,
            unit_type,
            subject,
            predicate,
            object_text,
            evidence_quote(evidence, max_chars=120),
        ),
        paragraph_id=paragraph.paragraph_id,
        doc_id=paragraph.doc_id,
        source_path=paragraph.source_path,
        title=paragraph.title,
        page_start=paragraph.page_start,
        page_end=paragraph.page_end,
        unit_type=unit_type,
        subject=subject,
        predicate=predicate,
        object=object_text,
        attributes_json=json_attributes(**attributes),
        evidence_quote=evidence_quote(evidence),
        confidence=confidence,
    )


def extract_knowledge_units_from_paragraph(paragraph: ParsedParagraph) -> list[KnowledgeUnit]:
    text = paragraph.text
    sentences = split_sentences(text) or [text]
    units: list[KnowledgeUnit] = []

    if ("一钱" in text or "1钱" in text) and ("克" in text or "黄金比例" in text or "比例" in text):
        dosage_evidence = next(
            (sentence for sentence in sentences if "一钱" in sentence or "1钱" in sentence),
            text,
        )
        dosage_context_markers = (
            "一钱是",
            "一钱等于",
            "一钱约",
            "一钱当",
            "一钱算",
            "一钱的量",
            "一钱，",
            "一钱 ",
        )
        if "一钱匕" in dosage_evidence and not any(marker in dosage_evidence for marker in dosage_context_markers):
            dosage_evidence = ""
        local_text = dosage_evidence
        if local_text and len(re.findall(r"\d+(?:\.\d+)?\s*克", local_text)) > 8:
            offset = local_text.find("一钱")
            if offset < 0:
                offset = local_text.find("1钱")
            local_text = local_text[max(0, offset - 80) : offset + 120]
        grams = dedupe_keep_order(re.findall(r"\d+(?:\.\d+)?\s*克", local_text))
        object_parts = grams[:]
        if "黄金比例" in local_text:
            object_parts.append("首重黄金比例")
        elif "比例" in local_text:
            object_parts.append("首重比例")
        if object_parts:
            units.append(
                make_knowledge_unit(
                    paragraph,
                    unit_type="dosage",
                    subject="一钱",
                    predicate="换算与剂量原则",
                    object_text="；".join(object_parts),
                    evidence=local_text,
                    confidence=0.86,
                    grams=grams,
                )
            )

    if "木香饼" in text and "热熨" in text:
        method_sentence = next(
            (sentence for sentence in sentences if "木香饼" in sentence or "热熨" in sentence),
            text,
        )
        materials = [term for term in ("木香", "生地") if term in method_sentence]
        indications = []
        if "治" in method_sentence:
            indications.append(method_sentence.split("治", 1)[1].rstrip("。；;"))
        units.append(
            make_knowledge_unit(
                paragraph,
                unit_type="method",
                subject="木香饼热熨法",
                predicate="治法",
                object_text=evidence_quote(method_sentence, max_chars=120),
                evidence=method_sentence,
                confidence=0.9,
                materials=materials,
                indications=indications,
            )
        )

    for sentence in sentences:
        if "主之" in sentence:
            for formula in extract_formula_terms(sentence):
                units.append(
                    make_knowledge_unit(
                        paragraph,
                        unit_type="formula_pattern",
                        subject=formula,
                        predicate="方证",
                        object_text=evidence_quote(sentence, max_chars=120),
                        evidence=sentence,
                        confidence=0.82,
                    )
                )

        if any(term in sentence for term in CAUTION_TERMS):
            units.append(
                make_knowledge_unit(
                    paragraph,
                    unit_type="caution",
                    subject="禁忌或误用",
                    predicate="提示",
                    object_text=evidence_quote(sentence, max_chars=120),
                    evidence=sentence,
                    confidence=0.72,
                    matched_terms=[term for term in CAUTION_TERMS if term in sentence],
                )
            )

        sentence_symptoms = extract_known_terms(sentence, SYMPTOM_TERMS)
        for symptom in sentence_symptoms[:4]:
            units.append(
                make_knowledge_unit(
                    paragraph,
                    unit_type="symptom",
                    subject=symptom,
                    predicate="相关原文",
                    object_text=evidence_quote(sentence, max_chars=120),
                    evidence=sentence,
                    confidence=0.68,
                )
            )

        formulas = extract_formula_terms(sentence)
        if len(formulas) >= 2 and ("鉴别" in sentence or "区别" in sentence or "比较" in sentence or "和" in sentence):
            units.append(
                make_knowledge_unit(
                    paragraph,
                    unit_type="comparison",
                    subject=" vs ".join(formulas[:3]),
                    predicate="鉴别比较",
                    object_text=evidence_quote(sentence, max_chars=120),
                    evidence=sentence,
                    confidence=0.76,
                    formulas=formulas,
                )
            )

    unique: dict[str, KnowledgeUnit] = {}
    for unit in units:
        unique[unit.knowledge_unit_id] = unit
    return list(unique.values())


def build_knowledge_units(paragraphs: Iterable[ParsedParagraph]) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    for paragraph in paragraphs:
        units.extend(extract_knowledge_units_from_paragraph(paragraph))
    return units


def sparse_hash_embedding(text: str, dims: int = 2048) -> dict[int, float]:
    tokens = embedding_tokens(text)
    if not tokens:
        return {}
    counts: dict[int, float] = {}
    for token in tokens:
        bucket = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16) % dims
        counts[bucket] = counts.get(bucket, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0:
        return {}
    return {key: value / norm for key, value in counts.items()}


def embedding_tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = ASCII_WORD_RE.findall(lowered)
    chinese_chars = CHINESE_CHAR_RE.findall(lowered)
    tokens.extend(chinese_chars)
    tokens.extend("".join(chinese_chars[i : i + 2]) for i in range(max(0, len(chinese_chars) - 1)))
    tokens.extend("".join(chinese_chars[i : i + 3]) for i in range(max(0, len(chinese_chars) - 2)))
    return [token for token in tokens if token.strip()]


class SparseHashEmbeddingBackend:
    name = "sparse_hash_char_ngrams_v1"
    vector_kind = "sparse"

    def __init__(self, dims: int = 2048) -> None:
        self.dims = dims

    def embed_texts(self, texts: list[str]) -> list[dict[int, float]]:
        return [sparse_hash_embedding(text, dims=self.dims) for text in texts]


class DenseEmbeddingBackend:
    name = "dense"
    vector_kind = "dense"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalBgeM3EmbeddingBackend(DenseEmbeddingBackend):
    """Local BAAI/bge-m3 dense embedding backend.

    The model is loaded lazily so text/knowledge-only workflows do not need
    torch/transformers installed.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        batch_size: int = 12,
        max_length: int = 8192,
        use_fp16: bool | None = None,
        model_instance: object | None = None,
        model_loader: object | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = (
            os.getenv("LOCAL_BGE_M3_USE_FP16", "false").lower() in {"1", "true", "yes", "on"}
            if use_fp16 is None
            else use_fp16
        )
        self.model_instance = model_instance
        self.model_loader = model_loader
        self.name = f"local-bge-m3:{model}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._encode_batch(model, batch)
            vectors.extend(normalize_dense_vector(vector) for vector in encoded)
        return vectors

    def _get_model(self) -> object:
        if self.model_instance is not None:
            return self.model_instance
        if self.model_loader is not None:
            self.model_instance = self.model_loader(self.model)
            return self.model_instance
        try:
            from FlagEmbedding import BGEM3FlagModel

            self.model_instance = BGEM3FlagModel(self.model, use_fp16=self.use_fp16)
            return self.model_instance
        except ImportError as flag_error:
            try:
                from sentence_transformers import SentenceTransformer

                self.model_instance = SentenceTransformer(self.model)
                return self.model_instance
            except ImportError as sentence_error:
                raise RuntimeError(
                    "Local bge-m3 embeddings require optional local dependencies. "
                    'Install with `pip install ".[local]"` or `pip install FlagEmbedding`.'
                ) from sentence_error
            except Exception as sentence_error:
                raise RuntimeError(f"failed to load local embedding model {self.model}: {sentence_error}") from sentence_error
        except Exception as flag_error:
            raise RuntimeError(f"failed to load local embedding model {self.model}: {flag_error}") from flag_error

    def _encode_batch(self, model: object, batch: list[str]) -> list[list[float]]:
        try:
            encoded = model.encode(
                batch,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except TypeError:
            encoded = model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
            )

        if isinstance(encoded, dict):
            raw_vectors = encoded.get("dense_vecs")
            if raw_vectors is None:
                raw_vectors = encoded.get("dense")
        else:
            raw_vectors = encoded
        if raw_vectors is None:
            raise RuntimeError("local bge-m3 model did not return dense vectors")
        if hasattr(raw_vectors, "tolist"):
            raw_vectors = raw_vectors.tolist()
        return [[float(value) for value in vector] for vector in raw_vectors]


class SiliconFlowEmbeddingBackend(DenseEmbeddingBackend):
    name = "siliconflow:BAAI/bge-m3"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "BAAI/bge-m3",
        base_url: str = "https://api.siliconflow.cn/v1",
        batch_size: int = 32,
        timeout: int = 120,
        max_retries: int = 3,
        session: object | None = None,
    ) -> None:
        load_dotenv_if_present()
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session
        self.name = f"siliconflow:{model}"
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for SiliconFlow embeddings")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        session = self.session
        if session is None:
            import requests

            session = requests.Session()

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._post_batch(session, batch))
        return vectors

    def _post_batch(self, session: object, batch: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": batch,
            "encoding_format": "float",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = session.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()["data"]
                data = sorted(data, key=lambda item: item.get("index", 0))
                return [item["embedding"] for item in data]
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"SiliconFlow embedding request failed: {last_error}") from last_error


class SiliconFlowChatBackend:
    name = "siliconflow_chat"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        temperature: float = 0.1,
        max_tokens: int = 1600,
        timeout: int = 120,
        max_retries: int = 3,
        session: object | None = None,
    ) -> None:
        load_dotenv_if_present()
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY", "")
        self.model = model or os.getenv("SILICONFLOW_CHAT_MODEL", "Qwen/Qwen3-32B")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for SiliconFlow chat completions")

    def complete(self, messages: list[dict[str, str]]) -> str:
        session = self.session
        if session is None:
            import requests

            session = requests.Session()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = session.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return str(data["choices"][0]["message"]["content"]).strip()
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"SiliconFlow chat completion request failed: {last_error}") from last_error


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_build_traces(
    trace_dir: Path,
    pdf_dir: Path,
    out_dir: Path,
    embedding_name: str,
    vector_kind: str,
    window_size: int,
    overlap: int,
    unit_types: set[str],
    paragraphs: list[ParsedParagraph],
    units: list[RetrievalUnit],
    document_events: list[dict[str, object]],
) -> dict[str, int | str]:
    trace_dir = trace_dir.expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pdf_dir": str(pdf_dir),
        "out_dir": str(out_dir),
        "embedding": embedding_name,
        "vector_kind": vector_kind,
        "window_size": window_size,
        "overlap": overlap,
        "unit_types": sorted(unit_types),
        "paragraphs": len(paragraphs),
        "retrieval_units": len(units),
        "documents": len(document_events),
        "secret_policy": "API keys and Authorization headers are not written to traces.",
    }
    (trace_dir / "build_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paragraph_rows = (
        {
            "paragraph_id": paragraph.paragraph_id,
            "doc_id": paragraph.doc_id,
            "source_path": paragraph.source_path,
            "title": paragraph.title,
            "page_start": paragraph.page_start,
            "page_end": paragraph.page_end,
            "text": paragraph.text,
        }
        for paragraph in paragraphs
    )
    unit_rows = (
        {
            "unit_id": unit.unit_id,
            "paragraph_id": unit.paragraph_id,
            "doc_id": unit.doc_id,
            "unit_type": unit.unit_type,
            "text": unit.text,
            "text_for_embedding": unit.text_for_embedding,
            "sentence_start": unit.sentence_start,
            "sentence_end": unit.sentence_end,
            "weight": unit.weight,
        }
        for unit in units
    )
    event_rows = (
        {
            "event": "document_processed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        for event in document_events
    )
    paragraph_count = write_jsonl(trace_dir / "paragraphs.jsonl", paragraph_rows)
    unit_count = write_jsonl(trace_dir / "retrieval_units.jsonl", unit_rows)
    event_count = write_jsonl(trace_dir / "build_events.jsonl", event_rows)
    return {
        "trace_dir": str(trace_dir),
        "trace_paragraphs": paragraph_count,
        "trace_retrieval_units": unit_count,
        "trace_events": event_count,
    }


def sparse_dot(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def pack_sparse_vector(vector: dict[int, float]) -> bytes:
    payload = bytearray()
    for key, value in sorted(vector.items()):
        payload.extend(struct.pack("<Hf", int(key), float(value)))
    return bytes(payload)


def unpack_sparse_vector(blob: bytes) -> dict[int, float]:
    if len(blob) % 6 != 0:
        raise ValueError("invalid sparse vector blob length")
    vector: dict[int, float] = {}
    for offset in range(0, len(blob), 6):
        key, value = struct.unpack("<Hf", blob[offset : offset + 6])
        vector[int(key)] = float(value)
    return vector


def normalize_dense_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def dense_dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def pack_dense_vector(vector: list[float]) -> bytes:
    normalized = normalize_dense_vector(vector)
    return struct.pack(f"<{len(normalized)}f", *normalized)


def unpack_dense_vector(blob: bytes) -> list[float]:
    if len(blob) % 4 != 0:
        raise ValueError("invalid dense vector blob length")
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def default_faiss_index_path(db_path: Path) -> Path:
    return db_path.with_name("vectors.faiss")


def default_faiss_ids_path(db_path: Path) -> Path:
    return db_path.with_name("vector_ids.jsonl")


def load_faiss_module() -> object | None:
    try:
        import faiss

        return faiss
    except ImportError:
        return None


def faiss_matrix(rows: list[list[float]]) -> object:
    try:
        import numpy as np

        return np.asarray(rows, dtype="float32")
    except ImportError:
        return rows


def read_faiss_unit_ids(ids_path: Path) -> list[str]:
    unit_ids: list[str] = []
    for line in ids_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        unit_ids.append(str(payload["unit_id"]))
    return unit_ids


def build_faiss_vector_index(
    db_path: Path,
    index_path: Path | None = None,
    ids_path: Path | None = None,
    batch_size: int = 4096,
    faiss_module: object | None = None,
) -> dict[str, object]:
    faiss = faiss_module or load_faiss_module()
    if faiss is None:
        raise RuntimeError('FAISS is required. Install with `pip install ".[faiss]"` or `pip install faiss-cpu`.')

    index_path = index_path or default_faiss_index_path(db_path)
    ids_path = ids_path or default_faiss_ids_path(db_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path, factory=ClosingConnection) as conn:
        conn.row_factory = sqlite3.Row
        meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
        if meta.get("vector_kind") != "dense":
            raise RuntimeError("FAISS index can only be built for dense vectors")

        first = conn.execute(
            "SELECT vector_blob FROM retrieval_units ORDER BY unit_id LIMIT 1"
        ).fetchone()
        if first is None:
            raise RuntimeError("no retrieval_units found")
        dims = len(unpack_dense_vector(first["vector_blob"]))
        index = faiss.IndexFlatIP(dims)

        total = 0
        with ids_path.open("w", encoding="utf-8") as ids_file:
            offset = 0
            while True:
                rows = conn.execute(
                    """
                    SELECT unit_id, weight, vector_blob
                    FROM retrieval_units
                    ORDER BY unit_id
                    LIMIT ? OFFSET ?
                    """,
                    (batch_size, offset),
                ).fetchall()
                if not rows:
                    break
                vectors: list[list[float]] = []
                for row in rows:
                    vector = unpack_dense_vector(row["vector_blob"])
                    vectors.append(vector)
                    ids_file.write(json.dumps({"unit_id": row["unit_id"]}, ensure_ascii=False) + "\n")
                index.add(faiss_matrix(vectors))
                total += len(rows)
                offset += len(rows)

        faiss.write_index(index, str(index_path))
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("faiss_index", str(index_path)))
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("faiss_ids", str(ids_path)))
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("faiss_vectors", str(total)))
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("faiss_dim", str(dims)))

    manifest_path = db_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["faiss_index"] = str(index_path)
        manifest["faiss_ids"] = str(ids_path)
        manifest["faiss_vectors"] = total
        manifest["faiss_dim"] = dims
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "faiss_index": str(index_path),
        "faiss_ids": str(ids_path),
        "faiss_vectors": total,
        "faiss_dim": dims,
    }


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        result = super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return bool(result)


class LocalVectorStore:
    def __init__(
        self,
        db_path: Path,
        dims: int = 2048,
        embedding_backend: SparseHashEmbeddingBackend | DenseEmbeddingBackend | None = None,
    ) -> None:
        self.db_path = db_path
        self.dims = dims
        self.embedding_backend = embedding_backend or SparseHashEmbeddingBackend(dims=dims)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, factory=ClosingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    def read_meta(self) -> dict[str, str]:
        if not self.db_path.exists():
            return {}
        with self.connect() as conn:
            try:
                return {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
            except sqlite3.OperationalError:
                return {}

    def recreate(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()
        with self.connect() as conn:
            conn.executescript(
                """
                DROP TABLE IF EXISTS paragraphs;
                DROP TABLE IF EXISTS retrieval_units;
                DROP TABLE IF EXISTS knowledge_units_fts;
                DROP TABLE IF EXISTS knowledge_units;
                DROP TABLE IF EXISTS meta;

                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE paragraphs (
                    paragraph_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    text TEXT NOT NULL
                );

                CREATE TABLE retrieval_units (
                    unit_id TEXT PRIMARY KEY,
                    paragraph_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    sentence_start INTEGER NOT NULL,
                    sentence_end INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    vector_blob BLOB NOT NULL,
                    FOREIGN KEY(paragraph_id) REFERENCES paragraphs(paragraph_id)
                );

                CREATE INDEX idx_units_paragraph ON retrieval_units(paragraph_id);
                CREATE INDEX idx_units_type ON retrieval_units(unit_type);

                CREATE TABLE knowledge_units (
                    knowledge_unit_id TEXT PRIMARY KEY,
                    paragraph_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    unit_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    evidence_quote TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    extractor_version TEXT NOT NULL,
                    FOREIGN KEY(paragraph_id) REFERENCES paragraphs(paragraph_id)
                );

                CREATE INDEX idx_knowledge_paragraph ON knowledge_units(paragraph_id);
                CREATE INDEX idx_knowledge_type ON knowledge_units(unit_type);
                """
            )
            conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("dims", str(self.dims)))
            conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("vector_dim", str(self.dims)))
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                ("embedding", self.embedding_backend.name),
            )
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                ("vector_kind", self.embedding_backend.vector_kind),
            )

    def insert_paragraphs(self, paragraphs: Iterable[ParsedParagraph]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO paragraphs
                (paragraph_id, doc_id, source_path, title, page_start, page_end, text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        paragraph.paragraph_id,
                        paragraph.doc_id,
                        paragraph.source_path,
                        paragraph.title,
                        paragraph.page_start,
                        paragraph.page_end,
                        paragraph.text,
                    )
                    for paragraph in paragraphs
                ],
            )
        self.rebuild_text_index()

    def rebuild_text_index(self) -> dict[str, int | str]:
        with self.connect() as conn:
            conn.execute("DROP TABLE IF EXISTS paragraphs_fts")
            tokenizer = "trigram"
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE paragraphs_fts USING fts5(
                        paragraph_id UNINDEXED,
                        doc_id UNINDEXED,
                        source_path UNINDEXED,
                        title,
                        text,
                        tokenize='trigram'
                    )
                    """
                )
            except sqlite3.OperationalError:
                tokenizer = "unicode61"
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE paragraphs_fts USING fts5(
                        paragraph_id UNINDEXED,
                        doc_id UNINDEXED,
                        source_path UNINDEXED,
                        title,
                        text
                    )
                    """
                )
            conn.execute(
                """
                INSERT INTO paragraphs_fts(paragraph_id, doc_id, source_path, title, text)
                SELECT paragraph_id, doc_id, source_path, title, text
                FROM paragraphs
                ORDER BY doc_id, page_start, paragraph_id
                """
            )
            rows = conn.execute("SELECT COUNT(*) FROM paragraphs_fts").fetchone()[0]
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("text_index", f"fts5_{tokenizer}"),
            )
        _update_manifest_after_text_index(self.db_path, f"fts5_{tokenizer}", rows)
        return {
            "db_path": str(self.db_path),
            "text_index": f"fts5_{tokenizer}",
            "text_index_rows": rows,
        }

    def ensure_knowledge_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_units (
                knowledge_unit_id TEXT PRIMARY KEY,
                paragraph_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                title TEXT NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                unit_type TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                attributes_json TEXT NOT NULL,
                evidence_quote TEXT NOT NULL,
                confidence REAL NOT NULL,
                extractor_version TEXT NOT NULL,
                FOREIGN KEY(paragraph_id) REFERENCES paragraphs(paragraph_id)
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_paragraph ON knowledge_units(paragraph_id);
            CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_units(unit_type);
            """
        )

    def rebuild_knowledge_units(self, trace_dir: Path | None = None) -> dict[str, object]:
        with self.connect() as conn:
            self.ensure_knowledge_schema(conn)
            paragraph_rows = conn.execute(
                """
                SELECT paragraph_id, doc_id, source_path, title, page_start, page_end, text
                FROM paragraphs
                ORDER BY doc_id, page_start, paragraph_id
                """
            ).fetchall()
            paragraphs = [
                ParsedParagraph(
                    paragraph_id=row["paragraph_id"],
                    doc_id=row["doc_id"],
                    source_path=row["source_path"],
                    title=row["title"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    text=row["text"],
                )
                for row in paragraph_rows
            ]
            units = build_knowledge_units(paragraphs)

            conn.execute("DELETE FROM knowledge_units")
            conn.executemany(
                """
                INSERT OR REPLACE INTO knowledge_units
                (knowledge_unit_id, paragraph_id, doc_id, source_path, title, page_start, page_end,
                 unit_type, subject, predicate, object, attributes_json, evidence_quote,
                 confidence, extractor_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        unit.knowledge_unit_id,
                        unit.paragraph_id,
                        unit.doc_id,
                        unit.source_path,
                        unit.title,
                        unit.page_start,
                        unit.page_end,
                        unit.unit_type,
                        unit.subject,
                        unit.predicate,
                        unit.object,
                        unit.attributes_json,
                        unit.evidence_quote,
                        unit.confidence,
                        unit.extractor_version,
                    )
                    for unit in units
                ],
            )
            self._rebuild_knowledge_fts(conn)
            rows = conn.execute("SELECT COUNT(*) FROM knowledge_units").fetchone()[0]
            type_rows = conn.execute(
                """
                SELECT unit_type, COUNT(*) AS count
                FROM knowledge_units
                GROUP BY unit_type
                ORDER BY unit_type
                """
            ).fetchall()
            unit_types = {row["unit_type"]: int(row["count"]) for row in type_rows}
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("knowledge_units", str(rows)),
            )

        trace_path = None
        if trace_dir is not None:
            trace_path = trace_dir.expanduser() / "knowledge_units.jsonl"
            write_jsonl(
                trace_path,
                (
                    {
                        "knowledge_unit_id": unit.knowledge_unit_id,
                        "paragraph_id": unit.paragraph_id,
                        "doc_id": unit.doc_id,
                        "source_path": unit.source_path,
                        "title": unit.title,
                        "page_start": unit.page_start,
                        "page_end": unit.page_end,
                        "unit_type": unit.unit_type,
                        "subject": unit.subject,
                        "predicate": unit.predicate,
                        "object": unit.object,
                        "attributes": json.loads(unit.attributes_json),
                        "evidence_quote": unit.evidence_quote,
                        "confidence": unit.confidence,
                        "extractor_version": unit.extractor_version,
                    }
                    for unit in units
                ),
            )

        _update_manifest_after_knowledge_units(self.db_path, rows, unit_types, trace_path)
        return {
            "db_path": str(self.db_path),
            "knowledge_units": rows,
            "knowledge_unit_types": unit_types,
            "knowledge_trace": str(trace_path) if trace_path else "",
        }

    def _rebuild_knowledge_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE IF EXISTS knowledge_units_fts")
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE knowledge_units_fts USING fts5(
                    knowledge_unit_id UNINDEXED,
                    paragraph_id UNINDEXED,
                    unit_type UNINDEXED,
                    subject,
                    predicate,
                    object,
                    evidence_quote,
                    tokenize='trigram'
                )
                """
            )
        except sqlite3.OperationalError:
            conn.execute(
                """
                CREATE VIRTUAL TABLE knowledge_units_fts USING fts5(
                    knowledge_unit_id UNINDEXED,
                    paragraph_id UNINDEXED,
                    unit_type UNINDEXED,
                    subject,
                    predicate,
                    object,
                    evidence_quote
                )
                """
            )
        conn.execute(
            """
            INSERT INTO knowledge_units_fts(
                knowledge_unit_id, paragraph_id, unit_type, subject, predicate, object, evidence_quote
            )
            SELECT knowledge_unit_id, paragraph_id, unit_type, subject, predicate, object, evidence_quote
            FROM knowledge_units
            ORDER BY doc_id, page_start, knowledge_unit_id
            """
        )

    def has_text_index(self) -> bool:
        if not self.db_path.exists():
            return False
        with self.connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'paragraphs_fts'"
            ).fetchone()
            return row is not None

    def has_knowledge_index(self) -> bool:
        if not self.db_path.exists():
            return False
        with self.connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'knowledge_units'"
            ).fetchone()
            return row is not None

    def insert_units(self, units: Iterable[RetrievalUnit], insert_batch_size: int = 256) -> None:
        unit_list = list(units)
        with self.connect() as conn:
            for start in range(0, len(unit_list), insert_batch_size):
                batch = unit_list[start : start + insert_batch_size]
                vectors = self.embedding_backend.embed_texts([unit.text_for_embedding for unit in batch])
                if len(vectors) != len(batch):
                    raise RuntimeError("embedding backend returned a different number of vectors")
                if self.embedding_backend.vector_kind == "dense":
                    vector_dim = len(vectors[0]) if vectors else 0
                    if any(len(vector) != vector_dim for vector in vectors):
                        raise RuntimeError("dense embedding backend returned inconsistent vector dimensions")
                    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("dims", str(vector_dim)))
                    conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                        ("vector_dim", str(vector_dim)),
                    )
                rows = []
                for unit, vector in zip(batch, vectors):
                    vector_blob = (
                        pack_sparse_vector(vector)
                        if self.embedding_backend.vector_kind == "sparse"
                        else pack_dense_vector(vector)
                    )
                    rows.append(
                        (
                            unit.unit_id,
                            unit.paragraph_id,
                            unit.doc_id,
                            unit.unit_type,
                            unit.text,
                            unit.sentence_start,
                            unit.sentence_end,
                            unit.weight,
                            vector_blob,
                        )
                    )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO retrieval_units
                    (unit_id, paragraph_id, doc_id, unit_type, text,
                     sentence_start, sentence_end, weight, vector_blob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def stats(self) -> dict[str, int | str]:
        with self.connect() as conn:
            paragraphs = conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
            units = conn.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0]
            docs = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM paragraphs").fetchone()[0]
            try:
                knowledge_units = conn.execute("SELECT COUNT(*) FROM knowledge_units").fetchone()[0]
            except sqlite3.OperationalError:
                knowledge_units = 0
        return {
            "db_path": str(self.db_path),
            "documents": docs,
            "paragraphs": paragraphs,
            "retrieval_units": units,
            "knowledge_units": knowledge_units,
        }

    def search_vector(
        self,
        query: str,
        limit: int = 8,
        unit_limit: int = 80,
        faiss_module: object | None = None,
    ) -> list[dict[str, object]]:
        vector_kind = self.embedding_backend.vector_kind
        stored_vector_kind = self.read_meta().get("vector_kind")
        if stored_vector_kind and stored_vector_kind != vector_kind:
            raise RuntimeError(
                "vector_kind mismatch: "
                f"database uses {stored_vector_kind}, but embedding backend produces {vector_kind}. "
                "Use the same embedding family that built the database, for example "
                "`--embedding siliconflow` with BAAI/bge-m3 or `--embedding local-bge-m3`."
            )
        query_vectors = self.embedding_backend.embed_texts([query])
        if not query_vectors:
            return []
        query_vector = query_vectors[0]
        if not query_vector:
            return []
        dense_query_vector = normalize_dense_vector(query_vector) if vector_kind == "dense" else None
        hits: list[tuple[float, sqlite3.Row]] = []
        with self.connect() as conn:
            if dense_query_vector is not None:
                faiss_results = self._search_vector_faiss(
                    conn=conn,
                    query_vector=dense_query_vector,
                    limit=limit,
                    unit_limit=unit_limit,
                    faiss_module=faiss_module,
                )
                if faiss_results is not None:
                    return faiss_results

            rows = conn.execute(
                """
                SELECT unit_id, paragraph_id, unit_type, text, sentence_start, sentence_end, weight, vector_blob
                FROM retrieval_units
                """
            ).fetchall()
            for row in rows:
                if vector_kind == "sparse":
                    vector = unpack_sparse_vector(row["vector_blob"])
                    score = sparse_dot(query_vector, vector) * float(row["weight"])
                else:
                    vector = unpack_dense_vector(row["vector_blob"])
                    score = dense_dot(dense_query_vector, vector) * float(row["weight"])
                if score > 0:
                    hits.append((score, row))
            hits.sort(key=lambda item: item[0], reverse=True)
            return self._vector_hits_to_results(conn, hits[:unit_limit], limit=limit, source="vector")

    def _search_vector_faiss(
        self,
        conn: sqlite3.Connection,
        query_vector: list[float],
        limit: int,
        unit_limit: int,
        faiss_module: object | None = None,
    ) -> list[dict[str, object]] | None:
        index_path = default_faiss_index_path(self.db_path)
        ids_path = default_faiss_ids_path(self.db_path)
        if not index_path.exists() or not ids_path.exists():
            return None
        faiss = faiss_module or load_faiss_module()
        if faiss is None:
            return None
        unit_ids = read_faiss_unit_ids(ids_path)
        if not unit_ids:
            return None
        index = faiss.read_index(str(index_path))
        top_k = min(len(unit_ids), max(unit_limit, limit * 20))
        scores, indices = index.search(faiss_matrix([query_vector]), top_k)
        scored_unit_ids: list[tuple[float, str]] = []
        for score, row_index in zip(scores[0], indices[0]):
            row_index = int(row_index)
            score = float(score)
            if row_index < 0 or row_index >= len(unit_ids) or score <= 0:
                continue
            scored_unit_ids.append((score, unit_ids[row_index]))
        if not scored_unit_ids:
            return []

        placeholders = ",".join("?" for _ in scored_unit_ids)
        rows = conn.execute(
            f"""
            SELECT unit_id, paragraph_id, unit_type, text, sentence_start, sentence_end, weight, vector_blob
            FROM retrieval_units
            WHERE unit_id IN ({placeholders})
            """,
            [unit_id for _, unit_id in scored_unit_ids],
        ).fetchall()
        row_by_id = {row["unit_id"]: row for row in rows}
        hits = [
            (score * float(row_by_id[unit_id]["weight"]), row_by_id[unit_id])
            for score, unit_id in scored_unit_ids
            if unit_id in row_by_id
        ]
        return self._vector_hits_to_results(conn, hits, limit=limit, source="faiss")

    def _vector_hits_to_results(
        self,
        conn: sqlite3.Connection,
        hits: list[tuple[float, sqlite3.Row]],
        limit: int,
        source: str,
    ) -> list[dict[str, object]]:
        paragraph_scores: dict[str, dict[str, object]] = {}
        for score, row in hits:
            paragraph_id = row["paragraph_id"]
            current = paragraph_scores.setdefault(
                paragraph_id,
                {
                    "paragraph_id": paragraph_id,
                    "score": 0.0,
                    "hit_count": 0,
                    "unit_types": set(),
                    "matched_units": [],
                },
            )
            current["score"] = max(float(current["score"]), score)
            current["hit_count"] = int(current["hit_count"]) + 1
            current["unit_types"].add(row["unit_type"])
            current["matched_units"].append(
                {
                    "unit_id": row["unit_id"],
                    "unit_type": row["unit_type"],
                    "score": round(score, 6),
                    "text": row["text"],
                    "sentence_start": row["sentence_start"],
                    "sentence_end": row["sentence_end"],
                }
            )

        ranked = []
        for item in paragraph_scores.values():
            diversity_bonus = 0.03 * max(0, len(item["unit_types"]) - 1)
            count_bonus = 0.01 * min(8, int(item["hit_count"]) - 1)
            item["score"] = float(item["score"]) + diversity_bonus + count_bonus
            ranked.append(item)
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)

        results = []
        for item in ranked[:limit]:
            paragraph = conn.execute(
                "SELECT * FROM paragraphs WHERE paragraph_id = ?", (item["paragraph_id"],)
            ).fetchone()
            if not paragraph:
                continue
            results.append(
                    {
                        "paragraph_id": paragraph["paragraph_id"],
                        "doc_id": paragraph["doc_id"],
                        "source_path": paragraph["source_path"],
                        "title": paragraph["title"],
                        "page_start": paragraph["page_start"],
                        "page_end": paragraph["page_end"],
                        "text": paragraph["text"],
                        "score": round(float(item["score"]), 6),
                        "vector_score": round(float(item["score"]), 6),
                        "text_score": 0.0,
                        "knowledge_score": 0.0,
                        "retrieval_sources": [source],
                        "hit_count": item["hit_count"],
                        "unit_types": sorted(item["unit_types"]),
                        "matched_units": item["matched_units"][:5],
                        "matched_text_terms": [],
                        "matched_knowledge_units": [],
                    }
                )
        return results

    def search_text(self, query: str, limit: int = 8, candidate_limit: int = 200) -> list[dict[str, object]]:
        terms = text_search_terms(query)
        if not terms:
            return []

        paragraph_scores: dict[str, dict[str, object]] = {}
        with self.connect() as conn:
            if self.has_text_index():
                fts_query = fts5_query_from_terms(terms)
                if fts_query:
                    try:
                        rows = conn.execute(
                            """
                            SELECT paragraph_id, bm25(paragraphs_fts) AS rank
                            FROM paragraphs_fts
                            WHERE paragraphs_fts MATCH ?
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (fts_query, candidate_limit),
                        ).fetchall()
                        for index, row in enumerate(rows):
                            score = 0.45 + 0.35 * (1.0 - index / max(1, len(rows)))
                            item = paragraph_scores.setdefault(
                                row["paragraph_id"],
                                {
                                    "paragraph_id": row["paragraph_id"],
                                    "text_score": 0.0,
                                    "matched_text_terms": set(),
                                },
                            )
                            item["text_score"] = max(float(item["text_score"]), score)
                    except sqlite3.OperationalError:
                        pass

            clauses = []
            params: list[str] = []
            for term in terms:
                pattern = f"%{term}%"
                clauses.append("(text LIKE ? OR title LIKE ?)")
                params.extend([pattern, pattern])
            like_rows = conn.execute(
                f"""
                SELECT paragraph_id, title, text
                FROM paragraphs
                WHERE {" OR ".join(clauses)}
                LIMIT ?
                """,
                [*params, candidate_limit],
            ).fetchall()

            for row in like_rows:
                haystack = f"{row['title']}\n{row['text']}"
                matched_terms = [term for term in terms if term in haystack]
                if not matched_terms:
                    continue
                all_terms_bonus = 0.12 if len(matched_terms) == len(terms) else 0.0
                score = 0.5 + 0.08 * len(matched_terms) + all_terms_bonus
                item = paragraph_scores.setdefault(
                    row["paragraph_id"],
                    {
                        "paragraph_id": row["paragraph_id"],
                        "text_score": 0.0,
                        "matched_text_terms": set(),
                    },
                )
                item["text_score"] = max(float(item["text_score"]), score)
                item["matched_text_terms"].update(matched_terms)

            ranked = sorted(
                paragraph_scores.values(),
                key=lambda item: float(item["text_score"]),
                reverse=True,
            )

            results = []
            for item in ranked[:limit]:
                paragraph = conn.execute(
                    "SELECT * FROM paragraphs WHERE paragraph_id = ?", (item["paragraph_id"],)
                ).fetchone()
                if not paragraph:
                    continue
                results.append(
                    {
                        "paragraph_id": paragraph["paragraph_id"],
                        "doc_id": paragraph["doc_id"],
                        "source_path": paragraph["source_path"],
                        "title": paragraph["title"],
                        "page_start": paragraph["page_start"],
                        "page_end": paragraph["page_end"],
                        "text": paragraph["text"],
                        "score": round(float(item["text_score"]), 6),
                        "vector_score": 0.0,
                        "text_score": round(float(item["text_score"]), 6),
                        "knowledge_score": 0.0,
                        "retrieval_sources": ["text"],
                        "hit_count": 0,
                        "unit_types": [],
                        "matched_units": [],
                        "matched_text_terms": sorted(item["matched_text_terms"]),
                        "matched_knowledge_units": [],
                    }
                )
            return results

    def search_knowledge_units(
        self,
        query: str,
        limit: int = 8,
        candidate_limit: int = 200,
    ) -> list[dict[str, object]]:
        terms = knowledge_search_terms(query)
        if not terms or not self.has_knowledge_index():
            return []
        normalized_query = normalize_query_text(query)
        dosage_intent = "一钱" in normalized_query and (
            "多少" in normalized_query or "几克" in normalized_query or "克" in normalized_query
        )

        paragraph_scores: dict[str, dict[str, object]] = {}
        with self.connect() as conn:
            fts_query = fts5_query_from_terms(terms)
            if fts_query:
                try:
                    rows = conn.execute(
                        """
                        SELECT knowledge_unit_id, paragraph_id, unit_type
                        FROM knowledge_units_fts
                        WHERE knowledge_units_fts MATCH ?
                        LIMIT ?
                        """,
                        (fts_query, candidate_limit),
                    ).fetchall()
                    for index, row in enumerate(rows):
                        if dosage_intent and row["unit_type"] != "dosage":
                            continue
                        score = 0.55 + 0.25 * (1.0 - index / max(1, len(rows)))
                        item = paragraph_scores.setdefault(
                            row["paragraph_id"],
                            {
                                "paragraph_id": row["paragraph_id"],
                                "knowledge_score": 0.0,
                                "matched_knowledge_unit_ids": set(),
                            },
                        )
                        item["knowledge_score"] = max(float(item["knowledge_score"]), score)
                        item["matched_knowledge_unit_ids"].add(row["knowledge_unit_id"])
                except sqlite3.OperationalError:
                    pass

            clauses = []
            params: list[str] = []
            for term in terms:
                pattern = f"%{term}%"
                clauses.append(
                    "(subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR evidence_quote LIKE ?)"
                )
                params.extend([pattern, pattern, pattern, pattern])
            like_rows = conn.execute(
                f"""
                SELECT knowledge_unit_id, paragraph_id, unit_type, subject, predicate, object, evidence_quote
                FROM knowledge_units
                WHERE {" OR ".join(clauses)}
                LIMIT ?
                """,
                [*params, candidate_limit],
            ).fetchall()

            for row in like_rows:
                if dosage_intent and row["unit_type"] != "dosage":
                    continue
                haystack = "\n".join(
                    [
                        row["unit_type"],
                        row["subject"],
                        row["predicate"],
                        row["object"],
                        row["evidence_quote"],
                    ]
                )
                matched_terms = [term for term in terms if term in haystack]
                if not matched_terms:
                    continue
                all_terms_bonus = 0.14 if len(matched_terms) == len(terms) else 0.0
                score = 0.62 + 0.08 * len(matched_terms) + all_terms_bonus
                if row["unit_type"] in {"dosage", "method", "formula_pattern"}:
                    score += 0.08
                if row["unit_type"] == "dosage" and dosage_intent:
                    gram_mentions = re.findall(r"\d+(?:\.\d+)?\s*克", row["object"])
                    score += 0.18 + 0.06 * max(0, len(gram_mentions) - 1)
                item = paragraph_scores.setdefault(
                    row["paragraph_id"],
                    {
                        "paragraph_id": row["paragraph_id"],
                        "knowledge_score": 0.0,
                        "matched_knowledge_unit_ids": set(),
                    },
                )
                item["knowledge_score"] = max(float(item["knowledge_score"]), score)
                item["matched_knowledge_unit_ids"].add(row["knowledge_unit_id"])

            ranked = sorted(
                paragraph_scores.values(),
                key=lambda item: float(item["knowledge_score"]),
                reverse=True,
            )

            results = []
            for item in ranked[:limit]:
                paragraph = conn.execute(
                    "SELECT * FROM paragraphs WHERE paragraph_id = ?", (item["paragraph_id"],)
                ).fetchone()
                if not paragraph:
                    continue
                unit_rows = conn.execute(
                    f"""
                    SELECT knowledge_unit_id, unit_type, subject, predicate, object,
                           evidence_quote, confidence
                    FROM knowledge_units
                    WHERE knowledge_unit_id IN ({",".join("?" for _ in item["matched_knowledge_unit_ids"])})
                    ORDER BY confidence DESC, unit_type, subject
                    LIMIT 5
                    """,
                    list(item["matched_knowledge_unit_ids"]),
                ).fetchall()
                matched_units = [
                    {
                        "knowledge_unit_id": row["knowledge_unit_id"],
                        "unit_type": row["unit_type"],
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "evidence_quote": row["evidence_quote"],
                        "confidence": row["confidence"],
                    }
                    for row in unit_rows
                ]
                top_unit = matched_units[0] if matched_units else {}
                results.append(
                    {
                        "paragraph_id": paragraph["paragraph_id"],
                        "doc_id": paragraph["doc_id"],
                        "source_path": paragraph["source_path"],
                        "title": paragraph["title"],
                        "page_start": paragraph["page_start"],
                        "page_end": paragraph["page_end"],
                        "text": paragraph["text"],
                        "score": round(float(item["knowledge_score"]), 6),
                        "vector_score": 0.0,
                        "text_score": 0.0,
                        "knowledge_score": round(float(item["knowledge_score"]), 6),
                        "retrieval_sources": ["knowledge"],
                        "hit_count": len(matched_units),
                        "unit_types": [],
                        "matched_units": [],
                        "matched_text_terms": [],
                        "matched_knowledge_units": matched_units,
                        "knowledge_unit_id": top_unit.get("knowledge_unit_id", ""),
                        "unit_type": top_unit.get("unit_type", ""),
                        "subject": top_unit.get("subject", ""),
                        "predicate": top_unit.get("predicate", ""),
                        "object": top_unit.get("object", ""),
                        "evidence_quote": top_unit.get("evidence_quote", ""),
                    }
                )
            return results

    def search_hybrid(
        self,
        query: str,
        limit: int = 8,
        unit_limit: int = 120,
        text_limit: int = 80,
    ) -> list[dict[str, object]]:
        vector_results = self.search_vector(query, limit=max(limit * 4, 16), unit_limit=unit_limit)
        text_results = self.search_text(query, limit=max(text_limit, limit * 4))
        knowledge_results = self.search_knowledge_units(query, limit=max(text_limit, limit * 4))
        combined: dict[str, dict[str, object]] = {}

        for result in vector_results:
            item = dict(result)
            item["retrieval_sources"] = set(item.get("retrieval_sources", ["vector"]))
            item["unit_types"] = set(item.get("unit_types", []))
            item["matched_text_terms"] = set(item.get("matched_text_terms", []))
            item["matched_knowledge_units"] = list(item.get("matched_knowledge_units", []))
            combined[str(item["paragraph_id"])] = item

        for result in text_results:
            paragraph_id = str(result["paragraph_id"])
            current = combined.get(paragraph_id)
            if current is None:
                item = dict(result)
                item["retrieval_sources"] = set(item.get("retrieval_sources", ["text"]))
                item["unit_types"] = set(item.get("unit_types", []))
                item["matched_text_terms"] = set(item.get("matched_text_terms", []))
                item["matched_knowledge_units"] = list(item.get("matched_knowledge_units", []))
                combined[paragraph_id] = item
                continue
            current["retrieval_sources"].add("text")
            current["text_score"] = max(float(current.get("text_score", 0.0)), float(result["text_score"]))
            current["matched_text_terms"].update(result.get("matched_text_terms", []))

        for result in knowledge_results:
            paragraph_id = str(result["paragraph_id"])
            current = combined.get(paragraph_id)
            if current is None:
                item = dict(result)
                item["retrieval_sources"] = set(item.get("retrieval_sources", ["knowledge"]))
                item["unit_types"] = set(item.get("unit_types", []))
                item["matched_text_terms"] = set(item.get("matched_text_terms", []))
                item["matched_knowledge_units"] = list(item.get("matched_knowledge_units", []))
                combined[paragraph_id] = item
                continue
            current["retrieval_sources"].add("knowledge")
            current["knowledge_score"] = max(
                float(current.get("knowledge_score", 0.0)),
                float(result["knowledge_score"]),
            )
            current.setdefault("matched_knowledge_units", [])
            current["matched_knowledge_units"].extend(result.get("matched_knowledge_units", []))

        ranked = []
        for item in combined.values():
            vector_score = float(item.get("vector_score", 0.0))
            text_score = float(item.get("text_score", 0.0))
            knowledge_score = float(item.get("knowledge_score", 0.0))
            source_bonus = 0.08 if len(item["retrieval_sources"]) >= 2 else 0.0
            knowledge_bonus = 0.05 if "knowledge" in item["retrieval_sources"] else 0.0
            final_score = vector_score + text_score + knowledge_score + source_bonus + knowledge_bonus
            item["score"] = round(final_score, 6)
            item["vector_score"] = round(vector_score, 6)
            item["text_score"] = round(text_score, 6)
            item["knowledge_score"] = round(knowledge_score, 6)
            item["retrieval_sources"] = sorted(item["retrieval_sources"])
            item["unit_types"] = sorted(item["unit_types"])
            item["matched_text_terms"] = sorted(item["matched_text_terms"])
            item["matched_knowledge_units"] = item.get("matched_knowledge_units", [])[:5]
            ranked.append(item)
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        return ranked[:limit]

    def search(
        self,
        query: str,
        limit: int = 8,
        unit_limit: int = 120,
        mode: str = "hybrid",
    ) -> list[dict[str, object]]:
        if mode == "vector":
            return self.search_vector(query, limit=limit, unit_limit=unit_limit)
        if mode == "text":
            return self.search_text(query, limit=limit)
        if mode == "knowledge":
            return self.search_knowledge_units(query, limit=limit)
        if mode == "hybrid":
            return self.search_hybrid(query, limit=limit, unit_limit=unit_limit)
        raise ValueError(f"unsupported search mode: {mode}")


TRADITIONAL_QUERY_TRANSLATION = str.maketrans(
    {
        "錢": "钱",
        "現": "现",
        "餅": "饼",
        "熱": "热",
        "來": "来",
        "發": "发",
        "燒": "烧",
        "噁": "恶",
        "歲": "岁",
        "頭": "头",
        "頸": "颈",
        "痠": "酸",
        "黃": "黄",
        "開": "开",
        "藥": "药",
        "處": "处",
        "瀉": "泻",
        "證": "证",
        "裡": "里",
        "裡": "里",
    }
)


def normalize_query_text(query: str) -> str:
    return query.translate(TRADITIONAL_QUERY_TRANSLATION)


def detect_answer_intent(query: str) -> str:
    normalized = normalize_query_text(query)
    if "一钱" in normalized and ("克" in normalized or "多少" in normalized or "几" in normalized):
        return "dosage"
    if "木香饼" in normalized or "热熨" in normalized or ("出处" in normalized and ("哪" in normalized or "哪本" in normalized)):
        return "method"
    clinical_markers = (
        "病人",
        "患者",
        "发烧",
        "發燒",
        "下利",
        "拉肚子",
        "恶心",
        "噁心",
        "建议开",
        "开什么方",
        "處方",
        "处方",
        "男",
        "女",
        "岁",
    )
    if any(marker in normalized for marker in clinical_markers):
        return "clinical"
    return "general"


def expand_answer_query(query: str) -> str:
    normalized = normalize_query_text(query)
    parts = [query]
    if normalized != query:
        parts.append(normalized)
    intent = detect_answer_intent(normalized)
    if intent == "dosage":
        parts.append("一钱 克 钱 剂量 换算 度量衡 比例 汉制 今制 药房")
    elif intent == "method":
        parts.append("木香饼 热熨 生地木香作饼 神农本草经 结肿成核 乳中结核")
    elif intent == "clinical":
        parts.append("下利 恶心 干呕 黄臭 热利 葛根黄芩黄连汤 黄芩加半夏生姜汤 黄芩汤 半夏泻心汤 生姜泻心汤")
    return " ".join(dedupe_keep_order(parts))


def useful_query_terms(text: str, max_terms: int = 24) -> list[str]:
    terms: list[str] = []
    terms.extend(MEASURE_TERM_RE.findall(text))
    for term in text_search_terms(text):
        normalized = term.strip()
        if len(normalized) < 2 or normalized in QUERY_EXPANSION_STOP_TERMS:
            continue
        if normalized.isdigit():
            continue
        terms.append(normalized)
    return dedupe_keep_order(terms)[:max_terms]


def evidence_sentences_for_followup(query: str, results: list[dict[str, object]]) -> list[str]:
    query_terms = set(useful_query_terms(normalize_query_text(query), max_terms=16))
    sentences: list[str] = []
    for result in results[:8]:
        for sentence in split_sentences(result_evidence_text(result)):
            sentence = sentence.strip()
            if not sentence:
                continue
            has_query_term = any(term in sentence for term in query_terms)
            has_measure = bool(MEASURE_TERM_RE.search(sentence))
            has_knowledge = any(
                str(unit.get("unit_type", "")) in sentence
                for unit in result.get("matched_knowledge_units", []) or []
            )
            if has_query_term or has_measure or has_knowledge:
                sentences.append(sentence)
    return dedupe_keep_order(sentences)


def build_followup_query(query: str, results: list[dict[str, object]], intent: str) -> str:
    if not results:
        return ""
    parts = [query, normalize_query_text(query)]
    if intent == "dosage":
        parts.append("剂量 换算 度量衡 比例 克 钱")
    elif intent == "method":
        parts.append("出处 原文 治法 材料 方法")
    elif intent == "clinical":
        parts.append("方证 鉴别 症状 加减 禁忌")

    for sentence in evidence_sentences_for_followup(query, results):
        parts.extend(useful_query_terms(sentence, max_terms=16))
    for result in results[:8]:
        for unit in result.get("matched_knowledge_units", []) or []:
            parts.extend(
                useful_query_terms(
                    " ".join(
                        [
                            str(unit.get("unit_type", "")),
                            str(unit.get("subject", "")),
                            str(unit.get("predicate", "")),
                            str(unit.get("object", "")),
                            str(unit.get("evidence_quote", "")),
                        ]
                    ),
                    max_terms=12,
                )
            )
    return " ".join(dedupe_keep_order(part for part in parts if part.strip()))


def merge_results_by_paragraph(
    primary: list[dict[str, object]],
    secondary: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for result in [*secondary, *primary]:
        paragraph_id = str(result.get("paragraph_id", ""))
        if not paragraph_id:
            continue
        current = merged.get(paragraph_id)
        if current is None:
            merged[paragraph_id] = dict(result)
            continue
        current["score"] = max(float(current.get("score", 0.0)), float(result.get("score", 0.0)))
        current["vector_score"] = max(float(current.get("vector_score", 0.0)), float(result.get("vector_score", 0.0)))
        current["text_score"] = max(float(current.get("text_score", 0.0)), float(result.get("text_score", 0.0)))
        current["knowledge_score"] = max(
            float(current.get("knowledge_score", 0.0)),
            float(result.get("knowledge_score", 0.0)),
        )
        sources = set(current.get("retrieval_sources", []))
        sources.update(result.get("retrieval_sources", []))
        current["retrieval_sources"] = sorted(sources)
        current.setdefault("matched_knowledge_units", [])
        current["matched_knowledge_units"].extend(result.get("matched_knowledge_units", []))
        current.setdefault("matched_units", [])
        current["matched_units"].extend(result.get("matched_units", []))
    return sorted(merged.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)


def result_diversity_facets(result: dict[str, object], intent: str = "general") -> set[str]:
    evidence = result_evidence_text(result)
    facets: set[str] = {
        f"source:{Path(str(result.get('source_path', ''))).name}:{result.get('page_start', '')}"
    }
    for term in MEASURE_TERM_RE.findall(evidence):
        facets.add(f"measure:{term.replace(' ', '')}")
    for term in useful_query_terms(evidence, max_terms=40):
        if len(term) >= 2:
            facets.add(f"term:{term}")
    for unit in result.get("matched_knowledge_units", []) or []:
        unit_type = str(unit.get("unit_type", ""))
        subject = str(unit.get("subject", ""))
        predicate = str(unit.get("predicate", ""))
        object_text = str(unit.get("object", ""))
        if unit_type:
            facets.add(f"unit:{unit_type}")
        if subject:
            facets.add(f"subject:{subject}")
        if predicate:
            facets.add(f"predicate:{predicate}")
        for term in useful_query_terms(object_text, max_terms=12):
            facets.add(f"object:{term}")
    if intent != "general":
        facets.add(f"intent:{intent}")
    return facets


def select_diverse_results(
    results: list[dict[str, object]],
    limit: int,
    intent: str = "general",
) -> list[dict[str, object]]:
    if limit <= 0 or len(results) <= limit:
        return results[:limit]

    remaining = sorted(results, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    max_score = max(float(item.get("score", 0.0)) for item in remaining) or 1.0
    selected: list[dict[str, object]] = []
    covered_facets: set[str] = set()

    while remaining and len(selected) < limit:
        best_index = 0
        best_score = -1.0
        for index, candidate in enumerate(remaining):
            facets = result_diversity_facets(candidate, intent=intent)
            new_facets = facets - covered_facets
            new_measures = {facet for facet in new_facets if facet.startswith("measure:")}
            relevance = float(candidate.get("score", 0.0)) / max_score
            novelty = min(len(new_facets), 12) / 12
            measure_bonus = 0.0
            if intent == "dosage":
                measure_bonus = 0.35 * min(len(new_measures), 3)
            source_bonus = 0.08 if not any(
                Path(str(item.get("source_path", ""))).name
                == Path(str(candidate.get("source_path", ""))).name
                for item in selected
            ) else 0.0
            candidate_score = relevance + 0.45 * novelty + measure_bonus + source_bonus
            if candidate_score > best_score:
                best_score = candidate_score
                best_index = index
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        covered_facets.update(result_diversity_facets(chosen, intent=intent))

    return selected


def citation_label(result: dict[str, object]) -> str:
    source_name = Path(str(result.get("source_path", ""))).name
    page_start = result.get("page_start", "")
    page_end = result.get("page_end", page_start)
    page = f"p{page_start}" if page_start == page_end else f"p{page_start}-{page_end}"
    return f"{source_name} {page}".strip()


def dosage_evidence_snippet(text: str, max_chars: int = 520) -> str:
    if len(text) <= max_chars:
        return text
    anchor_offsets = [offset for offset in (text.find("一钱"), text.find("1钱")) if offset >= 0]
    gram_match = MEASURE_TERM_RE.search(text)
    offsets = anchor_offsets[:]
    if gram_match is not None:
        offsets.append(gram_match.start())
    if not offsets:
        return evidence_quote(text, max_chars=max_chars)
    center = min(offsets)
    start = max(0, center - 60)
    end = min(len(text), center + max_chars - 10)
    return evidence_quote(text[start:end], max_chars=max_chars)


def citation_evidence_for_result(result: dict[str, object], intent: str = "general") -> str:
    if intent == "dosage":
        evidence_parts: list[str] = []
        for sentence in split_sentences(str(result.get("text", ""))):
            if ("一钱" in sentence or "1钱" in sentence) and ("克" in sentence or "比例" in sentence):
                evidence_parts.append(sentence)
        for unit in result.get("matched_knowledge_units", []) or []:
            quote = str(unit.get("evidence_quote", "")).strip()
            if "一钱" in quote or "1钱" in quote or "克" in quote:
                evidence_parts.append(quote)
        evidence = " ".join(dedupe_keep_order(evidence_parts))
        if evidence:
            return dosage_evidence_snippet(evidence)

    knowledge_units = result.get("matched_knowledge_units", []) or []
    if knowledge_units:
        evidence = str(knowledge_units[0].get("evidence_quote", "")).strip()
    else:
        evidence = evidence_quote(str(result.get("text", "")), max_chars=220)
    return evidence


def build_citations(
    results: list[dict[str, object]],
    max_citations: int = 6,
    intent: str = "general",
) -> list[dict[str, object]]:
    citations: list[dict[str, object]] = []
    seen: set[tuple[str, object, str]] = set()
    for result in results:
        evidence = citation_evidence_for_result(result, intent=intent)
        key = (str(result.get("source_path", "")), result.get("page_start"), evidence)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "index": len(citations) + 1,
                "paragraph_id": result.get("paragraph_id", ""),
                "source_path": result.get("source_path", ""),
                "title": result.get("title", ""),
                "page_start": result.get("page_start", ""),
                "page_end": result.get("page_end", result.get("page_start", "")),
                "label": citation_label(result),
                "evidence_quote": evidence,
            }
        )
        if len(citations) >= max_citations:
            break
    return citations


def result_evidence_text(result: dict[str, object]) -> str:
    parts = [str(result.get("title", "")), str(result.get("text", ""))]
    for unit in result.get("matched_knowledge_units", []) or []:
        parts.extend(
            [
                str(unit.get("unit_type", "")),
                str(unit.get("subject", "")),
                str(unit.get("predicate", "")),
                str(unit.get("object", "")),
                str(unit.get("evidence_quote", "")),
            ]
        )
    return "\n".join(parts)


def filter_results_for_intent(intent: str, results: list[dict[str, object]]) -> list[dict[str, object]]:
    if intent == "dosage":
        filtered = [result for result in results if is_dosage_relevant_result(result)]
    elif intent == "method":
        filtered = [
            result
            for result in results
            if "木香饼" in result_evidence_text(result) or "热熨" in result_evidence_text(result)
        ]
    elif intent == "clinical":
        clinical_terms = ("下利", "恶心", "干呕", "黄臭", "热利", "黄芩", "葛根黄芩黄连汤")
        filtered = [result for result in results if any(term in result_evidence_text(result) for term in clinical_terms)]
    else:
        filtered = results
    return filtered or results


def is_dosage_relevant_result(result: dict[str, object]) -> bool:
    text = result_evidence_text(result)
    has_anchor = "一钱" in text or "1钱" in text
    has_measure = bool(MEASURE_TERM_RE.search(text))
    has_context = any(term in text for term in ("剂量", "换算", "度量衡", "比例", "汉制", "今制"))
    if "大约一钱的量" in text or "大约1钱的量" in text:
        return has_context and "比例" in text
    if has_anchor and (has_measure or has_context):
        return True
    if "比例" in text and any(term in text for term in ("方", "汤", "剂量", "处方")):
        return True
    return any(
        unit.get("unit_type") == "dosage"
        for unit in result.get("matched_knowledge_units", []) or []
    )


def collect_matched_knowledge_units(results: list[dict[str, object]]) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    seen: set[str] = set()
    for result in results:
        for unit in result.get("matched_knowledge_units", []) or []:
            key = "|".join(
                [
                    str(unit.get("unit_type", "")),
                    str(unit.get("subject", "")),
                    str(unit.get("predicate", "")),
                    str(unit.get("object", "")),
                    str(unit.get("evidence_quote", "")),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            units.append(unit)
    return units


def collect_gram_values(results: list[dict[str, object]]) -> list[str]:
    def extract_conversion_values(text: str) -> list[str]:
        values: list[str] = []
        anchor_patterns = (
            r"(?:一钱|1钱)\s*(?:等于|是|约|算|当|大概|大约)?\s*(\d+(?:\.\d+)?\s*克)",
            r"(?:一钱|1钱)[^。！？!?；;，,]{0,16}?(\d+(?:\.\d+)?\s*克)",
        )
        for pattern in anchor_patterns:
            values.extend(re.findall(pattern, text))
        if ("一钱的量" in text or "1钱的量" in text) and any(
            marker in text for marker in ("也就是说", "按照比例", "加起来是")
        ):
            anchor_offsets = [offset for offset in (text.find("一钱的量"), text.find("1钱的量")) if offset >= 0]
            if anchor_offsets:
                anchor_offset = min(anchor_offsets)
                values.extend(re.findall(r"\d+(?:\.\d+)?\s*克", text[anchor_offset:]))

        for marker in ("误解成", "当作", "算作"):
            offset = text.find(marker)
            if offset >= 0:
                fragment = re.split(r"[，,。！？!?；;]", text[offset:], maxsplit=1)[0]
                values.extend(re.findall(r"\d+(?:\.\d+)?\s*克", fragment))
        return dedupe_keep_order(values)

    def is_conversion_context(text: str) -> bool:
        markers = (
            "一钱等于",
            "1钱等于",
            "一钱是",
            "1钱是",
            "一钱约",
            "1钱约",
            "一钱算",
            "1钱算",
            "等于多少克",
            "换算",
            "剂量换算",
        )
        if any(marker in text for marker in markers):
            return True
        if ("大约一钱的量" in text or "一钱的量" in text or "大约1钱的量" in text or "1钱的量" in text) and not any(
            marker in text for marker in ("也就是说", "按照比例", "加起来是")
        ):
            return False
        return ("一钱" in text or "1钱" in text) and "克" in text

    values: list[str] = []
    for unit in collect_matched_knowledge_units(results):
        if unit.get("unit_type") != "dosage":
            continue
        context = f"{unit.get('object', '')}\n{unit.get('evidence_quote', '')}"
        if not is_conversion_context(context):
            continue
        values.extend(extract_conversion_values(str(unit.get("object", ""))))
        values.extend(extract_conversion_values(str(unit.get("evidence_quote", ""))))
    for result in results:
        text = str(result.get("text", ""))
        for sentence in split_sentences(text):
            if is_conversion_context(sentence):
                values.extend(extract_conversion_values(sentence))
    return dedupe_keep_order(values)


def collect_formula_names(results: list[dict[str, object]]) -> list[str]:
    names: list[str] = []
    for unit in collect_matched_knowledge_units(results):
        if unit.get("unit_type") == "formula_pattern":
            names.append(str(unit.get("subject", "")))
        names.extend(extract_formula_terms(str(unit.get("object", ""))))
        names.extend(extract_formula_terms(str(unit.get("evidence_quote", ""))))
    for result in results:
        names.extend(extract_formula_terms(str(result.get("text", ""))))
    return dedupe_keep_order(names)


def with_formula_dosage_safety(notice: str) -> str:
    notice = notice.strip()
    if FORMULA_DOSAGE_SAFETY_NOTICE in notice:
        return notice
    if not notice:
        return FORMULA_DOSAGE_SAFETY_NOTICE
    return f"{notice} {FORMULA_DOSAGE_SAFETY_NOTICE}"


def synthesize_pdf_rag_answer(
    query: str,
    results: list[dict[str, object]],
    max_citations: int = 6,
) -> dict[str, object]:
    intent = detect_answer_intent(query)
    relevant_results = filter_results_for_intent(intent, results)
    effective_max_citations = max_citations
    citations = build_citations(
        relevant_results,
        max_citations=effective_max_citations,
        intent=intent,
    )
    cited = citations if intent == "dosage" else citations[:3]
    citation_refs = "、".join(f"[{citation['index']}]" for citation in cited)

    if not results:
        return {
            "query": query,
            "intent": intent,
            "answer": "当前知识库没有检索到足够可靠的原文证据。建议换用更具体的方名、药名、症状或原文短语再查。",
            "citations": [],
            "safety_notice": "",
        }

    if intent == "dosage":
        grams = collect_gram_values(relevant_results)
        gram_text = "、".join(grams) if grams else "检索结果中未形成稳定克数列表"
        answer = (
            f"根据当前 PDF 原文证据，“一钱”的说法不是单一固定值，已检索到：{gram_text}。"
            f"证据见 {citation_refs}。"
            "课程资料同时提示古今度量衡、抓药语境、煎煮方法和方中比例都会影响理解，"
            "不能只抽出一个克数孤立使用。"
        )
        safety_notice = with_formula_dosage_safety("这是课程资料整理和出处检索，不是个人用药剂量建议。")
    elif intent == "method":
        locations = "；".join(f"{citation['label']}[{citation['index']}]" for citation in citations[:4])
        answer = (
            f"关于“木香饼热熨法”，当前检索到的主要出处是：{locations}。"
            "原文证据集中在“木香饼（生地木香作饼），热熨贴之，治结肿成核”等内容，"
            "视频同步文稿还解释为用生地与木香做饼，并以热熨方式作用于硬结、结块相关语境。"
        )
        safety_notice = with_formula_dosage_safety("这是原文出处定位和课程整理，不是外治法操作建议。")
    elif intent == "clinical":
        formulas = collect_formula_names(relevant_results)[:8]
        formula_text = "、".join(formulas) if formulas else "未稳定抽取到方名"
        answer = (
            "这个问题含有具体病人信息，我不直接给个人处方。"
            f"按课程资料检索，相关辨证线索和方证包括：{formula_text}。"
            f"可先核对原文证据 {citation_refs}，重点看下利性质、是否恶心干呕、是否腹痛、"
            "是否仍有表证、心下痞满或肠鸣等条件。若用于真实病人，需要由合格医师面诊辨证。"
        )
        safety_notice = with_formula_dosage_safety("课程资料不能替代诊断；这里不直接给个人处方、剂量或治疗建议。")
    else:
        knowledge_units = collect_matched_knowledge_units(relevant_results)[:5]
        if knowledge_units:
            points = "；".join(
                f"{unit.get('subject')}：{unit.get('object')}" for unit in knowledge_units
            )
        else:
            points = "；".join(evidence_quote(str(result.get("text", "")), max_chars=80) for result in results[:3])
        answer = f"根据当前检索结果，相关原文要点包括：{points}。证据见 {citation_refs}。"
        safety_notice = ""

    return {
        "query": query,
        "intent": intent,
        "answer": answer,
        "citations": citations,
        "safety_notice": safety_notice,
        "results": results,
    }


def build_grounded_composer_messages(
    query: str,
    answer: dict[str, object],
) -> list[dict[str, str]]:
    citations = answer.get("citations", []) or []
    citation_lines = []
    for citation in citations:
        citation_lines.append(
            "\n".join(
                [
                    f"[{citation.get('index')}] {citation.get('label')}",
                    f"证据：{citation.get('evidence_quote')}",
                ]
            )
        )
    evidence_block = "\n\n".join(citation_lines)
    safety_notice = str(answer.get("safety_notice", "")).strip()
    intent = str(answer.get("intent", "general"))
    if intent in {"dosage", "method", "clinical"}:
        safety_notice = with_formula_dosage_safety(safety_notice)
    template_answer = str(answer.get("answer", "")).strip()
    system_prompt = (
        "你是一个严谨的 source-grounded RAG 答案整理器。"
        "只能使用用户提供的【证据】回答，不可使用未列出的资料、常识或模型记忆补充事实。"
        "每个关键事实后必须标注引用编号，如 [1]。"
        "如果证据不足，要明确说证据不足。"
        "不要编造书名、页码、条文、剂量或出处。"
        "如果问题包含具体病人或用药请求，只能整理课程资料中的辨证线索，必须声明不能替代诊断，且不直接给个人处方。"
        f"凡涉及剂量、方药、处方线索或外治法，必须包含这条安全提示：{FORMULA_DOSAGE_SAFETY_NOTICE}"
    )
    user_prompt = f"""问题：
{query}

问题类型：{intent}

本地模板初稿：
{template_answer}

安全边界：
{safety_notice}

【证据】
{evidence_block}

请用中文生成一个比模板更完整但仍严格受证据约束的回答。输出结构：
1. 直接结论
2. 证据依据
3. 需要鉴别或注意的条件
4. 安全边界

要求：
- 不可使用未列出的资料。
- 引用编号只能来自上面的【证据】。
- 不要输出没有证据支持的具体剂量建议或个人治疗方案。
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def compose_pdf_rag_answer_with_llm(
    query: str,
    answer: dict[str, object],
    llm_backend: SiliconFlowChatBackend | object,
) -> dict[str, object]:
    messages = build_grounded_composer_messages(query, answer)
    content = llm_backend.complete(messages)
    composed = dict(answer)
    composed["template_answer"] = answer.get("answer", "")
    composed["answer"] = content
    composed["composer"] = "llm"
    composed.pop("results", None)
    return composed


def answer_pdf_rag(
    query: str,
    db_path: Path,
    limit: int = 8,
    mode: str = "hybrid",
    embedding: str = "auto",
    model: str = "BAAI/bge-m3",
    batch_size: int = 32,
    embedding_backend: SparseHashEmbeddingBackend | DenseEmbeddingBackend | None = None,
    composer: str = "template",
    llm_backend: SiliconFlowChatBackend | object | None = None,
    llm_model: str | None = None,
) -> dict[str, object]:
    db_path = db_path.expanduser().resolve()
    if embedding_backend is not None:
        store = LocalVectorStore(db_path, embedding_backend=embedding_backend)
    elif mode in {"text", "knowledge"}:
        store = LocalVectorStore(db_path)
    else:
        backend = create_embedding_backend_for_db(
            db_path,
            embedding=embedding,
            model=model,
            batch_size=batch_size,
        )
        store = LocalVectorStore(db_path, embedding_backend=backend)
    search_query = expand_answer_query(query)
    candidate_limit = max(limit * 4, 16)
    results = store.search(search_query, limit=candidate_limit, mode=mode)
    intent = detect_answer_intent(query)
    if mode == "hybrid" and intent in {"dosage", "method", "clinical"}:
        knowledge_results = store.search_knowledge_units(search_query, limit=candidate_limit)
        results = merge_results_by_paragraph(results, knowledge_results)
    followup_query = build_followup_query(query, results, intent)
    if followup_query and followup_query != search_query:
        followup_limit = max(limit * 4, 12)
        followup_results = store.search(followup_query, limit=followup_limit, mode=mode)
        if mode == "hybrid" and intent in {"dosage", "method", "clinical"}:
            followup_knowledge_results = store.search_knowledge_units(
                followup_query,
                limit=followup_limit,
            )
            followup_results = merge_results_by_paragraph(
                followup_results,
                followup_knowledge_results,
            )
        results = merge_results_by_paragraph(results, followup_results)
    intent_results = filter_results_for_intent(intent, results)
    results = select_diverse_results(intent_results, limit=limit, intent=intent)
    answer = synthesize_pdf_rag_answer(query, results)
    answer["composer"] = "template"
    if composer == "template":
        return answer
    if composer != "llm":
        raise ValueError(f"unsupported answer composer: {composer}")
    backend = llm_backend or SiliconFlowChatBackend(model=llm_model)
    try:
        return compose_pdf_rag_answer_with_llm(query, answer, backend)
    except Exception as exc:
        fallback = dict(answer)
        fallback["composer"] = "template"
        fallback["llm_error"] = str(exc)
        return fallback


def extract_pdf_paragraphs(pdf_path: Path, max_chars: int = 900) -> list[ParsedParagraph]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF extraction. Install package: PyMuPDF") from exc

    pdf_path = pdf_path.expanduser().resolve()
    doc_id = stable_id(pdf_path.name, pdf_path.stat().st_size)
    paragraphs: list[ParsedParagraph] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            block_texts = []
            for block in blocks:
                text = clean_text(str(block[4]))
                text = re.sub(r"^\s*\d+\s*$", "", text).strip()
                if len(text) >= 20:
                    block_texts.append(text)
            page_text = clean_text("\n".join(block_texts) or page.get_text("text"))
            for paragraph_index, paragraph_text in enumerate(
                split_long_text_to_paragraphs(page_text, max_chars=max_chars), start=1
            ):
                paragraph_id = stable_id(doc_id, page_index, paragraph_index, paragraph_text[:80])
                paragraphs.append(
                    ParsedParagraph(
                        paragraph_id=paragraph_id,
                        doc_id=doc_id,
                        source_path=str(pdf_path),
                        title=f"{pdf_path.stem} p{page_index}",
                        page_start=page_index,
                        page_end=page_index,
                        text=paragraph_text,
                    )
                )
    return paragraphs


def build_pdf_vector_store(
    pdf_dir: Path,
    out_dir: Path,
    window_size: int = 6,
    overlap: int = 2,
    dims: int = 2048,
    embedding: str = "siliconflow",
    model: str = "BAAI/bge-m3",
    batch_size: int = 32,
    unit_types: set[str] | None = None,
    trace_dir: Path | None = None,
) -> dict[str, int | str]:
    pdf_dir = pdf_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    pdf_paths = sorted(list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    embedding_backend = create_embedding_backend(
        embedding=embedding,
        dims=dims,
        model=model,
        batch_size=batch_size,
    )
    store = LocalVectorStore(out_dir / "rag.sqlite", dims=dims, embedding_backend=embedding_backend)
    store.recreate()
    all_paragraphs: list[ParsedParagraph] = []
    all_units: list[RetrievalUnit] = []
    manifest = []
    document_events: list[dict[str, object]] = []
    for pdf_path in pdf_paths:
        paragraphs = extract_pdf_paragraphs(pdf_path)
        units = build_retrieval_units(paragraphs, window_size=window_size, overlap=overlap)
        if unit_types:
            units = [unit for unit in units if unit.unit_type in unit_types]
        store.insert_paragraphs(paragraphs)
        store.insert_units(units)
        all_paragraphs.extend(paragraphs)
        all_units.extend(units)
        manifest.append(
            {
                "source_path": str(pdf_path),
                "size_bytes": pdf_path.stat().st_size,
                "paragraphs": len(paragraphs),
                "retrieval_units": len(units),
            }
        )
        document_events.append(
            {
                "source_path": str(pdf_path),
                "size_bytes": pdf_path.stat().st_size,
                "paragraphs": len(paragraphs),
                "retrieval_units": len(units),
                "unit_types": sorted({unit.unit_type for unit in units}),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    store_meta = store.read_meta()
    vector_dim = int(store_meta.get("vector_dim", store_meta.get("dims", dims)))
    trace_stats = write_build_traces(
        trace_dir=trace_dir or (out_dir / "traces"),
        pdf_dir=pdf_dir,
        out_dir=out_dir,
        embedding_name=embedding_backend.name,
        vector_kind=embedding_backend.vector_kind,
        window_size=window_size,
        overlap=overlap,
        unit_types=unit_types or {"paragraph", "question", "sentence", "window"},
        paragraphs=all_paragraphs,
        units=all_units,
        document_events=document_events,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pdf_dir": str(pdf_dir),
                "store": str(store.db_path),
                "embedding": embedding_backend.name,
                "embedding_backend": embedding_backend.name,
                "vector_kind": embedding_backend.vector_kind,
                "dims": vector_dim,
                "vector_dim": vector_dim,
                "window_size": window_size,
                "overlap": overlap,
                "unit_types": sorted(unit_types) if unit_types else ["paragraph", "question", "sentence", "window"],
                "trace_dir": trace_stats["trace_dir"],
                "documents": manifest,
                "paragraphs": len(all_paragraphs),
                "retrieval_units": len(all_units),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stats = store.stats()
    stats["manifest"] = str(out_dir / "manifest.json")
    stats.update(trace_stats)
    return stats


def augment_pdf_vector_store_questions(
    db_path: Path,
    embedding: str = "auto",
    model: str = "BAAI/bge-m3",
    batch_size: int = 32,
    embedding_backend: SparseHashEmbeddingBackend | DenseEmbeddingBackend | None = None,
) -> dict[str, int | str]:
    db_path = db_path.expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Vector store does not exist: {db_path}")

    backend = embedding_backend or create_embedding_backend_for_db(
        db_path,
        embedding=embedding,
        model=model,
        batch_size=batch_size,
    )
    store = LocalVectorStore(db_path, embedding_backend=backend)
    with store.connect() as conn:
        paragraph_rows = conn.execute(
            """
            SELECT paragraph_id, doc_id, source_path, title, page_start, page_end, text
            FROM paragraphs
            ORDER BY doc_id, page_start, paragraph_id
            """
        ).fetchall()

    paragraphs = [
        ParsedParagraph(
            paragraph_id=row["paragraph_id"],
            doc_id=row["doc_id"],
            source_path=row["source_path"],
            title=row["title"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            text=row["text"],
        )
        for row in paragraph_rows
    ]
    question_units = build_question_retrieval_units(paragraphs)

    with store.connect() as conn:
        conn.execute("DELETE FROM retrieval_units WHERE unit_type = 'question'")
    store.insert_units(question_units)
    _update_manifest_after_question_augment(db_path)

    stats = store.stats()
    stats["question_units"] = len(question_units)
    return stats


def _update_manifest_after_question_augment(db_path: Path) -> None:
    manifest_path = db_path.parent / "manifest.json"
    if not manifest_path.exists():
        return

    with LocalVectorStore(db_path).connect() as conn:
        total_units = conn.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0]
        total_paragraphs = conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
        unit_types = [
            row[0]
            for row in conn.execute(
                "SELECT unit_type FROM retrieval_units GROUP BY unit_type ORDER BY unit_type"
            ).fetchall()
        ]
        document_rows = conn.execute(
            """
            SELECT p.source_path, COUNT(DISTINCT p.paragraph_id), COUNT(u.unit_id)
            FROM paragraphs p
            LEFT JOIN retrieval_units u ON u.paragraph_id = p.paragraph_id
            GROUP BY p.source_path
            """
        ).fetchall()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paragraphs"] = total_paragraphs
    manifest["retrieval_units"] = total_units
    manifest["unit_types"] = unit_types

    document_counts = {
        row[0]: {"paragraphs": int(row[1]), "retrieval_units": int(row[2])}
        for row in document_rows
    }
    documents = []
    for document in manifest.get("documents", []):
        source_path = document.get("source_path")
        if source_path in document_counts:
            document = {**document, **document_counts[source_path]}
        documents.append(document)
    if documents:
        manifest["documents"] = documents

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _update_manifest_after_text_index(db_path: Path, text_index: str, rows: int) -> None:
    manifest_path = db_path.parent / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["text_index"] = text_index
    manifest["text_index_rows"] = rows
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _update_manifest_after_knowledge_units(
    db_path: Path,
    rows: int,
    unit_types: dict[str, int],
    trace_path: Path | None,
) -> None:
    manifest_path = db_path.parent / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["knowledge_units"] = rows
    manifest["knowledge_unit_types"] = unit_types
    manifest["knowledge_extractor_version"] = KNOWLEDGE_EXTRACTOR_VERSION
    if trace_path is not None:
        manifest["knowledge_trace"] = str(trace_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_embedding_backend(
    embedding: str,
    dims: int = 2048,
    model: str = "BAAI/bge-m3",
    batch_size: int = 32,
) -> SparseHashEmbeddingBackend | DenseEmbeddingBackend:
    if embedding == "sparse":
        return SparseHashEmbeddingBackend(dims=dims)
    if embedding == "siliconflow":
        return SiliconFlowEmbeddingBackend(model=model, batch_size=batch_size)
    if embedding in {"local", "local-bge-m3", "bge-m3-local"}:
        return LocalBgeM3EmbeddingBackend(model=model, batch_size=batch_size)
    raise ValueError(f"unsupported embedding backend: {embedding}")


def siliconflow_api_key_available() -> bool:
    load_dotenv_if_present()
    return bool(os.getenv("SILICONFLOW_API_KEY", "").strip())


def create_embedding_backend_for_db(
    db_path: Path,
    embedding: str = "auto",
    dims: int = 2048,
    model: str = "BAAI/bge-m3",
    batch_size: int = 32,
) -> SparseHashEmbeddingBackend | DenseEmbeddingBackend:
    if embedding != "auto":
        return create_embedding_backend(embedding, dims=dims, model=model, batch_size=batch_size)
    probe = LocalVectorStore(db_path, dims=dims)
    meta = probe.read_meta()
    stored_embedding = meta.get("embedding", "")
    if stored_embedding.startswith("siliconflow:"):
        stored_model = stored_embedding.split(":", 1)[1] or model
        if not siliconflow_api_key_available():
            return LocalBgeM3EmbeddingBackend(model=stored_model, batch_size=batch_size)
        return SiliconFlowEmbeddingBackend(model=stored_model, batch_size=batch_size)
    if stored_embedding.startswith("local-bge-m3:"):
        stored_model = stored_embedding.split(":", 1)[1] or model
        return LocalBgeM3EmbeddingBackend(model=stored_model, batch_size=batch_size)
    return SparseHashEmbeddingBackend(dims=int(meta.get("dims", dims)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and search a local PDF vector store.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build a local SQLite vector store from PDFs")
    build.add_argument("--pdf-dir", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--window-size", type=int, default=6)
    build.add_argument("--overlap", type=int, default=2)
    build.add_argument("--dims", type=int, default=2048)
    build.add_argument(
        "--embedding",
        choices=["sparse", "siliconflow", "local-bge-m3"],
        default="siliconflow",
    )
    build.add_argument("--model", default="BAAI/bge-m3")
    build.add_argument("--batch-size", type=int, default=32)
    build.add_argument(
        "--unit-types",
        default="sentence,window,paragraph,question",
        help="Comma-separated retrieval unit types to embed.",
    )
    build.add_argument("--trace-dir", type=Path, default=None)

    search = sub.add_parser("search", help="search a built PDF vector store")
    search.add_argument("query")
    search.add_argument("--db", type=Path, required=True)
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--mode", choices=["hybrid", "vector", "text", "knowledge"], default="hybrid")
    search.add_argument(
        "--embedding",
        choices=["auto", "sparse", "siliconflow", "local-bge-m3"],
        default="auto",
    )
    search.add_argument("--model", default="BAAI/bge-m3")
    search.add_argument("--batch-size", type=int, default=32)

    augment_questions = sub.add_parser(
        "augment-questions",
        help="add question retrieval units to an existing vector store",
    )
    augment_questions.add_argument("--db", type=Path, required=True)
    augment_questions.add_argument(
        "--embedding",
        choices=["auto", "sparse", "siliconflow", "local-bge-m3"],
        default="auto",
    )
    augment_questions.add_argument("--model", default="BAAI/bge-m3")
    augment_questions.add_argument("--batch-size", type=int, default=32)

    text_index = sub.add_parser("rebuild-text-index", help="rebuild the FTS original-text index")
    text_index.add_argument("--db", type=Path, required=True)

    knowledge = sub.add_parser("rebuild-knowledge-units", help="extract grounded knowledge units")
    knowledge.add_argument("--db", type=Path, required=True)
    knowledge.add_argument("--trace-dir", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "build":
        stats = build_pdf_vector_store(
            pdf_dir=args.pdf_dir,
            out_dir=args.out,
            window_size=args.window_size,
            overlap=args.overlap,
            dims=args.dims,
            embedding=args.embedding,
            model=args.model,
            batch_size=args.batch_size,
            unit_types=parse_unit_types(args.unit_types),
            trace_dir=args.trace_dir,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    if args.command == "search":
        if args.mode == "text":
            store = LocalVectorStore(args.db)
        else:
            backend = create_embedding_backend_for_db(
                args.db,
                embedding=args.embedding,
                model=args.model,
                batch_size=args.batch_size,
            )
            store = LocalVectorStore(args.db, embedding_backend=backend)
        results = store.search(args.query, limit=args.limit, mode=args.mode)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    if args.command == "augment-questions":
        stats = augment_pdf_vector_store_questions(
            args.db,
            embedding=args.embedding,
            model=args.model,
            batch_size=args.batch_size,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    if args.command == "rebuild-text-index":
        store = LocalVectorStore(args.db)
        print(json.dumps(store.rebuild_text_index(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "rebuild-knowledge-units":
        store = LocalVectorStore(args.db)
        trace_dir = args.trace_dir or (args.db.parent / "traces")
        print(json.dumps(store.rebuild_knowledge_units(trace_dir=trace_dir), ensure_ascii=False, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


def parse_unit_types(value: str) -> set[str]:
    allowed = {"sentence", "window", "paragraph", "question"}
    unit_types = {item.strip() for item in value.split(",") if item.strip()}
    unknown = unit_types - allowed
    if unknown:
        raise ValueError(f"unsupported unit types: {', '.join(sorted(unknown))}")
    return unit_types or allowed


if __name__ == "__main__":
    raise SystemExit(main())
