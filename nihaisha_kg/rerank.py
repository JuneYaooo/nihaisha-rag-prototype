from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


DEFAULT_SILICONFLOW_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


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
        timeout: float = 60,
        max_retries: int = 2,
        strict: bool = False,
        session: object | None = None,
    ) -> None:
        configured_key = api_key if api_key is not None else os.getenv("SILICONFLOW_API_KEY", "")
        self.api_key = configured_key.strip()
        configured_model = model if model is not None else os.getenv("SILICONFLOW_RERANK_MODEL")
        self.model = (configured_model or DEFAULT_SILICONFLOW_RERANK_MODEL).strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.strict = strict
        self.session = session

        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for SiliconFlow reranking")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries <= 0:
            raise ValueError("max_retries must be a positive integer")

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, object]],
        limit: int,
    ) -> RerankOutcome:
        if not candidates or limit <= 0:
            return RerankOutcome(results=[], model=self.model)

        documents: list[str] = []
        original_indices: list[int] = []
        for original_index, candidate in enumerate(candidates):
            title = str(candidate.get("title", "")).strip()
            text = str(candidate.get("text", "")).strip()
            document = "\n".join(part for part in (title, text) if part)
            if not document:
                continue
            documents.append(document)
            original_indices.append(original_index)
        if not documents:
            return RerankOutcome(results=[], model=self.model)

        top_n = min(limit, len(documents))
        payload: dict[str, object] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
            "top_n": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        session = self.session
        if session is None:
            import requests

            session = requests.Session()

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
                ranked = self._validated_results(
                    response.json(),
                    candidates=candidates,
                    original_indices=original_indices,
                    limit=top_n,
                )
                return RerankOutcome(results=ranked, model=self.model)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        assert last_error is not None
        if self.strict:
            raise RuntimeError(
                f"SiliconFlow rerank request failed: {self._safe_error(last_error)}"
            ) from last_error
        return RerankOutcome(
            results=[dict(candidate) for candidate in candidates[:limit]],
            model=self.model,
            degraded_feature="siliconflow_rerank",
            error=self._safe_error(last_error),
        )

    @staticmethod
    def _validated_results(
        payload: object,
        *,
        candidates: Sequence[dict[str, object]],
        original_indices: Sequence[int],
        limit: int,
    ) -> list[dict[str, object]]:
        if not isinstance(payload, Mapping):
            raise ValueError("rerank response must be an object")
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("rerank response results must be a list")

        validated: list[tuple[int, float]] = []
        seen_indices: set[int] = set()
        for item in raw_results:
            if not isinstance(item, Mapping):
                raise ValueError("rerank result must be an object")
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("rerank result index must be an integer")
            if index < 0 or index >= len(original_indices):
                raise ValueError("rerank result index is out of range")
            if index in seen_indices:
                raise ValueError("rerank result indices must be unique")
            seen_indices.add(index)

            score = item.get("relevance_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("rerank relevance_score must be numeric")
            score_value = float(score)
            if not math.isfinite(score_value):
                raise ValueError("rerank relevance_score must be finite")
            validated.append((index, score_value))

        ranked: list[dict[str, object]] = []
        for document_index, score in validated[:limit]:
            candidate = dict(candidates[original_indices[document_index]])
            candidate["rerank_score"] = score
            ranked.append(candidate)
        return ranked

    def _safe_error(self, error: Exception) -> str:
        message = str(error)
        message = re.sub(r"(?i)bearer\s+[^\s,}\]]+", "[REDACTED]", message)
        if self.api_key:
            message = message.replace(self.api_key, "[REDACTED]")
        return message
