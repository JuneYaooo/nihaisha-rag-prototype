from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit


DEFAULT_SILICONFLOW_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
PUBLIC_RERANK_ERROR_MAX_CHARS = 240


def sanitize_rerank_error(
    error: object,
    *,
    api_key: str | None = None,
    max_chars: int = PUBLIC_RERANK_ERROR_MAX_CHARS,
) -> str:
    message = re.sub(r"[\x00-\x1f\x7f-\x9f]+", " ", str(error))
    secrets = [api_key, os.getenv("SILICONFLOW_API_KEY")]
    for secret in secrets:
        if isinstance(secret, str) and secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(
        r"(?i)\bauthorization\s*[:=]\s*(?:bearer\s+)?[^\s,;}\]]+",
        "Authorization: [REDACTED]",
        message,
    )
    message = re.sub(r"(?i)\bbearer\s+[^\s,;}\]]+", "Bearer [REDACTED]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{6,}\b", "[REDACTED]", message)
    message = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|token)\s*[:=]\s*[^\s,;}\]]+",
        r"\1=[REDACTED]",
        message,
    )
    message = " ".join(message.split())
    if max_chars <= 0:
        return ""
    if len(message) > max_chars:
        message = message[: max_chars - 1].rstrip() + "…"
    return message


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
        resolved_key = api_key if api_key is not None else os.getenv("SILICONFLOW_API_KEY", "")
        if not isinstance(resolved_key, str):
            raise ValueError("api_key must be a string")
        normalized_key = resolved_key.strip()
        if not normalized_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for SiliconFlow reranking")

        if model is not None:
            resolved_model = model
        else:
            environment_model = os.getenv("SILICONFLOW_RERANK_MODEL")
            resolved_model = (
                DEFAULT_SILICONFLOW_RERANK_MODEL
                if environment_model is None
                else environment_model
            )
        if not isinstance(resolved_model, str):
            raise ValueError("model must be a string")
        normalized_model = resolved_model.strip()
        if not normalized_model:
            raise ValueError("model must be a nonempty string")

        if not isinstance(base_url, str):
            raise ValueError("base_url must be a string")
        normalized_base_url = base_url.strip().rstrip("/")
        try:
            parsed_base_url = urlsplit(normalized_base_url)
        except ValueError as exc:
            raise ValueError("base_url must be an absolute http/https URL") from exc
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ValueError("base_url must be an absolute http/https URL")

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries <= 0:
            raise ValueError("max_retries must be a positive integer")

        self.api_key = normalized_key
        self.model = normalized_model
        self.base_url = normalized_base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.strict = strict
        self.session = session

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

        validated.sort(key=lambda pair: (-pair[1], original_indices[pair[0]]))
        ranked: list[dict[str, object]] = []
        for document_index, score in validated[:limit]:
            candidate = dict(candidates[original_indices[document_index]])
            candidate["rerank_score"] = score
            ranked.append(candidate)
        return ranked

    def _safe_error(self, error: Exception) -> str:
        return sanitize_rerank_error(error, api_key=self.api_key)
