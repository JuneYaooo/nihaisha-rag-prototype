from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

from nihaisha_kg.rerank import SiliconFlowReranker, sanitize_rerank_error


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeSession:
    def __init__(self, payload: object = None, error: Exception | None = None) -> None:
        self.payload = {} if payload is None else payload
        self.error = error
        self.calls: list[tuple[str, dict[str, str], dict[str, object], float]] = []

    def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append((url, headers, json, timeout))
        if self.error is not None:
            raise self.error
        return FakeResponse(self.payload)


class RerankTests(unittest.TestCase):
    def test_sanitizer_redacts_quoted_and_unquoted_authorization_values_atomically(self) -> None:
        messages = (
            "Authorization: Bearer supersecret",
            'Authorization="Bearer supersecret"',
            "Authorization='Bearer supersecret'",
        )

        for message in messages:
            with self.subTest(message=message):
                sanitized = sanitize_rerank_error(message)
                self.assertNotIn("supersecret", sanitized)
                self.assertNotIn("Bearer", sanitized)
                self.assertIn("[REDACTED]", sanitized)

    def test_posts_documented_request_and_maps_score_order_without_mutation(self) -> None:
        session = FakeSession(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.21},
                ]
            }
        )
        backend = SiliconFlowReranker(api_key="secret", session=session, max_retries=1)
        candidates = [
            {"paragraph_id": "p1", "title": " 甲 ", "text": "无关"},
            {"paragraph_id": "p2", "title": "乙", "text": " 桂枝汤主之 "},
        ]
        original = [dict(candidate) for candidate in candidates]

        outcome = backend.rerank("桂枝汤", candidates, limit=2)

        self.assertEqual([row["paragraph_id"] for row in outcome.results], ["p2", "p1"])
        self.assertEqual([row["rerank_score"] for row in outcome.results], [0.91, 0.21])
        self.assertEqual(candidates, original)
        self.assertIsNot(outcome.results[0], candidates[1])
        self.assertEqual(len(session.calls), 1)
        url, headers, payload, timeout = session.calls[0]
        self.assertEqual(url, "https://api.siliconflow.cn/v1/rerank")
        self.assertEqual(
            headers,
            {"Authorization": "Bearer secret", "Content-Type": "application/json"},
        )
        self.assertEqual(
            payload,
            {
                "model": "BAAI/bge-reranker-v2-m3",
                "query": "桂枝汤",
                "documents": ["甲\n无关", "乙\n桂枝汤主之"],
                "return_documents": False,
                "top_n": 2,
            },
        )
        self.assertEqual(timeout, 60)

    def test_unsorted_response_is_ranked_by_score_with_original_index_ties(self) -> None:
        session = FakeSession(
            {
                "results": [
                    {"index": 2, "relevance_score": 0.4},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.9},
                ]
            }
        )
        backend = SiliconFlowReranker(api_key="secret", session=session, max_retries=1)
        candidates = [
            {"paragraph_id": "p0", "text": "甲"},
            {"paragraph_id": "p1", "text": "乙"},
            {"paragraph_id": "p2", "text": "丙"},
        ]

        outcome = backend.rerank("问题", candidates, limit=2)

        self.assertEqual([row["paragraph_id"] for row in outcome.results], ["p0", "p1"])
        self.assertEqual([row["rerank_score"] for row in outcome.results], [0.9, 0.9])

    def test_blank_documents_are_omitted_and_response_indices_map_to_original_candidates(self) -> None:
        session = FakeSession({"results": [{"index": 1, "relevance_score": 0.8}]})
        backend = SiliconFlowReranker(api_key="secret", session=session, max_retries=1)
        candidates = [
            {"paragraph_id": "blank", "title": " ", "text": "\n"},
            {"paragraph_id": "first", "title": "甲", "text": "正文"},
            {"paragraph_id": "second", "text": "第二段"},
        ]

        outcome = backend.rerank("问题", candidates, limit=3)

        self.assertEqual(session.calls[0][2]["documents"], ["甲\n正文", "第二段"])
        self.assertEqual(session.calls[0][2]["top_n"], 2)
        self.assertEqual(outcome.results[0]["paragraph_id"], "second")

    def test_non_strict_network_failure_falls_back_without_sleep_or_secret_leak(self) -> None:
        secret = "top-secret"
        session = FakeSession(error=RuntimeError(f"offline Bearer {secret}"))
        backend = SiliconFlowReranker(
            api_key=secret,
            session=session,
            max_retries=1,
            strict=False,
        )
        candidates = [
            {"paragraph_id": "p1", "text": "甲"},
            {"paragraph_id": "p2", "text": "乙"},
            {"paragraph_id": "p3", "text": "丙"},
        ]

        with patch("nihaisha_kg.rerank.time.sleep") as sleep:
            outcome = backend.rerank("问题", candidates, limit=2)

        self.assertEqual([row["paragraph_id"] for row in outcome.results], ["p1", "p2"])
        self.assertEqual(outcome.degraded_feature, "siliconflow_rerank")
        self.assertIn("offline", outcome.error)
        self.assertNotIn(secret, outcome.error)
        self.assertIsNot(outcome.results[0], candidates[0])
        sleep.assert_not_called()

    def test_strict_failure_raises_runtime_error_chained_from_cause(self) -> None:
        cause = OSError("connection reset")
        backend = SiliconFlowReranker(
            api_key="secret",
            session=FakeSession(error=cause),
            max_retries=1,
            strict=True,
        )

        with self.assertRaisesRegex(RuntimeError, "SiliconFlow rerank request failed") as raised:
            backend.rerank("问题", [{"text": "甲"}], limit=1)

        self.assertIs(raised.exception.__cause__, cause)

    def test_malformed_results_are_rejected_as_a_whole_with_deterministic_fallback(self) -> None:
        malformed_payloads = {
            "missing results": {},
            "results not a list": {"results": {}},
            "item not an object": {"results": ["bad"]},
            "boolean index": {"results": [{"index": True, "relevance_score": 0.5}]},
            "out of range": {"results": [{"index": 2, "relevance_score": 0.5}]},
            "duplicate index": {
                "results": [
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 0, "relevance_score": 0.7},
                ]
            },
            "boolean score": {"results": [{"index": 0, "relevance_score": True}]},
            "non-finite score": {"results": [{"index": 0, "relevance_score": math.inf}]},
        }
        candidates = [{"paragraph_id": "p1", "text": "甲"}, {"paragraph_id": "p2", "text": "乙"}]

        for name, payload in malformed_payloads.items():
            with self.subTest(name=name):
                backend = SiliconFlowReranker(
                    api_key="secret",
                    session=FakeSession(payload),
                    max_retries=1,
                )
                outcome = backend.rerank("问题", candidates, limit=2)
                self.assertEqual(
                    [row["paragraph_id"] for row in outcome.results],
                    ["p1", "p2"],
                )
                self.assertEqual(outcome.degraded_feature, "siliconflow_rerank")
                self.assertTrue(outcome.error)

    def test_empty_candidates_nonpositive_limit_and_all_blank_skip_http(self) -> None:
        session = FakeSession({"results": []})
        backend = SiliconFlowReranker(api_key="secret", session=session, max_retries=1)

        self.assertEqual(backend.rerank("问题", [], limit=2).results, [])
        self.assertEqual(backend.rerank("问题", [{"text": "甲"}], limit=0).results, [])
        self.assertEqual(backend.rerank("问题", [{"title": " ", "text": "\n"}], limit=2).results, [])
        self.assertEqual(session.calls, [])

    def test_model_precedence_is_explicit_then_environment_then_default(self) -> None:
        with patch.dict(os.environ, {"SILICONFLOW_RERANK_MODEL": "env-model"}):
            explicit = SiliconFlowReranker(api_key="secret", model="explicit-model")
            environment = SiliconFlowReranker(api_key="secret")
        with patch.dict(os.environ, {}, clear=True):
            default = SiliconFlowReranker(api_key="secret")

        self.assertEqual(explicit.model, "explicit-model")
        self.assertEqual(environment.model, "env-model")
        self.assertEqual(default.model, "BAAI/bge-reranker-v2-m3")

    def test_missing_key_and_invalid_configuration_fail_before_http(self) -> None:
        session = FakeSession({"results": []})
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SILICONFLOW_API_KEY"):
                SiliconFlowReranker(session=session)
        for kwargs in ({"timeout": 0}, {"timeout": -1}, {"max_retries": 0}, {"max_retries": -1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SiliconFlowReranker(api_key="secret", session=session, **kwargs)
        self.assertEqual(session.calls, [])

    def test_constructor_rejects_invalid_resolved_strings_and_numeric_options(self) -> None:
        invalid_cases = [
            ("whitespace key", {"api_key": " "}, RuntimeError, "SILICONFLOW_API_KEY"),
            ("non-string key", {"api_key": 123}, ValueError, "api_key"),
            ("whitespace model", {"api_key": "key", "model": " "}, ValueError, "model"),
            ("non-string model", {"api_key": "key", "model": 123}, ValueError, "model"),
            ("whitespace url", {"api_key": "key", "base_url": " "}, ValueError, "base_url"),
            ("non-string url", {"api_key": "key", "base_url": 123}, ValueError, "base_url"),
            ("relative url", {"api_key": "key", "base_url": "/v1"}, ValueError, "base_url"),
            ("invalid scheme", {"api_key": "key", "base_url": "ftp://host/v1"}, ValueError, "base_url"),
            ("missing host", {"api_key": "key", "base_url": "https:///v1"}, ValueError, "base_url"),
            ("boolean timeout", {"api_key": "key", "timeout": True}, ValueError, "timeout"),
            ("zero timeout", {"api_key": "key", "timeout": 0}, ValueError, "timeout"),
            ("negative timeout", {"api_key": "key", "timeout": -1}, ValueError, "timeout"),
            ("nan timeout", {"api_key": "key", "timeout": math.nan}, ValueError, "timeout"),
            ("string timeout", {"api_key": "key", "timeout": "60"}, ValueError, "timeout"),
            ("boolean retries", {"api_key": "key", "max_retries": True}, ValueError, "max_retries"),
            ("zero retries", {"api_key": "key", "max_retries": 0}, ValueError, "max_retries"),
            ("negative retries", {"api_key": "key", "max_retries": -1}, ValueError, "max_retries"),
            ("float retries", {"api_key": "key", "max_retries": 1.5}, ValueError, "max_retries"),
        ]

        for name, kwargs, error_type, message in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(error_type, message):
                    SiliconFlowReranker(**kwargs)

    def test_constructor_strips_valid_strings_and_trailing_url_slash(self) -> None:
        backend = SiliconFlowReranker(
            api_key=" key ",
            model=" model ",
            base_url=" https://example.test/v1/ ",
        )

        self.assertEqual(backend.api_key, "key")
        self.assertEqual(backend.model, "model")
        self.assertEqual(backend.base_url, "https://example.test/v1")


if __name__ == "__main__":
    unittest.main()
