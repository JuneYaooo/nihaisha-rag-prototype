from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nihaisha_kg.pdf_vector import (
    DenseEmbeddingBackend,
    LocalBgeM3EmbeddingBackend,
    LocalVectorStore,
    ParsedParagraph,
    RetrievalUnit,
    SiliconFlowEmbeddingBackend,
    SiliconFlowChatBackend,
    answer_pdf_rag,
    augment_pdf_vector_store_questions,
    compose_pdf_rag_answer_with_llm,
    expand_answer_query,
    extract_knowledge_units_from_paragraph,
    synthesize_pdf_rag_answer,
    write_build_traces,
    build_retrieval_units,
    build_faiss_vector_index,
    create_embedding_backend,
    create_embedding_backend_for_db,
    generate_paragraph_questions,
    pack_dense_vector,
    pack_sparse_vector,
    parse_unit_types,
    split_sentences,
    sparse_dot,
    unpack_dense_vector,
    unpack_sparse_vector,
)


class FakeFaissIndex:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.vectors: list[list[float]] = []

    def add(self, vectors: object) -> None:
        if hasattr(vectors, "tolist"):
            rows = vectors.tolist()
        else:
            rows = vectors
        self.vectors.extend([[float(value) for value in row] for row in rows])

    def search(self, queries: object, top_k: int) -> tuple[list[list[float]], list[list[int]]]:
        if hasattr(queries, "tolist"):
            query = queries.tolist()[0]
        else:
            query = queries[0]
        scored = []
        for index, vector in enumerate(self.vectors):
            score = sum(float(a) * float(b) for a, b in zip(query, vector))
            scored.append((score, index))
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = scored[:top_k]
        return [[score for score, _ in scored]], [[index for _, index in scored]]


class FakeFaiss:
    def __init__(self) -> None:
        self.indexes: dict[str, FakeFaissIndex] = {}

    def IndexFlatIP(self, dims: int) -> FakeFaissIndex:
        return FakeFaissIndex(dims)

    def write_index(self, index: FakeFaissIndex, path: str) -> None:
        self.indexes[path] = index
        Path(path).write_text("fake-faiss-index", encoding="utf-8")

    def read_index(self, path: str) -> FakeFaissIndex:
        return self.indexes[path]


class PdfVectorTests(unittest.TestCase):
    def test_split_sentences_keeps_chinese_clause_punctuation(self) -> None:
        text = "太阳中风，阳浮而阴弱。桂枝汤主之！若无汗，不可误作同证？"

        self.assertEqual(
            split_sentences(text),
            ["太阳中风，阳浮而阴弱。", "桂枝汤主之！", "若无汗，不可误作同证？"],
        )

    def test_build_retrieval_units_links_sentence_windows_to_paragraph(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p1",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="太阳病篇",
            page_start=3,
            page_end=3,
            text="太阳中风，阳浮而阴弱。桂枝汤主之。若无汗，不可误作同证。还须看恶风。",
        )

        units = build_retrieval_units([paragraph], window_size=2, overlap=1)
        unit_types = {unit.unit_type for unit in units}

        self.assertIn("sentence", unit_types)
        self.assertIn("window", unit_types)
        self.assertIn("paragraph", unit_types)
        self.assertTrue(all(unit.paragraph_id == "p1" for unit in units))
        self.assertTrue(any("太阳病篇" in unit.text_for_embedding for unit in units))

    def test_paragraph_generates_multiple_question_units_mapped_to_source(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p1",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="方证比较",
            page_start=9,
            page_end=9,
            text="太阳中风，汗出恶风，桂枝汤主之。太阳伤寒，无汗而喘，麻黄汤主之。少阴咽痛，可以讨论猪肤汤和桔梗汤。",
        )

        questions = generate_paragraph_questions(paragraph)
        units = build_retrieval_units([paragraph], window_size=2, overlap=1)
        question_units = [unit for unit in units if unit.unit_type == "question"]

        self.assertGreaterEqual(len(questions), 4)
        self.assertGreaterEqual(len(question_units), 4)
        self.assertTrue(all(unit.paragraph_id == "p1" for unit in question_units))
        self.assertTrue(all(unit.text.endswith("？") for unit in question_units))
        self.assertTrue(any("桂枝汤" in unit.text for unit in question_units))
        self.assertTrue(any("麻黄汤" in unit.text for unit in question_units))
        self.assertTrue(any("问题：" in unit.text_for_embedding for unit in question_units))

    def test_extract_knowledge_units_detects_grounded_dosage_method_and_formula(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p1",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="神农本草经 木香",
            page_start=33,
            page_end=33,
            text=(
                "一钱有人说是3.75克，也有人说4克，还有说5克，重点是黄金比例。"
                "木香饼（生地木香作饼），热熨贴之，治结肿成核，消乳中结核酸痛。"
                "太阳中风，汗出恶风，桂枝汤主之。若无汗，不可误作同证。"
            ),
        )

        units = extract_knowledge_units_from_paragraph(paragraph)
        by_type = {unit.unit_type: unit for unit in units}

        self.assertIn("dosage", by_type)
        self.assertEqual(by_type["dosage"].subject, "一钱")
        self.assertIn("3.75克", by_type["dosage"].object)
        self.assertIn("4克", by_type["dosage"].object)
        self.assertIn("5克", by_type["dosage"].object)
        self.assertIn("method", by_type)
        self.assertEqual(by_type["method"].subject, "木香饼热熨法")
        self.assertIn("结肿成核", by_type["method"].evidence_quote)
        self.assertIn("formula_pattern", by_type)
        self.assertEqual(by_type["formula_pattern"].subject, "桂枝汤")
        self.assertIn("caution", by_type)
        self.assertTrue(all(unit.paragraph_id == "p1" for unit in units))
        self.assertTrue(all(unit.source_path == "/tmp/doc.pdf" for unit in units))

    def test_rebuild_knowledge_units_is_idempotent_and_writes_trace(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="神农本草经 木香",
            page_start=33,
            page_end=33,
            text="木香饼（生地木香作饼），热熨贴之，治结肿成核，消乳中结核酸痛。",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps({"paragraphs": 1}), encoding="utf-8")
            store = LocalVectorStore(base / "rag.sqlite")
            store.recreate()
            store.insert_paragraphs([paragraph])

            first = store.rebuild_knowledge_units(trace_dir=base / "traces")
            second = store.rebuild_knowledge_units(trace_dir=base / "traces")
            results = store.search_knowledge_units("木香饼热熨法 出处", limit=3)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            trace_exists = (base / "traces" / "knowledge_units.jsonl").exists()

        self.assertEqual(first["knowledge_units"], 1)
        self.assertEqual(second["knowledge_units"], 1)
        self.assertEqual(results[0]["paragraph_id"], "p-a")
        self.assertEqual(results[0]["unit_type"], "method")
        self.assertEqual(manifest["knowledge_units"], 1)
        self.assertEqual(manifest["knowledge_unit_types"]["method"], 1)
        self.assertEqual(manifest["knowledge_trace"], str(base / "traces" / "knowledge_units.jsonl"))
        self.assertTrue(trace_exists)

    def test_hybrid_search_includes_grounded_knowledge_units(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="剂量说明",
            page_start=7,
            page_end=7,
            text="一钱有人说是3.75克，也有人说4克，还有说5克，重点是黄金比例。",
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalVectorStore(Path(tmpdir) / "rag.sqlite", embedding_backend=ToyDenseBackend())
            store.recreate()
            store.insert_paragraphs([paragraph])
            store.insert_units(build_retrieval_units([paragraph], window_size=2, overlap=1))
            store.rebuild_knowledge_units()

            results = store.search_hybrid("古时候一钱是多少克", limit=3)

        self.assertEqual(results[0]["paragraph_id"], "p-a")
        self.assertIn("knowledge", results[0]["retrieval_sources"])
        self.assertEqual(results[0]["matched_knowledge_units"][0]["unit_type"], "dosage")
        self.assertIn("3.75克", results[0]["matched_knowledge_units"][0]["object"])

    def test_synthesize_answer_aggregates_dosage_evidence_with_citations(self) -> None:
        results = [
            {
                "paragraph_id": "p-a",
                "source_path": "/tmp/仲景心法.pdf",
                "title": "仲景心法 p20",
                "page_start": 20,
                "page_end": 20,
                "text": "倪师在南宁讲一钱等于3.75克，人纪教程是一钱约等于5克。",
                "matched_knowledge_units": [
                    {
                        "unit_type": "dosage",
                        "subject": "一钱",
                        "predicate": "换算与剂量原则",
                        "object": "3.75克；5克",
                        "evidence_quote": "倪师在南宁讲一钱等于3.75克，人纪教程是一钱约等于5克。",
                    }
                ],
            },
            {
                "paragraph_id": "p-b",
                "source_path": "/tmp/伤寒视频.pdf",
                "title": "伤寒视频 p25",
                "page_start": 25,
                "page_end": 25,
                "text": "中国人跟我讲，我们是一钱是5克。有的人跟我说一钱是3.6克。",
                "matched_knowledge_units": [
                    {
                        "unit_type": "dosage",
                        "subject": "一钱",
                        "predicate": "换算与剂量原则",
                        "object": "5克",
                        "evidence_quote": "中国人跟我讲，我们是一钱是5克。",
                    }
                ],
            },
        ]

        answer = synthesize_pdf_rag_answer("古时候一钱是多少克？", results)

        self.assertEqual(answer["intent"], "dosage")
        self.assertIn("3.75克", answer["answer"])
        self.assertIn("5克", answer["answer"])
        self.assertIn("3.6克", answer["answer"])
        self.assertIn("[1]", answer["answer"])
        self.assertEqual(len(answer["citations"]), 2)
        self.assertIn("3.6克", answer["citations"][1]["evidence_quote"])
        self.assertIn("不是个人用药剂量建议", answer["safety_notice"])
        self.assertIn("不同人的体质不同", answer["safety_notice"])
        self.assertIn("不要私自购药", answer["safety_notice"])
        self.assertIn("线下正规中医", answer["safety_notice"])

    def test_dosage_query_expansion_uses_generic_terms_not_fixed_answer_values(self) -> None:
        expanded = expand_answer_query("古时候的一钱，是现代的多少克？")

        self.assertIn("剂量", expanded)
        self.assertIn("换算", expanded)
        self.assertIn("度量衡", expanded)
        self.assertIn("比例", expanded)
        self.assertNotIn("3.75克", expanded)
        self.assertNotIn("3.6克", expanded)
        self.assertNotIn("5克", expanded)

    def test_answer_pdf_rag_runs_followup_search_for_diverse_dosage_evidence(self) -> None:
        seed = ParsedParagraph(
            paragraph_id="p-seed",
            doc_id="doc",
            source_path="/tmp/伤寒.pdf",
            title="伤寒 p25",
            page_start=25,
            page_end=25,
            text="一钱是5克，也有人说一钱是3.6克，古今度量衡不同，不要斤斤计较，还要看比例。",
        )
        conversion = ParsedParagraph(
            paragraph_id="p-conversion",
            doc_id="doc",
            source_path="/tmp/本草.pdf",
            title="本草 p20",
            page_start=20,
            page_end=20,
            text="中国用克，几克几克，看不懂，大概1钱3.3克，有尾数的3.3克，差个1克2克，误解成4克5克，这一点点倒无所谓，1克的剂量差不了多少。",
        )
        ratio = ParsedParagraph(
            paragraph_id="p-ratio",
            doc_id="doc",
            source_path="/tmp/伤寒.pdf",
            title="伤寒 p49",
            page_start=49,
            page_end=49,
            text="所以黄金比例，就是4：3：2：2，黄金比例很重要。葛根汤重用葛根，再来重用麻黄。",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rag.sqlite"
            store = LocalVectorStore(db_path)
            store.recreate()
            store.insert_paragraphs([seed, conversion, ratio])
            store.rebuild_text_index()
            store.rebuild_knowledge_units()

            answer = answer_pdf_rag(
                "古时候的一钱，是现代的多少克？",
                db_path=db_path,
                mode="text",
                limit=3,
            )

        citation_text = "\n".join(str(citation["evidence_quote"]) for citation in answer["citations"])
        self.assertIn("3.3克", answer["answer"])
        self.assertIn("3.6克", answer["answer"])
        self.assertIn("5克", answer["answer"])
        self.assertNotIn("1克", answer["answer"])
        self.assertNotIn("2克", answer["answer"])
        self.assertIn("3.3克", citation_text)
        self.assertIn("黄金比例", citation_text)
        self.assertEqual(
            {"p-seed", "p-conversion", "p-ratio"},
            {str(citation["paragraph_id"]) for citation in answer["citations"]},
        )

    def test_answer_pdf_rag_returns_clinical_safety_boundary(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/金匮.pdf",
            title="金匮 p232",
            page_start=232,
            page_end=232,
            text="干呕就是恶心。下利，黄芩加半夏生姜汤。腹痛的时候一律加白芍。",
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rag.sqlite"
            store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
            store.recreate()
            store.insert_paragraphs([paragraph])
            store.insert_units(build_retrieval_units([paragraph], window_size=2, overlap=1))
            store.rebuild_knowledge_units()

            answer = answer_pdf_rag(
                "病人发烧后下利黄臭恶心，建议开什么方？",
                db_path=db_path,
                embedding_backend=ToyDenseBackend(),
                limit=3,
            )

        self.assertEqual(answer["intent"], "clinical")
        self.assertIn("不能替代诊断", answer["safety_notice"])
        self.assertIn("不同人的体质不同", answer["safety_notice"])
        self.assertIn("药效也和以前差很多", answer["safety_notice"])
        self.assertIn("不要私自购药", answer["safety_notice"])
        self.assertIn("不直接给个人处方", answer["answer"])
        self.assertIn("黄芩加半夏生姜汤", answer["answer"])
        self.assertGreaterEqual(len(answer["citations"]), 1)

    def test_siliconflow_chat_backend_posts_grounded_messages(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "基于证据回答。[1]"}}]}

        class FakeSession:
            def __init__(self) -> None:
                self.calls = []

            def post(self, url: str, headers: dict, json: dict, timeout: int) -> FakeResponse:
                self.calls.append((url, headers, json, timeout))
                return FakeResponse()

        session = FakeSession()
        backend = SiliconFlowChatBackend(
            api_key="secret",
            model="Qwen/Qwen3-32B",
            session=session,
        )

        content = backend.complete(
            [
                {"role": "system", "content": "只能用证据回答"},
                {"role": "user", "content": "证据[1]：一钱等于3.75克"},
            ]
        )

        self.assertEqual(content, "基于证据回答。[1]")
        self.assertEqual(session.calls[0][0], "https://api.siliconflow.cn/v1/chat/completions")
        self.assertEqual(session.calls[0][2]["model"], "Qwen/Qwen3-32B")
        self.assertIn("只能用证据回答", session.calls[0][2]["messages"][0]["content"])
        self.assertNotIn("secret", json.dumps(session.calls[0][2], ensure_ascii=False))

    def test_local_bge_m3_backend_uses_injected_flagembedding_model(self) -> None:
        class FakeBgeM3Model:
            def __init__(self) -> None:
                self.calls = []

            def encode(
                self,
                texts: list[str],
                batch_size: int,
                max_length: int,
                return_dense: bool,
                return_sparse: bool,
                return_colbert_vecs: bool,
            ) -> dict[str, list[list[float]]]:
                self.calls.append(
                    {
                        "texts": texts,
                        "batch_size": batch_size,
                        "max_length": max_length,
                        "return_dense": return_dense,
                        "return_sparse": return_sparse,
                        "return_colbert_vecs": return_colbert_vecs,
                    }
                )
                return {"dense_vecs": [[3.0, 4.0]]}

        model = FakeBgeM3Model()
        backend = LocalBgeM3EmbeddingBackend(model_instance=model, batch_size=7, max_length=4096)

        vectors = backend.embed_texts(["桂枝汤"])

        self.assertEqual(backend.name, "local-bge-m3:BAAI/bge-m3")
        self.assertEqual(backend.vector_kind, "dense")
        self.assertAlmostEqual(vectors[0][0], 0.6)
        self.assertAlmostEqual(vectors[0][1], 0.8)
        self.assertEqual(model.calls[0]["texts"], ["桂枝汤"])
        self.assertEqual(model.calls[0]["batch_size"], 7)
        self.assertEqual(model.calls[0]["max_length"], 4096)
        self.assertTrue(model.calls[0]["return_dense"])
        self.assertFalse(model.calls[0]["return_sparse"])
        self.assertFalse(model.calls[0]["return_colbert_vecs"])

    def test_create_embedding_backend_supports_local_bge_m3(self) -> None:
        backend = create_embedding_backend("local-bge-m3")

        self.assertIsInstance(backend, LocalBgeM3EmbeddingBackend)
        self.assertEqual(backend.name, "local-bge-m3:BAAI/bge-m3")

    def test_auto_embedding_uses_siliconflow_when_key_exists_and_local_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rag.sqlite"
            store = LocalVectorStore(db_path)
            store.recreate()
            with store.connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    ("embedding", "siliconflow:BAAI/bge-m3"),
                )

            with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "secret"}, clear=False):
                api_backend = create_embedding_backend_for_db(db_path)
            with patch.dict(os.environ, {"SILICONFLOW_API_KEY": ""}, clear=False):
                local_backend = create_embedding_backend_for_db(db_path)

        self.assertIsInstance(api_backend, SiliconFlowEmbeddingBackend)
        self.assertIsInstance(local_backend, LocalBgeM3EmbeddingBackend)

    def test_compose_answer_with_llm_uses_only_citation_evidence(self) -> None:
        class FakeChatBackend:
            def __init__(self) -> None:
                self.messages = []

            def complete(self, messages: list[dict[str, str]]) -> str:
                self.messages = messages
                return "一钱可见3.75克、5克等说法，仍需回到比例语境。[1][2]"

        answer = {
            "query": "一钱是多少克？",
            "intent": "dosage",
            "answer": "模板答案",
            "safety_notice": "不是个人用药剂量建议。",
            "citations": [
                {
                    "index": 1,
                    "label": "仲景心法 p20",
                    "evidence_quote": "一钱等于3.75克，人纪教程约等于5克。",
                },
                {
                    "index": 2,
                    "label": "伤寒 p25",
                    "evidence_quote": "一钱是5克。有的人说一钱是3.6克。",
                },
            ],
            "results": [{"text": "这段不应直接暴露给 LLM composer"}],
        }
        backend = FakeChatBackend()

        composed = compose_pdf_rag_answer_with_llm("一钱是多少克？", answer, backend)
        prompt_text = "\n".join(message["content"] for message in backend.messages)

        self.assertIn("一钱可见3.75克", composed["answer"])
        self.assertEqual(composed["composer"], "llm")
        self.assertEqual(composed["template_answer"], "模板答案")
        self.assertIn("一钱等于3.75克", prompt_text)
        self.assertIn("不可使用未列出的资料", prompt_text)
        self.assertIn("不要私自购药", prompt_text)
        self.assertIn("线下正规中医", prompt_text)
        self.assertNotIn("这段不应直接暴露", prompt_text)

    def test_parse_unit_types_accepts_question_units(self) -> None:
        self.assertEqual(
            parse_unit_types("sentence,question,paragraph"),
            {"sentence", "question", "paragraph"},
        )

    def test_vector_store_search_returns_paragraphs_deduplicated_by_score(self) -> None:
        paragraph_a = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )
        paragraph_b = ParsedParagraph(
            paragraph_id="p-b",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="麻黄汤证",
            page_start=2,
            page_end=2,
            text="太阳伤寒，无汗而喘，麻黄汤主之。",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalVectorStore(Path(tmpdir) / "rag.sqlite")
            store.recreate()
            store.insert_paragraphs([paragraph_a, paragraph_b])
            store.insert_units(build_retrieval_units([paragraph_a, paragraph_b], window_size=2, overlap=1))

            results = store.search("桂枝汤 汗出 恶风", limit=4)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["paragraph_id"], "p-a")
        self.assertIn("桂枝汤", results[0]["text"])
        self.assertEqual(len({item["paragraph_id"] for item in results}), len(results))

    def test_text_search_returns_exact_original_paragraph_without_embedding(self) -> None:
        paragraph_a = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )
        paragraph_b = ParsedParagraph(
            paragraph_id="p-b",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="少阴咽痛",
            page_start=2,
            page_end=2,
            text="少阴咽痛，可以讨论猪肤汤和桔梗汤。",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalVectorStore(Path(tmpdir) / "rag.sqlite")
            store.recreate()
            store.insert_paragraphs([paragraph_a, paragraph_b])

            results = store.search_text("猪肤汤", limit=2)

        self.assertEqual(results[0]["paragraph_id"], "p-b")
        self.assertIn("text", results[0]["retrieval_sources"])
        self.assertIn("猪肤汤", results[0]["matched_text_terms"])

    def test_hybrid_search_combines_vector_and_text_sources(self) -> None:
        paragraph_a = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )
        paragraph_b = ParsedParagraph(
            paragraph_id="p-b",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="少阴咽痛",
            page_start=2,
            page_end=2,
            text="少阴咽痛，可以讨论猪肤汤和桔梗汤。",
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                vectors = []
                for text in texts:
                    if "桂枝" in text or "汗出" in text or "恶风" in text:
                        vectors.append([1.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0])
                return vectors

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalVectorStore(Path(tmpdir) / "rag.sqlite", embedding_backend=ToyDenseBackend())
            store.recreate()
            store.insert_paragraphs([paragraph_a, paragraph_b])
            store.insert_units(build_retrieval_units([paragraph_a, paragraph_b], window_size=2, overlap=1))

            results = store.search_hybrid("猪肤汤", limit=4)

        by_id = {item["paragraph_id"]: item for item in results}
        self.assertIn("p-b", by_id)
        self.assertIn("text", by_id["p-b"]["retrieval_sources"])
        self.assertIn("猪肤汤", by_id["p-b"]["matched_text_terms"])

    def test_build_faiss_vector_index_writes_index_and_unit_mapping(self) -> None:
        paragraph_a = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )
        paragraph_b = ParsedParagraph(
            paragraph_id="p-b",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="麻黄汤证",
            page_start=2,
            page_end=2,
            text="太阳伤寒，无汗而喘，麻黄汤主之。",
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                vectors = []
                for text in texts:
                    if "桂枝" in text or "汗出" in text or "恶风" in text:
                        vectors.append([1.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0])
                return vectors

        fake_faiss = FakeFaiss()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            db_path = base / "rag.sqlite"
            store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
            store.recreate()
            store.insert_paragraphs([paragraph_a, paragraph_b])
            store.insert_units(build_retrieval_units([paragraph_a, paragraph_b], window_size=2, overlap=1))

            stats = build_faiss_vector_index(
                db_path,
                index_path=base / "vectors.faiss",
                ids_path=base / "vector_ids.jsonl",
                faiss_module=fake_faiss,
            )
            unit_count = store.stats()["retrieval_units"]
            mapping_lines = (base / "vector_ids.jsonl").read_text(encoding="utf-8").splitlines()
            index_exists = (base / "vectors.faiss").exists()

        self.assertEqual(stats["faiss_vectors"], unit_count)
        self.assertEqual(stats["faiss_dim"], 2)
        self.assertEqual(len(mapping_lines), unit_count)
        self.assertIn("unit_id", json.loads(mapping_lines[0]))
        self.assertTrue(index_exists)

    def test_build_faiss_vector_index_keeps_vectors_unweighted(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )
        unit = RetrievalUnit(
            unit_id="u-a",
            paragraph_id="p-a",
            doc_id="doc",
            unit_type="sentence",
            text="桂枝汤",
            text_for_embedding="桂枝汤",
            sentence_start=0,
            sentence_end=0,
            weight=3.0,
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        fake_faiss = FakeFaiss()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            db_path = base / "rag.sqlite"
            index_path = base / "vectors.faiss"
            store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
            store.recreate()
            store.insert_paragraphs([paragraph])
            store.insert_units([unit])
            build_faiss_vector_index(
                db_path,
                index_path=index_path,
                ids_path=base / "vector_ids.jsonl",
                faiss_module=fake_faiss,
            )
            indexed_vector = fake_faiss.indexes[str(index_path)].vectors[0]

        self.assertEqual(indexed_vector, [1.0, 0.0])

    def test_vector_search_uses_faiss_index_when_available(self) -> None:
        paragraph_a = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )
        paragraph_b = ParsedParagraph(
            paragraph_id="p-b",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="麻黄汤证",
            page_start=2,
            page_end=2,
            text="太阳伤寒，无汗而喘，麻黄汤主之。",
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                vectors = []
                for text in texts:
                    if "桂枝" in text or "汗出" in text or "恶风" in text:
                        vectors.append([1.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0])
                return vectors

        fake_faiss = FakeFaiss()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            db_path = base / "rag.sqlite"
            store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
            store.recreate()
            store.insert_paragraphs([paragraph_a, paragraph_b])
            store.insert_units(build_retrieval_units([paragraph_a, paragraph_b], window_size=2, overlap=1))
            build_faiss_vector_index(
                db_path,
                index_path=base / "vectors.faiss",
                ids_path=base / "vector_ids.jsonl",
                faiss_module=fake_faiss,
            )

            search_store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
            results = search_store.search_vector("桂枝汤 汗出", limit=2, faiss_module=fake_faiss)

        self.assertEqual(results[0]["paragraph_id"], "p-a")
        self.assertIn("faiss", results[0]["retrieval_sources"])
        self.assertGreater(results[0]["vector_score"], 0)

    def test_vector_search_rejects_embedding_kind_mismatch(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rag.sqlite"
            store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
            store.recreate()
            store.insert_paragraphs([paragraph])
            store.insert_units(build_retrieval_units([paragraph], window_size=2, overlap=1))

            mismatched_store = LocalVectorStore(db_path, embedding_backend=create_embedding_backend("sparse"))
            with self.assertRaisesRegex(RuntimeError, "vector_kind mismatch"):
                mismatched_store.search_vector("桂枝汤", limit=1)

    def test_rebuild_text_index_is_idempotent(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalVectorStore(Path(tmpdir) / "rag.sqlite")
            store.recreate()
            store.insert_paragraphs([paragraph])
            first = store.rebuild_text_index()
            second = store.rebuild_text_index()
            meta = store.read_meta()

        self.assertEqual(first["text_index_rows"], 1)
        self.assertEqual(second["text_index_rows"], 1)
        self.assertTrue(meta["text_index"].startswith("fts5_"))

    def test_rebuild_text_index_updates_manifest_when_present(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps({"paragraphs": 0}), encoding="utf-8")
            store = LocalVectorStore(base / "rag.sqlite")
            store.recreate()
            store.insert_paragraphs([paragraph])
            store.rebuild_text_index()

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["text_index"], "fts5_trigram")
        self.assertEqual(manifest["text_index_rows"], 1)

    def test_write_build_traces_records_intermediate_artifacts_without_secrets(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )
        units = build_retrieval_units([paragraph], window_size=2, overlap=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            write_build_traces(
                trace_dir=trace_dir,
                pdf_dir=Path("/tmp/pdfs"),
                out_dir=Path(tmpdir),
                embedding_name="siliconflow:BAAI/bge-m3",
                vector_kind="dense",
                window_size=2,
                overlap=1,
                unit_types={"sentence", "window", "paragraph", "question"},
                paragraphs=[paragraph],
                units=units,
                document_events=[
                    {
                        "source_path": "/tmp/doc.pdf",
                        "paragraphs": 1,
                        "retrieval_units": len(units),
                    }
                ],
            )

            paragraphs_path = trace_dir / "paragraphs.jsonl"
            units_path = trace_dir / "retrieval_units.jsonl"
            events_path = trace_dir / "build_events.jsonl"
            config_path = trace_dir / "build_config.json"
            trace_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in [paragraphs_path, units_path, events_path, config_path]
            )
            files_exist = all(
                path.exists()
                for path in [paragraphs_path, units_path, events_path, config_path]
            )

        self.assertTrue(files_exist)
        self.assertIn('"paragraph_id": "p-a"', trace_text)
        self.assertIn('"unit_type": "question"', trace_text)
        self.assertNotIn("SILICONFLOW_API_KEY", trace_text)
        self.assertNotIn("sk-", trace_text)

    def test_sparse_vector_blob_round_trips_with_cosine_score(self) -> None:
        vector = {1: 0.5, 42: 0.25, 2047: 0.75}

        restored = unpack_sparse_vector(pack_sparse_vector(vector))

        self.assertEqual(set(restored), set(vector))
        self.assertAlmostEqual(sparse_dot(vector, restored), sparse_dot(vector, vector), places=5)

    def test_dense_vector_blob_round_trips_normalized_vector(self) -> None:
        vector = [3.0, 4.0, 0.0]

        restored = unpack_dense_vector(pack_dense_vector(vector))

        self.assertAlmostEqual(restored[0], 0.6, places=5)
        self.assertAlmostEqual(restored[1], 0.8, places=5)
        self.assertAlmostEqual(sum(value * value for value in restored), 1.0, places=5)

    def test_siliconflow_backend_posts_bge_m3_batches(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "data": [
                        {"index": 0, "embedding": [1.0, 0.0]},
                        {"index": 1, "embedding": [0.0, 2.0]},
                    ]
                }

        class FakeSession:
            def __init__(self) -> None:
                self.calls = []

            def post(self, url: str, headers: dict, json: dict, timeout: int) -> FakeResponse:
                self.calls.append((url, headers, json, timeout))
                return FakeResponse()

        session = FakeSession()
        backend = SiliconFlowEmbeddingBackend(api_key="secret", session=session, batch_size=2)

        vectors = backend.embed_texts(["桂枝汤", "麻黄汤"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 2.0]])
        self.assertEqual(session.calls[0][0], "https://api.siliconflow.cn/v1/embeddings")
        self.assertEqual(session.calls[0][2]["model"], "BAAI/bge-m3")
        self.assertEqual(session.calls[0][2]["input"], ["桂枝汤", "麻黄汤"])
        self.assertEqual(session.calls[0][2]["encoding_format"], "float")

    def test_vector_store_can_use_dense_embedding_backend(self) -> None:
        paragraph_a = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )
        paragraph_b = ParsedParagraph(
            paragraph_id="p-b",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="麻黄汤证",
            page_start=2,
            page_end=2,
            text="太阳伤寒，无汗而喘，麻黄汤主之。",
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                vectors = []
                for text in texts:
                    if "桂枝" in text or "汗出" in text or "恶风" in text:
                        vectors.append([1.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0])
                return vectors

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalVectorStore(Path(tmpdir) / "rag.sqlite", embedding_backend=ToyDenseBackend())
            store.recreate()
            store.insert_paragraphs([paragraph_a, paragraph_b])
            store.insert_units(build_retrieval_units([paragraph_a, paragraph_b], window_size=2, overlap=1))

            results = store.search("桂枝汤 汗出 恶风", limit=2)

        self.assertEqual(results[0]["paragraph_id"], "p-a")

    def test_dense_vector_store_records_actual_vector_dimension(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="桂枝汤证",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。",
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalVectorStore(
                Path(tmpdir) / "rag.sqlite",
                dims=2048,
                embedding_backend=ToyDenseBackend(),
            )
            store.recreate()
            store.insert_paragraphs([paragraph])
            store.insert_units(build_retrieval_units([paragraph], window_size=2, overlap=1))

            meta = store.read_meta()

        self.assertEqual(meta["dims"], "2")
        self.assertEqual(meta["vector_dim"], "2")

    def test_augment_pdf_vector_store_questions_is_idempotent(self) -> None:
        paragraph = ParsedParagraph(
            paragraph_id="p-a",
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            title="方证比较",
            page_start=1,
            page_end=1,
            text="太阳中风，汗出恶风，桂枝汤主之。太阳伤寒，无汗而喘，麻黄汤主之。",
        )

        class ToyDenseBackend(DenseEmbeddingBackend):
            name = "toy_dense"

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rag.sqlite"
            store = LocalVectorStore(db_path, embedding_backend=ToyDenseBackend())
            store.recreate()
            store.insert_paragraphs([paragraph])
            base_units = [
                unit
                for unit in build_retrieval_units([paragraph], window_size=2, overlap=1)
                if unit.unit_type != "question"
            ]
            store.insert_units(base_units)

            first = augment_pdf_vector_store_questions(db_path, embedding_backend=ToyDenseBackend())
            second = augment_pdf_vector_store_questions(db_path, embedding_backend=ToyDenseBackend())

            with store.connect() as conn:
                question_count = conn.execute(
                    "SELECT COUNT(*) FROM retrieval_units WHERE unit_type = 'question'"
                ).fetchone()[0]

        self.assertGreaterEqual(first["question_units"], 4)
        self.assertEqual(second["question_units"], first["question_units"])
        self.assertEqual(question_count, first["question_units"])


if __name__ == "__main__":
    unittest.main()
