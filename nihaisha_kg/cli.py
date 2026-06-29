from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pdf_vector import (
    LocalVectorStore,
    answer_pdf_rag,
    build_faiss_vector_index,
    create_embedding_backend_for_db,
)


DEFAULT_DB = Path("data/pdf_rag_bge_m3/rag.sqlite")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nihaisha-rag",
        description="Search and answer from the local Nihaisha PDF RAG database.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="show database counts")
    stats.add_argument("--db", type=Path, default=DEFAULT_DB)

    build_faiss = sub.add_parser("build-faiss", help="build FAISS index from stored dense vectors")
    build_faiss.add_argument("--db", type=Path, default=DEFAULT_DB)
    build_faiss.add_argument("--index", type=Path, default=None)
    build_faiss.add_argument("--ids", type=Path, default=None)
    build_faiss.add_argument("--batch-size", type=int, default=4096)

    search = sub.add_parser("search", help="search the local PDF RAG database")
    search.add_argument("query")
    search.add_argument("--db", type=Path, default=DEFAULT_DB)
    search.add_argument("--limit", type=int, default=5)
    search.add_argument(
        "--mode",
        choices=["hybrid", "vector", "text", "knowledge"],
        default="hybrid",
    )
    search.add_argument(
        "--embedding",
        choices=["auto", "sparse", "siliconflow", "local-bge-m3"],
        default="auto",
    )
    search.add_argument("--model", default="BAAI/bge-m3")
    search.add_argument("--batch-size", type=int, default=32)
    search.add_argument("--json", action="store_true")

    answer = sub.add_parser("answer", help="answer with grounded citations")
    answer.add_argument("query")
    answer.add_argument("--db", type=Path, default=DEFAULT_DB)
    answer.add_argument("--limit", type=int, default=8)
    answer.add_argument(
        "--mode",
        choices=["hybrid", "vector", "text", "knowledge"],
        default="hybrid",
    )
    answer.add_argument(
        "--embedding",
        choices=["auto", "sparse", "siliconflow", "local-bge-m3"],
        default="auto",
    )
    answer.add_argument("--model", default="BAAI/bge-m3")
    answer.add_argument("--batch-size", type=int, default=32)
    answer.add_argument("--composer", choices=["template", "llm"], default="template")
    answer.add_argument("--llm-model", default=None)
    answer.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "stats":
        store = LocalVectorStore(args.db)
        payload = store.stats()
        payload.update(store.read_meta())
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-faiss":
        payload = build_faiss_vector_index(
            args.db,
            index_path=args.index,
            ids_path=args.ids,
            batch_size=args.batch_size,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        if args.mode in {"text", "knowledge"}:
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
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0
        for index, item in enumerate(results, start=1):
            print(
                f"[{index}] score={item['score']} "
                f"sources={','.join(item.get('retrieval_sources', []))} "
                f"{Path(str(item['source_path'])).name} p{item['page_start']}"
            )
            for unit in item.get("matched_knowledge_units", [])[:3]:
                print(
                    "knowledge: "
                    f"{unit['unit_type']} | {unit['subject']} | {unit['predicate']} | {unit['object']}"
                )
            print(str(item["text"])[:500])
            print()
        return 0

    if args.command == "answer":
        payload = answer_pdf_rag(
            args.query,
            db_path=args.db,
            limit=args.limit,
            mode=args.mode,
            embedding=args.embedding,
            model=args.model,
            batch_size=args.batch_size,
            composer=args.composer,
            llm_model=args.llm_model,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        answer_text = str(payload["answer"])
        print(answer_text)
        answer_already_has_safety = args.composer == "llm" and "安全边界" in answer_text
        if payload.get("safety_notice") and not answer_already_has_safety:
            print()
            print(f"安全边界：{payload['safety_notice']}")
        print()
        print("引用：")
        for citation in payload.get("citations", []):
            print(f"[{citation['index']}] {citation['label']}")
            print(str(citation["evidence_quote"])[:260])
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
