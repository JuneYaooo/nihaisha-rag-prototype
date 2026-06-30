from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pdf_vector import (
    LocalVectorStore,
    answer_pdf_rag,
    augment_pdf_vector_store_questions,
    build_faiss_vector_index,
    build_pdf_vector_store,
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

    build = sub.add_parser("build", help="build a complete local PDF RAG database")
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
    build.add_argument("--trace-dir", type=Path, default=None)

    augment_questions = sub.add_parser(
        "augment-questions",
        help="add paragraph question retrieval units to an existing database",
    )
    augment_questions.add_argument("--db", type=Path, default=DEFAULT_DB)
    augment_questions.add_argument(
        "--embedding",
        choices=["auto", "sparse", "siliconflow", "local-bge-m3"],
        default="auto",
    )
    augment_questions.add_argument("--model", default="BAAI/bge-m3")
    augment_questions.add_argument("--batch-size", type=int, default=32)

    text_index = sub.add_parser("rebuild-text-index", help="rebuild paragraph FTS index")
    text_index.add_argument("--db", type=Path, default=DEFAULT_DB)

    knowledge = sub.add_parser("rebuild-knowledge-units", help="rebuild grounded knowledge units")
    knowledge.add_argument("--db", type=Path, default=DEFAULT_DB)
    knowledge.add_argument("--trace-dir", type=Path, default=None)

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
    answer.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "stats":
        store = LocalVectorStore(args.db)
        payload = store.stats()
        payload.update(store.read_meta())
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build":
        payload = build_pdf_vector_store(
            pdf_dir=args.pdf_dir,
            out_dir=args.out,
            window_size=args.window_size,
            overlap=args.overlap,
            dims=args.dims,
            embedding=args.embedding,
            model=args.model,
            batch_size=args.batch_size,
            trace_dir=args.trace_dir,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "augment-questions":
        backend = create_embedding_backend_for_db(
            args.db,
            embedding=args.embedding,
            model=args.model,
            batch_size=args.batch_size,
        )
        payload = augment_pdf_vector_store_questions(args.db, embedding_backend=backend)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
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
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        answer_text = str(payload["answer"])
        print(answer_text)
        if payload.get("safety_notice"):
            print()
            print(f"安全边界：{payload['safety_notice']}")
        print()
        print("引用：")
        for citation in payload.get("citations", []):
            print(f"[{citation['index']}] {citation['label']}")
            print(str(citation["evidence_quote"])[:260])
        related_units = payload.get("related_knowledge_units", []) or []
        if related_units:
            print()
            print("关联知识点：")
            for unit in related_units[:8]:
                label = str(unit.get("label", "")).strip()
                if label:
                    print(f"- 来源：{label}")
                print(
                    f"  {unit.get('unit_type')} | {unit.get('subject')} | "
                    f"{unit.get('predicate')} | {unit.get('object')}"
                )
                quote = str(unit.get("evidence_quote", "")).strip()
                if quote:
                    print(f"  原文线索：{quote[:180]}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
