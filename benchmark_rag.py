import argparse
import json
import statistics
import time
from pathlib import Path

import psutil

from database import get_documents, optimize_pgvector_indexes
from embedding_engine import (
    answer_question,
    bm25_retrieve,
    get_embeddings,
    rerank_chunks,
    semantic_search,
    vector_retrieve,
)


def percentile(values: list[float], percentage: int) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = round((percentage / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def summarize_latencies(values: list[float]) -> dict:
    return {
        "count": len(values),
        "avg_ms": statistics.mean(values) if values else 0.0,
        "p50_ms": statistics.median(values) if values else 0.0,
        "p95_ms": percentile(values, 95),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        return [
            {"question": "What is this knowledge base about?", "expected_terms": []},
            {"question": "What sources are indexed in this project?", "expected_terms": []},
            {"question": "How does reranking improve retrieval?", "expected_terms": []},
        ]

    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_embedding_speed(sample_texts: list[str]) -> dict:
    if not sample_texts:
        return {
            "sample_count": 0,
            "elapsed_sec": 0.0,
            "embeddings_per_sec": 0.0,
            "process_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        }

    embeddings = get_embeddings()
    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()
    embeddings.embed_documents(sample_texts)
    elapsed_sec = time.perf_counter() - start

    ram_after_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "sample_count": len(sample_texts),
        "elapsed_sec": elapsed_sec,
        "embeddings_per_sec": len(sample_texts) / elapsed_sec if elapsed_sec else 0.0,
        "process_ram_before_mb": ram_before_mb,
        "process_ram_after_mb": ram_after_mb,
        "process_ram_delta_mb": max(0.0, ram_after_mb - ram_before_mb),
    }


def benchmark_retrieval_pipeline(
    questions: list[dict],
    chunk_strategy: str,
    chunk_size: int,
    overlap: int,
    vector_top_k: int,
    bm25_top_k: int,
    final_k: int,
) -> dict:
    vector_times = []
    bm25_times = []
    rerank_times = []
    answer_times = []
    total_times = []
    accuracy_hits = 0
    accuracy_count = 0

    for item in questions:
        question = item["question"]

        start_total = time.perf_counter()

        start = time.perf_counter()
        vector_results = vector_retrieve(question, chunk_strategy, vector_top_k)
        vector_times.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        bm25_results = bm25_retrieve(question, chunk_strategy, chunk_size, overlap, bm25_top_k)
        bm25_times.append((time.perf_counter() - start) * 1000)

        candidates = []
        seen = set()
        for chunk in vector_results + bm25_results:
            key = (
                chunk.metadata.get("document_id"),
                chunk.metadata.get("chunk_index"),
                chunk.metadata.get("source_name"),
            )
            if key not in seen:
                candidates.append(chunk)
                seen.add(key)

        start = time.perf_counter()
        reranked = rerank_chunks(question, candidates, final_k)
        rerank_times.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        answer_question(
            question,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            final_k=final_k,
        )
        answer_times.append((time.perf_counter() - start) * 1000)

        total_times.append((time.perf_counter() - start_total) * 1000)

        expected_terms = [term.lower() for term in item.get("expected_terms", []) if term]
        if expected_terms:
            accuracy_count += 1
            retrieved_text = " ".join(chunk["chunk_text"].lower() for chunk in reranked)
            if any(term in retrieved_text for term in expected_terms):
                accuracy_hits += 1

    return {
        "vector_search": summarize_latencies(vector_times),
        "bm25_search": summarize_latencies(bm25_times),
        "reranker": summarize_latencies(rerank_times),
        "answer_generation": summarize_latencies(answer_times),
        "end_to_end": summarize_latencies(total_times),
        "top_k_accuracy": {
            "question_count": accuracy_count,
            "hits": accuracy_hits,
            "accuracy": accuracy_hits / accuracy_count if accuracy_count else None,
        },
    }


def collect_sample_texts(limit: int) -> list[str]:
    texts = []

    for document in get_documents():
        if document.extracted_text:
            texts.append(document.extracted_text[:1500])
        if len(texts) >= limit:
            break

    return texts


def print_resume_lines(results: dict) -> None:
    vector_p50 = results["retrieval"]["vector_search"]["p50_ms"]
    total_p50 = results["retrieval"]["end_to_end"]["p50_ms"]
    embeddings_per_sec = results["embedding"]["embeddings_per_sec"]
    ram_mb = results["embedding"]["process_ram_after_mb"]
    accuracy = results["retrieval"]["top_k_accuracy"]["accuracy"]

    print("\nResume-ready lines:")
    print(
        "- Built a LangChain RAG system with pgvector semantic retrieval, "
        f"achieving {vector_p50:.0f} ms median vector-search latency."
    )
    print(
        "- Optimized local embedding generation with Hugging Face MiniLM, "
        f"processing {embeddings_per_sec:.1f} embeddings/sec at {ram_mb:.1f} MB process RAM."
    )
    print(
        "- Implemented hybrid retrieval with BM25, vector search, reranking, and cited answers, "
        f"with {total_p50:.0f} ms median end-to-end Q&A latency."
    )

    if accuracy is not None:
        print(
            "- Evaluated retrieval quality with labeled benchmark questions, "
            f"achieving {accuracy * 100:.1f}% top-k accuracy."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the RAG project without running Streamlit.")
    parser.add_argument("--questions", type=Path, default=Path("benchmark_questions.json"))
    parser.add_argument("--chunk-strategy", default="recursive")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--vector-top-k", type=int, default=8)
    parser.add_argument("--bm25-top-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--embedding-sample-size", type=int, default=10)
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    if not args.skip_index:
        print(optimize_pgvector_indexes()["message"])

    questions = load_questions(args.questions)
    sample_texts = collect_sample_texts(args.embedding_sample_size)

    results = {
        "settings": {
            "chunk_strategy": args.chunk_strategy,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "vector_top_k": args.vector_top_k,
            "bm25_top_k": args.bm25_top_k,
            "final_k": args.final_k,
        },
        "embedding": benchmark_embedding_speed(sample_texts),
        "retrieval": benchmark_retrieval_pipeline(
            questions=questions,
            chunk_strategy=args.chunk_strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            final_k=args.final_k,
        ),
    }

    print(json.dumps(results, indent=2))
    print_resume_lines(results)


if __name__ == "__main__":
    main()
