import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.documents import Document as LangChainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from chunking import make_text_splitter
from database import DATABASE_URL, get_document, get_documents


load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_PREFIX = os.getenv("PGVECTOR_COLLECTION_PREFIX", "knowledge_collector")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "180"))
RERANK_CANDIDATE_LIMIT = int(os.getenv("RERANK_CANDIDATE_LIMIT", "8"))


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    # HuggingFaceEmbeddings runs locally, so indexing/search no longer needs a paid embeddings API key.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=8)
def get_vector_store(chunk_strategy: str = "recursive") -> PGVector:
    # PGVector is unchanged: LangChain still stores embeddings in PostgreSQL + pgvector.
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=f"{COLLECTION_PREFIX}_{chunk_strategy}",
        connection=DATABASE_URL,
        use_jsonb=True,
    )


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    # Loading the cross-encoder once avoids counting model startup time in every query.
    return CrossEncoder(RERANKER_MODEL)


def split_document_with_langchain(
    document_id: int,
    chunk_strategy: str = "recursive",
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[LangChainDocument]:
    document = get_document(document_id)

    if document is None:
        raise ValueError(f"Document {document_id} was not found.")

    splitter = make_text_splitter(
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    source_document = LangChainDocument(
        page_content=document.extracted_text,
        metadata={
            "document_id": document.id,
            "source_type": document.source_type,
            "source_name": document.source_name,
            "chunk_strategy": chunk_strategy,
        },
    )

    chunks = splitter.split_documents([source_document])

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        # Store the local embedding model name so retrieved chunks show how they were indexed.
        chunk.metadata["embedding_model"] = EMBEDDING_MODEL

    return chunks


def index_document(
    document_id: int,
    chunk_strategy: str = "recursive",
    chunk_size: int = 1000,
    overlap: int = 200,
) -> int:
    chunks = split_document_with_langchain(
        document_id=document_id,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        return 0

    vector_store = get_vector_store(chunk_strategy)
    ids = [f"{document_id}-{chunk_strategy}-{index}" for index in range(len(chunks))]

    vector_store.delete(ids=ids)
    vector_store.add_documents(chunks, ids=ids)
    clear_retrieval_caches()

    return len(chunks)


def semantic_search(
    query: str,
    chunk_strategy: str = "recursive",
    limit: int = 5,
) -> list[dict]:
    vector_store = get_vector_store(chunk_strategy)
    results = vector_store.similarity_search_with_score(query=query, k=limit)

    formatted_results = []
    for chunk, score in results:
        formatted_results.append(
            {
                "document_id": chunk.metadata.get("document_id"),
                "source_type": chunk.metadata.get("source_type"),
                "source_name": chunk.metadata.get("source_name"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "chunk_text": chunk.page_content,
                "distance": score,
            }
        )

    return formatted_results


def tokenize_for_bm25(text: str) -> list[str]:
    # BM25 works on keyword tokens, so this normalizes text for keyword retrieval.
    return re.findall(r"\b\w+\b", text.lower())


@lru_cache(maxsize=16)
def load_chunk_documents(
    chunk_strategy: str = "recursive",
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[LangChainDocument]:
    # BM25 needs plain text chunks, so this rebuilds chunks from the saved source documents.
    splitter = make_text_splitter(
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    chunk_documents = []

    for document in get_documents():
        source_document = LangChainDocument(
            page_content=document.extracted_text,
            metadata={
                "document_id": document.id,
                "source_type": document.source_type,
                "source_name": document.source_name,
                "chunk_strategy": chunk_strategy,
            },
        )
        chunks = splitter.split_documents([source_document])

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk_documents.append(chunk)

    return chunk_documents


@lru_cache(maxsize=16)
def get_bm25_index(
    chunk_strategy: str = "recursive",
    chunk_size: int = 1000,
    overlap: int = 200,
) -> tuple[list[LangChainDocument], BM25Okapi | None]:
    # Building BM25 over all chunks is expensive, so cache it until new documents are indexed.
    chunks = load_chunk_documents(
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        return chunks, None

    tokenized_chunks = [tokenize_for_bm25(chunk.page_content) for chunk in chunks]
    return chunks, BM25Okapi(tokenized_chunks)


def clear_retrieval_caches() -> None:
    # New or re-indexed documents change the retrieval corpus, so cached BM25 chunks must reset.
    load_chunk_documents.cache_clear()
    get_bm25_index.cache_clear()


def vector_retrieve(query: str, chunk_strategy: str, top_k: int) -> list[LangChainDocument]:
    # Vector search retrieves semantically similar chunks from PostgreSQL + pgvector.
    vector_store = get_vector_store(chunk_strategy)
    return vector_store.similarity_search(query=query, k=top_k)


def bm25_retrieve(
    query: str,
    chunk_strategy: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
) -> list[LangChainDocument]:
    # BM25 complements vector search by catching exact keywords, names, and acronyms.
    chunks, bm25 = get_bm25_index(
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks or bm25 is None:
        return []

    scores = bm25.get_scores(tokenize_for_bm25(query))
    ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    return [chunks[index] for index in ranked_indexes[:top_k] if scores[index] > 0]


def merge_candidates(candidate_groups: list[list[LangChainDocument]]) -> list[LangChainDocument]:
    # Hybrid retrieval can return duplicates, so this keeps one copy of each source chunk.
    merged = {}

    for group in candidate_groups:
        for chunk in group:
            key = (
                chunk.metadata.get("document_id"),
                chunk.metadata.get("chunk_index"),
                chunk.metadata.get("source_name"),
            )
            merged.setdefault(key, chunk)

    return list(merged.values())


def rerank_chunks(
    query: str,
    chunks: list[LangChainDocument],
    final_k: int,
) -> list[dict]:
    # The reranker scores query + chunk pairs, improving the final top-k context quality.
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(query, chunk.page_content) for chunk in chunks]
    scores = reranker.predict(pairs)
    scored_chunks = sorted(
        zip(chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        {
            "rank": index + 1,
            "score": float(score),
            "document_id": chunk.metadata.get("document_id"),
            "source_type": chunk.metadata.get("source_type"),
            "source_name": chunk.metadata.get("source_name"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "chunk_text": chunk.page_content,
        }
        for index, (chunk, score) in enumerate(scored_chunks[:final_k])
    ]


def build_context(reranked_chunks: list[dict], max_chars_per_chunk: int = 900) -> str:
    # Citation labels are embedded into context so generated answers can cite sources.
    context_blocks = []

    for chunk in reranked_chunks:
        citation = f'[{chunk["rank"]}] {chunk["source_type"]}: {chunk["source_name"]}'
        chunk_text = chunk["chunk_text"].replace("\n", " ").strip()
        context_blocks.append(f"{citation}\n{chunk_text[:max_chars_per_chunk]}")

    return "\n\n".join(context_blocks)


def generate_extractive_answer(reranked_chunks: list[dict]) -> str:
    # This free local fallback returns cited evidence without calling a paid LLM.
    if not reranked_chunks:
        return "I could not find enough relevant information in the indexed documents."

    lines = ["Most relevant retrieved evidence:", ""]

    for chunk in reranked_chunks[:3]:
        snippet = chunk["chunk_text"].replace("\n", " ").strip()
        lines.append(f'[{chunk["rank"]}] {snippet[:700]}')

    lines.append("")
    lines.append("Set LLM_PROVIDER=openai to generate a GPT-4o-mini answer from these cited chunks.")
    return "\n".join(lines)


def generate_openai_answer(question: str, reranked_chunks: list[dict]) -> str:
    # GPT-4o-mini is optional and only runs when LLM_PROVIDER=openai plus OPENAI_API_KEY are set.
    from langchain_openai import ChatOpenAI

    context = build_context(reranked_chunks)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    prompt = (
        "Answer the question using only the context below. "
        "Cite every factual claim with source numbers like [1] or [2]. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}"
    )
    response = llm.invoke(prompt)
    return response.content


def generate_ollama_answer(question: str, reranked_chunks: list[dict]) -> str:
    from langchain_ollama import ChatOllama

    context = build_context(reranked_chunks, max_chars_per_chunk=700)

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=OLLAMA_NUM_PREDICT,
    )

    prompt = (
        "Answer using only the context. Be concise. "
        "Use source citations like [1]. "
        "If the answer is not present, say you do not know.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}"
    )

    response = llm.invoke(prompt)

    return response.content


def answer_question(
    question: str,
    chunk_strategy: str = "recursive",
    chunk_size: int = 1000,
    overlap: int = 200,
    vector_top_k: int = 8,
    bm25_top_k: int = 8,
    final_k: int = 5,
) -> dict:
    # Week 3 RAG pipeline: query -> vector + BM25 retrieval -> rerank -> answer with citations.
    vector_chunks = vector_retrieve(question, chunk_strategy, vector_top_k)
    bm25_chunks = bm25_retrieve(question, chunk_strategy, chunk_size, overlap, bm25_top_k)
    candidates = merge_candidates([vector_chunks, bm25_chunks])[:RERANK_CANDIDATE_LIMIT]
    reranked_chunks = rerank_chunks(question, candidates, final_k)

    if LLM_PROVIDER == "openai":
        answer = generate_openai_answer(question, reranked_chunks)

    elif LLM_PROVIDER == "ollama":
        answer = generate_ollama_answer(question, reranked_chunks)

    else:
        answer = generate_extractive_answer(reranked_chunks)

    return {
        "answer": answer,
        "reranked_chunks": reranked_chunks,
        "candidate_count": len(candidates),
        "vector_count": len(vector_chunks),
        "bm25_count": len(bm25_chunks),
        "llm_provider": LLM_PROVIDER,
    }
