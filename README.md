## Second Brain - GEN AI Local RAG Platform

Second Brain is a Streamlit application that turns PDFs, website articles, and YouTube videos into a searchable RAG knowledge base. It extracts text, chunks documents with LangChain, stores embeddings in PostgreSQL with pgvector, and answers questions with hybrid retrieval, reranking, and source citations.

The project is designed to run locally. Embeddings use Hugging Face `sentence-transformers/all-MiniLM-L6-v2`, so the core retrieval pipeline does not require paid OpenAI embedding credits.

## What This Project Demonstrates

- ETL pipelines for PDF, website, and YouTube ingestion.
- LangChain document chunking with recursive character splitting and overlap.
- Local embedding generation with Hugging Face sentence transformers.
- PostgreSQL + pgvector vector storage and semantic similarity search.
- Hybrid retrieval using pgvector vector search plus BM25 keyword search.
- Cross-encoder reranking for better final context selection.
- Cited RAG answers through extractive mode, Ollama, or optional OpenAI.
- Separate benchmark script for latency, embedding throughput, memory, and top-k accuracy.

## Architecture

```text
PDF / Website / YouTube
        |
        v
Text extraction
        |
        v
LangChain RecursiveCharacterTextSplitter
        |
        v
HuggingFaceEmbeddings all-MiniLM-L6-v2
        |
        v
PostgreSQL + pgvector
        |
        v
Vector Search + BM25 Keyword Search
        |
        v
Cross-Encoder Reranker
        |
        v
Answer with Source Citations
```

## Tech Stack

| Layer | Tools |
|---|---|
| App UI | Streamlit |
| Orchestration | LangChain |
| PDF extraction | PyMuPDF |
| Website extraction | requests, BeautifulSoup |
| YouTube ingestion | yt-dlp, Whisper |
| Embeddings | HuggingFaceEmbeddings, `all-MiniLM-L6-v2` |
| Vector database | PostgreSQL, pgvector, LangChain PGVector |
| Keyword retrieval | BM25 |
| Reranking | sentence-transformers CrossEncoder |
| Optional answer generation | Ollama or OpenAI GPT-4o-mini |

## Main Files

```text
app.py                         Streamlit application
extractors.py                  PDF, website, and YouTube text extraction
chunking.py                    LangChain text splitting strategies
embedding_engine.py            Embeddings, retrieval, reranking, and Q&A
database.py                    PostgreSQL setup, document storage, pgvector index helper
benchmark_rag.py               Standalone benchmark runner
benchmark_questions.example.json Example labeled benchmark questions
requirements.txt               Python dependencies
```

## Features

### Document Ingestion

The app supports three source types:

```text
PDF -> PyMuPDF -> extracted text
Website URL -> BeautifulSoup -> article text
YouTube URL -> yt-dlp + Whisper -> transcript
```

Each source is saved in PostgreSQL, then indexed into pgvector.

### Chunking

Documents are split with LangChain’s `RecursiveCharacterTextSplitter`.

Default settings:

```text
chunk_size = 1000
chunk_overlap = 200
strategy = recursive
```

The UI also supports `fixed` and `paragraph` chunking modes for experimentation.

### Embeddings and Vector Storage

Embeddings are generated locally:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding dimension is:

```text
384
```

Vectors are stored in LangChain’s pgvector tables:

```text
langchain_pg_collection
langchain_pg_embedding
```

The original extracted text is stored in:

```text
documents
```

### RAG Q&A

The Q&A tab uses this retrieval pipeline:

```text
User question
-> pgvector semantic search
-> BM25 keyword search
-> merge candidate chunks
-> cross-encoder reranking
-> cited answer
```

The app returns:

- answer text
- source chunks
- document ID
- source name
- rerank score
- retrieval counts

## Setup

### 1. Install PostgreSQL with pgvector

Docker is the easiest option:

```powershell
docker run --name knowledge-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=knowledge_collector -p 5433:5432 -d pgvector/pgvector:pg16
```

If you installed PostgreSQL manually, create the database and extension:

```sql
CREATE DATABASE knowledge_collector;
```

Then connect to `knowledge_collector` and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Open the Project Folder

```cmd
cd C:\Users\kevin\Documents\Codex\2026-06-20\t\outputs\knowledge_collector_app
```

### 3. Create and Activate a Virtual Environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies

```cmd
pip install -r requirements.txt
```

### 5. Configure `.env`

Create a `.env` file in the project folder:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/knowledge_collector

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_CANDIDATE_LIMIT=8

WHISPER_MODEL=base

LLM_PROVIDER=extractive
OPENAI_MODEL=gpt-4o-mini

OLLAMA_MODEL=llama3.2:1b
OLLAMA_NUM_CTX=2048
OLLAMA_NUM_PREDICT=180
```

Use `LLM_PROVIDER=extractive` for a fully local/free evidence-based answer.

Use `LLM_PROVIDER=ollama` for local generated answers. Pull the model first:

```cmd
ollama pull llama3.2:1b
```

Use `LLM_PROVIDER=openai` only if you want GPT-4o-mini answer generation:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
```

## Run the App

```cmd
streamlit run app.py
```

Typical workflow:

```text
1. Upload a PDF, enter a website URL, or enter a YouTube URL.
2. Let the app extract text and index chunks into pgvector.
3. Open the Q&A tab.
4. Ask a question.
5. Review the cited answer and source chunks.
```
