from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from database import create_tables, get_documents, optimize_pgvector_indexes, save_document
from embedding_engine import (
    answer_question,
    index_document,
    semantic_search,
)
from extractors import extract_pdf_text, extract_website_text, transcribe_youtube


APP_DIR = Path(__file__).parent
YOUTUBE_OUTPUT_DIR = APP_DIR / "youtube_output"


st.set_page_config(page_title="Knowledge Collector", page_icon="KB", layout="wide")

try:
    create_tables()
except SQLAlchemyError as error:
    st.error("Database setup failed. Check PostgreSQL, your DATABASE_URL, and pgvector installation.")
    st.code(str(error))
    st.stop()

st.title("Knowledge Collector")
st.write(
    "Extract text from PDFs, websites, and YouTube videos, then save it into PostgreSQL "
    "and make it searchable with embeddings + pgvector."
)

with st.sidebar:
    st.header("Vector Settings")
    chunk_strategy = st.selectbox(
        "Chunking strategy",
        ["recursive", "fixed", "paragraph"],
        help="Recursive is the best default because it tries to keep paragraphs and sentences together.",
    )
    chunk_size = st.slider("Chunk size", min_value=300, max_value=2000, value=1000, step=100)
    overlap = st.slider("Overlap", min_value=0, max_value=500, value=200, step=50)
    st.caption("These settings control how text is split before embeddings are created.")

tab_pdf, tab_website, tab_youtube, tab_qa, tab_search, tab_saved = st.tabs(
    ["PDF", "Website", "YouTube", "Q&A", "Search", "Saved Data"]
)


with tab_pdf:
    st.header("PDF to Text")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("Extract PDF Text", disabled=uploaded_pdf is None):
        with st.spinner("Reading PDF..."):
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_pdf.read())
                temp_pdf_path = Path(temp_pdf.name)

            text = extract_pdf_text(temp_pdf_path)
            document = save_document("pdf", uploaded_pdf.name, text)

        st.success(f"Saved PDF text to PostgreSQL with ID {document.id}.")
        st.text_area("Extracted Text", text, height=300)

        with st.spinner("Creating chunks and embeddings..."):
            chunk_count = index_document(document.id, chunk_strategy, chunk_size, overlap)
        st.success(f"Indexed {chunk_count} chunks in pgvector.")
        st.info(optimize_pgvector_indexes()["message"])


with tab_website:
    st.header("Website Article to Text")
    website_url = st.text_input("Website URL")

    if st.button("Extract Website Text", disabled=not website_url):
        with st.spinner("Reading website..."):
            text = extract_website_text(website_url)

            if not text:
                st.error("No readable text was found on this page.")
            else:
                document = save_document("website", website_url, text)
                st.success(f"Saved website text to PostgreSQL with ID {document.id}.")
                st.text_area("Extracted Text", text, height=300)

                with st.spinner("Creating chunks and embeddings..."):
                    chunk_count = index_document(document.id, chunk_strategy, chunk_size, overlap)
                st.success(f"Indexed {chunk_count} chunks in pgvector.")
                st.info(optimize_pgvector_indexes()["message"])


with tab_youtube:
    st.header("YouTube to Transcript")
    youtube_url = st.text_input("YouTube URL")

    if st.button("Download and Transcribe", disabled=not youtube_url):
        with st.spinner("Downloading audio and transcribing with Whisper. This can take a while..."):
            text = transcribe_youtube(youtube_url, YOUTUBE_OUTPUT_DIR)
            document = save_document("youtube", youtube_url, text)

        st.success(f"Saved YouTube transcript to PostgreSQL with ID {document.id}.")
        st.text_area("Transcript", text, height=300)

        with st.spinner("Creating chunks and embeddings..."):
            chunk_count = index_document(document.id, chunk_strategy, chunk_size, overlap)
        st.success(f"Indexed {chunk_count} chunks in pgvector.")
        st.info(optimize_pgvector_indexes()["message"])


with tab_qa:
    st.header("RAG Q&A")
    st.write("Runs vector search + BM25, reranks the top chunks, then returns a cited answer.")
    question = st.text_input("Ask a question about your indexed documents")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        vector_top_k = st.number_input("Vector top-k", min_value=1, max_value=20, value=8)
    with col_b:
        bm25_top_k = st.number_input("BM25 top-k", min_value=1, max_value=20, value=8)
    with col_c:
        final_k = st.number_input("Final reranked chunks", min_value=1, max_value=10, value=5)

    if st.button("Run RAG Pipeline", disabled=not question):
        with st.spinner("Retrieving, reranking, and building a cited answer..."):
            qa_result = answer_question(
                question,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                vector_top_k=vector_top_k,
                bm25_top_k=bm25_top_k,
                final_k=final_k,
            )

        st.subheader("Answer")
        st.write(qa_result["answer"])
        st.caption(
            f'Provider: {qa_result["llm_provider"]} | '
            f'Vector hits: {qa_result["vector_count"]} | '
            f'BM25 hits: {qa_result["bm25_count"]} | '
            f'Merged candidates: {qa_result["candidate_count"]}'
        )

        st.subheader("Sources")
        for chunk in qa_result["reranked_chunks"]:
            with st.expander(
                f'[{chunk["rank"]}] {chunk["source_type"]} | {chunk["source_name"]} | '
                f'rerank score {chunk["score"]:.3f}'
            ):
                st.caption(f'Document ID {chunk["document_id"]}, chunk {chunk["chunk_index"]}')
                st.write(chunk["chunk_text"])


with tab_search:
    st.header("Semantic Search")
    query = st.text_input("Ask a question or search by meaning")
    result_limit = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("Search Knowledge Base", disabled=not query):
        with st.spinner("Embedding your query and searching similar chunks..."):
            results = semantic_search(query, chunk_strategy=chunk_strategy, limit=result_limit)

        if not results:
            st.info("No vector results found yet. Add a PDF, website, or YouTube transcript first.")
        else:
            for result in results:
                similarity = 1 - float(result["distance"])
                with st.expander(
                    f'{result["source_type"]} | {result["source_name"]} | similarity {similarity:.3f}'
                ):
                    st.caption(
                        f'Document ID {result["document_id"]}, chunk {result["chunk_index"]}, '
                        f'distance {float(result["distance"]):.4f}'
                    )
                    st.write(result["chunk_text"])


with tab_saved:
    st.header("Saved Data in PostgreSQL")
    documents = get_documents()

    if not documents:
        st.info("No saved documents yet.")
    else:
        for document in documents:
            with st.expander(f"{document.id} | {document.source_type} | {document.source_name}"):
                st.caption(f"Saved at: {document.created_at}")
                st.text_area(
                    "Stored Text",
                    document.extracted_text,
                    height=250,
                    key=f"document-{document.id}",
                )
