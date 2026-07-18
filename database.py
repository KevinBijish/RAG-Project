import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5433/knowledge_collector",
).replace(
    "postgresql+psycopg2://",
    "postgresql+psycopg://",
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(30), nullable=False)
    source_name = Column(String(500), nullable=False)
    extracted_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def create_tables() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)


def optimize_pgvector_indexes() -> dict:
    """Create pgvector indexes used by LangChain's PGVector tables when they exist."""
    results = {
        "hnsw_created": False,
        "ivfflat_created": False,
        "message": "",
    }

    with engine.begin() as connection:
        table_exists = connection.execute(
            text("SELECT to_regclass('public.langchain_pg_embedding') IS NOT NULL")
        ).scalar()

    if not table_exists:
        results["message"] = "LangChain PGVector table does not exist yet. Index documents first."
        return results

    try:
        with engine.begin() as connection:
            # LangChain creates a generic vector column. pgvector indexes need fixed dimensions.
            connection.execute(
                text(
                    f"""
                    ALTER TABLE langchain_pg_embedding
                    ALTER COLUMN embedding TYPE vector({EMBEDDING_DIMENSION})
                    USING embedding::vector({EMBEDDING_DIMENSION})
                    """
                )
            )
            # Recreate our indexes after the column type is normalized.
            connection.execute(text("DROP INDEX IF EXISTS langchain_pg_embedding_hnsw_idx"))
            connection.execute(text("DROP INDEX IF EXISTS langchain_pg_embedding_ivfflat_idx"))
    except Exception as error:
        results["message"] = (
            "Could not convert embeddings to a fixed vector dimension. "
            "If old embeddings used another model, delete/re-index them. "
            f"Details: {error}"
        )
        return results

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw_idx
                    ON langchain_pg_embedding
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            )
            connection.execute(text("ANALYZE langchain_pg_embedding"))

        results["hnsw_created"] = True
        results["message"] = "HNSW index is ready."
        return results

    except Exception as hnsw_error:
        try:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        CREATE INDEX IF NOT EXISTS langchain_pg_embedding_ivfflat_idx
                        ON langchain_pg_embedding
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                        """
                    )
                )
                connection.execute(text("ANALYZE langchain_pg_embedding"))

            results["ivfflat_created"] = True
            results["message"] = "IVFFlat index is ready. HNSW was not available."
            return results
        except Exception as ivfflat_error:
            results["message"] = (
                "Could not create a pgvector index. Retrieval still works, but may be slower. "
                f"HNSW error: {hnsw_error}. IVFFlat error: {ivfflat_error}"
            )
            return results


def save_document(source_type: str, source_name: str, extracted_text: str) -> Document:
    session = SessionLocal()
    try:
        document = Document(
            source_type=source_type,
            source_name=source_name,
            extracted_text=extracted_text,
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        return document
    finally:
        session.close()


def get_documents() -> list[Document]:
    session = SessionLocal()
    try:
        return session.query(Document).order_by(Document.created_at.desc()).all()
    finally:
        session.close()


def get_document(document_id: int) -> Document | None:
    session = SessionLocal()
    try:
        return session.query(Document).filter(Document.id == document_id).first()
    finally:
        session.close()
