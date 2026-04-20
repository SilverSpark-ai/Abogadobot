import os
import sys
import uuid
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    select,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.exc import OperationalError


# =========================================================
# Paths y entorno
# =========================================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR = Path(os.getenv("RAW_DIR", str(DEFAULT_RAW_DIR))).resolve()
DATABASE_URL = os.getenv("DATABASE_URL")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks_local")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_DOC_BATCH = 25

if not DATABASE_URL:
    print("❌ Falta DATABASE_URL en el entorno", file=sys.stderr)
    sys.exit(1)

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)
Base = declarative_base()


# =========================================================
# Helpers
# =========================================================
def fail(msg: str):
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)


def utcnow():
    return datetime.now(timezone.utc)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def read_txt(path: Path) -> str:
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start += max(1, chunk_size - overlap)

    return chunks


def new_session():
    return SessionLocal()


def safe_commit(session, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            session.commit()
            return
        except OperationalError as e:
            last_error = e
            print(f"⚠️ Error de conexión en commit. Reintentando ({attempt + 1}/{retries})...")
            try:
                session.rollback()
            except Exception:
                pass
        except Exception:
            try:
                session.rollback()
            except Exception:
                pass
            raise
    raise last_error


# =========================================================
# Models
# =========================================================
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    doc_id = Column(String(255), nullable=False, unique=True, index=True)
    source_file = Column(String(255), nullable=False)
    content = Column(LONGTEXT, nullable=False)
    char_count = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="ingested", index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_chunks_chunk_id"),
    )

    id = Column(Integer, primary_key=True)
    chunk_id = Column(String(64), nullable=False, index=True)
    doc_id = Column(String(255), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(LONGTEXT, nullable=False)
    char_count = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    is_indexed = Column(Boolean, nullable=False, default=False, index=True)
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    document = relationship("Document", back_populates="chunks")


# =========================================================
# DB
# =========================================================
def test_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Conexión a MySQL OK")
    except Exception as e:
        fail(f"No se pudo conectar a MySQL: {e}")


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("✅ Tablas verificadas/creadas")


def fix_longtext():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE documents MODIFY content LONGTEXT NOT NULL;"))
        conn.execute(text("ALTER TABLE chunks MODIFY text LONGTEXT NOT NULL;"))
    print("✅ DB actualizada: content/text ahora son LONGTEXT")


# =========================================================
# ETAPA 1: INGESTA
# =========================================================
def ingest_documents():
    if not RAW_DIR.exists():
        fail(f"No existe la carpeta raw: {RAW_DIR}")

    files = [f for f in RAW_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    print(f"📂 TXT encontrados: {len(files)}")
    print(f"📍 RAW_DIR: {RAW_DIR}")

    if not files:
        print("ℹ️ No hay TXT para ingerir")
        return

    session = new_session()

    try:
        for file_path in tqdm(files, desc="Ingestando"):
            file_name = file_path.name
            content = read_txt(file_path).strip()

            if not content:
                continue

            content_hash = sha256_text(content)
            char_count = len(content)

            existing = session.execute(
                select(Document).where(Document.doc_id == file_name)
            ).scalar_one_or_none()

            if existing:
                if existing.content_hash == content_hash:
                    continue

                existing.source_file = file_name
                existing.content = content
                existing.char_count = char_count
                existing.content_hash = content_hash
                existing.status = "ingested"
                existing.updated_at = utcnow()

                session.query(Chunk).filter(
                    Chunk.doc_id == existing.doc_id
                ).delete(synchronize_session=False)

            else:
                session.add(
                    Document(
                        doc_id=file_name,
                        source_file=file_name,
                        content=content,
                        char_count=char_count,
                        content_hash=content_hash,
                        status="ingested",
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                )

            safe_commit(session)

    finally:
        session.close()

    print("✅ Ingesta completada en SQL")


# =========================================================
# ETAPA 2A: CHUNKING DE UN CONJUNTO ESPECÍFICO
# =========================================================
def chunk_specific_documents(doc_ids: list[str], chunk_size: int, overlap: int):
    if not doc_ids:
        print("ℹ️ No hay documentos para chunkear en este lote")
        return 0

    with new_session() as session:
        rows = session.execute(
            select(Document.doc_id, Document.content).where(Document.doc_id.in_(doc_ids))
        ).all()

    total_docs_ok = 0

    for doc_id, content in tqdm(rows, desc="Chunking por documento"):
        session = new_session()
        try:
            doc_chunks = chunk_text(content, chunk_size, overlap)

            # Rehacemos chunks del documento completo
            session.query(Chunk).filter(Chunk.doc_id == doc_id).delete(synchronize_session=False)

            chunk_objects = []
            for idx, chunk in enumerate(doc_chunks):
                chunk_id_raw = f"{doc_id}::{idx}::{sha256_text(chunk)}"
                chunk_id = sha256_text(chunk_id_raw)

                chunk_objects.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        chunk_index=idx,
                        text=chunk,
                        char_count=len(chunk),
                        content_hash=sha256_text(chunk),
                        is_indexed=False,
                        created_at=utcnow(),
                    )
                )

            session.bulk_save_objects(chunk_objects)

            doc_row = session.execute(
                select(Document).where(Document.doc_id == doc_id)
            ).scalar_one_or_none()

            if doc_row:
                doc_row.status = "chunked"
                doc_row.updated_at = utcnow()

            safe_commit(session)
            total_docs_ok += 1

        except OperationalError:
            try:
                session.rollback()
            except Exception:
                pass

            try:
                err_session = new_session()
                doc_row = err_session.execute(
                    select(Document).where(Document.doc_id == doc_id)
                ).scalar_one_or_none()
                if doc_row:
                    doc_row.status = "chunk_error"
                    doc_row.updated_at = utcnow()
                    safe_commit(err_session)
                err_session.close()
            except Exception:
                pass

            print(f"⚠️ Conexión caída durante chunking de {doc_id}. Se podrá retomar luego.")
            session.close()
            raise

        except Exception as e:
            try:
                session.rollback()
            except Exception:
                pass

            try:
                err_session = new_session()
                doc_row = err_session.execute(
                    select(Document).where(Document.doc_id == doc_id)
                ).scalar_one_or_none()
                if doc_row:
                    doc_row.status = "chunk_error"
                    doc_row.updated_at = utcnow()
                    safe_commit(err_session)
                err_session.close()
            except Exception:
                pass

            print(f"⚠️ Error en documento {doc_id}: {e}")

        finally:
            session.close()

    return total_docs_ok


# =========================================================
# ETAPA 2B: CHUNKING DE UNA TANDA
# =========================================================
def chunk_documents(chunk_size: int, overlap: int, only_pending: bool = True, limit_docs: int | None = None):
    with new_session() as session:
        stmt = select(Document.doc_id).order_by(Document.id)
        if only_pending:
            stmt = stmt.where(Document.status.in_(["ingested", "chunk_error"]))

        pending_doc_ids = session.execute(stmt).scalars().all()

    if not pending_doc_ids:
        print("ℹ️ No hay documentos pendientes para chunking")
        return

    if limit_docs is not None:
        pending_doc_ids = pending_doc_ids[:limit_docs]

    print(f"📄 Documentos a chunkear en esta corrida: {len(pending_doc_ids)}")
    total_docs_ok = chunk_specific_documents(
        doc_ids=pending_doc_ids,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    print(f"✅ Chunking completado en esta corrida. Docs procesados: {total_docs_ok}")


# =========================================================
# ETAPA 2C: CHUNKING AUTOMÁTICO EN LOOP
# =========================================================
def chunk_documents_loop(chunk_size: int, overlap: int, batch_docs: int = 25, only_pending: bool = True):
    total_rounds = 0
    total_docs_processed = 0

    while True:
        with new_session() as session:
            stmt = select(Document.doc_id).order_by(Document.id)
            if only_pending:
                stmt = stmt.where(Document.status.in_(["ingested", "chunk_error"]))

            pending_doc_ids = session.execute(stmt).scalars().all()

        if not pending_doc_ids:
            print("✅ No quedan documentos pendientes para chunking")
            break

        current_batch = pending_doc_ids[:batch_docs]

        print(f"\n🔁 Lote #{total_rounds + 1}")
        print(f"📄 Documentos en este lote: {len(current_batch)}")
        print(f"📚 Pendientes antes del lote: {len(pending_doc_ids)}")

        processed = chunk_specific_documents(
            doc_ids=current_batch,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        total_rounds += 1
        total_docs_processed += processed

        print(f"✅ Lote #{total_rounds} terminado. Docs procesados en el lote: {processed}")
        show_stats()

    print(f"\n✅ Chunking automático finalizado")
    print(f"🔁 Total de lotes: {total_rounds}")
    print(f"📄 Total docs procesados: {total_docs_processed}")


# =========================================================
# Qdrant
# =========================================================
def ensure_collection(qdrant_client: QdrantClient, vector_size: int):
    existing = qdrant_client.get_collections().collections
    names = [c.name for c in existing]

    if QDRANT_COLLECTION in names:
        print(f"ℹ️ La colección '{QDRANT_COLLECTION}' ya existe")
        return

    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        ),
    )
    print(f"✅ Colección '{QDRANT_COLLECTION}' creada")


# =========================================================
# ETAPA 3: INDEXADO
# =========================================================
def index_chunks(limit: int | None, batch_size: int, reindex: bool = False):
    with new_session() as session:
        stmt = select(Chunk).order_by(Chunk.id)
        if not reindex:
            stmt = stmt.where(Chunk.is_indexed.is_(False))

        chunks = session.execute(stmt).scalars().all()

    if limit is not None:
        chunks = chunks[:limit]

    if not chunks:
        print("ℹ️ No hay chunks pendientes para indexar")
        return

    print(f"📦 Chunks a indexar: {len(chunks)}")
    print(f"🤖 Modelo embeddings: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    qdrant_client = QdrantClient(url=QDRANT_URL)

    first_batch = chunks[: min(batch_size, len(chunks))]
    first_texts = [c.text for c in first_batch]

    first_vectors = model.encode(
        first_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    if len(first_vectors) == 0:
        fail("No se generaron embeddings en el primer lote")

    vector_size = len(first_vectors[0])
    print(f"📏 Tamaño vector: {vector_size}")

    ensure_collection(qdrant_client, vector_size)

    session = new_session()
    try:
        for start_idx in tqdm(range(0, len(chunks), batch_size), desc="Embedding + upsert"):
            batch = chunks[start_idx:start_idx + batch_size]
            texts = [c.text for c in batch]

            vectors = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            points = []
            for chunk, vector in zip(batch, vectors):
                points.append(
                    rest.PointStruct(
                        id=qdrant_point_id(chunk.chunk_id),
                        vector=vector.tolist(),
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "chunk_index": int(chunk.chunk_index),
                            "text": chunk.text,
                            "char_count": int(chunk.char_count),
                            "content_hash": chunk.content_hash,
                        },
                    )
                )

            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points,
            )

            indexed_at = utcnow()
            chunk_ids = [c.chunk_id for c in batch]
            doc_ids = {c.doc_id for c in batch}

            db_chunks = session.execute(
                select(Chunk).where(Chunk.chunk_id.in_(chunk_ids))
            ).scalars().all()

            for db_chunk in db_chunks:
                db_chunk.is_indexed = True
                db_chunk.indexed_at = indexed_at

            safe_commit(session)

            for doc_id in doc_ids:
                remaining = session.execute(
                    select(Chunk).where(
                        Chunk.doc_id == doc_id,
                        Chunk.is_indexed.is_(False),
                    )
                ).scalars().first()

                if remaining is None:
                    doc = session.execute(
                        select(Document).where(Document.doc_id == doc_id)
                    ).scalar_one_or_none()
                    if doc:
                        doc.status = "indexed"
                        doc.updated_at = utcnow()

            safe_commit(session)
            session.close()
            session = new_session()

    finally:
        session.close()

    print(f"✅ Indexado completo en '{QDRANT_COLLECTION}'")


# =========================================================
# Stats
# =========================================================
def show_stats():
    with new_session() as session:
        doc_count = session.query(Document).count()
        chunk_count = session.query(Chunk).count()
        pending_chunks = session.query(Chunk).filter(Chunk.is_indexed.is_(False)).count()

        ingested_docs = session.query(Document).filter(Document.status == "ingested").count()
        chunked_docs = session.query(Document).filter(Document.status == "chunked").count()
        indexed_docs = session.query(Document).filter(Document.status == "indexed").count()
        chunk_error_docs = session.query(Document).filter(Document.status == "chunk_error").count()

    print(f"📄 Documentos: {doc_count}")
    print(f"🧩 Chunks: {chunk_count}")
    print(f"⏳ Chunks pendientes de indexar: {pending_chunks}")
    print(f"📥 Docs ingested: {ingested_docs}")
    print(f"✂️ Docs chunked: {chunked_docs}")
    print(f"🧠 Docs indexed: {indexed_docs}")
    print(f"⚠️ Docs con chunk_error: {chunk_error_docs}")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline SQL + Qdrant del corpus")
    parser.add_argument(
        "--modo",
        choices=["todo", "test", "init", "fixdb", "ingest", "chunk", "chunk_loop", "index", "stats"],
        default="todo",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--limit", type=int, default=None, help="Límite de chunks para pruebas en indexado")
    parser.add_argument("--limit-docs", type=int, default=None, help="Límite de documentos en una corrida o tamaño de lote en chunk_loop")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--all-docs",
        action="store_true",
        help="Procesar todos los documentos en chunking, no solo pendientes",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindexar chunks aunque ya estén marcados como indexados",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.modo in ["todo", "test"]:
        print("\n=== TEST CONEXIÓN DB ===")
        test_connection()
        if args.modo == "test":
            return

    if args.modo in ["todo", "init"]:
        print("\n=== ETAPA 0: INIT DB ===")
        create_tables()

    if args.modo in ["todo", "fixdb"]:
        print("\n=== ETAPA 0.1: FIX LONGTEXT ===")
        fix_longtext()
        if args.modo == "fixdb":
            return

    if args.modo in ["todo", "ingest"]:
        print("\n=== ETAPA 1: INGESTA ===")
        ingest_documents()

    if args.modo == "chunk":
        print("\n=== ETAPA 2: CHUNKING ===")
        chunk_documents(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            only_pending=not args.all_docs,
            limit_docs=args.limit_docs,
        )
        return

    if args.modo == "chunk_loop":
        print("\n=== ETAPA 2B: CHUNKING AUTOMÁTICO POR LOTES ===")
        chunk_documents_loop(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            batch_docs=args.limit_docs or DEFAULT_DOC_BATCH,
            only_pending=not args.all_docs,
        )
        return

    if args.modo in ["todo", "index"]:
        print("\n=== ETAPA 3: INDEXADO ===")
        index_chunks(
            limit=args.limit,
            batch_size=args.batch_size,
            reindex=args.reindex,
        )

    if args.modo in ["todo", "stats"]:
        print("\n=== STATS ===")
        show_stats()


if __name__ == "__main__":
    main()