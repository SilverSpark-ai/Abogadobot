import os
import sys
import uuid
import hashlib
import argparse

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv()

RAW_DIR = "data/raw"
DOCS_PATH = "data/db/documents.parquet"
CHUNKS_PATH = "data/db/chunks.parquet"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks_local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
DEFAULT_BATCH_SIZE = 64


def fail(msg: str):
    print(f"❌ {msg}")
    sys.exit(1)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def read_txt(path: str) -> str:
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def ingest_documents():
    if not os.path.exists(RAW_DIR):
        fail(f"No existe la carpeta raw: {RAW_DIR}")

    docs = []
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".txt")]

    print(f"📂 TXT encontrados: {len(files)}")

    for file_name in tqdm(files, desc="Ingestando"):
        full_path = os.path.join(RAW_DIR, file_name)
        content = read_txt(full_path).strip()

        if not content:
            continue

        docs.append({
            "doc_id": file_name,
            "source_file": file_name,
            "content": content,
            "char_count": len(content),
            "content_hash": sha256_text(content),
        })

    df = pd.DataFrame(docs)
    os.makedirs("data/db", exist_ok=True)
    df.to_parquet(DOCS_PATH, index=False)

    print(f"✅ Documentos guardados en: {DOCS_PATH}")
    print(f"📄 Total documentos: {len(df)}")


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


def chunk_documents(chunk_size: int, overlap: int):
    if not os.path.exists(DOCS_PATH):
        fail(f"No existe {DOCS_PATH}. Corre primero la ingesta.")

    df = pd.read_parquet(DOCS_PATH)
    if df.empty:
        fail("documents.parquet está vacío")

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        doc_chunks = chunk_text(row["content"], chunk_size, overlap)

        for idx, chunk in enumerate(doc_chunks):
            chunk_id_raw = f"{row['doc_id']}::{idx}::{sha256_text(chunk)}"
            rows.append({
                "chunk_id": sha256_text(chunk_id_raw),
                "doc_id": row["doc_id"],
                "chunk_index": idx,
                "text": chunk,
                "char_count": len(chunk),
                "content_hash": sha256_text(chunk),
            })

    chunk_df = pd.DataFrame(rows)
    chunk_df.to_parquet(CHUNKS_PATH, index=False)

    print(f"✅ Chunks guardados en: {CHUNKS_PATH}")
    print(f"🧩 Total chunks: {len(chunk_df)}")


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


def index_chunks(limit: int | None, batch_size: int):
    if not os.path.exists(CHUNKS_PATH):
        fail(f"No existe {CHUNKS_PATH}. Corre primero chunking.")

    df = pd.read_parquet(CHUNKS_PATH)
    if df.empty:
        fail("chunks.parquet está vacío")

    if limit is not None:
        df = df.head(limit).copy()

    print(f"📦 Chunks a indexar: {len(df)}")
    print(f"🤖 Modelo embeddings: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    qdrant_client = QdrantClient(url=QDRANT_URL)

    first_batch = df.iloc[: min(batch_size, len(df))]
    first_vectors = model.encode(
        first_batch["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    if len(first_vectors) == 0:
        fail("No se generaron embeddings en el primer lote")

    vector_size = len(first_vectors[0])
    print(f"📏 Tamaño vector: {vector_size}")

    ensure_collection(qdrant_client, vector_size)

    all_points = []

    for start_idx in tqdm(range(0, len(df), batch_size), desc="Embedding"):
        batch_df = df.iloc[start_idx:start_idx + batch_size]
        texts = batch_df["text"].tolist()

        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        for row, vector in zip(batch_df.to_dict("records"), vectors):
            all_points.append(
                rest.PointStruct(
                    id=qdrant_point_id(row["chunk_id"]),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "chunk_index": int(row["chunk_index"]),
                        "text": row["text"],
                        "char_count": int(row["char_count"]),
                        "content_hash": row["content_hash"],
                    },
                )
            )

    print("📡 Subiendo a Qdrant...")
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=all_points,
    )

    print(f"✅ Indexado completo en '{QDRANT_COLLECTION}'")
    print(f"📌 Total puntos: {len(all_points)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline unificado del corpus")
    parser.add_argument("--modo", choices=["todo", "ingest", "chunk", "index"], default="todo")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--limit", type=int, default=None, help="Límite de chunks para pruebas en indexado")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.modo in ["todo", "ingest"]:
        print("\n=== ETAPA 1: INGESTA ===")
        ingest_documents()

    if args.modo in ["todo", "chunk"]:
        print("\n=== ETAPA 2: CHUNKING ===")
        chunk_documents(args.chunk_size, args.overlap)

    if args.modo in ["todo", "index"]:
        print("\n=== ETAPA 3: INDEXADO ===")
        index_chunks(args.limit, args.batch_size)


if __name__ == "__main__":
    main()