import os
import sys
import uuid

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

# Cargar variables de entorno desde .env en la raíz del proyecto
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks_local")

INPUT_PATH = "data/db/chunks.parquet"

# Para pruebas. Cambia a None cuando quieras indexar TODO el corpus.
TEST_LIMIT = 100

BATCH_SIZE = 64
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def fail(msg: str):
    print(f"❌ {msg}")
    sys.exit(1)


def qdrant_point_id(chunk_id: str) -> str:
    """
    Convierte el chunk_id original en un UUID determinístico válido para Qdrant.
    El mismo chunk_id siempre producirá el mismo UUID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def load_chunks() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        fail(f"No existe el archivo: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    if df.empty:
        fail("chunks.parquet está vacío")

    required_columns = {
        "chunk_id",
        "doc_id",
        "chunk_index",
        "text",
        "char_count",
        "content_hash",
    }

    missing = required_columns - set(df.columns)
    if missing:
        fail(f"Faltan columnas en chunks.parquet: {missing}")

    if TEST_LIMIT is not None:
        df = df.head(TEST_LIMIT).copy()

    return df


def ensure_collection(qdrant_client: QdrantClient, vector_size: int):
    existing = qdrant_client.get_collections().collections
    existing_names = [c.name for c in existing]

    if QDRANT_COLLECTION in existing_names:
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


def main():
    print("🚀 Iniciando indexado local...")
    print(f"🤖 Modelo local: {MODEL_NAME}")
    print(f"📂 Leyendo chunks desde: {INPUT_PATH}")
    print(f"🗂️ Colección destino: {QDRANT_COLLECTION}")

    df = load_chunks()
    print(f"📦 Chunks a procesar: {len(df)}")

    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    except Exception as e:
        fail(f"No se pudo conectar a Qdrant: {e}")

    print("📥 Cargando modelo local de embeddings...")
    model = SentenceTransformer(MODEL_NAME)

    first_batch = df.iloc[: min(BATCH_SIZE, len(df))]
    first_vectors = model.encode(
        first_batch["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    if len(first_vectors) == 0:
        fail("No se generaron embeddings en el primer lote")

    vector_size = len(first_vectors[0])
    print(f"📏 Tamaño del vector detectado: {vector_size}")

    ensure_collection(qdrant_client, vector_size)

    all_points = []

    print("🧠 Generando embeddings por lotes...")
    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding local"):
        batch_df = df.iloc[start_idx:start_idx + BATCH_SIZE]
        texts = batch_df["text"].tolist()

        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        for row, vector in zip(batch_df.to_dict("records"), vectors):
            point = rest.PointStruct(
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
            all_points.append(point)

    print("📡 Subiendo puntos a Qdrant...")
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=all_points,
    )

    print("✅ Indexado local completo")
    print(f"📌 Colección: {QDRANT_COLLECTION}")
    print(f"📦 Total puntos subidos: {len(all_points)}")


if __name__ == "__main__":
    main()