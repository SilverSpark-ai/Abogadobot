import os
import sys
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from openai import OpenAI

# Cargar variables de entorno desde .env en la raíz del proyecto
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INPUT_PATH = "data/db/chunks.parquet"

# Cambia esto a None cuando quieras indexar TODO
TEST_LIMIT = 100

BATCH_SIZE = 100


def fail(msg: str):
    print(f"❌ {msg}")
    sys.exit(1)


def ensure_env():
    if not OPENAI_API_KEY:
        fail("No se encontró OPENAI_API_KEY en el archivo .env")


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


def create_clients():
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL)
    return openai_client, qdrant_client


def embed_texts(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


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
    print("🚀 Iniciando indexado semántico...")
    ensure_env()

    print(f"📂 Leyendo chunks desde: {INPUT_PATH}")
    df = load_chunks()
    print(f"📦 Chunks a procesar: {len(df)}")

    openai_client, qdrant_client = create_clients()

    print("🧪 Generando embeddings del primer lote para detectar tamaño del vector...")
    first_batch = df.iloc[:BATCH_SIZE]
    first_texts = first_batch["text"].tolist()
    first_vectors = embed_texts(openai_client, first_texts)

    if not first_vectors:
        fail("No se generaron embeddings en el primer lote")

    vector_size = len(first_vectors[0])
    print(f"📏 Tamaño del vector detectado: {vector_size}")

    ensure_collection(qdrant_client, vector_size)

    all_points = []

    print("🧠 Generando embeddings por lotes...")
    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding"):
        batch_df = df.iloc[start_idx:start_idx + BATCH_SIZE]
        texts = batch_df["text"].tolist()

        vectors = embed_texts(openai_client, texts)

        for row, vector in zip(batch_df.to_dict("records"), vectors):
            point = rest.PointStruct(
                id=row["chunk_id"],
                vector=vector,
                payload={
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

    print("✅ Indexado completo")
    print(f"📌 Colección: {QDRANT_COLLECTION}")
    print(f"📦 Total puntos subidos: {len(all_points)}")


if __name__ == "__main__":
    main()