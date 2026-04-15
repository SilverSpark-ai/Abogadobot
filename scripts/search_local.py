import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks_local")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def fail(msg: str):
    print(f"❌ {msg}")
    sys.exit(1)


def run_query(qdrant_client: QdrantClient, query_vector: list[float], limit: int = 5):
    """
    Compatibilidad con distintas versiones del cliente de Qdrant.
    Primero intenta query_points (API nueva).
    Si no existe, intenta search (API vieja).
    """
    if hasattr(qdrant_client, "query_points"):
        response = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
        )

        # Algunas versiones devuelven un objeto con .points
        if hasattr(response, "points"):
            return response.points

        # Por si devuelve lista directa
        return response

    if hasattr(qdrant_client, "search"):
        return qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=limit,
        )

    fail("Tu versión de QdrantClient no tiene ni 'query_points' ni 'search'")


def main():
    if len(sys.argv) < 2:
        fail('Uso: python app\\scripts\\search_local.py "tu consulta aquí"')

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        fail("La consulta está vacía")

    print(f"🔎 Consulta: {query}")
    print(f"🤖 Modelo local: {MODEL_NAME}")
    print(f"🗂️ Colección: {QDRANT_COLLECTION}")

    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    except Exception as e:
        fail(f"No se pudo conectar a Qdrant: {e}")

    print("📥 Cargando modelo local...")
    model = SentenceTransformer(MODEL_NAME)

    print("🧠 Generando embedding de la consulta...")
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    print("📡 Buscando en Qdrant...")
    results = run_query(qdrant_client, query_vector, limit=5)

    if not results:
        print("⚠️ No se encontraron resultados")
        return

    print("\n===== RESULTADOS =====\n")
    for i, result in enumerate(results, start=1):
        payload = getattr(result, "payload", {}) or {}
        score = getattr(result, "score", None)

        doc_id = payload.get("doc_id", "N/A")
        chunk_index = payload.get("chunk_index", "N/A")
        text = payload.get("text", "")

        if score is not None:
            print(f"[{i}] score={score:.4f} | doc_id={doc_id} | chunk_index={chunk_index}")
        else:
            print(f"[{i}] doc_id={doc_id} | chunk_index={chunk_index}")

        print(text[:800])
        print("-" * 80)


if __name__ == "__main__":
    main()