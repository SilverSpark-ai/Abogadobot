import os
import sys
import textwrap

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks_local")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = 5
MAX_CONTEXT_CHARS = 5000


def fail(msg: str):
    print(f"❌ {msg}")
    sys.exit(1)


def run_query(qdrant_client: QdrantClient, query_vector: list[float], limit: int = 5):
    if hasattr(qdrant_client, "query_points"):
        response = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
        )
        if hasattr(response, "points"):
            return response.points
        return response

    if hasattr(qdrant_client, "search"):
        return qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=limit,
        )

    fail("Tu versión de QdrantClient no tiene ni 'query_points' ni 'search'")


def build_context(results, max_chars=MAX_CONTEXT_CHARS):
    parts = []
    used = 0

    for i, result in enumerate(results, start=1):
        payload = getattr(result, "payload", {}) or {}
        text = (payload.get("text") or "").strip()
        doc_id = payload.get("doc_id", "N/A")
        chunk_index = payload.get("chunk_index", "N/A")
        score = getattr(result, "score", None)

        if not text:
            continue

        header = f"[Fuente {i}] doc_id={doc_id} chunk={chunk_index}"
        if score is not None:
            header += f" score={score:.4f}"

        block = f"{header}\n{text}\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                block = block[:remaining]
                parts.append(block)
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts)


def build_answer(query: str, results):
    if not results:
        return "No encontré contexto relevante para responder."

    payloads = [(getattr(r, "payload", {}) or {}) for r in results]
    snippets = []
    sources = []

    for idx, p in enumerate(payloads, start=1):
        text = (p.get("text") or "").strip()
        doc_id = p.get("doc_id", "N/A")
        chunk_index = p.get("chunk_index", "N/A")

        if text:
            snippet = text[:700].replace("\n", " ").strip()
            snippets.append(f"- {snippet}")
            sources.append(f"[{idx}] doc_id={doc_id} chunk={chunk_index}")

    answer = []
    answer.append(f"Pregunta: {query}\n")
    answer.append("Respuesta basada en los fragmentos recuperados:\n")

    if snippets:
        answer.append("Los textos más cercanos sugieren lo siguiente:")
        answer.extend(snippets[:3])

    if len(snippets) > 3:
        answer.append("\nAdemás, hay contexto complementario en otros fragmentos relacionados.")

    answer.append("\nFuentes usadas:")
    answer.extend(sources)

    return "\n".join(answer)


def main():
    if len(sys.argv) < 2:
        fail('Uso: python app\\scripts\\answer_local.py "tu pregunta aquí"')

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        fail("La pregunta está vacía")

    print(f"🧠 Pregunta: {query}")
    print(f"🤖 Modelo local: {MODEL_NAME}")
    print(f"🗂️ Colección: {QDRANT_COLLECTION}")

    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    except Exception as e:
        fail(f"No se pudo conectar a Qdrant: {e}")

    print("📥 Cargando modelo local...")
    model = SentenceTransformer(MODEL_NAME)

    print("🔎 Generando embedding de la pregunta...")
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    print("📡 Recuperando contexto...")
    results = run_query(qdrant_client, query_vector, limit=TOP_K)

    if not results:
        print("⚠️ No se encontraron resultados")
        return

    context = build_context(results)
    answer = build_answer(query, results)

    print("\n" + "=" * 90)
    print("RESPUESTA")
    print("=" * 90)
    print(textwrap.fill(answer, width=100))
    print("\n" + "=" * 90)
    print("CONTEXTO RECUPERADO")
    print("=" * 90)
    print(context)


if __name__ == "__main__":
    main()