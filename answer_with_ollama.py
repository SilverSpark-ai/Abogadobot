import os
import sys
import textwrap
import requests

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks_local")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

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
    blocks = []
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
                blocks.append(block[:remaining])
            break

        blocks.append(block)
        used += len(block)

    return "\n".join(blocks)


def ask_ollama(prompt: str) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def main():
    if len(sys.argv) < 2:
        fail('Uso: python app\\scripts\\answer_with_ollama.py "tu pregunta aquí"')

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        fail("La pregunta está vacía")

    print(f"🧠 Pregunta: {question}")
    print(f"🤖 Embedding local: {EMBED_MODEL}")
    print(f"🗂️ Colección: {QDRANT_COLLECTION}")
    print(f"🦙 Ollama: {OLLAMA_MODEL}")

    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    except Exception as e:
        fail(f"No se pudo conectar a Qdrant: {e}")

    print("📥 Cargando modelo de embeddings...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("🔎 Recuperando contexto relevante...")
    query_vector = embedder.encode(question, normalize_embeddings=True).tolist()
    results = run_query(qdrant_client, query_vector, limit=TOP_K)

    if not results:
        print("⚠️ No se encontraron resultados")
        return

    context = build_context(results)

    prompt = f"""
Eres un asistente que responde SOLO con base en el contexto dado.
Si el contexto no basta, dilo claramente.
No inventes.
Resume de forma clara y útil.
Cita las fuentes usando [Fuente 1], [Fuente 2], etc. cuando corresponda.

Pregunta:
{question}

Contexto:
{context}

Respuesta:
""".strip()

    print("🧾 Pidiéndole respuesta al modelo local...")
    answer = ask_ollama(prompt)

    print("\n" + "=" * 90)
    print("RESPUESTA")
    print("=" * 90)
    print(textwrap.fill(answer, width=100))

    print("\n" + "=" * 90)
    print("CONTEXTO USADO")
    print("=" * 90)
    print(context)


if __name__ == "__main__":
    main()