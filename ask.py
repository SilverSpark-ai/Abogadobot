import os
import sys
import textwrap
import argparse
import requests

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks_local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))


def fail(msg: str):
    print(f"❌ {msg}")
    sys.exit(1)


def run_query(qdrant_client: QdrantClient, query_vector: list[float], collection: str, limit: int = 5):
    if hasattr(qdrant_client, "query_points"):
        response = qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
        )
        if hasattr(response, "points"):
            return response.points
        return response

    if hasattr(qdrant_client, "search"):
        return qdrant_client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
        )

    fail("Tu versión de QdrantClient no tiene ni 'query_points' ni 'search'")


def build_context(results, max_chars: int):
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


def ask_ollama(prompt: str, model_name: str) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def build_prompt(question: str, context: str) -> str:
    return f"""
Eres un asistente de análisis jurídico.

INSTRUCCIONES OBLIGATORIAS:
- Responde SOLO con base en el CONTEXTO.
- NO uses conocimiento externo.
- Si el contexto no basta, responde exactamente:
  "No hay suficiente información en el contexto para responder."
- Responde en español.
- Sé claro, preciso y breve.
- Cita las fuentes dentro de la respuesta como [Fuente 1], [Fuente 2], etc.
- No inventes artículos, fechas ni autoridades que no aparezcan en el contexto.

FORMATO DE RESPUESTA:
1. Respuesta breve
2. Fundamento
3. Fuentes citadas

PREGUNTA:
{question}

CONTEXTO:
{context}

RESPUESTA:
""".strip()


def answer_question(
    question: str,
    collection: str,
    top_k: int,
    max_context_chars: int,
    model_name: str,
    show_context: bool,
):
    print(f"🧠 Pregunta: {question}")
    print(f"🤖 Embeddings: {EMBEDDING_MODEL}")
    print(f"🦙 Ollama: {model_name}")
    print(f"🗂️ Colección: {collection}")
    print(f"📚 Top K: {top_k}")

    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
    except Exception as e:
        fail(f"No se pudo conectar a Qdrant: {e}")

    print("📥 Cargando modelo de embeddings...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("🔎 Recuperando contexto relevante...")
    query_vector = embedder.encode(question, normalize_embeddings=True).tolist()
    results = run_query(qdrant_client, query_vector, collection=collection, limit=top_k)

    if not results:
        print("⚠️ No se encontraron resultados")
        return

    context = build_context(results, max_chars=max_context_chars)
    prompt = build_prompt(question, context)

    print("🧾 Pidiéndole respuesta al modelo local...")
    answer = ask_ollama(prompt, model_name=model_name)

    print("\n" + "=" * 90)
    print("RESPUESTA")
    print("=" * 90)
    print(textwrap.fill(answer, width=100))

    if show_context:
        print("\n" + "=" * 90)
        print("CONTEXTO USADO")
        print("=" * 90)
        print(context)


def interactive_mode(args):
    print("💬 Modo interactivo activo")
    print("Escribe tu pregunta y presiona Enter")
    print("Escribe 'salir' para terminar\n")

    while True:
        try:
            question = input("Pregunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Saliendo")
            return

        if not question:
            continue

        if question.lower() in {"salir", "exit", "quit"}:
            print("👋 Saliendo")
            return

        answer_question(
            question=question,
            collection=args.collection,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            model_name=args.model,
            show_context=args.show_context,
        )
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Consulta tu base vectorial y responde con Ollama.")
    parser.add_argument("--q", type=str, help="Pregunta a realizar")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Cantidad de chunks a recuperar")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help="Límite de caracteres del contexto enviado al modelo",
    )
    parser.add_argument("--collection", type=str, default=QDRANT_COLLECTION, help="Colección de Qdrant")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Modelo de Ollama a usar")
    parser.add_argument("--show-context", action="store_true", help="Muestra el contexto recuperado")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.interactive:
        interactive_mode(args)
        return

    if not args.q:
        fail('Debes usar --q "tu pregunta" o --interactive')

    answer_question(
        question=args.q,
        collection=args.collection,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        model_name=args.model,
        show_context=args.show_context,
    )


if __name__ == "__main__":
    main()