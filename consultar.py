import os
import sys
import io
import argparse
import contextlib
import requests

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Encoding seguro para Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Silenciar bastante ruido de librerías
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks_local")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def fail(msg: str):
    print(msg, file=sys.stderr)
    sys.exit(1)


def get_embedding_model():
    # Silencia stdout/stderr de carga del modelo
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return SentenceTransformer(EMBEDDING_MODEL)


def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)


def embed_text(model, text: str):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return model.encode(text, normalize_embeddings=True).tolist()


def search_qdrant(client: QdrantClient, vector, top_k: int = 5):
    # Compatibilidad con versiones nuevas y viejas del cliente
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=top_k,
        )
        if hasattr(response, "points"):
            return response.points
        return response

    if hasattr(client, "search"):
        return client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k,
        )

    fail("Tu version de QdrantClient no tiene ni 'query_points' ni 'search'")


def build_context(results):
    chunks = []

    for r in results:
        payload = getattr(r, "payload", None) or {}
        text = (payload.get("text") or "").strip()
        if text:
            chunks.append(text)

    return "\n\n".join(chunks)


def ask_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )

    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.text}")

    data = response.json()
    return (data.get("response") or "").strip()


def build_prompt(question: str, context: str) -> str:
    return f"""
Eres un asistente de análisis jurídico.

INSTRUCCIONES OBLIGATORIAS:
- Responde SOLO con base en el CONTEXTO.
- NO uses conocimiento externo.
- Si el contexto no basta, responde exactamente:
  "No tengo suficiente información en los documentos."
- Responde en español.
- Sé claro, preciso y breve.
- No inventes artículos, fechas ni autoridades que no aparezcan en el contexto.

Pregunta:
{question}

Contexto:
{context}

Respuesta:
""".strip()


def answer_question(question: str, top_k: int = 5, show_context: bool = False):
    model = get_embedding_model()
    client = get_qdrant_client()

    query_vector = embed_text(model, question)
    results = search_qdrant(client, query_vector, top_k=top_k)

    if not results:
        print("No se encontraron resultados")
        return

    context = build_context(results)
    prompt = build_prompt(question, context)
    answer = ask_ollama(prompt)

    # Por defecto imprime SOLO la respuesta
    print(answer.strip())

    # Contexto opcional, solo si tú lo pides desde consola
    if show_context:
        print("\n--- CONTEXTO ---\n")
        print(context)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pregunta",
        type=str,
        required=True,
        help="Pregunta a consultar",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Cantidad de chunks a recuperar",
    )
    parser.add_argument(
        "--show_context",
        action="store_true",
        help="Mostrar contexto recuperado",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    answer_question(
        question=args.pregunta,
        top_k=args.top_k,
        show_context=args.show_context,
    )


if __name__ == "__main__":
    main()