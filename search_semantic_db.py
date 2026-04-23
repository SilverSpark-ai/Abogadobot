#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import argparse
import traceback
from pathlib import Path
from urllib.parse import urlparse, unquote

import pymysql
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ajusta estos nombres si tu esquema real usa otros
TABLE_CHUNKS = "chunks"
TABLE_DOCS = "documents"
TABLE_EMB = "embeddings"          # <- cambia si tu tabla real no se llama así

CHUNK_ID_COL = "chunk_id"
CHUNK_TEXT_COL = "text"
CHUNK_DOC_ID_COL = "doc_id"
CHUNK_INDEX_COL = "chunk_index"

DOC_ID_COL = "id"
DOC_NAME_COL = "filename"         # <- cambia a title/name/file_name si aplica

EMB_CHUNK_ID_COL = "chunk_id"
EMB_VECTOR_COL = "embedding"      # <- puede ser embedding_json, vector, etc.


# =========================
# UTILS
# =========================
def log(msg):
    print(msg, flush=True)


def load_environment():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"No encontré .env en: {env_path}")
    load_dotenv(dotenv_path=env_path, override=True)


def parse_db_url(db_url):
    cleaned = db_url.replace("mysql+pymysql://", "mysql://")
    parsed = urlparse(cleaned)

    if not parsed.hostname:
        raise ValueError("DATABASE_URL no tiene host válido")
    if not parsed.username:
        raise ValueError("DATABASE_URL no tiene usuario")
    if parsed.password is None:
        raise ValueError("DATABASE_URL no tiene password")
    if not parsed.path or parsed.path == "/":
        raise ValueError("DATABASE_URL no tiene nombre de base de datos")

    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username),
        "password": unquote(parsed.password),
        "database": parsed.path.lstrip("/"),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "connect_timeout": 20,
        "read_timeout": 60,
        "write_timeout": 60,
        "autocommit": True,
    }


def connect():
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL no está definido en el .env")
    return pymysql.connect(**parse_db_url(db_url))


def get_tables(conn):
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES;")
        rows = cur.fetchall()
    return [list(r.values())[0] for r in rows]


def get_columns(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SHOW COLUMNS FROM `{table}`;")
        rows = cur.fetchall()
    return [r["Field"] for r in rows]


def parse_embedding(raw):
    """
    Soporta embeddings guardados como:
    - JSON string: "[0.1, 0.2, ...]"
    - bytes de JSON
    - lista ya materializada
    """
    if raw is None:
        return None

    if isinstance(raw, list):
        return [float(x) for x in raw]

    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")

    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return [float(x) for x in data]
        except Exception:
            return None

    return None


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return -1.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return -1.0

    return dot / (norm_a * norm_b)


# =========================
# DB READ
# =========================
def fetch_joined_rows(conn, limit=None):
    """
    Trae chunks + embedding + documento.
    Ajusta DOC_NAME_COL / TABLE_EMB si tu esquema real cambia.
    """
    sql = f"""
        SELECT
            c.`{CHUNK_ID_COL}`   AS chunk_id,
            c.`{CHUNK_DOC_ID_COL}` AS doc_id,
            c.`{CHUNK_INDEX_COL}` AS chunk_index,
            c.`{CHUNK_TEXT_COL}` AS chunk_text,
            d.`{DOC_NAME_COL}`   AS document_name,
            e.`{EMB_VECTOR_COL}` AS embedding
        FROM `{TABLE_CHUNKS}` c
        INNER JOIN `{TABLE_EMB}` e
            ON c.`{CHUNK_ID_COL}` = e.`{EMB_CHUNK_ID_COL}`
        LEFT JOIN `{TABLE_DOCS}` d
            ON c.`{CHUNK_DOC_ID_COL}` = d.`{DOC_ID_COL}`
    """

    if limit:
        sql += " LIMIT %s"

    with conn.cursor() as cur:
        if limit:
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)
        return cur.fetchall()


# =========================
# SEARCH
# =========================
def semantic_search(query, rows, model, top_k=5):
    query_vec = model.encode(query, normalize_embeddings=False).tolist()

    scored = []
    skipped = 0

    for row in rows:
        emb = parse_embedding(row.get("embedding"))
        if not emb:
            skipped += 1
            continue

        score = cosine_similarity(query_vec, emb)
        if score < -0.5:
            skipped += 1
            continue

        scored.append({
            "score": score,
            "chunk_id": row.get("chunk_id"),
            "doc_id": row.get("doc_id"),
            "chunk_index": row.get("chunk_index"),
            "document_name": row.get("document_name"),
            "chunk_text": row.get("chunk_text"),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k], skipped, len(scored)


def preview(text, max_len=500):
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ...[recortado]"


def print_results(results):
    if not results:
        log("\nNo se encontraron chunks relevantes.")
        return

    log(f"\nTOP {len(results)} CHUNKS:\n")
    for i, r in enumerate(results, 1):
        log("=" * 100)
        log(
            f"[{i}] score={r['score']:.6f} | "
            f"chunk_id={r['chunk_id']} | doc_id={r['doc_id']} | chunk_index={r['chunk_index']}"
        )
        log(f"documento: {r['document_name']}")
        log("\nchunk:")
        log(preview(r["chunk_text"]))
        log("=" * 100)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Búsqueda semántica estilo IUSTITIA antes de Ollama."
    )
    parser.add_argument("query", help="Texto de búsqueda")
    parser.add_argument("--top_k", type=int, default=5, help="Cantidad de chunks a devolver")
    parser.add_argument("--limit_db", type=int, default=None, help="Limita filas leídas de DB para pruebas")
    parser.add_argument("--debug", action="store_true", help="Muestra tablas y columnas")
    args = parser.parse_args()

    load_environment()
    conn = connect()

    try:
        if args.debug:
            tables = get_tables(conn)
            log("\nTABLAS:")
            for t in tables:
                log(f"- {t}")
            for t in tables:
                try:
                    cols = get_columns(conn, t)
                    log(f"\nCOLUMNAS DE {t}:")
                    for c in cols:
                        log(f"  - {c}")
                except Exception as e:
                    log(f"No pude leer columnas de {t}: {e}")

        log("\nCargando modelo de embeddings...")
        model = SentenceTransformer(MODEL_NAME)
        log("Modelo cargado.")

        log("Leyendo chunks + embeddings desde la DB...")
        rows = fetch_joined_rows(conn, limit=args.limit_db)
        log(f"Filas leídas: {len(rows)}")

        results, skipped, valid = semantic_search(
            query=args.query,
            rows=rows,
            model=model,
            top_k=args.top_k
        )

        log(f"Embeddings válidos: {valid}")
        log(f"Filas saltadas: {skipped}")

        print_results(results)

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("\nERROR FATAL:")
        log(str(e))
        log("\nTRACEBACK:")
        traceback.print_exc()
        sys.exit(1)