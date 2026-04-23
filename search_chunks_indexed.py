#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import traceback
from pathlib import Path
from urllib.parse import urlparse, unquote

import pymysql
from dotenv import load_dotenv


def log(msg):
    print(msg, flush=True)


def load_environment():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"No encontré el .env en: {env_path}")
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
        "connect_timeout": 15,
        "read_timeout": 30,
        "write_timeout": 30,
        "autocommit": True,
    }


def connect():
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL no está definido en el .env")
    return pymysql.connect(**parse_db_url(db_url))


def get_indexes(conn, table="chunks"):
    sql = f"SHOW INDEX FROM `{table}`"
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def has_fulltext_on_text(conn, table="chunks", text_col="text"):
    indexes = get_indexes(conn, table)
    for idx in indexes:
        if (
            idx.get("Column_name") == text_col
            and str(idx.get("Index_type", "")).upper() == "FULLTEXT"
        ):
            return True
    return False


def show_indexes(conn, table="chunks"):
    rows = get_indexes(conn, table)
    if not rows:
        log(f"No hay índices en `{table}`")
        return

    log(f"\nÍNDICES EN `{table}`:")
    for r in rows:
        log(
            f"- key={r.get('Key_name')} | col={r.get('Column_name')} | "
            f"type={r.get('Index_type')} | unique={r.get('Non_unique') == 0}"
        )


def search_fulltext(conn, query, limit=5, table="chunks", text_col="text"):
    """
    Busca usando FULLTEXT indexado real.
    BOOLEAN MODE permite cosas como:
      +contrato +arrendamiento
      derecho laboral
      "debido proceso"
    """
    sql = f"""
        SELECT
            id,
            chunk_id,
            doc_id,
            chunk_index,
            char_count,
            is_indexed,
            created_at,
            MATCH(`{text_col}`) AGAINST (%s IN BOOLEAN MODE) AS score,
            `{text_col}` AS chunk_text
        FROM `{table}`
        WHERE MATCH(`{text_col}`) AGAINST (%s IN BOOLEAN MODE)
        ORDER BY score DESC, id DESC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, (query, query, limit))
        return cur.fetchall()


def preview(text, max_len=350):
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ...[recortado]"


def print_results(rows):
    if not rows:
        log("\nNo se encontraron chunks.")
        return

    log(f"\nRESULTADOS: {len(rows)}\n")
    for i, row in enumerate(rows, 1):
        log("=" * 100)
        log(f"[{i}] id={row.get('id')} | chunk_id={row.get('chunk_id')} | doc_id={row.get('doc_id')}")
        log(f"chunk_index={row.get('chunk_index')} | char_count={row.get('char_count')} | score={row.get('score')}")
        log(f"is_indexed={row.get('is_indexed')} | created_at={row.get('created_at')}")
        log("\npreview:")
        log(preview(row.get("chunk_text")))
        log("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Busca chunks usando FULLTEXT en MySQL.")
    parser.add_argument("query", help='Consulta de búsqueda. Ej: +contrato +arrendamiento')
    parser.add_argument("--limit", type=int, default=5, help="Cantidad máxima de resultados")
    parser.add_argument("--show-indexes", action="store_true", help="Muestra índices de la tabla")
    args = parser.parse_args()

    load_environment()
    conn = connect()

    try:
        if args.show_indexes:
            show_indexes(conn, "chunks")

        if not has_fulltext_on_text(conn, "chunks", "text"):
            raise RuntimeError(
                "La columna `text` NO tiene índice FULLTEXT.\n"
                "Sin eso, no puedes hacer búsqueda indexada real con MATCH ... AGAINST.\n"
                "Crea el índice y vuelve a probar."
            )

        rows = search_fulltext(
            conn=conn,
            query=args.query,
            limit=args.limit,
            table="chunks",
            text_col="text"
        )
        print_results(rows)

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