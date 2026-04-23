#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse, unquote

import pymysql
from dotenv import load_dotenv


def log(msg):
    print(msg, flush=True)


def load_environment():
    env_path = Path(__file__).resolve().parent / ".env"
    log(f"Buscando .env en: {env_path}")

    if not env_path.exists():
        raise FileNotFoundError(f"No encontré el archivo .env en: {env_path}")

    load_dotenv(dotenv_path=env_path, override=True)
    log(".env cargado correctamente")


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

    safe_url = db_url
    if "@" in safe_url and "://" in safe_url:
        left, right = safe_url.split("@", 1)
        if ":" in left:
            proto_user, _pwd = left.rsplit(":", 1)
            safe_url = proto_user + ":****@" + right

    log(f"Intentando conectar con DATABASE_URL: {safe_url}")
    conn = pymysql.connect(**parse_db_url(db_url))
    log("Conexión MySQL exitosa")
    return conn


def show_tables(conn):
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES;")
        rows = cur.fetchall()

    tables = [list(row.values())[0] for row in rows]

    log("\nTABLAS EN LA DB:")
    for t in tables:
        log(f"- {t}")

    return tables


def show_columns(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SHOW COLUMNS FROM `{table}`;")
        rows = cur.fetchall()

    cols = [row["Field"] for row in rows]

    log(f"\nCOLUMNAS DE `{table}`:")
    for c in cols:
        log(f"- {c}")

    return cols


def count_rows(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS total FROM `{table}`;")
        row = cur.fetchone()
    return int(row["total"])


def preview_chunks(conn, table="chunks", text_col="text", order_col="id", limit=3, preview_len=300):
    sql = f"""
        SELECT `{order_col}` AS row_id, `{text_col}` AS chunk_text
        FROM `{table}`
        ORDER BY `{order_col}` DESC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()

    log(f"\nMOSTRANDO {len(rows)} CHUNKS DE PRUEBA:\n")

    if not rows:
        log("No hay filas en la tabla.")
        return

    for i, row in enumerate(rows, 1):
        text = row.get("chunk_text") or ""
        preview = text[:preview_len].replace("\r", " ").replace("\n", " ")

        log("=" * 80)
        log(f"[{i}] id = {row.get('row_id')}")
        log(f"largo = {len(text)} caracteres")
        log("preview:")
        log(preview)
        if len(text) > preview_len:
            log("...[recortado]")
        log("=" * 80)


def main():
    log("INICIO")
    load_environment()

    db_url_exists = "sí" if os.getenv("DATABASE_URL") else "no"
    log(f"DATABASE_URL cargado: {db_url_exists}")

    conn = connect()
    try:
        tables = show_tables(conn)
        if "chunks" not in tables:
            raise RuntimeError("La tabla 'chunks' no existe")

        cols = show_columns(conn, "chunks")

        if "text" not in cols:
            raise RuntimeError("La columna 'text' no existe en la tabla 'chunks'")

        total = count_rows(conn, "chunks")
        log(f"\nTotal de filas en `chunks`: {total}")

        preview_chunks(
            conn,
            table="chunks",
            text_col="text",
            order_col="id",
            limit=3,
            preview_len=300
        )

        log("\nOK: los chunks sí están subidos en la base de datos.")

    finally:
        conn.close()
        log("\nConexión cerrada")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("\nERROR FATAL:")
        log(str(e))
        log("\nTRACEBACK:")
        traceback.print_exc()
        sys.exit(1)