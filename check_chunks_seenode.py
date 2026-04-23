#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse, unquote

import pymysql
from dotenv import load_dotenv


def log(msg: str) -> None:
    print(msg, flush=True)


def load_environment() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    log(f"📄 Buscando .env en: {env_path}")

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        log("✅ .env cargado correctamente")
    else:
        log("⚠️ No encontré .env en la misma carpeta del script")


def parse_db_url(db_url: str) -> dict:
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
        "read_timeout": 15,
        "write_timeout": 15,
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

    log(f"🔌 Intentando conectar con DATABASE_URL: {safe_url}")

    config = parse_db_url(db_url)
    conn = pymysql.connect(**config)
    log("✅ Conexión MySQL exitosa")
    return conn


def show_tables(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES;")
        rows = cur.fetchall()

    tables = [list(row.values())[0] for row in rows]

    log("\n📦 TABLAS EN LA DB:")
    if not tables:
        log("   (no hay tablas)")
    else:
        for t in tables:
            log(f" - {t}")

    return tables


def show_columns(conn, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(f"SHOW COLUMNS FROM `{table}`;")
        rows = cur.fetchall()

    cols = [row["Field"] for row in rows]

    log(f"\n🧬 COLUMNAS DE `{table}`:")
    if not cols:
        log("   (sin columnas)")
    else:
        for c in cols:
            log(f" - {c}")

    return cols


def guess_table(tables: list[str]) -> str | None:
    preferred_names = [
        "chunks",
        "document_chunks",
        "documents",
        "embeddings",
        "legal_chunks",
        "chunked_documents",
    ]

    lowered = {t.lower(): t for t in tables}

    for name in preferred_names:
        if name.lower() in lowered:
            return lowered[name.lower()]

    for t in tables:
        tl = t.lower()
        if "chunk" in tl or "embed" in tl or "doc" in tl:
            return t

    return tables[0] if tables else None


def guess_text_column(columns: list[str]) -> str | None:
    preferred = [
        "chunk_text",
        "text",
        "content",
        "chunk",
        "document_text",
        "body",
    ]

    lowered = {c.lower(): c for c in columns}

    for name in preferred:
        if name.lower() in lowered:
            return lowered[name.lower()]

    for c in columns:
        cl = c.lower()
        if "text" in cl or "content" in cl or cl == "chunk":
            return c

    return None


def guess_order_column(columns: list[str]) -> str | None:
    preferred = ["id", "chunk_id", "created_at", "updated_at"]

    lowered = {c.lower(): c for c in columns}

    for name in preferred:
        if name.lower() in lowered:
            return lowered[name.lower()]

    return columns[0] if columns else None


def count_rows(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS total FROM `{table}`;")
        row = cur.fetchone()
    return int(row["total"])


def show_sample_rows(conn, table: str, limit: int = 5) -> None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM `{table}` LIMIT %s;", (limit,))
        rows = cur.fetchall()

    log(f"\n🧪 MUESTRA BRUTA DE `{table}`:")
    if not rows:
        log("   (sin filas)")
        return

    for i, row in enumerate(rows, 1):
        log("-" * 80)
        log(f"[fila {i}]")
        for k, v in row.items():
            text = str(v)
            if len(text) > 300:
                text = text[:300] + " ...[recortado]"
            log(f"{k}: {text}")


def show_chunks(conn, table: str, text_col: str, order_col: str | None, limit: int = 10) -> None:
    order_sql = f"ORDER BY `{order_col}` DESC" if order_col else ""
    sql = f"SELECT * FROM `{table}` {order_sql} LIMIT %s;"

    with conn.cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()

    log(f"\n🔥 MOSTRANDO {len(rows)} FILAS DESDE `{table}`:\n")

    if not rows:
        log("No hay datos en esa tabla.")
        return

    for i, row in enumerate(rows, 1):
        log("=" * 100)
        log(f"[{i}]")

        if order_col and order_col in row:
            log(f"{order_col}: {row[order_col]}")

        if text_col in row:
            value = row[text_col]
            log(f"\n📝 {text_col}:\n")
            log(str(value) if value is not None else "(NULL)")
        else:
            log(f"⚠️ No se encontró la columna de texto '{text_col}' en la fila")
            log("Columnas disponibles: " + ", ".join(row.keys()))

        log("=" * 100)


def main() -> None:
    log("🚀 Script iniciado")
    load_environment()

    database_url = os.getenv("DATABASE_URL")
    log(f"🔍 DATABASE_URL cargado: {'sí' if database_url else 'no'}")

    conn = connect()

    try:
        tables = show_tables(conn)
        if not tables:
            log("\n⚠️ La base de datos está vacía o no tiene tablas.")
            return

        table = guess_table(tables)
        log(f"\n🎯 Tabla elegida automáticamente: {table}")

        columns = show_columns(conn, table)
        if not columns:
            log("\n⚠️ Esa tabla no tiene columnas.")
            return

        text_col = guess_text_column(columns)
        order_col = guess_order_column(columns)

        log(f"\n🧠 Columna de texto detectada: {text_col}")
        log(f"🧠 Columna de orden detectada: {order_col}")

        total = count_rows(conn, table)
        log(f"\n📊 Total de filas en `{table}`: {total}")

        show_sample_rows(conn, table, limit=3)

        if text_col:
            show_chunks(conn, table, text_col, order_col, limit=10)
        else:
            log("\n⚠️ No pude detectar automáticamente una columna de texto.")
            log("Revisa la muestra bruta de arriba y dime qué columna contiene los chunks.")

    finally:
        conn.close()
        log("\n🔒 Conexión cerrada")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("\n💀 ERROR FATAL:")
        log(str(e))
        log("\n🧵 TRACEBACK:")
        traceback.print_exc()
        sys.exit(1)