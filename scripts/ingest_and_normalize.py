import os
import hashlib
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import unicodedata
import re
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
OUTPUT_PATH = "data/db/documents.parquet"


def normalize_text(text: str) -> str:
    if not text:
        return ""

    # Normalización unicode (mantiene acentos)
    text = unicodedata.normalize("NFC", text)

    # Unificar saltos de línea
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Eliminar caracteres de control (excepto \n y \t)
    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C"
    )

    # Colapsar espacios
    text = re.sub(r"[ \t]+", " ", text)

    # Colapsar saltos de línea excesivos
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except:
        try:
            return path.read_text(encoding="latin-1")
        except:
            return path.read_text(errors="replace")


def main():
    raw_path = Path(RAW_DATA_DIR)
    files = list(raw_path.rglob("*.txt"))

    print(f"📂 Archivos encontrados: {len(files)}")

    rows = []

    for file_path in tqdm(files, desc="Procesando archivos"):
        raw_text = read_file(file_path)
        normalized = normalize_text(raw_text)

        rows.append({
            "doc_id": hash_text(str(file_path)),
            "source_path": str(file_path),
            "filename": file_path.name,
            "text_raw": raw_text,
            "text_normalized": normalized,
            "char_count": len(normalized),
            "content_hash": hash_text(normalized),
        })

    df = pd.DataFrame(rows)

    os.makedirs("data/db", exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n✅ Guardado en: {OUTPUT_PATH}")
    print(f"📊 Documentos procesados: {len(df)}")


if __name__ == "__main__":
    main()