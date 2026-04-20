import os
import json
import math
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
OUT_DIR = "semantic_index"
EMBED_PATH = os.path.join(OUT_DIR, "embeddings.npy")
META_PATH = os.path.join(OUT_DIR, "metadata.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)

engine = create_engine(DATABASE_URL, future=True)

def count_chunks():
    with engine.connect() as conn:
        row = conn.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()
        return row[0]

def fetch_chunks(offset: int, limit: int):
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT doc_id, chunk_index, text
                FROM chunks
                ORDER BY id
                LIMIT :limit OFFSET :offset
            """),
            {"limit": limit, "offset": offset},
        ).fetchall()
    return rows

def main():
    total = count_chunks()
    print(f"Total chunks: {total}")

    model = SentenceTransformer(MODEL_NAME)

    dim = 384
    mmap = np.memmap(
        EMBED_PATH,
        dtype="float32",
        mode="w+",
        shape=(total, dim),
    )

    written = 0

    with open(META_PATH, "w", encoding="utf-8") as meta_f:
        for offset in range(0, total, BATCH_SIZE):
            rows = fetch_chunks(offset, BATCH_SIZE)
            texts = [r[2] for r in rows]

            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            batch_n = len(rows)
            mmap[written:written + batch_n] = embeddings

            for r in rows:
                meta_f.write(json.dumps({
                    "doc_id": r[0],
                    "chunk_index": r[1],
                    "text": r[2],
                }, ensure_ascii=False) + "\n")

            written += batch_n
            print(f"Embeddings: {written}/{total}")

    mmap.flush()
    print("OK: embeddings generados")

if __name__ == "__main__":
    main()