import os
import hashlib
import pandas as pd

INPUT_PATH = "data/db/documents.parquet"
OUTPUT_PATH = "data/db/chunks.parquet"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = end - overlap

    return chunks


def main():
    df = pd.read_parquet(INPUT_PATH)

    rows = []

    for _, row in df.iterrows():
        doc_id = row["doc_id"]
        text = row["text_normalized"]

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": hash_text(f"{doc_id}_{idx}_{chunk[:100]}"),
                "doc_id": doc_id,
                "chunk_index": idx,
                "text": chunk,
                "char_count": len(chunk),
                "content_hash": hash_text(chunk),
            })

    chunks_df = pd.DataFrame(rows)

    os.makedirs("data/db", exist_ok=True)
    chunks_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"✅ Guardado en: {OUTPUT_PATH}")
    print(f"📦 Total chunks: {len(chunks_df)}")
    print(f"📄 Documentos origen: {df['doc_id'].nunique()}")


if __name__ == "__main__":
    main()