import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine)


def buscar_chunks(pregunta, limite=10):
    session = SessionLocal()

    try:
        query = text("""
            SELECT doc_id, chunk_index, text
            FROM chunks
            WHERE text LIKE :q
            LIMIT :limite
        """)

        resultados = session.execute(
            query,
            {
                "q": f"%{pregunta}%",
                "limite": limite
            }
        ).fetchall()

        return resultados

    finally:
        session.close()


def main():
    while True:
        pregunta = input("\n🧠 Pregunta: ").strip()

        if not pregunta:
            continue

        resultados = buscar_chunks(pregunta)

        print("\n📚 RESULTADOS:\n")

        for r in resultados:
            print(f"[{r.doc_id} - chunk {r.chunk_index}]")
            print(r.text[:500])
            print("-" * 80)


if __name__ == "__main__":
    main()