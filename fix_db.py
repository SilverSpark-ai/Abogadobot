from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

with engine.begin() as conn:
    conn.execute(text("ALTER TABLE documents MODIFY content LONGTEXT NOT NULL;"))
    conn.execute(text("ALTER TABLE chunks MODIFY text LONGTEXT NOT NULL;"))

print("✅ DB actualizada: content/text ahora son LONGTEXT")