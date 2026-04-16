from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = r"C:\Users\Spark\Desktop\IA cuerpo"
PYTHON_EXE = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
SCRIPT_PATH = os.path.join(BASE_DIR, "app", "scripts", "consultar.py")


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "IA cuerpo viva"}


@app.post("/ask")
def ask_question(q: Question):
    try:
        result = subprocess.run(
            [
                PYTHON_EXE,
                SCRIPT_PATH,
                "--pregunta",
                q.question,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=BASE_DIR,
            timeout=300,
        )

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            return {
                "ok": False,
                "error": stderr if stderr else "Error desconocido",
            }

        return {
            "ok": True,
            "response": stdout,
        }

    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": "Timeout ejecutando consultar.py",
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }