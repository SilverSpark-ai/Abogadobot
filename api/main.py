from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "IA cuerpo viva 🧠"}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    try:
        result = subprocess.run(
            [
                "venv\\Scripts\\python.exe",
                "app\\scripts\\consultar.py",
                "--pregunta",
                req.question
            ],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        return {
            "question": req.question,
            "answer": result.stdout
        }

    except Exception as e:
        return {"error": str(e)}