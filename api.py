"""Optional REST API:  uvicorn api:app --reload   (docs at /docs)"""
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "src"))
from finbase_rag.rag import RAGPipeline  # noqa: E402

app = FastAPI(title="FinBase Support RAG API", version="1.0")
pipe = RAGPipeline()


class Turn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Turn] = []


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(pipe.retriever.chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    r = pipe.ask(req.question, [t.model_dump() for t in req.history])
    return {
        "answer": r.answer,
        "answerable": r.answerable,
        "abstained_by": r.abstained_by,
        "confidence": round(r.confidence, 3),
        "standalone_query": r.standalone_query,
        "sources": [{"id": c.chunk_id, "citation": c.citation, "text": c.text} for c in r.citations],
        "warnings": r.warnings,
        "latency_ms": r.latency_ms,
    }
