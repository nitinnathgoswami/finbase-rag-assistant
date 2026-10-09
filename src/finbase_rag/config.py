"""Central configuration. Everything is overridable via environment variables / .env."""
import os
from pathlib import Path

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
CHROMA_DIR = ROOT / "data" / "chroma"

# --- Chunking ---------------------------------------------------------------
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1400"))

# --- Embeddings / reranker (both run locally on CPU through ONNX / fastembed) -
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# --- Retrieval --------------------------------------------------------------
# bm25 | dense | hybrid | hybrid_rerank
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid_rerank")
CANDIDATES = int(os.getenv("CANDIDATES", "15"))  # per retriever, before fusion
TOP_K = int(os.getenv("TOP_K", "4"))  # passages sent to the LLM
RRF_K = 60
# If even the best passage scores below this (cross-encoder logit) we abstain
# without calling the LLM. Tune with `python -m eval.run_eval --scores`.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-6.0"))

# --- LLM --------------------------------------------------------------------
# gemini | openai | anthropic
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))  # past turns used for follow-ups
