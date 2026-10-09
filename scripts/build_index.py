"""Re-parse the PDFs, rebuild chunks.json and the Chroma index, print stats.   python scripts/build_index.py"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finbase_rag.ingest import build_all  # noqa: E402
from finbase_rag.retrieval import Retriever  # noqa: E402

chunks = build_all()
print(f"{len(chunks)} chunks:", dict(Counter(c.kind for c in chunks)))
r = Retriever(chunks)
print("Chroma collection size:", r.collection.count())
