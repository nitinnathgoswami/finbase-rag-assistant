"""Offline tests: heavy models and the LLM are replaced by deterministic fakes,
so these verify plumbing (ingestion, Chroma, fusion, guards, citations) in seconds."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finbase_rag import config, rag  # noqa: E402
from finbase_rag.ingest import build_all  # noqa: E402
from finbase_rag.retrieval import Retriever, tokenize  # noqa: E402


class FakeEmbedder:
    def _vec(self, text):
        v = np.zeros(256)
        for t in tokenize(text):
            v[hash(t) % 256] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed(self, texts):
        return (self._vec(t) for t in texts)

    def query_embed(self, texts):
        return (self._vec(t) for t in texts)


class FakeReranker:
    def rerank(self, query, docs):
        q = set(tokenize(query))
        return [len(q & set(tokenize(d))) - 8.0 for d in docs]


@pytest.fixture(scope="session")
def chunks(tmp_path_factory):
    return build_all(out=tmp_path_factory.mktemp("proc") / "chunks.json")


@pytest.fixture()
def retriever(chunks, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    r = Retriever(chunks=chunks)
    r._embedder, r._reranker = FakeEmbedder(), FakeReranker()
    return r


# ---- ingestion ----------------------------------------------------------------
def test_ingest_is_clean_and_deduplicated(chunks):
    text = "\n".join(c.text for c in chunks)
    assert "(cid:" not in text and "■" not in text
    assert "₹1,00,000" in text
    assert 90 <= len(chunks) <= 130  # ~560k chars of PDF text -> ~100 chunks
    faq = [c for c in chunks if c.kind == "faq"]
    assert len(faq) == 60 and all(len(c.meta["faq_ids"]) == 10 for c in faq)
    assert not any("Table of Contents" in c.text for c in chunks)


def test_foreclosure_chunk_and_citation(chunks):
    c = next(c for c in chunks if c.chunk_id == "sample_5-s6.2")
    assert "3% of the outstanding principal" in c.text
    assert c.citation.endswith("Section 6.2: Foreclosure Charges & Rules")


# ---- retrieval ----------------------------------------------------------------
@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_all_modes_return_k_hits(retriever, mode):
    hits = retriever.search("foreclosure charge personal loan 18 months", k=4, mode=mode)
    assert len(hits) == 4
    assert any("foreclos" in h.chunk.text.lower() for h in hits)


def test_collection_is_reused(retriever):
    retriever.search("upi limit", mode="dense")
    n = retriever.collection.count()
    assert n == len(retriever.chunks)


# ---- pipeline -----------------------------------------------------------------
def test_guard_abstains_without_calling_llm(retriever, monkeypatch):
    monkeypatch.setattr(rag, "generate_json", lambda *a, **k: pytest.fail("LLM must not be called"))
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", 100.0)
    out = rag.RAGPipeline(retriever).ask("What is the CEO's favourite colour?")
    assert out.abstained_by == "retrieval_guard" and not out.answerable and not out.citations


def test_grounded_answer_with_citation_and_invalid_id_dropped(retriever, monkeypatch):
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", -100.0)
    monkeypatch.setattr(
        rag, "generate_json",
        lambda system, user: {"answerable": True, "answer": "It is 3% (about 4,500 on a sample).",
                              "citations": [1, 99, "x"], "confidence": "high"},
    )
    out = rag.RAGPipeline(retriever).ask("foreclosure charge personal loan before 24 months")
    assert out.answerable and len(out.citations) == 1  # 99 and "x" rejected
    assert any("4500" in w for w in out.warnings)  # a number not found in the sources is flagged


def test_llm_abstention_clears_citations(retriever, monkeypatch):
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", -100.0)
    monkeypatch.setattr(
        rag, "generate_json",
        lambda s, u: {"answerable": False, "answer": "Not in the knowledge base.", "citations": [1]},
    )
    out = rag.RAGPipeline(retriever).ask("What are the NEFT cut-off timings?")
    assert out.abstained_by == "llm" and out.citations == []


def test_followup_is_rewritten(retriever, monkeypatch):
    calls = []

    def fake(system, user):
        calls.append(system)
        if system is rag.REWRITE_PROMPT:
            return {"standalone": "foreclosure charge personal loan after 30 months"}
        return {"answerable": True, "answer": "1.5%", "citations": [1], "confidence": "high"}

    monkeypatch.setattr(rag, "generate_json", fake)
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", -100.0)
    hist = [{"role": "user", "content": "foreclosure charge?"}, {"role": "assistant", "content": "3%"}]
    out = rag.RAGPipeline(retriever).ask("and after 30 months?", history=hist)
    assert out.standalone_query.startswith("foreclosure") and rag.REWRITE_PROMPT in calls


def test_ungrounded_numbers_helper():
    assert rag.ungrounded_numbers("Fee is ₹1,000 and 18% GST", "fee ₹1,000 plus 18% GST") == []
    assert rag.ungrounded_numbers("Fee is ₹2,000", "fee ₹1,000") == ["2000"]
