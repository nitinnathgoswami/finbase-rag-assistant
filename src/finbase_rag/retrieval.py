"""Hybrid retrieval: BM25 (lexical) + dense (bge-small in Chroma), fused with
Reciprocal Rank Fusion, then optionally re-scored by a cross-encoder.

Why hybrid: bank documents are full of exact tokens (T+2, 1.5%, Form 15H, U69,
₹5,000) that dense embeddings blur but BM25 nails, while paraphrased customer
questions ("money got cut but payment failed") need the dense side.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .ingest import Chunk, load_chunks

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float  # final ordering score (mode dependent)
    bm25_rank: int | None = None
    dense_rank: int | None = None
    dense_sim: float | None = None
    rerank_score: float | None = None

    @property
    def confidence(self) -> float:
        """0-1 relevance estimate for display (sigmoid of the cross-encoder logit)."""
        if self.rerank_score is not None:
            return 1 / (1 + math.exp(-self.rerank_score))
        if self.dense_sim is not None:
            return max(0.0, min(1.0, self.dense_sim))
        return 0.0


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, mode: str | None = None):
        self.chunks = chunks or load_chunks()
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.mode = mode or config.RETRIEVAL_MODE
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in self.chunks])
        self._embedder = None
        self._reranker = None
        self._collection = None

    # ---- lazy heavy components -------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=config.EMBED_MODEL)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(model_name=config.RERANK_MODEL)
        return self._reranker

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._build_collection()
        return self._collection

    def _signature(self) -> str:
        h = hashlib.sha1(config.EMBED_MODEL.encode())
        for c in self.chunks:
            h.update(c.chunk_id.encode())
            h.update(c.embed_text.encode())
        return h.hexdigest()

    def _build_collection(self):
        import chromadb

        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        sig = self._signature()
        try:
            col = client.get_collection("finbase")
            if col.metadata.get("sig") == sig and col.count() == len(self.chunks):
                return col
            client.delete_collection("finbase")
        except Exception:
            pass
        col = client.create_collection("finbase", metadata={"hnsw:space": "cosine", "sig": sig})
        texts = [c.embed_text for c in self.chunks]
        vecs = [v.tolist() for v in self.embedder.embed(texts)]
        col.add(
            ids=[c.chunk_id for c in self.chunks],
            embeddings=vecs,
            documents=texts,
            metadatas=[
                {"doc_id": c.doc_id, "kind": c.kind, "section": c.section} for c in self.chunks
            ],
        )
        return col

    # ---- individual retrievers -------------------------------------------
    def _bm25_search(self, query: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:n]
        return [self.chunks[i].chunk_id for i in order if scores[i] > 0]

    def _dense_search(self, query: str, n: int) -> list[tuple[str, float]]:
        qv = next(iter(self.embedder.query_embed([query]))).tolist()
        res = self.collection.query(query_embeddings=[qv], n_results=min(n, len(self.chunks)))
        # chroma cosine distance = 1 - cosine similarity
        return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    # ---- public API ------------------------------------------------------
    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[Hit]:
        k = k or config.TOP_K
        mode = mode or self.mode
        n = config.CANDIDATES
        hits: dict[str, Hit] = {}

        def get(cid: str) -> Hit:
            return hits.setdefault(cid, Hit(chunk=self.by_id[cid], score=0.0))

        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            for rank, cid in enumerate(self._bm25_search(query, n), 1):
                h = get(cid)
                h.bm25_rank = rank
                h.score += 1 / (config.RRF_K + rank)
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            for rank, (cid, sim) in enumerate(self._dense_search(query, n), 1):
                h = get(cid)
                h.dense_rank, h.dense_sim = rank, sim
                h.score += 1 / (config.RRF_K + rank)

        ranked = sorted(hits.values(), key=lambda h: -h.score)
        if mode == "hybrid_rerank" and ranked:
            pool = ranked[: max(n, k)]
            scores = list(self.reranker.rerank(query, [h.chunk.embed_text for h in pool]))
            for h, s in zip(pool, scores):
                h.rerank_score = float(s)
            ranked = sorted(pool, key=lambda h: -h.rerank_score)
            for h in ranked:
                h.score = h.rerank_score
        return ranked[:k]
