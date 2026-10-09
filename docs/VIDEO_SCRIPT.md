# 5-minute video outline (≈ 700 words at a calm pace)

| Time | Cover | Say / show |
|---|---|---|
| 0:00–0:30 | Problem understanding | Support bot for FinBase; must be grounded, cite sources, admit when the KB lacks the answer. Six PDFs: savings, FD & wealth, UPI, cards, loans, KYC. |
| 0:30–1:15 | Architecture | Show the README diagram: PDFs → clean → chunk → bge-small + BM25 → RRF → rerank → guard → LLM → verified answer. |
| 1:15–2:00 | Preprocessing + chunking | Data insight: 560K chars but only 104 unique chunks; 17 filler sections and 10× repeated FAQs; `■`→`₹`; chunk by section with a header; FAQ = one Q+A chunk. Mention the trade-off: dedup vs keeping provenance (`repeats_in`). |
| 2:00–2:45 | Embeddings, retrieval, LLM | Why bge-small on ONNX, Chroma, hybrid + RRF (exact tokens vs paraphrase), cross-encoder rerank, Gemini Flash at temperature 0 with strict JSON prompt; provider is swappable. |
| 2:45–3:30 | Evaluation | 48 questions incl. 7 unanswerable; metrics; show the ablation table (BM25 0.83 Hit@4 → hybrid+rerank result) and the end-to-end summary. |
| 3:30–4:30 | Live demo | (1) foreclosure after 18 months → ₹ 3%, Section 6.2 source. (2) follow-up "what if after 30 months?". (3) "Does FinBase offer crypto?" → not offered. (4) "NEFT cut-off timings?" → not in KB. (5) UPI refund question showing T+2 vs T+5 inconsistency. |
| 4:30–5:00 | Trade-offs, challenges, next steps | Challenges: glyph/format issues, near-duplicate chunks, conflicting clauses, citation self-references that point to wrong sections. Next: query decomposition, bigger eval set, caching/streaming, eval dashboard. |
