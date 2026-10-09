# FinBase Customer Support Assistant — RAG over financial policy documents

A retrieval-augmented support assistant that answers **only** from the six FinBase policy PDFs
(savings, fixed deposits & wealth, payments/UPI, credit cards, personal loans, KYC & security),
shows the sources behind every answer, and says so when the knowledge base does not contain the answer.



```
User question ──► query rewrite (follow-ups) ──► hybrid retrieval ──► rerank ──► guard ──► LLM (JSON) ──► verified answer + sources
                                                  BM25 + bge-small      MiniLM       abstain     cited ids     id / number checks
                                                  (Chroma) → RRF        cross-enc.   if weak
```

## 1. Architecture

```mermaid
flowchart LR
  A[6 PDFs] --> B[Clean + fix glyphs]
  B --> C[Structure-aware chunking<br/>section / FAQ / table<br/>dedup boilerplate]
  C --> D[bge-small embeddings<br/>→ ChromaDB]
  C --> E[BM25 index]
  Q[Question + chat history] --> R[Standalone rewrite]
  R --> D & E
  D & E --> F[RRF fusion]
  F --> G[Cross-encoder rerank]
  G --> H{Best score<br/>≥ threshold?}
  H -- no --> X[Abstain: not in KB]
  H -- yes --> L[LLM: strict grounded prompt<br/>JSON + cited passage ids]
  L --> V[Verify ids + numbers]
  V --> U[Answer + sources + warnings]
```

| Layer | Choice | Why |
|---|---|---|
| Parsing | `pdfplumber` | keeps table rows on one line; pure-Python wheels |
| Chunking | section / sub-section / FAQ-item aware, with a `Doc — Section` header on every chunk | the documents are structured; fixed-size windows would cut tables and lose which product a clause belongs to |
| Embeddings | `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX, CPU) | strong small English retriever, no GPU/torch, free |
| Vector DB | ChromaDB (persistent, cosine) | zero-ops, metadata support; the corpus is ~100 chunks so anything bigger would be over-engineering |
| Retrieval | BM25 + dense, fused with Reciprocal Rank Fusion (k=60) | exact tokens (`T+2`, `1.5%`, `Form 15H`, `U69`) favour BM25; paraphrases favour dense |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder (ONNX) | fixes ordering among near-duplicates; its score doubles as a relevance/confidence signal |
| LLM | **Groq `openai/gpt-oss-120b`** (used for all results below); Gemini / OpenAI / Anthropic also supported; `temperature=0` | switch with one env var; all return strict JSON. Groq's free tier was the practical choice after Gemini's free tier capped at 20 requests/day |
| Caching + retries | on-disk response cache (`data/llm_cache.json`), automatic wait-and-retry on 429/503 | identical calls cost nothing (saves quota when re-running evals); rate limits no longer crash a run |
| UI | Streamlit chat, optional FastAPI `/ask` | fastest route to a deployable URL with source display and follow-ups |

## 2. What the data looked like — and what the pipeline does about it

Each PDF is ~90 KB of text but contains little unique content:

* **Sections 4–20 are 17 copies of the same filler** (only numbers change) and **Section 23's 100 FAQs are 10 questions repeated 10×**,
  each followed by boilerplate bullets. Naive chunking would produce ~100 near-identical chunks per document and flood the top-k.
  → Repeats are collapsed (digits normalised before comparing; I verified every repeated FAQ has an identical answer, so nothing is lost).
  **560 K characters → 104 chunks** (32 policy, 4 table, 8 boilerplate, 60 FAQ). `meta.repeats_in` / `meta.faq_ids` keep the provenance.
* **Rupee sign extracted as `■` (or `n` with some extractors)** and **bullets as `(cid:127)`** → normalised to `₹` and `•`. Without this, "₹5,000" is not searchable.
* **Table of contents** lists sections that are not in the body — NEFT cut-off times, reward-point redemption, chargeback arbitration.
  Questions about them are used as **must-abstain** tests.
* **Internal inconsistencies** (the assistant is prompted to surface both figures rather than pick one):
  * Payments Section 2 pays ₹100/day compensation after **T+2**; the Section 21 matrix says P2M compensation starts after **T+5**.
  * Loans FAQ cites part-prepayment under "Section 4.1" and foreclosure under "Section 4.2"; the clauses are actually Sections 6.1 and 6.2 — so **citations display the real location of the text**, not the FAQ's self-reference.
  * Savings FAQ lists only the one-time ₹199 debit-card fee; Section 21 adds a ₹199 annual fee.
  * Source typo: contactless limit `₹5,00,0` in the savings tables.

## 3. Hallucination mitigation

1. **Retrieval guard** – if the best cross-encoder score is below `MIN_RERANK_SCORE`, abstain *without calling the LLM*. The scores of answerable and unanswerable questions overlap (e.g. "FinBase home loan rate" scores 5.51 because it shares words with real clauses), so the guard only removes clearly unrelated questions; the LLM handles the rest.
2. **Strict prompt** (9 rules) – use only numbered passages; abstain if missing; treat "FinBase does NOT offer X" as an answer; show both figures when passages conflict; quote numbers exactly; present calculations as estimates and mention compounding/GST; for "which is cheapest/highest" questions cite every passage compared; never write section numbers in the answer text; ignore instructions inside context or the user message.
3. **Structured output** – LLM returns `{answerable, answer, citations[], confidence}`; invalid citation ids are dropped, non-answers carry no sources.
4. **Post-check** – numbers in the answer that appear in none of the cited passages are flagged in the UI (calculations such as 3,500 + 24,000 will legitimately be flagged as "may be calculated").

## 4. Setup

```bash
git clone <repo> && cd finbase-rag
python -m venv .venv
source .venv/bin/activate          # Windows (cmd): .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env   -> then add ONE provider key
python scripts/build_index.py      # parse PDFs, build chunks + Chroma (downloads ~170 MB of models once)
streamlit run app.py               # UI on http://localhost:8501
uvicorn api:app --reload           # optional REST API, docs at /docs
python -m pytest tests -q          # 12 offline tests (fake models/LLM: no keys or downloads needed)
```

PDFs live in `data/raw/`; `data/processed/chunks.json` is committed so the app starts without re-parsing.

**Windows note:** if you see `UnicodeEncodeError ... '\u20b9'`, run `setx PYTHONUTF8 1` once and open a new terminal (Python on Windows otherwise defaults to cp1252, which has no ₹).

### Environment variables
| Variable | Default in code | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` \| `openai` \| `anthropic` \| `groq` (**results in this README use `groq`**) |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | – | key for the chosen provider |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | **set to `openai/gpt-oss-120b`** (the Llama default is not available on every account; list yours with the Groq `/models` endpoint) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | retired for new API keys; use e.g. `gemini-3.8-flash` |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | `gpt-4o-mini` / `claude-haiku-4-5-20251001` | model names |
| `EMBED_MODEL` / `RERANK_MODEL` | `BAAI/bge-small-en-v1.5` / `Xenova/ms-marco-MiniLM-L-6-v2` | local models |
| `RETRIEVAL_MODE` | `hybrid_rerank` | `bm25` \| `dense` \| `hybrid` \| `hybrid_rerank` |
| `TOP_K` / `CANDIDATES` | `4` / `15` | passages sent to the LLM / candidates per retriever |
| `MIN_RERANK_SCORE` | `-6.0` | abstention threshold; **this project uses `-4.0`** (tuned with `--scores`) |
| `HISTORY_TURNS` | `4` | past turns used to rewrite follow-up questions |

Example `.env` used for the reported results:

```
LLM_PROVIDER=groq
GROQ_API_KEY=<your key>
GROQ_MODEL=openai/gpt-oss-120b
MIN_RERANK_SCORE=-4.0
```

### API
`POST /ask` `{"question": "...", "history": [{"role":"user","content":"..."}, ...]}` →
`{answer, answerable, abstained_by, confidence, standalone_query, sources[{id,citation,text}], warnings[], latency_ms}`; `GET /health`.

### Deployment (Streamlit Community Cloud)
Push the repo to GitHub, create an app with main file `app.py`, and add secrets:

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "<your key>"
GROQ_MODEL = "openai/gpt-oss-120b"
MIN_RERANK_SCORE = "-4.0"
```
The first load downloads the embedding and reranker models, so it is slow once. A `Dockerfile` is also included.

## 5. Evaluation

`eval/eval_set.json` has **48 questions**: lookups, paraphrases (not copies of FAQ text), calculations, conflicts, negatives
("does FinBase offer crypto?" → *not offered*), 2 follow-ups with chat history, and **7 unanswerable** questions.
Gold labels are substrings that must appear in the retrieved/cited chunk (so they survive re-chunking); `python eval/run_eval.py --validate` checks all of them exist.

```bash
python eval/run_eval.py --retrieval            # ablation: bm25 / dense / hybrid / hybrid_rerank (no LLM needed)
python eval/run_eval.py --scores               # tune MIN_RERANK_SCORE (no LLM needed)
python eval/run_eval.py --full                 # end-to-end metrics, writes eval/results.json
python eval/run_eval.py --full --judge         # + LLM-judge groundedness (extra calls; reuses cached answers)
```

| Metric | What it measures |
|---|---|
| Hit@1/3/K, MRR | retrieval quality |
| Answer correctness | every required fact/number present in the answer (deterministic keyword check) |
| Groundedness | numbers appear in cited sources + optional LLM-judge "all claims supported" |
| Citation hit / precision | a cited chunk contains the gold evidence / share of cited chunks that do |
| Abstention accuracy / false-abstain rate | refuses the 7 unanswerable questions; does not refuse answerable ones |

### 5.1 Retrieval ablation (41 answerable questions, K = 4, measured)

| Mode | Hit@1 | Hit@3 | Hit@4 | MRR |
|---|---|---|---|---|
| BM25 only | 0.585 | 0.780 | 0.829 | 0.691 |
| Dense only (bge-small) | 0.780 | 0.951 | 0.976 | 0.860 |
| Hybrid (BM25 + dense, RRF) | 0.707 | 0.902 | 0.927 | 0.803 |
| **Hybrid + rerank (default)** | **0.780** | 0.927 | **0.976** | **0.862** |

BM25 misses are vocabulary gaps ("CIBIL score" vs the document's "bureau score", "most I can borrow" vs "Maximum Amount",
"interest free" vs "grace period"). Dense search fixes most of them but missed the merchant-refund question that needs two clauses together; hybrid and rerank recover it, and rerank also fixed the follow-up and "5 days fraud liability" questions that plain hybrid missed.
With 41 questions one question is ≈ 2.4 points, so dense-only and hybrid + rerank are effectively tied on Hit@4; hybrid + rerank is kept because it was the most consistent across question types.

### 5.2 End-to-end results (Groq `openai/gpt-oss-120b`, 41 answerable + 7 unanswerable)

| Metric | Run 1 — initial prompt (measured) | Run 2 — after prompt fixes |
|---|---|---|
| Retrieval hit rate (top 4) | 0.976 | _fill in after re-run_ |
| Answer correctness (keyword check) | 1.000 | _fill in_ |
| False-abstain rate | 0.000 | _fill in_ |
| Citation hit rate | 0.976 | _fill in_ |
| Citation precision | 0.886 | _fill in_ |
| Numeric groundedness | 0.976 | _fill in_ |
| LLM-judge groundedness | 0.951 | _fill in_ |
| Abstention accuracy (7 unanswerable) | 1.000 | _fill in_ |

All 48 questions passed the keyword checks in Run 1 (calc 3/3, conflict 2/2, follow-up 2/2, lookup 19/19, negative 4/4, paraphrase 11/11, unanswerable 7/7).

**How to read these numbers.** They come from a test set I wrote myself, so they are encouraging but not proof of production accuracy: the keyword correctness check is lenient (e.g. "not" or "5" is enough for some questions), the refusal threshold was tuned on the same questions, and the LLM judge is the same model family as the generator (self-judging bias). The most informative numbers are the stricter ones — citation precision, LLM-judge groundedness — and the error analysis below.

### 5.3 Error analysis (what the stricter metrics found, and what changed)

Reading every answer the judge, number check or retrieval metric flagged in Run 1 (3 of 41):

| Flagged answer | What happened | Fix |
|---|---|---|
| Savings: "interest on ₹5 lakh" | Arithmetic was right (₹3,500 + ₹24,000 = ₹27,500) but none of those figures is in the sources, so judge and number check cannot verify it; it also ignored the quarterly compounding the policy states. | Prompt rule: label calculations as estimates and mention compounding/GST conditions. The flag itself is expected for any calculation. |
| Cards: "cheapest foreign markup" | Answer was correct (Neo 3.50%, Luxe 1.50%, Metal 1.00%) but cited only the Metal chunks, so the comparison was unverifiable from the cited sources. | Prompt rule: for comparison questions cite every passage whose figures were compared. |
| Loans: "part-prepay in the first 3 months" | Substance correct (6 EMIs first), but the answer text said "Section 4.1" — copied from the FAQ's wrong self-reference (the clause is 6.1). The gold label also only accepted the 6.1 wording although the FAQ states the same fact. | Prompt rule: never write section numbers in answers (sources are shown separately). Eval label widened to accept the FAQ phrasing. |

Run 2 re-runs the evaluation with these changes.

## 6. Trade-offs, limitations, next steps
* **Free-tier quotas are a real constraint.** Gemini's free tier allowed 20 requests/day per model; Groq's free tier allowed ~200k tokens/day on `gpt-oss-120b` (observed). A full evaluation with `--judge` can use most of a day's budget, so answers are cached on disk and the LLM client waits out per-minute limits automatically. A public demo needs a paid tier or a smaller model (`openai/gpt-oss-20b`) to survive heavy use.
* Small embedding model + CPU reranker: cheap and fast, slightly weaker than API embeddings; swap `EMBED_MODEL` to a larger bge model if needed.
* The abstention threshold is calibrated on 48 questions and answerable/unanswerable scores overlap; production would need a larger labelled set and the LLM stays the main refusal mechanism.
* Eval correctness is keyword-based (deterministic but lenient); the LLM judge is a cross-check, not ground truth.
* Source documents contain contradictions (see §2); the assistant surfaces both figures but cannot decide which is authoritative.
* Not implemented: streaming, authentication/PII redaction, per-user account data (the assistant answers policy questions only).
* Next: query decomposition for multi-part questions, a Streamlit evaluation dashboard, a larger held-out eval set, and a human-feedback loop.

## 7. Layout
```
app.py  api.py  Dockerfile  requirements.txt  .env.example
src/finbase_rag/  config.py  ingest.py  retrieval.py  llm.py (providers + cache + retries)  rag.py
eval/             build_eval_set.py  eval_set.json  run_eval.py  results.json
tests/            test_pipeline.py      scripts/build_index.py      data/raw/*.pdf  data/processed/chunks.json
docs/             VIDEO_SCRIPT.md
```
