"""RAG evaluation.

  python eval/run_eval.py --validate                 # check every gold label exists in the chunk store
  python eval/run_eval.py --retrieval                # Hit@k / MRR for each retrieval mode (no LLM needed)
  python eval/run_eval.py --retrieval --modes bm25,hybrid
  python eval/run_eval.py --full [--judge]           # end-to-end (needs an LLM key)
  python eval/run_eval.py --scores                   # rerank-score distribution to tune MIN_RERANK_SCORE

Metrics
  Retrieval  : Hit@1 / Hit@3 / Hit@K, MRR  (a hit = retrieved chunk contains a gold substring)
  Correctness: every `must_include` group is satisfied by the answer (deterministic, case-insensitive)
  Groundedness: (a) numbers in the answer appear in the cited sources; (b) optional LLM-judge `--judge`
  Citation   : at least one cited chunk contains a gold substring; plus citation precision
  Abstention : unanswerable questions are refused; answerable ones are not refused
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finbase_rag import config  # noqa: E402
from finbase_rag.ingest import load_chunks  # noqa: E402
from finbase_rag.rag import RAGPipeline, ungrounded_numbers  # noqa: E402
from finbase_rag.retrieval import Retriever  # noqa: E402

EVAL_SET = json.loads((ROOT / "eval" / "eval_set.json").read_text())


def has_gold(text: str, gold: list[str]) -> bool:
    return any(g.lower() in text.lower() for g in gold)


def correct(answer: str, groups: list[list[str]]) -> bool:
    a = answer.lower()
    return all(any(alt.lower() in a for alt in grp) for grp in groups)


# ---------------------------------------------------------------------------
def validate():
    chunks = load_chunks()
    bad = [x["q"] for x in EVAL_SET if x["answerable"] and not any(has_gold(c.text, x["gold"]) for c in chunks)]
    print(f"{len(EVAL_SET)} questions, {len(bad)} with gold labels missing from the chunk store")
    for q in bad:
        print("  MISSING:", q)
    return not bad


def retrieval_eval(modes: list[str], k: int):
    rows = [x for x in EVAL_SET if x["answerable"]]
    retr = Retriever()
    out = {}
    for mode in modes:
        ranks, by_type = [], defaultdict(list)
        for x in rows:
            query = x["q"]
            if x.get("history"):  # use the intended standalone form for retrieval-only runs
                query = x["history"][0]["content"] + " " + x["q"]
            hits = retr.search(query, k=k, mode=mode)
            r = next((i for i, h in enumerate(hits, 1) if has_gold(h.chunk.text, x["gold"])), None)
            ranks.append(r)
            by_type[x["type"]].append(r)
        n = len(ranks)
        res = {
            "Hit@1": sum(1 for r in ranks if r == 1) / n,
            "Hit@3": sum(1 for r in ranks if r and r <= 3) / n,
            f"Hit@{k}": sum(1 for r in ranks if r) / n,
            "MRR": sum(1 / r for r in ranks if r) / n,
        }
        out[mode] = res
        print(f"\n[{mode}] n={n}  " + "  ".join(f"{m}={v:.3f}" for m, v in res.items()))
        for t, rs in sorted(by_type.items()):
            print(f"    {t:<10} n={len(rs):<2} hit@{k}={sum(1 for r in rs if r)/len(rs):.2f}")
        misses = [x["q"] for x, r in zip(rows, ranks) if not r]
        for q in misses:
            print("    miss:", q)
    return out


def score_distribution():
    retr = Retriever(mode="hybrid_rerank")
    ans, unans = [], []
    for x in EVAL_SET:
        if x.get("history"):
            continue
        top = retr.search(x["q"], k=1)[0]
        (ans if x["answerable"] else unans).append((top.rerank_score, x["q"]))
    for name, arr in (("answerable", ans), ("unanswerable", unans)):
        vals = [s for s, _ in arr]
        print(f"{name:<13} min={min(vals):.2f} median={st.median(vals):.2f} max={max(vals):.2f}")
    print("\nLowest answerable scores:")
    for s, q in sorted(ans)[:5]:
        print(f"  {s:6.2f}  {q}")
    print("Highest unanswerable scores:")
    for s, q in sorted(unans, reverse=True)[:5]:
        print(f"  {s:6.2f}  {q}")
    print("\nPick MIN_RERANK_SCORE just below the lowest answerable score.")


JUDGE_SYSTEM = (
    "You are a strict fact-checker. Given CONTEXT and an ANSWER, decide whether every factual claim in the "
    'ANSWER is supported by the CONTEXT. Return JSON: {"supported": true|false, "unsupported_claims": [..]}'
)


def full_eval(judge: bool):
    from finbase_rag.llm import generate_json

    pipe = RAGPipeline()
    results, by_type = [], defaultdict(list)
    for x in EVAL_SET:
        resp = pipe.ask(x["q"], history=x.get("history"))
        r = {"q": x["q"], "type": x["type"], "answerable": x["answerable"],
             "answer": resp.answer, "model_answerable": resp.answerable,
             "abstained_by": resp.abstained_by, "standalone": resp.standalone_query,
             "cited": [c.chunk_id for c in resp.citations],
             "retrieved": [h.chunk.chunk_id for h in resp.hits]}
        if x["answerable"]:
            r["retrieval_hit"] = any(has_gold(h.chunk.text, x["gold"]) for h in resp.hits)
            r["correct"] = resp.answerable and correct(resp.answer, x["must_include"])
            r["false_abstain"] = not resp.answerable
            cited_text = "\n".join(c.text for c in resp.citations)
            r["citation_hit"] = bool(resp.citations) and has_gold(cited_text, x["gold"])
            r["citation_precision"] = (
                sum(has_gold(c.text, x["gold"]) for c in resp.citations) / len(resp.citations)
                if resp.citations else 0.0)
            r["numbers_grounded"] = not ungrounded_numbers(resp.answer, cited_text, x["q"]) if resp.citations else False
            if judge and resp.citations:
                ctx = "\n\n".join(c.text for c in resp.citations)
                j = generate_json(JUDGE_SYSTEM, f"CONTEXT:\n{ctx}\n\nANSWER:\n{resp.answer}")
                r["judge_supported"] = bool(j.get("supported"))
        else:
            r["correct_abstain"] = not resp.answerable
        results.append(r)
        by_type[x["type"]].append(r)
        print(("OK  " if r.get("correct", r.get("correct_abstain")) else "FAIL"), x["type"][:10].ljust(10), x["q"])

    ans = [r for r in results if r["answerable"]]
    un = [r for r in results if not r["answerable"]]

    def rate(rows, key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "n_answerable": len(ans), "n_unanswerable": len(un),
        "retrieval_hit_rate": rate(ans, "retrieval_hit"),
        "answer_correctness": rate(ans, "correct"),
        "false_abstain_rate": rate(ans, "false_abstain"),
        "citation_hit_rate": rate(ans, "citation_hit"),
        "citation_precision": rate(ans, "citation_precision"),
        "numeric_groundedness": rate(ans, "numbers_grounded"),
        "abstention_accuracy (unanswerable)": rate(un, "correct_abstain"),
    }
    if judge:
        summary["llm_judge_groundedness"] = rate(ans, "judge_supported")
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:<38} {v:.3f}" if isinstance(v, float) else f"{k:<38} {v}")
    print("\nBy question type (correct / n):")
    for t, rs in sorted(by_type.items()):
        key = "correct_abstain" if t == "unanswerable" else "correct"
        print(f"  {t:<12} {sum(bool(r.get(key)) for r in rs)}/{len(rs)}")

    out = ROOT / "eval" / "results.json"
    out.write_text(json.dumps({"config": {
        "mode": config.RETRIEVAL_MODE, "top_k": config.TOP_K, "llm": config.LLM_PROVIDER,
        "embed": config.EMBED_MODEL, "rerank": config.RERANK_MODEL},
        "summary": summary, "results": results}, ensure_ascii=False, indent=1))
    print("\nWrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--modes", default="bm25,dense,hybrid,hybrid_rerank")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--scores", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
    if a.retrieval:
        retrieval_eval(a.modes.split(","), a.k)
    if a.scores:
        score_distribution()
    if a.full:
        full_eval(a.judge)
