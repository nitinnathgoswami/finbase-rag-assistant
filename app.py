"""Streamlit customer-support UI:  streamlit run app.py"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
try:  # Streamlit Cloud / HF Spaces: copy secrets into env vars before config is imported
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass

from finbase_rag import config  # noqa: E402
from finbase_rag.rag import RAGPipeline  # noqa: E402

st.set_page_config(page_title="FinBase Support Assistant", page_icon="🏦", layout="centered")

KEY_VAR = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
}[config.LLM_PROVIDER]
if not os.getenv(KEY_VAR):
    st.error(f"Set `{KEY_VAR}` (in .env or Streamlit secrets) to use provider `{config.LLM_PROVIDER}`.")
    st.stop()

BUSY_MESSAGE = (
    "The assistant is temporarily unavailable (the AI service is busy or its usage limit was reached). "
    "Please try again in a few minutes."
)


@st.cache_resource(show_spinner="Loading knowledge base and models (first run downloads them)…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipe = get_pipeline()

EXAMPLES = [
    "What is the foreclosure charge if I close my personal loan after 18 months?",
    "My UPI payment failed but money was deducted. When do I get a refund?",
    "What rate does a senior citizen get on a 1 year FD?",
    "How much interest do I earn on 5 lakh in my savings account?",
    "Does FinBase offer cryptocurrency trading?",
    "What are the NEFT cut-off timings?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("🏦 FinBase Assistant")
    st.caption("Answers come only from FinBase's policy documents and always show their sources.")
    st.subheader("Try asking")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    show_debug = st.toggle("Show retrieved passages & latency", value=False)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        f"Retrieval: `{config.RETRIEVAL_MODE}` · LLM: `{config.LLM_PROVIDER}` · embeddings: `{config.EMBED_MODEL}`"
    )


def render_sources(m: dict):
    if m.get("citations"):
        with st.expander(f"📄 Sources ({len(m['citations'])})", expanded=True):
            for c in m["citations"]:
                st.markdown(f"**{c['citation']}**")
                st.caption(c["text"])
    for w in m.get("warnings", []):
        st.warning(w, icon="⚠️")
    if show_debug and m.get("hits"):
        with st.expander("🔎 Retrieved passages"):
            st.caption(f"Standalone query: _{m['standalone']}_ · latency (ms): {m['latency']}")
            for h in m["hits"]:
                st.markdown(f"`{h['confidence']:.2f}` {h['citation']}")


def render_message(m: dict):
    """Draw one stored chat message (used for history replay)."""
    if m.get("error"):
        st.error(m["content"])
    elif m.get("abstained"):
        st.info(m["content"], icon="ℹ️")
    else:
        st.markdown(m["content"])
    if m["role"] == "assistant":
        render_sources(m)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render_message(m)

question = st.chat_input("Ask about savings, FDs, UPI, cards, loans or KYC…") or st.session_state.pop(
    "pending", None
)
if question:
    # error messages are not sent back to the model as conversation history
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if not m.get("error")
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        r, err = None, None
        with st.spinner("Searching the knowledge base…"):
            try:
                r = pipe.ask(question, history=history)
            except Exception as e:  # network / quota / key errors
                err = e
        if err is not None:
            # full details go to the server log only, never to the user (they can contain account ids)
            print(f"LLM error: {err}", flush=True)
            m = {"role": "assistant", "content": BUSY_MESSAGE, "error": True}
        else:
            m = {
                "role": "assistant", "content": r.answer, "warnings": r.warnings,
                "abstained": bool(r.abstained_by),
                "citations": [{"citation": c.citation, "text": c.text} for c in r.citations],
                "hits": [{"citation": h.chunk.citation, "confidence": h.confidence} for h in r.hits],
                "standalone": r.standalone_query, "latency": r.latency_ms,
            }
        render_message(m)
    st.session_state.messages.append(m)