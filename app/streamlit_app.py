"""Streamlit demo: grounded clinical Q&A over the medical manual.

Run:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAGConfig
from src.qa import answer_question, retrieve_context

st.set_page_config(page_title="Clinical RAG Assistant", page_icon="⚕️")
st.title("⚕️ Clinical RAG Assistant")
st.caption(
    "Retrieval-augmented Q&A over a 4,000+ page medical reference manual. "
    "Answers are generated strictly from retrieved passages."
)
st.warning("Educational prototype — not medical advice.", icon="⚠️")

EXAMPLES = [
    "What is the protocol for managing sepsis in a critical care unit?",
    "What are the typical clinical symptoms of appendicitis, and when is surgery indicated?",
    "Outline the first aid, diagnostic steps, and treatment plan for a leg fracture.",
]

question = st.text_input("Ask a clinical question", placeholder=EXAMPLES[0])
col1, col2, col3 = st.columns(3)
for col, ex in zip((col1, col2, col3), EXAMPLES):
    if col.button("Try example", key=ex):
        question = ex

if question:
    config = RAGConfig()
    with st.spinner("Retrieving passages and generating answer..."):
        try:
            result = answer_question(question, config)
            hits = retrieve_context(question, config)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Retrieved evidence")
    for i, hit in enumerate(hits, start=1):
        with st.expander(f"Passage {i} — manual page {hit['page']}"):
            st.write(hit["text"][:1200] + ("…" if len(hit["text"]) > 1200 else ""))

    st.caption(result["disclaimer"])
