from __future__ import annotations

import json
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend import (
    extract_pdf_text,
    generate_report,
    load_dataset,
)

st.set_page_config(page_title="Papermind", layout="wide")


@st.cache_resource(show_spinner="Loading arXiv dataset…")
def get_backend():
    return load_dataset()


get_backend()

for key, default in [
    ("chat_history", []),
    ("papers", []),               # list of {"name": str, "text": str}
    ("results_per_paper", []),    # list of {"name": str, "summary": str, "gaps": str, "recs": str}
    ("is_thinking", False),
    ("pending_query", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _clear_results():
    st.session_state.results_per_paper = []
    st.session_state.chat_history = []


def _render_paper_results(r: dict):
    if r.get("summary"):
        st.markdown("### 📋 Summary")
        st.write(r["summary"])

    if r.get("gaps"):
        st.markdown("---")
        st.markdown("### 🔍 Research Gaps")
        st.write(r["gaps"])

    if r.get("recs"):
        st.markdown("---")
        st.markdown("### 📚 Recommended Papers")
        recs = json.loads(r["recs"])
        with st.container(height=380):
            for i, paper in enumerate(recs):
                st.markdown(f"**{paper['title']}**")
                st.caption(f"{paper['categories']} · Match: {paper['similarity']:.0%}")
                if paper.get("authors"):
                    st.caption(f"_{paper['authors'][:80]}_")
                if paper.get("reason"):
                    st.info(paper["reason"])
                st.write(paper["abstract"][:250] + "…")
                if i < len(recs) - 1:
                    st.divider()


st.title("I'm Papermind")

left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.markdown("### Chat with Papermind")

    # Paper status bar — one row per uploaded paper with individual remove
    for i, paper in enumerate(st.session_state.papers):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.success(f"📄 **{paper['name']}**")
        with c2:
            if st.button("✕", key=f"remove_{i}", use_container_width=True):
                st.session_state.papers.pop(i)
                _clear_results()
                st.rerun()

    chat_box = st.container(height=420)
    with chat_box:
        if not st.session_state.chat_history:
            st.caption("No messages yet. Ask a research question or upload a paper using ＋")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='display:flex;justify-content:flex-end;margin:6px 0'>"
                    f"<div style='background:#0084ff;color:white;padding:8px 14px;"
                    f"border-radius:18px 18px 4px 18px;max-width:80%;word-wrap:break-word'>"
                    f"{msg['content']}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(msg["content"])

        if st.session_state.is_thinking:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown("_Thinking…_")

    plus_col, form_col = st.columns([1, 11])

    with plus_col:
        with st.popover("＋", use_container_width=True):
            st.markdown("**Upload PDF papers**")
            uploaded_files = st.file_uploader(
                "PDF",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="inline_uploader",
            )
            if uploaded_files:
                existing_names = {p["name"] for p in st.session_state.papers}
                new_files = [f for f in uploaded_files if f.name not in existing_names]
                if new_files:
                    for f in new_files:
                        with st.spinner(f"Extracting {f.name}…"):
                            text = extract_pdf_text(f)
                        st.session_state.papers.append({"name": f.name, "text": text[:5000]})
                    _clear_results()
                    st.rerun()
                for p in st.session_state.papers:
                    st.success(f"✓ {p['name']}")

    with form_col:
        with st.form(key="chat_input_form", clear_on_submit=True):
            user_input = st.text_input("Type your question…", label_visibility="collapsed")
            send = st.form_submit_button("Send ➤", use_container_width=True)


with right_col:
    has_results = bool(
        st.session_state.results_per_paper
        and any(r.get("summary") or r.get("gaps") or r.get("recs")
                for r in st.session_state.results_per_paper)
    )

    if not has_results:
        st.markdown("### About Papermind")
        st.markdown("""
Papermind is an AI-powered academic research assistant built on the arXiv dataset.

---

**📄 Analyse Academic Papers**
Upload any research paper and Papermind will read and understand its content.

**🏷️ Extract Metadata**
Automatically extracts authors, research domain, and key topics using TF-IDF and NER.

**📋 Summarise Findings**
Generates a concise summary of a paper's main research direction and results.

**🔍 Identify Research Gaps**
Surfaces topics and methods missing or underexplored compared to the research domain.

**📚 Recommend Relevant Papers**
Retrieves semantically relevant papers using dense embeddings.

---
""")
        if not st.session_state.papers:
            st.info("No paper uploaded — questions will search the arXiv dataset.")
    else:
        results = st.session_state.results_per_paper

        if len(results) == 1:
            _render_paper_results(results[0])
        else:
            tabs = st.tabs([r["name"] for r in results])
            for tab, r in zip(tabs, results):
                with tab:
                    _render_paper_results(r)

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        _clear_results()
        st.rerun()


if send and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
    st.session_state.pending_query = user_input.strip()
    st.session_state.is_thinking = True
    st.rerun()

if st.session_state.is_thinking:
    query = st.session_state.pending_query

    if not os.environ.get("GEMINI_API_KEY"):
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "No API key found. Please add your Gemini API key to the .env file.",
        })
        st.session_state.is_thinking = False
        st.session_state.pending_query = ""
        st.rerun()

    try:
        df, tfidf_vec, tfidf_matrix, terms, classifier, embed_model, retriever = get_backend()

        if st.session_state.papers:
            all_results = []
            reply_parts = []
            for paper in st.session_state.papers:
                report = generate_report(
                    query,
                    paper["text"],
                    paper["name"],
                    df, tfidf_vec, tfidf_matrix, terms, classifier, embed_model, retriever,
                )
                all_results.append({
                    "name": paper["name"],
                    "summary": report["summary"],
                    "gaps":    report["gaps"],
                    "recs":    report["recs"],
                })
                if report["reply"]:
                    reply_parts.append(f"**{paper['name']}:** {report['reply']}")
            st.session_state.results_per_paper = all_results
            reply_text = "\n\n".join(reply_parts) if reply_parts else "Done. See results on the right."
        else:
            report = generate_report(
                query, "", "",
                df, tfidf_vec, tfidf_matrix, terms, classifier, embed_model, retriever,
            )
            st.session_state.results_per_paper = [{
                "name": "",
                "summary": report["summary"],
                "gaps":    report["gaps"],
                "recs":    report["recs"],
            }]
            reply_text = report["reply"]

        st.session_state.chat_history.append({"role": "assistant", "content": reply_text})

    except Exception as e:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Something went wrong: {e}",
        })

    finally:
        st.session_state.is_thinking = False
        st.session_state.pending_query = ""

    st.rerun()
