from __future__ import annotations

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


# Load models immediately at startup so they are cached in Streamlit's process
# before the first user request arrives. Without this, @st.cache_resource only
# populates on the first request, causing a 2-3 minute cold-start delay.
get_backend()

for key, default in [
    ("chat_history", []),
    ("paper_text", ""),
    ("paper_name", ""),
    ("result_summary", ""),
    ("result_gaps", ""),
    ("result_recs", ""),
    ("is_thinking", False),
    ("pending_query", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


st.title("I'm Papermind")


left_col, right_col = st.columns([1, 1.2], gap="large")


with left_col:
    st.markdown("### Chat with Papermind")

    # Paper status bar
    if st.session_state.paper_name:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.success(f"📄 **{st.session_state.paper_name}**")
        with c2:
            if st.button("✕", use_container_width=True):
                st.session_state.paper_text = ""
                st.session_state.paper_name = ""
                st.session_state.chat_history = []
                st.session_state.result_summary = ""
                st.session_state.result_gaps = ""
                st.session_state.result_recs = ""
                st.rerun()

    # Chat messages display
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
            st.markdown("**Upload a PDF paper**")
            uploaded_inline = st.file_uploader(
                "PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="inline_uploader",
            )
            if uploaded_inline:
                if uploaded_inline.name != st.session_state.paper_name:
                    with st.spinner("Extracting text…"):
                        text = extract_pdf_text(uploaded_inline)
                    st.session_state.paper_text = text[:5000]
                    st.session_state.paper_name = uploaded_inline.name
                    st.session_state.chat_history = []
                    st.session_state.result_summary = ""
                    st.session_state.result_gaps = ""
                    st.session_state.result_recs = ""
                st.success(f"✓ {st.session_state.paper_name}")

    with form_col:
        with st.form(key="chat_input_form", clear_on_submit=True):
            user_input = st.text_input("Type your question…", label_visibility="collapsed")
            send = st.form_submit_button("Send ➤", use_container_width=True)


with right_col:
    st.markdown("### About Papermind")
    st.markdown("""
Papermind is an AI-powered academic research assistant built on the arXiv dataset.

---

**📄 Analyse Academic Papers**
Upload any research paper and Papermind will read and understand its content, identifying the core contributions, methodology, and findings.

**🏷️ Extract Metadata**
Automatically extracts structured information including authors, research domain, and key topics using TF-IDF keyword extraction and Named Entity Recognition.

**📋 Summarise Findings**
Generates a concise, human-readable summary of a paper's main research direction and results — going beyond copying sentences to synthesising meaning.

**🔍 Identify Research Gaps**
Compares the paper's coverage against its research domain to surface topics and methods that are missing or underexplored.

**📚 Recommend Relevant Papers**
Retrieves the most semantically relevant papers from the arXiv database using dense embeddings, then explains why each paper is a good match.

---
""")

    if not st.session_state.paper_name:
        st.info("No paper uploaded — questions will search the arXiv dataset.")

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.result_summary = ""
        st.session_state.result_gaps = ""
        st.session_state.result_recs = ""
        st.rerun()

st.markdown("---")
st.markdown("### Analysis Results")

if st.session_state.result_summary or st.session_state.result_gaps or st.session_state.result_recs:
    res_col1, res_col2, res_col3 = st.columns(3, gap="medium")

    with res_col1:
        st.markdown("#### 📋 Summary")
        if st.session_state.result_summary:
            st.write(st.session_state.result_summary)
        else:
            st.caption("No summary yet.")

    with res_col2:
        st.markdown("#### 🔍 Research Gaps")
        if st.session_state.result_gaps:
            st.write(st.session_state.result_gaps)
        else:
            st.caption("No gaps identified yet.")

    with res_col3:
        st.markdown("#### 📚 Recommended Papers")
        if st.session_state.result_recs:
            st.write(st.session_state.result_recs)
        else:
            st.caption("No recommendations yet.")
else:
    st.write("Analysis results will appear here once you ask a question.")

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

        report = generate_report(
            query,
            st.session_state.paper_text,
            st.session_state.paper_name,
            df, tfidf_vec, tfidf_matrix, terms, classifier, embed_model, retriever,
        )

        st.session_state.result_summary = report["summary"]
        st.session_state.result_gaps    = report["gaps"]
        st.session_state.result_recs    = report["recs"]
        st.session_state.chat_history.append({"role": "assistant", "content": report["reply"]})



    except Exception as e:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Something went wrong: {e}",
        })

    finally:
        st.session_state.is_thinking = False
        st.session_state.pending_query = ""

    st.rerun()
