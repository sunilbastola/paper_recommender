import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def _log(msg):
    sys.stderr.write(f"[PAPERMIND] {msg}\n")
    sys.stderr.flush()

import numpy as np  # must be imported before torch
import pandas as pd
import pdfplumber
import requests
from dotenv import load_dotenv

load_dotenv()

import kagglehub
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

_log("backend.py imports complete")

MAX_PAPERS = 8000
DATA_FILE = Path(os.environ.get("ARXIV_DATA_FILE", str(Path.home() / ".cache/kagglehub/datasets/Cornell-University/arxiv/versions/288/arxiv-metadata-oai-snapshot.json")))
CACHE_DIR = Path(__file__).parent / ".cache"
EMBEDDINGS_CACHE = CACHE_DIR / f"embeddings_{MAX_PAPERS}.npy"
PAPERS_CACHE = CACHE_DIR / f"papers_{MAX_PAPERS}.parquet"
_log(f"DATA_FILE = {DATA_FILE}, exists = {DATA_FILE.exists()}")

STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","in","is",
    "it","its","of","on","or","that","the","this","to","we","with","using","used","via",
}



def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return " ".join(t for t in text.split() if len(t) > 2 and t not in STOPWORDS)


def load_dataset():
    """Load arXiv papers and build all models needed by the 5 goals."""
    # Download dataset if not already cached
    if not DATA_FILE.exists():
        print("Downloading arXiv dataset from Kaggle...")
        kagglehub.dataset_download("Cornell-University/arxiv")
        print("Dataset downloaded and cached.")

    _log("Step 1: Reading dataset file...")
    rows = []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cats = item.get("categories", "")
            if any(c.startswith(("cs.", "stat.ML")) for c in cats.split()):
                rows.append(item)
            if len(rows) >= MAX_PAPERS:
                break
    _log(f"Step 1 done: loaded {len(rows)} papers.")

    df = pd.DataFrame(rows).sample(
        n=min(MAX_PAPERS, len(rows)),
        random_state=42
    ).reset_index(drop=True)
    
    useful = ["id", "title", "authors", "abstract", "categories", "update_date"]
    df = df[[c for c in useful if c in df.columns]].dropna(
        subset=["title", "abstract", "categories"]
    )
    df["title"]    = df["title"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["abstract"] = df["abstract"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["paper_text"]        = df["title"] + ". " + df["abstract"]
    df["primary_category"]  = df["categories"].str.split().str[0]
    df["clean_text"]        = df["paper_text"].map(_preprocess)


    tfidf_vec    = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=3, max_df=0.75)
    tfidf_matrix = tfidf_vec.fit_transform(df["clean_text"])
    terms        = np.array(tfidf_vec.get_feature_names_out())

    def _top_kw(idx: int, k: int = 6) -> list[str]:
        row = tfidf_matrix[idx].toarray().ravel()
        return terms[row.argsort()[-k:][::-1]].tolist() if row.sum() else []

    df["keywords"] = [_top_kw(i) for i in range(len(df))]


    common_cats = df["primary_category"].value_counts().head(8).index
    clf_data    = df[df["primary_category"].isin(common_cats)]
    X_train, _, y_train, _ = train_test_split(
        clf_data["clean_text"], clf_data["primary_category"],
        test_size=0.25, random_state=42, stratify=clf_data["primary_category"],
    )
    classifier = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    classifier.fit(X_train, y_train)


    _log("Step 3: Loading sentence-transformer model...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if EMBEDDINGS_CACHE.exists():
        _log("Step 4: Loading cached embeddings (fast)...")
        embeddings = np.load(str(EMBEDDINGS_CACHE))
    else:
        _log("Step 4: Encoding papers (first run, takes ~2 min)...")
        CACHE_DIR.mkdir(exist_ok=True)
        raw = embed_model.encode(
            df["paper_text"].tolist(), batch_size=64,
            show_progress_bar=True, normalize_embeddings=True,
            convert_to_tensor=True,
        )
        embeddings = np.array(raw.tolist(), dtype=np.float32)
        np.save(str(EMBEDDINGS_CACHE), embeddings)
        _log("Step 4 done: embeddings saved to cache.")

    retriever = NearestNeighbors(n_neighbors=8, metric="cosine")
    retriever.fit(embeddings)

    return df, tfidf_vec, tfidf_matrix, terms, classifier, embed_model, retriever


_last_llm_call: float = 0.0
_MIN_CALL_INTERVAL = 13.0  # seconds between calls (free tier: 5 RPM = 12s minimum)

def call_llm(system: str, messages: list[dict], model: str = "gemini-2.5-flash") -> str:
    global _last_llm_call
    # Throttle: wait if last call was too recent
    elapsed = time.time() - _last_llm_call
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
    }
    for attempt in range(4):
        _last_llm_call = time.time()
        response = requests.post(
            url,
            params={"key": os.environ["GEMINI_API_KEY"]},
            json=body,
        )
        if response.status_code in (429, 503):
            wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s
            _log(f"Rate limited (attempt {attempt+1}), retrying in {wait}s…")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError(
        "Gemini API rate limit exceeded. Your free-tier daily quota may be exhausted. "
        "Get a new key at https://aistudio.google.com/apikey and update .env."
    )


_INTENT_SYSTEM = """You are an intent classifier for an academic research assistant called Papermind.

Given the user's message, reply with ONLY a comma-separated list of these labels (no explanation):
  SUMMARY        – user wants a summary or overview of the paper / findings
  GAPS           – user wants research gaps, limitations, or future work identified
  RECOMMENDATIONS – user wants related or similar paper recommendations
  GENERAL        – user has a general question not covered by the above

Rules:
- Use multiple labels if the request covers more than one (e.g. SUMMARY,GAPS).
- If the user says "analyse" or "full analysis" or "everything", reply: SUMMARY,GAPS,RECOMMENDATIONS
- If none of the specific labels apply, reply: GENERAL

Examples:
  "summarise the paper"              → SUMMARY
  "what are the research gaps?"      → GAPS
  "recommend similar papers"         → RECOMMENDATIONS
  "summarise and find the gaps"      → SUMMARY,GAPS
  "give me a full analysis"          → SUMMARY,GAPS,RECOMMENDATIONS
  "what method did the authors use?" → GENERAL
"""


def _classify_intent(query: str) -> set[str]:
    """Classify the user's intent via Gemini. Returns a set of intent labels."""
    raw = call_llm(_INTENT_SYSTEM, [{"role": "user", "content": query}])
    return {label.strip().upper() for label in raw.split(",")}


def generate_report(
    query: str,
    paper_text: str,
    paper_name: str,
    df: pd.DataFrame,
    tfidf_vec: TfidfVectorizer,
    tfidf_matrix,
    terms: np.ndarray,
    classifier: Pipeline,
    embed_model,
    retriever,
) -> dict:
    """
    Detects the user's intent, then runs only the matching function(s).
    Returns {"reply": str, "summary": str, "gaps": str, "recs": str}.
    """
    intents = _classify_intent(query)

    wants_summary = "SUMMARY" in intents
    wants_gaps    = "GAPS" in intents
    wants_recs    = "RECOMMENDATIONS" in intents
    is_general    = "GENERAL" in intents or not intents

    # Clear all fields — only populate what was requested
    result = {"reply": "", "summary": "", "gaps": "", "recs": ""}

    _paper_ref_phrases = {"this paper", "the paper", "uploaded", "attached", "referring to", "the document", "the file"}
    _generic_rec_words = {"recommend", "paper", "similar", "related", "find", "show", "suggest", "other", "more", "what", "can", "you", "me", "please", "some", "any", "another", "give", "list", "papers"}
    query_lower = query.lower()
    refers_to_paper = paper_text and any(p in query_lower for p in _paper_ref_phrases)
    query_has_topic = len(set(query_lower.split()) - _generic_rec_words) > 2
    search_query = paper_text[:1500] if (refers_to_paper or (paper_text and not query_has_topic)) else query

    rec = recommend_papers(
        search_query, df, tfidf_vec, tfidf_matrix, terms,
        classifier, embed_model, retriever,
        top_k=8 if paper_text else 4,
    )

    if paper_text:
        analysis = analyse_paper(paper_text, classifier, tfidf_vec, terms)
        context_block = (
            f'Uploaded paper: "{paper_name}"\n'
            f'Domain: {analysis["domain"]}\n'
            f'Keywords: {", ".join(analysis["keywords"])}\n\n'
            f'Content:\n{paper_text}'
        )
    else:
        context_block = (
            "No paper uploaded. Using retrieved arXiv papers:\n\n"
            + "\n\n---\n\n".join(
                f"Paper ID: {r['id']}\nTitle: {r['title']}\n"
                f"Keywords: {', '.join(r['keywords'])}\nAbstract: {r['abstract'][:600]}"
                for _, r in rec["advanced_recommendations"].iterrows()
            )
        )

    actions = []

    if wants_summary:
        summ = summarise_findings(query, rec["advanced_recommendations"], tfidf_vec, terms, classifier, paper_text=paper_text)
        result["summary"] = summ["advanced"]
        actions.append("summarised the paper" if paper_text else "summarised the relevant findings")

    if wants_gaps:
        gaps = identify_research_gaps(
            query, rec["advanced_recommendations"], df, tfidf_vec, terms, classifier, paper_text=paper_text
        )
        result["gaps"] = gaps["advanced_gaps"]
        actions.append("identified the research gaps")

    if wants_recs:
        if paper_text:
            reranked = rerank_recommendations(paper_text, rec["advanced_recommendations"])
            recs_data = reranked if reranked else [
                {"id": r["id"], "title": r["title"], "authors": str(r.get("authors", "")),
                 "categories": r["categories"], "similarity": float(r["similarity"]),
                 "abstract": str(r.get("abstract", ""))[:400], "reason": ""}
                for _, r in rec["advanced_recommendations"].iterrows()
            ]
        else:
            recs_data = [
                {"id": r["id"], "title": r["title"], "authors": str(r.get("authors", "")),
                 "categories": r["categories"], "similarity": float(r["similarity"]),
                 "abstract": str(r.get("abstract", ""))[:400], "reason": ""}
                for _, r in rec["advanced_recommendations"].iterrows()
            ]
        result["recs"] = json.dumps(recs_data)
        actions.append("found relevant paper recommendations")

    if actions:
        action_str = " and ".join(actions)
        result["reply"] = f"I have {action_str}. Please see the result below."
    else:
        # General question — answer conversationally via LLM
        system = (
            "You are Papermind, an academic research assistant specialising in AI and NLP papers.\n"
            + context_block
            + "\n\nBe concise and cite paper IDs or sections when relevant."
        )
        result["reply"] = call_llm(system, [{"role": "user", "content": query}])

    return result




def analyse_paper(
    text: str,
    classifier: Pipeline,
    tfidf_vec: TfidfVectorizer,
    terms: np.ndarray,
) -> dict:
    """
    Classify a paper into its research domain and extract its top keywords.
    Combines TF-IDF keyword extraction (basic) with LR classification (basic).
    """
    clean = _preprocess(text)
    domain = classifier.predict([clean])[0]

    vec = tfidf_vec.transform([clean])
    row = vec.toarray().ravel()
    keywords = terms[row.argsort()[-8:][::-1]].tolist() if row.sum() else []

    return {"domain": domain, "keywords": keywords}



INSTITUTION_PATTERN = re.compile(
    r'\b(?:University|Institute|Laboratory|Lab|College|School|Centre|Center)'
    r'\s+of\s+[A-Z][A-Za-z\-]+|'
    r'\b[A-Z][A-Za-z\-]+\s+'
    r'(?:University|Institute|Laboratory|Lab|College|School|Centre|Center)\b'
)


def extract_metadata(paper_row: pd.Series) -> dict:
    """
    Extract structured metadata from a paper:
    authors (rule-based), institutions (regex NER), keywords (TF-IDF already in row).
    """
    authors = [
        name.strip()
        for name in re.split(r',| and ', paper_row.get("authors", "") or "")
        if name.strip()
    ]
    institutions = sorted(set(
        m.group(0)
        for m in INSTITUTION_PATTERN.finditer(paper_row.get("abstract", "") or "")
    ))
    return {
        "title":        paper_row["title"],
        "authors":      authors,
        "institutions": institutions,
        "keywords":     paper_row.get("keywords", []),
        "domain":       paper_row.get("primary_category", ""),
        "categories":   paper_row.get("categories", ""),
    }



def summarise_findings(
    query: str,
    results: pd.DataFrame,
    tfidf_vec: TfidfVectorizer,
    terms: np.ndarray,
    classifier: Pipeline,
    paper_text: str = "",
) -> dict:
    if paper_text:
        basic_summary = _textrank_summary(paper_text, n=3)
        prompt = (
            f"Summarise the following research paper.\n\n"
            f"[FORMAT]\n## Summary\n<3-4 sentences covering the main theme, methods, and key findings>\n\n"
            f"[CONSTRAINTS]\n- Only use information from the paper.\n- Be specific about contributions and results.\n\n"
            f"[Key sentences pre-extracted by TextRank]\n{basic_summary}\n\n"
            f"[PAPER CONTENT]\n{paper_text[:4000]}"
        )
        advanced_summary = call_llm(
            system="You are an expert academic research assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
    else:
        combined_text = " ".join(results["abstract"].tolist())
        basic_summary = _textrank_summary(combined_text, n=3)
        enriched_ctx = _build_enriched_context(query, results, tfidf_vec, terms, classifier)
        assisted_ctx = (
            f"{enriched_ctx}\n\n"
            f"[Key sentences pre-extracted by TextRank]\n{basic_summary}"
        )
        advanced_summary = call_llm(
            system="You are an expert academic research assistant.",
            messages=[{"role": "user", "content": _instruction_prompt(query, assisted_ctx)}],
        )

    return {"basic": basic_summary, "advanced": advanced_summary}



def identify_research_gaps(
    query: str,
    results: pd.DataFrame,
    df: pd.DataFrame,
    tfidf_vec: TfidfVectorizer,
    terms: np.ndarray,
    classifier: Pipeline,
    paper_text: str = "",
) -> dict:
    source_text = paper_text if paper_text else query
    query_domain = classifier.predict([_preprocess(source_text)])[0]

    domain_papers = df[df["primary_category"] == query_domain]
    domain_tfidf  = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=2)
    domain_matrix = domain_tfidf.fit_transform(domain_papers["clean_text"])
    domain_scores = np.asarray(domain_matrix.mean(axis=0)).ravel()
    domain_terms  = np.array(domain_tfidf.get_feature_names_out())
    top_domain_kw = set(domain_terms[domain_scores.argsort()[-30:][::-1]])

    vec      = tfidf_vec.transform([_preprocess(source_text)])
    row      = vec.toarray().ravel()
    query_kw = set(terms[row.argsort()[-20:][::-1]].tolist()) if row.sum() else set()
    basic_gaps = sorted(top_domain_kw - query_kw)[:10]

    if paper_text:
        context = f"Paper content:\n{paper_text[:3000]}"
    else:
        context = _build_enriched_context(query, results, tfidf_vec, terms, classifier)

    assisted_ctx = (
        f"{context}\n\n"
        f"[Keyword-level gaps pre-identified by TF-IDF domain comparison]\n"
        f"{', '.join(basic_gaps)}"
    )
    advanced_gaps = call_llm(
        system="You are a research analyst.",
        messages=[{"role": "user", "content": _gap_cot_prompt(query, assisted_ctx)}],
    )
    for marker in ("**Final Answer:**", "Final Answer:"):
        if marker in advanced_gaps:
            advanced_gaps = advanced_gaps.split(marker, 1)[1].strip()
            break

    return {
        "domain":        query_domain,
        "basic_gaps":    basic_gaps,
        "advanced_gaps": advanced_gaps,
    }



def recommend_papers(
    query: str,
    df: pd.DataFrame,
    tfidf_vec: TfidfVectorizer,
    tfidf_matrix,
    terms: np.ndarray,
    classifier: Pipeline,
    embed_model: SentenceTransformer,
    retriever: NearestNeighbors,
    top_k: int = 5,
) -> dict:
    """
    Recommend papers using:
    - Basic: TF-IDF cosine similarity (keyword match)
    - Advanced: Dense embedding retrieval + RAG instruction prompt
    Basic output feeds into advanced — TF-IDF candidate IDs are injected
    into the enriched context so the LLM can cross-reference both methods.
    """

    query_clean = _preprocess(query)

    # TF-IDF candidate retrieval: exact keyword matching in the paper corpus.
    query_vec = tfidf_vec.transform([query_clean])
    if query_vec.nnz == 0:
        basic_recs = df.iloc[[]][["id", "title", "categories", "keywords"]].copy()
        basic_recs["similarity"] = np.array([], dtype=float)
        tfidf_ids = []
        tfidf_scores = {}
    else:
        scores = cosine_similarity(query_vec, tfidf_matrix).ravel()
        top_idx = scores.argsort()[-top_k:][::-1]
        basic_recs = df.iloc[top_idx][["id", "title", "categories", "keywords"]].copy()
        basic_recs["similarity"] = scores[top_idx]
        tfidf_ids = basic_recs["id"].tolist()
        tfidf_scores = dict(zip(tfidf_ids, basic_recs["similarity"].tolist()))

    # Embedding retrieval: semantic matching using dense sentence embeddings.
    qvec = np.array(
        embed_model.encode([query], normalize_embeddings=True, convert_to_tensor=True).tolist(),
        dtype=np.float32,
    )
    dists, idxs = retriever.kneighbors(qvec, n_neighbors=top_k)
    embed_results = df.iloc[idxs[0]].copy()
    embed_results["similarity"] = 1 - dists[0]
    embed_scores = dict(zip(embed_results["id"].tolist(), embed_results["similarity"].tolist()))
    embed_results = embed_results[
        ["id", "title", "authors", "categories", "keywords", "similarity", "abstract"]
    ]

    # Combine both candidate sets to reduce cases where one retriever misses relevant papers.
    combined_ids = []
    for rid in tfidf_ids:
        if rid not in combined_ids:
            combined_ids.append(rid)
    for rid in embed_results["id"].tolist():
        if rid not in combined_ids:
            combined_ids.append(rid)

    combined_results = df.set_index("id").loc[combined_ids].reset_index()
    combined_results["similarity"] = combined_results["id"].map(embed_scores).fillna(
        combined_results["id"].map(tfidf_scores)
    ).fillna(0.0).astype(float)
    combined_results = combined_results[
        ["id", "title", "authors", "categories", "keywords", "similarity", "abstract"]
    ]
    combined_results = combined_results.sort_values(by="similarity", ascending=False).head(top_k)

    enriched_ctx = _build_enriched_context(query, combined_results, tfidf_vec, terms, classifier)
    assisted_ctx = (
        f"{enriched_ctx}\n\n"
        f"[Papers also flagged by TF-IDF keyword matching]\n"
        f"{', '.join(tfidf_ids)}"
    )
    llm_answer = call_llm(
        system="You are an academic research assistant.",
        messages=[{"role": "user", "content": _rag_instruction_prompt(query, assisted_ctx)}],
    )

    return {
        "basic_recommendations":    basic_recs,
        "advanced_recommendations": combined_results,
        "llm_answer":               llm_answer,
    }


def rerank_recommendations(
    paper_text: str,
    candidates: pd.DataFrame,
) -> list[dict]:
    """
    Cross-reference function: re-ranks candidate papers by relevance to the
    uploaded paper using Gemini scoring (0-10). Returns only papers scoring >= 5,
    sorted by score descending.
    """
    candidate_blocks = "\n---\n".join(
        f"Paper ID: {r['id']}\nTitle: {r['title']}\n"
        f"Categories: {r['categories']}\nAbstract: {str(r.get('abstract', ''))[:400]}"
        for _, r in candidates.iterrows()
    )
    prompt = f"""You are an expert academic relevance evaluator.

Uploaded paper (excerpt):
{paper_text[:1200]}

For each candidate paper below, score its relevance to the uploaded paper from 0 to 10.
Consider shared methodology, related topics, complementary findings, or directly citable work.

Candidates:
{candidate_blocks}

Reply with ONLY this format for each paper (no extra text):
Paper ID: <id>
Score: <0-10>
Reason: <one sentence why it is or isn't relevant>
---"""

    response = call_llm(
        system="You are a precise academic relevance evaluator. Follow the output format exactly.",
        messages=[{"role": "user", "content": prompt}],
    )

    scored = []
    for block in response.split("---"):
        block = block.strip()
        if not block:
            continue
        lines = {}
        for line in block.split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                lines[key.strip()] = val.strip()
        paper_id = lines.get("Paper ID", "").strip()
        reason = lines.get("Reason", "")
        try:
            score = float(lines.get("Score", "0"))
        except ValueError:
            score = 0.0
        if paper_id and score >= 5:
            scored.append({"id": paper_id, "score": score, "reason": reason})

    scored.sort(key=lambda x: -x["score"])
    score_map = {s["id"]: s for s in scored}

    results = []
    for _, r in candidates.iterrows():
        if r["id"] in score_map:
            results.append({
                "id": r["id"],
                "title": r["title"],
                "authors": str(r.get("authors", "")),
                "categories": r["categories"],
                "similarity": score_map[r["id"]]["score"] / 10,
                "abstract": str(r.get("abstract", ""))[:400],
                "reason": score_map[r["id"]]["reason"],
            })
    return results


def _textrank_summary(text: str, n: int = 2) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    if len(sentences) <= n:
        return text
    vecs    = TfidfVectorizer().fit_transform(sentences)
    scores  = cosine_similarity(vecs).sum(axis=1)
    top     = sorted(scores.argsort()[-n:])
    return ' '.join(sentences[i] for i in top)


def _build_enriched_context(
    query: str,
    results: pd.DataFrame,
    tfidf_vec: TfidfVectorizer,
    terms: np.ndarray,
    classifier: Pipeline,
) -> str:
    vec          = tfidf_vec.transform([_preprocess(query)])
    row          = vec.toarray().ravel()
    kw           = terms[row.argsort()[-8:][::-1]].tolist() if row.sum() else []
    domain       = classifier.predict([_preprocess(query)])[0]
    rag_ctx      = _build_rag_context(results)
    return f"Query domain: {domain}\nQuery keywords: {', '.join(kw)}\n\n{rag_ctx}"


def _build_rag_context(results: pd.DataFrame, max_chars: int = 700) -> str:
    blocks = []
    for _, r in results.iterrows():
        blocks.append(
            f"Paper ID: {r['id']}\nTitle: {r['title']}\nAuthors: {r['authors']}\n"
            f"Categories: {r['categories']}\nKeywords: {', '.join(r['keywords'])}\n"
            f"Abstract: {r['abstract'][:max_chars]}"
        )
    return "\n\n---\n\n".join(blocks)


def _instruction_prompt(topic: str, context: str) -> str:
    return f"""[ROLE]
You are an expert academic research assistant specialising in {topic}.

[TASK]
Summarise what the retrieved papers say about: {topic}
Pay attention to the Query domain and Query keywords at the top of the context.

[FORMAT]
## Summary
<3 sentences covering the main theme, methods used, and key findings>

[CONSTRAINTS]
- Only use information from the context.
- Be specific, not generic.

[CONTEXT]
{context}"""


def _gap_cot_prompt(topic: str, context: str) -> str:
    return f"""You are a research analyst. Think step by step before writing your answer.

<scratchpad>
Step 1 – Note the query domain and keywords at the top of the context.
Step 2 – Read each paper and note its core contribution in one sentence.
Step 3 – Check which query keywords are NOT covered by any paper.
Step 4 – Identify what topics in this domain none of the papers address.
</scratchpad>

After the scratchpad, write **Final Answer:** with 2 specific research gaps about: {topic}

Papers:
{context}"""


def _rag_instruction_prompt(topic: str, context: str) -> str:
    return f"""You are an academic research assistant.
Use only the retrieved paper context below.

Task:
1. Summarise the main research direction in 4 bullet points.
2. Recommend the 3 most relevant papers and explain why.
3. Identify 2 likely research gaps related to the query keywords.
4. Suggest one future research question in the query domain.

Grounding rules:
- Cite paper IDs from the context.
- Use the Query domain and Query keywords to focus your answer.
- Do not invent papers, authors, or results.

User question: {topic}

Retrieved context:
{context}

Answer:""".strip()


def extract_pdf_text(uploaded_file) -> str:
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages[:10])
