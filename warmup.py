"""
warmup.py — Pre-loads all models and datasets at deploy time so the first
user request is instant rather than waiting 2-3 minutes for cold-start loading.

Exit codes:
  0 — all components loaded successfully
  1 — one or more components failed to load
"""

import os
import sys

# Suppress noisy transformers / tokenizer warnings in deployment logs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings
warnings.filterwarnings("ignore")

print("=== Papermind warmup starting ===")

try:
    print("[1/2] Importing backend...")
    from backend import load_dataset
    print("[1/2] Backend imported successfully.")
except Exception as e:
    print(f"[ERROR] Failed to import backend: {e}", file=sys.stderr)
    sys.exit(1)

try:
    print("[2/2] Loading arXiv dataset and building all models...")
    print("      (This includes: dataset, TF-IDF vectorizer, logistic regression")
    print("       classifier, sentence-transformer embeddings, and KNN retriever)")
    result = load_dataset()
except Exception as e:
    print(f"[ERROR] load_dataset() raised an exception: {e}", file=sys.stderr)
    sys.exit(1)

# Validate that load_dataset returned all 7 expected components
if result is None or len(result) != 7:
    print(
        f"[ERROR] load_dataset() returned unexpected value "
        f"(expected 7-tuple, got {type(result).__name__} of length "
        f"{len(result) if result is not None else 'N/A'}).",
        file=sys.stderr,
    )
    sys.exit(1)

df, tfidf_vec, tfidf_matrix, terms, classifier, embed_model, retriever = result

# Validate each component is non-None and has the expected type/shape
errors = []

if df is None or len(df) == 0:
    errors.append("dataset (df) is empty or None")

if tfidf_vec is None:
    errors.append("TF-IDF vectorizer is None")

if tfidf_matrix is None:
    errors.append("TF-IDF matrix is None")

if terms is None or len(terms) == 0:
    errors.append("TF-IDF terms array is empty or None")

if classifier is None:
    errors.append("logistic regression classifier is None")

if embed_model is None:
    errors.append("sentence-transformer embed_model is None")

if retriever is None:
    errors.append("KNN retriever is None")

if errors:
    for err in errors:
        print(f"[ERROR] Validation failed — {err}", file=sys.stderr)
    sys.exit(1)

print(f"=== Warmup complete ===")
print(f"    Papers loaded   : {len(df):,}")
print(f"    TF-IDF features : {tfidf_matrix.shape[1]:,}")
print(f"    Vocabulary size : {len(terms):,}")
print(f"    Categories      : {df['primary_category'].nunique()}")
print(f"    Embed model     : {embed_model}")
print(f"    KNN retriever   : {retriever}")
print("All models are cached and ready. Starting Streamlit...")
sys.exit(0)
