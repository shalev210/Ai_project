"""RAG Module — embed the query, retrieve the closest chunks from pgvector,
and return them with cosine-similarity scores.

Embeddings: fastembed (ONNX, runs locally on CPU, no extra API key) with the
multilingual model paraphrase-multilingual-MiniLM-L12-v2 (384 dims), so Hebrew
and English questions both match the English recipe collection.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any

import numpy as np

import config
import db

log = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()

# "fastembed" (default) or "hash" — a deterministic bag-of-words embedder used
# ONLY by the offline test-suite, where the model cannot be downloaded.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "fastembed").lower()


class RAGError(Exception):
    pass


def _hash_embed(text: str) -> np.ndarray:
    vec = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    for tok in text.lower().split():
        tok = "".join(ch for ch in tok if ch.isalnum())
        if tok:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % config.EMBEDDING_DIM] += 1.0
    return vec


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding
                log.info("Loading embedding model %s (first run downloads ~220MB)", config.EMBEDDING_MODEL)
                _model = TextEmbedding(model_name=config.EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> list[np.ndarray]:
    """Return L2-normalised embeddings (so dot product == cosine similarity)."""
    if EMBEDDING_BACKEND == "hash":
        raw = [_hash_embed(t) for t in texts]
    else:
        try:
            raw = list(_get_model().embed(texts))
        except Exception as e:
            raise RAGError(f"Embedding model unavailable: {e}") from e
    out = []
    for v in raw:
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        out.append(v / n if n else v)
    return out


def retrieve(query: str, top_k: int | None = None, min_similarity: float | None = None) -> dict[str, Any]:
    """Returns {"chunks": [...], "all_scores": [...], "empty_kb": bool}.

    `chunks` holds only results above the similarity threshold; `all_scores`
    keeps every candidate so the router/UI can explain low-relevance cases.
    """
    top_k = top_k or config.RAG_TOP_K
    min_similarity = config.RAG_MIN_SIMILARITY if min_similarity is None else min_similarity

    if db.count_chunks() == 0:
        return {"chunks": [], "all_scores": [], "empty_kb": True}

    q_vec = embed([query])[0]
    rows = db.search_chunks(q_vec, top_k)
    relevant = [r for r in rows if r["similarity"] >= min_similarity]
    return {
        "chunks": relevant,
        "all_scores": [{"source": r["source"], "section": r["section"], "similarity": r["similarity"]} for r in rows],
        "empty_kb": False,
    }


def format_context(chunks: list[dict]) -> str:
    """Numbered context block the LLM cites as [1], [2], ..."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {c['source']} — {c['title']} / {c['section']}, "
                     f"similarity {c['similarity']:.2f})\n{c['content']}")
    return "\n\n".join(parts)


def sources_for_response(chunks: list[dict]) -> list[dict]:
    return [
        {"ref": i, "source": c["source"], "title": c["title"], "section": c["section"],
         "similarity": c["similarity"], "snippet": c["content"][:220]}
        for i, c in enumerate(chunks, 1)
    ]
