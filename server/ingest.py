"""Knowledge-base ingestion: data/recipes/*.md -> chunks -> embeddings -> pgvector.

Chunking strategy (structure-aware, two stages):
  1. MarkdownHeaderTextSplitter splits each file on its "#" / "##" headers, so a
     chunk never mixes e.g. "Ingredients" with "Storage". Recipe sections are
     the natural unit of meaning for questions like "how long does hummus keep?".
  2. RecursiveCharacterTextSplitter (700 chars, 100 overlap) further splits any
     section that is still too long, preferring paragraph/line boundaries.
Each chunk is prefixed with "Recipe: <title> | Section: <section>" (a contextual
chunk header). Without it, a chunk like "Keeps 4 days in the fridge" would not
say WHICH recipe it belongs to, and retrieval quality would drop.

Run manually:   python ingest.py          (re-indexes from scratch)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config
import db
import rag

log = logging.getLogger(__name__)

_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True
)
_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def build_chunks(recipes_dir: Path | None = None) -> list[dict]:
    recipes_dir = Path(recipes_dir or config.RECIPES_DIR)
    files = sorted(recipes_dir.glob("*.md"))
    chunks: list[dict] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        idx = 0
        for doc in _header_splitter.split_text(text):
            title = doc.metadata.get("title", path.stem)
            section = doc.metadata.get("section", "Overview")
            for piece in _char_splitter.split_text(doc.page_content):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append({
                    "source": path.name,
                    "title": title,
                    "section": section,
                    "chunk_index": idx,
                    "content": f"Recipe: {title} | Section: {section}\n{piece}",
                })
                idx += 1
    return chunks


def ingest(recipes_dir: Path | None = None) -> int:
    chunks = build_chunks(recipes_dir)
    if not chunks:
        log.warning("No documents found in %s — knowledge base is empty", recipes_dir or config.RECIPES_DIR)
        return 0
    vectors = rag.embed([c["content"] for c in chunks])
    for c, v in zip(chunks, vectors):
        c["embedding"] = v
    n = db.replace_chunks(chunks)
    log.info("Ingested %d chunks from %d files", n, len({c['source'] for c in chunks}))
    return n


def ensure_ingested() -> None:
    """Called at server start-up: index once if the vector table is empty."""
    try:
        if db.count_chunks() == 0:
            log.info("Vector store empty — ingesting knowledge base…")
            ingest()
    except Exception:
        # The server must still start (e.g. model download offline); /chat
        # will then report the RAG path as unavailable instead of crashing.
        log.exception("Auto-ingestion failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = ingest(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(f"Indexed {count} chunks.")
