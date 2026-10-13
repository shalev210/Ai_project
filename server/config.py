"""Central configuration. Every secret and tunable comes from environment
variables (loaded from the project-root .env) — nothing is hard-coded."""
import os
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT_DIR = SERVER_DIR.parent

# Load .env from the project root (local runs). In Docker, compose injects it.
load_dotenv(ROOT_DIR / ".env")


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- Paths (work both locally and inside the container at /app) ---
DATA_DIR = Path(os.getenv("DATA_DIR") or _first_existing(SERVER_DIR / "data", ROOT_DIR / "data"))
RECIPES_DIR = DATA_DIR / "recipes"
NUTRITION_FILE = DATA_DIR / "nutrition_table.json"
CLIENT_DIR = Path(os.getenv("CLIENT_DIR") or _first_existing(SERVER_DIR / "client", ROOT_DIR / "client"))

# --- Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "claude-haiku-5-5")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "claude-haiku-5-5")
ROUTER_MODE = os.getenv("ROUTER_MODE", "llm").lower()  # "llm" | "rules"
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# --- Database ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "souschef"),
    "user": os.getenv("POSTGRES_USER", "souschef"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

# --- RAG ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIM = 384
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.35"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
AUTO_INGEST = _bool("AUTO_INGEST", True)

# --- Server ---
SERVER_PORT = int(os.getenv("SERVER_PORT", "5000"))
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "6"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "2000"))
