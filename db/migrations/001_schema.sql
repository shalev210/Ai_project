-- =====================================================================
-- SousChef — schema migration 001
-- PostgreSQL 16 + pgvector. Runs automatically on first container start
-- (mounted into /docker-entrypoint-initdb.d by db/Dockerfile).
-- Idempotent: safe to re-run by hand with psql -f.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------
-- Conversation history (per session)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id           BIGSERIAL PRIMARY KEY,
    session_id   VARCHAR(64)  NOT NULL,
    role         VARCHAR(16)  NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT         NOT NULL,
    route        VARCHAR(16)  CHECK (route IN ('rag', 'tool', 'direct', 'error')),
    route_reason TEXT,
    tool_name    VARCHAR(64),
    sources      JSONB        NOT NULL DEFAULT '[]'::jsonb,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversations_session
    ON conversations (session_id, created_at);

-- ---------------------------------------------------------------------
-- Tool-call log: which tool, with what input, what it returned
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tool_calls (
    id           BIGSERIAL PRIMARY KEY,
    session_id   VARCHAR(64)  NOT NULL,
    tool_name    VARCHAR(64)  NOT NULL,
    tool_input   JSONB        NOT NULL,
    tool_output  JSONB,
    success      BOOLEAN      NOT NULL,
    error        TEXT,
    duration_ms  INTEGER,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session
    ON tool_calls (session_id, created_at);

-- ---------------------------------------------------------------------
-- RAG vector store (pgvector). 384 dims = paraphrase-multilingual-MiniLM-L12-v2 (Hebrew + English)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id           BIGSERIAL PRIMARY KEY,
    source       VARCHAR(255) NOT NULL,   -- file name, e.g. shakshuka.md
    title        VARCHAR(255),            -- recipe title
    section      VARCHAR(255),            -- e.g. "Ingredients", "Method"
    chunk_index  INTEGER      NOT NULL,
    content      TEXT         NOT NULL,
    embedding    vector(384)  NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (source, chunk_index)
);
-- HNSW index for fast cosine-distance search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- Evaluation results (router accuracy + retrieval hit-rate runs)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluation_results (
    id              BIGSERIAL PRIMARY KEY,
    run_id          VARCHAR(64)  NOT NULL,
    query           TEXT         NOT NULL,
    expected_route  VARCHAR(16),
    actual_route    VARCHAR(16),
    expected_source VARCHAR(255),
    retrieved_sources JSONB,
    route_correct   BOOLEAN,
    retrieval_hit   BOOLEAN,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluation_results (run_id);
