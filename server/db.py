"""Data-access layer for PostgreSQL (+pgvector). All SQL lives here."""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

import config


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(**config.DB_CONFIG, row_factory=dict_row, connect_timeout=5)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ----------------------------------------------------------------- health
def ping() -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 AS ok, (SELECT count(*) FROM document_chunks) AS chunks"
        ).fetchone()
    return {"ok": row["ok"] == 1, "chunks": row["chunks"]}


# ---------------------------------------------------------- conversations
def save_message(
    session_id: str,
    role: str,
    content: str,
    route: str | None = None,
    route_reason: str | None = None,
    tool_name: str | None = None,
    sources: list | None = None,
) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """INSERT INTO conversations
               (session_id, role, content, route, route_reason, tool_name, sources)
               VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb) RETURNING id""",
            (session_id, role, content, route, route_reason, tool_name,
             json.dumps(sources or [])),
        ).fetchone()
    return row["id"]


def get_history(session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Most recent `limit` turns, returned oldest-first."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, role, content, route, route_reason, tool_name, sources, created_at
               FROM conversations WHERE session_id = %s
               ORDER BY created_at DESC, id DESC LIMIT %s""",
            (session_id, limit),
        ).fetchall()
    rows.reverse()
    for r in rows:
        r["created_at"] = r["created_at"].isoformat()
    return rows


# -------------------------------------------------------------- tool logs
def log_tool_call(session_id: str, tool_name: str, tool_input: dict, tool_output: Any,
                  success: bool, error: str | None, duration_ms: int) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO tool_calls
               (session_id, tool_name, tool_input, tool_output, success, error, duration_ms)
               VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)""",
            (session_id, tool_name, json.dumps(tool_input),
             json.dumps(tool_output) if tool_output is not None else None,
             success, error, duration_ms),
        )


def get_tool_calls(session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT tool_name, tool_input, tool_output, success, error, duration_ms, created_at
               FROM tool_calls WHERE session_id = %s ORDER BY id DESC LIMIT %s""",
            (session_id, limit),
        ).fetchall()
    for r in rows:
        r["created_at"] = r["created_at"].isoformat()
    return rows


# ------------------------------------------------------------ vector store
def count_chunks() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT count(*) AS n FROM document_chunks").fetchone()["n"]


def replace_chunks(chunks: list[dict[str, Any]]) -> int:
    """Re-index from scratch (keeps ingestion idempotent)."""
    with get_conn() as conn:
        conn.execute("TRUNCATE document_chunks RESTART IDENTITY")
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO document_chunks
                   (source, title, section, chunk_index, content, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                [(c["source"], c["title"], c["section"], c["chunk_index"],
                  c["content"], c["embedding"]) for c in chunks],
            )
    return len(chunks)


def search_chunks(embedding, top_k: int) -> list[dict[str, Any]]:
    """Cosine similarity search. pgvector's <=> is cosine *distance*."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT source, title, section, chunk_index, content,
                      1 - (embedding <=> %s) AS similarity
               FROM document_chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (embedding, embedding, top_k),
        ).fetchall()
    for r in rows:
        r["similarity"] = round(float(r["similarity"]), 4)
    return rows


# -------------------------------------------------------------- evaluation
def save_eval_result(run_id: str, query: str, expected_route: str | None, actual_route: str,
                     expected_source: str | None, retrieved_sources: list[str],
                     route_correct: bool, retrieval_hit: bool | None) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO evaluation_results
               (run_id, query, expected_route, actual_route, expected_source,
                retrieved_sources, route_correct, retrieval_hit)
               VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)""",
            (run_id, query, expected_route, actual_route, expected_source,
             json.dumps(retrieved_sources), route_correct, retrieval_hit),
        )
