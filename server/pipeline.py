"""The mandatory AI pipeline, in order:

    USER QUERY → AGENT ROUTER → RAG / TOOLS → LLM → RESPONSE

No request reaches the LLM without first passing through the router.
"""
from __future__ import annotations

import logging
import time

import config
import db
import formatter
import llm
import rag
import router

log = logging.getLogger(__name__)


class PipelineError(Exception):
    def __init__(self, code: str, message: str, http_status: int, route: str = "error"):
        super().__init__(message)
        self.code, self.message, self.http_status, self.route = code, message, http_status, route


def run(query: str, session_id: str) -> dict:
    t0 = time.perf_counter()
    warnings: list[str] = []

    # ---- 1. Agent Router ------------------------------------------------
    try:
        recent = db.get_history(session_id, limit=4)
    except Exception:
        recent = []
    decision = router.route(query, recent)
    log.info("route=%s method=%s reason=%s", decision.route, decision.method, decision.reason)

    sources: list[dict] = []
    tool_calls: list[dict] = []
    rag_fallback = False
    route_key = decision.route
    context = ""

    # ---- 2. RAG / Tools -------------------------------------------------
    if decision.route == "rag":
        try:
            result = rag.retrieve(decision.search_query or query)
            if result["empty_kb"]:
                warnings.append("The knowledge base is empty — run ingestion (python server/ingest.py).")
            if result["chunks"]:
                sources = rag.sources_for_response(result["chunks"])
                context = "KNOWLEDGE-BASE EXCERPTS:\n" + rag.format_context(result["chunks"])
            else:
                rag_fallback, route_key = True, "rag_empty"
                if not result["empty_kb"]:
                    best = max((s["similarity"] for s in result["all_scores"]), default=0)
                    warnings.append(f"No excerpt passed the relevance threshold "
                                    f"({config.RAG_MIN_SIMILARITY}); best similarity was {best:.2f}.")
        except Exception as e:
            log.exception("RAG retrieval failed")
            rag_fallback, route_key = True, "rag_empty"
            warnings.append(f"Retrieval failed ({type(e).__name__}); answered from general knowledge.")

    elif decision.route == "tool":
        try:
            history = llm.load_memory(session_id)
            tool_calls = llm.run_tool_agent(query, session_id, history, decision.suggested_tools)
        except llm.LLMError as e:
            raise PipelineError("llm_unavailable", str(e), 502, route="tool") from e
        failed = [c for c in tool_calls if not c["ok"]]
        if failed:
            warnings.append(f"{len(failed)} tool call(s) failed: " + "; ".join(c["error"] for c in failed))
        context = llm.format_tool_context(tool_calls)

    # ---- 3. LLM (LCEL chain with memory, retry + fallback) ----------------
    try:
        answer = llm.generate_answer(query, session_id, route_key, context)
    except llm.LLMError as e:
        raise PipelineError("llm_unavailable", str(e), 502, route=decision.route) from e

    # ---- 4. Persist + format --------------------------------------------
    try:
        db.save_message(session_id, "user", query, route=decision.route, route_reason=decision.reason)
        db.save_message(
            session_id, "assistant", answer, route=decision.route, route_reason=decision.reason,
            tool_name=",".join(dict.fromkeys(c["name"] for c in tool_calls)) or None, sources=sources,
        )
    except Exception:
        log.exception("Failed to persist conversation")
        warnings.append("The answer could not be saved to history.")

    return formatter.success(
        answer=answer, route=decision.route, decision=decision.to_dict(), session_id=session_id,
        latency_ms=int((time.perf_counter() - t0) * 1000), sources=sources,
        tool_calls=tool_calls, warnings=warnings, rag_fallback=rag_fallback,
    )
