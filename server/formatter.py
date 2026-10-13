"""Response Formatter — every /chat reply has the same JSON contract.

Success (HTTP 200):
{
  "answer":      str,
  "route":       "rag" | "tool" | "direct",
  "route_label": "RAG" | "Tool: Nutrition Calculator" | "Direct LLM" | ...,
  "router":      {"method", "reason", "confidence", "search_query", "suggested_tools"},
  "sources":     [{"ref", "source", "title", "section", "similarity", "snippet"}],
  "tool_calls":  [{"name", "input", "ok", "output"|"error"}],
  "warnings":    [str],
  "session_id":  str,
  "latency_ms":  int
}
Error (HTTP 4xx/5xx):
{ "error": {"code": str, "message": str}, "route": "error" | <route>, "session_id": str|null }
"""
from __future__ import annotations

from typing import Any

TOOL_LABELS = {
    "lookup_recipe_ingredients": "Recipe Lookup",
    "calculate_nutrition": "Nutrition Calculator",
    "convert_units": "Unit Converter",
    "scale_recipe": "Recipe Scaler",
}


def route_label(route: str, tool_calls: list[dict] | None = None, rag_fallback: bool = False) -> str:
    if route == "rag":
        return "RAG → Direct LLM (no matching docs)" if rag_fallback else "RAG"
    if route == "tool":
        names = []
        for c in tool_calls or []:
            label = TOOL_LABELS.get(c["name"], c["name"])
            if label not in names:
                names.append(label)
        return "Tool: " + (" + ".join(names) if names else "none called")
    if route == "direct":
        return "Direct LLM"
    return route


def success(*, answer: str, route: str, decision: dict, session_id: str, latency_ms: int,
            sources: list[dict] | None = None, tool_calls: list[dict] | None = None,
            warnings: list[str] | None = None, rag_fallback: bool = False) -> dict[str, Any]:
    return {
        "answer": answer,
        "route": route,
        "route_label": route_label(route, tool_calls, rag_fallback),
        "router": decision,
        "sources": sources or [],
        "tool_calls": [
            {"name": c["name"], "input": c["input"], "ok": c["ok"],
             **({"output": c.get("result")} if c["ok"] else {"error": c.get("error")})}
            for c in (tool_calls or [])
        ],
        "warnings": warnings or [],
        "session_id": session_id,
        "latency_ms": latency_ms,
    }


def error(code: str, message: str, session_id: str | None = None, route: str = "error") -> dict[str, Any]:
    return {"error": {"code": code, "message": message}, "route": route, "session_id": session_id}
