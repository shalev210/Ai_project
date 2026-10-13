"""Test-suite. Needs the database running (docker compose up -d db) but NOT an
Anthropic key: LLM calls are replaced by fakes, and embeddings use the offline
"hash" backend so no model download is needed.

    cd server && EMBEDDING_BACKEND=hash ROUTER_MODE=rules python -m pytest -q
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("ROUTER_MODE", "rules")
# the toy hash embedder produces lower similarities than the real model
os.environ.setdefault("RAG_MIN_SIMILARITY", "0.1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

import app as app_module  # noqa: E402
import config  # noqa: E402
import db  # noqa: E402
import ingest  # noqa: E402
import llm  # noqa: E402
import router  # noqa: E402
import tools  # noqa: E402

SESSION = "pytest-session-0001"


@pytest.fixture(scope="session", autouse=True)
def knowledge_base():
    ingest.ingest()
    yield


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")

    def fake_answer(question, session_id, route_key, context=""):
        return f"[{route_key}] answer to: {question} | ctx={len(context)}"

    def fake_tool_agent(question, session_id, history=None, suggested_tools=None):
        return [{"name": "convert_units", "input": {"amount": 2, "from_unit": "cup", "to_unit": "g",
                                                    "ingredient": "flour"},
                 **tools.execute_tool("convert_units", {"amount": 2, "from_unit": "cup", "to_unit": "g",
                                                         "ingredient": "flour"}, session_id)}]

    monkeypatch.setattr(llm, "generate_answer", fake_answer)
    monkeypatch.setattr(llm, "run_tool_agent", fake_tool_agent)
    return app_module.create_app().test_client()


# ------------------------------------------------------------------ router
@pytest.mark.parametrize("query,expected", [
    ("How do I make shakshuka?", "rag"),
    ("איך מכינים חומוס?", "rag"),
    ("What can I use instead of eggs?", "rag"),
    ("How many calories are in 100 g tahini?", "tool"),
    ("Convert 2 cups of flour to grams", "tool"),
    ("Scale the challah for 24 people", "tool"),
    ("כמה קלוריות יש בשקשוקה?", "tool"),
    ("Hello!", "direct"),
    ("What is umami?", "direct"),
])
def test_rule_router(query, expected):
    d = router.route_by_rules(query)
    assert d.route == expected
    assert d.reason


def test_llm_router_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_MODE", "llm")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "bad-key")
    monkeypatch.setattr(router, "route_by_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    d = router.route("Convert 1 cup sugar to grams")
    assert d.route == "tool" and d.method == "rules (llm fallback)"


# ------------------------------------------------------------------- tools
def test_tool_schemas_are_valid_anthropic_format():
    assert len(tools.TOOL_SCHEMAS) >= 2
    for t in tools.TOOL_SCHEMAS:
        assert set(t) == {"name", "description", "input_schema"}
        assert t["input_schema"]["type"] == "object"
        assert t["name"] in tools.TOOL_FUNCTIONS


def test_nutrition_and_conversion():
    r = tools.calculate_nutrition([{"name": "sugar", "grams": 100}], servings=2)
    assert r["total"]["kcal"] == 387.0 and r["per_serving"]["kcal"] == 193.5
    assert tools.convert_units(100, "c", "f")["result"] == 212.0
    assert tools.convert_units(1, "kg", "g")["result"] == 1000.0


def test_tool_failure_is_logged_not_raised():
    out = tools.execute_tool("scale_recipe", {"recipe_name": "pizza unicorn", "target_servings": -1}, SESSION)
    assert out["ok"] is False
    logs = db.get_tool_calls(SESSION, 5)
    assert logs[0]["tool_name"] == "scale_recipe" and logs[0]["success"] is False


# ---------------------------------------------------------------- endpoints
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["database"] == "up"


def test_chat_rag_route(client):
    r = client.post("/chat", json={"query": "How do I make shakshuka?", "session_id": SESSION})
    body = r.get_json()
    assert r.status_code == 200
    assert body["route"] == "rag" and body["route_label"].startswith("RAG")
    assert body["sources"] and all("similarity" in s for s in body["sources"])


def test_chat_tool_route(client):
    r = client.post("/chat", json={"query": "Convert 2 cups of flour to grams", "session_id": SESSION})
    body = r.get_json()
    assert body["route"] == "tool"
    assert body["route_label"] == "Tool: Unit Converter"
    assert body["tool_calls"][0]["output"]["result"] == 254.4


def test_chat_direct_route(client):
    r = client.post("/chat", json={"query": "Hello!", "session_id": SESSION})
    assert r.get_json()["route_label"] == "Direct LLM"


def test_history_persists(client):
    r = client.get(f"/history?session_id={SESSION}&include_tools=true")
    body = r.get_json()
    assert r.status_code == 200 and body["count"] >= 2
    assert {t["role"] for t in body["turns"]} == {"user", "assistant"}
    assert "tool_calls" in body


@pytest.mark.parametrize("payload,raw", [
    (None, "not json at all"),
    ({"query": ""}, None),
    ({"query": 42}, None),
    ({"query": "hi", "session_id": "bad id!"}, None),
    ({"query": "x" * 5000}, None),
])
def test_malformed_requests_return_400(client, payload, raw):
    if raw is not None:
        r = client.post("/chat", data=raw, content_type="application/json")
    else:
        r = client.post("/chat", json=payload)
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "bad_request"


def test_empty_knowledge_base_does_not_crash(client):
    db.replace_chunks([])
    try:
        r = client.post("/chat", json={"query": "How do I make hummus?", "session_id": SESSION})
        body = r.get_json()
        assert r.status_code == 200
        assert body["route_label"] == "RAG → Direct LLM (no matching docs)"
        assert any("empty" in w for w in body["warnings"])
    finally:
        ingest.ingest()


def test_llm_error_returns_clean_502(client, monkeypatch):
    def broken(*a, **k):
        raise llm.LLMError("The language model is unavailable right now (APIConnectionError).")
    monkeypatch.setattr(llm, "generate_answer", broken)
    r = client.post("/chat", json={"query": "What is umami?", "session_id": SESSION})
    assert r.status_code == 502
    assert r.get_json()["error"]["code"] == "llm_unavailable"


def test_unknown_route_is_json_404(client):
    r = client.get("/nope")
    assert r.status_code == 404 and "error" in r.get_json()
