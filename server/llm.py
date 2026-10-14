"""LLM Module.

  * answer_chain — a LangChain LCEL chain that builds the final prompt:
        RunnablePassthrough.assign(history=<load memory from Postgres>)
        | ChatPromptTemplate (system + context, MessagesPlaceholder(history), question)
        | ChatAnthropic(primary).with_retry(...).with_fallbacks([ChatAnthropic(fallback)])
        | StrOutputParser()
    Conversational memory: the last MEMORY_TURNS exchanges of the session are
    loaded from the `conversations` table on every call, so memory survives
    server restarts and is shared by every path (RAG / tool / direct).

  * run_tool_agent — the tool-use loop with the Anthropic SDK and the
    Anthropic tool schemas from tools.py. Claude chooses tools and
    arguments; Python executes them; results are fed back until Claude stops
    calling tools. The collected tool outputs then go through answer_chain.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import config
import db
import tools

log = logging.getLogger(__name__)


class LLMError(Exception):
    """The LLM could not produce an answer even after retries and fallback."""


# ======================================================================
# Prompt
# ======================================================================
SYSTEM_PROMPT = """You are SousChef, a friendly, precise kitchen assistant for home cooks.
Always answer in the same language as the user's latest message (Hebrew or English).
Be concise and practical; use short lists for ingredients and steps.

{route_instructions}

{context}"""

ROUTE_INSTRUCTIONS = {
    "rag": (
        "Answer using ONLY the numbered knowledge-base excerpts below. Cite them inline as [1], [2]. "
        "If the excerpts do not contain the answer, say so clearly and then give brief general advice, "
        "labelled as general knowledge."
    ),
    "rag_empty": (
        "The knowledge base had no relevant excerpts for this question. Tell the user that this is not "
        "in the SousChef recipe collection, then answer from general cooking knowledge and label it as such."
    ),
    "tool": (
        "Exact results computed by SousChef's tools are given below as JSON. Base every number in your "
        "answer on these results — do not recalculate or invent figures. Mention that nutrition values "
        "are approximate. If a tool returned an error, explain the problem plainly and suggest how the "
        "user can rephrase."
    ),
    "direct": (
        "Answer from your general cooking knowledge. SousChef's own recipes are: {recipes}. "
        "If the user asks what you can do, mention recipe Q&A, nutrition calculation, unit conversion "
        "and recipe scaling."
    ),
}

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


# ======================================================================
# Memory
# ======================================================================
def load_memory(session_id: str) -> list:
    """Last MEMORY_TURNS user/assistant exchanges as LangChain messages."""
    try:
        turns = db.get_history(session_id, limit=config.MEMORY_TURNS * 2)
    except Exception:
        log.exception("Could not load memory; continuing without history")
        return []
    msgs = []
    for t in turns:
        if t["route"] == "error":
            continue
        msgs.append(HumanMessage(t["content"]) if t["role"] == "user" else AIMessage(t["content"]))
    # Anthropic requires the first message to be from the user
    while msgs and isinstance(msgs[0], AIMessage):
        msgs.pop(0)
    return msgs


# ======================================================================
# Chain construction (lazy, so the app starts even without an API key)
# ======================================================================
_answer_chain = None


def _chat_model(model: str) -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        api_key=config.ANTHROPIC_API_KEY,
        max_tokens=1024,
        timeout=config.LLM_TIMEOUT,
        max_retries=0,  # retries are handled by .with_retry below
    )


def get_answer_chain():
    global _answer_chain
    if _answer_chain is None:
        primary = _chat_model(config.LLM_MODEL).with_retry(
            stop_after_attempt=config.LLM_MAX_RETRIES, wait_exponential_jitter=True
        )
        fallback = _chat_model(config.LLM_FALLBACK_MODEL).with_retry(stop_after_attempt=2)
        llm = primary.with_fallbacks([fallback])
        _answer_chain = (
            RunnablePassthrough.assign(history=RunnableLambda(lambda x: load_memory(x["session_id"])))
            | prompt
            | llm
            | StrOutputParser()
        )
    return _answer_chain


def generate_answer(question: str, session_id: str, route_key: str, context: str = "") -> str:
    if not config.ANTHROPIC_API_KEY:
        raise LLMError("ANTHROPIC_API_KEY is not configured on the server.")
    import recipes as _recipes  # local import avoids a cycle at module load
    instructions = ROUTE_INSTRUCTIONS[route_key].format(recipes=", ".join(_recipes.recipe_titles()))
    try:
        answer = get_answer_chain().invoke({
            "question": question,
            "session_id": session_id,
            "route_instructions": instructions,
            "context": context,
        })
    except Exception as e:
        log.exception("LLM chain failed after retries and fallback")
        raise LLMError(f"The language model is unavailable right now ({type(e).__name__}).") from e
    if not answer or not answer.strip():
        raise LLMError("The language model returned an empty answer.")
    return answer.strip()


# ======================================================================
# Tool-use agent loop (Anthropic SDK + Anthropic tool schemas)
# ======================================================================
_client: anthropic.Anthropic | None = None
MAX_TOOL_ROUNDS = 5

TOOL_AGENT_SYSTEM = """You are SousChef's tool-calling component. Use the provided tools to get exact
data for the user's request. Rules:
- If the user names a recipe and asks about its nutrition or scaling, call lookup_recipe_ingredients
  first, then calculate_nutrition with the grams from that lookup (use 'nutrition_grams' when present,
  e.g. for deep-frying oil) and the recipe's base servings — or call scale_recipe.
- For free-form ingredient lists, convert to grams (convert_units) before calculate_nutrition.
- Call independent tools in parallel when possible. Stop calling tools once you have the data;
  reply with one short sentence."""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY,
                                      max_retries=config.LLM_MAX_RETRIES, timeout=config.LLM_TIMEOUT)
    return _client


def create_message(client: anthropic.Anthropic, **kwargs) -> Any:
    """messages.create that degrades a forced tool_choice gracefully.

    Some models (e.g. Claude Sonnet 5.5) reject tool_choice of type "any"/"tool".
    In that case we retry once with "auto"; the system prompt still tells the
    model to call a tool, so behaviour stays the same.
    """
    try:
        return client.messages.create(**kwargs)
    except anthropic.BadRequestError as e:
        forced = kwargs.get("tool_choice", {}).get("type") in ("any", "tool")
        if forced and "tool_choice" in str(e):
            log.info("Model %s does not support forced tool_choice; retrying with auto", kwargs.get("model"))
            return client.messages.create(**{**kwargs, "tool_choice": {"type": "auto"}})
        raise


def _create_with_fallback(**kwargs) -> Any:
    """SDK already retries 429/5xx/timeouts; on final failure try the fallback model."""
    try:
        return create_message(_get_client(), model=config.LLM_MODEL, **kwargs)
    except anthropic.APIError as e:
        log.warning("Primary model failed in tool loop (%s); trying fallback %s", e, config.LLM_FALLBACK_MODEL)
        return create_message(_get_client(), model=config.LLM_FALLBACK_MODEL, **kwargs)


def run_tool_agent(question: str, session_id: str, history: list | None = None,
                   suggested_tools: list[str] | None = None) -> list[dict]:
    """Returns the list of executed tool calls: [{name, input, ok, result|error}]."""
    if not config.ANTHROPIC_API_KEY:
        raise LLMError("ANTHROPIC_API_KEY is not configured on the server.")

    messages: list[dict] = []
    for m in (history or [])[-4:]:  # a little context so "it" resolves to a recipe
        messages.append({"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    hint = f"\n\n(Router hint — likely tools: {', '.join(suggested_tools)})" if suggested_tools else ""
    messages.append({"role": "user", "content": question + hint})
    # Anthropic requires alternating roles; merge consecutive user turns if history ended on user
    merged: list[dict] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    messages = merged

    executed: list[dict] = []
    for round_no in range(MAX_TOOL_ROUNDS):
        try:
            resp = _create_with_fallback(
                max_tokens=1024,
                system=TOOL_AGENT_SYSTEM,
                tools=tools.TOOL_SCHEMAS,
                # the router already decided a tool is needed: force at least one call
                tool_choice={"type": "any"} if round_no == 0 else {"type": "auto"},
                messages=messages,
            )
        except anthropic.APIError as e:
            raise LLMError(f"The language model is unavailable right now ({type(e).__name__}).") from e

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            break
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        results_block = []
        for tu in tool_uses:
            out = tools.execute_tool(tu.name, tu.input, session_id)
            executed.append({"name": tu.name, "input": tu.input, **out})
            results_block.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(out.get("result") if out["ok"] else {"error": out["error"]},
                                      ensure_ascii=False),
                "is_error": not out["ok"],
            })
        messages.append({"role": "user", "content": results_block})
        if resp.stop_reason != "tool_use":
            break
    return executed


def format_tool_context(executed: list[dict]) -> str:
    if not executed:
        return "No tool was executed."
    parts = []
    for i, c in enumerate(executed, 1):
        payload = c.get("result") if c["ok"] else {"error": c["error"]}
        parts.append(f"Tool call {i}: {c['name']}({json.dumps(c['input'], ensure_ascii=False)})\n"
                     f"Result: {json.dumps(payload, ensure_ascii=False)}")
    return "TOOL RESULTS:\n" + "\n\n".join(parts)
