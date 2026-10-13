"""Agent Router — decides, for every query, which path processes it:

    rag     : the answer lives in the recipe / kitchen-guide knowledge base
    tool    : the query needs exact computation or structured data
              (nutrition math, unit conversion, recipe scaling)
    direct  : greetings, chit-chat, or general cooking knowledge that is not
              in the knowledge base

Two strategies, layered:
  1. LLM classifier (default, ROUTER_MODE=llm): a small Claude model is FORCED
     to call a `route_query` tool, so its decision comes back as validated,
     structured JSON (route + reason + confidence + an English search query
     for RAG). It also sees the last few turns, so follow-ups such as
     "and how long does it keep?" are routed correctly.
  2. Rule-based classifier: keyword/regex signals in Hebrew and English.
     Used when ROUTER_MODE=rules, and as the automatic fallback whenever the
     LLM classifier fails (no API key, timeout, malformed output...).

Every decision carries a human-readable `reason`, which is returned to the
client and stored in the DB, so routing is explainable and auditable.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

import anthropic

import config
import recipes

log = logging.getLogger(__name__)

ROUTES = ("rag", "tool", "direct")


@dataclass
class RouteDecision:
    route: str
    reason: str
    method: str                 # "llm" | "rules" | "rules (llm fallback)"
    confidence: float = 0.0
    search_query: str = ""      # query text used for retrieval
    suggested_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ======================================================================
# 1) Rule-based router
# ======================================================================
_TOOL_PATTERNS: dict[str, list[str]] = {
    "calculate_nutrition": [
        r"\bcalori", r"\bkcal\b", r"\bnutrition", r"\bmacro", r"\bprotein\b", r"\bcarb", r"\bfat\b",
        r"\bfiber\b", r"קלורי", r"ערכים? תזונתי", r"ערך תזונתי", r"חלבון", r"פחמימות", r"שומן",
    ],
    "convert_units": [
        r"\bconvert", r"\bhow many (grams|cups|tablespoons|teaspoons|ml|ounces|tbsp|tsp)\b",
        r"\b(grams?|g|ml|cups?|tbsp|tsp|oz|lbs?)\s+(to|in|into)\s+", r"\b(fahrenheit|celsius)\b",
        r"\d+\s*°?\s*[fc]\b", r"להמיר", r"המר", r"המרה", r"כמה (גרם|כוסות|כפות|כפיות|מ\"ל|מל)",
        r"פרנהייט", r"צלזיוס",
    ],
    "scale_recipe": [
        r"\bscale\b", r"\bdouble\b", r"\btriple\b", r"\bhalve\b", r"\bfor \d+ (people|servings|guests|persons)\b",
        r"\b\d+ (people|servings|guests)\b", r"להכפיל", r"כפול", r"חצי כמות", r"ל-?\s?\d+ (אנשים|סועדים|מנות)",
        r"\d+ (אנשים|סועדים|מנות)",
    ],
}

_RAG_PATTERNS = [
    r"\brecipe", r"\bhow (do|to|can) (i|you|we)? ?(make|cook|bake|prepare|store|keep|freeze)",
    r"\bingredient", r"\bstore\b", r"\bstorage\b", r"\bfreeze", r"\bkeep\b", r"\bsubstitut",
    r"\breplace\b", r"\binstead of\b", r"\btemperature\b", r"\bsafe\b", r"\ballerg", r"\bkosher\b",
    r"\bpareve\b", r"\bvegan\b", r"\bvegetarian\b", r"\bgluten", r"\bhow long\b", r"\btips?\b",
    r"\bvariation", r"\bsteps?\b", r"\bmethod\b", r"\bfridge\b",
    r"מתכון", r"איך (מכינים|להכין|אופים|לאפות|מבשלים|לבשל|שומרים|לשמור|מקפיאים)", r"מרכיבים", r"רכיבים",
    r"לאחסן", r"אחסון", r"להקפיא", r"תחליף", r"במקום", r"טמפרטורה", r"אלרג", r"כשר", r"פרווה",
    r"טבעוני", r"צמחוני", r"גלוטן", r"כמה זמן", r"טיפ", r"מקרר",
]

_DIRECT_PATTERNS = [
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening)|bye)\b", r"\bwho are you\b",
    r"\bwhat can you do\b", r"^\s*(שלום|היי|הי|תודה|בוקר טוב|ערב טוב|להתראות)", r"מי אתה", r"מה אתה יודע",
]

_RECIPE_NAME_PATTERNS = [
    r"shakshuka", r"hummus", r"lentil", r"schnitzel", r"challah", r"falafel", r"israeli salad",
    r"chopped salad", r"banana bread", r"pomodoro", r"spaghetti", r"tahini cookie",
    r"שקשוקה", r"חומוס", r"עדשים", r"שניצל", r"חלה", r"פלאפל", r"סלט ישראלי", r"סלט קצוץ",
    r"עוגת בננה", r"לחם בננה", r"פסטה", r"עוגיות טחינה",
]


def _matches(patterns: list[str], text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.I)]


def route_by_rules(query: str) -> RouteDecision:
    q = query.strip()
    tool_hits = {name: _matches(pats, q) for name, pats in _TOOL_PATTERNS.items()}
    tool_hits = {k: v for k, v in tool_hits.items() if v}
    rag_hits = _matches(_RAG_PATTERNS, q)
    recipe_hits = _matches(_RECIPE_NAME_PATTERNS, q)
    direct_hits = _matches(_DIRECT_PATTERNS, q)

    if tool_hits:
        tools = sorted(tool_hits)
        # a named recipe + a computation -> the tool path will look the recipe up first
        if recipe_hits and ("calculate_nutrition" in tools or "scale_recipe" in tools):
            tools = ["lookup_recipe_ingredients"] + tools
        return RouteDecision(
            route="tool",
            reason=f"Computation keywords matched for {', '.join(sorted(tool_hits))}"
                   + (" on a known recipe" if recipe_hits else ""),
            method="rules", confidence=0.8, search_query=q, suggested_tools=tools,
        )
    if recipe_hits or rag_hits:
        what = []
        if recipe_hits:
            what.append("a recipe from the collection is named")
        if rag_hits:
            what.append("knowledge-base keywords (recipe / storage / substitution / safety)")
        return RouteDecision(route="rag", reason="; ".join(what).capitalize(),
                             method="rules", confidence=0.75 if recipe_hits else 0.6, search_query=q)
    if direct_hits:
        return RouteDecision(route="direct", reason="Greeting or small talk",
                             method="rules", confidence=0.9, search_query=q)
    return RouteDecision(route="direct",
                         reason="No knowledge-base or computation signals — general question",
                         method="rules", confidence=0.5, search_query=q)


# ======================================================================
# 2) LLM-based router (structured output via a forced tool call)
# ======================================================================
_ROUTER_TOOL = {
    "name": "route_query",
    "description": "Record the routing decision for the user's latest message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": list(ROUTES)},
            "reason": {"type": "string", "description": "One short sentence explaining the decision."},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "search_query": {
                "type": "string",
                "description": "For 'rag': a standalone ENGLISH search query, with pronouns from the "
                               "conversation resolved (e.g. 'it' -> 'hummus'). Otherwise repeat the message.",
            },
            "suggested_tools": {
                "type": "array",
                "items": {"type": "string", "enum": ["lookup_recipe_ingredients", "calculate_nutrition",
                                                     "convert_units", "scale_recipe"]},
            },
        },
        "required": ["route", "reason", "confidence", "search_query"],
    },
}

_ROUTER_SYSTEM = """You are the routing component of SousChef, a recipe & nutrition assistant.
Classify the user's LATEST message into exactly one route:

- "rag": the answer should come from SousChef's knowledge base. It contains these recipes:
  {recipes}
  plus a Kitchen Safety & Substitutions guide (safe internal temperatures, fridge/freezer rules,
  ingredient substitutions, kosher notes, allergens). Use "rag" for how-to-make, ingredients,
  method, tips, storage, substitutions, food safety, allergens of these items.
- "tool": the message needs exact computation or structured data: calories / macros / nutrition,
  unit or temperature conversion, scaling a recipe to a different number of servings, or listing
  the exact ingredient quantities of a recipe. Computation always wins over "rag".
- "direct": greetings, thanks, questions about you, or general cooking knowledge NOT covered above
  (e.g. "what is umami?", "difference between baking soda and baking powder").

Use the recent conversation to resolve references like "it" or "that recipe".
The user may write in Hebrew or English. Always respond by calling route_query."""

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=2, timeout=20)
    return _client


def route_by_llm(query: str, recent_turns: list[dict] | None = None) -> RouteDecision:
    convo = ""
    if recent_turns:
        lines = [f"{t['role']}: {t['content'][:300]}" for t in recent_turns[-4:]]
        convo = "Recent conversation:\n" + "\n".join(lines) + "\n\n"
    resp = _get_client().messages.create(
        model=config.ROUTER_MODEL,
        max_tokens=300,
        system=_ROUTER_SYSTEM.format(recipes=", ".join(recipes.recipe_titles())),
        tools=[_ROUTER_TOOL],
        tool_choice={"type": "tool", "name": "route_query"},
        messages=[{"role": "user", "content": f"{convo}Latest message: {query}"}],
    )
    block = next((b for b in resp.content if b.type == "tool_use"), None)
    if block is None:
        raise ValueError("Router model returned no tool_use block")
    data = block.input
    route = data.get("route")
    if route not in ROUTES:
        raise ValueError(f"Router returned invalid route: {route!r}")
    return RouteDecision(
        route=route,
        reason=str(data.get("reason", "")).strip() or "LLM classifier decision",
        method="llm",
        confidence=float(data.get("confidence", 0.0)),
        search_query=str(data.get("search_query") or query),
        suggested_tools=[t for t in data.get("suggested_tools", []) or [] if isinstance(t, str)],
    )


# ======================================================================
# Public entry point
# ======================================================================
def route(query: str, recent_turns: list[dict] | None = None) -> RouteDecision:
    if config.ROUTER_MODE == "llm" and config.ANTHROPIC_API_KEY:
        try:
            return route_by_llm(query, recent_turns)
        except Exception as e:
            log.warning("LLM router failed (%s) — falling back to rules", e)
            d = route_by_rules(query)
            d.method = "rules (llm fallback)"
            return d
    return route_by_rules(query)
