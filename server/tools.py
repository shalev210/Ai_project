"""Tools Module.

Four custom tools, each declared with the Anthropic tool-use (function-calling)
JSON schema and backed by a deterministic Python function. Every execution is
logged to the `tool_calls` table (name, input, output, success, duration).

Why tools instead of the LLM alone? Arithmetic on nutrition values, unit
conversions and recipe scaling must be exact and reproducible — LLMs are
unreliable at multi-step arithmetic, and the ingredient data is structured.
"""
from __future__ import annotations

import json
import logging
import math
import time
from functools import lru_cache
from typing import Any, Callable

import config
import db
import recipes

log = logging.getLogger(__name__)


class ToolError(Exception):
    """Expected, user-facing tool failure (bad input, unknown item...)."""


# ======================================================================
# Anthropic tool schemas
# ======================================================================
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "lookup_recipe_ingredients",
        "description": (
            "Structured lookup of a recipe in the SousChef collection. Returns the exact "
            "ingredient list (name, quantity, unit) and the base number of servings. Use it "
            "before calculate_nutrition or scale_recipe when the user refers to a recipe by name. "
            "Accepts English or Hebrew names (e.g. 'shakshuka', 'שקשוקה')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string", "description": "Recipe name, e.g. 'hummus'."}
            },
            "required": ["recipe_name"],
        },
    },
    {
        "name": "calculate_nutrition",
        "description": (
            "Calculates total and per-serving calories, protein, fat, carbohydrates and fiber "
            "for a list of ingredients given in grams, using a reference nutrition table. "
            "Use for any question about calories or macros. Ingredients that are not in the "
            "table are returned in 'unknown_ingredients' and excluded from totals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Ingredient name, e.g. 'olive oil'."},
                            "grams": {"type": "number", "description": "Weight in grams (> 0)."},
                        },
                        "required": ["name", "grams"],
                    },
                    "minItems": 1,
                },
                "servings": {"type": "integer", "description": "Number of servings to divide by (default 1).", "minimum": 1},
            },
            "required": ["ingredients"],
        },
    },
    {
        "name": "convert_units",
        "description": (
            "Converts cooking quantities between units. Supports weight (g, kg, oz, lb), "
            "volume (ml, l, tsp, tbsp, cup, fl_oz) and temperature (c, f). Converting between "
            "volume and weight (e.g. cups of flour to grams) requires the 'ingredient' field "
            "so its density can be used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "The quantity to convert."},
                "from_unit": {"type": "string", "description": "Source unit, e.g. 'cup'."},
                "to_unit": {"type": "string", "description": "Target unit, e.g. 'g'."},
                "ingredient": {"type": "string", "description": "Ingredient name; required for volume<->weight."},
            },
            "required": ["amount", "from_unit", "to_unit"],
        },
    },
    {
        "name": "scale_recipe",
        "description": (
            "Scales every ingredient of a recipe from its base servings to a target number of "
            "servings and returns the new quantities. Use when the user wants to cook a recipe "
            "for a different number of people."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string", "description": "Recipe name, e.g. 'challah'."},
                "target_servings": {"type": "number", "description": "Desired number of servings (> 0)."},
            },
            "required": ["recipe_name", "target_servings"],
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_SCHEMAS]


# ======================================================================
# Nutrition table helpers
# ======================================================================
@lru_cache(maxsize=1)
def _nutrition_table() -> dict[str, dict]:
    with open(config.NUTRITION_FILE, encoding="utf-8") as f:
        return json.load(f)["items"]


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for name, item in _nutrition_table().items():
        idx[name.lower()] = name
        for a in item.get("aliases", []):
            idx[a.lower()] = name
    return idx


def resolve_ingredient(name: str) -> str | None:
    """Exact / alias match first, then the longest alias contained in the text."""
    q = " ".join(name.lower().replace("-", " ").split())
    idx = _alias_index()
    if q in idx:
        return idx[q]
    if q.endswith("s") and q[:-1] in idx:
        return idx[q[:-1]]
    candidates = [k for k in idx if k in q]
    if candidates:
        return idx[max(candidates, key=len)]
    return None


# ======================================================================
# Tool implementations
# ======================================================================
def lookup_recipe_ingredients(recipe_name: str) -> dict:
    r = recipes.find_recipe(recipe_name or "")
    if not r:
        raise ToolError(
            f"Recipe '{recipe_name}' not found. Available: {', '.join(recipes.recipe_titles())}"
        )
    return {
        "recipe": r["title"],
        "source": r["source"],
        "base_servings": r["servings"],
        "ingredients": [
            {k: v for k, v in i.items() if v is not None} for i in r["ingredients"]
        ],
    }


def calculate_nutrition(ingredients: list[dict], servings: int = 1) -> dict:
    if not isinstance(ingredients, list) or not ingredients:
        raise ToolError("'ingredients' must be a non-empty list of {name, grams}.")
    servings = int(servings or 1)
    if servings < 1:
        raise ToolError("'servings' must be at least 1.")

    table = _nutrition_table()
    keys = ["kcal", "protein_g", "fat_g", "carbs_g", "fiber_g"]
    totals = dict.fromkeys(keys, 0.0)
    breakdown, unknown = [], []
    for ing in ingredients:
        name = str(ing.get("name", "")).strip()
        try:
            grams = float(ing.get("grams"))
        except (TypeError, ValueError):
            raise ToolError(f"Ingredient '{name}' has an invalid 'grams' value.")
        if grams <= 0 or math.isnan(grams):
            raise ToolError(f"Ingredient '{name}' must have grams > 0.")
        match = resolve_ingredient(name)
        if not match:
            unknown.append(name)
            continue
        row = table[match]
        contrib = {k: row[k] * grams / 100 for k in keys}
        for k in keys:
            totals[k] += contrib[k]
        breakdown.append({"ingredient": name, "matched_as": match, "grams": grams,
                          "kcal": round(contrib["kcal"], 1)})

    r1 = lambda d: {k: round(v, 1) for k, v in d.items()}  # noqa: E731
    return {
        "servings": servings,
        "total": r1(totals),
        "per_serving": r1({k: v / servings for k, v in totals.items()}),
        "breakdown": breakdown,
        "unknown_ingredients": unknown,
        "note": "Approximate values from a USDA-based reference table.",
    }


_WEIGHT = {"g": 1.0, "kg": 1000.0, "oz": 28.3495, "lb": 453.592}
_VOLUME = {"ml": 1.0, "l": 1000.0, "tsp": 4.92892, "tbsp": 14.7868, "cup": 240.0, "fl_oz": 29.5735}
_TEMP = {"c", "f"}
_UNIT_ALIASES = {
    "gram": "g", "grams": "g", "gr": "g", "גרם": "g", "kilogram": "kg", "kilograms": "kg", "קילו": "kg",
    "ounce": "oz", "ounces": "oz", "pound": "lb", "pounds": "lb", "lbs": "lb",
    "milliliter": "ml", "milliliters": "ml", "מ\"ל": "ml", "מל": "ml", "liter": "l", "liters": "l", "litre": "l", "ליטר": "l",
    "teaspoon": "tsp", "teaspoons": "tsp", "כפית": "tsp", "כפיות": "tsp",
    "tablespoon": "tbsp", "tablespoons": "tbsp", "כף": "tbsp", "כפות": "tbsp",
    "cups": "cup", "כוס": "cup", "כוסות": "cup", "fl oz": "fl_oz", "fluid ounce": "fl_oz",
    "celsius": "c", "°c": "c", "fahrenheit": "f", "°f": "f",
}


def _norm_unit(u: str) -> str:
    u = (u or "").strip().lower()
    return _UNIT_ALIASES.get(u, u)


def convert_units(amount: float, from_unit: str, to_unit: str, ingredient: str | None = None) -> dict:
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ToolError("'amount' must be a number.")
    fu, tu = _norm_unit(from_unit), _norm_unit(to_unit)

    if fu in _TEMP or tu in _TEMP:
        if not (fu in _TEMP and tu in _TEMP):
            raise ToolError("Temperature can only be converted to another temperature unit.")
        result = amount if fu == tu else (amount * 9 / 5 + 32 if fu == "c" else (amount - 32) * 5 / 9)
        return {"amount": amount, "from_unit": fu, "to_unit": tu, "result": round(result, 1)}

    known = set(_WEIGHT) | set(_VOLUME)
    for u, raw in ((fu, from_unit), (tu, to_unit)):
        if u not in known:
            raise ToolError(f"Unknown unit '{raw}'. Supported: {', '.join(sorted(known | _TEMP))}.")

    density, matched = None, None
    if (fu in _WEIGHT) != (tu in _WEIGHT):  # crossing weight <-> volume
        if not ingredient:
            raise ToolError("Converting between volume and weight requires the 'ingredient' field.")
        matched = resolve_ingredient(ingredient)
        density = _nutrition_table().get(matched or "", {}).get("density_g_per_ml")
        if not density:
            raise ToolError(f"No density data for '{ingredient}', cannot convert volume<->weight.")

    # normalise to grams or ml
    base = amount * (_WEIGHT[fu] if fu in _WEIGHT else _VOLUME[fu])
    if fu in _WEIGHT and tu in _VOLUME:
        base = base / density           # g -> ml
    elif fu in _VOLUME and tu in _WEIGHT:
        base = base * density           # ml -> g
    result = base / (_WEIGHT[tu] if tu in _WEIGHT else _VOLUME[tu])

    out = {"amount": amount, "from_unit": fu, "to_unit": tu, "result": round(result, 2)}
    if density:
        out.update({"ingredient": matched, "density_g_per_ml": density})
    return out


def scale_recipe(recipe_name: str, target_servings: float) -> dict:
    try:
        target = float(target_servings)
    except (TypeError, ValueError):
        raise ToolError("'target_servings' must be a number.")
    if target <= 0:
        raise ToolError("'target_servings' must be greater than 0.")
    base = lookup_recipe_ingredients(recipe_name)
    if not base["base_servings"]:
        raise ToolError(f"Recipe '{base['recipe']}' has no base servings defined.")
    factor = target / base["base_servings"]
    return {
        "recipe": base["recipe"],
        "base_servings": base["base_servings"],
        "target_servings": target,
        "scale_factor": round(factor, 3),
        "ingredients": [
            {"name": i["name"], "quantity": round(i["quantity"] * factor, 1), "unit": i["unit"]}
            for i in base["ingredients"]
        ],
        "note": "Cooking times do not scale linearly; baking pans may need resizing.",
    }


TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "lookup_recipe_ingredients": lookup_recipe_ingredients,
    "calculate_nutrition": calculate_nutrition,
    "convert_units": convert_units,
    "scale_recipe": scale_recipe,
}


# ======================================================================
# Executor (with DB logging)
# ======================================================================
def execute_tool(name: str, tool_input: dict, session_id: str) -> dict:
    """Run a tool, never raising. Returns {"ok": bool, "result"|"error": ...}."""
    start = time.perf_counter()
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        out = {"ok": False, "error": f"Unknown tool '{name}'."}
    else:
        try:
            out = {"ok": True, "result": func(**(tool_input or {}))}
        except ToolError as e:
            out = {"ok": False, "error": str(e)}
        except TypeError as e:  # wrong / missing arguments from the model
            out = {"ok": False, "error": f"Invalid arguments for {name}: {e}"}
        except Exception as e:  # unexpected bug — log, don't crash the request
            log.exception("Tool %s crashed", name)
            out = {"ok": False, "error": f"Tool '{name}' failed unexpectedly: {e}"}
    duration_ms = int((time.perf_counter() - start) * 1000)

    try:
        db.log_tool_call(session_id, name, tool_input or {}, out.get("result"),
                         out["ok"], out.get("error"), duration_ms)
    except Exception:
        log.exception("Failed to log tool call (continuing)")
    return out
