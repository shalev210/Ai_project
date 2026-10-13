"""Structured parsing of the recipe Markdown files in data/recipes.

The same files feed two consumers:
  * the RAG ingester (free-text chunks)        -> ingest.py
  * the tools (structured ingredient queries)  -> tools.py
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import config

_ING_RE = re.compile(r"^-\s*(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>g|ml)\s+(?P<name>[^(]+?)\s*(?:\((?P<note>[^)]*)\))?\s*$")
_ABSORBED_RE = re.compile(r"about\s+(\d+(?:\.\d+)?)\s*g\s+is\s+absorbed", re.I)
_SERVINGS_RE = re.compile(r"^Servings:\s*(\d+)", re.I | re.M)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@lru_cache(maxsize=1)
def load_recipes() -> dict[str, dict]:
    """slug -> {title, source, servings, ingredients[...]}"""
    recipes: dict[str, dict] = {}
    for path in sorted(Path(config.RECIPES_DIR).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_m = re.search(r"^#\s+(.+)$", text, re.M)
        title = title_m.group(1).strip() if title_m else path.stem
        serv_m = _SERVINGS_RE.search(text)
        ingredients = []
        in_ing = False
        for line in text.splitlines():
            if line.startswith("## "):
                in_ing = line[3:].strip().lower() == "ingredients"
                continue
            if in_ing:
                m = _ING_RE.match(line.strip())
                if m:
                    qty = float(m.group("qty"))
                    note = (m.group("note") or "").strip()
                    absorbed = _ABSORBED_RE.search(note)
                    ingredients.append({
                        "name": m.group("name").strip(),
                        "quantity": qty,
                        "unit": m.group("unit"),
                        "note": note or None,
                        # grams actually eaten — differs for deep-frying oil
                        "nutrition_grams": float(absorbed.group(1)) if absorbed else qty,
                    })
        if not ingredients:  # guides etc. are not recipes
            continue
        recipes[_slug(path.stem)] = {
            "title": title,
            "source": path.name,
            "servings": int(serv_m.group(1)) if serv_m else None,
            "ingredients": ingredients,
        }
    return recipes


# Hebrew / common names -> recipe slug, so "שקשוקה" or "schnitzel" both resolve
_RECIPE_ALIASES = {
    "שקשוקה": "shakshuka", "חומוס": "hummus", "מרק עדשים": "lentil_soup",
    "שניצל": "chicken_schnitzel", "schnitzel": "chicken_schnitzel",
    "חלה": "challah", "פלאפל": "falafel", "סלט ישראלי": "israeli_salad",
    "סלט קצוץ": "israeli_salad", "עוגת בננה": "banana_bread", "לחם בננה": "banana_bread",
    "פסטה": "pasta_pomodoro", "pomodoro": "pasta_pomodoro", "spaghetti": "pasta_pomodoro",
    "עוגיות טחינה": "tahini_cookies", "lentil soup": "lentil_soup",
    "salad": "israeli_salad", "banana": "banana_bread",
}


def find_recipe(name: str) -> dict | None:
    recipes = load_recipes()
    q = name.strip().lower()
    if not q:
        return None
    for alias, slug in _RECIPE_ALIASES.items():
        if alias in q:
            return recipes.get(slug)
    qs = _slug(q)
    if qs in recipes:
        return recipes[qs]
    # word-overlap match against slug and title
    q_words = set(qs.split("_")) - {"the", "a", "my", "recipe", "classic"}
    best, best_score = None, 0
    for slug, r in recipes.items():
        words = set(slug.split("_")) | set(_slug(r["title"]).split("_"))
        score = len(q_words & words)
        if score > best_score:
            best, best_score = r, score
    return best


def recipe_titles() -> list[str]:
    return [r["title"] for r in load_recipes().values()]
