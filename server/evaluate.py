"""Offline evaluation of the two most heavily graded components:
  * Agent Router accuracy (expected vs. actual route)
  * RAG retrieval hit-rate (is the expected source file among the top-k chunks?)

Results are stored in the `evaluation_results` table under a run_id.

    python evaluate.py            # router in the configured ROUTER_MODE
    python evaluate.py --rules    # force the rule-based router (no API calls)
"""
from __future__ import annotations

import argparse
import sys
import uuid

import config
import db
import rag
import router

# (query, expected_route, expected_source_for_rag)
EVAL_SET = [
    ("How do I make shakshuka?", "rag", "shakshuka.md"),
    ("איך מכינים חומוס חלק?", "rag", "hummus.md"),
    ("How long can I keep lentil soup in the freezer?", "rag", "lentil_soup.md"),
    ("What can I use instead of eggs in baking?", "rag", "kitchen_guide.md"),
    ("What temperature is chicken safe to eat?", "rag", "kitchen_guide.md"),
    ("Why do my falafel balls fall apart?", "rag", "falafel.md"),
    ("איך לשמור עוגיות טחינה?", "rag", "tahini_cookies.md"),
    ("Can I bake schnitzel in the oven instead of frying?", "rag", "chicken_schnitzel.md"),
    ("How many calories per serving are in the shakshuka?", "tool", None),
    ("כמה קלוריות יש ב-200 גרם טחינה?", "tool", None),
    ("Convert 2 cups of flour to grams", "tool", None),
    ("What is 350 F in Celsius?", "tool", None),
    ("Scale the challah recipe for 24 people", "tool", None),
    ("I need the hummus recipe for 12 servings", "tool", None),
    ("Hi! What can you do?", "direct", None),
    ("What is umami?", "direct", None),
    ("תודה רבה!", "direct", None),
    ("What's the difference between a sauté pan and a skillet?", "direct", None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", action="store_true", help="use the rule-based router only")
    args = ap.parse_args()

    run_id = uuid.uuid4().hex[:12]
    route_ok = hits = rag_total = 0
    print(f"Evaluation run {run_id}  (router: {'rules' if args.rules else config.ROUTER_MODE})\n")
    for query, expected_route, expected_source in EVAL_SET:
        d = router.route_by_rules(query) if args.rules else router.route(query)
        correct = d.route == expected_route
        route_ok += correct

        retrieved, hit = [], None
        if expected_source:
            rag_total += 1
            try:
                res = rag.retrieve(d.search_query or query, min_similarity=0.0)
                retrieved = [s["source"] for s in res["all_scores"]]
                hit = expected_source in retrieved
                hits += bool(hit)
            except Exception as e:
                print(f"   retrieval error: {e}")
        db.save_eval_result(run_id, query, expected_route, d.route, expected_source,
                            retrieved, correct, hit)
        mark = "✓" if correct else "✗"
        extra = "" if hit is None else (" | retrieval " + ("hit" if hit else "MISS"))
        print(f"{mark} [{d.route:6}] expected {expected_route:6} | {query}{extra}")

    print(f"\nRouter accuracy : {route_ok}/{len(EVAL_SET)} = {route_ok / len(EVAL_SET):.0%}")
    if rag_total:
        print(f"Retrieval hit@{config.RAG_TOP_K}: {hits}/{rag_total} = {hits / rag_total:.0%}")
    print(f"Saved to evaluation_results (run_id={run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
