# Topic Proposal — SousChef (Appendix A)

**Project title:** SousChef — an agentic recipe & nutrition assistant

**Target user (who will use this, and why?):**
Home cooks, including beginners, who want quick, trustworthy answers while cooking: how to make a dish, how long it keeps, what to substitute, safe cooking temperatures, how many calories a serving has, how to convert "cups" to grams, or how to scale a recipe for more guests. Many Israeli home cooks mix Hebrew and English, so the assistant accepts both.

**Knowledge source(s) for RAG:**
A curated collection of 10 recipes written in a consistent Markdown format (title, servings, tags, allergens, ingredients in grams, method, tips & variations, storage) plus a Kitchen Safety & Substitutions guide (safe internal temperatures, fridge/freezer rules, common substitutions, kosher notes, allergens). Stored in `data/recipes/`, chunked by section and embedded into pgvector.

**Planned tools:**
1. `lookup_recipe_ingredients` — structured lookup of a recipe's exact ingredient list and base servings.
2. `calculate_nutrition` — totals and per-serving calories, protein, fat, carbs and fiber from a USDA-based reference table.
3. `convert_units` — weight / volume / temperature conversion, with ingredient density for cups ↔ grams.
4. `scale_recipe` — rescales every ingredient to a target number of servings.

**Database choice and vector storage approach:**
PostgreSQL 16 in Docker, with the pgvector extension for the vector store (single container for relational data and embeddings).

**Anticipated risks / open questions:**
- Hebrew questions against English documents → mitigated with a multilingual embedding model and an LLM router that rewrites the search query in English.
- Nutrition values are approximate → answers say so; values come only from the tool, never from the LLM's memory.
- Ambiguous queries that are both "recipe" and "calculation" → router rule: computation wins, and the tool path can look up the recipe itself.
- Embedding model download on first run (~220 MB) → cached in a Docker volume.
