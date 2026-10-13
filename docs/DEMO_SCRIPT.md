# Demo Script (≈ 5 minutes)

Before recording: `docker compose up --build`, wait until `/health` shows the chunk count, open http://localhost:5000 and click **New chat**.

| # | Show | Type this | What to point out |
|---|---|---|---|
| 1 | Setup | `docker compose ps`, then open http://localhost:5000/health | Both containers healthy, DB up, chunks indexed |
| 2 | **RAG** | `How do I make shakshuka?` | Green **RAG** tag, router reason, open **Sources** — chunks from `shakshuka.md` with similarity scores, inline [1] citations |
| 3 | RAG in Hebrew | `איך שומרים מרק עדשים במקפיא?` | Hebrew answer; source is the English *Red Lentil Soup → Storage* chunk (multilingual embeddings) |
| 4 | **Tool** | `How many calories per serving are in the hummus?` | Saffron **Tool: Recipe Lookup + Nutrition Calculator** tag; open **Tool calls** — lookup, then nutrition with the exact grams |
| 5 | Tool | `Convert 2 cups of flour to grams` | **Tool: Unit Converter**, density used |
| 6 | **Direct LLM** | `What is umami?` | Slate **Direct LLM** tag, no sources, no tools |
| 7 | Memory | `How do I make falafel?` → `How long does it keep in the freezer?` | Second question has no recipe name; router resolves "it" = falafel |
| 8 | Error handling | Stop the DB: `docker compose stop db`, refresh the page | Status dot turns red (`/health` → 503). Ask `What is umami?`: the server still answers, with a visible warning that the answer could not be saved to history — no crash. `docker compose start db` to recover |
| 9 | Persistence | `docker compose exec db psql -U souschef -d souschef -c "SELECT tool_name, success, tool_input FROM tool_calls ORDER BY id DESC LIMIT 5;"` | Tool-call log in PostgreSQL |
| 10 | Evaluation | `docker compose exec server python evaluate.py` | Router accuracy and retrieval hit-rate, saved to `evaluation_results` |
