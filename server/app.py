"""SousChef — Flask server (sole owner of all AI logic, secrets and DB access).

Endpoints
  POST /chat      run the full AI pipeline for one user query
  GET  /history   past conversation turns for a session  (?session_id=...&limit=50)
  GET  /health    server + database liveness
  GET  /          serves the HTML/JS client (static files from ../client)
"""
from __future__ import annotations

import logging
import re
import threading
import uuid

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

import config
import db
import formatter
import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("souschef")

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    # Allows opening client/index.html directly from disk or another port during development
    CORS(app, resources={r"/(chat|history|health)": {"origins": "*"}})

    # ----------------------------------------------------------- client
    @app.get("/")
    def index():
        return send_from_directory(config.CLIENT_DIR, "index.html")

    @app.get("/<path:filename>")
    def client_static(filename: str):  # style.css, app.js (lowest-priority route)
        return send_from_directory(config.CLIENT_DIR, filename)

    # ----------------------------------------------------------- /health
    @app.get("/health")
    def health():
        try:
            info = db.ping()
            return jsonify({"status": "ok", "server": "up", "database": "up",
                            "knowledge_base_chunks": info["chunks"],
                            "llm_configured": bool(config.ANTHROPIC_API_KEY),
                            "router_mode": config.ROUTER_MODE}), 200
        except Exception as e:
            log.warning("Health check DB failure: %s", e)
            return jsonify({"status": "degraded", "server": "up", "database": "down",
                            "error": type(e).__name__}), 503

    # ------------------------------------------------------------- /chat
    @app.post("/chat")
    def chat():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(formatter.error("bad_request", "Request body must be a JSON object.")), 400

        query = body.get("query", body.get("message"))
        if not isinstance(query, str) or not query.strip():
            return jsonify(formatter.error("bad_request", "Field 'query' (non-empty string) is required.")), 400
        query = query.strip()
        if len(query) > config.MAX_QUERY_CHARS:
            return jsonify(formatter.error("bad_request",
                                           f"'query' is too long (max {config.MAX_QUERY_CHARS} characters).")), 400

        session_id = body.get("session_id") or uuid.uuid4().hex
        if not isinstance(session_id, str) or not SESSION_RE.match(session_id):
            return jsonify(formatter.error("bad_request",
                                           "'session_id' must be 8-64 characters of letters, digits, '-' or '_'.")), 400

        try:
            return jsonify(pipeline.run(query, session_id)), 200
        except pipeline.PipelineError as e:
            return jsonify(formatter.error(e.code, e.message, session_id, e.route)), e.http_status
        except Exception:
            log.exception("Unhandled error in /chat")
            return jsonify(formatter.error("internal_error",
                                           "Something went wrong while answering. Please try again.",
                                           session_id)), 500

    # ---------------------------------------------------------- /history
    @app.get("/history")
    def history():
        session_id = request.args.get("session_id", "")
        if not SESSION_RE.match(session_id):
            return jsonify(formatter.error("bad_request", "Query parameter 'session_id' is required.")), 400
        try:
            limit = max(1, min(int(request.args.get("limit", 50)), 200))
        except ValueError:
            return jsonify(formatter.error("bad_request", "'limit' must be an integer.")), 400
        try:
            turns = db.get_history(session_id, limit)
            tool_log = db.get_tool_calls(session_id, limit) if request.args.get("include_tools") == "true" else None
        except Exception as e:
            log.exception("History query failed")
            return jsonify(formatter.error("db_unavailable", f"Database error ({type(e).__name__}).",
                                           session_id)), 503
        payload = {"session_id": session_id, "count": len(turns), "turns": turns}
        if tool_log is not None:
            payload["tool_calls"] = tool_log
        return jsonify(payload), 200

    # -------------------------------------------------- JSON for all errors
    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        return jsonify(formatter.error(e.name.lower().replace(" ", "_"), e.description)), e.code

    return app


app = create_app()

if __name__ == "__main__":
    if config.AUTO_INGEST:
        import ingest
        # background thread: the server is reachable while the embedding model downloads
        threading.Thread(target=ingest.ensure_ingested, daemon=True).start()
    app.run(host="0.0.0.0", port=config.SERVER_PORT, debug=False)
