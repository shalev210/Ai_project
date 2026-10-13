/* SousChef client — vanilla JS. Talks to the Flask server over REST/JSON only:
 *   POST /chat      {query, session_id}  -> {answer, route, route_label, router, sources, tool_calls, ...}
 *   GET  /history   ?session_id=...       -> {turns: [...]}
 *   GET  /health                          -> {status, database, ...}
 * No API keys, no direct LLM or database access live here.
 */
(() => {
  "use strict";

  // Same origin when served by Flask; localhost:5000 when index.html is opened from disk.
  const API_BASE = location.protocol === "file:" ? "http://localhost:5000" : "";
  const SESSION_KEY = "souschef_session_id";

  const $messages = document.getElementById("messages");
  const $empty = document.getElementById("empty");
  const $form = document.getElementById("composer");
  const $input = document.getElementById("input");
  const $send = document.getElementById("send");
  const $status = document.getElementById("status");
  const $statusText = document.getElementById("status-text");
  const thinkingTpl = document.getElementById("tpl-thinking");

  // ---------------------------------------------------------------- session
  function newSessionId() {
    if (crypto.randomUUID) return crypto.randomUUID().replace(/-/g, "");
    return Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, "0")).join("");
  }
  function getSessionId() {
    let id = null;
    try { id = localStorage.getItem(SESSION_KEY); } catch (_) { /* storage blocked */ }
    if (!id) {
      id = newSessionId();
      try { localStorage.setItem(SESSION_KEY, id); } catch (_) { /* ignore */ }
    }
    return id;
  }
  let sessionId = getSessionId();

  // ------------------------------------------------------------------- http
  async function api(path, options = {}) {
    let res;
    try {
      res = await fetch(API_BASE + path, {
        ...options,
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      });
    } catch (_) {
      throw new Error("Can't reach the SousChef server. Check that it is running on port 5000.");
    }
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) {
      const msg = data && data.error ? data.error.message : `Server returned ${res.status}.`;
      const err = new Error(msg);
      err.route = data && data.route;
      throw err;
    }
    return data;
  }

  // ---------------------------------------------------------------- render
  const ROUTE_CLASS = { rag: "tag-rag", tool: "tag-tool", direct: "tag-direct" };

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    for (const c of children) if (c) node.append(c);
    return node;
  }

  function scrollToEnd() { $messages.scrollTop = $messages.scrollHeight; }

  function addUser(text) {
    $empty.hidden = true;
    $messages.append(el("div", { class: "msg user" }, el("div", { class: "bubble", dir: "auto", text })));
    scrollToEnd();
  }

  function addAssistant(data) {
    const wrap = el("div", { class: `msg assistant ${ROUTE_CLASS[data.route] || ""}` });

    const router = data.router || {};
    const tagTitle = router.reason
      ? `Router (${router.method}, confidence ${Number(router.confidence || 0).toFixed(2)}): ${router.reason}`
      : "";
    wrap.append(el("span", { class: "route-tag", title: tagTitle, text: data.route_label || data.route }));
    wrap.append(el("div", { class: "bubble", dir: "auto", text: data.answer }));

    const meta = el("div", { class: "meta" });

    if (router.reason) {
      meta.append(el("div", { class: "reason", text: `Why this path: ${router.reason} (${router.method})` }));
    }

    if (data.sources && data.sources.length) {
      const list = el("ol");
      data.sources.forEach(s => {
        list.append(el("li", {},
          el("strong", { text: `${s.title} — ${s.section}` }), " ",
          el("span", { class: "score", text: `${(s.similarity * 100).toFixed(0)}% match` }),
          el("div", { dir: "auto", text: s.snippet + "…" })));
      });
      meta.append(el("details", {}, el("summary", { text: `Sources (${data.sources.length})` }), list));
    }

    if (data.tool_calls && data.tool_calls.length) {
      const box = el("details", {}, el("summary", { text: `Tool calls (${data.tool_calls.length})` }));
      data.tool_calls.forEach(c => {
        const body = c.ok ? c.output : { error: c.error };
        box.append(el("div", { text: `${c.ok ? "✓" : "✗"} ${c.name}` }),
          el("pre", { text: `input:  ${JSON.stringify(c.input)}\noutput: ${JSON.stringify(body, null, 2)}` }));
      });
      meta.append(box);
    }

    (data.warnings || []).forEach(w => meta.append(el("div", { class: "warn", text: w })));
    if (meta.childNodes.length) wrap.append(meta);

    $messages.append(wrap);
    scrollToEnd();
  }

  function addError(message) {
    $messages.append(el("div", { class: "msg assistant error" },
      el("div", { class: "bubble", role: "alert", text: message })));
    scrollToEnd();
  }

  // --------------------------------------------------------------- actions
  let busy = false;

  async function send(text) {
    const query = text.trim();
    if (!query || busy) return;
    busy = true;
    $send.disabled = true;
    addUser(query);
    $input.value = "";
    autosize();

    const thinking = thinkingTpl.content.firstElementChild.cloneNode(true);
    $messages.append(thinking);
    scrollToEnd();

    try {
      const data = await api("/chat", { method: "POST", body: JSON.stringify({ query, session_id: sessionId }) });
      thinking.remove();
      addAssistant(data);
    } catch (err) {
      thinking.remove();
      addError(err.message);
    } finally {
      busy = false;
      $send.disabled = false;
      $input.focus();
    }
  }

  async function loadHistory() {
    try {
      const data = await api(`/history?session_id=${encodeURIComponent(sessionId)}&limit=50`, { method: "GET" });
      data.turns.forEach(t => {
        if (t.role === "user") addUser(t.content);
        else addAssistant({
          answer: t.content, route: t.route, sources: t.sources,
          route_label: { rag: "RAG", tool: t.tool_name ? `Tool: ${t.tool_name}` : "Tool", direct: "Direct LLM" }[t.route] || t.route,
          router: { reason: t.route_reason, method: "saved" },
        });
      });
    } catch (_) { /* no history yet or DB down — health indicator shows it */ }
  }

  async function checkHealth() {
    try {
      const h = await api("/health", { method: "GET" });
      $status.className = "status ok";
      $statusText.textContent = h.llm_configured ? `Ready · ${h.knowledge_base_chunks} chunks` : "No API key set";
      if (!h.llm_configured) $status.className = "status down";
    } catch (err) {
      $status.className = "status down";
      $statusText.textContent = "Offline";
      $status.title = err.message;
    }
  }

  function autosize() {
    $input.style.height = "auto";
    $input.style.height = Math.min($input.scrollHeight, 160) + "px";
  }

  // ---------------------------------------------------------------- events
  $form.addEventListener("submit", e => { e.preventDefault(); send($input.value); });
  $input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($input.value); }
  });
  $input.addEventListener("input", autosize);
  document.querySelectorAll(".suggestion").forEach(b => b.addEventListener("click", () => send(b.dataset.q)));
  document.getElementById("new-chat").addEventListener("click", () => {
    sessionId = newSessionId();
    try { localStorage.setItem(SESSION_KEY, sessionId); } catch (_) { /* ignore */ }
    $messages.querySelectorAll(".msg").forEach(n => n.remove());
    $empty.hidden = false;
    $input.focus();
  });

  checkHealth();
  loadHistory();
})();
