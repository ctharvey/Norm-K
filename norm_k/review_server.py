"""Local-only, dependency-free human annotation UI.

Do not expose this HTTP server to a network. It binds to 127.0.0.1, checks
Host/Origin, requires a per-process CSRF token, and never returns oracle labels.
"""

from __future__ import annotations

import html
import json
import secrets
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .review import AnnotationLog, ReviewItem, latest_reviews


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="review-token" content="__TOKEN__">
  <title>‖K‖ — independent statement review</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #0d1425; color: #e9efff; }
    main { max-width: 980px; margin: 0 auto; padding: 24px 20px 60px; }
    h1 { font-size: 1.9rem; margin-bottom: 3px; }
    .muted { color: #a6b4cf; font-size: .91rem; }
    .notice { border-left: 4px solid #facb6d; background: #27253a; padding: 12px 16px; }
    .top { display: flex; gap: 15px; flex-wrap: wrap; align-items: end; margin: 20px 0; }
    label { display: block; font-size: .9rem; color: #aebdda; margin-bottom: 6px; }
    input, textarea, select { background: #19243a; color: #fff; border: 1px solid #52617e;
      border-radius: 9px; padding: 10px; font: inherit; box-sizing: border-box; }
    input { width: 240px; } textarea { width: 100%; min-height: 90px; }
    .card { border: 1px solid #364667; background: #172037; border-radius: 16px;
      padding: 23px; margin: 17px 0; }
    .statement { background: #101a2d; border-radius: 10px; padding: 20px; margin: 10px 0;
      font-size: 1.17rem; white-space: pre-wrap; overflow-wrap: anywhere; }
    .actions { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 9px; }
    button { color: #e9efff; background: #263959; border: 1px solid #4b678e;
      border-radius: 10px; padding: 12px; font: inherit; cursor: pointer; }
    button:hover, button:focus-visible { background: #335680; outline: 2px solid #84aff5; }
    button:disabled { opacity: .4; cursor: default; }
    .nav { display: flex; justify-content: space-between; gap: 10px; margin-top: 15px; }
    .status { color: #a8edcc; } .error { color: #ffc2c2; }
    .progress { height: 8px; background: #283a55; border-radius: 8px; overflow: hidden; }
    .bar { height: 8px; background: #63b7a3; width: 0; transition: width .15s; }
  </style>
</head>
<body>
<main>
  <h1>‖K‖ Independent Review</h1>
  <p class="muted">Label pairs by what they mean, including subject, time, and scope. No model predictions or reference labels are shown.</p>
  <p class="notice">Run this only on your own computer. Annotations are saved locally to an append-only JSONL log. Uncertain cases belong in <strong>Ambiguous</strong>.</p>
  <div class="top">
    <div><label for="reviewer">Reviewer ID (use the same ID to resume)</label><input id="reviewer" maxlength="80" autocomplete="off" placeholder="e.g. charlie"></div>
    <button id="load">Load my queue</button>
    <span id="status" class="muted" role="status" aria-live="polite"></span>
  </div>
  <div class="progress"><div class="bar" id="bar"></div></div>
  <p id="progress" class="muted"></p>
  <section class="card" id="reviewCard" hidden>
    <div class="muted" id="pairId"></div>
    <h2>Statement A</h2><div class="statement" id="textA"></div>
    <h2>Statement B</h2><div class="statement" id="textB"></div>
    <div class="top"><div><label for="confidence">Confidence</label><select id="confidence"><option>high</option><option selected>medium</option><option>low</option></select></div></div>
    <label for="notes">Rationale / relevant time or scope (optional)</label>
    <textarea id="notes" maxlength="2000" placeholder="Why do these statements conflict—or not?"></textarea>
    <p class="muted">Keyboard: 1–6 selects a label when not typing.</p>
    <div class="actions" id="labels">
      <button data-label="contradiction">1 · Contradiction</button>
      <button data-label="equivalent">2 · Equivalent</button>
      <button data-label="compatible">3 · Compatible</button>
      <button data-label="temporal_update">4 · Temporal update</button>
      <button data-label="unrelated">5 · Unrelated</button>
      <button data-label="ambiguous">6 · Ambiguous / unclear</button>
    </div>
    <div class="nav"><button id="previous">Previous</button><button id="next">Next / skip</button></div>
  </section>
</main>
<script>
const token = document.querySelector('meta[name="review-token"]').content;
const $ = id => document.getElementById(id);
let items = [], completed = {}, position = 0, busy = false;
function status(message, isError=false) { $('status').textContent = message; $('status').className = isError ? 'error' : 'status'; }
function render() {
  const total = items.length, done = Object.keys(completed).length;
  $('bar').style.width = (total ? (100*done/total) : 0) + '%';
  $('progress').textContent = `${done} of ${total} reviewed`;
  $('reviewCard').hidden = !total;
  if (!total) return;
  position = Math.max(0, Math.min(total-1, position));
  const item = items[position];
  $('pairId').textContent = `Pair ${position+1} / ${total}${completed[item.id] ? ' · previously reviewed' : ''}`;
  $('textA').textContent = item.text_a;
  $('textB').textContent = item.text_b;
  $('notes').value = '';
  $('confidence').value = 'medium';
  $('previous').disabled = position === 0;
  $('next').disabled = position === total - 1;
}
function nextUnreviewed(from=0) {
  for (let i=from; i<items.length; i++) if (!completed[items[i].id]) return i;
  for (let i=0; i<from; i++) if (!completed[items[i].id]) return i;
  return Math.min(items.length-1, from);
}
async function load() {
  const reviewer = $('reviewer').value.trim();
  if (!reviewer) return status('Enter a reviewer ID first.', true);
  try {
    const response = await fetch('/api/queue?reviewer='+encodeURIComponent(reviewer));
    if (!response.ok) throw Error('Unable to load queue');
    const data = await response.json();
    items = data.items; completed = data.completed;
    position = nextUnreviewed(0); render(); status('Queue ready.');
  } catch (error) { status(error.message, true); }
}
async function save(label) {
  if (busy || !items.length || !$('reviewer').value.trim()) return;
  busy = true;
  document.querySelectorAll('#labels button').forEach(button => button.disabled = true);
  try {
    const response = await fetch('/api/reviews', {method:'POST', headers:{
      'Content-Type':'application/json', 'X-Review-Token':token}, body:JSON.stringify({
      pair_id:items[position].id, reviewer:$('reviewer').value.trim(), label,
      confidence:$('confidence').value, notes:$('notes').value})});
    if (!response.ok) throw Error((await response.json()).error || 'Unable to save');
    completed[items[position].id] = label;
    position = nextUnreviewed(position+1); render(); status('Saved '+label+'.');
  } catch (error) { status(error.message, true); }
  finally { busy = false; document.querySelectorAll('#labels button').forEach(button => button.disabled = false); }
}
$('load').onclick = load;
$('labels').querySelectorAll('button').forEach(button => button.onclick=()=>save(button.dataset.label));
$('previous').onclick = () => {position--;render();};
$('next').onclick = () => {position++;render();};
document.addEventListener('keydown',event => {
  if (event.target.matches('input,textarea,select') || !items.length) return;
  if ('123456'.includes(event.key) && event.key.length === 1) {
    const choices=['contradiction','equivalent','compatible','temporal_update','unrelated','ambiguous'];
    save(choices[Number(event.key)-1]);
  }
});
</script>
</body></html>"""


def create_handler(items: list[ReviewItem], log: AnnotationLog,
                   token: str | None = None) -> type[BaseHTTPRequestHandler]:
    token = token or secrets.token_urlsafe(32)
    public_items = [item.for_reviewer() for item in items]

    class Handler(BaseHTTPRequestHandler):
        def _trusted(self) -> bool:
            port = self.server.server_address[1]
            permitted_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in permitted_hosts:
                return False
            origin = self.headers.get("Origin")
            return not origin or origin in {f"http://{host}" for host in permitted_hosts}

        def _reply(self, status: int, content: bytes, mime: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def _json(self, status: int, value: dict) -> None:
            self._reply(status, json.dumps(value).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if not self._trusted():
                return self._json(403, {"error": "Untrusted local origin/host"})
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page = _HTML.replace("__TOKEN__", html.escape(token, quote=True))
                return self._reply(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if parsed.path == "/api/queue":
                reviewer = parse_qs(parsed.query).get("reviewer", [""])[0].strip()
                if not 1 <= len(reviewer) <= 80:
                    return self._json(400, {"error": "Reviewer ID required (1–80 characters)"})
                latest = latest_reviews(log.read_all())
                completed = {pair_id: review.label for (pair_id, who), review in latest.items()
                             if who == reviewer}
                return self._json(200, {"items": public_items, "completed": completed})
            return self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if not self._trusted():
                return self._json(403, {"error": "Untrusted local origin/host"})
            if urlparse(self.path).path != "/api/reviews":
                return self._json(404, {"error": "Not found"})
            if self.headers.get("X-Review-Token") != token:
                return self._json(403, {"error": "Review token required"})
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self._json(415, {"error": "JSON content type required"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 2 or size > 16_384:
                    return self._json(413, {"error": "Review payload size outside permitted range"})
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict) or any(not isinstance(payload.get(k), str)
                                                         for k in ("pair_id", "reviewer", "label", "confidence", "notes")):
                    raise ValueError("Expected string-valued review fields")
                review = log.add(**{key: payload[key] for key in
                                    ("pair_id", "reviewer", "label", "confidence", "notes")})
                return self._json(201, {"review_id": review.review_id, "saved": True})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})

        def log_message(self, _format: str, *_args: object) -> None:
            # Avoid printing text or reviewer identifiers into service logs.
            return

    return Handler


def serve_review_ui(items: list[ReviewItem], log_path: str, port: int = 8765) -> None:
    if not 0 <= port <= 65535:
        raise ValueError("Port must be in 0..65535")
    log = AnnotationLog(log_path, [item.id for item in items])
    server = ThreadingHTTPServer(("127.0.0.1", port), create_handler(items, log))
    server.daemon_threads = True
    print(f"Norm-K local reviewer: http://127.0.0.1:{server.server_port}/")
    print("Reference labels are hidden. Press Ctrl+C to stop. Never expose this port publicly.")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
