"""Public read-only viewer for the control center.

    python -m secondlook.viewer [port]        (default 8772; ./sl viewer)

Sits in front of the control center (127.0.0.1:8770) and is the ONLY thing a public tunnel should point at
(Tailscale Funnel: `tailscale funnel --bg 8772`). It forwards an allowlist of read-only requests, strips
local details (file paths, process ids, log names), caches answers for a moment so any number of visitors
cost the laptop about one request per endpoint every two seconds, and refuses everything else: no start,
stop, labels, model switching, voice or scene images (the top/side cameras can see people; only the wrist
camera's inspection views are shown).

It also serves the dashboard page itself in view-only mode, so the tunnel URL alone is a working live page.
The Netlify copy (./sl publish) calls this server for live data and falls back to the last snapshot.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CENTER = "http://127.0.0.1:8770"
STATIC = Path(__file__).with_name("static")

# path -> (allowed query keys, cache seconds)
API = {
    "/api/status": ((), 1.5), "/api/live": ((), 1.5), "/api/scene": ((), 5), "/api/results": ((), 10),
    "/api/inspections": ((), 10), "/api/history": (("limit", "run", "truth", "verdict", "picked"), 4),
    "/api/history/runs": ((), 10), "/api/picks": (("limit",), 10), "/api/learning": ((), 10),
    "/api/models": ((), 30), "/api/voice": ((), 10), "/api/trends": ((), 10), "/api/aac": ((), 300),
}
FRAME = "/live/frame.jpg"
IMG = re.compile(r"/img/([\w.-]+\.png)")           # wrist-camera crops only, never /img/full or /scene
LIMITS = {"limit": 200}
MAX_FETCHES = threading.BoundedSemaphore(4)       # never more than 4 requests at once to the control center

_cache: dict[str, tuple[float, int, bytes, str]] = {}
_cache_lock = threading.Lock()
_hits: dict[str, list[float]] = {}
_hits_lock = threading.Lock()
RATE = (120, 10.0)                                # per visitor: 120 requests per 10 s (the page uses ~20)


def _clean_query(path, raw):
    allowed = API.get(path, ((), 0))[0]
    q = urllib.parse.parse_qs(raw)
    out = {}
    for k in allowed:
        if k in q:
            v = q[k][0][:64]
            if k in LIMITS:
                try:
                    v = str(max(1, min(LIMITS[k], int(v))))
                except ValueError:
                    continue
            out[k] = v
    return urllib.parse.urlencode(sorted(out.items()))


def _scrub(path, data):
    """Remove local details before anything leaves the laptop."""
    if path == "/api/status" and isinstance(data, dict):
        job = data.get("job") or {}
        job.pop("log", None)
        job["external"] = [1] * len(job.get("external") or [])     # keep "something else is running", drop pids
        data["owners"] = {k: (None if v is None else []) for k, v in (data.get("owners") or {}).items()}
        m = data.get("model") or {}
        if m.get("inspection"):
            m["inspection"] = "PatchCore"
    if path == "/api/models" and isinstance(data, dict):
        for x in data.get("learned") or []:
            x.pop("path", None)
        d = data.get("defect") or {}
        d.pop("path", None)
        for v in d.get("versions") or []:
            v.pop("path", None)
    if path == "/api/inspections" and isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                x.pop("path", None)
    return data


def fetch(path, query=""):
    """(status, body, content type) from the control center, through the cache."""
    key = path + ("?" + query if query else "")
    ttl = 300 if path.startswith("/img/") else 1.0 if path == FRAME else API[path][1]
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1:]
    with MAX_FETCHES:
        with _cache_lock:                           # someone else may have fetched it while we waited
            hit = _cache.get(key)
            if hit and time.time() - hit[0] < ttl:
                return hit[1:]
        try:
            r = urllib.request.urlopen(CENTER + key, timeout=5)
            status, body, ctype = r.status, r.read(), r.headers.get("Content-Type", "application/octet-stream")
        except urllib.error.HTTPError as e:
            status, body, ctype = e.code, b'{"error": "not available"}', "application/json"
        except Exception:
            status, body, ctype = 502, b'{"error": "station not reachable"}', "application/json"
    if status == 200 and ctype.startswith("application/json"):
        try:
            body = json.dumps(_scrub(path, json.loads(body))).encode()
        except ValueError:
            pass
    with _cache_lock:
        _cache[key] = (time.time(), status, body, ctype)
        if len(_cache) > 2000:                      # photos: keep memory bounded
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:500]:
                _cache.pop(k, None)
    return status, body, ctype


def page_config(api="", snapshot=None, built=None):
    return ("window.SL_PUBLIC = " + json.dumps({"api": api, "snapshot": snapshot, "built": built}) + ";\n").encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "SecondLookViewer"
    sys_version = ""

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", cache="no-store"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _limited(self):
        who = (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        if not who:
            return False                            # no visitor address (local); the cache protects the station
        now, (n, win) = time.time(), RATE
        with _hits_lock:
            q = [t for t in _hits.get(who, []) if now - t < win]
            q.append(now)
            _hits[who] = q
            if len(_hits) > 5000:
                _hits.clear()
        return len(q) > n

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self):
        path, _, raw = self.path.partition("?")
        if self._limited():
            return self._send(429, {"error": "too many requests, slow down"})
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "center.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/access.js":
                return self._send(200, (STATIC / "access.js").read_bytes(), "text/javascript; charset=utf-8")
            if path == "/config.js":
                return self._send(200, page_config(), "text/javascript; charset=utf-8")
            if path in API:
                return self._send(*fetch(path, _clean_query(path, raw)))
            if path == FRAME:
                status, body, ctype = fetch(FRAME)
                return self._send(status, body, ctype)
            m = IMG.fullmatch(path)
            if m:
                status, body, ctype = fetch("/img/" + m.group(1))
                return self._send(status, body, ctype, "public, max-age=86400" if status == 200 else "no-store")
            return self._send(404, {"error": "not available on the public view"})
        except Exception as e:                      # never leak a traceback
            print(f"viewer error on {path}: {type(e).__name__}: {e}", flush=True)
            return self._send(500, {"error": "internal error"})

    do_HEAD = do_GET

    def _refuse(self):
        self._send(405, {"error": "View only: this is the public SecondLook dashboard."})

    do_POST = do_PUT = do_PATCH = do_DELETE = _refuse


def main(port=8772):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    print(f"SecondLook public viewer (read-only) on http://127.0.0.1:{port} -> {CENTER}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8772)
