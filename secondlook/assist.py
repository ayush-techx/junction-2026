"""AccessInspect helpers (Ava voice + Gemini defect naming), reached through accessinspect/sl_bridge.py.

The bridge runs in AccessInspect's own Python (accessinspect/.venv), so nothing is installed into the
Studio environment SecondLook runs in. Everything here is best-effort: if the bridge, the internet or
the API key is missing, calls return None and SecondLook carries on with its offline voice and PatchCore.

    assist.available()                  bridge + venv present
    assist.cached_voice(text)           path of an already-made Ava mp3, or None (no process needed)
    assist.synth(text, timeout)         make (or fetch) the Ava mp3 for text -> path or None
    assist.phrase(event, defect, color) AccessInspect's wording for pass / reject -> text or None
    assist.check(path) -> Pending       Gemini look at a saved brick photo; .get(timeout) -> dict or None

Gemini only NAMES the defect and the colour for the announcement and the log. PatchCore decides the bowl:
on the 29 real-grasp photos (2026-11-15) Gemini cleared 5 of 6 PatchCore false alarms but missed 5 of 13
real defects, including the only defect inside the unsure band.
"""
import hashlib
import itertools
import json
import os
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI = os.path.join(ROOT, "accessinspect")
PY = os.path.join(AI, ".venv", "bin", "python")
BRIDGE = os.path.join(AI, "sl_bridge.py")
CACHE = os.path.join(AI, "voice_cache")
VOICE = "en-US-AvaMultilingualNeural"          # must match accessinspect/speak_alert.py
LOG = os.path.join(ROOT, "runs", "logs", "accessinspect_bridge.log")

_lock = threading.Lock()
_proc = None
_ids = itertools.count(1)
_waiting = {}                                   # id -> Pending
_dead_until = 0.0                               # after a crash, do not respawn for a while


class Pending:
    def __init__(self):
        self._ev = threading.Event()
        self.value = None
        self.error = None

    def _set(self, msg):
        self.error = msg.get("error")
        self.value = None if self.error else msg
        self._ev.set()

    def done(self):
        return self._ev.is_set()

    def get(self, timeout=0.0):
        """The reply dict, or None if it failed or is not back within timeout seconds."""
        self._ev.wait(max(0.0, timeout))
        return self.value if self._ev.is_set() else None


def available():
    return os.path.exists(PY) and os.path.exists(BRIDGE)


def cached_voice(text):
    p = os.path.join(CACHE, hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:16] + ".mp3")
    return p if os.path.exists(p) and os.path.getsize(p) > 0 else None


def _reader(proc):
    for line in proc.stdout:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        p = _waiting.pop(msg.get("id"), None)
        if p is not None:
            p._set(msg)
    for p in list(_waiting.values()):              # bridge exited: fail whoever is still waiting
        p._set({"error": "bridge exited"})
    _waiting.clear()


def _bridge():
    global _proc, _dead_until
    if _proc is not None and _proc.poll() is None:
        return _proc
    if not available() or time.time() < _dead_until:
        return None
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    try:
        _proc = subprocess.Popen([PY, "-W", "ignore", BRIDGE], cwd=AI, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=open(LOG, "a"), text=True, bufsize=1)
    except OSError:
        _dead_until = time.time() + 60
        return None
    threading.Thread(target=_reader, args=(_proc,), daemon=True).start()
    return _proc


def request(op, **kw):
    """Send one request; returns a Pending (already failed if the bridge can't start)."""
    p = Pending()
    with _lock:
        proc = _bridge()
        if proc is None:
            p._set({"error": "bridge unavailable"})
            return p
        rid = next(_ids)
        _waiting[rid] = p
        try:
            proc.stdin.write(json.dumps(dict(kw, op=op, id=rid)) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            _waiting.pop(rid, None)
            p._set({"error": "bridge write failed"})
    return p


def synth(text, timeout=5.0):
    path = cached_voice(text)
    if path:
        return path
    r = request("synth", text=text).get(timeout)
    return r.get("path") if r else None


def phrase(event, defect=None, color=None, timeout=1.0):
    r = request("text", event=event, defect=defect, color=color).get(timeout)
    return r.get("text") if r else None


def check(path):
    """Start a Gemini check of a saved photo. Returns a Pending; .get(t) -> result dict or None."""
    p = request("check", path=os.path.abspath(path))
    out = Pending()

    def unwrap():
        r = p.get(60)
        out._set(r["result"] if r and "result" in r else {"error": p.error or "timeout"})
    threading.Thread(target=unwrap, daemon=True).start()
    return out


def close():
    global _proc
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.stdin.close()
            _proc.wait(2)
        except Exception:
            _proc.kill()
    _proc = None
