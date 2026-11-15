"""The station's voice. Offline, never blocks the robot loop, never crashes it over audio.

Engines, best first (runs/voice.json "engine": auto | ava | piper | espeak):
  Ava      AccessInspect's neural voice (accessinspect/, edge-tts via secondlook/assist.py); phrases are
           cached as mp3 so the common ones work offline, new ones need internet once
  Piper    neural text-to-speech on the CPU (tts-env/ + voices/*.onnx), one long-running process
  espeak-ng / spd-say   always there as the fallback

What it says is built from what the cameras and the defect model measured (no language model):
  describe_table(bricks)          "Five bricks on the table: two blue, one red, one yellow and one green."
  brick_line(verdict, counts, …)  "Good brick. Pink bowl. Three left."
  summary_line(counts, seconds)   "Table clear. Five bricks sorted in one minute: four good, one defective."

Voice on/off lives in runs/voice.json so the dashboard can switch it during a run.
"""
import glob
import json
import os
import queue
import shutil
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(ROOT, "runs", "voice.json")
PIPER = os.path.join(ROOT, "tts-env", "bin", "piper")
VOICES = sorted(glob.glob(os.path.join(ROOT, "voices", "*.onnx")))
_ESPEAK = next((e for e in ("espeak-ng", "espeak", "spd-say") if shutil.which(e)), None)
_FFPLAY = shutil.which("ffplay") or shutil.which("mpg123")
_PLAYER = next((p for p in ("paplay", "aplay", "pw-play") if shutil.which(p)), None)

PHRASES = {
    "good": "Good brick.",
    "defective": "Defect found.",
    "uncertain": "Not sure. Taking a second look.",
    "missed": "Missed it. Trying again.",
    "ready": "Ready.",
    "handfed": "Hand me a brick.",
    "stopped": "Stopped.",
    "table_clear": "The table is clear.",
    "scanning": "Looking at the table.",
}
# "Speak for me" buttons for a non-speaking operator: the station says it out loud (fixed list, no free text).
AAC = {
    "help": "I need help, please.",
    "wait": "Please wait a moment.",
    "yes": "Yes.",
    "no": "No.",
    "thanks": "Thank you.",
    "check": "Can you check this brick, please?",
    "done": "I am finished.",
    "again": "Please say that again.",
}
NUM = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def settings():
    try:
        with open(SETTINGS) as f:
            return dict({"on": True}, **json.load(f))
    except (OSError, ValueError):
        return {"on": True}


def set_voice(on):
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    s = settings()
    s["on"] = bool(on)
    with open(SETTINGS, "w") as f:
        json.dump(s, f)
    return s


def set_engine(name):
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    s = settings()
    s["engine"] = name
    with open(SETTINGS, "w") as f:
        json.dump(s, f)
    return s


def _ava_ok():
    try:
        from . import assist
        return bool(_FFPLAY) and assist.available()
    except Exception:
        return False


def _offline_engine():
    if os.path.exists(PIPER) and VOICES and _PLAYER:
        return "piper"
    return _ESPEAK


def engine():
    want = settings().get("engine", "auto")
    if want in ("auto", "ava") and _ava_ok():
        return "ava"
    if want == "espeak":
        return _ESPEAK
    return _offline_engine()


def n(x):
    return NUM[x] if 0 <= x < len(NUM) else str(x)


def plural(x, word):
    return f"{n(x)} {word}{'' if x == 1 else 's'}"


def _join(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# --- what to say --------------------------------------------------------------------------
def describe_table(bricks, obstacles=0):
    """bricks: list of dicts with 'color' (from scene.detect)."""
    if not bricks:
        text = "The table is clear."
    else:
        by = {}
        for b in bricks:
            c = b.get("color") or "other"
            by[c] = by.get(c, 0) + 1
        order = sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))
        text = f"{plural(len(bricks), 'brick').capitalize()} on the table"
        if len(order) > 1 or order[0][0] not in ("other", "white"):
            text += ": " + _join([f"{n(k)} {c}" for c, k in order])
        text += "."
    if obstacles:
        text += " There is also something on the table that is not a brick."
    return text


def brick_line(verdict, counts=None, left=None, bowl_colour=None):
    text = PHRASES.get(verdict, verdict)
    if bowl_colour:
        text += f" {bowl_colour.capitalize()} bowl."
    if left is not None:
        text += " That was the last one." if left <= 0 else f" {n(left).capitalize()} left."
    elif counts and verdict in ("good", "defective"):
        text += f" {n(counts.get(verdict, 0)).capitalize()} so far."
    return text


def summary_line(counts, seconds=None):
    g, d = counts.get("good", 0), counts.get("defective", 0)
    total = g + d
    if not total:
        return "Nothing was sorted."
    took = ""
    if seconds:
        m, s = divmod(int(round(seconds)), 60)
        took = f" in {plural(m, 'minute')} {s} seconds" if m else f" in {s} seconds"
    every = "both" if total == 2 else f"all {n(total)}"
    split = (f"{every} good" if not d else f"{every} defective" if not g else f"{n(g)} good, {n(d)} defective")
    if total == 1:
        split = "it was good" if g else "it was defective"
    return f"Table clear. {plural(total, 'brick').capitalize()} sorted{took}: {split}."


# --- speaking -----------------------------------------------------------------------------
class _Piper:
    """One piper process reading lines on stdin, its raw audio piped to the speaker."""

    def __init__(self):
        cfg = VOICES[0] + ".json"
        rate = 22050
        try:
            with open(cfg) as f:
                rate = json.load(f)["audio"]["sample_rate"]
        except (OSError, KeyError, ValueError):
            pass
        play = {"paplay": f"paplay --raw --rate={rate} --channels=1 --format=s16le",
                "aplay": f"aplay -q -r {rate} -f S16_LE -t raw -c 1 -",
                "pw-play": f"pw-play --rate={rate} --channels=1 --format=s16 -"}[_PLAYER]
        self.proc = subprocess.Popen(
            f"'{PIPER}' --model '{VOICES[0]}' --output-raw 2>/dev/null | {play}",
            shell=True, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    def say(self, text):
        self.proc.stdin.write(text.replace("\n", " ") + "\n")
        self.proc.stdin.flush()

    def alive(self):
        return self.proc.poll() is None


_lock = threading.Lock()
_piper = None
_proc = None
_last = {"text": None, "t": 0.0}


def say(text, rate=160, force=False):
    """Speak text (if the voice is on). Repeats of the same line within 3 s are dropped."""
    global _piper, _proc
    if not text or (not force and not settings().get("on", True)):
        return
    now = time.time()
    if text == _last["text"] and now - _last["t"] < 3:
        return
    _last.update(text=text, t=now)
    if engine() == "ava":
        _ava_put(text)
        return
    _say_offline(text, rate)


def say_result(verdict, defect=None, color=None, extra="", fallback=None, force=False):
    """The result line in AccessInspect's wording ("Oops, there's some tape on that red brick. ...").

    verdict: good | defective. defect/color come from the Gemini look (may be None). extra is appended
    (e.g. "Three left."). fallback is SecondLook's own line, used by the offline engines.
    """
    if not force and not settings().get("on", True):
        return
    if engine() != "ava":
        say(fallback or brick_line(verdict), force=force)
        return
    _last.update(text=None, t=0.0)
    _aq.put(({"event": "pass" if verdict == "good" else "reject", "defect": defect, "color": color,
              "extra": extra, "fallback": fallback}, time.time()))
    _ava_start()


# --- Ava: one worker thread, lines in order, never overlapping, never blocking the caller ------
_aq = queue.Queue()
_ava = {"thread": None, "offline_until": 0.0, "proc": None}


def _ava_put(text):
    _aq.put((text, time.time()))
    _ava_start()


def _ava_start():
    with _lock:
        if _ava["thread"] is None or not _ava["thread"].is_alive():
            _ava["thread"] = threading.Thread(target=_ava_worker, daemon=True)
            _ava["thread"].start()


def _play(path):
    args = ([_FFPLAY, "-nodisp", "-autoexit", "-loglevel", "quiet", path] if "ffplay" in _FFPLAY
            else [_FFPLAY, "-q", path])
    p = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _ava["proc"] = p
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()


def _ava_worker():
    from . import assist
    while True:
        item, t = _aq.get()
        try:
            if time.time() - t > 15:                  # stale (queue backed up): skip rather than lag
                continue
            if isinstance(item, dict):                # result line: AccessInspect wording + extra
                text = None
                if time.time() > _ava["offline_until"]:
                    text = assist.phrase(item["event"], item.get("defect"), item.get("color"))
                if not text:
                    text = item.get("fallback") or ("Good brick." if item["event"] == "pass" else "Defect found.")
                    extra = ""
                else:
                    extra = item.get("extra") or ""
                lines = [text] + ([extra] if extra else [])
            else:
                lines = [item]
            for line in lines:
                path = assist.cached_voice(line)
                if not path and time.time() > _ava["offline_until"]:
                    # A new line takes 5-12 s to make; wait briefly, otherwise speak it offline now.
                    # The bridge keeps making it, so it is cached (instant) next time.
                    pend = assist.request("synth", text=line)
                    r = pend.get(1.5)
                    path = r.get("path") if r else None
                    if not path and pend.done():      # real failure (no internet / bridge): offline for 60 s
                        _ava["offline_until"] = time.time() + 60
                if path:
                    _play(path)
                else:
                    _say_offline(line)
        except Exception:
            pass                                      # audio must never stop the robot


def _say_offline(text, rate=160):
    global _piper, _proc
    try:
        with _lock:
            if _offline_engine() == "piper":
                if _piper is None or not _piper.alive():
                    _piper = _Piper()
                _piper.say(text)
                return
            if _ESPEAK is None:
                return
            if _proc and _proc.poll() is None:
                _proc.terminate()            # espeak: a new line interrupts the old one, never lags
            args = [_ESPEAK, text] if _ESPEAK == "spd-say" else [_ESPEAK, "-v", "en-gb-x-rp", "-s", str(rate), text]
            _proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass                                  # audio must never stop the robot


def access_api(method, path, body=None):
    """Shared by the run dashboard (web.py) and the control center: voice switch + speak-for-me.
    Returns (status, dict) or None if the path is not ours."""
    body = body or {}
    if path == "/api/voice":
        if method == "POST":
            if "engine" in body and body["engine"] in ("auto", "ava", "piper", "espeak"):
                set_engine(body["engine"])
            if "on" in body:
                set_voice(bool(body["on"]))
                if body["on"]:
                    say("Voice on.", force=True)
        return 200, dict(settings(), engine=engine())
    if path == "/api/say" and method == "POST":
        text = AAC.get(str(body.get("key")))
        if not text:
            return 400, {"error": "unknown phrase"}
        say(text, force=True)                 # the operator asked for it: speak even if results are muted
        return 200, {"ok": True, "text": text}
    if path == "/api/aac" and method == "GET":
        return 200, AAC
    return None


def announce_result(verdict, bowl=None, count=None):
    say(brick_line(verdict, {verdict: count} if count is not None else None))


if __name__ == "__main__":
    print("engine:", engine() or "none (install espeak-ng)", "| voice on:", settings().get("on"))
    say(describe_table([{"color": "blue"}, {"color": "blue"}, {"color": "red"}, {"color": "yellow"}]), force=True)
    time.sleep(5)
    say(brick_line("defective", left=3, bowl_colour="blue"), force=True)
    time.sleep(4)
    say(summary_line({"good": 3, "defective": 1}, 83), force=True)
    time.sleep(6)
