"""Bridge so SecondLook (which runs in the Studio Python) can use AccessInspect's voice and Gemini check
without installing edge-tts / google-genai into the Studio environment.

One long-running process, JSON lines on stdin -> JSON lines on stdout:
    {"op": "synth", "id": 1, "text": "Good brick."}                      -> {"id": 1, "path": ".../voice_cache/xx.mp3"}
    {"op": "text",  "id": 2, "event": "reject", "defect": "tape", "color": "red"}   -> {"id": 2, "text": "..."}
    {"op": "check", "id": 3, "path": "/abs/brick.png"}                    -> {"id": 3, "result": {...}}
Errors come back as {"id": n, "error": "..."}. Every request runs in its own thread, so a slow Gemini call
never holds up speech.

Started by secondlook/assist.py as:  accessinspect/.venv/bin/python accessinspect/sl_bridge.py
"""
import asyncio
import json
import sys
import threading

import gemini_defect
import speak_alert

_out = threading.Lock()


def reply(msg):
    with _out:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def handle(req):
    rid = req.get("id")
    try:
        op = req.get("op")
        if op == "synth":
            reply({"id": rid, "path": str(asyncio.run(speak_alert.synth(req["text"])))})
        elif op == "text":
            reply({"id": rid, "text": speak_alert.make_text(
                req["event"], defect=req.get("defect"), color=req.get("color"),
                passed=req.get("passed", 0), rejected=req.get("rejected", 0))})
        elif op == "check":
            r = gemini_defect.check_brick(req["path"])
            reply({"id": rid, "result": r.model_dump(mode="json")})
        elif op == "ping":
            reply({"id": rid, "ok": True})
        else:
            reply({"id": rid, "error": f"unknown op {op!r}"})
    except BaseException as e:  # noqa: BLE001 - report every failure to the caller, never die
        reply({"id": rid, "error": f"{type(e).__name__}: {e}"[:300]})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        threading.Thread(target=handle, args=(req,), daemon=True).start()


if __name__ == "__main__":
    main()
