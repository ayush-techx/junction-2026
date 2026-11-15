"""AccessInspect voice announcements (Microsoft Ava via edge-tts).

CLI:
    python speak_alert.py start
    python speak_alert.py pass   [--color red]
    python speak_alert.py reject [tape|sticker|marker] [--color red]
    python speak_alert.py complete PASSED REJECTED      # e.g. complete 1 1
    python speak_alert.py --pregen                      # cache phrases for offline use

From Python (returns the text immediately so the app can show it on screen;
speech plays in the background so the robot loop never pauses):
    from speak_alert import announce, wait_for_speech
    msg = announce("reject", defect="tape", color="red")
    wait_for_speech()   # optional: block until everything queued has been spoken

Requires: pip install edge-tts
Linux playback needs mpg123 (sudo apt install mpg123) or ffplay.
"""
import asyncio
import atexit
import hashlib
import queue
import random
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import edge_tts

VOICE = "en-US-AvaMultilingualNeural"
CACHE = Path(__file__).parent / "voice_cache"

# {brick} becomes "brick" or e.g. "red brick" when a color is detected.
PHRASES = {
    "start": [
        "Inspection started.",
        "Okay, starting the inspection now.",
        "Alright, inspection is running.",
    ],
    "pass": [
        "No defect detected. Brick placed in the good bowl.",
        "Looks good! This {brick} goes in the good bowl.",
        "All clear. I've placed the {brick} in the good bowl.",
    ],
    "reject": [
        "Defect detected. Brick placed in the reject bowl.",
        "Hmm, this {brick} has a defect. Moving it to the reject bowl.",
        "Heads up, defect spotted. I've placed the {brick} in the reject bowl.",
    ],
    "reject_typed": [
        "Defect detected: {defect}. Brick placed in the reject bowl.",
        "Hey, I found {defect} on this {brick}. I've placed it in the reject bowl.",
        "Oops, there's {defect} on that {brick}. Moving it to the reject bowl.",
    ],
}

# Defect name from the detector -> how it is spoken in a sentence.
DEFECTS = {
    "tape": "some tape",
    "sticker": "a sticker",
    "marker": "marker ink",
}


def _n(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def complete_text(passed: int, rejected: int) -> str:
    total = passed + rejected
    return (
        f"Inspection complete. {_n(total, 'brick')} checked. "
        f"{passed} passed and {rejected} {'was' if rejected == 1 else 'were'} rejected."
    )


def make_text(event: str, defect: str | None = None, color: str | None = None,
              passed: int = 0, rejected: int = 0) -> str:
    if event == "complete":
        return complete_text(passed, rejected)
    key = "reject_typed" if event == "reject" and defect else event
    templates = PHRASES[key]
    if color:  # only templates that can mention the brick's color
        templates = [t for t in templates if "{brick}" in t] or templates
    brick = f"{color} brick" if color else "brick"
    spoken_defect = DEFECTS.get((defect or "").lower(), defect)
    return random.choice(templates).format(brick=brick, defect=spoken_defect)


async def synth(text: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / (hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:16] + ".mp3")
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(".part")
    for attempt in range(3):  # the TTS service occasionally drops a request
        try:
            await edge_tts.Communicate(text, VOICE).save(str(tmp))
            tmp.replace(path)  # only a complete file ever lands in the cache
            return path
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            await asyncio.sleep(1 + attempt)
    return path


def play(path: Path) -> None:
    if sys.platform == "win32":
        ps = (
            "Add-Type -AssemblyName PresentationCore;"
            "$p=New-Object System.Windows.Media.MediaPlayer;"
            f"$p.Open([Uri]'{path}');"
            "while(-not $p.NaturalDuration.HasTimeSpan){Start-Sleep -Milliseconds 50};"
            "$p.Play();Start-Sleep -Milliseconds ([int]$p.NaturalDuration.TimeSpan.TotalMilliseconds+200);$p.Close()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)
    elif shutil.which("mpg123"):
        subprocess.run(["mpg123", "-q", str(path)], check=False)
    elif shutil.which("ffplay"):
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)], check=False)
    else:
        sys.exit("No audio player found: install mpg123 (sudo apt install mpg123).")


_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None


def _speak_worker() -> None:
    while True:
        text = _queue.get()
        try:
            play(asyncio.run(synth(text)))
        except Exception as e:  # never crash the robot loop over audio
            print(f"[speech error] {e}", file=sys.stderr)
        finally:
            _queue.task_done()


def wait_for_speech() -> None:
    """Block until every queued announcement has finished playing."""
    _queue.join()


def announce(event: str, wait: bool = False, **kwargs) -> str:
    """Queue an announcement and return its text immediately for on-screen display.

    Speech plays in a background thread, in order, without overlapping, so the
    robot control loop keeps running. Pass wait=True to block until it is spoken.
    """
    global _worker
    text = make_text(event, **kwargs)
    print(text, flush=True)
    if _worker is None:
        _worker = threading.Thread(target=_speak_worker, daemon=True)
        _worker.start()
        atexit.register(wait_for_speech)  # finish speaking before the program exits
    _queue.put(text)
    if wait:
        wait_for_speech()
    return text


async def pregen() -> None:
    texts = PHRASES["start"] + PHRASES["pass"] + PHRASES["reject"]
    texts = [t.format(brick="brick") for t in texts]
    for d in DEFECTS.values():
        texts += [t.format(brick="brick", defect=d) for t in PHRASES["reject_typed"]]
    texts += [complete_text(p, r) for total in range(1, 11) for p in range(total + 1)
              for r in [total - p]]
    failed = []
    for t in texts:
        try:
            await synth(t)
        except Exception as e:
            failed.append(t)
            print(f"FAILED: {t} ({e})")
    print(f"Cached {len(texts) - len(failed)}/{len(texts)} phrases in {CACHE}")
    if failed:
        sys.exit("Some phrases failed; run --pregen again to retry them.")


def main() -> None:
    args = sys.argv[1:]
    if args == ["--pregen"]:
        asyncio.run(pregen())
        return
    color = None
    if "--color" in args:
        i = args.index("--color")
        color = args[i + 1] if i + 1 < len(args) else None
        args = args[:i] + args[i + 2:]
    if not args or args[0] not in ("start", "pass", "reject", "complete"):
        sys.exit(__doc__)
    event = args[0]
    if event == "complete":
        if len(args) != 3:
            sys.exit("Usage: python speak_alert.py complete PASSED REJECTED")
        announce(event, wait=True, passed=int(args[1]), rejected=int(args[2]))
    else:
        announce(event, wait=True, defect=" ".join(args[1:]) or None, color=color)


if __name__ == "__main__":
    main()
