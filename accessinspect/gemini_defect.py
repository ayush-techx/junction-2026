"""AccessInspect defect check with Gemini vision.

Looks at a photo of one brick and decides: none | tape | sticker | marker.

CLI:
    python gemini_defect.py brick.jpg            # print the result as JSON
    python gemini_defect.py brick.jpg --speak    # also announce it with speak_alert

From Python:
    from gemini_defect import check_brick
    r = check_brick("brick.jpg")            # or JPEG/PNG bytes
    if r.defective: ...                     # r.defect, r.color, r.confident, r.reason

Setup:
    pip install google-genai
    Put GEMINI_API_KEY=... in a .env file next to this script (git-ignored),
    or set it as an environment variable. Free key: https://aistudio.google.com/apikey
    Optional: GEMINI_MODEL=<model id> to try that model first.

Results are cached by image content in defect_cache.json, so re-checking a demo
photo is instant and works without internet.
"""
import hashlib
import json
import os
import sys
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

HERE = Path(__file__).parent
CACHE_FILE = HERE / "defect_cache.json"
# Tried in order; the next one is used if a model is overloaded (503), rate-limited
# (429) or missing (404). Pinned ids (not "-latest") so nothing changes before the demo.
# Chosen from a live test on 2026-11-15: 3.7/3.8-flash timed out under load, 2.5-flash
# is retired for new keys; these four answered in 1-6 s.
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
TIMEOUT_MS = 10_000  # per model; a slow model is skipped instead of freezing the robot

PROMPT = """You are the quality inspector for a toy building-brick sorting robot.
The photo shows one brick. A brick is DEFECTIVE if anything foreign was added to it:
- "tape": any piece of tape stuck on the brick
- "sticker": any sticker or label stuck on the brick
- "marker": any marker / pen ink drawn on the brick
Otherwise the defect is "none". Molded logos, studs, shadows, reflections and the
background are NOT defects. If several defects are present, report the most visible one.
Also give the brick's main color in one plain word (e.g. red, blue, yellow), or
"unknown" if you cannot tell. Set confident=false if the photo is blurry, the brick
is partly hidden, or you are unsure."""


class Defect(str, Enum):
    none = "none"
    tape = "tape"
    sticker = "sticker"
    marker = "marker"


class BrickResult(BaseModel):
    defect: Defect
    color: str = Field(description="Main brick color in one word, or 'unknown'")
    confident: bool
    reason: str = Field(description="One short sentence on what you saw")

    @property
    def defective(self) -> bool:
        return self.defect != Defect.none


def _load_env() -> None:
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _read_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def check_brick(image: str | Path | bytes, use_cache: bool = True) -> BrickResult:
    """Classify one brick photo. Raises on API/network errors so the caller can fall back."""
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    key = hashlib.sha256(data).hexdigest()

    cache = _read_cache() if use_cache else {}
    if key in cache:
        return BrickResult.model_validate(cache[key])

    _load_env()
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY found. Add it to .env next to this script.")

    from google.genai import errors

    # attempts=1: we do our own fallback across models instead of the SDK retrying one model.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)))
    models = FALLBACK_MODELS
    if os.environ.get("GEMINI_MODEL"):
        models = [os.environ["GEMINI_MODEL"]] + [m for m in models if m != os.environ["GEMINI_MODEL"]]
    last_err = None
    for model in models:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=data, mime_type=_mime(data)), PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BrickResult,
                    temperature=0,
                ),
            )
            break
        except errors.APIError as e:
            if e.code not in (404, 429, 500, 503, 504):
                raise  # e.g. bad key: falling back won't help
            last_err = e
            print(f"[gemini] {model} unavailable ({e.code}), trying next model", file=sys.stderr)
        except Exception as e:  # timeouts / dropped connections
            last_err = e
            print(f"[gemini] {model} failed ({type(e).__name__}), trying next model", file=sys.stderr)
    else:
        raise last_err
    result = resp.parsed if isinstance(resp.parsed, BrickResult) else BrickResult.model_validate_json(resp.text)

    if use_cache:
        cache = _read_cache()
        cache[key] = result.model_dump(mode="json")
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    return result


def main() -> None:
    args = sys.argv[1:]
    speak = "--speak" in args
    args = [a for a in args if a != "--speak"]
    if len(args) != 1:
        sys.exit(__doc__)
    r = check_brick(args[0])
    print(json.dumps(r.model_dump(mode="json"), indent=2))
    if speak:
        from speak_alert import announce
        color = r.color if r.color.lower() != "unknown" else None
        if r.defective:
            announce("reject", wait=True, defect=r.defect.value, color=color)
        else:
            announce("pass", wait=True, color=color)


if __name__ == "__main__":
    main()
