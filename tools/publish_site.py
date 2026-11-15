"""Build the public dashboard for Netlify: the page + the last data from the station.

    ./sl publish            build site/ (nothing leaves the laptop)
    ./sl publish --deploy   build, then upload site/ to Netlify (netlify-cli; `npx netlify-cli login` once)

The data comes from the public viewer (secondlook/viewer.py, http://127.0.0.1:8772), so the snapshot has
exactly what the live public page shows: no paths, no scene camera images, wrist crops only.
The page calls the live station at `public.api_url` (config.yaml) and falls back to this snapshot when the
laptop is offline. Run it once before closing the laptop so the snapshot is the latest data.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from secondlook.viewer import page_config  # noqa: E402

STATIC = ROOT / "secondlook" / "static"
# what the page asks for -> snapshot file (the page maps a URL to last/<path>.json, ignoring the query)
ENDPOINTS = ["/api/status", "/api/live", "/api/scene", "/api/results", "/api/inspections",
             "/api/history?limit=200&picked=1", "/api/history/runs", "/api/picks?limit=300", "/api/learning",
             "/api/models", "/api/voice", "/api/trends", "/api/aac"]
HEADERS = """/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  X-Frame-Options: DENY
/
  Cache-Control: no-cache
/config.js
  Cache-Control: no-cache
/last/*
  Cache-Control: no-cache
"""


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as r:
        return r.status, r.read()


def build(out: Path, viewer: str, api_url: str | None):
    tmp = out.with_name(out.name + ".new")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "last" / "api" / "history").mkdir(parents=True)
    (tmp / "last" / "img").mkdir(parents=True)
    (tmp / "last" / "live").mkdir(parents=True)
    shutil.copy(STATIC / "center.html", tmp / "index.html")
    shutil.copy(STATIC / "access.js", tmp / "access.js")
    built = time.strftime("%a %H:%M")
    (tmp / "config.js").write_bytes(page_config(api=api_url, snapshot="last/", built=built))
    (tmp / "_headers").write_text(HEADERS)

    history = []
    for ep in ENDPOINTS:
        status, body = get(viewer, ep)
        if status != 200:
            raise SystemExit(f"{ep}: HTTP {status} from the viewer")
        path = urllib.parse.urlsplit(ep).path
        (tmp / "last" / (path.lstrip("/") + ".json")).write_bytes(body)
        if path == "/api/history":
            history = json.loads(body)
    photos = sorted({p for r in history for p in (r.get("photos") or [])})
    for name in photos:
        try:
            status, body = get(viewer, "/img/" + urllib.parse.quote(name))
            if status == 200:
                (tmp / "last" / "img" / name).write_bytes(body)
        except Exception as e:
            print(f"  skipped photo {name}: {e}")
    try:
        status, body = get(viewer, "/live/frame.jpg")
        if status == 200 and body:
            (tmp / "last" / "live" / "frame.jpg").write_bytes(body)
    except Exception:
        pass
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"built {out}/:{len(history)} bricks, {len(photos)} photos, {size:.1f} MB, "
          f"snapshot {built}, live API {api_url or 'none (snapshot only)'}")


def main():
    import yaml
    cfg = (yaml.safe_load((ROOT / "secondlook" / "config.yaml").read_text()) or {}).get("public") or {}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", action="store_true", help="upload site/ to Netlify after building")
    ap.add_argument("--viewer", default=f"http://127.0.0.1:{cfg.get('viewer_port', 8772)}")
    ap.add_argument("--api", default=cfg.get("api_url"), help="public URL of the viewer (Tailscale Funnel)")
    ap.add_argument("--site", default=cfg.get("netlify_site"), help="Netlify site name or id")
    ap.add_argument("--out", default=str(ROOT / "site"))
    a = ap.parse_args()
    try:
        get(a.viewer, "/api/voice")
    except Exception:
        raise SystemExit(f"The public viewer is not running at {a.viewer}. Start it: ./sl viewer "
                         "(or systemctl --user start secondlook-viewer)")
    api = a.api.rstrip("/") if a.api else None
    build(Path(a.out), a.viewer, api)
    if not a.deploy:
        print("Not uploaded. To publish: ./sl publish --deploy  (or drag the site/ folder onto app.netlify.com/drop)")
        return
    cmd = ["npx", "--yes", "netlify-cli", "deploy", "--prod", "--dir", a.out]
    if a.site:
        cmd += ["--site", a.site]
    print("uploading to Netlify:", " ".join(cmd))
    sys.exit(subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, NETLIFY_TELEMETRY_DISABLED="1")))


if __name__ == "__main__":
    main()
