"""Zip the project for another machine: code, docs, tests, config. No data, models, logs,
photos, environments or git history.

    python3 tools/make_bundle.py ../secondlook.zip
"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", ".claude", "__pycache__", "data", "runs", "exports", "anomalib-env", ".venv"}
SKIP_PREFIX = ("junction2026_", "IMG")
SKIP_EXT = (".zip", ".pyc", ".jpg", ".jpeg", ".webp", ".png", ".mp4", ".ckpt", ".pt")
SKIP_FILES = {"poses.yaml"}
EXECUTABLE = {"sl", "setup.sh"}


def files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(SKIP_PREFIX))
        for f in sorted(filenames):
            if f in SKIP_FILES or f.startswith(SKIP_PREFIX) or f.lower().endswith(SKIP_EXT):
                continue
            yield os.path.join(dirpath, f)


def main(out):
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files():
            rel = os.path.relpath(path, ROOT)
            info = zipfile.ZipInfo(os.path.join("secondlook", rel), date_time=(2026, 11, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if rel in EXECUTABLE else 0o644) << 16
            with open(path, "rb") as fh:
                z.writestr(info, fh.read())
            n += 1
    print(f"{out}: {n} files, {os.path.getsize(out) / 1024:.0f} KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(ROOT), "secondlook.zip"))
