"""Turn raw camera frames into a clean Anomalib folder.

Crops the brick region, drops blurry frames and near-duplicates, writes PNGs plus a
manifest saying why each frame was kept or dropped.

    python tools/prepare_dataset.py --src runs/raw_wrist --out data/bricks/good --roi 100 0 440 480
    python tools/prepare_dataset.py --src a b c --out data/bricks/defect --min-sharpness 4

Only needs Pillow.
"""
import argparse
import csv
import os

from PIL import Image, ImageFilter, ImageStat

EXTS = (".jpg", ".jpeg", ".png", ".webp")


LAPLACIAN = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


def sharpness(img):
    """Variance of the Laplacian: drops sharply with motion blur and bad focus."""
    return ImageStat.Stat(img.convert("L").filter(LAPLACIAN)).var[0]


def ahash(img, size=8):
    g = img.convert("L").resize((size, size), Image.BILINEAR)
    px = list(g.get_flattened_data() if hasattr(g, "get_flattened_data") else g.getdata())
    mean = sum(px) / len(px)
    return sum(1 << i for i, p in enumerate(px) if p > mean)


def hamming(a, b):
    return bin(a ^ b).count("1")


def prepare(sources, out, roi=None, size=None, min_sharpness=0.0, dup_bits=4, skip=()):
    os.makedirs(out, exist_ok=True)
    files = []
    for src in sources:
        if os.path.isdir(src):
            files += [os.path.join(src, f) for f in sorted(os.listdir(src)) if f.lower().endswith(EXTS)]
        elif src.lower().endswith(EXTS):
            files.append(src)

    rows, hashes = [], []
    for path in files:
        name = os.path.basename(path)
        row = {"source": path, "kept": 0, "reason": "", "sharpness": ""}
        if name in skip:
            row["reason"] = "skip list"
            rows.append(row)
            continue
        img = Image.open(path).convert("RGB")
        if roi:
            x, y, w, h = roi
            img = img.crop((x, y, x + w, y + h))
        if size:
            img = img.resize(size, Image.BILINEAR)
        s = sharpness(img)
        row["sharpness"] = round(s, 2)
        if s < min_sharpness:
            row["reason"] = "blurry"
        else:
            h = ahash(img)
            if any(hamming(h, o) <= dup_bits for o in hashes):
                row["reason"] = "duplicate"
            else:
                hashes.append(h)
                img.save(os.path.join(out, os.path.splitext(name)[0] + ".png"))
                row["kept"], row["reason"] = 1, "ok"
        rows.append(row)

    with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["source", "kept", "reason", "sharpness"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True, help="image folders and/or files")
    p.add_argument("--out", required=True)
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    p.add_argument("--size", nargs=2, type=int, metavar=("W", "H"))
    p.add_argument("--min-sharpness", type=float, default=0.0,
                   help="drop frames below this; run once with 0 and read manifest.csv to choose")
    p.add_argument("--dup-bits", type=int, default=4, help="hash distance treated as duplicate (-1 = keep all)")
    p.add_argument("--skip", nargs="*", default=[], help="file names to drop by hand")
    a = p.parse_args()
    rows = prepare(a.src, a.out, a.roi, a.size, a.min_sharpness, a.dup_bits, set(a.skip))
    kept = sum(r["kept"] for r in rows)
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print(f"kept {kept}/{len(rows)} -> {a.out}  {reasons}")


if __name__ == "__main__":
    main()
