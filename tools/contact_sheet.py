"""Tile images into numbered contact sheets so a whole folder can be reviewed at a glance.

    python tools/contact_sheet.py <image_dir> <out_prefix> [--cols 10] [--thumb 128]

Writes <out_prefix>_1.jpg, _2.jpg, ... (100 images per sheet). Each tile is labelled
with its index in the sorted file list, so you can say "keep 12-30" afterwards.
"""
import argparse
import os

from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("out")
p.add_argument("--cols", type=int, default=10)
p.add_argument("--thumb", type=int, default=128)
a = p.parse_args()

files = sorted(f for f in os.listdir(a.src) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
per = a.cols * 10
tw, th = a.thumb, a.thumb * 3 // 4
for s in range(0, len(files), per):
    chunk = files[s:s + per]
    rows = (len(chunk) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (a.cols * tw, rows * th), "black")
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(chunk):
        im = Image.open(os.path.join(a.src, f)).convert("RGB").resize((tw, th))
        x, y = (i % a.cols) * tw, (i // a.cols) * th
        sheet.paste(im, (x, y))
        d.rectangle([x, y, x + 26, y + 12], fill="black")
        d.text((x + 2, y), str(s + i), fill="yellow")
    out = f"{a.out}_{s // per + 1}.jpg"
    sheet.save(out, quality=85)
    print(out, len(chunk))
