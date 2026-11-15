"""prepare_dataset: crop, blur filter, duplicate filter, manifest."""
import csv
import os
import sys
import tempfile
import unittest

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    Image = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def brick(path, color, offset=0, blur=0):
    img = Image.new("RGB", (640, 480), (235, 235, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([200 + offset, 150, 420 + offset, 330], fill=color)
    for i in range(4):
        d.ellipse([220 + offset + i * 50, 170, 250 + offset + i * 50, 200], outline=(40, 40, 40), width=3)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    img.save(path)


@unittest.skipIf(Image is None, "Pillow not installed")
class TestPrepare(unittest.TestCase):
    def test_filters_and_manifest(self):
        from prepare_dataset import prepare
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "raw"), os.path.join(d, "good")
            os.makedirs(src)
            brick(os.path.join(src, "a.jpg"), (230, 200, 60))
            brick(os.path.join(src, "b.jpg"), (230, 200, 60))             # duplicate of a
            brick(os.path.join(src, "c.jpg"), (60, 140, 220), offset=60)
            brick(os.path.join(src, "d.jpg"), (120, 200, 60), blur=8)     # blurry
            brick(os.path.join(src, "e.webp"), (220, 60, 60), offset=-60)
            rows = prepare([src], out, roi=(100, 50, 440, 380), min_sharpness=120.0, skip={"e.webp"})
            by_name = {os.path.basename(r["source"]): r["reason"] for r in rows}
            self.assertEqual(by_name, {"a.jpg": "ok", "b.jpg": "duplicate", "c.jpg": "ok",
                                       "d.jpg": "blurry", "e.webp": "skip list"})
            kept = sorted(f for f in os.listdir(out) if f.endswith(".png"))
            self.assertEqual(kept, ["a.png", "c.png"])
            self.assertEqual(Image.open(os.path.join(out, "a.png")).size, (440, 380))
            with open(os.path.join(out, "manifest.csv")) as f:
                self.assertEqual(len(list(csv.DictReader(f))), 5)


if __name__ == "__main__":
    unittest.main()
