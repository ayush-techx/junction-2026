"""Score good/defect images with the exported model and choose the threshold.

    python tools/evaluate_anomaly.py --model exports/padim/int8/weights/openvino/model.xml \
        --root data/bricks --device NPU --overlays

Prints score ranges, the suggested threshold/band (paste into config.yaml), recall and
false-reject rate, and writes runs/eval_anomaly/scores.csv. --overlays saves heatmap images
for the README. Images must already be cropped like the runtime crop (prepare_dataset / collect).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secondlook.calibrate import suggest  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".webp")


def main():
    import cv2
    from secondlook.inspection import Inspector

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--root", default="data/bricks")
    p.add_argument("--device", default="CPU")
    p.add_argument("--out", default="runs/eval_anomaly")
    p.add_argument("--overlays", action="store_true")
    a = p.parse_args()

    # Images are pre-cropped, so no ROI here.
    inspector = Inspector({"inspection": {"model": a.model, "device": a.device, "roi": None}})
    os.makedirs(a.out, exist_ok=True)
    scores, rows, times = {"good": [], "defect": []}, [], []
    for label in ("good", "defect"):
        folder = os.path.join(a.root, label)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith(EXTS):
                continue
            img = cv2.imread(os.path.join(folder, f))
            s, heat, crop = inspector.score(img)
            times.append(inspector.last_ms)
            scores[label].append(s)
            rows.append({"label": label, "file": f, "score": round(s, 4)})
            if a.overlays and heat is not None:
                from secondlook.hud import overlay_heatmap
                cv2.imwrite(os.path.join(a.out, f"{label}_{os.path.splitext(f)[0]}.jpg"),
                            overlay_heatmap(crop, heat, vmax=1.0))

    with open(os.path.join(a.out, "scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, ["label", "file", "score"])
        w.writeheader()
        w.writerows(rows)

    g, d = scores["good"], scores["defect"]
    print(f"good:   n={len(g)}  min={min(g):.3f}  max={max(g):.3f}" if g else "good: none")
    print(f"defect: n={len(d)}  min={min(d):.3f}  max={max(d):.3f}" if d else "defect: none")
    thr, band, info = suggest(g, d)
    print(f"suggested threshold {thr:.3f}  band {band:.3f}  ({info['mode']})")
    if d:
        print(f"at that threshold: defects caught {info['recall']:.0%}, good rejected {info['false_reject']:.0%}")
    times = sorted(t for t in times if t is not None)
    if times:
        print(f"latency on {a.device}: median {times[len(times) // 2]:.1f} ms over {len(times)} images")
    print("scores ->", os.path.join(a.out, "scores.csv"))


if __name__ == "__main__":
    main()
