"""One-window demo display for the inspection cell (OpenCV only).

Readable from ~2 m at a peer-review table: inspection views with heatmap,
big verdict, score bar with threshold, latency/device/power lines, counters.

Usage from the loop:
    from hud import Hud
    hud = Hud(fullscreen=True)
    key = hud.show(views=[(img1, heat1, s1), (img2, heat2, s2)],
                   state={"stage": "INSPECT", "verdict": "defective", "score": 0.71,
                          "thr": 0.5, "band": 0.08, "counts": {"good": 4, "defective": 2},
                          "cycle_s": 14.2, "policy_ms": 38, "policy_dev": "GPU",
                          "anomaly_ms": 6, "anomaly_dev": "NPU", "watts": 9.4, "temp_c": 61},
                   top=top_frame)
    if key in (ord("s"), 27): stop_robot()

Run `python hud.py` to see it with synthetic data.
"""
import cv2
import numpy as np

W, H = 1280, 720
GREEN, RED, AMBER = (60, 180, 75), (40, 40, 220), (0, 190, 255)
WHITE, GREY, BLACK = (245, 245, 245), (150, 150, 150), (18, 18, 18)
VERDICT_COLOR = {"good": GREEN, "defective": RED, "uncertain": AMBER}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def overlay_heatmap(img, heat, vmax=None, alpha=0.45):
    """Blend an anomaly map onto a BGR image.

    vmax fixes the colour scale (use ~2x threshold) so good bricks stay blue.
    With vmax=None every image is stretched to full range, which makes good
    bricks look hot too.
    """
    if heat is None:
        return img
    h = cv2.resize(np.squeeze(np.asarray(heat, np.float32)), (img.shape[1], img.shape[0]))  # model gives (1,1,H,W)
    h = h / vmax if vmax else (h - h.min()) / (np.ptp(h) + 1e-8)
    h = (np.clip(h, 0, 1) * 255).astype(np.uint8)
    return cv2.addWeighted(img, 1 - alpha, cv2.applyColorMap(h, cv2.COLORMAP_JET), alpha, 0)


def _fit(img, w, h):
    """Letterbox img into a w x h black box."""
    out = np.full((h, w, 3), BLACK, np.uint8)
    if img is None:
        return out
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def _text(canvas, txt, xy, scale=0.8, color=WHITE, thick=2):
    cv2.putText(canvas, str(txt), xy, FONT, scale, color, thick, cv2.LINE_AA)


def render(views, state, top=None):
    """Build the 1280x720 frame. views = [(bgr_img, heatmap_or_None, score), ...]."""
    c = np.full((H, W, 3), BLACK, np.uint8)
    thr, band = state.get("thr", 0.5), state.get("band", 0.0)
    vmax = state.get("vmax", 2 * thr if thr else None)

    # Left: up to two inspection views side by side, top camera below.
    n = max(1, min(2, len(views)))
    vw, vh = 760 // n, 400
    for i, (img, heat, score) in enumerate(views[:2]):
        x = 10 + i * vw
        c[10:10 + vh, x:x + vw - 10] = _fit(overlay_heatmap(img, heat, vmax), vw - 10, vh)
        col = VERDICT_COLOR["defective"] if score >= thr else VERDICT_COLOR["good"]
        _text(c, f"view {i + 1}/{len(views)}  score {score:.2f}", (x + 8, vh + 2), 0.7, col)
    if top is not None:
        c[430:710, 10:380] = _fit(top, 370, 280)
        _text(c, "top camera", (18, 700), 0.6, GREY, 1)

    # Right panel.
    x0 = 790
    verdict = state.get("verdict")
    if verdict:
        col = VERDICT_COLOR.get(verdict, WHITE)
        cv2.rectangle(c, (x0, 10), (W - 10, 120), col, -1)
        _text(c, verdict.upper(), (x0 + 20, 90), 2.0, BLACK, 5)
    _text(c, state.get("stage", ""), (x0, 160), 1.0, WHITE, 2)

    # Score bar: threshold line + uncertain band.
    score = state.get("score")
    if score is not None:
        bx, by, bw, bh = x0, 185, W - 10 - x0, 26
        smax = vmax or max(1.0, score)
        px = lambda v: bx + int(bw * min(max(v / smax, 0), 1))
        cv2.rectangle(c, (px(thr - band), by), (px(thr + band), by + bh), (40, 70, 90), -1)
        cv2.rectangle(c, (bx, by), (px(score), by + bh),
                      VERDICT_COLOR["defective"] if score >= thr else VERDICT_COLOR["good"], -1)
        cv2.line(c, (px(thr), by - 6), (px(thr), by + bh + 6), WHITE, 2)
        cv2.rectangle(c, (bx, by), (bx + bw, by + bh), GREY, 1)
        _text(c, f"anomaly score {score:.2f}  (threshold {thr:.2f})", (bx, by + bh + 26), 0.6, GREY, 1)

    lines = [
        ("policy", state.get("policy_ms"), state.get("policy_dev"), "ms"),
        ("anomaly", state.get("anomaly_ms"), state.get("anomaly_dev"), "ms"),
    ]
    y = 290
    for name, val, dev, unit in lines:
        if val is not None:
            _text(c, f"{name:8s} {val:6.0f} {unit}  on {dev or '?'}", (x0, y), 0.8)
            y += 40
    for label, key, fmt in [("cycle", "cycle_s", "{:.1f} s / brick"),
                            ("power", "watts", "+{:.1f} W over idle"),
                            ("temp", "temp_c", "{:.0f} C")]:
        if state.get(key) is not None:
            _text(c, f"{label:8s} " + fmt.format(state[key]), (x0, y), 0.8)
            y += 40

    counts = state.get("counts", {})
    _text(c, f"GOOD {counts.get('good', 0)}", (x0, 640), 1.3, GREEN, 3)
    _text(c, f"DEFECTIVE {counts.get('defective', 0)}", (x0 + 200, 640), 1.3, RED, 3)
    _text(c, "all inference on-device  |  s / Esc = STOP", (x0, 700), 0.55, GREY, 1)
    return c


class Hud:
    def __init__(self, name="SecondLook", fullscreen=False):
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, views, state, top=None, wait_ms=1):
        """Draw and return the pressed key (-1 if none). Call every loop tick."""
        cv2.imshow(self.name, render(views, state, top))
        return cv2.waitKey(wait_ms) & 0xFF if wait_ms else -1

    def close(self):
        cv2.destroyWindow(self.name)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = np.full((480, 640, 3), (60, 140, 220), np.uint8)
    cv2.rectangle(img, (220, 160), (420, 320), (40, 40, 200), -1)
    heat = rng.random((64, 64)) * 0.2
    heat[25:35, 30:42] = 0.9
    state = {"stage": "INSPECT  view 2/2", "verdict": "defective", "score": 0.9, "thr": 0.5,
             "band": 0.08, "counts": {"good": 7, "defective": 3}, "cycle_s": 13.8,
             "policy_ms": 41, "policy_dev": "GPU", "anomaly_ms": 6, "anomaly_dev": "NPU",
             "watts": 9.2, "temp_c": 62}
    frame = render([(img, heat * 0.3, 0.12), (img, heat, 0.9)], state, top=img)
    cv2.imwrite("hud_preview.png", frame)
    print("wrote hud_preview.png")
