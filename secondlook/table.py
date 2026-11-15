"""Top-camera table scan: find bricks anywhere in the pick area and pick them without a learned policy.

How it works
  1. `./sl table` (once): photo of the empty table, then 6+ calibration points. For each point a
     brick lies on the table, the top camera finds it (pixel u, v) and you hold the open gripper
     around it ("pick") and a few cm above it ("above").
  2. A smooth map  pixel (u, v) -> joint angles  is fitted per joint (bilinear least squares).
  3. At run time the arm parks at `home`, the top camera is compared with the empty-table photo,
     every new blob inside the pick area is a brick, and the map gives the poses to pick it.

Accuracy is best inside the area the calibration points cover; spread them over the whole area.
"""
import os
import time

import numpy as np
import yaml

from .robot import ARM

MAP_FILE = "secondlook/table_map.yaml"
BACKGROUND_FILE = "secondlook/table_empty.png"


# --- detection --------------------------------------------------------------------------------
def _change(a_bgr, b_bgr, thresh):
    import cv2
    a = cv2.GaussianBlur(cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB).astype(np.int16), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB).astype(np.int16), (5, 5), 0)
    d = np.abs(a - b)
    return (d[..., 0] * 0.6 + d[..., 1] * 1.5 + d[..., 2] * 1.5) > thresh    # colour counts more than light


def fixed_objects(background, grow=25):
    """Mask of big coloured things already in the empty-table photo (the bowls), grown by `grow`
    pixels so a nudged bowl is never mistaken for a brick."""
    import cv2
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    col = ((hsv[..., 1] > 70) & (hsv[..., 2] > 60)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(col)
    out = np.zeros(col.shape, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 4000:
            out[labels == i] = 255
    return cv2.dilate(out, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0


def detect_bricks(bgr, background, region=None, min_area=600, max_area=40000, thresh=40,
                  previous=None, exclude=None):
    """Bricks = blobs that differ from the empty-table photo (and, if `previous` is given, also
    from that photo: only bricks that newly appeared). `region` is a polygon (list of [u, v]);
    `exclude` a boolean mask of places to ignore (bowls). Returns dicts {u, v, area, box},
    largest first."""
    import cv2
    if background is None:
        raise ValueError("no empty-table photo; run `./sl table` first")
    changed = _change(bgr, background, thresh)
    if previous is not None:
        changed &= _change(bgr, previous, thresh)
    if exclude is not None:
        changed &= ~exclude
    mask = changed.astype(np.uint8) * 255
    if region is not None and len(region) >= 3:
        roi = np.zeros_like(mask)
        cv2.fillPoly(roi, [np.asarray(region, np.int32)], 255)
        mask &= roi
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = (int(v) for v in stats[i, :4])
        out.append({"u": float(cents[i][0]), "v": float(cents[i][1]), "area": area,
                    "box": [x, y, w, h]})
    return sorted(out, key=lambda r: -r["area"])


def table_region(background, shrink=0.03):
    """Polygon around the white table top in the empty-table photo, so hands, people and dark
    surroundings at the edge of the view are never mistaken for bricks."""
    import cv2
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 150) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = np.column_stack(np.where(labels == biggest))[:, ::-1].astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    c = hull.mean(axis=0)
    return ((hull - c) * (1 - shrink) + c).round().astype(int).tolist()


def pick_area(points, grow=0.25):
    """Convex hull of the calibration points, grown outwards by `grow` (fraction)."""
    import cv2
    pts = np.asarray(points, np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    c = hull.mean(axis=0)
    return ((hull - c) * (1 + grow) + c).round().astype(int).tolist()


# --- pixel -> joints map ----------------------------------------------------------------------
def _features(u, v):
    u, v = np.asarray(u, float) / 640.0, np.asarray(v, float) / 480.0
    return np.stack([np.ones_like(u), u, v, u * v], axis=-1)        # bilinear


class PoseMap:
    """Per-joint bilinear fit from top-camera pixel (u, v) to joint angles, for `pick` and `above`."""

    def __init__(self, data):
        self.data = data            # {"points": [...], "coef": {"pick": {joint: [4]}, "above": {...}}, "region": [...]}

    @classmethod
    def fit(cls, points):
        """points: [{"u", "v", "pick": {joint: deg}, "above": {joint: deg}}, ...] (4 or more)."""
        if len(points) < 4:
            raise ValueError("need at least 4 calibration points (6+ recommended)")
        X = _features([p["u"] for p in points], [p["v"] for p in points])
        coef, err = {}, {}
        for kind in ("pick", "above"):
            coef[kind], err[kind] = {}, {}
            for j in ARM:
                y = np.array([p[kind][j] for p in points], float)
                w, *_ = np.linalg.lstsq(X, y, rcond=None)
                coef[kind][j] = [float(x) for x in w]
                err[kind][j] = float(np.abs(X @ w - y).max())
        data = {"points": points, "coef": coef, "fit_max_error_deg": err,
                "region": pick_area([[p["u"], p["v"]] for p in points])}
        return cls(data)

    def poses(self, u, v):
        x = _features(u, v)
        return {kind: {j: float(x @ np.array(self.data["coef"][kind][j])) for j in ARM}
                for kind in ("pick", "above")}

    @property
    def region(self):
        return self.data.get("region")

    def save(self, path=None):
        with open(path or MAP_FILE, "w") as f:
            yaml.safe_dump(self.data, f, sort_keys=False)

    @classmethod
    def load(cls, path=None):
        path = path or MAP_FILE
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; run `./sl table` to calibrate the table scan")
        with open(path) as f:
            return cls(yaml.safe_load(f))


# --- picker -----------------------------------------------------------------------------------
class TableScanPicker:
    """Looks at the table with the top camera and picks the biggest brick in the pick area.
    pick() returns "grasped", "missed" or "empty" (nothing left on the table)."""

    def __init__(self, cfg, camera="top"):
        import cv2
        self.cam = camera
        if camera not in cfg["cameras"]:
            raise KeyError(f"table scan needs the '{camera}' camera in config.yaml")
        self.map = PoseMap.load()
        if not os.path.exists(BACKGROUND_FILE):
            raise FileNotFoundError(f"{BACKGROUND_FILE} missing; run `./sl table`")
        self.background = cv2.imread(BACKGROUND_FILE)
        self.table = table_region(self.background)
        self.bowls = fixed_objects(self.background)
        areas = [p["area"] for p in self.map.data["points"] if "area" in p]
        self.max_area = int(3 * max(areas)) if areas else 8000
        self.last = None            # poses of the last pick, for the second look
        self.found = 0

    def _on_table(self, b):
        import cv2
        if not self.table:
            return True
        return cv2.pointPolygonTest(np.asarray(self.table, np.float32), (b["u"], b["v"]), False) >= 0

    def scan(self, robot):
        robot.go("home")            # park out of the top camera's view of the pick area
        time.sleep(0.4)
        frame = robot.frame(self.cam)
        # A Duplo brick is ~600-4000 px from the top camera; bowls (~15000 px) and people are bigger.
        bricks = detect_bricks(frame, self.background, self.map.region, exclude=self.bowls, max_area=self.max_area)
        bricks = [b for b in bricks if self._on_table(b)]
        self.found = len(bricks)
        return frame, bricks

    def pick(self, robot, ui):
        _, bricks = self.scan(robot)
        if not bricks:
            ui.show(stage="TABLE CLEAR")
            return "empty"
        b = bricks[0]
        ui.show(stage=f"PICK  brick at ({b['u']:.0f}, {b['v']:.0f}), {len(bricks)} on the table")
        p = self.map.poses(b["u"], b["v"])
        self.last = p
        robot.gripper(open=True)
        robot.move(p["above"])
        robot.move(p["pick"])
        robot.gripper(open=False)
        time.sleep(0.3)
        robot.move(p["above"])
        return "grasped" if robot.holding() else "missed"

    def put_back(self, robot):
        """Second look: put the brick back where it came from (it is found again by the scan)."""
        if self.last is None:
            robot.go("putback")
            robot.gripper(open=True)
            return
        robot.move(self.last["above"])
        robot.move(self.last["pick"])
        robot.gripper(open=True)
        robot.move(self.last["above"])
