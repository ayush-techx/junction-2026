"""Top-camera scene understanding with colour + depth (RealSense D415).

  height map   depth -> 3D points -> table plane (RANSAC) -> height of every pixel above the table
  bricks       small blobs 6-60 mm tall, brick-sized in millimetres (not pixels), with orientation
  bowls        big coloured discs (pink / blue), found live every scan, so a moved bowl is no problem
  obstacles    anything else on the table that is too big or too tall to be a brick (a box, a hand)

No empty-table photo is needed, so the scan keeps working when the scene drifts. Without depth it
falls back to colour only (bricks = coloured or dark blobs on the white table) and a stored plane.
Pure numpy + OpenCV; no hardware imports (capture() opens only the camera, never the arm).
"""
import json
import os
import time

import numpy as np

OUT_DIR = os.path.join("runs", "scene")

DEFAULTS = {
    "min_height_mm": 6.0,        # lower = table noise / paper edges
    "max_height_mm": 60.0,       # higher = arm, people, boxes
    "min_area_mm2": 300.0,       # Duplo 2x2 top is ~1000 mm2, 2x4 ~2000 mm2
    "max_area_mm2": 2800.0,
    "max_length_mm": 80.0,       # 2x4 Duplo is 64 mm long
    "max_width_mm": 52.0,        # allow depth/projection edge noise on 32 mm Duplo bricks
    "bowl_min_area_px": 5000,
    "bowl_margin_px": 18,
    "grasp_height_m": 0.012,     # tool point height above the table when grasping (same as calibration)
}


# --- geometry -----------------------------------------------------------------------------
def K_dict(fx, fy, cx, cy, w=640, h=480):
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy), "w": int(w), "h": int(h)}


def rs_intrinsics(i):
    return K_dict(i.fx, i.fy, i.ppx, i.ppy, i.width, i.height)


def align_depth(depth_m, d_K, c_K, R, t):
    """Re-project a depth image (metres, depth-sensor frame) into the colour camera.
    R, t: depth -> colour extrinsics. Returns an HxW float32 depth image aligned to colour (0 = none)."""
    h, w = depth_m.shape
    v, u = np.nonzero(depth_m > 0)
    z = depth_m[v, u]
    x = (u - d_K["cx"]) * z / d_K["fx"]
    y = (v - d_K["cy"]) * z / d_K["fy"]
    p = np.asarray(R, float).reshape(3, 3) @ np.vstack([x, y, z]) + np.asarray(t, float).reshape(3, 1)
    ok = p[2] > 0.05
    p = p[:, ok]
    uc = np.round(p[0] * c_K["fx"] / p[2] + c_K["cx"]).astype(int)
    vc = np.round(p[1] * c_K["fy"] / p[2] + c_K["cy"]).astype(int)
    inside = (uc >= 0) & (uc < c_K["w"]) & (vc >= 0) & (vc < c_K["h"])
    uc, vc, zc = uc[inside], vc[inside], p[2][inside]
    out = np.zeros((c_K["h"], c_K["w"]), np.float32)
    order = np.argsort(-zc)                       # far first, near overwrites (z-buffer)
    out[vc[order], uc[order]] = zc[order]
    # Fill the 1-px holes left by the forward projection.
    import cv2
    filled = cv2.dilate(out, np.ones((3, 3), np.uint8))
    holes = out == 0
    out[holes] = filled[holes]
    return out


def points(depth_m, K):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    return np.stack([(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z], -1)


def fit_plane(depth_m, K, mask=None, iters=250, tol=0.006, seed=0):
    """Largest plane (the table) by RANSAC + least-squares refit. Returns [nx, ny, nz, d] with
    n.p + d = height above the table (positive towards the camera), and the inlier fraction."""
    P = points(depth_m, K)
    ok = (depth_m > 0.2) & (depth_m < 2.0)
    if mask is not None:
        ok &= mask
    pts = P[ok]
    if len(pts) < 500:
        raise ValueError("not enough depth points to find the table")
    rng = np.random.default_rng(seed)
    sub = pts[rng.choice(len(pts), min(len(pts), 40000), replace=False)]
    best_n, best = 0, None
    for _ in range(iters):
        s = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ s[0]
        cnt = int((np.abs(sub @ n + d) < tol).sum())
        if cnt > best_n:
            best_n, best = cnt, (n, d)
    n, d = best
    inl = np.abs(pts @ n + d) < tol
    c = pts[inl].mean(0)
    _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    n = vt[-1]
    d = -n @ c
    if d < 0:                                      # camera (origin) must be on the positive side
        n, d = -n, -d
    return [float(n[0]), float(n[1]), float(n[2]), float(d)], float(inl.mean())


def heights(depth_m, K, plane):
    h = points(depth_m, K) @ np.asarray(plane[:3]) + plane[3]
    h[depth_m <= 0] = np.nan
    return h.astype(np.float32)


def ray_plane(u, v, K, plane, lift=0.0):
    """3D point (camera frame) where pixel (u, v) meets the plane raised by `lift` metres."""
    r = np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0])
    n, d = np.asarray(plane[:3]), plane[3]
    s = -(d - lift) / (n @ r)
    return r * s


def project(p, K):
    p = np.asarray(p, float)
    return float(p[0] * K["fx"] / p[2] + K["cx"]), float(p[1] * K["fy"] / p[2] + K["cy"])


def plane_basis(plane):
    n = np.asarray(plane[:3], float)
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1), n


def plane_change(a, b):
    """(angle between normals in degrees, change of camera height in mm) of two planes."""
    na, nb = np.asarray(a[:3]), np.asarray(b[:3])
    ang = float(np.degrees(np.arccos(np.clip(abs(na @ nb), -1, 1))))
    return ang, float(abs(a[3] - b[3]) * 1000)


# --- detection ----------------------------------------------------------------------------
def _hue_name(h, s, v):
    if s < 60:
        return "grey" if v < 140 else "white"
    if h < 8 or h >= 170:
        return "red"
    if h < 22:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 80:
        return "green"
    if h < 130:
        return "blue"
    return "pink"


BOWL_HUES = {"pink": lambda h, s, v: ((h >= 140) | (h < 6)) & (s > 45) & (v > 110),
             "blue": lambda h, s, v: (h >= 85) & (h < 130) & (s > 110) & (v > 50)}


def find_bowls(bgr, min_area=5000):
    """Big round discs of one colour (pink / blue), segmented per colour so two touching bowls stay
    apart. Bricks lying in a bowl are inside its hull. Returns [{color, u, v, r, hull}]."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., i].astype(np.int16) for i in range(3))
    bowls = []
    for name, rule in BOWL_HUES.items():
        col = rule(H, S, V).astype(np.uint8) * 255
        col = cv2.morphologyEx(col, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        col = cv2.morphologyEx(col, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(col)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            ys, xs = np.nonzero(labels == i)
            hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.float32))
            (u, v), r = cv2.minEnclosingCircle(hull)
            circ = cv2.contourArea(hull) / max(1.0, np.pi * r * r)
            if circ < 0.6:                          # not round: a box or the arm, not a bowl
                continue
            bowls.append({"color": name, "u": float(u), "v": float(v), "r": float(r), "round": round(float(circ), 2),
                          "hull": hull.reshape(-1, 2).round().astype(int).tolist()})
    return sorted(bowls, key=lambda b: -b["r"])


def table_region(hm, tol_mm=5.0, shrink=0.03):
    """Polygon of the table top: the biggest connected patch that lies ON the fitted plane."""
    import cv2
    if hm is None:
        return None
    flat = (np.abs(np.nan_to_num(hm, nan=99.0)) < tol_mm).astype(np.uint8) * 255
    flat = cv2.morphologyEx(flat, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(flat)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == big)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.float32)).reshape(-1, 2)
    c = hull.mean(axis=0)
    return ((hull - c) * (1 - shrink) + c).round().astype(int).tolist()


def detect(bgr, depth_m=None, K=None, plane=None, region=None, params=None, robot_mask=None):
    """Find bricks, bowls and obstacles. `region` = polygon (pixels) of the pick area, or None.
    Returns {"bricks": [...], "bowls": [...], "obstacles": [...], "in_bowls": n, "height": HxW|None}."""
    import cv2
    p = dict(DEFAULTS, **(params or {}))
    h_img, w_img = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bowls = find_bowls(bgr, p["bowl_min_area_px"])
    bowl_mask = np.zeros((h_img, w_img), np.uint8)
    for b in bowls:
        cv2.fillPoly(bowl_mask, [np.asarray(b["hull"], np.int32)], 255)
    bowl_mask = cv2.dilate(bowl_mask, np.ones((2 * p["bowl_margin_px"] + 1,) * 2, np.uint8)) > 0

    hm = None
    if depth_m is not None and K is not None and plane is not None:
        hm = heights(depth_m, K, plane) * 1000.0                      # mm
        hz = np.nan_to_num(hm, nan=0.0)
        raised = hz > p["min_height_mm"]
        tableish = (hsv[..., 1] < 45) & (hsv[..., 2] > 150)
        strong = (hsv[..., 1] > 110) & (hsv[..., 2] > 60)             # clearly coloured (not the white table)
        # Raised and not table-coloured; or clearly coloured and a little raised (depth smooths small bricks
        # far from the camera); or clearly raised (white bricks).
        cand = (raised & ~tableish) | (strong & (hz > 2.5)) | (hz > 2.5 * p["min_height_mm"])
        # Cut away anything far taller than a brick (the arm, a hand) so a brick touching it stays separate.
        tall = hz > p["max_height_mm"] + 15
        cand &= ~tall
        tall_near = cv2.dilate(tall.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    else:
        colourful = (hsv[..., 1] > 80) & (hsv[..., 2] > 50)
        dark = hsv[..., 2] < 90
        cand = colourful | dark
    cand = cand.astype(np.uint8) * 255
    if region is not None and len(region) >= 3:
        roi = np.zeros_like(cand)
        cv2.fillPoly(roi, [np.asarray(region, np.int32)], 255)
        cand &= roi
    if robot_mask is not None:
        cand[robot_mask] = 0
    in_bowls = cand.copy()
    cand[bowl_mask] = 0
    k = np.ones((5, 5), np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=2)

    # Count blobs sitting inside bowls (left-over bricks), for the scene report only.
    inside = np.zeros_like(in_bowls)
    for b in bowls:
        cv2.fillPoly(inside, [np.asarray(b["hull"], np.int32)], 255)
    in_bowls &= inside
    in_bowls = cv2.morphologyEx(in_bowls, cv2.MORPH_OPEN, k)
    n_in = 0
    if hm is not None:
        tall = ((np.nan_to_num(hm, nan=0.0) > 22) & (inside > 0)).astype(np.uint8)   # above the bowl floor+rim
        tall = cv2.morphologyEx(tall, cv2.MORPH_OPEN, k)
        nb, _, sb, _ = cv2.connectedComponentsWithStats(tall)
        n_in = int(sum(1 for i in range(1, nb) if sb[i, cv2.CC_STAT_AREA] > 250))

    bricks, obstacles = [], []
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand)
    pieces = []                                       # (mask, arm_part or None = decide by colour)
    hue, sat, val = (hsv[..., i].astype(np.int16) for i in range(3))
    brickish = (sat > 110) & (val > 35) & ~((hue >= 85) & (hue < 135))      # clearly coloured, not arm-blue
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 120:
            continue
        m = labels == i
        if hm is not None and tall_near[m].any():
            # Touches the arm: keep each clearly coloured, non-blue piece as its own candidate, the rest is arm.
            sub = cv2.morphologyEx((m & brickish).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            ns, ls, ss, _ = cv2.connectedComponentsWithStats(sub)
            taken = np.zeros_like(m)
            for j in range(1, ns):
                if ss[j, cv2.CC_STAT_AREA] >= 150:
                    pm = ls == j
                    pieces.append((pm, False))
                    taken |= pm
            rest = m & ~(cv2.dilate(taken.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
            if rest.sum() >= 120:
                pieces.append((rest, True))
        else:
            pieces.append((m, None))
    # Closing the height mask can join neighbouring bricks of different colours.
    # Split only oversized candidates; keep ordinary bricks (including small stickers) intact.
    separated = []
    split_ids = set()
    for mask, arm_flag in pieces:
        ys, xs = np.nonzero(mask)
        oversized = False
        if not arm_flag and K is not None and plane is not None:
            corners = np.array([ray_plane(x, y, K, plane, 0.015)
                                for x, y in ((xs.min(), ys.min()), (xs.max(), ys.min()),
                                             (xs.max(), ys.max()), (xs.min(), ys.max()))])
            e1, e2, _ = plane_basis(plane)
            q = np.column_stack([corners @ e1, corners @ e2]).astype(np.float32) * 1000
            _, dims, _ = cv2.minAreaRect(q)
            oversized = max(dims) > p["max_length_mm"] or min(dims) > p["max_width_mm"]
        parts = []
        if oversized:
            hue_groups = ((hue < 8) | (hue >= 170), (hue >= 8) & (hue < 22),
                          (hue >= 22) & (hue < 35), (hue >= 35) & (hue < 80),
                          (hue >= 80) & (hue < 130), (hue >= 130) & (hue < 170))
            for group in hue_groups:
                sub = (mask & group & (sat > 60) & (val > 50)).astype(np.uint8)
                sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                count, labs, sizes, _ = cv2.connectedComponentsWithStats(sub)
                for j in range(1, count):
                    if sizes[j, cv2.CC_STAT_AREA] >= 150:
                        part = labs == j
                        # A hand, arm, or tall background fragment is never a brick target.
                        if hm is not None:
                            values = hm[part]
                            valid = values[np.isfinite(values)]
                            if not len(valid) or np.percentile(valid, 90) >= p["max_height_mm"] - 5:
                                continue
                        parts.append(part)
        if len(parts) >= 2:
            for part in parts:
                separated.append((part, arm_flag))
                split_ids.add(id(part))
            taken = np.logical_or.reduce(parts)
            rest = mask & ~(cv2.dilate(taken.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
            if rest.sum() >= 120:
                separated.append((rest, arm_flag))
        else:
            separated.append((mask, arm_flag))
    for m, arm_flag in separated:
        ys, xs = np.nonzero(m)
        area_px = int(len(xs))
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
        u, v = float(xs.mean()), float(ys.mean())
        touches_edge = x <= 1 or y <= 1 or x + w >= w_img - 1 or y + h >= h_img - 1
        hh = float(np.nanmedian(hm[m])) if hm is not None and np.isfinite(hm[m]).any() else None
        htop = float(np.nanpercentile(hm[m], 90)) if hm is not None and np.isfinite(hm[m]).any() else None
        med = np.median(hsv[m], axis=0)
        colour = _hue_name(*med)
        # A blue / grey / white piece right next to something tall is the arm's own lower part, not a brick.
        arm_part = arm_flag if arm_flag is not None else \
            (hm is not None and bool(tall_near[m].any()) and colour in ("blue", "grey", "white"))
        rec = {"u": round(u, 1), "v": round(v, 1), "area_px": area_px, "box": [x, y, w, h], "color": colour,
               "height_mm": None if htop is None else round(htop, 1)}
        if plane is not None and K is not None:
            # Size and orientation in millimetres on the table plane (top face projected down).
            lift = (hh or 15.0) / 1000.0
            shape = m.astype(np.uint8)
            if hm is not None and id(m) not in split_ids:
                # The depth blob is soft at the edges; the brick's colour gives a crisper outline.
                near = cv2.dilate(shape, np.ones((9, 9), np.uint8)) > 0
                crisp = near & (((hsv[..., 1] > 80) & (hsv[..., 2] > 50)) | (hsv[..., 2] < 90))
                if crisp.sum() > 0.6 * area_px:
                    shape = cv2.morphologyEx(crisp.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
            e1, e2, nrm = plane_basis(plane)
            P3 = np.array([ray_plane(a, b, K, plane, lift) for a, b in cnt])
            Q = np.column_stack([P3 @ e1, P3 @ e2]).astype(np.float32) * 1000.0     # mm in the plane
            (qc1, qc2), (s1, s2), ang = cv2.minAreaRect(Q)
            area_mm2 = float(cv2.contourArea(Q))
            th = np.radians(ang)
            d1 = np.cos(th) * e1 + np.sin(th) * e2
            d2 = -np.sin(th) * e1 + np.cos(th) * e2
            long_axis = d1 if s1 >= s2 else d2
            centre_top = (qc1 / 1000.0) * e1 + (qc2 / 1000.0) * e2
            # Point on the table plane under the top-face centre, then the grasp height above it.
            on_plane = centre_top - nrm * (centre_top @ nrm + plane[3])
            grasp = on_plane + nrm * p["grasp_height_m"]
            length, width = max(s1, s2), min(s1, s2)
            rec.update(area_mm2=round(area_mm2), size_mm=[round(length), round(width)],
                       elongated=bool(length > 1.35 * max(width, 1.0)),
                       long_axis_cam=[round(float(c), 4) for c in long_axis],
                       grasp_cam=[round(float(c), 4) for c in grasp])
            is_brick = (p["min_area_mm2"] <= area_mm2 <= p["max_area_mm2"] and length <= p["max_length_mm"]
                        and width <= p["max_width_mm"] and not touches_edge and (htop is None or htop <= p["max_height_mm"]))
        else:
            is_brick = 400 <= area_px <= 6000 and not touches_edge
        if arm_part:
            is_brick = False
        if is_brick:
            bricks.append(rec)
        else:
            why = "part of the arm" if arm_part else "touches the image edge (arm or person?)" if touches_edge else \
                  "too tall" if htop is not None and htop > p["max_height_mm"] else \
                  "too small" if rec.get("area_mm2", 999) < p["min_area_mm2"] else "too big for a brick"
            rec["why"] = why
            obstacles.append(rec)
    # Free space around each brick (mm to the nearest other object), so isolated bricks go first.
    # Only things on the table count (the wall, the floor and people beside the table do not).
    table = table_region(hm)
    if table is not None:
        poly = np.asarray(table, np.float32)
        on = lambda o: cv2.pointPolygonTest(poly, (o["u"], o["v"]), False) >= 0
        bricks = [b for b in bricks if on(b)]
        obstacles = [o for o in obstacles if on(o) and not (o.get("why") == "too small")]
    else:
        obstacles = [o for o in obstacles if o.get("why") != "too small"]
    objs = bricks + obstacles
    for b in bricks:
        others = [o for o in objs if o is not b]
        b["clear_px"] = round(min((np.hypot(o["u"] - b["u"], o["v"] - b["v"]) for o in others), default=999.0), 1)
    return {"bricks": bricks, "bowls": bowls, "obstacles": obstacles, "in_bowls": n_in, "height": hm, "table": table}


# --- pictures for the dashboard -----------------------------------------------------------
ACCENT = (12, 89, 232)          # BGR of the dashboard accent (#E8590C)
BOWL_NAME = {"pink": "Good", "blue": "Defective"}


def _label(img, text, org, fg=(255, 255, 255), bg=(20, 20, 20)):
    import cv2
    font, sc, th = cv2.FONT_HERSHEY_DUPLEX, 0.5, 1
    (w, h), base = cv2.getTextSize(text, font, sc, th)
    x, y = int(org[0]), int(org[1])
    x = max(2, min(img.shape[1] - w - 12, x))
    y = max(h + 8, y)
    cv2.rectangle(img, (x, y - h - 7), (x + w + 10, y + 4), bg, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + 5, y - 1), font, sc, fg, th, cv2.LINE_AA)


def overlay(bgr, res, region=None, target=None, picks=None, ignore=None):
    """Annotated top view for the dashboard: numbered bricks, bowls by what they hold, other objects in
    grey, the next pick ringed. `ignore` = (u, v) of the parked arm (never drawn as an object)."""
    import cv2
    img = bgr.copy()
    for b in res.get("bowls", []):
        cv2.polylines(img, [np.asarray(b["hull"], np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
        name = BOWL_NAME.get(b["color"], b["color"].capitalize())
        tw = cv2.getTextSize(f"{name} bowl", cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0][0]
        _label(img, f"{name} bowl", (b["u"] - tw / 2 - 5, b["v"] - b["r"] * 0.55))
    for o in res.get("obstacles", []):
        if o.get("why", "").startswith(("touches", "part of the arm")):
            continue
        if ignore is not None and np.hypot(o["u"] - ignore[0], o["v"] - ignore[1]) < 120:
            continue
        x, y, w, h = o["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (150, 150, 150), 2, cv2.LINE_AA)
        _label(img, "Not a brick", (x, y - 4), fg=(255, 255, 255), bg=(110, 110, 110))
    for i, b in enumerate(res.get("bricks", []), 1):
        c = (int(b["u"]), int(b["v"]))
        r = int(max(14, 0.6 * max(b["box"][2], b["box"][3])))
        nxt = target is not None and abs(b["u"] - target[0]) < 2 and abs(b["v"] - target[1]) < 2
        cv2.circle(img, c, r, ACCENT, 3 if nxt else 2, cv2.LINE_AA)
        if nxt:
            cv2.circle(img, c, r + 6, ACCENT, 1, cv2.LINE_AA)
        cv2.circle(img, (c[0] + r - 2, c[1] - r + 2), 10, ACCENT, -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.putText(img, str(i), (c[0] + r - 2 - tw // 2, c[1] - r + 2 + th // 2), cv2.FONT_HERSHEY_DUPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    for pk in picks or []:
        col = (69, 122, 30) if pk.get("ok") else (44, 48, 201)
        cv2.circle(img, (int(pk["u"]), int(pk["v"])), 6, col, -1, cv2.LINE_AA)
    return img


def height_image(hm, max_mm=60.0):
    import cv2
    if hm is None:
        return None
    v = np.clip(np.nan_to_num(hm, nan=0.0) / max_mm, 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~np.isfinite(hm)] = (30, 30, 30)
    return img


def report(res, plane=None, ref=None, inlier=None):
    """Plain-language scene check for the operator (what to fix before a run)."""
    warn = []
    if ref:
        if plane is not None and ref.get("plane"):
            ang, dh = plane_change(plane, ref["plane"])
            if ang > 1.5 or dh > 15:
                warn.append(f"Top camera moved since calibration (tilt {ang:.1f} deg, height {dh:.0f} mm): "
                            "re-run ./sl table --auto")
        for b in res["bowls"]:
            r = (ref.get("bowls") or {}).get(b["color"])
            if r:
                dpx = float(np.hypot(b["u"] - r[0], b["v"] - r[1]))
                if dpx > 25:
                    warn.append(f"The {b['color']} bowl moved {dpx:.0f} px since the bowl poses were taught: "
                                "put it back, or turn on scan.place_by_vision")
    colours = [b["color"] for b in res["bowls"]]
    for c in ("pink", "blue"):
        if c not in colours:
            warn.append(f"No {c} bowl seen by the top camera")
    for o in res["obstacles"]:
        if o.get("why", "").startswith(("touches the image edge", "part of the arm")):
            continue                                   # cables / people at the border, the arm itself
        warn.append(f"Something that is not a brick at ({o['u']:.0f}, {o['v']:.0f}): {o.get('why')}"
                    + (f", {o['height_mm']:.0f} mm tall" if o.get("height_mm") else ""))
    if res.get("in_bowls"):
        warn.append(f"{res['in_bowls']} brick(s) left in the bowls (fine, just not counted)")
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "bricks": len(res["bricks"]),
            "brick_list": [{k: b.get(k) for k in ("u", "v", "color", "height_mm", "size_mm", "elongated", "clear_px")}
                           for b in res["bricks"]],
            "bowls": [{k: b[k] for k in ("color", "u", "v", "r")} for b in res["bowls"]],
            "obstacles": [{k: o.get(k) for k in ("u", "v", "why", "height_mm", "size_mm")} for o in res["obstacles"]
                          if not o.get("why", "").startswith(("touches the image edge", "part of the arm"))],
            "in_bowls": res.get("in_bowls", 0), "plane_inliers": inlier, "depth": res.get("height") is not None,
            "warnings": warn, "ok": not [w for w in warn if "left in the bowls" not in w]}


def save(bgr, res, rep, region=None, target=None, out_dir=OUT_DIR, ignore=None):
    """Write latest.jpg (annotated), latest_raw.jpg, latest_height.jpg and latest.json for the dashboard."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "latest_raw.jpg"), bgr)
    cv2.imwrite(os.path.join(out_dir, "latest.jpg"), overlay(bgr, res, region, target, ignore=ignore))
    hi = height_image(res.get("height"))
    if hi is not None:
        cv2.imwrite(os.path.join(out_dir, "latest_height.jpg"), hi)
    elif os.path.exists(os.path.join(out_dir, "latest_height.jpg")):
        os.remove(os.path.join(out_dir, "latest_height.jpg"))
    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)


# --- camera-only capture (no arm) ---------------------------------------------------------
def rgbd_from_camera(cam, frames=5):
    """(bgr, depth aligned to colour in metres or None, K) from a connected LeRobot RealSenseCamera."""
    rgb = cam.async_read()
    bgr = np.ascontiguousarray(np.asarray(rgb)[..., ::-1])
    if not getattr(cam, "use_depth", False):
        return bgr, None, None
    import pyrealsense2 as rs
    prof = cam.rs_profile
    ds = prof.get_stream(rs.stream.depth).as_video_stream_profile()
    cs = prof.get_stream(rs.stream.color).as_video_stream_profile()
    d_K, c_K = rs_intrinsics(ds.get_intrinsics()), rs_intrinsics(cs.get_intrinsics())
    ex = ds.get_extrinsics_to(cs)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    stack = []
    for _ in range(frames):
        d = np.asarray(cam.async_read_depth(), np.float32).squeeze() * scale
        stack.append(d)
        time.sleep(1 / 30)
    st = np.stack(stack)
    st[st <= 0] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # pixels with no depth in any frame
        d = np.nan_to_num(np.nanmedian(st, axis=0), nan=0.0).astype(np.float32)
    aligned = align_depth(d, d_K, c_K, np.asarray(ex.rotation).reshape(3, 3).T, np.asarray(ex.translation))
    return bgr, aligned, c_K


def capture(cfg, name="top"):
    """Open ONLY the top camera (colour + depth), grab one averaged frame, close it. The arm is not touched."""
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    c = cfg["cameras"][name]
    cam = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name=str(c["serial"]), fps=c["fps"],
                                                width=c["width"], height=c["height"], use_depth=True))
    cam.connect()
    try:
        time.sleep(1.0)                                # auto-exposure settles
        return rgbd_from_camera(cam)
    finally:
        cam.disconnect()
