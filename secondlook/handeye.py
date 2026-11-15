"""Top camera -> robot base transform ("hand-eye" calibration), learned from point pairs.

A pair = the same physical spot seen two ways:
  cam   grasp point of a brick in top-camera coordinates (metres, from depth or the table plane)
  base  where the tool point was (forward kinematics of the joints) when it held / released that brick

Sources of pairs, best first:
  drop      `./sl table --auto`: the arm puts a brick down at a known spot, the camera finds it
  putback   a second look puts a brick back; the next scan sees where it really landed
  manual    the old hand-held table_map.yaml points (people hold the arm differently, so these are
            only used until there are enough drop/putback pairs)

The fit is a rigid transform (rotation + translation, Kabsch) with outlier rejection. Every pair's
error in mm is kept so the dashboard can show how good the calibration is.
"""
import os
import time

import numpy as np
import yaml

FILE = os.path.join("secondlook", "handeye.yaml")
MAX_PAIRS = 60
MIN_AUTO = 4                    # with this many drop/putback pairs, manual pairs are no longer used


def kabsch(A, B, w=None):
    """R, t minimising sum w*|R a + t - b|^2."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ca, cb = (A * w[:, None]).sum(0), (B * w[:, None]).sum(0)
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


class HandEye:
    def __init__(self, data=None):
        self.data = data or {"pairs": []}

    # --- persistence ----------------------------------------------------------------------
    @classmethod
    def load(cls, path=None):
        path = path or FILE
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run `./sl table --from-map` or `./sl table --auto`")
        with open(path) as f:
            return cls(yaml.safe_load(f))

    @classmethod
    def exists(cls, path=None):
        return os.path.exists(path or FILE)

    def save(self, path=None):
        path = path or FILE
        if os.path.exists(path):
            import shutil
            shutil.copy(path, path + ".prev")
        with open(path, "w") as f:
            yaml.safe_dump(self.data, f, sort_keys=False, default_flow_style=None)

    # --- transform ------------------------------------------------------------------------
    @property
    def R(self):
        return np.asarray(self.data["R"], float)

    @property
    def t(self):
        return np.asarray(self.data["t"], float)

    def to_base(self, p_cam):
        return self.R @ np.asarray(p_cam, float) + self.t

    def to_cam(self, p_base):
        return self.R.T @ (np.asarray(p_base, float) - self.t)

    def dir_to_base(self, v_cam):
        return self.R @ np.asarray(v_cam, float)

    def up_base(self):
        """Table normal (pointing up, away from the table) in the base frame."""
        n = self.R @ np.asarray(self.data["plane"][:3], float)
        return n / np.linalg.norm(n)

    @property
    def plane(self):
        return self.data.get("plane")

    @property
    def K(self):
        return self.data.get("K")

    # --- fitting --------------------------------------------------------------------------
    def used_pairs(self):
        pairs = self.data["pairs"]
        auto = [p for p in pairs if p["source"] != "manual" and not p.get("rejected")]
        if len(auto) >= MIN_AUTO:
            return auto
        return [p for p in pairs if not p.get("rejected")]

    def fit(self):
        pairs = [p for p in self.data["pairs"]]
        for p in pairs:
            p.pop("rejected", None)
        use = self.used_pairs()
        if len(use) < 3:
            raise ValueError(f"need at least 3 point pairs, have {len(use)}")
        weight = lambda p: 1.0 if p["source"] == "manual" else 3.0
        for _ in range(3):                                   # refit without gross outliers
            A = [p["cam"] for p in use]
            B = [p["base"] for p in use]
            R, t = kabsch(A, B, [weight(p) for p in use])
            err = np.linalg.norm(np.asarray(A) @ R.T + t - np.asarray(B), axis=1) * 1000
            lim = max(25.0, 3.0 * float(np.median(err)))
            bad = [p for p, e in zip(use, err) if e > lim]
            if not bad or len(use) - len(bad) < 4:
                break
            for p in bad:
                p["rejected"] = True
            use = [p for p in use if not p.get("rejected")]
        self.data["R"] = R.round(6).tolist()
        self.data["t"] = t.round(6).tolist()
        for p in self.data["pairs"]:
            p["err_mm"] = round(float(np.linalg.norm(R @ np.asarray(p["cam"]) + t - np.asarray(p["base"])) * 1000), 1)
        e = [p["err_mm"] for p in use]
        self.data["fit"] = {"pairs_used": len(use), "mean_mm": round(float(np.mean(e)), 1),
                            "max_mm": round(float(np.max(e)), 1), "sources": sorted({p["source"] for p in use}),
                            "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        return self.data["fit"]

    def add_pair(self, cam, base, source, refit=True, **extra):
        pair = {"cam": [round(float(c), 5) for c in cam], "base": [round(float(c), 5) for c in base],
                "source": source, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
        self.data["pairs"].append(pair)
        # Keep the newest pairs; manual ones are kept only while needed.
        if len(self.data["pairs"]) > MAX_PAIRS:
            manual = [p for p in self.data["pairs"] if p["source"] == "manual"]
            auto = [p for p in self.data["pairs"] if p["source"] != "manual"]
            self.data["pairs"] = (manual if len(auto) < MIN_AUTO else []) + auto[-MAX_PAIRS:]
        return self.fit() if refit else None

    # --- bootstrap from the old hand-held map ------------------------------------------------
    @classmethod
    def from_table_map(cls, points, plane, K, arm, grasp_height=0.012):
        from .scene import ray_plane
        he = cls({"plane": [float(x) for x in plane], "K": K, "pairs": []})
        for p in points:
            cam = ray_plane(p["u"], p["v"], K, plane, grasp_height)
            he.add_pair(cam, arm.tcp(p["pick"]), "manual", refit=False, u=round(p["u"], 1), v=round(p["v"], 1))
        he.fit()
        return he
