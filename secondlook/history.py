"""Persistent history of every brick and every pick attempt (SQLite, runs/history.db).

  bricks   one row per cycle: run, scores, verdict, bowl, true label, photos, timing, where it was picked
  picks    one row per grasp attempt: method, pixel + base position, outcome (grasped / missed / slipped)

The control center's History and Learning tabs read it; labels set there go back into the
learning set (data/bricks_live) so the next retrain uses them. Old runs (runs/*.csv) are imported
once with `python -m secondlook.history backfill`.
"""
import csv
import glob
import json
import os
import re
import shutil
import sqlite3
import sys
import time

DB = os.path.join("runs", "history.db")
LIVE = os.path.join("data", "bricks_live")
INSPECT = os.path.join("runs", "inspections")

SCHEMA = """
CREATE TABLE IF NOT EXISTS bricks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, run TEXT, trial INTEGER, mode TEXT, policy TEXT,
  pick_ok INTEGER, regrasps INTEGER, dropped INTEGER,
  score_v1 REAL, score_v2 REAL, thr REAL, band REAL,
  verdict TEXT, final_bowl TEXT, truth TEXT,
  cycle_s REAL, wait_s REAL, grip REAL,
  pick_u REAL, pick_v REAL, pick_x REAL, pick_y REAL,
  photos TEXT, notes TEXT,
  UNIQUE(run, trial)
);
CREATE TABLE IF NOT EXISTS picks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, run TEXT, method TEXT,
  u REAL, v REAL, x REAL, y REAL, z REAL,
  outcome TEXT, grip REAL, height_mm REAL, size TEXT, color TEXT,
  roll REAL, tilt_deg REAL, ik_err_mm REAL, joints TEXT
);
CREATE INDEX IF NOT EXISTS bricks_run ON bricks(run);
CREATE INDEX IF NOT EXISTS picks_run ON picks(run);
"""

BRICK_COLS = ["ts", "run", "trial", "mode", "policy", "pick_ok", "regrasps", "dropped", "score_v1", "score_v2", "thr",
              "band", "verdict", "final_bowl", "truth", "cycle_s", "wait_s", "grip", "pick_u", "pick_v", "pick_x",
              "pick_y", "photos", "notes"]
PICK_COLS = ["ts", "run", "method", "u", "v", "x", "y", "z", "outcome", "grip", "height_mm", "size", "color", "roll",
             "tilt_deg", "ik_err_mm", "joints"]


class _Conn(sqlite3.Connection):
    """`with connect() as con:` commits (or rolls back) AND closes, so a long-running server never leaks."""

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.close()


def connect(path=None):
    path = path or DB
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path, timeout=10, factory=_Conn)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def _clean(v):
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, default=float)
    if v == "":
        return None
    if hasattr(v, "item"):                       # numpy scalar
        return v.item()
    return v


def add_brick(path=None, **row):
    row.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    cols = [c for c in BRICK_COLS if c in row]
    with connect(path) as con:
        cur = con.execute(f"INSERT OR REPLACE INTO bricks ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          [_clean(row[c]) for c in cols])
        return cur.lastrowid


def add_pick(path=None, **row):
    row.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    cols = [c for c in PICK_COLS if c in row]
    with connect(path) as con:
        cur = con.execute(f"INSERT INTO picks ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          [_clean(row[c]) for c in cols])
        return cur.lastrowid


def bricks(path=None, limit=200, run=None, verdict=None, truth=None, only_picked=False):
    q, args = "SELECT * FROM bricks WHERE 1=1", []
    if run:
        q += " AND run = ?"
        args.append(run)
    if verdict:
        q += " AND verdict = ?"
        args.append(verdict)
    if truth == "none":
        q += " AND (truth IS NULL OR truth = '')"
    elif truth == "wrong":
        q += " AND truth IN ('good','defective') AND final_bowl IS NOT NULL AND truth != final_bowl"
    elif truth:
        q += " AND truth = ?"
        args.append(truth)
    if only_picked:
        q += " AND pick_ok = 1"
    q += " ORDER BY ts DESC, id DESC LIMIT ?"
    args.append(int(limit))
    with connect(path) as con:
        out = [dict(r) for r in con.execute(q, args)]
    for r in out:
        try:
            r["photos"] = json.loads(r["photos"]) if r.get("photos") else []
        except ValueError:
            r["photos"] = []
    return out


def picks(path=None, limit=500, run=None):
    q, args = "SELECT * FROM picks", []
    if run:
        q += " WHERE run = ?"
        args.append(run)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with connect(path) as con:
        return [dict(r) for r in con.execute(q, args)]


def runs(path=None):
    with connect(path) as con:
        return [dict(r) for r in con.execute(
            "SELECT run, MIN(ts) AS first, MAX(ts) AS last, COUNT(*) AS n, SUM(pick_ok) AS picked FROM bricks "
            "GROUP BY run ORDER BY last DESC")]


def set_truth(brick_id, truth, path=None, root="."):
    """Set (or clear) the true label of one brick and file its photos into the learning set.
    Returns a short message for the operator."""
    if truth not in ("good", "defective", None, ""):
        raise ValueError("truth must be good, defective or empty")
    with connect(path) as con:
        r = con.execute("SELECT photos FROM bricks WHERE id = ?", (brick_id,)).fetchone()
        if r is None:
            raise KeyError(f"no brick {brick_id}")
        con.execute("UPDATE bricks SET truth = ? WHERE id = ?", (truth or None, brick_id))
    photos = json.loads(r["photos"]) if r["photos"] else []
    moved = file_photos(photos, truth, root)
    return f"label saved; {moved} photo(s) {'added to the learning set' if truth else 'removed from the learning set'}"


def file_photos(photos, truth, root="."):
    """Copy inspection photos into data/bricks_live/<good|defect> (and out of the other folder)."""
    n = 0
    for f in photos:
        if not re.fullmatch(r"[\w.-]+\.png", f or ""):
            continue
        src = os.path.join(root, INSPECT, f)
        for lab in ("good", "defect"):
            other = os.path.join(root, LIVE, lab, f)
            if os.path.exists(other) and lab != {"good": "good", "defective": "defect"}.get(truth):
                os.remove(other)
        if truth and os.path.exists(src):
            dst = os.path.join(root, LIVE, "good" if truth == "good" else "defect")
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, os.path.join(dst, f))
            n += 1
    return n


def learning_stats(path=None):
    """Numbers for the Learning tab: score distributions by true label, best threshold on real data,
    pick success by method."""
    with connect(path) as con:
        lab = [dict(r) for r in con.execute(
            "SELECT score_v1, score_v2, truth, final_bowl, thr, ts, run, policy FROM bricks "
            "WHERE pick_ok = 1 AND truth IN ('good','defective') AND score_v1 IS NOT NULL ORDER BY ts")]
        pk = [dict(r) for r in con.execute("SELECT method, outcome, COUNT(*) AS n FROM picks GROUP BY method, outcome")]
        pol = [dict(r) for r in con.execute(
            "SELECT policy, COUNT(*) AS n, SUM(pick_ok) AS ok FROM bricks GROUP BY policy")]
    good = [r["score_v1"] for r in lab if r["truth"] == "good"]
    bad = [r["score_v1"] for r in lab if r["truth"] == "defective"]
    best = None
    if good and bad:
        cands = sorted(set(good + bad))
        for i in range(len(cands)):
            thr = cands[i] + 1e-4 if i == len(cands) - 1 else (cands[i] + cands[i + 1]) / 2
            fa = sum(1 for s in good if s >= thr)
            miss = sum(1 for s in bad if s < thr)
            # Missing a defect is worse than a false alarm (a human re-checks the reject bowl).
            cost = fa + 2 * miss
            if best is None or cost < best["cost"]:
                best = {"thr": round(thr, 3), "false_alarms": fa, "missed": miss, "cost": cost,
                        "correct": len(good) + len(bad) - fa - miss, "n": len(good) + len(bad)}
    methods = {}
    for r in pk:
        m = methods.setdefault(r["method"] or "?", {"grasped": 0, "missed": 0, "slipped": 0})
        m[r["outcome"]] = m.get(r["outcome"], 0) + r["n"]
    return {"good_scores": good, "defect_scores": bad, "best_threshold": best, "labelled": lab,
            "pick_methods": methods, "policies": pol}


# --- one-off import of the CSV runs ---------------------------------------------------------
def _photo_index(root):
    out = []
    for f in glob.glob(os.path.join(root, INSPECT, "*.png")):
        name = os.path.basename(f)
        m = re.match(r"(?:(\d{8})-)?(\d{6})", name)
        if m:
            out.append((m[2], name, os.path.getmtime(f)))
    return out


def backfill(root=".", path=None):
    """Import runs/*.csv (skips rows already in the DB) and link inspection photos by time."""
    photos = _photo_index(root)
    n = 0
    for f in sorted(glob.glob(os.path.join(root, "runs", "*.csv"))):
        day = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(f)))
        rows = list(csv.DictReader(open(f)))
        for r in rows:
            ts = r.get("ts") or ""
            # CSV times have no date; rows before midnight in a file written after midnight are the day before.
            d = day
            if ts and ts > "12:00:00" and time.strftime("%H", time.localtime(os.path.getmtime(f))) < "12":
                d = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(f) - 86400))
            linked = []
            if r.get("pick_ok") == "1" and r.get("cycle_s") and ts:
                start_s = _sec(ts) - min(float(r["cycle_s"]), 120)
                linked = sorted(name for hms, name, _ in photos if start_s - 2 <= _sec(hms) <= _sec(ts) + 2
                                and _day_ok(name, d))
            notes = r.get("notes") or ""
            m = re.search(r"dropped=(\d+)", notes)
            with connect(path) as con:
                exists = con.execute("SELECT 1 FROM bricks WHERE run = ? AND trial = ?",
                                     (r.get("run"), int(r.get("trial") or 0))).fetchone()
            if exists:
                continue
            add_brick(path, ts=f"{d} {ts}", run=r.get("run"), trial=int(r.get("trial") or 0), mode=r.get("layout"),
                      policy=r.get("policy"), pick_ok=int(r.get("pick_ok") or 0),
                      regrasps=int(r["regrasps"]) if r.get("regrasps") else None,
                      dropped=int(m[1]) if m else 0, score_v1=_f(r.get("score_v1")), score_v2=_f(r.get("score_v2")),
                      thr=_f(r.get("thr")), verdict=r.get("verdict") or None, final_bowl=r.get("final_bowl") or None,
                      truth=r.get("truth") or None, cycle_s=_f(r.get("cycle_s")), photos=linked, notes=notes)
            n += 1
    return n


def _day_ok(name, day):
    m = re.match(r"(\d{8})-", name)
    return not m or m[1] == day.replace("-", "")


def _sec(hms):
    hms = hms.replace(":", "")
    return int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6])


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    if sys.argv[1:2] == ["backfill"]:
        print("imported", backfill(), "rows into", DB)
    else:
        print(__doc__)
