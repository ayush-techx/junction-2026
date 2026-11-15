"""SecondLook control center: a persistent dashboard (always on, independent of runs).

Start/stop runs, see the live run, results of every run, recent inspection photos, learning data,
model settings, hardware owners and run logs. Standard library only; binds to localhost.

  python -m secondlook.center            (normally run by the systemd --user service)
  http://127.0.0.1:8770
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("static") / "center.html"
RUN_UI = "http://127.0.0.1:8765"
LOGS = ROOT / "runs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

MODES = {
    "sort": None,                       # built from the selected pick engine (see mode_args)
    "sort_label": None,                 # sort the table and ask for the true label of every brick
    "handfed": ["run", "--handfed", "--eval", "--learn-every", "5"],
    "describe": ["scene", "--speak"],
    "scene_check": ["scene"],
    "check_policy": ["check", "--policy"],
    "recalibrate": ["recalibrate"],
}
LABELS = {"sort": "Sort the table", "sort_label": "Sort and label", "handfed": "Hand-feed bricks",
          "describe": "Describe the table", "scene_check": "Scan the table", "check_policy": "System check",
          "recalibrate": "Retrain the defect model"}
STUDIO_MODELS = Path.home() / ".local/share/physicalai/models"
STUDIO_DB = Path.home() / ".local/share/physicalai/data/physicalai.db"
SCENE = ROOT / "runs" / "scene"

_job = {"proc": None, "mode": None, "log": None, "started": None}
_lock = threading.Lock()


def _running_external():
    out = subprocess.run(["pgrep", "-af", "secondlook.cell"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if "pgrep" not in l]


def _job_state():
    with _lock:
        p = _job["proc"]
        alive = p is not None and p.poll() is None
        return {"mode": _job["mode"], "label": LABELS.get(_job["mode"]), "alive": alive,
                "exit": None if p is None or alive else p.returncode, "started": _job["started"],
                "log": _job["log"] and os.path.basename(_job["log"]), "external": _running_external()}


def start(mode, name=None):
    if mode not in MODES:
        return 400, {"error": "unknown mode"}
    st = _job_state()
    if st["alive"] or st["external"]:
        return 409, {"error": "a SecondLook run is already active; stop it first"}
    args = mode_args(mode)
    if args[0] == "run":
        args += ["--name", name or f"{mode}_{time.strftime('%H%M')}"]
    log = LOGS / f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}.log"
    f = open(log, "w")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["DISPLAY"] = env.get("DISPLAY", ":0")
    # Don't pop a new browser tab: the run's live view is embedded here.
    env["SECONDLOOK_NO_BROWSER"] = "1"
    proc = subprocess.Popen([str(ROOT / "sl"), *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    f.close()                                      # the child has its own copy of the file
    with _lock:
        _job.update(proc=proc, mode=mode, log=str(log), started=time.strftime("%H:%M:%S"))
    return 200, {"ok": True, "log": log.name}


def picker_engine():
    return (_yaml(ROOT / "secondlook" / "config.yaml").get("station") or {}).get("picker", "vision")


def mode_args(mode):
    if mode in ("sort", "sort_label"):
        base = ["run", "--scan"] if picker_engine() == "vision" else ["run"]
        args = base + ["--auto", "--until-clear"]
        if mode == "sort_label":
            args += ["--eval", "--learn-every", "5"]
        return args
    return list(MODES[mode])


def stop():
    """Graceful: press STOP in the run (motors off safely), never kill the robot process."""
    try:
        req = urllib.request.Request(RUN_UI + "/cmd", json.dumps({"cmd": "stop"}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
        return 200, {"ok": True, "how": "STOP sent to the run"}
    except Exception:
        with _lock:
            p = _job["proc"]
        if p is not None and p.poll() is None and _job["mode"] == "recalibrate":
            p.terminate()
            return 200, {"ok": True, "how": "recalibration cancelled"}
        return 409, {"error": "no running dashboard to send STOP to"}


def _yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def status():
    cfg = _yaml(ROOT / "secondlook" / "config.yaml")
    ins, pol = cfg.get("inspection", {}), cfg.get("policy", {})
    cnt = lambda d: len(glob.glob(str(ROOT / d / "*.png")))
    live = None
    try:
        live = json.load(urllib.request.urlopen(RUN_UI + "/state", timeout=1))
    except Exception:
        pass
    owners = {}
    for name, pat in (("follower arm", "/dev/serial/by-id/*5AE6082572*"), ("leader arm", "/dev/serial/by-id/*5B91043227*"),
                      ("wrist camera", "/dev/v4l/by-id/*CAM1*index0")):
        devs = glob.glob(pat)
        if devs:
            out = subprocess.run(["fuser", os.path.realpath(devs[0])], capture_output=True, text=True).stdout
            owners[name] = out.split() or []
        else:
            owners[name] = None
    for pid_list in owners.values():
        if pid_list:
            for i, pid in enumerate(pid_list):
                pid = re.sub(r"\D", "", pid)
                try:
                    with open(f"/proc/{pid}/cmdline") as fh:
                        cmd = fh.read().replace("\0", " ")
                except OSError:
                    cmd = ""
                who = "SecondLook" if "secondlook" in cmd else "Studio" if "physicalai" in cmd else "other"
                pid_list[i] = f"{who} ({pid})"
    pol_path = pol.get("path") or ""
    pol_name = pol_path.split("models/")[-1].split("/")[0] if pol_path else None
    if pol_name and len(pol_name) == 36:      # Studio model id (uuid): show the short form
        pol_name = pol_name[:8]
    return {
        "job": _job_state(), "live": live,
        "model": {"inspection": ins.get("model"), "threshold": ins.get("threshold"), "band": ins.get("band"),
                  "device": ins.get("device"), "policy": pol_name,
                  "policy_task": pol.get("task")},
        "data": {"good": cnt("data/bricks/good"), "defect": cnt("data/bricks/defect"),
                 "live_good": cnt("data/bricks_live/good"), "live_defect": cnt("data/bricks_live/defect")},
        "owners": owners, "time": time.strftime("%H:%M:%S"),
    }


def results():
    out = []
    for f in sorted(glob.glob(str(ROOT / "runs" / "*.csv")), key=os.path.getmtime, reverse=True):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        picked = [r for r in rows if r.get("pick_ok") == "1"]
        lab = [r for r in picked if r.get("truth") in ("good", "defective")]
        fp = sum(1 for r in lab if r["truth"] == "good" and r["final_bowl"] == "defective")
        fn = sum(1 for r in lab if r["truth"] == "defective" and r["final_bowl"] == "good")
        ok = sum(1 for r in lab if r["truth"] == r["final_bowl"])
        cyc = [float(r["cycle_s"]) for r in picked if r.get("cycle_s")]
        out.append({"run": Path(f).stem, "when": time.strftime("%H:%M", time.localtime(os.path.getmtime(f))),
                    "attempts": len(rows), "picked": len(picked), "labelled": len(lab), "correct": ok,
                    "false_alarms": fp, "missed_defects": fn,
                    "second_looks": sum(1 for r in picked if r.get("regrasps") not in ("", "0", None)),
                    "drops": sum(1 for r in rows if "dropped" in (r.get("notes") or "")),
                    "cycle_s": round(sorted(cyc)[len(cyc) // 2], 1) if cyc else None,   # median: one long wait can't skew it
                    "policy": rows[-1].get("policy")})
    return out


def inspections(n=24):
    files = sorted(glob.glob(str(ROOT / "runs" / "inspections" / "*.png")), key=os.path.getmtime, reverse=True)[:n]
    out = []
    for f in files:
        name = os.path.basename(f)
        m = re.match(r"(?:\d{8}-)?(\d{6})(?:-\d+)?_([\d.]+)_(\w+)\.png", name)
        lab = "good" if os.path.exists(ROOT / "data/bricks_live/good" / name) else \
              "defective" if os.path.exists(ROOT / "data/bricks_live/defect" / name) else None
        out.append({"file": name, "t": m and f"{m[1][:2]}:{m[1][2:4]}:{m[1][4:]}", "score": m and float(m[2]),
                    "verdict": m and m[3], "label": lab})
    return out


def log_tail(n=80):
    with _lock:
        log = _job["log"]
    if not log:
        logs = sorted(glob.glob(str(LOGS / "*.log")), key=os.path.getmtime)
        log = logs[-1] if logs else None
    if not log:
        return {"log": None, "lines": []}
    with open(log, errors="replace") as fh:
        lines = fh.read().splitlines()
    lines = [l for l in lines if " DEBUG " not in l and "Gtk-Message" not in l]
    return {"log": os.path.basename(log), "lines": lines[-n:]}


def _history():
    sys.path.insert(0, str(ROOT))
    from secondlook import history
    os.chdir(ROOT)
    return history


def _qs(path):
    from urllib.parse import parse_qs, urlparse
    return {k: v[-1] for k, v in parse_qs(urlparse(path).query).items()}


def scene_state():
    f = SCENE / "latest.json"
    if not f.exists():
        return {"report": None}
    rep = json.loads(f.read_text())
    he = _yaml(ROOT / "secondlook" / "handeye.yaml")
    return {"report": rep, "mtime": os.path.getmtime(f), "age_s": int(time.time() - os.path.getmtime(f)),
            "height": (SCENE / "latest_height.jpg").exists(),
            "calibration": he.get("fit"), "pairs": [{k: p.get(k) for k in ("source", "err_mm", "u", "v", "ts", "rejected")}
                                                     for p in he.get("pairs", [])]}


def storage():
    def size(pattern):
        files = glob.glob(str(ROOT / pattern), recursive=True)
        return len(files), sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    n_ins, b_ins = size("runs/inspections/**/*.png")
    db = ROOT / "runs" / "history.db"
    st = os.statvfs(ROOT)
    return {"photos": n_ins, "photos_mb": round(b_ins / 1e6, 1), "db_mb": round(db.stat().st_size / 1e6, 2) if db.exists() else 0,
            "free_gb": round(st.f_bavail * st.f_frsize / 1e9, 1)}


def learning():
    h = _history()
    out = h.learning_stats()
    cfg = _yaml(ROOT / "secondlook" / "config.yaml")
    ins = cfg.get("inspection", {})
    out["threshold"], out["band"] = ins.get("threshold"), ins.get("band")
    out["calibration"] = _yaml(ROOT / "secondlook" / "handeye.yaml").get("fit")
    st = status()["data"]
    out["data"] = st
    with h.connect() as con:
        unl = con.execute("SELECT COUNT(*) FROM bricks WHERE pick_ok = 1 AND (truth IS NULL OR truth = '')").fetchone()[0]
        wrong = con.execute("SELECT COUNT(*) FROM bricks WHERE truth IN ('good','defective') AND final_bowl IS NOT NULL "
                            "AND truth != final_bowl").fetchone()[0]
    tips = []
    if unl:
        tips.append(f"{unl} inspected brick{'s have' if unl != 1 else ' has'} no true label yet. Label them below: every label becomes a training photo.")
    if wrong:
        tips.append(f"{wrong} brick{'s' if wrong != 1 else ''} went to the wrong bowl (filter History by “Wrong bowl”). "
                    "Retrain once they are labelled.")
    b = out.get("best_threshold")
    cur_cost = None
    if out["threshold"] is not None and (out["good_scores"] or out["defect_scores"]):
        t = out["threshold"]
        cur_cost = sum(1 for x in out["good_scores"] if x >= t) + 2 * sum(1 for x in out["defect_scores"] if x < t)
    if b and cur_cost is not None and b["cost"] < cur_cost and abs(b["thr"] - out["threshold"]) > 0.03:
        tips.append(f"On real robot photos a cut-off of {b['thr']} would get {b['correct']} of {b['n']} right "
                    f"(now {out['threshold']}). Retraining sets it from all labelled photos.")
    fit = out["calibration"]
    if not fit:
        tips.append("Vision pick is not calibrated yet. In a terminal: ./sl table --from-map, then ./sl table --auto.")
    elif fit.get("mean_mm", 0) > 7 or fit.get("sources") == ["manual"]:
        tips.append(f"Vision pick is accurate to about {fit.get('mean_mm')} mm. Let the arm calibrate itself with one brick "
                    "(terminal: ./sl table --auto) to get under 6 mm.")
    for m, c in (out.get("pick_methods") or {}).items():
        tot = sum(c.values())
        if tot >= 5 and c.get("grasped", 0) / tot < 0.7:
            tips.append(f"The {'vision pick' if m == 'ik' else m} grasps {c.get('grasped', 0)} of {tot}. The Picks map on "
                        "the Station shows where it misses.")
    if b and cur_cost is not None and b["cost"] >= cur_cost:
        b["same_as_now"] = True
    if out["good_scores"] and out["policies"]:
        hi = [x for x in out["good_scores"] if out["threshold"] is not None and x >= out["threshold"]]
        if hi:
            tips.append(f"{len(hi)} good brick{'s' if len(hi) != 1 else ''} scored above the cut-off. Most were picked by the "
                        "AI pick, which holds bricks at new angles. Label more of those good bricks, then retrain.")
    out["tips"] = tips
    out.pop("labelled", None)
    out["storage"] = storage()
    return out


# --- live run (proxied, so the dashboard draws it itself) -----------------------------------
def live_state():
    try:
        return json.load(urllib.request.urlopen(RUN_UI + "/state", timeout=1))
    except Exception:
        return None


def live_frame():
    try:
        r = urllib.request.urlopen(RUN_UI + "/frame.jpg", timeout=2)
        return r.read() if r.status == 200 else None
    except Exception:
        return None


def live_cmd(cmd):
    if cmd not in ("start", "good", "defective", "skip", "fix_good", "fix_defect", "stop"):
        return 400, {"error": "unknown command"}
    try:
        req = urllib.request.Request(RUN_UI + "/cmd", json.dumps({"cmd": cmd}).encode(), {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
        return 200, {"ok": True}
    except Exception:
        return 409, {"error": "no run is listening"}


# --- models ---------------------------------------------------------------------------------
def _config_text():
    return (ROOT / "secondlook" / "config.yaml").read_text()


def _write_config(text, tag):
    import shutil
    p = ROOT / "secondlook" / "config.yaml"
    shutil.copy(p, str(p) + f".bak_{time.strftime('%H%M%S')}{tag}")
    p.write_text(text)


def models():
    cfg = _yaml(ROOT / "secondlook" / "config.yaml")
    pol_path = (cfg.get("policy") or {}).get("path") or ""
    names = {}
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{STUDIO_DB}?mode=ro", uri=True, timeout=2)
        for mid, name, policy, created in con.execute("SELECT id, name, policy, created_at FROM models"):
            names[mid] = {"name": name, "policy": policy, "created": created}
        con.close()
    except Exception:
        pass
    learned = []
    active_real = os.path.realpath(ROOT / pol_path) if pol_path else ""
    seen = set()
    # models/<name>/ (downloaded with tools/fetch_models.sh, or symlinks) first, then Studio's own folder
    found = [(d, d.parent.name, str(d.parent.relative_to(ROOT))) for d in sorted((ROOT / "models").glob("*/manifest.json"))]
    found += [(d, d.parts[-4], str(d.parent)) for d in sorted(STUDIO_MODELS.glob("*/exports/openvino/manifest.json"))]
    for d, mid, path in found:
        real = os.path.realpath(d.parent)
        if real in seen:
            continue
        seen.add(real)
        try:
            man = json.loads(d.read_text())
        except ValueError:
            continue
        studio_id = Path(real).parts[-3] if real.startswith(str(STUDIO_MODELS)) else mid
        policy = (man.get("policy") or {}).get("name") or names.get(studio_id, {}).get("policy") or "policy"
        info = names.get(studio_id, {})
        learned.append({"id": mid, "name": info.get("name") or (mid if path.startswith("models/") else policy.upper()),
                        "policy": policy,
                        "created": info.get("created") or time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
                        "path": path, "active": real == active_real})
    engine = picker_engine()
    he = _yaml(ROOT / "secondlook" / "handeye.yaml").get("fit")
    ins = cfg.get("inspection") or {}
    versions = []
    for d in sorted(ROOT.glob("exports_before_recal_*"), reverse=True):
        t = d.name.rsplit("_", 1)[-1]
        bak = ROOT / "secondlook" / f"config.yaml.bak_recal_{t}"
        thr = (_yaml(bak).get("inspection") or {}).get("threshold") if bak.exists() else None
        m = next(iter(d.glob("patchcore/fp16/weights/openvino/model.xml")), None)
        if m:
            versions.append({"id": d.name, "when": time.strftime("%b %d %H:%M", time.localtime(m.stat().st_mtime)),
                             "threshold": thr, "path": str(m.relative_to(ROOT))})
    cur = ROOT / (ins.get("model") or "")
    return {"engine": engine, "learned": learned, "vision": {"calibration": he},
            "defect": {"model": "PatchCore", "path": ins.get("model"), "threshold": ins.get("threshold"),
                       "band": ins.get("band"), "device": ins.get("device"),
                       "trained": time.strftime("%b %d %H:%M", time.localtime(cur.stat().st_mtime)) if cur.exists() else None,
                       "versions": versions},
            "task": (cfg.get("policy") or {}).get("task")}


def select_picker(body):
    """{"engine": "vision"} or {"engine": "policy", "id": <studio model id>}"""
    import re as _re
    text = _config_text()
    engine = body.get("engine")
    if engine not in ("vision", "policy"):
        return 400, {"error": "engine must be vision or policy"}
    if engine == "policy":
        m = next((x for x in models()["learned"] if x["id"] == body.get("id")), None)
        if not m:
            return 404, {"error": "model not found"}
        text = _re.sub(r"(\npolicy:(?:\n[^\n]*)*?\n  path: )\S+", lambda mm: mm[1] + m["path"], text, count=1)
    if _re.search(r"\nstation:\n  picker: \w+", text):
        text = _re.sub(r"(\nstation:\n  picker: )\w+", lambda mm: mm[1] + engine, text, count=1)
    else:
        text = text.rstrip("\n") + f"\n\nstation:\n  picker: {engine}            # vision = depth + arm model; policy = learned model in policy.path\n"
    _write_config(text, "pick")
    return 200, {"ok": True, "engine": engine}


def select_defect(body):
    """Roll the defect model back to an earlier version (its model file and the threshold it used)."""
    import re as _re
    v = next((x for x in models()["defect"]["versions"] if x["id"] == body.get("id")), None)
    if not v or v["threshold"] is None:
        return 404, {"error": "version not found"}
    text = _config_text()
    text = _re.sub(r"(\ninspection:(?:\n[^\n]*)*?\n  model: )\S+", lambda mm: mm[1] + v["path"], text, count=1)
    text = _re.sub(r"(\n  threshold: )[0-9.]+", lambda mm: mm[1] + str(v["threshold"]), text, count=1)
    _write_config(text, "defect")
    return 200, {"ok": True}


# --- voice ----------------------------------------------------------------------------------
def _announce():
    sys.path.insert(0, str(ROOT))
    from secondlook import announce
    return announce


def describe():
    """Say what is on the table. During a run the camera is busy, so use the run's latest scan."""
    a = _announce()
    st = _job_state()
    if st["alive"] or st["external"]:
        f = SCENE / "latest.json"
        if not f.exists():
            return 409, {"error": "no table scan yet in this run"}
        rep = json.loads(f.read_text())
        text = a.describe_table(rep.get("brick_list") or [], len(rep.get("obstacles") or []))
        a.say(text, force=True)
        return 200, {"ok": True, "text": text}
    return start("describe")


def trends():
    """Per-run numbers for the charts (newest last)."""
    h = _history()
    with h.connect() as con:
        rows = [dict(r) for r in con.execute(
            "SELECT run, MIN(ts) AS started, policy, COUNT(*) AS attempts, SUM(pick_ok) AS picked, "
            "SUM(CASE WHEN pick_ok=1 AND final_bowl='good' THEN 1 ELSE 0 END) AS good, "
            "SUM(CASE WHEN pick_ok=1 AND final_bowl='defective' THEN 1 ELSE 0 END) AS defective, "
            "SUM(CASE WHEN truth IN ('good','defective') AND pick_ok=1 THEN 1 ELSE 0 END) AS labelled, "
            "SUM(CASE WHEN truth IN ('good','defective') AND pick_ok=1 AND truth=final_bowl THEN 1 ELSE 0 END) AS correct "
            "FROM bricks GROUP BY run ORDER BY started")]
        cyc = {}
        for r in con.execute("SELECT run, cycle_s FROM bricks WHERE pick_ok=1 AND cycle_s IS NOT NULL"):
            cyc.setdefault(r["run"], []).append(r["cycle_s"])
    for r in rows:
        c = sorted(cyc.get(r["run"], []))
        r["cycle_s"] = round(c[len(c) // 2], 1) if c else None
    return rows


def _backfill_once():
    try:
        n = _history().backfill(str(ROOT))
        if n:
            print(f"history: imported {n} old run rows", flush=True)
    except Exception as e:
        print("history backfill failed:", e, flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                                   # the browser went away mid-reply: nothing to do

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/access.js":
                return self._send(200, PAGE.with_name("access.js").read_bytes(), "text/javascript; charset=utf-8")
            if path == "/api/aac":
                return self._send(*_announce().access_api("GET", path))
            if path == "/api/status":
                return self._send(200, status())
            if path == "/api/results":
                return self._send(200, results())
            if path == "/api/inspections":
                return self._send(200, inspections())
            if path == "/api/log":
                return self._send(200, log_tail())
            if path == "/api/modes":
                return self._send(200, LABELS)
            if path == "/api/history":
                q = _qs(self.path)
                return self._send(200, _history().bricks(limit=int(q.get("limit", 150)), run=q.get("run") or None,
                                                         verdict=q.get("verdict") or None, truth=q.get("truth") or None,
                                                         only_picked=q.get("picked") == "1"))
            if path == "/api/history/runs":
                return self._send(200, _history().runs())
            if path == "/api/picks":
                return self._send(200, _history().picks(limit=int(_qs(self.path).get("limit", 400))))
            if path == "/api/learning":
                return self._send(200, learning())
            if path == "/api/scene":
                return self._send(200, scene_state())
            if path == "/api/live":
                return self._send(200, live_state() or {})
            if path == "/live/frame.jpg":
                f = live_frame()
                return self._send(200, f, "image/jpeg") if f else self._send(204, b"", "image/jpeg")
            if path == "/api/models":
                return self._send(200, models())
            if path == "/api/voice":
                a = _announce()
                return self._send(200, dict(a.settings(), engine=a.engine()))
            if path == "/api/trends":
                return self._send(200, trends())
            if path.startswith("/scene/"):
                name = os.path.basename(path)
                f = SCENE / name
                if name in ("latest.jpg", "latest_raw.jpg", "latest_height.jpg") and f.exists():
                    return self._send(200, f.read_bytes(), "image/jpeg")
            if path.startswith("/img/"):
                sub = "full" if path.startswith("/img/full/") else ""
                name = os.path.basename(path)
                f = ROOT / "runs" / "inspections" / sub / name
                if re.fullmatch(r"[\w.-]+\.png", name) and f.exists():
                    return self._send(200, f.read_bytes(), "image/png")
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        if self.path == "/api/start":
            code, out = start(body.get("mode"), body.get("name"))
        elif self.path == "/api/stop":
            code, out = stop()
        elif self.path == "/api/live/cmd":
            code, out = live_cmd(body.get("cmd"))
        elif self.path == "/api/models/picker":
            code, out = select_picker(body)
        elif self.path == "/api/models/defect":
            code, out = select_defect(body)
        elif self.path in ("/api/voice", "/api/say"):
            code, out = _announce().access_api("POST", self.path, body if isinstance(body, dict) else {})
        elif self.path == "/api/describe":
            code, out = describe()
        elif self.path == "/api/history/label":
            try:
                truth = body.get("truth") or None
                code, out = 200, {"ok": True, "msg": _history().set_truth(int(body.get("id")), truth, root=str(ROOT))}
            except (KeyError, ValueError, TypeError) as e:
                code, out = 400, {"error": str(e)}
        else:
            code, out = 404, {"error": "not found"}
        self._send(code, out)


def main(port=8770):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=_backfill_once, daemon=True).start()
    print(f"SecondLook control center on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8770)
