"""Station monitor: continuous background logging + live camera view for the SO-101 station.

Passive only: it never opens a serial port or a camera device. Cameras are read from the
shared streams Physical AI Studio already publishes; when Studio is not streaming a camera,
the page says so instead of grabbing the device.

Run (from the Studio backend folder so the right Python is used):
  cd ~/physical-ai-studio/application/backend
  uv run --extra xpu python ~/station_monitor/monitor.py
Page:   http://127.0.0.1:8090
Logs:   ~/station_monitor/events.jsonl  (one JSON event per line)
        ~/station_monitor/events.log    (human readable)
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path.home()
OUT = HOME / "station_monitor"
OUT.mkdir(exist_ok=True)
JSONL, TEXT = OUT / "events.jsonl", OUT / "events.log"
STUDIO_LOG = HOME / ".local/share/physicalai/logs/app.log"
STUDIO_API = "http://127.0.0.1:7860/api"
PORT = 8090

ARMS = {"5AE6082572": "follower arm", "5B91043227": "leader arm"}
CAMERAS = {"top": "310423022922", "side": "344223023487", "wrist": "USB2.0_CAM1"}
USB_NAMES = {"3-1": "wrist camera", "2-4.4.2": "top RealSense", "2-4.4.3": "side RealSense"}

# issue key -> (what, why, how to prevent)
RULES = {
    "wrist_usb": ("Wrist camera disconnected",
                  "USB cable at the moving wrist is pulled, loose, or the camera browns out",
                  "Give the cable slack loops at each joint, tape it along the arm, use a short good cable or a powered hub"),
    "realsense_usb": ("RealSense camera disconnected",
                      "Stacked USB hub or weak port (EPROTO -71 seen earlier on 2-4.4.x)",
                      "Plug the RealSense directly into the laptop USB-C/3 port or a powered USB3 hub"),
    "arm_usb": ("Robot arm USB disconnected",
                "Arm USB cable replugged or loose; ttyACM numbers can swap after replug",
                "Secure the cable; always select arms by serial (/dev/serial/by-id), never ttyACM0/1"),
    "multi_access": ("Two programs used the same arm port",
                     "Studio session and SecondLook (or a probe) talked to the servos at the same time",
                     "Close Studio Run-model/teleop/live tabs before running SecondLook, and quit SecondLook before Studio"),
    "voltage": ("Servo voltage error",
                "Supply dipped below 4 V under load, or over-voltage (12 V on 8 V-max servos)",
                "Use the follower's 5 V (max 7.4 V), >=5 A adapter; never 12 V; avoid pressing the gripper into the table"),
    "camera_missing": ("Studio could not open a camera",
                       "Camera held by another program, or its index/bus changed after a replug",
                       "Release it from the other program; re-select the camera in Studio if it moved ports"),
    "calibration": ("Robot calibration missing for a joint",
                    "Calibration was wiped or the robot was re-added without calibrating",
                    "Recalibrate the arm in Studio (Robots -> calibrate)"),
    "deadline": ("Control loop missed its 30 Hz deadline",
                 "Laptop busy (training/export/browser) or a slow camera frame",
                 "Close heavy apps while running the robot; occasional single misses are harmless"),
    "port_conflict": ("Hardware held by more than one process",
                      "Two programs opened the same arm/camera",
                      "Only one program may own the arm and cameras at a time"),
    "holder_change": ("Hardware owner changed", "A program started or stopped using it", "Informational"),
    "job": ("Training job state changed", "Remote training progress", "Informational"),
    "studio_error": ("Studio reported an error", "See message", "Check the message; ask Claude with the event id"),
}

_events, _lock = [], threading.Lock()


def emit(key, where, detail, level="warn"):
    what, why, fix = RULES.get(key, (key, "", ""))
    ev = {"id": len(_events) + 1, "time": datetime.now().strftime("%H:%M:%S"), "level": level,
          "issue": what, "where": where, "detail": detail[:300], "why": why, "prevent": fix}
    with _lock:
        _events.append(ev)
        del _events[:-300]
    with open(JSONL, "a") as f:
        f.write(json.dumps(ev) + "\n")
    with open(TEXT, "a") as f:
        f.write(f"[{ev['time']}] {level.upper():5} {what} @ {where}: {ev['detail']}\n"
                f"           why: {why}\n           prevent: {fix}\n")


# --- watchers ------------------------------------------------------------------------------
def watch_kernel():
    p = subprocess.Popen(["journalctl", "-k", "-f", "-n", "0", "-o", "short-iso"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in p.stdout:
        m = re.search(r"usb (\d+-[\d.]+): USB disconnect", line)
        if m:
            bus = m.group(1)
            if bus == "3-1":
                emit("wrist_usb", "USB 3-1 (wrist camera)", line.strip())
            elif bus in ("2-4.4.2", "2-4.4.3"):
                emit("realsense_usb", f"USB {bus} ({USB_NAMES[bus]})", line.strip())
            else:
                emit("arm_usb" if bus.startswith("3-2") else "holder_change", f"USB {bus}", line.strip(),
                     "warn" if bus.startswith("3-2") else "info")
        elif "error -71" in line or "EPROTO" in line:
            emit("realsense_usb", "USB protocol error", line.strip())


def follow(path):
    f = open(path, errors="replace")
    f.seek(0, 2)
    ino = os.fstat(f.fileno()).st_ino
    while True:
        line = f.readline()
        if line:
            yield line
            continue
        time.sleep(0.5)
        try:
            if os.stat(path).st_ino != ino:          # log rotated
                f = open(path, errors="replace")
                ino = os.fstat(f.fileno()).st_ino
        except FileNotFoundError:
            pass


def watch_studio_log():
    while not STUDIO_LOG.exists():
        time.sleep(5)
    for line in follow(STUDIO_LOG):
        try:
            r = json.loads(line)["record"]
        except Exception:
            continue
        msg, lvl = r["message"], r["level"]["name"]
        if lvl not in ("WARNING", "ERROR", "CRITICAL"):
            continue
        where = f"Studio {r.get('name', '')}"
        low = msg.lower()
        if "multiple access" in low or "incorrect status packet" in low:
            emit("multi_access", where, msg)
        elif "voltage" in low:
            emit("voltage", where, msg)
        elif "no camera found" in low or "failed to start camera publisher" in low:
            emit("camera_missing", where, msg)
        elif "keyerror" in low or re.search(r"'(shoulder_pan|gripper|elbow_flex)'", msg):
            emit("calibration", where, msg)
        elif "missed" in low and "deadline" in low:
            emit("deadline", where, msg, "info")
        elif lvl in ("ERROR", "CRITICAL"):
            emit("studio_error", where, msg, "error")


def _holders(dev):
    try:
        out = subprocess.run(["fuser", dev], capture_output=True, text=True, timeout=5)
        return sorted(set(re.findall(r"\d+", out.stdout)))
    except Exception:
        return []


def _name(pid):
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except OSError:
        return "gone"
    if "secondlook" in cmd:
        return "SecondLook"
    if "_publisher_worker" in cmd:
        return "Studio camera publisher"
    if "physicalai-studio" in cmd:
        return "Studio backend"
    if "spawn_main" in cmd:
        return "Studio worker"
    return cmd.split()[0].rsplit("/", 1)[-1] if cmd else "?"


def _camera_nodes():
    nodes = {}
    for v in glob.glob("/sys/class/video4linux/video*"):
        try:
            serial = (Path(v).joinpath("device").resolve().parent / "serial").read_text().strip()
        except OSError:
            continue
        for cam, s in CAMERAS.items():
            if s == serial:
                nodes.setdefault(cam, []).append("/dev/" + os.path.basename(v))
    return nodes


STATE = {"owners": {}, "jobs": {}, "updated": ""}


def watch_owners():
    last = {}
    while True:
        owners = {}
        for link in glob.glob("/dev/serial/by-id/*"):
            arm = next((n for s, n in ARMS.items() if s in link), None)
            if arm:
                pids = _holders(os.path.realpath(link))
                owners[arm] = [f"{_name(p)} ({p})" for p in pids]
        for cam, devs in _camera_nodes().items():
            pids = sorted({p for d in devs for p in _holders(d)})
            owners[cam + " camera"] = [f"{_name(p)} ({p})" for p in pids]
        for k, v in owners.items():
            kinds = {x.split(" (")[0] for x in v}
            if len(v) > 1 and not kinds <= {"Studio worker", "Studio backend", "Studio camera publisher"}:
                emit("port_conflict", k, ", ".join(v), "error")
            if last.get(k) != v and k in last:
                emit("holder_change", k, f"{', '.join(last[k]) or 'free'} -> {', '.join(v) or 'free'}", "info")
        last = owners
        STATE["owners"], STATE["updated"] = owners, datetime.now().strftime("%H:%M:%S")
        time.sleep(5)


def watch_jobs():
    last = {}
    while True:
        try:
            jobs = json.load(urllib.request.urlopen(STUDIO_API + "/jobs", timeout=10))
            cur = {j["id"][:8]: (j["payload"].get("model_name", j["type"]), j["status"], j["progress"])
                   for j in jobs if j["status"] in ("running", "pending") or j["id"][:8] in last}
            for k, (name, st, pr) in cur.items():
                if k in last and last[k][1] != st:
                    emit("job", name, f"{last[k][1]} -> {st}", "error" if st == "failed" else "info")
            last = cur
            STATE["jobs"] = {f"{n} ({k})": f"{st} {pr}%" for k, (n, st, pr) in cur.items()}
        except Exception as e:
            STATE["jobs"] = {"Studio API": f"unreachable ({type(e).__name__})"}
        time.sleep(30)


# --- live cameras (shared streams only) -------------------------------------------------------
_frames = {c: (None, "waiting") for c in CAMERAS}


def _services():
    import iceoryx2 as iox2
    out = set()
    for s in iox2.Service.list(iox2.config.global_config(), iox2.ServiceType.Ipc):
        n = s.name()
        out.add(n.to_string() if hasattr(n, "to_string") else str(n))
    return out


def grab_cameras():
    import cv2
    from physicalai.capture.transport import SharedCamera
    while True:
        # Only watch while a Studio session drives the arm. Reading a stream otherwise keeps Studio's
        # publisher (and the camera) busy, which blocks SecondLook from opening it.
        arm = STATE["owners"].get("follower arm", [])
        if not any(o.startswith("Studio") for o in arm):
            for cam in CAMERAS:
                _frames[cam] = (None, "paused: no Studio session (camera left free for SecondLook)")
            time.sleep(1.0)
            continue
        try:
            services = _services()
        except Exception:
            services = set()
        nodes = _camera_nodes()
        for cam, devs in nodes.items():
            idx = [d.removeprefix("/dev/video") for d in devs]
            svc = next((s for s in services for i in idx if s.endswith(f"/{i}/frame") and "/camera/" in s), None)
            if svc is None:
                _frames[cam] = (None, "not shared by Studio right now (idle, or used by SecondLook)")
                continue
            try:
                c = SharedCamera.from_publisher(svc)
                c.connect(timeout=2.0)
                try:
                    rgb = c.read(timeout=1.0).data
                finally:
                    c.disconnect()          # never keep Studio's publisher alive on our own
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                _frames[cam] = (jpg.tobytes(), f"live {rgb.shape[1]}x{rgb.shape[0]} {datetime.now():%H:%M:%S}")
            except Exception as e:
                _frames[cam] = (None, f"read failed: {type(e).__name__}")
        for cam in CAMERAS:
            if cam not in nodes:
                _frames[cam] = (None, "camera not connected (USB)")
        time.sleep(0.2)


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Station Monitor</title><style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e8eaf0;--mut:#9aa3b2;--warn:#f5a524;--err:#f31260;--ok:#17c964}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
header{padding:12px 16px;font-size:18px;font-weight:600}
.cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;padding:0 16px}
.card{background:var(--card);border-radius:10px;padding:10px}
.card img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#000;border-radius:6px}
.mut{color:var(--mut);font-size:12px}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px 16px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:5px;border-bottom:1px solid #262a33;vertical-align:top}
.warn{color:var(--warn)}.error{color:var(--err)}.info{color:var(--mut)}
@media(max-width:800px){.row{grid-template-columns:1fr}}
</style></head><body><header>Station Monitor <span class=mut id=upd></span></header>
<div class=cams>
<div class=card><b>Top</b> <span class=mut id=s_top></span><img id=i_top></div>
<div class=card><b>Side</b> <span class=mut id=s_side></span><img id=i_side></div>
<div class=card><b>Wrist</b> <span class=mut id=s_wrist></span><img id=i_wrist></div></div>
<div class=row><div class=card><b>Who holds the hardware</b><table id=own></table><br><b>Training jobs</b><table id=jobs></table></div>
<div class=card><b>Issues (newest first)</b><table id=ev></table></div></div>
<script>
const cams=['top','side','wrist'];
function tick(){cams.forEach(c=>{document.getElementById('i_'+c).src='/frame/'+c+'?t='+Date.now()});}
async function state(){const s=await (await fetch('/state')).json();
 document.getElementById('upd').textContent='updated '+s.updated;
 cams.forEach(c=>document.getElementById('s_'+c).textContent=s.cams[c]);
 document.getElementById('own').innerHTML=Object.entries(s.owners).map(([k,v])=>`<tr><td>${k}</td><td>${v.length?v.join('<br>'):'<span class=mut>free</span>'}</td></tr>`).join('');
 document.getElementById('jobs').innerHTML=Object.entries(s.jobs).map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
 document.getElementById('ev').innerHTML=s.events.map(e=>`<tr class=${e.level}><td>${e.time}</td><td><b>${e.issue}</b><br><span class=mut>${e.where}</span><br>${e.detail}<br><span class=mut>why: ${e.why}<br>prevent: ${e.prevent}</span></td></tr>`).join('')||'<tr><td class=mut>no issues yet</td></tr>';}
setInterval(tick,400);setInterval(state,2000);tick();state();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path.startswith("/frame/"):
            cam = self.path.split("/")[2].split("?")[0]
            jpg = _frames.get(cam, (None, ""))[0]
            if jpg:
                return self._send(200, jpg, "image/jpeg")
            return self._send(204, b"", "image/jpeg")
        if self.path.startswith("/state"):
            with _lock:
                ev = list(reversed(_events[-60:]))
            body = {"owners": STATE["owners"], "jobs": STATE["jobs"], "updated": STATE["updated"],
                    "cams": {c: _frames[c][1] for c in CAMERAS}, "events": ev}
            return self._send(200, json.dumps(body).encode(), "application/json")
        self._send(404, b"", "text/plain")


def main():
    emit("holder_change", "monitor", "station monitor started", "info")
    # Live camera reading is OFF: subscribing to Studio's shared camera streams from a second
    # process kept stale streams alive and blocked Studio's cameras (2026-11-14). Use Studio's own view.
    for cam in CAMERAS:
        _frames[cam] = (None, "camera view disabled here: use Studio's live view")
    for fn in (watch_kernel, watch_studio_log, watch_owners, watch_jobs):
        threading.Thread(target=fn, daemon=True, name=fn.__name__).start()
    print(f"Station monitor on http://127.0.0.1:{PORT}  | logs: {TEXT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
