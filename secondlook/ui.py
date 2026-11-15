"""Operator interface. Every result goes to the screen and is spoken aloud.

mode "web"    (default) dashboard in the browser at http://127.0.0.1:8765, clickable buttons
mode "window" plain OpenCV window, keyboard only
Keys / buttons: Space = start, S or Esc = STOP, G / D / X = true label during --eval.
"""
import time
import webbrowser

from .announce import announce_result, say
from .machine import Stop
from .power import Power

NONE_KEY = 255
COMMAND_KEYS = {"start": ord(" "), "good": ord("g"), "defective": ord("d"), "skip": ord("x")}
FIX = {"fix_good": "good", "fix_defect": "defective"}


def friendly(stage):
    """Operator-facing words for the cycle's stage codes (the codes stay in the log only)."""
    import re
    s = (stage or "").strip()
    head, _, rest = s.partition("  ")
    head, rest = head.strip().upper(), rest.strip()
    if head == "PICK":
        m = re.search(r"(\d+) on the table", rest)
        if m:
            return f"Picking a brick ({m[1]} on the table)"
        return "Picking up a brick" + (" (second try)" if "try 2" in rest else "")
    if head == "INSPECT":
        return "Second look" if "view 2" in rest else "Inspecting"
    if head == "PLACE":
        return "Into the good bowl" if "good" in rest else "Into the reject bowl"
    table = {"DONE": None, "DROPPED": "The brick slipped. Picking it again", "NO BRICK": "No brick picked up",
             "TABLE CLEAR": "The table is clear", "TABLE": "The table is clear", "READY": "Ready",
             "EVAL": "Waiting for your label", "OUT OF REACH": "Some bricks are out of reach: move them closer",
             "PLACE A BRICK IN THE GRIPPER, THEN PRESS SPACE": "Hand a brick to the gripper",
             "NO BRICK IN THE GRIPPER": "The gripper is empty: hand it a brick",
             "LOOKING": "Looking at the table", "SORTED": "Table sorted", "REVIEW": "Reviewing photos"}
    if s.upper().startswith("PLACE A BRICK"):
        return table["PLACE A BRICK IN THE GRIPPER, THEN PRESS SPACE"]
    if s.upper().startswith("NO BRICK IN THE GRIPPER"):
        return table["NO BRICK IN THE GRIPPER"]
    if head in table:
        return table[head]
    return s[:1].upper() + s[1:].lower() if s.isupper() else s


class UI:
    def __init__(self, cfg, inspector=None, picker=None):
        u = cfg["ui"]
        self.mode = u.get("mode", "web")
        self.voice = u.get("voice", True)
        self.power = Power(cfg["power"]["idle_watts"])
        self.inspector, self.picker = inspector, picker
        self.bowls = cfg.get("bowls") or {"good": "pink", "defective": "blue"}
        ins = cfg["inspection"]
        self.state = {"thr": ins["threshold"], "band": ins["band"], "counts": {"good": 0, "defective": 0},
                      "anomaly_dev": ins["device"], "policy_dev": cfg["policy"]["device"], "stage": "Starting"}
        self.views, self.history, self.events = [], [], []
        self.waited = 0.0             # seconds spent waiting for the operator (not counted as cycle time)
        self._new_views = False
        self.on_label = None          # callback(truth) from the app: file the last photos as a labelled example
        self.state["learn"] = None
        self.web = self.hud = None
        if self.mode == "web":
            from .web import WebServer
            self.web = WebServer(u.get("host", "127.0.0.1"), u.get("port", 8765)).start()
            print("Dashboard:", self.web.url)
            import threading
            self._alive = True
            threading.Thread(target=self._heartbeat, daemon=True).start()
            import os as _os
            if u.get("open_browser", True) and not _os.environ.get("SECONDLOOK_NO_BROWSER"):
                webbrowser.open(self.web.url)
        else:
            from .hud import Hud
            self.hud = Hud(fullscreen=u.get("fullscreen", True))

    def _heartbeat(self):
        """Keep the page live while the main loop is busy (moving, retraining): publish twice a second."""
        while getattr(self, "_alive", False):
            try:
                if getattr(self, "on_idle", None):
                    self.on_idle()
                self.web.publish(self._public_state())
            except Exception:
                pass
            time.sleep(0.5)

    def _public_state(self):
        st = dict(self.state, history=self.history[:12], events=self.events[:30], clock=time.strftime("%H:%M:%S"))
        g = getattr(self.picker, "last_grip", None)
        if g is not None:
            st["grip"] = round(g, 1)
            st["held"] = getattr(self.picker, "last_held", None)
        st["attempt"] = getattr(self.picker, "attempt", None)
        found = getattr(self.picker, "found", None)
        if isinstance(found, int):
            st["on_table"] = found
            st["table_colors"] = getattr(self.picker, "colors", None)
        st["bowls"] = self.bowls
        return st

    def event(self, text, level="info"):
        self.events.insert(0, {"t": time.strftime("%H:%M:%S"), "text": text, "level": level})
        del self.events[30:]

    def _frame_jpeg(self):
        import cv2
        import numpy as np
        from .hud import overlay_heatmap
        vmax = max(1.0, 2 * self.state["thr"])
        tiles = []
        for img, heat, _ in self.views[-2:]:
            t = overlay_heatmap(img, heat, vmax)
            tiles.append(cv2.resize(t, (int(t.shape[1] * 480 / t.shape[0]), 480)))
        ok, buf = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def tick(self):
        """Refresh the display and telemetry; raise Stop on STOP. Returns the key/button pressed."""
        self.power.sample()
        self.state.update(watts=self.power.above_idle(), temp_c=self.power.temp_c())
        if self.inspector is not None and self.inspector.last_ms is not None:
            self.state["anomaly_ms"] = self.inspector.last_ms
        if getattr(self.picker, "last_ms", None) is not None:
            self.state["policy_ms"] = self.picker.last_ms

        if self.web is not None:
            jpeg = None
            if self._new_views and self.views:
                jpeg = self._frame_jpeg()
                self._new_views = False
            now = time.time()
            if jpeg is not None or getattr(self, "_dirty", True) or now - getattr(self, "_last_pub", 0) > 0.2:
                self.web.publish(self._public_state(), jpeg)
                self._last_pub, self._dirty = now, False
            if self.web.stop_requested.is_set():
                self.web.stop_requested.clear()
                self.event("STOP pressed: motors off", "error")
                raise Stop()
            cmd = self.web.next_command()
            if cmd == "stop":
                raise Stop()
            if cmd in FIX:
                truth = FIX[cmd]
                if self.on_label:
                    msg = self.on_label(truth)
                    self.event(f"Corrected: last brick was {truth.upper()}. {msg}", "warn")
                    self.say_text(f"Thanks. Saved as {'good' if truth == 'good' else 'defective'}.")
                else:
                    self.event("Correction ignored: no run in progress", "warn")
                return NONE_KEY
            return COMMAND_KEYS.get(cmd, NONE_KEY)

        key = self.hud.show(self.views, self.state)
        if key in (ord("s"), 27):
            raise Stop()
        return key

    def check_stop(self):
        self.tick()

    def show(self, **kw):
        views = kw.pop("views", None)
        if views is not None:
            self.views = [v for v in views if v[0] is not None]
            self._new_views = True
        self._dirty = True                    # real change: publish now; idle loop ticks are throttled
        old_stage = self.state.get("stage")
        self.state.update({k: v for k, v in kw.items() if v is not None or k == "verdict"})
        if kw.get("stage") and kw["stage"] != old_stage:
            lvl = "error" if "STOP" in kw["stage"] else "info"
            text = friendly(kw["stage"])
            if text:
                self.event(text, lvl)
            self.state["activity"] = text or self.state.get("activity")
        if kw.get("verdict") and kw.get("score") is not None and kw.get("stage", "").startswith("INSPECT"):
            words = {"good": "Looks good", "defective": "Defect found", "uncertain": "Not sure"}.get(kw["verdict"], kw["verdict"])
            self.event(f"{words} (score {kw['score']:.2f})",
                       "ok" if kw["verdict"] == "good" else "error" if kw["verdict"] == "defective" else "warn")
        if kw.get("stage") == "DONE" and self.state.get("verdict"):
            self.history.insert(0, {"verdict": self.state["verdict"], "score": self.state.get("score"),
                                    "cycle_s": self.state.get("cycle_s"), "t": time.strftime("%H:%M:%S")})
        self.tick()

    def wait_key(self, keys):
        t0 = time.time()
        try:
            while True:
                k = self.tick()
                if k in keys:
                    return k
                if self.web is not None:
                    time.sleep(0.03)
        finally:
            self.waited = getattr(self, "waited", 0.0) + (time.time() - t0)

    def caption(self, text, kind="info"):
        """On-screen copy of what the robot says (deaf / hard-of-hearing view), shown even with voice off."""
        self._cap_n = getattr(self, "_cap_n", 0) + 1
        self.show(caption={"n": self._cap_n, "text": text, "kind": kind, "t": time.strftime("%H:%M:%S")})

    def say(self, key, count=None, defect=None, color=None):
        """defect/color: Gemini's look at the brick (AccessInspect), used only for the wording."""
        from .announce import PHRASES, n
        if key in ("good", "defective"):
            found = getattr(self.picker, "found", None)
            left = max(0, found - 1) if isinstance(found, int) and self.state.get("until_clear") else None
            extra = []
            if self.bowls.get(key):
                extra.append(f"The {self.bowls[key]} bowl.")
            if left is not None:
                extra.append("That was the last one." if left <= 0 else f"{n(left).capitalize()} left.")
            shown = ("Good brick" if key == "good" else "Defect found") + (f": {defect}" if defect else "")
            shown += (f" ({color})" if color else "") + ". " + " ".join(extra)
            text = shown.strip()
            self.caption(text, key)
            if self.voice:
                say(text)
        else:
            text = PHRASES.get(key, key)
            self.caption(text, "warn" if key in ("missed", "uncertain", "stopped") else "info")
            if self.voice:
                say(text)

    def say_text(self, text):
        self.caption(text)
        if self.voice:
            say(text)

    def close(self):
        self._alive = False
        if self.web is not None:
            self.web.stop()
        if self.hud is not None:
            self.hud.close()
