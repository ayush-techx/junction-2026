"""The inspection cycle as plain logic. No hardware imports, so it is unit-testable.

One cycle = one brick:
    HOME -> PICK (policy or hand-fed) -> [missed? retry] -> INSPECT
         -> [uncertain? put back, re-pick, re-inspect = the second look] -> PLACE -> HOME

Collaborators (duck-typed, see robot.py / policy.py / inspection.py / ui.py):
    robot.go(pose_name)            move to a taught pose
    robot.gripper(open: bool)      open or close the gripper
    robot.holding() -> bool        gripper closed on something (not fully closed)
    robot.frame(camera) -> image   latest frame from a camera
    picker.pick(robot, ui) -> str  "grasped" | "missed" | "timeout"
    inspector.score(image) -> (score, heatmap, crop)
    ui.show(**state) / ui.say(key) / ui.check_stop()
    advisor(photo_path) -> pending  optional second opinion (AccessInspect Gemini): pending.get(t) -> dict
                                   with defect/color/confident. Used for the spoken line and the log only;
                                   PatchCore alone decides the bowl.
"""
import os
import time

from .decide import final_bowl, verdict


class Stop(Exception):
    """Raised by ui.check_stop() when the operator presses STOP."""


class Cell:
    def __init__(self, robot, picker, inspector, ui, cfg):
        self.robot, self.picker, self.inspector, self.ui = robot, picker, inspector, ui
        ins = cfg["inspection"]
        self.cam, self.thr, self.band = ins["camera"], ins["threshold"], ins["band"]
        self.pick_retries = cfg["cycle"]["pick_retries"]
        self.second_looks = cfg["cycle"]["second_looks"]
        self.counts = {"good": 0, "defective": 0}
        self.empty = False
        self.save_dir = cfg.get("inspection_log", os.path.join(cfg.get("log_dir", "runs"), "inspections"))
        self.last_files = []                 # inspection photos of the current cycle
        self.advisor = None                  # set by cell.py when AccessInspect's Gemini check is on
        self.advisor_wait = 3.0              # extra seconds to wait for it after placing
        self.last_opinion = None

    def _pick(self):
        """Pick with retries. Returns True when holding a brick."""
        for attempt in range(self.pick_retries + 1):
            self.ui.check_stop()
            self.ui.show(stage=f"PICK  (try {attempt + 1})")
            result = self.picker.pick(self.robot, self.ui)
            if result == "empty":                # table scan: nothing left to pick
                self.empty = True
                return False
            if result == "blocked":              # bricks remain, but no reachable pick exists
                if hasattr(self.ui, "say_text"):
                    self.ui.say_text("Bricks remain out of reach. Move them closer to the robot.")
                return False
            if result == "grasped" and self.robot.holding():
                return True
            self.ui.say("missed")
            self.robot.gripper(open=True)
            self.robot.go("home")
        return False

    def _inspect(self):
        self.ui.check_stop()
        self.robot.go("inspect")
        time.sleep(0.3)                      # let the arm settle; avoids motion blur
        if not self.robot.holding():         # the brick slipped out on the way: never inspect an empty gripper
            self._note_grip(False)
            return None
        self._note_grip(True)
        frame = self.robot.frame(self.cam)
        score, heat, crop = self.inspector.score(frame)
        v = verdict(score, self.thr, self.band)
        self._save_inspection(frame, crop, score, v)
        return score, heat, crop, v

    def _note_grip(self, held):
        try:
            g = float(self.robot.observe()["gripper.pos"]) if hasattr(self.robot, "observe") else None
        except Exception:
            g = None
        if g is not None:
            self.picker.last_grip = g
        self.picker.last_held = 1 if held else 0

    def _pick_and_inspect(self, label):
        """Pick, then inspect; if the brick slipped before the photo, say so and pick again."""
        for attempt in range(self.pick_retries + 1):
            if not self._pick():
                return None
            out = self._inspect()
            if out is not None:
                return out
            self.dropped += 1
            if hasattr(self.picker, "note_slip"):
                self.picker.note_slip()
            if hasattr(self.ui, "event"):
                self.ui.event("Dropped the brick on the way to inspection: picking again", "warn")
            self.ui.say("missed")
            self.ui.show(stage=f"DROPPED  {label}: brick slipped, trying again")
            self.robot.gripper(open=True)
            self.robot.go("home")
        return None

    def _save_inspection(self, frame, crop, score, v):
        """Continuous log: every inspection photo, so false alarms can be reviewed and learned from."""
        if not self.save_dir or frame is None or not hasattr(frame, "shape"):
            return
        import cv2
        os.makedirs(os.path.join(self.save_dir, "full"), exist_ok=True)
        # Date + milliseconds: unique across days and for two views in the same second.
        now = time.time()
        name = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{int(now * 1000) % 1000:03d}_{score:.3f}_{v}.png"
        cv2.imwrite(os.path.join(self.save_dir, "full", name), frame)
        if crop is not None and hasattr(crop, "shape"):
            cv2.imwrite(os.path.join(self.save_dir, name), crop)
        self.last_files.append(name)

    def _ask_advisor(self):
        """Start the Gemini look at the last photo of this brick (crop = what PatchCore saw)."""
        self.last_opinion = None
        if self.advisor is None or not self.last_files or not self.save_dir:
            return None
        name = self.last_files[-1]
        path = os.path.join(self.save_dir, name)
        if not os.path.exists(path):
            path = os.path.join(self.save_dir, "full", name)
        try:
            return self.advisor(path)
        except Exception as e:
            print("second opinion not started:", e, flush=True)
            return None

    def _opinion(self, pending, bowl):
        """Gemini's answer (if back in time) -> words for the voice + a note for the log."""
        if pending is None:
            return None
        try:
            r = pending.get(self.advisor_wait)
        except Exception:
            r = None
        if not r or "defect" not in r:
            return {"note": "gemini=late" if r is None else "gemini=error"}
        defect = str(r.get("defect") or "none")
        color = str(r.get("color") or "").lower()
        gem_bowl = "good" if defect == "none" else "defective"
        op = {"defect": defect, "color": color, "confident": bool(r.get("confident")),
              "agrees": gem_bowl == bowl, "reason": r.get("reason", ""),
              # speak the defect type only when PatchCore also rejected it; colour whenever known
              "say_defect": defect if bowl == "defective" and defect != "none" else None,
              "say_color": color if color and color != "unknown" else None}
        op["note"] = f"gemini={defect}/{color or '?'}" + ("" if op["agrees"] else " gemini_disagrees")
        self.last_opinion = op
        if not op["agrees"] and hasattr(self.ui, "event"):
            what = f"sees {defect}" if defect != "none" else "sees no defect"
            self.ui.event(f"Second opinion (Gemini) {what} on this {bowl} brick: {op['reason']} "
                          f"Check it and press It was GOOD / It was DEFECTIVE if PatchCore was wrong.", "warn")
        return op

    def cycle(self):
        """Run one brick. Returns a result dict for logging (None if no brick picked)."""
        t0 = time.time()
        w0 = getattr(self.ui, "waited", 0.0)          # time spent waiting for the operator is not cycle time
        regrasps = 0
        self.empty = False
        self.last_files = []
        self.dropped = 0
        self.robot.go("home")
        first = self._pick_and_inspect("view 1")
        if first is None:
            self.ui.show(stage="NO BRICK", verdict=None)
            return None

        views = []
        score, heat, crop, v = first
        views.append((crop, heat, score))
        self.ui.show(stage="INSPECT  view 1", verdict=v, score=score, views=views)

        # Second look: an uncertain score triggers put-back + re-grasp + re-inspect.
        while v == "uncertain" and regrasps < self.second_looks:
            regrasps += 1
            self.ui.say("uncertain")
            if hasattr(self.picker, "put_back"):
                self.picker.put_back(self.robot)  # fixed spot: back where the next pick looks
            else:
                self.robot.go("putback")
                self.robot.gripper(open=True)
            self.robot.go("home")
            second = self._pick_and_inspect("second look")
            if second is None:
                break
            score2, heat, crop, v2 = second
            views.append((crop, heat, score2))
            # Any clearly defective view rejects the brick. Otherwise the new view decides, so a
            # false alarm can be cleared by a clean second look; still unsure -> reject (fail-safe).
            if v2 == "defective" or any(verdict(s, self.thr, self.band) == "defective" for _, _, s in views):
                v = "defective"
            else:
                v = v2
            score = max(s for _, _, s in views)
            self.ui.show(stage=f"INSPECT  view {len(views)}", verdict=v, score=score, views=views)

        bowl = final_bowl(v)
        pending = self._ask_advisor()        # runs while the arm moves to the bowl
        self.ui.check_stop()
        self.ui.show(stage=f"PLACE  {bowl} bowl")
        placed = False
        if hasattr(self.picker, "place"):
            try:
                placed = self.picker.place(self.robot, bowl)   # bowl where the camera last saw it
            except Exception as e:
                print("vision place failed, using the taught pose:", e, flush=True)
        if not placed:
            self.robot.go(f"bowl_{bowl}")
        self.robot.gripper(open=True)
        self.robot.go("home")
        self.counts[bowl] += 1
        op = self._opinion(pending, bowl)
        if v == "uncertain" and hasattr(self.ui, "say_text"):
            self.ui.say_text("Inspection uncertain. Brick placed in the reject bowl for a person to check.")
        elif op and op.get("defect"):
            self.ui.say(bowl, count=self.counts[bowl], defect=op["say_defect"], color=op["say_color"])
        else:
            self.ui.say(bowl, count=self.counts[bowl])
        wait_s = getattr(self.ui, "waited", 0.0) - w0
        cycle_s = time.time() - t0 - wait_s
        self.ui.show(stage="DONE", verdict=v, counts=dict(self.counts), cycle_s=cycle_s)
        out = {"verdict": v, "final_bowl": bowl, "pick_ok": 1, "regrasps": regrasps,
               "notes": " ".join(x for x in (f"dropped={self.dropped}" if self.dropped else "",
                                             op["note"] if op else "") if x),
               "cycle_s": round(cycle_s, 2), "wait_s": round(wait_s, 1), "score_v1": round(views[0][2], 4),
               "score_v2": round(views[1][2], 4) if len(views) > 1 else "",
               "thr": self.thr, "dropped": self.dropped, "placed_by": "vision" if placed else "pose"}
        lp = getattr(self.picker, "last_pick", None)
        if lp:
            out.update(pick_u=lp.get("u"), pick_v=lp.get("v"), pick_x=lp.get("x"), pick_y=lp.get("y"))
        if getattr(self.picker, "last_grip", None) is not None:
            out["grip"] = round(float(self.picker.last_grip), 1)
        return out
