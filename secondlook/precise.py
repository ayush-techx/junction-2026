"""Precise table pick: top camera colour + depth -> brick position and orientation -> robot base
frame (hand-eye) -> inverse kinematics on the SO-101 URDF.

Compared with the old per-joint pixel map (table.py PoseMap) this
  * needs no empty-table photo (height above the table finds bricks; bowls are found live),
  * turns the wrist so the fingers close across the brick (2x4 bricks have a long side),
  * keeps the wrist pointing down and the elbow on the same side everywhere (one arm model),
  * keeps learning: every second look that puts a brick back gives a new calibration pair, and
    `./sl table --auto` collects many pairs with no hands on the arm.

pick() returns "grasped", "missed" or "empty" like the other pickers.
"""
import time

import numpy as np

from . import scene
from .handeye import HandEye
from .kinematics import Arm

DEFAULT_REF = {"shoulder_pan": -45.0, "shoulder_lift": 60.0, "elbow_flex": -60.0, "wrist_flex": 78.0, "wrist_roll": -57.0}


class IKTablePicker:
    method = "ik"

    def __init__(self, cfg, camera="top", history=None):
        if camera not in cfg["cameras"]:
            raise KeyError(f"precise pick needs the '{camera}' camera in config.yaml")
        sc = cfg.get("scan") or {}
        self.cam = camera
        self.arm = Arm()
        self.he = HandEye.load()
        self.params = sc.get("detect") or {}
        self.above_m = float(sc.get("above_m", 0.06))
        self.max_tilt = float(sc.get("max_tilt_deg", 35))
        self.place_by_vision = bool(sc.get("place_by_vision", False))
        self.drop_height_m = float(sc.get("drop_height_m", 0.09))
        self.learn = bool(sc.get("learn_from_putback", True))
        self.bowl_colour = (cfg.get("bowls") or {"good": "pink", "defective": "blue"})
        self.ref = dict(DEFAULT_REF, **(self.he.data.get("ref") or {}))
        self.history = history                  # callable(**row) -> logs one pick attempt
        self.run = None
        self.last = None                        # plan of the last pick (for the second look)
        self.prefer = None                      # pixel to pick first (the brick that was put back)
        self.expect = None                      # where a put-back brick should be seen again
        self.found = 0
        self.scene = None                       # (result, plane, K) of the latest scan
        self.notes = []                         # warnings for the UI
        self.last_pick = None                   # info about the last attempt, for the run log
        self.colors, self.obstacles = [], 0

    # --- seeing ---------------------------------------------------------------------------
    def _snapshot(self, robot):
        bgr, depth, K = robot.rgbd(self.cam)
        inl = None
        if depth is not None:
            try:
                plane, inl = scene.fit_plane(depth, K)
            except ValueError:
                plane, depth = self.he.plane, None
                K = self.he.K
        else:
            plane, K = self.he.plane, self.he.K
        return bgr, depth, K, plane, inl

    def scan(self, robot, ui=None):
        robot.go("home")                       # park out of the top camera's view
        time.sleep(0.4)
        bgr, depth, K, plane, inl = self._snapshot(robot)
        res = scene.detect(bgr, depth, K, plane, params=self.params)
        self.notes = []
        if depth is not None and self.he.plane:
            ang, dh = scene.plane_change(plane, self.he.plane)
            if ang > 1.5 or dh > 15:
                self.notes.append(f"Top camera seems to have moved (tilt {ang:.1f} deg, {dh:.0f} mm): "
                                  "picks may miss; run ./sl table --auto")
        bricks = res["bricks"]
        for b in bricks:
            b["base"] = self.he.to_base(b["grasp_cam"]).tolist() if "grasp_cam" in b else None
        bricks = [b for b in bricks if b.get("base") is not None]
        self._learn_from_putback(bricks)
        if self.prefer is not None:
            pu, pv = self.prefer
            bricks.sort(key=lambda b: np.hypot(b["u"] - pu, b["v"] - pv))
        else:
            # Isolated bricks first (nothing to bump into), then the closest to the robot.
            bricks.sort(key=lambda b: (-min(b.get("clear_px", 999), 80), np.hypot(*b["base"][:2])))
        self.found = len(bricks)
        self.colors = [b.get("color") for b in bricks]
        self.scene = (res, plane, K)
        ignore = self._arm_pixel(K)
        rep = scene.report(res, plane, ref=self.he.data.get("scene_ref"), inlier=inl)
        if ignore is not None:
            rep["obstacles"] = [o for o in rep["obstacles"] if np.hypot(o["u"] - ignore[0], o["v"] - ignore[1]) > 120]
            rep["warnings"] = [w for w in rep["warnings"] if not self._is_arm_warning(w, ignore)]
        rep["warnings"] = self.notes + rep["warnings"]
        self.obstacles = len(rep["obstacles"])              # real objects only (not the arm, not the image edge)
        target = (bricks[0]["u"], bricks[0]["v"]) if bricks else None
        try:
            scene.save(bgr, res, rep, target=target, ignore=ignore)
        except Exception as e:                         # never stop a run for a dashboard picture
            print("scene snapshot not saved:", e, flush=True)
        if ui is not None and hasattr(ui, "event"):
            for w in self.notes:
                ui.event(w, "warn")
        return bgr, bricks

    def _arm_pixel(self, K):
        """Where the parked gripper appears in the top image (it is not an obstacle)."""
        try:
            import yaml
            with open("secondlook/poses.yaml") as f:
                home = yaml.safe_load(f)["home"]
            return scene.project(self.he.to_cam(self.arm.tcp(home)), K)
        except Exception:
            return None

    @staticmethod
    def _is_arm_warning(w, px):
        import re
        m = re.search(r"at \((\d+), (\d+)\)", w)
        return bool(m) and np.hypot(int(m[1]) - px[0], int(m[2]) - px[1]) <= 120

    def _learn_from_putback(self, bricks):
        """A brick we put back at a known spot is now seen again: one more calibration pair."""
        exp, self.expect = self.expect, None
        if not (exp and self.learn and bricks):
            return
        near = min(bricks, key=lambda b: np.linalg.norm(np.asarray(b["base"]) - exp["base"]))
        miss_mm = float(np.linalg.norm(np.asarray(near["base"]) - exp["base"]) * 1000)
        if miss_mm > 45:                                 # probably another brick: don't learn from it
            return
        fit = self.he.add_pair(near["grasp_cam"], exp["base"], "putback", u=near["u"], v=near["v"])
        self.he.save()
        print(f"calibration learned from put-back: was off by {miss_mm:.0f} mm, fit now "
              f"{fit['mean_mm']} mm mean over {fit['pairs_used']} pairs", flush=True)

    # --- planning -------------------------------------------------------------------------
    def plan(self, b):
        """IK for grasp + above poses, wrist turned to grasp across the brick."""
        up = self.he.up_base()
        tgt = np.asarray(b["base"], float)
        q, info = self.arm.ik(tgt, up, self.ref["wrist_roll"], self.ref, self.max_tilt)
        roll = self.ref["wrist_roll"]
        if b.get("long_axis_cam"):
            roll = self.arm.roll_for(q, self.he.dir_to_base(b["long_axis_cam"]), up, b.get("elongated", False))
            q, info = self.arm.ik(tgt, up, roll, self.ref, self.max_tilt, starts=[q])
        qa, ia = self.arm.ik(tgt + up * self.above_m, up, roll, self.ref, self.max_tilt, starts=[q])
        ok = info["ok"] and ia["pos_err_mm"] < 10
        return {"pick": q, "above": qa, "info": info, "target": tgt.tolist(), "roll": roll, "ok": ok,
                "u": b["u"], "v": b["v"], "brick": b}

    # --- acting ---------------------------------------------------------------------------
    def pick(self, robot, ui):
        _, bricks = self.scan(robot, ui)
        if not bricks:
            ui.show(stage="TABLE CLEAR")
            return "empty"
        plan = None
        for b in bricks:
            p = self.plan(b)
            if p["ok"]:
                plan = p
                break
            if hasattr(ui, "event"):
                ui.event(f"Brick at ({b['u']:.0f}, {b['v']:.0f}) is out of reach "
                         f"(off by {p['info']['pos_err_mm']:.0f} mm, wrist tilt {p['info']['tilt_deg']:.0f} deg): skipped", "warn")
        self.prefer = None
        if plan is None:
            ui.show(stage="OUT OF REACH  move the bricks closer to the robot")
            return "blocked"
        b = plan["brick"]
        ui.show(stage=f"PICK  brick at ({b['u']:.0f}, {b['v']:.0f}), {len(bricks)} on the table"
                      + (f", {b['size_mm'][0]}x{b['size_mm'][1]} mm" if b.get("size_mm") else ""))
        self.last = plan
        robot.gripper(open=True)
        robot.move(plan["above"])
        robot.move(plan["pick"])
        robot.gripper(open=False)
        time.sleep(0.3)
        try:
            grip = float(robot.observe()["gripper.pos"])
        except Exception:
            grip = None
        robot.move(plan["above"])
        held = robot.holding()
        self.last_grip, self.last_held = grip, 1 if held else 0
        self._log(plan, "grasped" if held else "missed", grip)
        return "grasped" if held else "missed"

    def _log(self, plan, outcome, grip):
        b = plan["brick"]
        self.last_pick = {"u": b["u"], "v": b["v"], "x": plan["target"][0], "y": plan["target"][1],
                          "method": self.method, "outcome": outcome}
        if self.history:
            try:
                self.history(run=self.run, method=self.method, u=b["u"], v=b["v"], x=plan["target"][0],
                             y=plan["target"][1], z=plan["target"][2], outcome=outcome, grip=grip,
                             height_mm=b.get("height_mm"), size=b.get("size_mm"), color=b.get("color"),
                             roll=plan["roll"], tilt_deg=plan["info"]["tilt_deg"], ik_err_mm=plan["info"]["pos_err_mm"],
                             joints=plan["pick"])
            except Exception as e:
                print("history log failed:", e, flush=True)

    def put_back(self, robot):
        """Second look: set the brick down where it came from; the next scan finds it again, and where it
        really landed teaches the calibration."""
        if self.last is None:
            robot.go("putback")
            robot.gripper(open=True)
            return
        robot.move(self.last["above"])
        robot.move(self.last["pick"])
        robot.gripper(open=True)
        robot.move(self.last["above"])
        self.prefer = (self.last["u"], self.last["v"])
        self.expect = {"base": np.asarray(self.arm.tcp(self.last["pick"]))}

    def note_slip(self):
        if self.last is not None:
            self._log(self.last, "slipped", None)

    def place(self, robot, bowl):
        """Drop into the bowl where the camera last saw it (bowls drift). False = use the taught pose."""
        if not self.place_by_vision or self.scene is None:
            return False
        res, plane, K = self.scene
        colour = self.bowl_colour.get(bowl)
        cand = [b for b in res["bowls"] if b["color"] == colour]
        if not cand or plane is None:
            return False
        bw = cand[0]
        tgt_cam = scene.ray_plane(bw["u"], bw["v"], K, plane, self.drop_height_m)
        up = self.he.up_base()
        q, info = self.arm.ik(self.he.to_base(tgt_cam), up, self.ref["wrist_roll"], self.ref, max_tilt_deg=50)
        if info["pos_err_mm"] > 15:
            print(f"bowl {colour} out of reach by vision ({info}); using the taught pose", flush=True)
            return False
        robot.move(q)
        return True


class TableWatcher:
    """Counts the bricks on the table with the top camera, for pickers that have no table scan of their
    own (the learned ACT / SmolVLA pick), so "sort the table" knows when it is done."""

    def __init__(self, cfg, camera="top"):
        self.cam = camera
        self.he = HandEye.load() if HandEye.exists() else None
        self.params = (cfg.get("scan") or {}).get("detect") or {}

    def look(self, robot):
        robot.go("home")                        # out of the camera's view
        time.sleep(0.4)
        bgr, depth, K = robot.rgbd(self.cam)
        plane = inl = None
        if depth is not None:
            try:
                plane, inl = scene.fit_plane(depth, K)
            except ValueError:
                depth = None
        if depth is None and self.he is not None:
            plane, K = self.he.plane, self.he.K
        res = scene.detect(bgr, depth, K, plane, params=self.params)
        rep = scene.report(res, plane, ref=self.he.data.get("scene_ref") if self.he else None, inlier=inl)
        ignore = None
        if self.he is not None:
            p = IKTablePicker.__new__(IKTablePicker)
            p.he, p.arm = self.he, Arm()
            ignore = p._arm_pixel(K)
            if ignore is not None:
                rep["obstacles"] = [o for o in rep["obstacles"] if np.hypot(o["u"] - ignore[0], o["v"] - ignore[1]) > 120]
                rep["warnings"] = [w for w in rep["warnings"] if not IKTablePicker._is_arm_warning(w, ignore)]
        try:
            scene.save(bgr, res, rep, target=(res["bricks"][0]["u"], res["bricks"][0]["v"]) if res["bricks"] else None,
                       ignore=ignore)
        except Exception as e:
            print("scene snapshot not saved:", e, flush=True)
        self.obstacles = len(rep["obstacles"])
        return res["bricks"]
