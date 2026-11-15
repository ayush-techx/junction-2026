"""Pickers: who puts a brick in the gripper.

PolicyPicker  learned pick with the exported ACT policy (OpenVINO, via physicalai InferenceModel).
HandFedPicker operator places the brick in the gripper; the demo still works without a policy.
FixedSpotPicker scripted pick from one marked spot on the table, using taught poses (backup, no learning).
"""
import os
import time

import numpy as np

from .robot import JOINTS


class PolicyPicker:
    """Learned pick with a Physical AI Studio export (ACT or SmolVLA), fed exactly like Studio's
    model worker: {"state": (1, 6) float32, "images.<cam>": (1, H, W, 3) uint8 RGB, "task": [text]}.
    The predicted action chunk is played step by step; the model is queried again every
    `replan_steps` steps so it keeps reacting to what the cameras see."""

    def __init__(self, cfg):
        pc = cfg["policy"]
        path = pc.get("path")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"policy.path is {path!r}. Set it in config.yaml to the exported OpenVINO policy "
                "folder, or run with --handfed until the trained model is available.")
        import json
        from physicalai.inference import InferenceModel
        # Run on the configured device (Intel GPU: SmolVLA ~0.1 s per chunk vs ~1.9 s on the CPU).
        self.model = InferenceModel(path, device=pc.get("device") or "auto")
        manifest = json.load(open(os.path.join(path, "manifest.json")))
        feats = [f["init_args"] for f in manifest["model"]["input_features"]]
        self.state_name = next(f["name"] for f in feats if f["ftype"] == "STATE")
        image_map = pc.get("image_map") or {}          # model input -> SecondLook camera, if names differ
        self.images = {}
        for f in feats:
            if f["ftype"] == "VISUAL":
                cam = image_map.get(f["name"], f["name"].split(".", 1)[-1])
                if cam not in cfg["cameras"]:
                    raise KeyError(f"model input {f['name']!r} needs camera {cam!r}; set policy.image_map")
                self.images[f["name"]] = cam
        self.cams = sorted(set(self.images.values()))
        self.task = pc.get("task")                     # SmolVLA reads it; ACT ignores it
        self.start_pose = pc.get("start_pose")         # where the recorded demos began
        self.replan = int(pc.get("replan_steps", 30))
        self.hz = cfg["robot"]["hz"]
        self.max_s, self.after_grasp = pc["max_seconds"], pc["after_grasp_steps"]
        self.last_ms, self._queue = None, []

    def to_obs(self, raw):
        obs = {self.state_name: np.array([[raw[f"{j}.pos"] for j in JOINTS]], np.float32)}
        for name, cam in self.images.items():
            obs[name] = np.asarray(raw[cam], np.uint8)[np.newaxis]
        if self.task:
            obs["task"] = [self.task]
        return obs

    def chunk(self, raw):
        t = time.perf_counter()
        actions = np.asarray(self.model.predict_action_chunk(self.to_obs(raw)), np.float32).reshape(-1, len(JOINTS))
        self.last_ms = (time.perf_counter() - t) * 1000
        return actions

    def act(self, raw):
        if not self._queue:
            self._queue = list(self.chunk(raw)[: self.replan])
        a = self._queue.pop(0)
        # Safety: never command outside the calibrated range (normalised units).
        lo = {j: -100.0 for j in JOINTS}; lo["gripper"] = 0.0
        return {j: float(np.clip(a[i], lo[j], 100.0)) for i, j in enumerate(JOINTS)}

    def pick(self, robot, ui):
        """Run the policy until the brick is held and lifted, then hand over to taught poses.
        The recorded demos continue to a bowl; we stop after_grasp steps after the grasp."""
        # Start where the demos started (arm raised, gripper open over the pick area), not folded at
        # home: ACT only knows how to begin from poses it saw, and stops once it is above a bowl.
        if self.start_pose:
            robot.move(self.start_pose)
            robot.gripper(open=True)
        self.model.reset()
        self._queue = []
        held, end, prev = 0, time.time() + self.max_s, None
        while time.time() < end:
            t = time.time()
            ui.check_stop()
            raw = robot.observe()
            robot.send(self.act(raw))
            g = float(raw["gripper.pos"])
            # Holding = gripper in the hold range AND no longer closing (stopped on the brick),
            # so a gripper that is merely passing through the range is not counted as a grasp.
            steady = prev is not None and abs(g - prev) < 1.5
            prev = g
            held = held + 1 if (robot.holding_from(raw) and steady) else 0
            if held >= self.after_grasp:
                return self._confirm(robot)
            ui.show(policy_ms=self.last_ms)
            time.sleep(max(0.0, 1 / self.hz - (time.time() - t)))
        return self._confirm(robot) if robot.holding() else "timeout"

    def _confirm(self, robot):
        """The policy only closes the gripper part way (~9), so an EMPTY gripper can look like
        'holding'. Squeeze fully, then read it: a brick stops the fingers (held=1), empty fingers
        close to ~1 (held=0). Only held=1 is ever inspected, so an empty grasp can't be a defect."""
        robot.gripper(open=False)
        time.sleep(0.4)
        g = float(robot.observe()["gripper.pos"])
        lo, hi = robot.g["hold_min"], robot.g["hold_max"]
        held = lo <= g <= hi
        self.last_grip, self.last_held = g, 1 if held else 0
        print(f"squeeze check: gripper={g:.1f} (held if {lo}-{hi}) -> held={1 if held else 0}", flush=True)
        return "grasped" if held else "missed"


class HandFedPicker:
    """Operator puts a brick between the open fingers and presses the button."""

    method = "handfed"

    def pick(self, robot, ui):
        robot.go("handoff")
        robot.gripper(open=True)
        ui.show(stage="PLACE A BRICK IN THE GRIPPER, THEN PRESS SPACE")
        ui.say("handfed")
        while True:
            ui.wait_key({ord(" ")})
            robot.gripper(open=False)
            time.sleep(0.3)
            g = float(robot.observe()["gripper.pos"])
            held = robot.g["hold_min"] <= g <= robot.g["hold_max"]
            self.last_grip, self.last_held = g, 1 if held else 0
            if held:
                return "grasped"
            # Closed on nothing (e.g. Space pressed to "start"): not a failed pick, just ask again.
            robot.gripper(open=True)
            if hasattr(ui, "event"):
                ui.event("The gripper closed on nothing: put a brick between the fingers, then press Space", "warn")
            ui.show(stage="NO BRICK IN THE GRIPPER  put one between the fingers, then press SPACE")


class FixedSpotPicker:
    """Scripted backup pick: the operator puts a brick on one marked spot on the table.

    Needs two taught poses:
      pick_above  gripper open, a few cm straight above the marked spot
      pick        fingers around the brick on the spot, ready to close
    """

    def __init__(self, robot_poses):
        missing = [p for p in ("pick_above", "pick") if p not in robot_poses]
        if missing:
            raise KeyError(f"fixed-spot pick needs taught poses {missing}; run `./sl teach` and add them")

    def pick(self, robot, ui):
        robot.gripper(open=True)
        robot.go("pick_above")
        robot.go("pick")
        robot.gripper(open=False)
        time.sleep(0.3)
        robot.go("pick_above")                 # lift straight up before the long move
        return "grasped" if robot.holding() else "missed"

    def put_back(self, robot):
        """Second look: return the brick to the marked spot so the next pick finds it."""
        robot.go("pick_above")
        robot.go("pick")
        robot.gripper(open=True)
        robot.go("pick_above")
