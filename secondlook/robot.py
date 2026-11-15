"""SO-101 follower arm + cameras through LeRobot, plus taught poses.

Physical AI Studio records LeRobot-format data and drives the SO-101 through LeRobot,
so we use the same driver here. If your team's hardware.py already works, you can
replace this class with a wrapper that has the same methods:
    connect, disconnect, observe, joints, frame, send, go, gripper, holding, holding_from,
    torque, save_pose
"""
import os
import time

import yaml

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = JOINTS[:-1]


def _lerobot_classes():
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    except ImportError:  # newer LeRobot layout
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    return SO101Follower, SO101FollowerConfig


def _camera_config(c):
    if c["type"] == "realsense":
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        return RealSenseCameraConfig(serial_number_or_name=str(c["serial"]), fps=c["fps"],
                                     width=c["width"], height=c["height"], use_depth=bool(c.get("depth", False)))
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    return OpenCVCameraConfig(index_or_path=c["path"], fps=c["fps"], width=c["width"], height=c["height"])


class SO101:
    def __init__(self, cfg, cameras=None):
        rc = cfg["robot"]
        self.hz, self.speed, self.g = rc["hz"], rc["max_deg_per_s"], rc["gripper"]
        self.poses_file = rc["poses_file"]
        self.poses = {}
        if os.path.exists(self.poses_file):
            with open(self.poses_file) as f:
                self.poses = yaml.safe_load(f) or {}
        names = cameras if cameras is not None else list(cfg["cameras"])
        cams = {n: _camera_config(cfg["cameras"][n]) for n in names}
        follower, config = _lerobot_classes()
        extra = {k: rc[k] for k in ("calibration_dir", "use_degrees") if rc.get(k) is not None}
        if "calibration_dir" in extra:     # LeRobot wants a Path; relative paths are from the repo root
            from pathlib import Path
            extra["calibration_dir"] = Path(extra["calibration_dir"]).expanduser().resolve()
        self.r = follower(config(port=rc["port"], id=rc["id"], cameras=cams, **extra))
        self.on_tick = None            # set by the app: called every control step (HUD refresh + STOP key)

    # --- connection -------------------------------------------------------
    def connect(self):
        self.r.connect(calibrate=False)    # reuse the existing calibration file for this id

    def disconnect(self):
        self.r.disconnect()

    def hold_here(self):
        """Motors on where the arm is now, without a jump: set the goal to the current position first."""
        self.send(self.joints())
        self.torque(True)

    def torque(self, on):
        (self.r.bus.enable_torque if on else self.r.bus.disable_torque)()

    # --- reading ----------------------------------------------------------
    def observe(self):
        """Raw LeRobot observation: '<joint>.pos' floats + one RGB array per camera name."""
        return self.r.get_observation()

    def joints(self, obs=None):
        obs = obs or self.observe()
        return {j: float(obs[f"{j}.pos"]) for j in JOINTS}

    def frame(self, cam):
        """BGR image (OpenCV order) from one camera."""
        return self.observe()[cam][..., ::-1].copy()

    def rgbd(self, cam, frames=5):
        """(BGR image, depth in metres aligned to it or None, colour intrinsics or None).
        Depth needs `depth: true` on that camera in config.yaml (RealSense only)."""
        from .scene import rgbd_from_camera
        c = self.r.cameras[cam]
        if not getattr(c, "use_depth", False):
            return self.frame(cam), None, None
        return rgbd_from_camera(c, frames)

    def holding_from(self, obs):
        if self.g["hold_min"] is None:
            raise ValueError("robot.gripper.hold_min/hold_max not set; run `check` and fill config.yaml")
        v = float(obs["gripper.pos"])
        return self.g["hold_min"] <= v <= self.g["hold_max"]

    def holding(self):
        return self.holding_from(self.observe())

    # --- moving -----------------------------------------------------------
    def send(self, target):
        self.r.send_action({f"{j}.pos": float(v) for j, v in target.items()})

    def _ramp(self, target, min_s=0.4):
        cur = self.joints()
        dist = max(abs(target[j] - cur[j]) for j in target)
        steps = max(1, int(max(dist / self.speed, min_s) * self.hz))
        for i in range(1, steps + 1):
            a = i / steps
            self.send({j: cur[j] + a * (target[j] - cur[j]) for j in target})
            if self.on_tick:
                self.on_tick()
            time.sleep(1 / self.hz)

    def go(self, pose):
        """Move the arm (not the gripper) to a taught pose."""
        if pose not in self.poses:
            raise KeyError(f"pose '{pose}' not taught; run `python -m secondlook.cell teach`")
        self._ramp({j: self.poses[pose][j] for j in ARM})
        time.sleep(0.15)

    def move(self, pose):
        """Move the arm (not the gripper) to a computed pose {joint: deg}, clamped to the safe range."""
        self._ramp({j: max(-100.0, min(100.0, float(pose[j]))) for j in ARM})
        time.sleep(0.15)

    def gripper(self, open):
        v = self.g["open"] if open else self.g["close"]
        if v is None:
            raise ValueError("robot.gripper.open/close not set; run `check` and fill config.yaml")
        self._ramp({"gripper": v}, min_s=0.5)

    def save_pose(self, name):
        self.poses[name] = {j: round(v, 2) for j, v in self.joints().items() if j in ARM}
        with open(self.poses_file, "w") as f:
            yaml.safe_dump(self.poses, f, sort_keys=False)
