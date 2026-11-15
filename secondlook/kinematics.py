"""SO-101 kinematics from the URDF: forward kinematics and inverse kinematics in SecondLook's
joint units (LeRobot normalised, -100..100).

Unit convention: the same one Physical AI Studio's 3D viewer uses, i.e. a joint value n maps
linearly onto the URDF joint limits:  angle = lower + (n + 100) / 200 * (upper - lower).
Checked against the 5 hand-taught table points: forward kinematics plus one rigid camera pose
reproduces them to 7-15 mm, including points taught elbow-up and elbow-down.

No hardware imports; numpy + scipy only.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.path.join(os.path.dirname(__file__), "urdf", "so101_follower.urdf")
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SAFE = 97.0                      # never command beyond +-97 (normalised units)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a, t):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def _origin(el):
    o = el.find("origin")
    m = np.eye(4)
    if o is not None:
        m[:3, :3] = _rpy(*map(float, (o.get("rpy") or "0 0 0").split()))
        m[:3, 3] = list(map(float, (o.get("xyz") or "0 0 0").split()))
    return m


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class Arm:
    """Kinematic chain base_link -> gripper_frame_link (the point between the finger tips)."""

    def __init__(self, urdf=None, tool="gripper_frame_link"):
        root = ET.parse(urdf or URDF).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], tool
        while link in by_child:
            j = by_child[link]
            chain.insert(0, j)
            link = j.find("parent").get("link")
        self.chain = [(j.get("name"), j.get("type"), _origin(j),
                       np.array(list(map(float, j.find("axis").get("xyz").split()))) if j.find("axis") is not None else None)
                      for j in chain]
        self.limits = {}
        for j in root.findall("joint"):
            lim = j.find("limit")
            if lim is not None and j.get("type") == "revolute":
                self.limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
        missing = [j for j in ARM if j not in self.limits]
        if missing:
            raise ValueError(f"URDF has no limits for {missing}")
        # Jaw hinge axis in the tool frame: the fingers close perpendicular to it and to the approach axis.
        jaw = next((j for j in root.findall("joint") if j.get("name") == "gripper"), None)
        self._closing = np.array([1.0, 0.0, 0.0])
        if jaw is not None:
            t_tool = self.fk({j: 0.0 for j in ARM})
            t_jaw = self._link_pose(root, by_child, jaw.find("parent").get("link")) @ _origin(jaw)
            hinge = np.linalg.inv(t_tool)[:3, :3] @ (t_jaw[:3, :3] @ np.array([0.0, 0.0, 1.0]))
            self._closing = unit(np.cross(hinge, [0.0, 0.0, 1.0]))

    def _link_pose(self, root, by_child, link):
        """Pose of a link at all-zero (normalised 0) joints; used once for the jaw axis."""
        chain = []
        while link in by_child:
            j = by_child[link]
            chain.insert(0, j)
            link = j.find("parent").get("link")
        t = np.eye(4)
        for j in chain:
            t = t @ _origin(j)
            if j.get("type") == "revolute" and j.get("name") in self.limits:
                lo, hi = self.limits[j.get("name")]
                r = np.eye(4)
                r[:3, :3] = _axis_angle(list(map(float, j.find("axis").get("xyz").split())), lo + 0.5 * (hi - lo))
                t = t @ r
        return t

    # --- units ----------------------------------------------------------------------------
    def to_rad(self, joint, n):
        lo, hi = self.limits[joint]
        return lo + (float(n) + 100.0) / 200.0 * (hi - lo)

    def rad_per_unit(self, joint):
        lo, hi = self.limits[joint]
        return (hi - lo) / 200.0

    # --- forward --------------------------------------------------------------------------
    def fk(self, joints):
        """4x4 pose of the tool point in the base frame. `joints` in normalised units."""
        t = np.eye(4)
        for name, kind, origin, axis in self.chain:
            t = t @ origin
            if kind == "revolute":
                r = np.eye(4)
                r[:3, :3] = _axis_angle(axis, self.to_rad(name, joints.get(name, 0.0)))
                t = t @ r
        return t

    def tcp(self, joints):
        return self.fk(joints)[:3, 3]

    def closing_axis(self, t):
        """Direction (base frame) along which the fingers close, for tool pose t."""
        return t[:3, :3] @ self._closing

    # --- inverse --------------------------------------------------------------------------
    def ik(self, target, up, roll, ref, max_tilt_deg=35.0, starts=None):
        """Joints that put the tool point on `target` (base frame, metres) with the fingers pointing
        down (against `up`, the table normal in the base frame), wrist_roll fixed to `roll`.
        `ref` (dict) is the posture to stay close to (keeps the elbow on the same side).
        Returns (joints, info) with info = {pos_err_mm, tilt_deg, ok}."""
        from scipy.optimize import least_squares
        target, up = np.asarray(target, float), unit(up)
        names = ARM[:4]
        x_ref = np.array([ref[j] for j in names], float)

        def residual(x):
            q = dict(zip(names, x))
            q["wrist_roll"] = roll
            t = self.fk(q)
            approach = t[:3, 2]                     # tool z points out of the fingers
            return np.r_[(t[:3, 3] - target) * 1000.0,          # mm
                         (approach + up) * 40.0,                 # 1 rad of tilt ~ 40 mm
                         (x - x_ref) * 0.03]                     # mild pull towards the reference posture

        best = None
        for x0 in (starts or []) + [x_ref]:
            x0 = np.clip(np.asarray([x0[j] for j in names] if isinstance(x0, dict) else x0, float), -SAFE, SAFE)
            r = least_squares(residual, x0, bounds=([-SAFE] * 4, [SAFE] * 4), xtol=1e-6, ftol=1e-6, max_nfev=400)
            if best is None or r.cost < best.cost:
                best = r
        q = dict(zip(names, (float(v) for v in best.x)))
        q["wrist_roll"] = float(roll)
        t = self.fk(q)
        err = float(np.linalg.norm(t[:3, 3] - target) * 1000.0)
        tilt = float(np.degrees(np.arccos(np.clip(-t[:3, 2] @ up, -1, 1))))
        return q, {"pos_err_mm": round(err, 1), "tilt_deg": round(tilt, 1), "ok": err < 6.0 and tilt <= max_tilt_deg}

    def roll_for(self, joints, long_axis, up, elongated):
        """wrist_roll value that makes the fingers close across the brick: perpendicular to its long
        axis for a 2x4, at the nearest 90 degrees for a square 2x2. Picks the smallest turn."""
        t = self.fk(joints)
        up = unit(up)
        c = unit(self.closing_axis(t) - up * (self.closing_axis(t) @ up))       # closing dir on the table
        a = unit(np.asarray(long_axis, float) - up * (np.asarray(long_axis, float) @ up))
        want = unit(np.cross(up, a))                      # perpendicular to the long axis, in the table plane
        ang = np.arctan2(np.cross(c, want) @ up, c @ want)          # signed angle c -> want about `up`
        period = np.pi if elongated else np.pi / 2
        ang = (ang + period / 2) % period - period / 2              # smallest equivalent turn
        # Which way does +1 unit of wrist_roll turn the fingers (seen along `up`)?
        q2 = dict(joints, wrist_roll=joints["wrist_roll"] + 1.0)
        c2 = self.closing_axis(self.fk(q2))
        c2 = unit(c2 - up * (c2 @ up))
        per_unit = np.arctan2(np.cross(c, c2) @ up, c @ c2)
        if abs(per_unit) < 1e-6:
            return joints["wrist_roll"]
        roll = joints["wrist_roll"] + ang / per_unit
        if abs(roll) > SAFE:                              # out of range: turn the other way (same grasp)
            roll -= np.sign(roll) * period / abs(per_unit)
        return float(np.clip(roll, -SAFE, SAFE))
