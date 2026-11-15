"""SO101 wrapper against a stand-in LeRobot driver: poses, ramps, gripper, hold detection."""
import os
import sys
import tempfile
import types
import unittest

import yaml

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class StubFollower:
    """Tracks commands perfectly, like an ideal servo."""

    def __init__(self, config):
        self.config = config
        self.pos = {f"{j}.pos": 0.0 for j in JOINTS}
        self.sent = []
        self.connected = False
        self.torque_on = True
        self.bus = types.SimpleNamespace(enable_torque=lambda: self._torque(True),
                                         disable_torque=lambda: self._torque(False))

    def _torque(self, on):
        self.torque_on = on

    def connect(self, calibrate=True):
        assert calibrate is False, "must reuse the existing calibration"
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        return dict(self.pos)

    def send_action(self, action):
        self.sent.append(action)
        self.pos.update(action)


class StubConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def install_stub_lerobot():
    pkg = types.ModuleType("lerobot")
    robots = types.ModuleType("lerobot.robots")
    so101 = types.ModuleType("lerobot.robots.so101_follower")
    so101.SO101Follower, so101.SO101FollowerConfig = StubFollower, StubConfig
    sys.modules.update({"lerobot": pkg, "lerobot.robots": robots, "lerobot.robots.so101_follower": so101})


def make_cfg(poses_file, **gripper):
    g = {"open": 80.0, "close": 0.0, "hold_min": 10.0, "hold_max": 60.0}
    g.update(gripper)
    return {"robot": {"port": "/dev/null", "id": "follower", "calibration_dir": "/cal", "use_degrees": None,
                      "poses_file": poses_file, "hz": 1000, "max_deg_per_s": 100000, "gripper": g},
            "cameras": {}}


class TestSO101(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_stub_lerobot()
        from secondlook.robot import SO101
        cls.SO101 = SO101

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.poses = os.path.join(self.tmp.name, "poses.yaml")

    def tearDown(self):
        self.tmp.cleanup()

    def test_config_passthrough(self):
        r = self.SO101(make_cfg(self.poses))
        self.assertEqual(str(r.r.config.calibration_dir), "/cal")
        self.assertFalse(hasattr(r.r.config, "use_degrees"))      # None means: leave LeRobot default

    def test_teach_then_go(self):
        r = self.SO101(make_cfg(self.poses))
        r.connect()
        r.r.pos.update({"shoulder_pan.pos": 30.0, "elbow_flex.pos": -45.0, "gripper.pos": 50.0})
        r.save_pose("inspect")
        self.assertNotIn("gripper", r.poses["inspect"])           # poses never move the gripper
        with open(self.poses) as f:
            self.assertEqual(yaml.safe_load(f)["inspect"]["shoulder_pan"], 30.0)

        r2 = self.SO101(make_cfg(self.poses))                     # reload from disk
        r2.connect()
        r2.r.pos["gripper.pos"] = 50.0
        r2.go("inspect")
        j = r2.joints()
        self.assertAlmostEqual(j["shoulder_pan"], 30.0)
        self.assertAlmostEqual(j["elbow_flex"], -45.0)
        self.assertEqual(j["gripper"], 50.0)
        self.assertTrue(all("gripper.pos" not in a for a in r2.r.sent))

    def test_ramp_is_gradual(self):
        r = self.SO101(make_cfg(self.poses))
        r.speed, r.hz = 100.0, 50                                 # 90 deg at 100 deg/s -> 45 steps
        r.poses = {"far": {j: 0.0 for j in JOINTS[:-1]}}
        r.poses["far"]["shoulder_pan"] = 90.0
        r.go("far")
        pans = [a["shoulder_pan.pos"] for a in r.r.sent]
        self.assertGreaterEqual(len(pans), 40)
        self.assertLessEqual(max(b - a for a, b in zip(pans, pans[1:])), 2.5)
        self.assertAlmostEqual(pans[-1], 90.0)

    def test_on_tick_called_and_can_stop(self):
        r = self.SO101(make_cfg(self.poses))
        r.poses = {"p": {j: 10.0 for j in JOINTS[:-1]}}
        calls = []

        def tick():
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt
        r.on_tick = tick
        r.speed = 1.0
        with self.assertRaises(KeyboardInterrupt):
            r.go("p")
        self.assertEqual(len(calls), 3)

    def test_gripper_and_holding(self):
        r = self.SO101(make_cfg(self.poses))
        r.gripper(open=True)
        self.assertEqual(r.joints()["gripper"], 80.0)
        self.assertFalse(r.holding())                              # fully open
        r.r.pos["gripper.pos"] = 30.0                              # stalled on a brick
        self.assertTrue(r.holding())
        r.gripper(open=False)
        self.assertFalse(r.holding())                              # fully closed = empty

    def test_unset_values_fail_clearly(self):
        r = self.SO101(make_cfg(self.poses, open=None, hold_min=None))
        with self.assertRaises(ValueError):
            r.gripper(open=True)
        with self.assertRaises(ValueError):
            r.holding()
        with self.assertRaises(KeyError):
            r.go("never_taught")

    def test_torque(self):
        r = self.SO101(make_cfg(self.poses))
        r.torque(False)
        self.assertFalse(r.r.torque_on)
        r.torque(True)
        self.assertTrue(r.r.torque_on)


if __name__ == "__main__":
    unittest.main()
