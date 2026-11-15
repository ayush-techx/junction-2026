"""Precise table pick: kinematics, depth scene detection, hand-eye fit, history store, picker flow.
No hardware: synthetic depth images and a fake robot."""
import os
import shutil
import tempfile
import unittest

import numpy as np

K = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "w": 640, "h": 480}
# Table 0.65 m below the camera, tilted a little (like the real D415 mount).
N = np.array([-0.07, -0.40, -0.91])
N = N / np.linalg.norm(N)
PLANE = [float(N[0]), float(N[1]), float(N[2]), 0.62]


def synthetic(bricks=(), bowls=(), box=None):
    """Colour + depth of a white table with bricks (u, v, size_px, colour), bowls (u, v, r, colour)."""
    import cv2
    bgr = np.full((480, 640, 3), (235, 235, 235), np.uint8)
    h = np.zeros((480, 640), np.float32)                       # height above table (m)
    for (u, v, r, col) in bowls:
        cv2.circle(bgr, (u, v), r, col, -1)
        ring = np.zeros_like(h, np.uint8)
        cv2.circle(ring, (u, v), r, 1, 8)
        h[ring > 0] = 0.014
    for (u, v, s, col) in bricks:
        w, l = s if isinstance(s, tuple) else (s, s)
        cv2.rectangle(bgr, (u - l // 2, v - w // 2), (u + l // 2, v + w // 2), col, -1)
        h[v - w // 2:v + w // 2, u - l // 2:u + l // 2] = 0.019
    if box:
        u, v, s = box
        cv2.rectangle(bgr, (u - s, v - s), (u + s, v + s), (40, 40, 40), -1)
        h[v - s:v + s, u - s:u + s] = 0.040
    # depth = distance along the ray to the raised surface: n.p + d = h  with p = r * z
    uu, vv = np.meshgrid(np.arange(640), np.arange(480))
    r = np.stack([(uu - K["cx"]) / K["fx"], (vv - K["cy"]) / K["fy"], np.ones_like(uu, float)], -1)
    z = (h - PLANE[3]) / (r @ N)
    return bgr, z.astype(np.float32)


class TestKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from secondlook.kinematics import Arm
        cls.arm = Arm()

    def test_ik_reaches_fk_targets(self):
        ref = {"shoulder_pan": -45, "shoulder_lift": 60, "elbow_flex": -60, "wrist_flex": 78, "wrist_roll": -57}
        up = np.array([0.0, 0.0, 1.0])
        for q0 in ({"shoulder_pan": -30, "shoulder_lift": 55, "elbow_flex": -50, "wrist_flex": 80, "wrist_roll": -57},
                   {"shoulder_pan": -60, "shoulder_lift": 70, "elbow_flex": -70, "wrist_flex": 85, "wrist_roll": -57}):
            target = self.arm.tcp(q0)
            q, info = self.arm.ik(target, up, -57, ref, max_tilt_deg=90)
            self.assertLess(info["pos_err_mm"], 3.0)
            for j in q:
                self.assertLessEqual(abs(q[j]), 97.0)

    def test_roll_makes_fingers_close_across_a_long_brick(self):
        from secondlook.kinematics import unit
        ref = {"shoulder_pan": -45, "shoulder_lift": 60, "elbow_flex": -60, "wrist_flex": 78, "wrist_roll": -57}
        up = np.array([0.0, 0.0, 1.0])
        q, _ = self.arm.ik(self.arm.tcp(ref), up, -57, ref)
        for ang in (0, 30, 75, 120):
            long_axis = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
            roll = self.arm.roll_for(q, long_axis, up, elongated=True)
            c = self.arm.closing_axis(self.arm.fk(dict(q, wrist_roll=roll)))
            c = unit(c - up * (c @ up))
            self.assertLess(abs(c @ long_axis), 0.08, f"brick at {ang} deg: fingers not across it")

    def test_units_match_studio_viewer(self):
        lo, hi = self.arm.limits["shoulder_pan"]
        self.assertAlmostEqual(self.arm.to_rad("shoulder_pan", -100), lo)
        self.assertAlmostEqual(self.arm.to_rad("shoulder_pan", 100), hi)


class TestScene(unittest.TestCase):
    def test_plane_fit_recovers_table(self):
        from secondlook import scene
        _, z = synthetic()
        plane, inl = scene.fit_plane(z, K)
        self.assertGreater(inl, 0.95)
        self.assertGreater(abs(np.dot(plane[:3], N)), 0.999)
        self.assertAlmostEqual(plane[3], PLANE[3], delta=0.003)

    def test_finds_bricks_skips_bowls_and_box(self):
        from secondlook import scene
        bgr, z = synthetic(bricks=[(200, 300, 30, (30, 60, 200)), (450, 330, (28, 56), (200, 120, 30))],
                           bowls=[(330, 150, 70, (180, 120, 230))], box=(520, 420, 40))
        plane, _ = scene.fit_plane(z, K)
        res = scene.detect(bgr, z, K, plane)
        centres = sorted((round(b["u"]), round(b["v"])) for b in res["bricks"])
        self.assertEqual(len(centres), 2, res["bricks"])
        self.assertAlmostEqual(centres[0][0], 200, delta=4)
        self.assertEqual([b["color"] for b in res["bowls"]], ["pink"])
        self.assertTrue(any("big" in o["why"] for o in res["obstacles"]), res["obstacles"])
        long_brick = max(res["bricks"], key=lambda b: b["size_mm"][0] / b["size_mm"][1])
        self.assertTrue(long_brick["elongated"])

    def test_brick_touching_the_arm_is_split_off(self):
        import cv2
        from secondlook import scene
        bgr, z = synthetic(bricks=[(300, 200, 30, (30, 40, 200))])            # red brick
        # The arm: a tall blue column touching the brick, with a low blue skirt around it.
        cv2.rectangle(bgr, (318, 150), (380, 250), (200, 90, 20), -1)
        uu, vv = np.meshgrid(np.arange(640), np.arange(480))
        r = np.stack([(uu - K["cx"]) / K["fx"], (vv - K["cy"]) / K["fy"], np.ones_like(uu, float)], -1)
        h = np.zeros((480, 640)); h[150:250, 318:380] = 0.15; h[150:250, 312:318] = 0.03
        h[185:215, 285:315] = np.maximum(h[185:215, 285:315], 0.019)
        z = ((h - PLANE[3]) / (r @ N)).astype(np.float32)
        res = scene.detect(bgr, z, K, scene.fit_plane(z, K)[0])
        self.assertEqual([b["color"] for b in res["bricks"]], ["red"], res["obstacles"])
        self.assertFalse(any(o.get("why") == "too tall" for o in res["obstacles"]))

    def test_brick_in_bowl_not_picked(self):
        from secondlook import scene
        bgr, z = synthetic(bricks=[(330, 150, 30, (30, 60, 200))], bowls=[(330, 150, 70, (200, 140, 20))])
        res = scene.detect(bgr, z, K, scene.fit_plane(z, K)[0])
        self.assertEqual(res["bricks"], [])

    def test_touching_different_coloured_bricks_are_separate(self):
        from secondlook import scene
        for gap in (0, 2):
            bgr, z = synthetic(bricks=[(250, 300, (32, 48), (30, 60, 200)),
                                       (298 + gap, 300, (32, 48), (200, 120, 30))])
            res = scene.detect(bgr, z, K, scene.fit_plane(z, K)[0])
            self.assertEqual(sorted(b["color"] for b in res["bricks"]), ["blue", "red"])

    def test_align_identity(self):
        from secondlook import scene
        _, z = synthetic()
        out = scene.align_depth(z, K, K, np.eye(3), np.zeros(3))
        self.assertLess(np.abs(out[10:-10, 10:-10] - z[10:-10, 10:-10]).max(), 1e-5)

    def test_report_flags_moved_camera_and_bowl(self):
        from secondlook import scene
        res = {"bricks": [], "bowls": [{"color": "pink", "u": 100, "v": 100, "r": 50}], "obstacles": [], "in_bowls": 0}
        tilted = [0.0, -0.5, -0.866, 0.62]
        rep = scene.report(res, tilted, ref={"plane": PLANE, "bowls": {"pink": [160, 100]}})
        text = " ".join(rep["warnings"])
        self.assertIn("camera moved", text)
        self.assertIn("pink bowl moved", text)
        self.assertIn("No blue bowl", text)


class TestHandEye(unittest.TestCase):
    def test_recovers_transform_and_rejects_outlier(self):
        from secondlook.handeye import HandEye
        rng = np.random.default_rng(1)
        a = np.radians(30)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ \
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        t = np.array([0.2, -0.1, 0.6])
        he = HandEye({"plane": PLANE, "K": K, "pairs": []})
        for i in range(8):
            cam = rng.uniform([-0.2, -0.15, 0.55], [0.2, 0.15, 0.7])
            base = R @ cam + t + rng.normal(0, 0.001, 3)
            if i == 3:
                base = base + 0.08                      # one bad pair (wrong brick)
            he.add_pair(cam, base, "drop", refit=False)
        fit = he.fit()
        self.assertLess(fit["mean_mm"], 3.0)
        self.assertEqual(fit["pairs_used"], 7)
        self.assertLess(np.abs(he.R - R).max(), 0.02)

    def test_manual_points_dropped_once_enough_auto_pairs(self):
        from secondlook.handeye import HandEye
        he = HandEye({"plane": PLANE, "K": K, "pairs": []})
        for i in range(4):
            he.add_pair([i * 0.05, 0.01 * i, 0.6], [i * 0.05, 0.01 * i, 0.0], "manual", refit=False)
        self.assertEqual({p["source"] for p in he.used_pairs()}, {"manual"})
        for i in range(4):
            he.add_pair([i * 0.04, 0.02, 0.6 + 0.01 * i], [i * 0.04, 0.02, 0.01 * i], "drop", refit=False)
        self.assertEqual({p["source"] for p in he.used_pairs()}, {"drop"})


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "h.db")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_store_query_label_and_learning(self):
        from secondlook import history
        os.makedirs(os.path.join(self.dir, "runs", "inspections"))
        photo = "20261115-010203-001_0.610_defective.png"
        with open(os.path.join(self.dir, "runs", "inspections", photo), "wb") as f:
            f.write(b"png")
        i = history.add_brick(self.db, run="r1", trial=1, pick_ok=1, score_v1=0.61, verdict="defective",
                              final_bowl="defective", photos=[photo])
        history.add_brick(self.db, run="r1", trial=2, pick_ok=1, score_v1=0.2, verdict="good", final_bowl="good",
                          truth="good")
        self.assertEqual(len(history.bricks(self.db, truth="none")), 1)
        msg = history.set_truth(i, "good", self.db, root=self.dir)
        self.assertIn("1 photo", msg)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "data", "bricks_live", "good", photo)))
        self.assertEqual(len(history.bricks(self.db, truth="wrong")), 1)        # good brick, defective bowl
        history.set_truth(i, "defective", self.db, root=self.dir)               # correction moves the photo
        self.assertFalse(os.path.exists(os.path.join(self.dir, "data", "bricks_live", "good", photo)))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "data", "bricks_live", "defect", photo)))
        history.add_pick(self.db, run="r1", method="ik", u=1, v=2, outcome="grasped")
        history.add_pick(self.db, run="r1", method="ik", u=3, v=4, outcome="missed")
        st = history.learning_stats(self.db)
        self.assertEqual(st["pick_methods"]["ik"], {"grasped": 1, "missed": 1, "slipped": 0})
        self.assertEqual(st["best_threshold"]["missed"], 0)

    def test_photo_names_rejected_if_unsafe(self):
        from secondlook import history
        self.assertEqual(history.file_photos(["../../etc/passwd"], "good", self.dir), 0)


class FakeRobot:
    """Arm that 'moves' instantly; the brick is held if the gripper closes within 15 mm of it."""

    def __init__(self, arm, brick_base):
        self.arm, self.brick, self.q, self.closed_on = arm, np.asarray(brick_base), None, False
        self.log = []

    def go(self, pose):
        self.log.append(pose)

    def move(self, q):
        self.q = dict(q)
        self.log.append("move")

    def gripper(self, open):
        if not open and self.q is not None:
            self.closed_on = np.linalg.norm(self.arm.tcp(self.q) - self.brick) < 0.015
        if open:
            self.closed_on = False
        self.log.append("open" if open else "close")

    def holding(self):
        return self.closed_on

    def observe(self):
        return {"gripper.pos": 20.0 if self.closed_on else 1.0}


class TestPrecisePicker(unittest.TestCase):
    def test_plan_and_pick_a_brick_seen_by_the_camera(self):
        from secondlook.handeye import HandEye
        from secondlook.kinematics import Arm
        from secondlook.precise import IKTablePicker
        arm = Arm()
        ref = {"shoulder_pan": -45, "shoulder_lift": 60, "elbow_flex": -60, "wrist_flex": 78, "wrist_roll": -57}
        target = arm.tcp(ref)
        # Camera frame == base frame rotated 180 deg about x, 0.65 m above the table plane through `target`.
        R = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
        t = np.array([target[0], target[1], target[2] - 0.012 + 0.65])
        he = HandEye({"plane": [0.0, 0.0, -1.0, 0.65], "K": K, "pairs": [], "R": R.tolist(), "t": t.tolist(), "ref": ref})
        p = IKTablePicker.__new__(IKTablePicker)
        p.arm, p.he, p.ref, p.max_tilt, p.above_m = arm, he, ref, 35.0, 0.06
        p.history, p.run, p.last_pick = None, None, None
        grasp_cam = he.to_cam(target)
        brick = {"u": 320.0, "v": 240.0, "grasp_cam": grasp_cam.tolist(), "base": he.to_base(grasp_cam).tolist(),
                 "long_axis_cam": [1.0, 0.0, 0.0], "elongated": True}
        plan = p.plan(brick)
        self.assertTrue(plan["ok"], plan["info"])
        self.assertLess(np.linalg.norm(arm.tcp(plan["pick"]) - target) * 1000, 6.0)
        self.assertGreater(arm.tcp(plan["above"])[2], arm.tcp(plan["pick"])[2] + 0.04)
        robot = FakeRobot(arm, target)
        p.scan = lambda robot, ui=None: (None, [brick])
        p.prefer = p.last = None

        class UI:
            def show(self, **kw):
                pass
        self.assertEqual(p.pick(robot, UI()), "grasped")
        self.assertEqual(p.last_pick["outcome"], "grasped")


class TestNarration(unittest.TestCase):
    def test_sentences_from_measurements(self):
        from secondlook.announce import brick_line, describe_table, summary_line
        self.assertEqual(describe_table([]), "The table is clear.")
        self.assertEqual(describe_table([{"color": "blue"}, {"color": "red"}, {"color": "blue"}]),
                         "Three bricks on the table: two blue and one red.")
        self.assertIn("not a brick", describe_table([{"color": "red"}], obstacles=1))
        self.assertEqual(brick_line("good", left=2, bowl_colour="pink"), "Good brick. Pink bowl. Two left.")
        self.assertEqual(brick_line("defective", left=0, bowl_colour="blue"), "Defect found. Blue bowl. That was the last one.")
        self.assertEqual(summary_line({"good": 3, "defective": 1}, 83),
                         "Table clear. Four bricks sorted in one minute 23 seconds: three good, one defective.")
        self.assertIn("both good", summary_line({"good": 2, "defective": 0}))
        self.assertIn("all three defective", summary_line({"good": 0, "defective": 3}))

    def test_voice_switch_is_shared_through_a_file(self):
        from secondlook import announce
        old = announce.SETTINGS
        with tempfile.TemporaryDirectory() as d:
            announce.SETTINGS = os.path.join(d, "voice.json")
            try:
                self.assertTrue(announce.settings()["on"])
                announce.set_voice(False)
                self.assertFalse(announce.settings()["on"])
            finally:
                announce.SETTINGS = old


class TestCycleExtras(unittest.TestCase):
    def test_wait_time_excluded_and_vision_place_used(self):
        from secondlook.machine import Cell
        from tests.test_machine import CFG, FakeInspector, FakeRobot as R, FakeUI

        class Picker:
            placed = []

            def pick(self, robot, ui):
                ui.waited = getattr(ui, "waited", 0.0) + 100.0       # operator took 100 s
                return "grasped"

            def place(self, robot, bowl):
                self.placed.append(bowl)
                return True

        r, ui, pk = R(), FakeUI(), Picker()
        res = Cell(r, pk, FakeInspector([0.1]), ui, CFG).cycle()
        self.assertLess(res["cycle_s"], 5)
        self.assertGreaterEqual(res["wait_s"], 100)
        self.assertEqual(pk.placed, ["good"])
        self.assertNotIn("bowl_good", r.moves)
        self.assertEqual(res["placed_by"], "vision")


if __name__ == "__main__":
    unittest.main()


class TestSelfCalibratingPick(unittest.TestCase):
    """station.picker: vision_learn puts the first grasped brick back down, learns where it really is, re-picks."""

    def _picker(self, self_calib):
        from secondlook.handeye import HandEye
        from secondlook.kinematics import Arm
        from secondlook.precise import IKTablePicker
        arm = Arm()
        ref = {"shoulder_pan": -45, "shoulder_lift": 60, "elbow_flex": -60, "wrist_flex": 78, "wrist_roll": -57}
        target = arm.tcp(ref)
        R = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
        t = np.array([target[0], target[1], target[2] - 0.012 + 0.65])
        he = HandEye({"plane": [0.0, 0.0, -1.0, 0.65], "K": K, "pairs": [], "R": R.tolist(), "t": t.tolist(), "ref": ref,
                      "fit": {"mean_mm": 9.0, "pairs_used": 6}})
        learned = []
        he.add_pair = lambda cam, base, source, **kw: learned.append(source) or {"mean_mm": 4.0, "pairs_used": 7}
        he.save = lambda *a, **k: None
        p = IKTablePicker.__new__(IKTablePicker)
        p.arm, p.he, p.ref, p.max_tilt, p.above_m = arm, he, ref, 35.0, 0.06
        p.history, p.run, p.last_pick, p.prefer, p.last, p.expect = None, None, None, None, None, None
        p.learn, p.self_calib, p.self_check_every, p._grasps, p._checking = True, self_calib, 3, 0, False
        grasp_cam = he.to_cam(target)
        brick = {"u": 320.0, "v": 240.0, "grasp_cam": grasp_cam.tolist(), "base": he.to_base(grasp_cam).tolist(),
                 "long_axis_cam": [1.0, 0.0, 0.0], "elongated": True}
        scans = []

        def scan(robot, ui=None):
            scans.append(1)
            p._learn_from_putback([brick])
            return None, [brick]
        p.scan = scan
        return p, FakeRobot(arm, target), learned, scans

    class UI:
        def __init__(self):
            self.events = []

        def show(self, **kw):
            pass

        def event(self, text, level="info"):
            self.events.append(text)

    def test_first_grasp_checks_aim_then_repicks(self):
        p, robot, learned, scans = self._picker(True)
        ui = self.UI()
        self.assertEqual(p.pick(robot, ui), "grasped")
        self.assertEqual([x for x in robot.log if x in ("open", "close")], ["open", "close", "open", "open", "close"])
        self.assertEqual(learned, ["putback"])
        self.assertEqual(len(scans), 2)
        self.assertTrue(any("Aim updated" in e for e in ui.events))
        # the next two grasps are normal, the 4th checks again
        for n in (2, 3):
            self.assertEqual(p.pick(robot, ui), "grasped")
            self.assertEqual(learned, ["putback"])
        p.pick(robot, ui)
        self.assertEqual(learned, ["putback", "putback"])

    def test_plain_vision_pick_never_puts_down(self):
        p, robot, learned, scans = self._picker(False)
        self.assertEqual(p.pick(robot, self.UI()), "grasped")
        self.assertEqual(learned, [])
        self.assertEqual(len(scans), 1)
