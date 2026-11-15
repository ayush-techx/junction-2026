"""Top-camera table scan: detection, pixel -> joints map, picker flow (no hardware)."""
import os
import tempfile
import unittest

import numpy as np

from secondlook.robot import ARM


def table(bricks=()):
    import cv2
    img = np.full((480, 640, 3), 235, np.uint8)                       # white table
    cv2.rectangle(img, (0, 0), (640, 40), (60, 60, 60), -1)            # robot base strip
    for (u, v, color) in bricks:
        cv2.rectangle(img, (u - 25, v - 25), (u + 25, v + 25), color, -1)
    return img


class TestDetect(unittest.TestCase):
    def test_finds_coloured_bricks_and_ignores_empty_table(self):
        from secondlook.table import detect_bricks
        bg = table()
        self.assertEqual(detect_bricks(bg.copy(), bg), [])
        found = detect_bricks(table([(200, 300, (200, 120, 30)), (450, 200, (40, 180, 90))]), bg)
        self.assertEqual(len(found), 2)
        centres = sorted((round(b["u"]), round(b["v"])) for b in found)
        self.assertEqual(centres, [(200, 300), (450, 200)])

    def test_region_limits_the_search(self):
        from secondlook.table import detect_bricks
        bg = table()
        img = table([(200, 300, (200, 120, 30)), (550, 400, (40, 180, 90))])
        found = detect_bricks(img, bg, region=[[100, 200], [300, 200], [300, 400], [100, 400]])
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["u"], 200, delta=2)


class TestCalibrationChanges(unittest.TestCase):
    def test_removed_brick_and_nudged_bowl_are_ignored(self):
        import cv2
        from secondlook.table import detect_bricks, fixed_objects
        empty = table()
        cv2.circle(empty, (480, 330), 70, (200, 120, 230), -1)              # a bowl
        before = empty.copy()
        cv2.rectangle(before, (175, 275), (225, 325), (200, 120, 30), -1)   # old brick, will be removed
        after = empty.copy()
        cv2.circle(after, (480, 330), 70, (235, 235, 235), -1)
        cv2.circle(after, (490, 335), 70, (200, 120, 230), -1)              # bowl nudged 10 px
        cv2.rectangle(after, (375, 125), (425, 175), (40, 180, 90), -1)     # the NEW brick
        found = detect_bricks(after, empty, previous=before, exclude=fixed_objects(empty))
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["u"], 400, delta=3)
        self.assertAlmostEqual(found[0]["v"], 150, delta=3)


class TestPoseMap(unittest.TestCase):
    def _points(self):
        pts = []
        for u in (150, 320, 490):
            for v in (200, 380):
                pick = {j: 0.05 * u - 0.03 * v + k for k, j in enumerate(ARM)}
                above = {j: pick[j] - 10 for j in ARM}
                pts.append({"u": u, "v": v, "pick": pick, "above": above})
        return pts

    def test_fit_recovers_a_smooth_map_and_saves(self):
        from secondlook.table import PoseMap
        m = PoseMap.fit(self._points())
        p = m.poses(250, 300)
        for k, j in enumerate(ARM):
            self.assertAlmostEqual(p["pick"][j], 0.05 * 250 - 0.03 * 300 + k, places=3)
            self.assertAlmostEqual(p["above"][j], p["pick"][j] - 10, places=3)
        self.assertGreaterEqual(len(m.region), 3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "map.yaml")
            m.save(path)
            self.assertAlmostEqual(PoseMap.load(path).poses(250, 300)["pick"]["shoulder_pan"],
                                   p["pick"]["shoulder_pan"], places=6)

    def test_needs_four_points(self):
        from secondlook.table import PoseMap
        with self.assertRaises(ValueError):
            PoseMap.fit(self._points()[:3])


class FakeRobot:
    def __init__(self, frames):
        self.frames, self.moves, self.grip, self.hold = list(frames), [], [], True

    def go(self, name):
        self.moves.append(name)

    def move(self, pose):
        self.moves.append(dict(pose))

    def frame(self, cam):
        return self.frames.pop(0)

    def gripper(self, open):
        self.grip.append(open)

    def holding(self):
        return self.hold


class FakeUI:
    def show(self, **kw):
        pass


class TestPicker(unittest.TestCase):
    def test_scan_pick_then_table_clear(self):
        import cv2
        from secondlook import table as T
        with tempfile.TemporaryDirectory() as d:
            T.MAP_FILE, T.BACKGROUND_FILE = os.path.join(d, "m.yaml"), os.path.join(d, "bg.png")
            cv2.imwrite(T.BACKGROUND_FILE, table())
            T.PoseMap.fit(TestPoseMap()._points()).save(T.MAP_FILE)
            p = T.TableScanPicker({"cameras": {"top": {}}})
            robot = FakeRobot([table([(320, 300, (200, 120, 30))]), table()])
            self.assertEqual(p.pick(robot, FakeUI()), "grasped")
            self.assertEqual(robot.moves[0], "home")                      # parks before looking
            self.assertEqual(robot.grip, [True, False])                   # open, then close
            self.assertEqual(len([m for m in robot.moves if isinstance(m, dict)]), 3)  # above, pick, above
            self.assertEqual(p.pick(robot, FakeUI()), "empty")
            # a bowl put back on the table (much bigger than a brick) is not a brick
            import cv2 as _cv
            bowl = table(); _cv.circle(bowl, (330, 300), 75, (200, 120, 230), -1)
            robot.frames = [bowl]
            self.assertEqual(p.pick(robot, FakeUI()), "empty")


if __name__ == "__main__":
    unittest.main()
