"""Logic tests for the inspection cycle, using stand-in objects (no hardware, no models).

Run: python -m unittest discover tests
"""
import unittest

from secondlook.decide import final_bowl, verdict
from secondlook.machine import Cell, Stop

CFG = {"inspection": {"camera": "wrist", "threshold": 0.5, "band": 0.08},
       "cycle": {"pick_retries": 1, "second_looks": 1}}


class FakeRobot:
    def __init__(self, holding=True):
        self.moves, self._holding = [], holding

    def go(self, pose):
        self.moves.append(pose)

    def gripper(self, open):
        self.moves.append("open" if open else "close")

    def holding(self):
        return self._holding

    def frame(self, cam):
        return cam


class FakePicker:
    def __init__(self, results):
        self.results = list(results)

    def pick(self, robot, ui):
        return self.results.pop(0)


class FakeInspector:
    def __init__(self, scores):
        self.scores = list(scores)

    def score(self, img):
        return self.scores.pop(0), None, None


class FakeUI:
    def __init__(self, stop_after=None):
        self.said, self.calls, self.stop_after = [], 0, stop_after

    def show(self, **kw):
        pass

    def say(self, key, count=None):
        self.said.append(key)

    def check_stop(self):
        self.calls += 1
        if self.stop_after is not None and self.calls > self.stop_after:
            raise Stop()


class TestDecide(unittest.TestCase):
    def test_verdict_bands(self):
        self.assertEqual(verdict(0.30, 0.5, 0.08), "good")
        self.assertEqual(verdict(0.55, 0.5, 0.08), "uncertain")
        self.assertEqual(verdict(0.70, 0.5, 0.08), "defective")

    def test_fail_safe(self):
        self.assertEqual(final_bowl("uncertain"), "defective")
        self.assertEqual(final_bowl("good"), "good")


class TestCycle(unittest.TestCase):
    def _run_scores(self, scores):
        r, ui = FakeRobot(), FakeUI()
        res = Cell(r, FakePicker(["grasped"] * len(scores)), FakeInspector(scores), ui, CFG).cycle()
        return r, ui, res

    def test_good_brick(self):
        r = FakeRobot()
        res = Cell(r, FakePicker(["grasped"]), FakeInspector([0.2]), FakeUI(), CFG).cycle()
        self.assertEqual(res["final_bowl"], "good")
        self.assertIn("bowl_good", r.moves)
        self.assertEqual(res["regrasps"], 0)

    def test_defect_brick(self):
        r = FakeRobot()
        res = Cell(r, FakePicker(["grasped"]), FakeInspector([0.9]), FakeUI(), CFG).cycle()
        self.assertEqual(res["final_bowl"], "defective")
        self.assertIn("bowl_defective", r.moves)

    def test_uncertain_triggers_second_look(self):
        r, ui = FakeRobot(), FakeUI()
        res = Cell(r, FakePicker(["grasped", "grasped"]), FakeInspector([0.52, 0.85]), ui, CFG).cycle()
        self.assertEqual(res["regrasps"], 1)
        self.assertIn("putback", r.moves)
        self.assertEqual(res["final_bowl"], "defective")
        self.assertIn("uncertain", ui.said)

    def test_uncertain_then_clean_second_look_is_good(self):
        r, ui, res = self._run_scores([0.52, 0.10])
        self.assertEqual(res["regrasps"], 1)
        self.assertEqual(res["final_bowl"], "good")          # false alarm cleared by the second look

    def test_uncertain_then_defect_second_look_is_defective(self):
        r, ui, res = self._run_scores([0.52, 0.80])
        self.assertEqual(res["final_bowl"], "defective")

    def test_brick_slips_before_inspection_is_not_inspected(self):
        class SlipRobot(FakeRobot):
            def __init__(self):
                super().__init__(); self.checks = [True, False, True, True]   # pick ok, slipped at inspect, re-pick ok
            def holding(self):
                return self.checks.pop(0) if self.checks else True
        r, ui = SlipRobot(), FakeUI()
        insp = FakeInspector([0.1])
        res = Cell(r, FakePicker(["grasped", "grasped"]), insp, ui, CFG).cycle()
        self.assertEqual(res["final_bowl"], "good")
        self.assertIn("dropped=1", res["notes"])
        self.assertEqual(insp.scores, [])                    # exactly one inspection: the empty one was skipped

    def test_uncertain_twice_goes_to_defective(self):
        res = Cell(FakeRobot(), FakePicker(["grasped", "grasped"]), FakeInspector([0.52, 0.47]),
                   FakeUI(), CFG).cycle()
        self.assertEqual(res["final_bowl"], "defective")   # worse view 0.52 still uncertain

    def test_missed_then_retry(self):
        ui = FakeUI()
        res = Cell(FakeRobot(), FakePicker(["missed", "grasped"]), FakeInspector([0.2]), ui, CFG).cycle()
        self.assertEqual(res["final_bowl"], "good")
        self.assertIn("missed", ui.said)

    def test_no_brick_after_retries(self):
        res = Cell(FakeRobot(holding=False), FakePicker(["grasped", "grasped"]), FakeInspector([]),
                   FakeUI(), CFG).cycle()
        self.assertIsNone(res)

    def test_stop_raises(self):
        with self.assertRaises(Stop):
            Cell(FakeRobot(), FakePicker(["grasped"]), FakeInspector([0.2]), FakeUI(stop_after=1), CFG).cycle()


if __name__ == "__main__":
    unittest.main()
