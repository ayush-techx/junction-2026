"""AccessInspect hook: Gemini is a second opinion for the voice/log only; PatchCore decides the bowl."""
import os
import tempfile
import unittest

from secondlook.machine import Cell
from tests.test_machine import CFG, FakeInspector, FakePicker, FakeRobot


class Done:
    def __init__(self, value):
        self.value = value

    def get(self, timeout=0):
        return self.value


class UI:
    def __init__(self):
        self.said, self.events = [], []

    def show(self, **kw):
        pass

    def say(self, key, count=None, defect=None, color=None):
        self.said.append((key, defect, color))

    def event(self, text, kind="info"):
        self.events.append(text)

    def check_stop(self):
        pass


class Img:                                    # stands in for a frame: has .shape, so a photo is "saved"
    shape = (4, 4, 3)


def run(score, gemini):
    d = tempfile.mkdtemp()
    cfg = dict(CFG, inspection_log=d)
    robot = FakeRobot()
    robot.frame = lambda cam: Img()
    ui = UI()
    cell = Cell(robot, FakePicker(["grasped"]), FakeInspector([score]), ui, cfg)
    cell._save_inspection = lambda frame, crop, s, v: (open(os.path.join(d, "a.png"), "w").close(),
                                                       cell.last_files.append("a.png"))
    asked = []
    cell.advisor = lambda path: (asked.append(path), Done(gemini))[1]
    return cell.cycle(), ui, asked


class TestSecondOpinion(unittest.TestCase):
    def test_names_the_defect_patchcore_found(self):
        res, ui, asked = run(0.9, {"defect": "tape", "color": "red", "confident": True, "reason": "tape"})
        self.assertEqual(res["final_bowl"], "defective")
        self.assertEqual(ui.said[-1], ("defective", "tape", "red"))
        self.assertIn("gemini=tape/red", res["notes"])
        self.assertTrue(asked[0].endswith("a.png"))

    def test_cannot_change_the_bowl(self):
        res, ui, _ = run(0.1, {"defect": "sticker", "color": "blue", "confident": True, "reason": "x"})
        self.assertEqual(res["final_bowl"], "good")            # PatchCore said good -> good bowl
        self.assertEqual(ui.said[-1], ("good", None, "blue"))    # no defect spoken for a passed brick
        self.assertIn("gemini_disagrees", res["notes"])
        self.assertTrue(ui.events)                               # operator is told to check it

    def test_late_or_missing_answer_is_harmless(self):
        res, ui, _ = run(0.9, None)
        self.assertEqual(res["final_bowl"], "defective")
        self.assertEqual(ui.said[-1], ("defective", None, None))
        self.assertIn("gemini=late", res["notes"])


class TestBridge(unittest.TestCase):
    def test_phrase_and_cached_voice(self):
        from secondlook import assist
        if not assist.available():
            self.skipTest("accessinspect/.venv not installed")
        text = assist.phrase("reject", defect="tape", timeout=20)
        self.assertIn("tape", text)
        self.assertTrue(assist.cached_voice("Inspection started."))
        assist.close()


if __name__ == "__main__":
    unittest.main()
