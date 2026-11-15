"""Parts that need numpy / OpenCV / OpenVINO / Anomalib. Skipped where a library is missing,
so the same suite runs on a dev machine and on the robot laptop.

Full training smoke test (downloads backbone weights, ~1-2 min):
    SECONDLOOK_TRAIN_SMOKE=1 anomalib-env/bin/python -m unittest tests.test_runtime_libs
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
has = lambda m: importlib.util.find_spec(m) is not None  # noqa: E731


@unittest.skipUnless(has("numpy"), "numpy not installed")
class TestNumpyParts(unittest.TestCase):
    def test_crop_and_fields(self):
        import numpy as np
        from secondlook.inspection import _field, crop
        img = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        self.assertEqual(crop(img, [100, 0, 440, 480]).shape, (480, 440, 3))
        self.assertIs(crop(img, None), img)
        self.assertEqual(_field({"pred_score": 0.7}, "pred_score"), 0.7)

        class Batch:
            pred_score = np.array([0.3])
        self.assertEqual(float(_field(Batch(), "pred_score")[0]), 0.3)

    def test_policy_obs_and_action_mapping(self):
        """Inputs match Studio's model worker; chunks are replayed; commands are clamped."""
        import numpy as np
        from secondlook.policy import PolicyPicker
        from secondlook.robot import JOINTS
        p = PolicyPicker.__new__(PolicyPicker)            # skip model loading
        p.state_name, p.images, p.task, p.replan, p._queue, p.start_pose = "state", {"images.top": "top", "images.wrist": "wrist"}, None, 2, [], None
        raw = {f"{j}.pos": float(i) for i, j in enumerate(JOINTS)}
        raw["top"] = raw["wrist"] = np.zeros((480, 640, 3), np.uint8)
        obs = p.to_obs(raw)
        self.assertEqual(obs["state"].shape, (1, 6))
        self.assertEqual(obs["state"][0].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(obs["images.top"].shape, (1, 480, 640, 3))
        self.assertEqual(set(obs), {"state", "images.top", "images.wrist"})
        p.task = "pick up the brick"
        self.assertEqual(p.to_obs(raw)["task"], ["pick up the brick"])

        calls = []

        class Model:
            def predict_action_chunk(self, o):
                calls.append(1)
                return (np.arange(6, dtype=np.float32) * 30)[None, None].repeat(5, axis=1)   # (1, 5, 6)
        p.model = Model()
        a = p.act(raw)
        self.assertEqual(a["gripper"], 100.0)               # 150 clamped to the safe range
        self.assertEqual(a["shoulder_lift"], 30.0)
        p.act(raw)
        self.assertEqual(len(calls), 1)                     # second step comes from the same chunk
        p.act(raw)
        self.assertEqual(len(calls), 2)                     # replan after `replan` steps
        self.assertIsNotNone(p.last_ms)

    def test_missing_model_paths_explain_themselves(self):
        from secondlook.inspection import Inspector
        from secondlook.policy import PolicyPicker
        with self.assertRaisesRegex(FileNotFoundError, "train_anomaly"):
            Inspector({"inspection": {"model": None, "device": "CPU", "roi": None}})
        with self.assertRaisesRegex(FileNotFoundError, "handfed"):
            PolicyPicker({"policy": {"path": None}, "robot": {"hz": 30}})


@unittest.skipUnless(has("cv2") and has("numpy"), "OpenCV not installed")
class TestHud(unittest.TestCase):
    def test_render(self):
        import numpy as np
        from secondlook.hud import overlay_heatmap, render
        img = np.full((480, 440, 3), 200, np.uint8)
        heat = np.zeros((64, 64), np.float32)
        heat[20:30, 20:30] = 1.0
        self.assertEqual(overlay_heatmap(img, heat, vmax=1.0).shape, img.shape)
        frame = render([(img, heat, 0.9), (img, None, 0.2)],
                       {"stage": "INSPECT", "verdict": "defective", "score": 0.9, "thr": 0.5, "band": 0.08,
                        "counts": {"good": 1, "defective": 2}, "policy_ms": 40, "anomaly_ms": 6,
                        "watts": 4.2, "temp_c": 60, "cycle_s": 12.0})
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertGreater(frame.std(), 10)


@unittest.skipUnless(has("openvino") and has("numpy"), "OpenVINO not installed")
class TestOpenVINO(unittest.TestCase):
    def test_raw_inspector_on_tiny_model(self):
        """Builds a 2-output model shaped like an Anomalib export and runs it through Inspector."""
        import numpy as np
        import openvino as ov
        from openvino import opset13 as ops
        x = ops.parameter([1, 3, 64, 64], np.float32, name="input")
        heat = ops.reduce_mean(x, ops.constant(np.array([1], np.int64)), keep_dims=True)
        score = ops.reduce_max(heat, ops.constant(np.array([1, 2, 3], np.int64)), keep_dims=False)
        heat.output(0).set_names({"anomaly_map"})
        score.output(0).set_names({"pred_score"})
        model = ov.Model([score, heat], [x], "tiny")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.xml")
            ov.save_model(model, path)
            from secondlook import inspection
            orig = sys.modules.get("anomalib")
            sys.modules["anomalib"] = None                 # force the plain-OpenVINO path
            try:
                insp = inspection.Inspector({"inspection": {"model": path, "device": "CPU", "roi": None}})
            finally:
                if orig is None:
                    sys.modules.pop("anomalib", None)
                else:
                    sys.modules["anomalib"] = orig
            bright = np.full((120, 120, 3), 255, np.uint8)
            s, h, c = insp.score(bright)
            self.assertAlmostEqual(s, 1.0, places=3)
            self.assertEqual(h.shape, (64, 64))
            self.assertIsNotNone(insp.last_ms)


@unittest.skipUnless(has("anomalib") and os.environ.get("SECONDLOOK_TRAIN_SMOKE"), "set SECONDLOOK_TRAIN_SMOKE=1 in anomalib-env")
class TestTrainSmoke(unittest.TestCase):
    def test_train_and_export_padim(self):
        from PIL import Image, ImageDraw
        with tempfile.TemporaryDirectory() as d:
            for label, n in (("good", 12), ("defect", 4)):
                os.makedirs(os.path.join(d, label))
                for i in range(n):
                    img = Image.new("RGB", (256, 256), (230, 230, 235))
                    dr = ImageDraw.Draw(img)
                    dr.rectangle([60 + i % 3, 80, 196, 176], fill=(230, 200, 60))
                    if label == "defect":
                        dr.line([70, 90, 180, 170], fill=(20, 20, 20), width=6)
                    img.save(os.path.join(d, label, f"{i}.png"))
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "train_anomaly.py"),
                                "--root", d, "--model", "padim", "--no-int8"],
                               cwd=d, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            xml = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(d, "exports")) for f in fs
                   if f == "model.xml"]
            self.assertTrue(xml, "no OpenVINO model exported")


if __name__ == "__main__":
    unittest.main()
