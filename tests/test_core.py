"""Pure-Python parts: calibration, eval log, power reader, config file, CLI wiring."""
import csv
import os
import subprocess
import sys
import tempfile
import unittest

import yaml

from secondlook import eval_log
from secondlook.calibrate import rates, suggest
from secondlook.power import Power

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCalibrate(unittest.TestCase):
    def test_separable(self):
        thr, band, info = suggest([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
        self.assertEqual(info["mode"], "separable")
        self.assertTrue(0.3 < thr < 0.7)
        self.assertEqual(rates([0.1, 0.2, 0.3], [0.7, 0.8, 0.9], thr), (1.0, 0.0))
        self.assertTrue(thr - band >= 0.3 - 1e-9 and thr + band <= 0.7 + 1e-9)

    def test_overlap_picks_best_tradeoff(self):
        good, defect = [0.1, 0.2, 0.55], [0.5, 0.8, 0.9]
        thr, band, info = suggest(good, defect)
        self.assertEqual(info["mode"], "overlap")
        recall, false_rej = rates(good, defect, thr)
        self.assertGreaterEqual(recall - false_rej, 2 / 3 - 1e-9)
        self.assertGreater(band, 0)

    def test_good_only(self):
        thr, band, info = suggest([0.2, 0.4])
        self.assertGreater(thr, 0.4)
        self.assertEqual(info["mode"], "good-only")

    def test_needs_good(self):
        with self.assertRaises(ValueError):
            suggest([], [0.9])


class TestEvalLog(unittest.TestCase):
    def test_log_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "runs", "t.csv")
            eval_log.log(path, truth="good", final_bowl="good", pick_ok=1, cycle_s=10.0,
                         score_v1=0.2, score_v2="", thr=0.5)
            eval_log.log(path, truth="defective", final_bowl="defective", pick_ok=1, cycle_s=12.0,
                         score_v1=0.4, score_v2=0.9, thr=0.5, extra="x")
            eval_log.log(path, truth="defective", final_bowl="good", pick_ok=0, regrasps=1)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["trial"] for r in rows], ["1", "2", "3"])
            self.assertIn("extra=x", rows[1]["notes"])
            s = eval_log.summarize(rows)
            self.assertEqual(s["trials"], 3)
            self.assertEqual(s["pick success"], "67% (2/3)")
            self.assertEqual(s["sort accuracy (of picked)"], "100% (2/2)")
            self.assertEqual(s["caught only by second view"], 1)
            self.assertEqual(s["re-grasps"], 1)
            self.assertEqual(s["cycle time median / p90"], "11.0 s / 12.0 s")
            unlabelled = eval_log.summarize([{"pick_ok": "1", "final_bowl": "good", "truth": ""}])
            self.assertEqual(unlabelled["sort accuracy (of picked)"], "n/a")


class TestPower(unittest.TestCase):
    def test_watts_from_energy_counter(self):
        with tempfile.TemporaryDirectory() as d:
            rapl = os.path.join(d, "energy_uj")
            zone = os.path.join(d, "thermal_zone0")
            os.makedirs(zone)
            with open(os.path.join(zone, "type"), "w") as f:
                f.write("x86_pkg_temp\n")
            with open(os.path.join(zone, "temp"), "w") as f:
                f.write("61000\n")
            with open(rapl, "w") as f:
                f.write("1000000\n")
            p = Power(idle_watts=3.0, rapl=rapl, thermal_glob=os.path.join(d, "thermal_zone*"))
            self.assertIsNone(p.sample())
            p._last = (p._last[0], p._last[1] - 1.0)       # pretend one second passed
            with open(rapl, "w") as f:
                f.write("11000000\n")                       # +10 J
            self.assertAlmostEqual(p.sample(), 10.0, delta=0.5)
            self.assertAlmostEqual(p.above_idle(), 7.0, delta=0.5)
            self.assertEqual(p.temp_c(), 61.0)

    def test_unreadable_is_none(self):
        p = Power(rapl="/nonexistent/energy_uj", thermal_glob="/nonexistent/*")
        self.assertIsNone(p.sample())
        self.assertIsNone(p.temp_c())


class TestConfig(unittest.TestCase):
    def test_config_has_everything_the_code_reads(self):
        with open(os.path.join(ROOT, "secondlook", "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        for key in ("port", "id", "poses_file", "hz", "max_deg_per_s", "gripper"):
            self.assertIn(key, cfg["robot"])
        for key in ("open", "close", "hold_min", "hold_max"):
            self.assertIn(key, cfg["robot"]["gripper"])
        for key in ("camera", "model", "device", "roi", "threshold", "band"):
            self.assertIn(key, cfg["inspection"])
        self.assertIn(cfg["inspection"]["camera"], cfg["cameras"])
        for cam in cfg["policy"]["cameras"]:
            self.assertIn(cam, cfg["cameras"])
        for key in ("pick_retries", "second_looks"):
            self.assertIn(key, cfg["cycle"])
        self.assertIn("idle_watts", cfg["power"])


class TestCli(unittest.TestCase):
    def run_help(self, *args):
        r = subprocess.run([sys.executable, "-m", "secondlook.cell", *args, "--help"],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_all_commands_parse(self):
        self.assertIn("check", self.run_help())
        for cmd in ("check", "teach", "collect", "run", "review"):
            self.run_help(cmd)
        self.assertIn("--handfed", self.run_help("run"))


if __name__ == "__main__":
    unittest.main()
