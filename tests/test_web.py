"""Dashboard server and the web-mode UI: page, state, frame, buttons, STOP."""
import json
import os
import unittest
import urllib.error
import urllib.request

from secondlook.machine import Stop
from secondlook.ui import UI
from secondlook.web import WebServer


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


class TestWebServer(unittest.TestCase):
    def setUp(self):
        self.web = WebServer(port=0).start()

    def tearDown(self):
        self.web.stop()

    def test_page_state_frame(self):
        status, ctype, body = get(self.web.url)
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"SecondLook", body)
        self.assertEqual(get(self.web.url + "frame.jpg")[0], 204)

        self.web.publish({"stage": "PICK", "score": 0.4}, b"\xff\xd8jpeg")
        _, _, body = get(self.web.url + "state")
        state = json.loads(body)
        self.assertEqual(state["stage"], "PICK")
        self.assertEqual(state["frame_version"], 1)
        status, ctype, body = get(self.web.url + "frame.jpg?v=1")
        self.assertEqual((status, ctype, body), (200, "image/jpeg", b"\xff\xd8jpeg"))

    def test_commands(self):
        self.assertEqual(post(self.web.url + "cmd", {"cmd": "start"}), 200)
        self.assertEqual(post(self.web.url + "cmd", {"cmd": "rm -rf"}), 400)
        self.assertEqual(post(self.web.url + "cmd", {}), 400)
        self.assertEqual(self.web.next_command(), "start")
        self.assertIsNone(self.web.next_command())

    def test_stop_is_instant_and_events_stream(self):
        import threading, time
        self.assertEqual(post(self.web.url + "cmd", {"cmd": "stop"}), 200)
        self.assertTrue(self.web.stop_requested.is_set())
        self.assertEqual(post(self.web.url + "cmd", {"cmd": "fix_good"}), 200)
        got = []
        def read():
            with urllib.request.urlopen(self.web.url + "events", timeout=5) as r:
                for line in r:
                    if line.startswith(b"data:") and b"stage" in line:
                        got.append(json.loads(line[5:]))
                        return
        t = threading.Thread(target=read, daemon=True); t.start()
        time.sleep(0.3)
        self.web.publish({"stage": "INSPECT"})
        t.join(4)
        self.assertTrue(got and got[0]["stage"] == "INSPECT")

    def test_unknown_path(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            get(self.web.url + "etc/passwd")
        self.assertEqual(e.exception.code, 404)


CFG = {"ui": {"mode": "web", "port": 0, "open_browser": False, "voice": False},
       "power": {"idle_watts": None},
       "inspection": {"threshold": 0.5, "band": 0.08, "device": "NPU"},
       "policy": {"device": "GPU"}}


class TestWebUI(unittest.TestCase):
    def setUp(self):
        self.ui = UI(CFG)

    def tearDown(self):
        self.ui.close()

    def test_state_and_history_reach_the_page(self):
        self.ui.show(stage="INSPECT", verdict="defective", score=0.83, counts={"good": 2, "defective": 1})
        self.ui.show(stage="DONE", cycle_s=13.4)
        state = json.loads(get(self.ui.web.url + "state")[2])
        self.assertEqual(state["verdict"], "defective")
        self.assertEqual(state["anomaly_dev"], "NPU")
        self.assertEqual(state["counts"], {"good": 2, "defective": 1})
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(state["history"][0]["verdict"], "defective")

    def test_buttons_map_to_keys_and_stop(self):
        post(self.ui.web.url + "cmd", {"cmd": "start"})
        self.assertEqual(self.ui.wait_key({ord(" ")}), ord(" "))
        post(self.ui.web.url + "cmd", {"cmd": "defective"})
        self.assertEqual(self.ui.tick(), ord("d"))
        post(self.ui.web.url + "cmd", {"cmd": "stop"})
        with self.assertRaises(Stop):
            self.ui.tick()


class TestPageFile(unittest.TestCase):
    def test_page_is_self_contained(self):
        page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "secondlook", "static", "index.html")
        with open(page) as f:
            html = f.read()
        self.assertNotIn("http://", html.replace("http://127", ""))   # no external assets: works offline
        self.assertNotIn("https://", html)
        for element in ('id="b-start"', 'id="b-stop"', 'aria-live="assertive"', 'id="view"', 'EventSource', 'id="b-fixgood"'):
            self.assertIn(element, html)


if __name__ == "__main__":
    unittest.main()
