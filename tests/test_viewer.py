"""Public read-only viewer: allowlist, write refusal and scrubbing, against a FAKE control center.

Never point these tests at the real station: the control center's POST endpoints move the robot.
"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from secondlook import viewer


class FakeCenter(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeCenter.calls.append(("GET", self.path))
        path = self.path.split("?")[0]
        if path == "/api/status":
            return self._send({"job": {"mode": "sort", "alive": True, "log": "x.log", "external": ["123 /home/me/python"]},
                               "owners": {"follower arm": [42], "wrist camera": None},
                               "model": {"inspection": "/home/me/exports/model.xml"}, "live": None})
        if path == "/api/models":
            return self._send({"learned": [{"id": "a", "path": "/home/me/models/a"}],
                               "defect": {"path": "/home/me/x.xml", "versions": [{"id": "v", "path": "/home/me/v"}]}})
        if path.startswith("/img/"):
            return self._send(b"\x89PNG fake", "image/png")
        return self._send({"path": self.path})

    def do_POST(self):
        FakeCenter.calls.append(("POST", self.path))
        self._send({"ok": True})


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.center = serve(FakeCenter)
        cls.old = viewer.CENTER
        viewer.CENTER = f"http://127.0.0.1:{cls.center.server_address[1]}"
        viewer._cache.clear()
        cls.view = serve(viewer.Handler)
        cls.base = f"http://127.0.0.1:{cls.view.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        viewer.CENTER = cls.old
        cls.view.shutdown()
        cls.center.shutdown()

    def req(self, path, method="GET"):
        r = urllib.request.Request(self.base + path, method=method, data=b"{}" if method == "POST" else None)
        try:
            with urllib.request.urlopen(r, timeout=5) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def test_writes_never_reach_the_center(self):
        FakeCenter.calls.clear()
        for p in ("/api/start", "/api/stop", "/api/live/cmd", "/api/history/label", "/api/voice", "/api/models/picker"):
            code, _, _ = self.req(p, "POST")
            self.assertEqual(code, 405, p)
        self.assertEqual(self.req("/api/status", "DELETE")[0], 405)
        self.assertFalse([c for c in FakeCenter.calls if c[0] == "POST"])

    def test_private_paths_are_not_served(self):
        for p in ("/api/log", "/api/modes", "/scene/latest.jpg", "/img/full/a.png", "/img/../secondlook/config.yaml",
                  "/secondlook/config.yaml", "/api/start"):
            self.assertEqual(self.req(p)[0], 404, p)

    def test_status_and_models_are_scrubbed(self):
        _, body, headers = self.req("/api/status")
        s = json.loads(body)
        self.assertNotIn("log", s["job"])
        self.assertEqual(s["owners"], {"follower arm": [], "wrist camera": None})
        self.assertNotIn("/home", body.decode())
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        _, body, _ = self.req("/api/models")
        self.assertNotIn("/home", body.decode())

    def test_query_is_filtered_and_capped(self):
        FakeCenter.calls.clear()
        viewer._cache.clear()
        _, body, _ = self.req("/api/history?limit=99999&run=r1&evil=1")
        forwarded = json.loads(body)["path"]
        self.assertIn("limit=200", forwarded)
        self.assertIn("run=r1", forwarded)
        self.assertNotIn("evil", forwarded)

    def test_cache_shields_the_center(self):
        viewer._cache.clear()
        FakeCenter.calls.clear()
        for _ in range(10):
            self.req("/api/trends")
        self.assertEqual(len([c for c in FakeCenter.calls if c[1].startswith("/api/trends")]), 1)

    def test_page_runs_in_public_mode(self):
        code, body, _ = self.req("/config.js")
        self.assertEqual(code, 200)
        self.assertIn(b"window.SL_PUBLIC", body)
        self.assertEqual(self.req("/")[0], 200)
        code, body, _ = self.req("/img/0001_0.5_good.png")
        self.assertEqual((code, body), (200, b"\x89PNG fake"))


if __name__ == "__main__":
    unittest.main()
