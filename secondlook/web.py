"""Local web dashboard: serves the operator page, live state as JSON, the latest inspection
image, and accepts button presses. Standard library only; binds to localhost by default."""
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("static") / "index.html"
COMMANDS = {"start", "stop", "good", "defective", "skip", "fix_good", "fix_defect"}


class WebServer:
    def __init__(self, host="127.0.0.1", port=8765):
        self._lock = threading.Lock()
        self._state = {}
        self._frame = None
        self._frame_version = 0
        self._version = 0
        self._streams = []
        self._changed = threading.Condition(self._lock)
        self.stop_requested = threading.Event()   # set the instant STOP arrives, even if the main loop is busy
        self.commands = queue.Queue()
        self._httpd = ThreadingHTTPServer((host, port), self._handler())
        self._httpd.daemon_threads = True
        self._thread = None

    @property
    def url(self):
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()

    def publish(self, state=None, frame_jpeg=None):
        with self._lock:
            if state is not None:
                self._state = dict(state)
            if frame_jpeg is not None:
                self._frame = frame_jpeg
                self._frame_version += 1
            self._version += 1
            self._changed.notify_all()

    def next_command(self):
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def _snapshot(self):
        with self._lock:
            return dict(self._state, frame_version=self._frame_version), self._frame

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body=b"", ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif path == "/access.js":
                    self._send(200, PAGE.with_name("access.js").read_bytes(), "text/javascript; charset=utf-8")
                elif path in ("/api/voice", "/api/aac"):
                    from .announce import access_api
                    code, out = access_api("GET", path)
                    self._send(code, json.dumps(out).encode())
                elif path == "/state":
                    state, _ = server._snapshot()
                    self._send(200, json.dumps(state, default=str).encode())
                elif path == "/events":
                    # Server-sent events: push every state change; heartbeat keeps the connection alive.
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seen, last_body, me = -1, None, object()
                    with server._lock:
                        server._streams.append(me)
                        while len(server._streams) > 3:          # old tabs/reconnects: drop the oldest
                            server._streams.pop(0)
                    try:
                        while me in server._streams:
                            with server._changed:
                                if server._version == seen:
                                    server._changed.wait(timeout=2.0)
                                version = server._version
                            state, _ = server._snapshot()
                            state.pop("clock", None)
                            body = json.dumps(state, default=str, sort_keys=True)
                            # unchanged content -> tiny heartbeat only; changed -> full state
                            self.wfile.write(f"data: {body if body != last_body else '{}'}\n\n".encode())
                            self.wfile.flush()
                            seen, last_body = version, body
                            time.sleep(0.2)                       # at most 5 messages per second
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        with server._lock:
                            if me in server._streams:
                                server._streams.remove(me)
                    return
                elif path == "/frame.jpg":
                    _, frame = server._snapshot()
                    if frame is None:
                        self._send(204)
                    else:
                        self._send(200, frame, "image/jpeg")
                else:
                    self._send(404, b'{"error": "not found"}')

            def do_POST(self):
                if self.path in ("/api/voice", "/api/say"):
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(length) or b"{}")
                    except ValueError:
                        body = {}
                    from .announce import access_api
                    code, out = access_api("POST", self.path, body if isinstance(body, dict) else {})
                    self._send(code, json.dumps(out).encode())
                    return
                if self.path != "/cmd":
                    self._send(404, b'{"error": "not found"}')
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    cmd = json.loads(self.rfile.read(length) or b"{}").get("cmd")
                except (ValueError, AttributeError):
                    cmd = None
                if cmd not in COMMANDS:
                    self._send(400, b'{"error": "unknown command"}')
                    return
                if cmd == "stop":
                    server.stop_requested.set()
                server.commands.put(cmd)
                self._send(200, b'{"ok": true}')

        return Handler
