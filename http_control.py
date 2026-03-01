"""HTTP health-check / shutdown server for Context Service (port 8020)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from config import HTTP_CONTROL_PORT, SERVICE_NAME

_shutdown_cb = None
_server: HTTPServer | None = None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "service": SERVICE_NAME, "port": HTTP_CONTROL_PORT})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/shutdown":
            self._json(200, {"status": "shutting_down"})
            if _shutdown_cb:
                threading.Thread(target=_shutdown_cb, daemon=True).start()
        else:
            self._json(404, {"error": "not found"})


def start(shutdown_callback):
    global _shutdown_cb, _server
    _shutdown_cb = shutdown_callback
    _server = HTTPServer(("0.0.0.0", HTTP_CONTROL_PORT), _Handler)
    t = threading.Thread(target=_server.serve_forever, daemon=True, name="CtxHTTP")
    t.start()
    print(f"✅ [CtxHTTP] Health server on :{HTTP_CONTROL_PORT} (/health, /shutdown)")


def stop():
    global _server
    if _server:
        _server.shutdown()
        _server = None