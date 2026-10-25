"""HTTP server: serves the dashboard and a JSON inference endpoint."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_BODY_BYTES = 16 * 1024
MAX_PROMPT_CHARS = 500


def make_handler(engine):
    index_html = (WEB_DIR / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "MicroTransformer/1.0"

        def log_message(self, format, *args):  # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, index_html, "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/forward":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                self._send_json(400, {"error": "missing or oversized request body"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                prompt = str(body.get("prompt", ""))[:MAX_PROMPT_CHARS]
                result = engine.generate(prompt, max_new_tokens=6, temperature=0.0)
            except (ValueError, AttributeError, TypeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, result)

    return Handler


def make_server(engine, host: str = "127.0.0.1", port: int = 5095) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(engine))
