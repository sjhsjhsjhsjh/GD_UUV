"""HTTP server for the Bellhop HTML monitor."""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict

try:
    from .data_loader import build_env_payload, build_manifest
except ImportError:
    # Support direct execution: python monitor/app.py
    from data_loader import build_env_payload, build_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"


def start_monitor_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    outputs_root: str | Path | None = None,
    open_browser: bool = True,
) -> ThreadingHTTPServer:
    """Start the monitor server and return the HTTPServer instance.

    Args:
        host: Bind host for the local server.
        port: Bind port for the local server.
        outputs_root: Env cache root. Defaults to the project env_output root.
        open_browser: Whether to open the monitor page in a browser.

    Returns:
        The started ThreadingHTTPServer instance.
    """

    handler = _build_handler(outputs_root)
    server = ThreadingHTTPServer((host, port), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if open_browser:
        webbrowser.open(f"http://{host}:{port}/")

    return server


def main() -> None:
    """Start the monitor server using the latest env cache directory."""

    server = start_monitor_server()
    host, port = server.server_address
    print(f"Monitor running at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def _build_handler(outputs_root: str | Path | None) -> type[BaseHTTPRequestHandler]:
    class MonitorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._serve_html()
                return
            if self.path == "/api/manifest":
                self._serve_json(build_manifest(outputs_root))
                return
            if self.path.startswith("/api/env/"):
                env_id = self.path[len("/api/env/"):].strip()
                try:
                    payload = build_env_payload(env_id, outputs_root)
                except FileNotFoundError as exc:
                    self._serve_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self._serve_json(payload)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def _serve_html(self) -> None:
            html = HTML_PATH.read_text(encoding="utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        def _serve_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return MonitorHandler


if __name__ == "__main__":
    main()
