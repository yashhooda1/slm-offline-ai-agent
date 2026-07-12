"""A local-only web UI for the agent. Python standard library only - no new dependencies.

Streams the agent's think -> tool -> observe trace to the browser over Server-Sent Events,
so you watch the loop run instead of reading about it afterwards.

  python -m src.server        then open http://127.0.0.1:8000

Binds to 127.0.0.1 on purpose: like Ollama itself, that address means "this computer only."
Nothing is served to your network.
"""

import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.agent import build_client, run_agent
from src.config import MAX_STEPS, MODEL
from src.tools import TOOL_FUNCTIONS

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
HOST = "127.0.0.1"
PORT = 8000


class AgentHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console clean for the trace
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/meta":
            self._send_json(
                {
                    "model": MODEL,
                    "max_steps": MAX_STEPS,
                    "tools": sorted(TOOL_FUNCTIONS),
                }
            )
        elif parsed.path == "/run":
            params = urllib.parse.parse_qs(parsed.query)
            task = (params.get("task") or [""])[0].strip()
            self._stream_run(task)
        else:
            self.send_error(404, "No such path")

    # --- responses ------------------------------------------------------------

    def _send_file(self, path: Path, content_type: str):
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, f"Missing {path.name}")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_run(self, task: str):
        """Run one task, pushing every loop event to the browser as it happens."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(kind: str, text: str) -> None:
            frame = json.dumps({"kind": kind, "text": text})
            self.wfile.write(f"data: {frame}\n\n".encode("utf-8"))
            self.wfile.flush()

        if not task:
            emit("error", "Type a task first.")
            emit("done", "")
            return

        try:
            run_agent(task, client=self.server.agent_client, on_event=emit)
        except BrokenPipeError:
            return  # browser navigated away mid-run
        except Exception as exc:
            emit("error", f"{type(exc).__name__}: {exc}")
        emit("done", "")


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), AgentHandler)
    try:
        server.agent_client = build_client()
    except Exception as exc:
        print(f"Could not reach Ollama: {exc}\nIs `ollama serve` running?")
        return 1

    print(f"Agent UI on http://{HOST}:{PORT}  (model={MODEL}, max_steps={MAX_STEPS})")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
