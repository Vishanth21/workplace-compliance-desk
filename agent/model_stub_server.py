"""Lightweight mock OpenAI-compatible HTTP inference server for demo and hermetic integration (D3)."""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model-stub")

_FAIL_MODE = False


class StubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/health", "/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        global _FAIL_MODE
        if self.path in ("/admin/trip", "/admin/fail"):
            _FAIL_MODE = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"fail_mode": true}')
            return

        if self.path in ("/admin/reset",):
            _FAIL_MODE = False
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"fail_mode": false}')
            return

        if _FAIL_MODE:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"error": "Upstream service temporarily unavailable (simulated)"}'
            )
            return

        if self.path in ("/chat/completions", "/v1/chat/completions"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body) if body else {}

            is_json = payload.get("response_format", {}).get("type") == "json_object"
            content = "OK"

            if is_json:
                if "Verdict" in body or "supported" in body:
                    content = json.dumps(
                        {"supported": True, "reason": "Verified against statutory proviso."}
                    )
                else:
                    content = json.dumps(
                        {
                            "action": "answer",
                            "answer": (
                                "Under the second and sixth provisos to section 53 of the "
                                "Code on Social Security, 2020, gratuity is payable on a "
                                "pro-rata basis for fixed-term employment without requiring "
                                "five years of continuous service."
                            ),
                            "citations": [
                                "css2020-s53-proviso_2",
                                "css2020-s53-proviso_6",
                            ],
                            "reasoning": "Formulated from section 53 statutory provisos.",
                        }
                    )

            reply = {
                "id": "chatcmpl-stub-001",
                "object": "chat.completion",
                "created": 1700000000,
                "model": payload.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        logger.info("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), format % args)


def run(port: int = 8080) -> None:
    server = HTTPServer(("0.0.0.0", port), StubHandler)
    logger.info("Model stub server listening on 0.0.0.0:%d", port)
    server.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    run(port)

