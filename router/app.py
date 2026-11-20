"""Prompt router for quota-proxy's "auto" model: classifies a prompt as simple, code, or math.

It embeds the prompt with a small CPU sentence-embedding model and picks the label whose
example prompts (examples.json) have the most similar average embedding. The previous reply,
when given, counts for a quarter of the score, so follow-ups stay with their conversation.
Prompts are never logged.
"""

from __future__ import annotations

import json
import os
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime
from fastembed import TextEmbedding

# Prompts reach this service, so it sends nothing anywhere (its NetworkPolicy also blocks egress).
onnxruntime.disable_telemetry_events()

MODEL = "BAAI/bge-small-en-v1.5"
# The model reads at most 512 tokens, so longer text cannot change the result.
MAX_CHARS = 2000
MAX_BODY_BYTES = 32 * 1024
CONTEXT_WEIGHT = 0.25


class Router:
    def __init__(self, examples: dict[str, list[str]], cache_dir: str | None = None):
        self.model = TextEmbedding(MODEL, cache_dir=cache_dir, threads=1)
        self.labels = list(examples)
        centroids = np.stack([self._embed(texts).mean(axis=0) for texts in examples.values()])
        self.centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.model.embed([text[:MAX_CHARS] for text in texts])))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def classify(self, text: str, context: str = "") -> tuple[str, dict[str, float]]:
        scores = self._embed([text])[0] @ self.centroids.T
        if context.strip():
            scores = (1 - CONTEXT_WEIGHT) * scores + CONTEXT_WEIGHT * (self._embed([context])[0] @ self.centroids.T)
        return self.labels[int(scores.argmax())], {label: round(float(s), 4) for label, s in zip(self.labels, scores)}


def make_handler(router: Router) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/classify":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send(411, {"error": "Content-Length required"})
                return
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": "request too large"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                text, context = body["text"], body.get("context", "")
                if not isinstance(text, str) or not text.strip() or not isinstance(context, str):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                self._send(400, {"error": "expected JSON with a nonempty text string and an optional context string"})
                return
            label, scores = router.classify(text, context)
            self._send(200, {"label": label, "scores": scores})

    return Handler


def main() -> None:
    examples = json.loads(pathlib.Path(__file__).with_name("examples.json").read_text(encoding="utf-8"))
    router = Router(examples, cache_dir=os.environ.get("ROUTER_MODEL_CACHE"))
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), make_handler(router))
    print(f"model-router ready: {MODEL}, labels {router.labels}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
