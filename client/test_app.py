import base64
import io
import json
import os
import pathlib
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from PIL import Image
from streamlit.testing.v1 import AppTest

import chat

MODEL = "qwen3.5-4b"
KEY = "test-key"
# The only fields cmd/quota-proxy accepts; a request with any other field is rejected.
ALLOWED_FIELDS = {"model", "messages", "max_tokens", "stream", "temperature"}


def encoded_image(size, image_format="PNG", **save_options):
    buffer = io.BytesIO()
    Image.new("RGB", size, "teal").save(buffer, image_format, **save_options)
    return buffer.getvalue()


def decoded_image(url):
    header, _, data = url.partition(",")
    return header, Image.open(io.BytesIO(base64.b64decode(data)))


class RequestMessagesTest(unittest.TestCase):
    def test_keeps_newest_turns_within_context(self):
        history = []
        for i in range(30):
            history.append({"role": "user", "content": f"question {i} " + "x" * 500})
            history.append({"role": "assistant", "content": "y" * 500, "reservation": (8192, 81920)})
        history.append({"role": "user", "content": "newest"})
        messages, dropped = chat.request_messages("Be brief.", history, 512)
        self.assertEqual(messages[0], {"role": "system", "content": "Be brief."})
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[-1], {"role": "user", "content": "newest"})
        self.assertTrue(all(set(m) == {"role", "content"} for m in messages))
        self.assertLessEqual(chat.token_upper_bound(messages), chat.CONTEXT_TOKENS - 512)
        self.assertEqual(dropped, len(history) - (len(messages) - 1))
        self.assertGreater(dropped, 0)

    def test_caps_message_count(self):
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "hi"} for i in range(301)]
        messages, _ = chat.request_messages("", history, 1)
        self.assertLessEqual(len(messages), chat.MAX_MESSAGES)
        self.assertEqual(messages[0]["role"], "user")

    def test_rejects_message_that_cannot_fit(self):
        # Two UTF-8 bytes per character: 8,000 bytes cannot fit beside 256 output tokens.
        with self.assertRaises(ValueError):
            chat.request_messages("", [{"role": "user", "content": "é" * 4000}], 256)

    def test_counts_bytes_after_nfc_normalization(self):
        # 4,000 raw bytes, but NFC turns each U+1D160 into three 4-byte characters (about
        # 9,000 tokens), which cannot fit the context.
        with self.assertRaises(ValueError):
            chat.request_messages("", [{"role": "user", "content": "\U0001d160" * 1000}], 256)

    def test_sends_images_within_the_image_limit(self):
        image = chat.image_data_url(encoded_image((64, 48)))
        history = [
            {"role": "user", "content": "first", "images": [image, image]},
            {"role": "assistant", "content": "A teal square."},
            {"role": "user", "content": "", "images": [image]},
        ]
        messages, dropped = chat.request_messages("", history, 512)
        self.assertEqual(messages, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}])
        self.assertEqual(dropped, 2)
        self.assertGreaterEqual(chat.token_upper_bound(messages), chat.IMAGE_TOKENS)
        with self.assertRaises(ValueError):
            chat.request_messages("", [{"role": "user", "content": "x", "images": [image] * 3}], 512)


class ImageDataUrlTest(unittest.TestCase):
    def test_scales_large_images_to_the_engine_pixel_limit(self):
        header, image = decoded_image(chat.image_data_url(encoded_image((4000, 3000))))
        self.assertEqual(header, "data:image/png;base64")
        self.assertLessEqual(image.width * image.height, chat.MAX_IMAGE_PIXELS)
        self.assertAlmostEqual(image.width / image.height, 4 / 3, places=2)

    def test_keeps_jpeg_and_applies_exif_rotation(self):
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display.
        header, image = decoded_image(chat.image_data_url(encoded_image((40, 20), "JPEG", exif=exif)))
        self.assertEqual(header, "data:image/jpeg;base64")
        self.assertEqual(image.size, (20, 40))

    def test_rejects_files_that_are_not_images(self):
        with self.assertRaises(ValueError):
            chat.image_data_url(b"not an image")


class ApiBaseUrlTest(unittest.TestCase):
    def test_requires_https_except_on_loopback(self):
        self.assertEqual(chat.api_base_url(" https://example.com/v1/ "), "https://example.com/v1")
        self.assertEqual(chat.api_base_url("http://127.0.0.1:8080"), "http://127.0.0.1:8080/v1")
        for endpoint in ("http://example.com", "https://example.com/other", "example.com"):
            with self.assertRaises(ValueError):
                chat.api_base_url(endpoint)


class MockGateway(ThreadingHTTPServer):
    """Speaks the quota-proxy contract: strict request fields, SSE replies, quota headers."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _GatewayHandler)
        self.chat_requests = []
        self.quota = 0


def _valid_messages(messages):
    """The quota proxy's message rules: text, or text and at most 2 data-URL images from users."""
    images = 0
    for i, m in enumerate(messages):
        if set(m) != {"role", "content"} or not m["content"]:
            return False
        if m["role"] not in ("user", "assistant") and not (m["role"] == "system" and i == 0):
            return False
        if isinstance(m["content"], str):
            continue
        if m["role"] != "user":
            return False
        for part in m["content"]:
            if part.get("type") == "text" and set(part) == {"type", "text"} and part["text"]:
                continue
            image_url = part.get("image_url")
            if not (
                part.get("type") == "image_url"
                and set(part) == {"type", "image_url"}
                and isinstance(image_url, dict)
                and set(image_url) == {"url"}
                and image_url["url"].startswith(("data:image/png;base64,", "data:image/jpeg;base64,"))
            ):
                return False
            images += 1
    return images <= 2 and any(m["role"] == "user" for m in messages)


class _GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        if self.headers.get("Authorization") == f"Bearer {KEY}":
            return True
        self._send_json(401, {"error": {"message": "invalid API key", "type": "Unauthorized"}})
        return False

    def do_GET(self):
        if self._authorized():
            model = {"id": MODEL, "object": "model", "created": 0, "owned_by": "test"}
            self._send_json(200, {"object": "list", "data": [model]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self._authorized():
            return
        self.server.chat_requests.append(body)
        valid = (
            set(body) <= ALLOWED_FIELDS
            and body.get("stream") is True
            and body.get("model") == MODEL
            and 1 <= body.get("max_tokens", 0) <= 512
            and _valid_messages(body.get("messages", []))
        )
        if not valid:
            self._send_json(400, {"error": {"message": "unsupported request", "type": "Bad Request"}})
            return
        if self.server.quota <= 0:
            message = "request, daily token, or daily chargeback budget exceeded"
            self._send_json(429, {"error": {"message": message, "type": "Too Many Requests"}})
            return
        self.server.quota -= 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-Quota-Reserved-Tokens", "8192")
        self.send_header("X-Quota-Reserved-Microusd", "81920")
        self.end_headers()
        for text in ("Hello", " from", " Qwen"):
            chunk = {
                "id": "chunk",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": MODEL,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        # Ray Serve LLM sends the terminator twice.
        self.wfile.write(b"data: [DONE]\n\ndata: [DONE]\n\n")


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.gateway = MockGateway()
        threading.Thread(target=self.gateway.serve_forever, daemon=True).start()
        self.addCleanup(self.gateway.server_close)
        self.addCleanup(self.gateway.shutdown)
        environment = {"INFERENCE_URL": f"http://127.0.0.1:{self.gateway.server_port}", "INFERENCE_API_KEY": KEY}
        patcher = mock.patch.dict(os.environ, environment)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(str(pathlib.Path(__file__).with_name("app.py")), default_timeout=30)

    def send(self, text):
        self.app.chat_input[0].set_value(text).run()
        self.assertFalse(self.app.exception)

    def test_streams_replies_and_leaves_failed_prompts_out_of_context(self):
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.sidebar.selectbox[0].value, MODEL)

        self.gateway.quota = 1
        self.send("first question")
        reply = [m for m in self.app.chat_message if m.name == "assistant"][-1]
        self.assertEqual(reply.markdown[0].value, "Hello from Qwen")
        self.assertIn("Reserved 8,192 tokens", reply.caption[0].value)

        self.send("second question")
        self.assertIn("Caller limit reached (HTTP 429)", self.app.error[0].value)
        self.assertEqual(len(self.gateway.chat_requests), 2, "a rejected request must not be retried")

        self.gateway.quota = 1
        self.send("third question")
        sent = [m["content"] for m in self.gateway.chat_requests[-1]["messages"]]
        self.assertEqual(sent, ["first question", "Hello from Qwen", "third question"])
        self.assertTrue(all(set(body) <= ALLOWED_FIELDS for body in self.gateway.chat_requests))
        # The failed request's error stays in the conversation after later runs.
        self.assertIn("Caller limit reached (HTTP 429)", self.app.error[0].value)

    def test_sends_attached_images(self):
        self.app.run()
        self.gateway.quota = 1
        self.app.chat_input[0].upload("pasted.png", encoded_image((3000, 2000)), "image/png")
        self.send("What is in this picture?")
        content = self.gateway.chat_requests[-1]["messages"][-1]["content"]
        self.assertEqual([part["type"] for part in content], ["image_url", "text"])
        header, image = decoded_image(content[0]["image_url"]["url"])
        self.assertEqual(header, "data:image/png;base64")
        self.assertLessEqual(image.width * image.height, chat.MAX_IMAGE_PIXELS)
        self.assertEqual(content[1]["text"], "What is in this picture?")
        reply = [m for m in self.app.chat_message if m.name == "assistant"][-1]
        self.assertEqual(reply.markdown[0].value, "Hello from Qwen")

    def test_clears_chat_history(self):
        def clear_buttons():
            return [button for button in self.app.button if button.key == "clear_history"]

        self.app.run()
        self.assertEqual(clear_buttons(), [])
        self.gateway.quota = 1
        self.send("first question")
        self.assertEqual(len(clear_buttons()), 1, "the button must appear right after the first reply")
        self.app.button(key="clear_history").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.chat_message), 0)
        self.assertEqual(clear_buttons(), [])


if __name__ == "__main__":
    unittest.main()
