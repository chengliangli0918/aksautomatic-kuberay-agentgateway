import json
import pathlib
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import app

HERE = pathlib.Path(__file__).parent
CODE_REPLY = "Here is the function:\n```go\nfunc Sum(xs []int) int {\n\ttotal := 0\n\treturn total\n}\n```"
MATH_REPLY = "Differentiate term by term: d/dx(x^3) = 3x^2 and d/dx(2x) = 2, so the derivative is 3x^2 + 2."


class RouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = app.Router(json.loads((HERE / "examples.json").read_text(encoding="utf-8")))

    def test_classifies_held_out_prompts(self):
        # eval.json shares no prompts with examples.json and includes tricky words such as
        # "zip code" and "function of red blood cells". 44 of 45 were right when written.
        evals = json.loads((HERE / "eval.json").read_text(encoding="utf-8"))
        results = [(label, self.router.classify(text)[0], text) for label, texts in evals.items() for text in texts]
        misses = [result for result in results if result[0] != result[1]]
        self.assertGreaterEqual(1 - len(misses) / len(results), 0.9, misses)

    def test_previous_reply_keeps_follow_ups_in_context(self):
        self.assertEqual(self.router.classify("Make it work for floats too.", CODE_REPLY)[0], "code")
        self.assertEqual(self.router.classify("Thanks! What is the capital of France?", CODE_REPLY)[0], "simple")
        self.assertEqual(self.router.classify("What about the second derivative?", MATH_REPLY)[0], "math")
        self.assertEqual(self.router.classify("Great, now write a poem about spring.", MATH_REPLY)[0], "simple")

    def test_http_api(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(self.router))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"

        def post(data: bytes):
            request = urllib.request.Request(base + "/classify", data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)

        status, body = post(json.dumps({"text": "Solve 3x = 12.", "context": ""}).encode())
        self.assertEqual((status, body["label"]), (200, "math"))
        self.assertEqual(set(body["scores"]), {"simple", "code", "math"})
        for bad in (b"{}", b"not json", json.dumps({"text": " "}).encode(), json.dumps({"text": "hi", "context": 3}).encode()):
            self.assertEqual(post(bad)[0], 400, bad)
        with urllib.request.urlopen(base + "/healthz") as response:
            self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
