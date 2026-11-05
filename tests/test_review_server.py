import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from norm_k.dataset import generate_pairs
from norm_k.review import AnnotationLog, queue_from_pairs
from norm_k.review_server import create_handler


class ReviewServerTests(unittest.TestCase):
    def test_queue_hides_oracle_and_post_requires_token(self):
        items = queue_from_pairs(generate_pairs(2), per_label=1)
        with tempfile.TemporaryDirectory() as temp:
            log = AnnotationLog(Path(temp) / "reviews.jsonl", [item.id for item in items])
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(items, log, token="test-only-token"))
            server.daemon_threads = True
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(base + "/api/queue?reviewer=charlie", timeout=3) as response:
                    payload = json.load(response)
                self.assertEqual(len(payload["items"]), len(items))
                self.assertNotIn("reference_label", payload["items"][0])

                body = json.dumps({"pair_id": items[0].id, "reviewer": "charlie",
                                   "label": "compatible", "confidence": "high", "notes": "can both hold"}).encode()
                request = urllib.request.Request(base + "/api/reviews", data=body,
                                                 headers={"Content-Type": "application/json"}, method="POST")
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request, timeout=3)
                self.assertEqual(error.exception.code, 403)

                request.add_header("X-Review-Token", "test-only-token")
                with urllib.request.urlopen(request, timeout=3) as response:
                    self.assertEqual(response.status, 201)
                self.assertEqual(len(log.read_all()), 1)
                with urllib.request.urlopen(base + "/api/queue?reviewer=charlie", timeout=3) as response:
                    after = json.load(response)
                self.assertEqual(after["completed"][items[0].id], "compatible")

                hostile = urllib.request.Request(base + "/api/queue?reviewer=charlie",
                                                headers={"Host": "example.com"})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(hostile, timeout=3)
                self.assertEqual(error.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
