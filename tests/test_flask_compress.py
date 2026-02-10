"""
Test Flask-Compress gzip behavior.

Verifies that:
1. Non-pre-compressed endpoints get gzip-compressed when the client sends Accept-Encoding: gzip
2. Pre-compressed endpoints (already have Content-Encoding: gzip) are NOT double-compressed
"""

import gzip
import json
import unittest

from flask import Flask, Response
from flask_compress import Compress


def create_test_app():
    """Create a minimal Flask app with Flask-Compress and two endpoints."""
    app = Flask(__name__)
    Compress(app)

    @app.route("/plain")
    def plain_endpoint():
        """Returns a normal JSON response (should be compressed by Flask-Compress)."""
        data = json.dumps({"status": "ok", "data": "x" * 500})
        return Response(data, content_type="application/json")

    @app.route("/pre-compressed")
    def pre_compressed_endpoint():
        """Returns already-gzipped data with Content-Encoding: gzip set.

        Flask-Compress should skip this response because the header is already present.
        """
        payload = json.dumps({"status": "ok", "data": "y" * 500}).encode()
        compressed = gzip.compress(payload)
        return Response(
            compressed,
            content_type="application/json",
            headers={"Content-Encoding": "gzip"},
        )

    return app


class TestFlaskCompress(unittest.TestCase):
    def setUp(self):
        self.app = create_test_app()
        self.client = self.app.test_client()

    def test_plain_endpoint_gets_compressed(self):
        """A normal endpoint should be gzip-compressed when the client accepts it."""
        response = self.client.get(
            "/plain", headers={"Accept-Encoding": "gzip"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Content-Encoding"), "gzip")

        # Decompress and verify the payload is intact
        decompressed = gzip.decompress(response.data)
        data = json.loads(decompressed)
        self.assertEqual(data["status"], "ok")

    def test_pre_compressed_endpoint_not_double_compressed(self):
        """A pre-compressed endpoint must not be compressed again."""
        response = self.client.get(
            "/pre-compressed", headers={"Accept-Encoding": "gzip"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Content-Encoding"), "gzip")

        # Should decompress exactly once to valid JSON
        decompressed = gzip.decompress(response.data)
        data = json.loads(decompressed)
        self.assertEqual(data["status"], "ok")

    def test_plain_endpoint_no_accept_encoding(self):
        """Without Accept-Encoding: gzip, the response should not be compressed."""
        response = self.client.get("/plain")
        self.assertEqual(response.status_code, 200)
        # Should be readable as plain JSON
        data = json.loads(response.data)
        self.assertEqual(data["status"], "ok")


if __name__ == "__main__":
    unittest.main()
