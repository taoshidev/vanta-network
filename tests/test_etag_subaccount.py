"""
Test ETag / 304 Not Modified behavior for the entity subaccount endpoint.

Tests the ETag logic extracted from ValidatorRestServer.get_subaccount_dashboard()
using a minimal Flask app with a mock entity client.
"""

import json
import hashlib
import time
import unittest
from unittest.mock import MagicMock
from flask import Flask, request, Response, jsonify


def create_test_app(mock_entity_client):
    """Create a minimal Flask app with the subaccount endpoint using ETag logic.

    Mirrors the logic in ValidatorRestServer.get_subaccount_dashboard() but
    uses stdlib json.JSONEncoder instead of CustomEncoder (test data is plain
    dicts so the output is identical).
    """
    app = Flask(__name__)

    @app.route("/entity/subaccount/<synthetic_hotkey>", methods=["GET"])
    def get_subaccount_dashboard(synthetic_hotkey):
        dashboard_data = mock_entity_client.get_subaccount_dashboard_data(synthetic_hotkey)

        if dashboard_data:
            # Serialize the dashboard payload (excluding the timestamp wrapper which changes every call)
            dashboard_json = json.dumps(dashboard_data, sort_keys=True)

            # Compute ETag from dashboard content
            etag = '"' + hashlib.md5(dashboard_json.encode()).hexdigest() + '"'

            # Check If-None-Match
            if_none_match = request.headers.get('If-None-Match')
            if if_none_match == etag:
                return Response(status=304, headers={'ETag': etag})

            # Build full response with ETag
            response_data = json.dumps({
                'status': 'success',
                'dashboard': dashboard_data,
                'timestamp': int(time.time() * 1000)
            })

            response = Response(response_data, status=200, content_type='application/json')
            response.headers['ETag'] = etag
            return response
        else:
            return jsonify({'error': f'Subaccount {synthetic_hotkey} not found'}), 404

    return app


SAMPLE_DASHBOARD = {
    "subaccount_info": {
        "synthetic_hotkey": "entity_alpha_0",
        "entity_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "subaccount_id": 0,
        "status": "active",
        "created_at_ms": 1700000000000,
        "eliminated_at_ms": None,
    },
    "challenge_period": {
        "bucket": "crypto",
        "start_time_ms": 1700000000000,
    },
    "positions": {
        "positions": [],
        "total_leverage": 0.0,
    },
    "statistics": {
        "total_pnl": 0.05,
        "sharpe_ratio": 1.2,
    },
}


class TestETagSubaccountEndpoint(unittest.TestCase):
    def setUp(self):
        self.mock_entity_client = MagicMock()
        self.mock_entity_client.get_subaccount_dashboard_data.return_value = SAMPLE_DASHBOARD
        self.app = create_test_app(self.mock_entity_client)
        self.client = self.app.test_client()

    def test_first_request_returns_200_with_etag(self):
        """First request should return 200 with an ETag header."""
        response = self.client.get("/entity/subaccount/entity_alpha_0")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ETag", response.headers)

        etag = response.headers["ETag"]
        self.assertTrue(etag.startswith('"') and etag.endswith('"'),
                        f"ETag should be quoted per RFC 7232, got: {etag}")

        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("dashboard", data)
        self.assertIn("timestamp", data)

    def test_second_request_with_matching_etag_returns_304(self):
        """Second request with matching If-None-Match should return 304 with empty body."""
        # First request to get the ETag
        response1 = self.client.get("/entity/subaccount/entity_alpha_0")
        self.assertEqual(response1.status_code, 200)
        etag = response1.headers["ETag"]

        # Second request with If-None-Match
        response2 = self.client.get(
            "/entity/subaccount/entity_alpha_0",
            headers={"If-None-Match": etag}
        )
        self.assertEqual(response2.status_code, 304)
        self.assertEqual(response2.headers["ETag"], etag)
        self.assertEqual(response2.data, b"", "304 response should have empty body")

    def test_mismatched_etag_returns_200(self):
        """Request with non-matching If-None-Match should return 200."""
        response = self.client.get(
            "/entity/subaccount/entity_alpha_0",
            headers={"If-None-Match": '"bogus-etag-value"'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("ETag", response.headers)

    def test_changed_data_invalidates_etag(self):
        """When dashboard data changes, the ETag should change and 200 is returned."""
        # First request
        response1 = self.client.get("/entity/subaccount/entity_alpha_0")
        etag1 = response1.headers["ETag"]

        # Change the data
        changed_dashboard = {**SAMPLE_DASHBOARD, "statistics": {"total_pnl": 0.10, "sharpe_ratio": 1.5}}
        self.mock_entity_client.get_subaccount_dashboard_data.return_value = changed_dashboard

        # Second request with old ETag — should get 200 with new ETag
        response2 = self.client.get(
            "/entity/subaccount/entity_alpha_0",
            headers={"If-None-Match": etag1}
        )
        self.assertEqual(response2.status_code, 200)
        etag2 = response2.headers["ETag"]
        self.assertNotEqual(etag1, etag2, "ETag should change when data changes")

    def test_no_data_returns_404(self):
        """When entity client returns None, endpoint should return 404."""
        self.mock_entity_client.get_subaccount_dashboard_data.return_value = None
        response = self.client.get("/entity/subaccount/entity_alpha_0")
        self.assertEqual(response.status_code, 404)

    def test_etag_is_deterministic_for_same_data(self):
        """Same data should always produce the same ETag (sort_keys=True ensures this)."""
        response1 = self.client.get("/entity/subaccount/entity_alpha_0")
        response2 = self.client.get("/entity/subaccount/entity_alpha_0")
        self.assertEqual(response1.headers["ETag"], response2.headers["ETag"])

    def test_etag_excludes_timestamp(self):
        """ETag should not change even though timestamp changes between requests."""
        # Both requests will have different timestamps but same dashboard data
        response1 = self.client.get("/entity/subaccount/entity_alpha_0")
        response2 = self.client.get("/entity/subaccount/entity_alpha_0")

        etag1 = response1.headers["ETag"]
        etag2 = response2.headers["ETag"]
        self.assertEqual(etag1, etag2, "ETag should be stable despite timestamp changes")

        # But the response bodies should have different timestamps
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)
        # Timestamps may or may not differ (depends on speed), but ETags must be equal
        self.assertEqual(data1["dashboard"], data2["dashboard"])

    def test_different_hotkeys_same_data_same_etag(self):
        """Different hotkeys returning the same data should produce the same ETag."""
        response1 = self.client.get("/entity/subaccount/entity_alpha_0")
        response2 = self.client.get("/entity/subaccount/entity_beta_1")
        # Same mock data → same ETag
        self.assertEqual(response1.headers["ETag"], response2.headers["ETag"])

    def test_no_if_none_match_header_returns_200(self):
        """Request without If-None-Match should always return 200."""
        response = self.client.get("/entity/subaccount/entity_alpha_0")
        self.assertEqual(response.status_code, 200)

    def test_response_content_type_is_json(self):
        """200 response should have application/json content type."""
        response = self.client.get("/entity/subaccount/entity_alpha_0")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.content_type)


if __name__ == "__main__":
    unittest.main()
