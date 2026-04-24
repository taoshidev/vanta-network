# developer: rizzo
# Copyright (c) 2024 Taoshi Inc
"""
Unit tests for the GET /hl-traders/<hl_address>/limits endpoint.

Tests the public (no-auth) endpoint that resolves a Hyperliquid address
and returns trading limits (account size, max position per pair, max portfolio,
challenge period status).

Uses a lightweight Flask test client with a mocked entity_client to
isolate endpoint logic from the full RPC stack.
"""
import json
import unittest
from unittest.mock import MagicMock

from flask import Flask
from vali_objects.vali_config import ValiConfig, TradePairCategory
from vali_objects.enums.miner_bucket_enum import MinerBucket


# ==================== Test constants ====================
VALID_HL_ADDRESS = "0x" + "a1b2c3d4" * 5
VALID_HL_ADDRESS_2 = "0x" + "1234567890abcdef" * 2 + "12345678"
ACCOUNT_SIZE = 50_000.0  # Tier 2 (<$200K)

# Expected limits — Tier 2 (SUBACCOUNT_FUNDED, account_size < $200K), crypto
TIER2_POSITIONAL = ValiConfig.TIER_POSITIONAL_LEVERAGE[2][TradePairCategory.CRYPTO]  # 1.0x
TIER2_PORTFOLIO  = ValiConfig.TIER_PORTFOLIO_LEVERAGE[2][TradePairCategory.CRYPTO]   # 2.0x

EXPECTED_MAX_POSITION = ACCOUNT_SIZE * TIER2_POSITIONAL   # 50_000
EXPECTED_MAX_PORTFOLIO = ACCOUNT_SIZE * TIER2_PORTFOLIO   # 100_000

# Expected limits — Tier 1 (SUBACCOUNT_CHALLENGE), crypto
TIER1_POSITIONAL = ValiConfig.TIER_POSITIONAL_LEVERAGE[1][TradePairCategory.CRYPTO]  # 0.5x
TIER1_PORTFOLIO  = ValiConfig.TIER_PORTFOLIO_LEVERAGE[1][TradePairCategory.CRYPTO]   # 1.0x

EXPECTED_CHALLENGE_MAX_POSITION = ACCOUNT_SIZE * TIER1_POSITIONAL  # 25_000
EXPECTED_CHALLENGE_MAX_PORTFOLIO = ACCOUNT_SIZE * TIER1_PORTFOLIO   # 50_000


def _build_limits_data(
    account_size=ACCOUNT_SIZE,
    asset_class="crypto",
    challenge_bucket=None,
):
    """Build a limits_data dict matching the shape returned by get_hl_subaccount_limits_data."""
    return {
        'account_size': account_size,
        'asset_class': asset_class,
        'challenge_bucket': challenge_bucket,
    }


class TestHlTraderLimitsEndpoint(unittest.TestCase):
    """
    Unit tests for the get_hl_trader_limits endpoint method.

    Creates a minimal Flask app and binds the real get_hl_trader_limits method
    with a mocked _entity_client, avoiding the heavy ValidatorRestServer
    constructor.
    """

    def setUp(self):
        from vanta_api.validator_rest_server import ValidatorRestServer

        # Create a bare object without calling __init__
        self.server = object.__new__(ValidatorRestServer)

        # Wire up the mock entity client
        self.mock_entity = MagicMock()
        self.server._entity_client = self.mock_entity

        # Create a minimal Flask app and register the route
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.route("/hl-traders/<hl_address>/limits", methods=["GET"])(self.server.get_hl_trader_limits)
        self.client = self.app.test_client()

    def _get(self, hl_address: str):
        """GET /hl-traders/<hl_address>/limits and return (status_code, parsed_json)."""
        resp = self.client.get(f"/hl-traders/{hl_address}/limits")
        return resp.status_code, json.loads(resp.data)

    # ==================== Happy path — normal (funded) ====================

    def test_success_normal(self):
        """200 with correct HS limits for a funded subaccount."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            challenge_bucket=MinerBucket.SUBACCOUNT_FUNDED.value
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['hl_address'], VALID_HL_ADDRESS)
        self.assertEqual(data['account_size'], ACCOUNT_SIZE)
        self.assertEqual(data['max_position_per_pair_usd'], EXPECTED_MAX_POSITION)
        self.assertEqual(data['max_portfolio_usd'], EXPECTED_MAX_PORTFOLIO)
        self.assertFalse(data['in_challenge_period'])
        self.assertIn('timestamp', data)
        self.assertIsInstance(data['timestamp'], int)

    def test_success_no_challenge_bucket(self):
        """200 with normal limits when challenge_bucket is None."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            challenge_bucket=None
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['max_position_per_pair_usd'], EXPECTED_MAX_POSITION)
        self.assertEqual(data['max_portfolio_usd'], EXPECTED_MAX_PORTFOLIO)
        self.assertFalse(data['in_challenge_period'])

    # ==================== Happy path — challenge period ====================

    def test_success_challenge_period(self):
        """200 with HS-reduced limits for a challenge-period subaccount (÷2 from funded)."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            challenge_bucket=MinerBucket.SUBACCOUNT_CHALLENGE.value
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertTrue(data['in_challenge_period'])
        self.assertEqual(data['max_position_per_pair_usd'], EXPECTED_CHALLENGE_MAX_POSITION)
        self.assertEqual(data['max_portfolio_usd'], EXPECTED_CHALLENGE_MAX_PORTFOLIO)

    # ==================== Response structure ====================

    def test_response_content_type_is_json(self):
        """Response Content-Type is application/json."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data()

        resp = self.client.get(f"/hl-traders/{VALID_HL_ADDRESS}/limits")

        self.assertIn('application/json', resp.content_type)

    def test_no_auth_required(self):
        """Endpoint returns non-401/403 without any auth header."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data()

        resp = self.client.get(f"/hl-traders/{VALID_HL_ADDRESS}/limits")

        self.assertNotIn(resp.status_code, (401, 403))

    def test_hl_address_echoed_in_response(self):
        """The hl_address in the response matches the one in the URL."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data()

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(data['hl_address'], VALID_HL_ADDRESS)

    # ==================== 404 paths ====================

    def test_unknown_hl_address_returns_404(self):
        """Unknown HL address returns 404."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = None

        status, data = self._get(VALID_HL_ADDRESS_2)

        self.assertEqual(status, 404)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'HL address not found')

    # ==================== 500 paths ====================

    def test_lookup_exception_returns_500(self):
        """500 when limits data lookup raises an exception."""
        self.mock_entity.get_hl_subaccount_limits_data.side_effect = RuntimeError("RPC down")

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 500)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Internal error')

    # ==================== 503 path ====================

    def test_entity_client_unavailable_returns_503(self):
        """503 when entity client is not available."""
        self.server._entity_client = None

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 503)
        self.assertIn('error', data)

    # ==================== RPC call ordering ====================

    def test_correct_entity_client_call(self):
        """Verifies the endpoint calls get_hl_subaccount_limits_data with the right arg."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data()

        self._get(VALID_HL_ADDRESS)

        self.mock_entity.get_hl_subaccount_limits_data.assert_called_once_with(VALID_HL_ADDRESS)

    def test_limits_data_not_called_when_entity_client_none(self):
        """Limits data is not fetched when entity client is unavailable."""
        self.server._entity_client = None

        self._get(VALID_HL_ADDRESS)

        self.mock_entity.get_hl_subaccount_limits_data.assert_not_called()

    # ==================== Different account sizes ====================

    def test_custom_account_size(self):
        """Limits scale correctly with a different account size."""
        custom_size = 100_000.0
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            account_size=custom_size,
            challenge_bucket=MinerBucket.SUBACCOUNT_FUNDED.value
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['account_size'], custom_size)
        self.assertEqual(data['max_position_per_pair_usd'], custom_size * TIER2_POSITIONAL)
        self.assertEqual(data['max_portfolio_usd'], custom_size * TIER2_PORTFOLIO)


if __name__ == '__main__':
    unittest.main()
