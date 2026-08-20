# developer: rizzo
# Copyright (c) 2024 Taoshi Inc
"""
Unit tests for the GET /hl-traders/<hl_address>/limits endpoint.

Tests the public (no-auth) endpoint that resolves a Hyperliquid address
and returns trading limits (account size, leverage tier, asset class,
per-class caps, overall portfolio cap, challenge period status).

Every HL subaccount is treated as HL_ALL: the handler ignores the stored
asset_class and always returns the cross-class overall cap plus the
per-class breakdown.

Uses a lightweight Flask test client with a mocked entity_client to
isolate endpoint logic from the full RPC stack.
"""
import json
import unittest
from unittest.mock import MagicMock

from flask import Flask
from vali_objects.vali_config import ValiConfig, TradePairCategory
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vanta_api.validator_rest_server import ValidatorRestServer


# ==================== Test constants ====================
VALID_HL_ADDRESS = "0x" + "a1b2c3d4" * 5
VALID_HL_ADDRESS_2 = "0x" + "1234567890abcdef" * 2 + "12345678"
ACCOUNT_SIZE = 50_000.0  # Tier 2 (<$200K)

# Expected limits — Tier 2 (SUBACCOUNT_FUNDED, account_size < $200K), HL_ALL
TIER2_POSITIONAL = ValidatorRestServer._ENDPOINT_TIER_POSITIONAL_LEVERAGE[2][MinerAssetClass.HL_ALL]   # 0.5x
TIER2_PORTFOLIO  = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.HL_ALL]   # 10.0x

EXPECTED_MAX_POSITION = ACCOUNT_SIZE * TIER2_POSITIONAL   # 25_000
EXPECTED_MAX_PORTFOLIO = ACCOUNT_SIZE * TIER2_PORTFOLIO   # 500_000

# Expected limits — Tier 1 (SUBACCOUNT_CHALLENGE), HL_ALL
TIER1_POSITIONAL = ValidatorRestServer._ENDPOINT_TIER_POSITIONAL_LEVERAGE[1][MinerAssetClass.HL_ALL]   # 0.5x
TIER1_PORTFOLIO  = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[1][MinerAssetClass.HL_ALL]   # 10.0x

EXPECTED_CHALLENGE_MAX_POSITION = ACCOUNT_SIZE * TIER1_POSITIONAL  # 25_000
EXPECTED_CHALLENGE_MAX_PORTFOLIO = ACCOUNT_SIZE * TIER1_PORTFOLIO   # 500_000


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
        self.assertEqual(data['tier'], 2)
        self.assertEqual(data['asset_class'], 'hl_all')
        self.assertEqual(data['max_position_per_pair_usd'], EXPECTED_MAX_POSITION)
        self.assertEqual(data['max_portfolio_usd'], EXPECTED_MAX_PORTFOLIO)
        self.assertIn('max_asset_class_usd', data)
        self.assertFalse(data['in_challenge_period'])
        self.assertIn('timestamp', data)
        self.assertIsInstance(data['timestamp'], int)

    def test_success_no_challenge_bucket(self):
        """challenge_bucket=None (no trades yet) is treated as challenge period (tier 1)."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            challenge_bucket=None
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['tier'], 1)
        self.assertEqual(data['max_position_per_pair_usd'], EXPECTED_CHALLENGE_MAX_POSITION)
        self.assertEqual(data['max_portfolio_usd'], EXPECTED_CHALLENGE_MAX_PORTFOLIO)
        self.assertTrue(data['in_challenge_period'])

    # ==================== Happy path — challenge period ====================

    def test_success_challenge_period(self):
        """200 with HS-reduced limits for a challenge-period subaccount (÷2 from funded)."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            challenge_bucket=MinerBucket.SUBACCOUNT_CHALLENGE.value
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertTrue(data['in_challenge_period'])
        self.assertEqual(data['tier'], 1)
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

    # ==================== Multi-class subaccount (HL_ALL) ====================

    def test_hl_all_response_includes_per_class_breakdown(self):
        """HL_ALL subaccount response carries the max_asset_class_usd dict."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            asset_class="hl_all",
            challenge_bucket=MinerBucket.SUBACCOUNT_FUNDED.value,
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertIn('max_asset_class_usd', data)
        # All five real categories listed, HL_ALL itself not in the breakdown
        self.assertEqual(
            set(data['max_asset_class_usd'].keys()),
            {'crypto', 'forex', 'equities', 'indices', 'commodities'},
        )

    def test_hl_all_overall_cap_from_asset_class_table(self):
        """HL_ALL max_portfolio_usd sources from the HL_ALL entry in TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS."""
        from vali_objects.vali_config import ValiConfig
        from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            asset_class="hl_all",
            challenge_bucket=MinerBucket.SUBACCOUNT_FUNDED.value,
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        expected_overall = ACCOUNT_SIZE * ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.HL_ALL]
        self.assertEqual(data['max_portfolio_usd'], expected_overall)

    def test_hl_all_per_class_values_match_table(self):
        """Each per-class entry matches TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY for the right tier."""
        from vali_objects.vali_config import ValiConfig
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            asset_class="hl_all",
            challenge_bucket=MinerBucket.SUBACCOUNT_FUNDED.value,
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        breakdown = data['max_asset_class_usd']
        for cat_str, cat in (
            ('crypto',     TradePairCategory.CRYPTO),
            ('forex',      TradePairCategory.FOREX),
            ('equities',   TradePairCategory.EQUITIES),
            ('indices',    TradePairCategory.INDICES),
            ('commodities', TradePairCategory.COMMODITIES),
        ):
            expected = ACCOUNT_SIZE * ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][cat]
            self.assertEqual(breakdown[cat_str], expected, f"per-class mismatch for {cat_str}")

    def test_hl_all_challenge_period_uses_tier_1(self):
        """HL_ALL during challenge period uses tier-1 entry in TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS."""
        from vali_objects.vali_config import ValiConfig
        from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            asset_class="hl_all",
            challenge_bucket=MinerBucket.SUBACCOUNT_CHALLENGE.value,
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertTrue(data['in_challenge_period'])
        self.assertEqual(
            data['max_portfolio_usd'],
            ACCOUNT_SIZE * ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[1][MinerAssetClass.HL_ALL],
        )

    def test_stored_single_class_still_gets_full_breakdown(self):
        """Stored asset_class is ignored: every HL subaccount is reported as hl_all
        with the full per-class breakdown."""
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            asset_class="crypto",
            challenge_bucket=MinerBucket.SUBACCOUNT_FUNDED.value,
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['asset_class'], 'hl_all')
        self.assertIn('max_asset_class_usd', data)
        self.assertEqual(data['max_portfolio_usd'], EXPECTED_MAX_PORTFOLIO)

    # ==================== Leverage tier field ====================

    def test_tier_field_by_bucket_and_size(self):
        """`tier` follows get_leverage_tier: challenge → 1; funded by size → 2/3/4."""
        cases = (
            (MinerBucket.SUBACCOUNT_CHALLENGE.value, 50_000.0, 1),
            (MinerBucket.SUBACCOUNT_FUNDED.value, 50_000.0, 2),
            (MinerBucket.SUBACCOUNT_FUNDED.value, 199_999.0, 2),
            (MinerBucket.SUBACCOUNT_FUNDED.value, 200_000.0, 3),   # boundary: >= is tier 3
            (MinerBucket.SUBACCOUNT_FUNDED.value, 250_000.0, 3),
            (MinerBucket.SUBACCOUNT_FUNDED.value, 1_000_000.0, 4), # boundary: >= is tier 4
            (MinerBucket.SUBACCOUNT_FUNDED.value, 2_000_000.0, 4),
        )
        for bucket, size, expected_tier in cases:
            with self.subTest(bucket=bucket, size=size):
                self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
                    account_size=size,
                    challenge_bucket=bucket,
                )

                status, data = self._get(VALID_HL_ADDRESS)

                self.assertEqual(status, 200)
                self.assertEqual(data['tier'], expected_tier)

    # ==================== Deprecated per-pair field ====================

    def test_deprecated_max_position_uses_canonical_base_not_raised_pairs(self):
        """max_position_per_pair_usd reports the minimum active HL_ALL base per tier.

        Pairs with a higher subaccount_tier_base_leverage (e.g. GOLDUSDC at 3.0)
        intentionally diverge from this deprecated class-level stand-in — clients
        must resolve per-pair caps from /trade-pairs instead.
        """
        from vali_objects.vali_config import TradePair
        from vali_objects.utils.leverage_utils import get_tier_positional_leverage
        self.mock_entity.get_hl_subaccount_limits_data.return_value = _build_limits_data(
            challenge_bucket=MinerBucket.SUBACCOUNT_FUNDED.value,
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        endpoint_reported = data['max_position_per_pair_usd'] / ACCOUNT_SIZE
        self.assertEqual(endpoint_reported, TIER2_POSITIONAL)
        # GOLDUSDC's order-path cap is higher than the deprecated stand-in reports
        gold_order_path = get_tier_positional_leverage(2, TradePair.GOLDUSDC)
        self.assertGreater(gold_order_path, endpoint_reported)


if __name__ == '__main__':
    unittest.main()
