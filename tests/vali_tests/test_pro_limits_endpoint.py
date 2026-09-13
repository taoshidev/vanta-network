"""
Unit tests for GET /subaccounts/<synthetic_hotkey>/limits.

The Vanta-native counterpart to /hl-traders/<hl_address>/limits, which pro accounts cannot reach
because they have no Hyperliquid address. Covers the pro-specific parts: the legacy curve and tier
a pro account is sized against, the correlated-exposure block, and the entity-collateral headroom
(whose "unknown balance" case must not read as "no headroom").

Uses a lightweight Flask test client with mocked clients, like test_hl_trader_limits_endpoint.
"""
import json
import unittest
from unittest.mock import MagicMock

from flask import Flask

from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_manager import CollateralRecord, MinerAccount
from vali_objects.trade_pair import TradePair, TradePairCategory
from vali_objects.utils.leverage_utils import compute_correlated_exposures, get_pro_positional_leverage
from vali_objects.vali_config import ValiConfig
from vanta_api.validator_rest_server import ValidatorRestServer

HOTKEY = "5GhDr3xy" + "a" * 40 + "_0"  # {entity_hotkey}_{subaccount_id}
PRO_ACCOUNT_SIZE = 400_000.0


def _pro_account(exposures=None):
    account = MinerAccount(
        miner_hotkey=HOTKEY,
        asset_class=MinerAssetClass.ALL_MARKETS,
        miner_bucket=MinerBucket.PRO_FUNDED,
    )
    account.collateral_records = [CollateralRecord(PRO_ACCOUNT_SIZE, PRO_ACCOUNT_SIZE / 5000, 0, True)]
    account.correlated_exposure_by_group = exposures or {}
    return account


class TestProSubaccountLimitsEndpoint(unittest.TestCase):

    def setUp(self):
        self.server = object.__new__(ValidatorRestServer)
        self.server._entity_client = MagicMock()
        self.server._miner_account_client = MagicMock()
        self.server._entity_collateral_client = MagicMock()
        # Auth is exercised elsewhere; these tests are about the payload.
        self.server._get_access_error_response = lambda *a, **kw: None

        self.server._entity_client.get_subaccount_dashboard.return_value = {"account_type": "pro"}
        self.server._miner_account_client.get_account.return_value = _pro_account()
        self.server._entity_collateral_client.get_entity_collateral_headroom.return_value = 100.0
        self.server._entity_collateral_client.compute_subaccount_margin_requirement.return_value = 1_234.0

        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.route("/subaccounts/<synthetic_hotkey>/limits", methods=["GET"])(
            self.server.get_subaccount_limits
        )
        self.client = self.app.test_client()

    def _get(self, hotkey=HOTKEY):
        resp = self.client.get(f"/subaccounts/{hotkey}/limits")
        return resp.status_code, json.loads(resp.data)

    def test_pro_account_reports_the_flat_pro_curve_and_its_caps(self):
        status, data = self._get()

        self.assertEqual(status, 200)
        self.assertTrue(data["is_pro"])
        self.assertEqual(data["tier_curve"], "pro")
        # The pro curve is flat, so there is no tier for a client to key a table on.
        self.assertIsNone(data["tier"])
        self.assertEqual(data["portfolio_multiplier"], ValiConfig.PRO_PORTFOLIO_LEVERAGE)
        self.assertAlmostEqual(
            data["max_portfolio_usd"], PRO_ACCOUNT_SIZE * ValiConfig.PRO_PORTFOLIO_LEVERAGE
        )
        self.assertAlmostEqual(
            data["max_asset_class_usd"]["crypto"],
            PRO_ACCOUNT_SIZE * ValiConfig.PRO_CLASS_LEVERAGE[TradePairCategory.CRYPTO],
        )

    def test_per_pair_leverage_matches_the_published_table(self):
        # Spot-check one pair per category against the spec values.
        expected = {
            TradePair.BTCUSDC: 5.0, TradePair.ADAUSDC: 1.5, TradePair.TRXUSDC: 1.0,
            TradePair.EURUSD: 20.0, TradePair.NZDJPY: 10.0,
            TradePair.GOLDUSDC: 8.0, TradePair.SILVERUSDC: 5.0,
            TradePair.SP500USDC: 10.0, TradePair.EWYUSDC: 5.0,
            TradePair.NVDA: 2.0,
        }
        for trade_pair, leverage in expected.items():
            with self.subTest(trade_pair=trade_pair.trade_pair_id):
                self.assertEqual(get_pro_positional_leverage(trade_pair), leverage)

    def test_correlated_exposure_is_reported_with_its_room(self):
        positions_exposure = compute_correlated_exposures([])
        self.assertEqual(positions_exposure, {})

        account = _pro_account({"currency:EUR": [PRO_ACCOUNT_SIZE * 20, 0.0]})
        self.server._miner_account_client.get_account.return_value = account

        _, data = self._get()
        group = data["correlation_limits"]["groups"]["currency:EUR"]
        limit = ValiConfig.PRO_CURRENCY_EXPOSURE_LIMITS["EUR"]
        self.assertEqual(group["limit_multiplier"], limit)
        self.assertAlmostEqual(group["long_room_usd"], PRO_ACCOUNT_SIZE * (limit - 20))
        self.assertAlmostEqual(group["short_room_usd"], PRO_ACCOUNT_SIZE * limit)

    def test_non_pro_account_gets_no_correlation_block(self):
        account = _pro_account()
        account.miner_bucket = MinerBucket.SUBACCOUNT_FUNDED
        self.server._miner_account_client.get_account.return_value = account

        _, data = self._get()
        self.assertNotIn("correlation_limits", data)
        self.assertNotEqual(data["tier_curve"], "pro")

    def test_unknown_entity_balance_is_not_reported_as_zero_headroom(self):
        self.server._entity_collateral_client.get_entity_collateral_headroom.return_value = None

        _, data = self._get()
        self.assertIsNone(data["entity_collateral"]["headroom_theta"])
        self.assertIsNone(data["entity_collateral"]["headroom_usd"])

    def test_headroom_is_converted_to_usd(self):
        _, data = self._get()
        self.assertEqual(
            data["entity_collateral"]["headroom_usd"], 100.0 * ValiConfig.ENTITY_COLLATERAL_CPT_RISK
        )

    def test_non_subaccount_hotkey_is_rejected(self):
        status, _ = self._get("not_a_subaccount")
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
