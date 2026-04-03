# developer: rizzo
# Copyright (c) 2024 Taoshi Inc
"""
Unit tests for the GET /hl-traders/<hl_address> endpoint.

Tests the public (no-auth) endpoint that resolves a Hyperliquid address
to a synthetic hotkey and returns trader info (positions, drawdown,
account size, payout address).

Uses a lightweight Flask test client with a mocked entity_client to
isolate endpoint logic from the full RPC stack.
"""
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from flask import Flask, jsonify, Response
from time_util.time_util import TimeUtil
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from vali_objects.vali_config import ValiConfig


# ==================== Test constants ====================
VALID_HL_ADDRESS = "0x" + "a1b2c3d4" * 5
VALID_HL_ADDRESS_2 = "0x" + "1234567890abcdef" * 2 + "12345678"
SYNTHETIC_HOTKEY = "entity_alpha_0"
VALID_PAYOUT_ADDRESS = "0x" + "deadbeef" * 5


def _build_dashboard(
    account_size=50_000,
    payout_address=VALID_PAYOUT_ADDRESS,
    positions=None,
    statistics=None,
    ledger=None,
    challenge_period=None,
    account_size_data=None,
):
    """Build a dashboard dict matching the shape returned by get_subaccount_dashboard_data."""
    return {
        'subaccount_info': {
            'synthetic_hotkey': SYNTHETIC_HOTKEY,
            'entity_hotkey': 'entity_alpha',
            'subaccount_id': 0,
            'status': 'active',
            'created_at_ms': 1700000000000,
            'eliminated_at_ms': None,
            'account_size': account_size,
            'asset_class': 'crypto',
            'hl_address': VALID_HL_ADDRESS,
            'payout_address': payout_address,
        },
        'challenge_period': challenge_period,
        'ledger': ledger,
        'positions': positions,
        'account_size_data': account_size_data,
        'statistics': statistics,
        'elimination': None,
    }


class TestHlTraderEndpoint(unittest.TestCase):
    """
    Unit tests for the get_hl_trader endpoint method.

    Creates a minimal Flask app and binds the real get_hl_trader method
    with a mocked _entity_client, avoiding the heavy ValidatorRestServer
    constructor.
    """

    def setUp(self):
        # Import the class but don't instantiate (too many deps)
        from vanta_api.validator_rest_server import ValidatorRestServer

        # Create a bare object without calling __init__
        self.server = object.__new__(ValidatorRestServer)

        # Wire up the mock entity client
        self.mock_entity = MagicMock()
        self.server._entity_client = self.mock_entity

        # Create a minimal Flask app and register the route
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.route("/hl-traders/<hl_address>", methods=["GET"])(self.server.get_hl_trader)
        self.client = self.app.test_client()

    def _get(self, hl_address: str):
        """GET /hl-traders/<hl_address> and return (status_code, parsed_json)."""
        resp = self.client.get(f"/hl-traders/{hl_address}")
        return resp.status_code, json.loads(resp.data)

    # ==================== Happy path ====================

    def test_success_basic(self):
        """200 with correct structure for a known HL address."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard()

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['synthetic_hotkey'], SYNTHETIC_HOTKEY)
        self.assertEqual(data['hl_address'], VALID_HL_ADDRESS)
        self.assertEqual(data['account_size'], 50_000)
        self.assertEqual(data['payout_address'], VALID_PAYOUT_ADDRESS)
        self.assertIn('timestamp', data)
        self.assertIsInstance(data['timestamp'], int)

    def test_success_with_positions(self):
        """Positions data is forwarded when present in dashboard."""
        positions = {'n_positions': 5, 'total_leverage': 0.5}
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(positions=positions)

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['positions']['n_positions'], 5)
        self.assertEqual(data['positions']['total_leverage'], 0.5)

    def test_success_no_positions(self):
        """Positions is None when trader has no positions."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(positions=None)

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertIsNone(data['positions'])

    def test_success_no_payout_address(self):
        """Payout address is None when not set."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(payout_address=None)

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertIsNone(data['payout_address'])

    def test_response_content_type_is_json(self):
        """Response Content-Type is application/json."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard()

        resp = self.client.get(f"/hl-traders/{VALID_HL_ADDRESS}")

        self.assertIn('application/json', resp.content_type)

    def test_no_auth_required(self):
        """Endpoint returns non-401/403 without any auth header."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard()

        resp = self.client.get(f"/hl-traders/{VALID_HL_ADDRESS}")

        self.assertNotIn(resp.status_code, (401, 403))

    # ==================== Drawdown extraction ====================

    def test_drawdown_none_when_no_stats_or_ledger(self):
        """Drawdown is None when neither statistics nor ledger data exist."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard()

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertIsNone(data['drawdown'])

    def test_drawdown_from_statistics_only(self):
        """Drawdown populated from statistics drawdowns when no ledger."""
        stats = {
            'hotkey': SYNTHETIC_HOTKEY,
            'drawdowns': {
                'instantaneous_max_drawdown': 0.05,
                'daily_max_drawdown': 0.03,
            }
        }
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(statistics=stats)

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertIsNotNone(data['drawdown'])
        self.assertEqual(data['drawdown']['instantaneous_max_drawdown'], 0.05)
        self.assertEqual(data['drawdown']['daily_max_drawdown'], 0.03)
        # No ledger_max_drawdown key since no ledger
        self.assertNotIn('ledger_max_drawdown', data['drawdown'])

    def test_drawdown_from_ledger_only(self):
        """Drawdown includes ledger_max_drawdown from last checkpoint."""
        ledger = {
            'checkpoints': [
                {'performance': {'max_drawdown': 0.08}},
                {'performance': {'max_drawdown': 0.12}},
            ]
        }
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(ledger=ledger)

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertIsNotNone(data['drawdown'])
        # Should use the LAST checkpoint's max_drawdown
        self.assertEqual(data['drawdown']['ledger_max_drawdown'], 0.12)

    def test_drawdown_combined_stats_and_ledger(self):
        """Drawdown merges statistics drawdowns with ledger max_drawdown."""
        stats = {
            'hotkey': SYNTHETIC_HOTKEY,
            'drawdowns': {
                'instantaneous_max_drawdown': 0.05,
                'daily_max_drawdown': 0.03,
            }
        }
        ledger = {
            'checkpoints': [
                {'performance': {'max_drawdown': 0.07}},
            ]
        }
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(
            statistics=stats, ledger=ledger
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        dd = data['drawdown']
        self.assertEqual(dd['instantaneous_max_drawdown'], 0.05)
        self.assertEqual(dd['daily_max_drawdown'], 0.03)
        self.assertEqual(dd['ledger_max_drawdown'], 0.07)

    def test_drawdown_empty_checkpoints(self):
        """Drawdown has no ledger_max_drawdown when checkpoints list is empty."""
        ledger = {'checkpoints': []}
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(ledger=ledger)

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        # Empty drawdown dict → serialized as None
        self.assertIsNone(data['drawdown'])

    def test_drawdown_checkpoint_missing_performance(self):
        """Handles checkpoint with no performance key gracefully."""
        ledger = {'checkpoints': [{'something_else': 42}]}
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(ledger=ledger)

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        # ledger_max_drawdown should be None since no performance key
        self.assertIsNotNone(data['drawdown'])
        self.assertIsNone(data['drawdown']['ledger_max_drawdown'])

    # ==================== Challenge progress ====================

    def test_challenge_progress_for_subaccount_challenge(self):
        """
        Challenge progress is populated for SUBACCOUNT_CHALLENGE subaccounts.
        """
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(
            account_size=50_000,
            challenge_period={
                'bucket': 'SUBACCOUNT_CHALLENGE',
                'start_time_ms': 1_700_000_000_000,
            },
            account_size_data={
                'balance': 53_000,   # current_return = 1.06
                'max_return': 1.08,
            },
        )

        now_ms = 1_700_864_000_000  # +10 days
        with patch("vanta_api.validator_rest_server.TimeUtil.now_in_millis", return_value=now_ms):
            status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        challenge_progress = data['challenge_progress']
        self.assertIsNotNone(challenge_progress)
        self.assertTrue(challenge_progress['in_challenge_period'])
        self.assertEqual(challenge_progress['bucket'], 'SUBACCOUNT_CHALLENGE')
        self.assertEqual(challenge_progress['elapsed_time_ms'], now_ms - 1_700_000_000_000)
        self.assertAlmostEqual(challenge_progress['time_progress_percent'], (10 / ValiConfig.CHALLENGE_PERIOD_MAXIMUM_DAYS) * 100.0)
        self.assertAlmostEqual(challenge_progress['current_return'], 1.06)
        self.assertAlmostEqual(challenge_progress['returns_percent'], 6.0)
        self.assertAlmostEqual(
            challenge_progress['target_return_percent'],
            ValiConfig.SUBACCOUNT_CRYPTO_CHALLENGE_RETURNS_THRESHOLD * 100.0
        )
        self.assertAlmostEqual(challenge_progress['returns_progress_percent'], 60.0)
        self.assertAlmostEqual(challenge_progress['challenge_completion_percent'], 60.0)
        self.assertAlmostEqual(challenge_progress['drawdown_limit_percent'], 5.0)
        self.assertAlmostEqual(challenge_progress['drawdown_percent'], (1 - (1.06 / 1.08)) * 100.0)

    def test_challenge_progress_for_non_challenge_bucket(self):
        """
        Challenge completion percent is None when not in SUBACCOUNT_CHALLENGE.
        """
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard(
            challenge_period={
                'bucket': 'SUBACCOUNT_FUNDED',
                'start_time_ms': 1_700_000_000_000,
            },
            account_size_data={
                'balance': 52_000,
                'max_return': 1.08,
            },
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        challenge_progress = data['challenge_progress']
        self.assertFalse(challenge_progress['in_challenge_period'])
        self.assertEqual(challenge_progress['bucket'], 'SUBACCOUNT_FUNDED')
        self.assertIsNone(challenge_progress['returns_progress_percent'])
        self.assertIsNone(challenge_progress['challenge_completion_percent'])

    # ==================== 404 paths ====================

    def test_unknown_hl_address_returns_404(self):
        """Unknown HL address returns 404."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = None

        status, data = self._get(VALID_HL_ADDRESS_2)

        self.assertEqual(status, 404)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'HL address not found')

    def test_dashboard_none_returns_404(self):
        """404 when hotkey resolves but dashboard returns None."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = None

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 404)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Trader data not available')

    # ==================== 500 paths ====================

    def test_lookup_exception_returns_500(self):
        """500 when HL address lookup raises an exception."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.side_effect = RuntimeError("RPC down")

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 500)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Internal error')

    def test_dashboard_exception_returns_500(self):
        """500 when dashboard aggregation raises an exception."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.side_effect = RuntimeError("Timeout")

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

    # ==================== HL address passthrough ====================

    def test_hl_address_echoed_in_response(self):
        """The hl_address in the response matches the one in the URL."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard()

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(data['hl_address'], VALID_HL_ADDRESS)

    def test_correct_entity_client_calls(self):
        """Verifies the endpoint calls entity_client methods with the right args."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard_data.return_value = _build_dashboard()

        self._get(VALID_HL_ADDRESS)

        self.mock_entity.get_synthetic_hotkey_for_hl_address.assert_called_once_with(VALID_HL_ADDRESS)
        self.mock_entity.get_subaccount_dashboard_data.assert_called_once_with(SYNTHETIC_HOTKEY)

    def test_dashboard_not_called_when_hotkey_not_found(self):
        """Dashboard aggregation is skipped when HL address lookup returns None."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = None

        self._get(VALID_HL_ADDRESS)

        self.mock_entity.get_subaccount_dashboard_data.assert_not_called()


if __name__ == '__main__':
    unittest.main()
