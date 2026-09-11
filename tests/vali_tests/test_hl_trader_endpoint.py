# developer: rizzo
# Copyright (c) 2024 Taoshi Inc
"""
Unit tests for the GET /hl-traders/<hl_address> endpoint.

Tests the public (no-auth) endpoint that resolves a Hyperliquid address
to a synthetic hotkey and returns a dashboard aggregated from the entity
client and the challenge period / elimination / miner account / position /
limit order RPC clients.

Uses a lightweight Flask test client with mocked clients to isolate
endpoint logic from the full RPC stack.
"""
import json
import unittest
from unittest.mock import MagicMock

from flask import Flask


# ==================== Test constants ====================
VALID_HL_ADDRESS = "0x" + "a1b2c3d4" * 5
VALID_HL_ADDRESS_2 = "0x" + "1234567890abcdef" * 2 + "12345678"
SYNTHETIC_HOTKEY = "entity_alpha_0"
VALID_PAYOUT_ADDRESS = "0x" + "deadbeef" * 5


def _build_subaccount_info(
    account_size=50_000,
    payout_address=VALID_PAYOUT_ADDRESS,
    hl_address=VALID_HL_ADDRESS,
):
    """Build a subaccount_info dict matching entity_manager.get_subaccount_dashboard's shape."""
    return {
        'synthetic_hotkey': SYNTHETIC_HOTKEY,
        'subaccount_uuid': 'uuid-0',
        'subaccount_id': 0,
        'asset_class': 'crypto',
        'account_size': account_size,
        'status': 'active',
        'created_at_ms': 1700000000000,
        'eliminated_at_ms': None,
        'hl_address': hl_address,
        'payout_address': payout_address,
    }


class TestHlTraderEndpoint(unittest.TestCase):
    """
    Unit tests for the get_hl_trader endpoint method.

    Creates a minimal Flask app and binds the real get_hl_trader method
    with mocked RPC clients, avoiding the heavy ValidatorRestServer
    constructor.
    """

    def setUp(self):
        # Import the class but don't instantiate (too many deps)
        from vanta_api.validator_rest_server import ValidatorRestServer

        # Create a bare object without calling __init__
        self.server = object.__new__(ValidatorRestServer)

        # Wire up mocked clients used by get_hl_trader
        self.mock_entity = MagicMock()
        self.server._entity_client = self.mock_entity

        self.mock_challenge_period = MagicMock()
        self.mock_challenge_period.get_dashboard.return_value = None
        self.mock_challenge_period.get_drawdown_stats.return_value = None
        self.server._challenge_period_client = self.mock_challenge_period

        self.mock_elimination = MagicMock()
        self.mock_elimination.get_dashboard.return_value = None
        self.server._elimination_client = self.mock_elimination

        self.mock_miner_account = MagicMock()
        self.mock_miner_account.get_dashboard.return_value = None
        self.server._miner_account_client = self.mock_miner_account

        self.mock_position = MagicMock()
        self.mock_position.get_dashboard.return_value = None
        self.server._position_client = self.mock_position

        self.mock_limit_order = MagicMock()
        self.mock_limit_order.get_dashboard.return_value = None
        self.server._limit_order_client = self.mock_limit_order

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
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['status'], 'success')
        self.assertIn('timestamp', data)
        self.assertIsInstance(data['timestamp'], int)
        info = data['dashboard']['subaccount_info']
        self.assertEqual(info['synthetic_hotkey'], SYNTHETIC_HOTKEY)
        self.assertEqual(info['hl_address'], VALID_HL_ADDRESS)
        self.assertEqual(info['account_size'], 50_000)
        self.assertEqual(info['payout_address'], VALID_PAYOUT_ADDRESS)

    def test_success_with_positions(self):
        """Positions data is forwarded when present."""
        positions = {'n_positions': 5, 'total_leverage': 0.5}
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()
        self.mock_position.get_dashboard.return_value = positions

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['dashboard']['positions']['n_positions'], 5)
        self.assertEqual(data['dashboard']['positions']['total_leverage'], 0.5)

    def test_success_no_positions(self):
        """Positions key is omitted when the position client returns None."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()
        self.mock_position.get_dashboard.return_value = None

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertNotIn('positions', data['dashboard'])

    def test_success_no_payout_address(self):
        """Payout address is absent from subaccount_info when not set."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        info = _build_subaccount_info()
        del info['payout_address']
        self.mock_entity.get_subaccount_dashboard.return_value = info

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertNotIn('payout_address', data['dashboard']['subaccount_info'])

    def test_response_content_type_is_json(self):
        """Response Content-Type is application/json."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()

        resp = self.client.get(f"/hl-traders/{VALID_HL_ADDRESS}")

        self.assertIn('application/json', resp.content_type)

    def test_no_auth_required(self):
        """Endpoint returns non-401/403 without any auth header."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()

        resp = self.client.get(f"/hl-traders/{VALID_HL_ADDRESS}")

        self.assertNotIn(resp.status_code, (401, 403))

    # ==================== Drawdown ====================

    def test_drawdown_none_when_client_returns_none(self):
        """Drawdown key is omitted when the challenge period client has no stats."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()
        self.mock_challenge_period.get_drawdown_stats.return_value = None

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertNotIn('drawdown', data['dashboard'])

    def test_drawdown_forwarded_from_challenge_period_client(self):
        """Drawdown section is forwarded verbatim from the challenge period client."""
        drawdown_stats = {
            'instantaneous_max_drawdown': 0.05,
            'daily_max_drawdown': 0.03,
        }
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()
        self.mock_challenge_period.get_drawdown_stats.return_value = drawdown_stats

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['dashboard']['drawdown'], drawdown_stats)

    def test_drawdown_section_error_is_swallowed(self):
        """An exception retrieving drawdown stats doesn't fail the whole request."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()
        self.mock_challenge_period.get_drawdown_stats.side_effect = RuntimeError("boom")

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertNotIn('drawdown', data['dashboard'])

    # ==================== Challenge period ====================

    def test_challenge_period_forwarded(self):
        """Challenge period section is forwarded verbatim from the challenge period client."""
        challenge_period = {
            'bucket': 'SUBACCOUNT_CHALLENGE',
            'start_time_ms': 1_700_000_000_000,
        }
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()
        self.mock_challenge_period.get_dashboard.return_value = challenge_period

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertEqual(data['dashboard']['challenge_period'], challenge_period)

    def test_challenge_period_none_when_client_returns_none(self):
        """Challenge period key is omitted when the client has no data."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()
        self.mock_challenge_period.get_dashboard.return_value = None

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(status, 200)
        self.assertNotIn('challenge_period', data['dashboard'])

    # ==================== 404 paths ====================

    def test_unknown_hl_address_returns_404(self):
        """Unknown HL address returns 404."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = None

        status, data = self._get(VALID_HL_ADDRESS_2)

        self.assertEqual(status, 404)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'HL address not found')

    def test_subaccount_info_none_returns_404(self):
        """404 when hotkey resolves but subaccount info lookup returns None."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = None

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

    def test_subaccount_info_exception_returns_500(self):
        """500 when subaccount info lookup raises an exception."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.side_effect = RuntimeError("Timeout")

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

    def test_hl_address_echoed_in_subaccount_info(self):
        """The hl_address in subaccount_info matches the one in the URL."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info(
            hl_address=VALID_HL_ADDRESS
        )

        status, data = self._get(VALID_HL_ADDRESS)

        self.assertEqual(data['dashboard']['subaccount_info']['hl_address'], VALID_HL_ADDRESS)

    def test_correct_entity_client_calls(self):
        """Verifies the endpoint calls entity_client methods with the right args."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()

        self._get(VALID_HL_ADDRESS)

        self.mock_entity.get_synthetic_hotkey_for_hl_address.assert_called_once_with(VALID_HL_ADDRESS)
        self.mock_entity.get_subaccount_dashboard.assert_called_once_with(SYNTHETIC_HOTKEY)

    def test_dashboard_not_called_when_hotkey_not_found(self):
        """Subaccount info lookup is skipped when HL address lookup returns None."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = None

        self._get(VALID_HL_ADDRESS)

        self.mock_entity.get_subaccount_dashboard.assert_not_called()

    def test_positions_time_ms_query_param_forwarded(self):
        """positions_time_ms query param is parsed and forwarded to the position client."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()

        self.client.get(f"/hl-traders/{VALID_HL_ADDRESS}?positions_time_ms=123")

        self.mock_position.get_dashboard.assert_called_once_with(SYNTHETIC_HOTKEY, 123)

    def test_limit_orders_time_ms_query_param_forwarded(self):
        """limit_orders_time_ms query param is parsed and forwarded to the limit order client."""
        self.mock_entity.get_synthetic_hotkey_for_hl_address.return_value = SYNTHETIC_HOTKEY
        self.mock_entity.get_subaccount_dashboard.return_value = _build_subaccount_info()

        self.client.get(f"/hl-traders/{VALID_HL_ADDRESS}?limit_orders_time_ms=456")

        self.mock_limit_order.get_dashboard.assert_called_once_with(SYNTHETIC_HOTKEY, 456)


if __name__ == '__main__':
    unittest.main()
