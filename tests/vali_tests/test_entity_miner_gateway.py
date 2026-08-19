# developer: jbonilla
# Copyright (c) 2025 Taoshi Inc
"""
Tests for Entity Miner Gateway implementation.

Covers:
- OrderEvent / OrderEventStore (ring buffer)
- WebSocket Server entity auth (signature, nonce, scope enforcement)
- HyperliquidTracker rejection broadcasts
- EntityMinerRestServer WS message handling
- Dynamic subaccount subscription
- _remove_client entity cleanup
"""
import asyncio
import json
import time
import unittest
from collections import defaultdict
from unittest.mock import MagicMock

from tests.vali_tests.base_objects.test_base import TestBase


# ==================== OrderEvent / OrderEventStore Tests ====================

class TestOrderEvent(TestBase):
    """Tests for the OrderEvent dataclass."""

    def test_order_event_creation(self):
        """OrderEvent stores all fields correctly."""
        from vanta_api.entity_miner_rest_server import OrderEvent

        event = OrderEvent(
            timestamp_ms=1700000000000,
            hl_address="0x" + "a1" * 20,
            trade_pair="BTCUSD",
            order_type="LONG",
            status="rejected",
            error_message="Rate limited. Please wait 5s.",
            fill_hash="0xabc123",
            synthetic_hotkey="entity_hotkey_0"
        )

        self.assertEqual(event.timestamp_ms, 1700000000000)
        self.assertEqual(event.hl_address, "0x" + "a1" * 20)
        self.assertEqual(event.status, "rejected")
        self.assertEqual(event.error_message, "Rate limited. Please wait 5s.")

    def test_order_event_to_dict(self):
        """OrderEvent.to_dict() returns all fields."""
        from vanta_api.entity_miner_rest_server import OrderEvent

        event = OrderEvent(
            timestamp_ms=1700000000000,
            hl_address="0xabc",
            trade_pair="ETHUSD",
            order_type="SHORT",
            status="accepted"
        )
        d = event.to_dict()

        self.assertIsInstance(d, dict)
        self.assertEqual(d["timestamp_ms"], 1700000000000)
        self.assertEqual(d["trade_pair"], "ETHUSD")
        self.assertEqual(d["status"], "accepted")
        self.assertEqual(d["error_message"], "")  # default

    def test_order_event_defaults(self):
        """OrderEvent default fields are empty strings."""
        from vanta_api.entity_miner_rest_server import OrderEvent

        event = OrderEvent(
            timestamp_ms=0, hl_address="0x1", trade_pair="", order_type="", status="rejected"
        )
        self.assertEqual(event.error_message, "")
        self.assertEqual(event.fill_hash, "")
        self.assertEqual(event.synthetic_hotkey, "")


class TestOrderEventStore(TestBase):
    """Tests for the OrderEventStore ring buffer."""

    def test_add_and_get_events(self):
        """Events can be stored and retrieved per HL address."""
        from vanta_api.entity_miner_rest_server import OrderEvent, OrderEventStore

        store = OrderEventStore()
        addr = "0x" + "ab" * 20

        event = OrderEvent(
            timestamp_ms=1000, hl_address=addr, trade_pair="BTCUSD",
            order_type="LONG", status="rejected", error_message="Rate limited"
        )
        store.add(event)

        events = store.get_events(addr)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["error_message"], "Rate limited")

    def test_get_events_empty(self):
        """Getting events for unknown address returns empty list."""
        from vanta_api.entity_miner_rest_server import OrderEventStore

        store = OrderEventStore()
        events = store.get_events("0xunknown")
        self.assertEqual(events, [])

    def test_get_events_since_filter(self):
        """Events can be filtered by timestamp."""
        from vanta_api.entity_miner_rest_server import OrderEvent, OrderEventStore

        store = OrderEventStore()
        addr = "0x" + "cd" * 20

        for ts in [1000, 2000, 3000, 4000, 5000]:
            store.add(OrderEvent(
                timestamp_ms=ts, hl_address=addr, trade_pair="BTCUSD",
                order_type="LONG", status="rejected"
            ))

        events = store.get_events(addr, since_ms=3000)
        self.assertEqual(len(events), 2)  # 4000, 5000
        self.assertEqual(events[0]["timestamp_ms"], 4000)
        self.assertEqual(events[1]["timestamp_ms"], 5000)

    def test_ring_buffer_eviction(self):
        """Events beyond MAX_EVENTS_PER_ADDRESS are evicted (oldest first)."""
        from vanta_api.entity_miner_rest_server import OrderEvent, OrderEventStore

        store = OrderEventStore()
        addr = "0x" + "ef" * 20

        # Add more than max
        for i in range(OrderEventStore.MAX_EVENTS_PER_ADDRESS + 20):
            store.add(OrderEvent(
                timestamp_ms=i, hl_address=addr, trade_pair="BTCUSD",
                order_type="LONG", status="rejected"
            ))

        events = store.get_events(addr)
        self.assertEqual(len(events), OrderEventStore.MAX_EVENTS_PER_ADDRESS)
        # Oldest should be evicted — first event starts at 20
        self.assertEqual(events[0]["timestamp_ms"], 20)

    def test_separate_addresses(self):
        """Events for different addresses are stored separately."""
        from vanta_api.entity_miner_rest_server import OrderEvent, OrderEventStore

        store = OrderEventStore()
        addr1 = "0x" + "11" * 20
        addr2 = "0x" + "22" * 20

        store.add(OrderEvent(timestamp_ms=1, hl_address=addr1, trade_pair="BTCUSD",
                             order_type="LONG", status="rejected"))
        store.add(OrderEvent(timestamp_ms=2, hl_address=addr2, trade_pair="ETHUSD",
                             order_type="SHORT", status="accepted"))

        self.assertEqual(len(store.get_events(addr1)), 1)
        self.assertEqual(len(store.get_events(addr2)), 1)
        self.assertEqual(store.get_events(addr1)[0]["trade_pair"], "BTCUSD")
        self.assertEqual(store.get_events(addr2)[0]["trade_pair"], "ETHUSD")


# ==================== Scope Enforcement Tests ====================

class TestEntityScopeEnforcement(TestBase):
    """Tests for entity client scope enforcement logic (subscribe, subscribe_subaccount)."""

    def test_parse_synthetic_hotkey_ownership(self):
        """parse_synthetic_hotkey correctly extracts entity_hotkey."""
        from entity_management.entity_utils import parse_synthetic_hotkey

        entity, sub_id = parse_synthetic_hotkey("5EntityKey_0")
        self.assertEqual(entity, "5EntityKey")
        self.assertEqual(sub_id, 0)

        entity, sub_id = parse_synthetic_hotkey("5EntityKey_42")
        self.assertEqual(entity, "5EntityKey")
        self.assertEqual(sub_id, 42)

    def test_ownership_check_different_entity(self):
        """parse_synthetic_hotkey distinguishes different entities."""
        from entity_management.entity_utils import parse_synthetic_hotkey

        # Entity "5Alice" trying to access "5Bob_0"
        parsed_entity, _ = parse_synthetic_hotkey("5Bob_0")
        self.assertNotEqual(parsed_entity, "5Alice")

    def test_ownership_check_same_entity(self):
        """parse_synthetic_hotkey confirms same entity."""
        from entity_management.entity_utils import parse_synthetic_hotkey

        parsed_entity, _ = parse_synthetic_hotkey("5Alice_3")
        self.assertEqual(parsed_entity, "5Alice")


# ==================== Remove Client Tests ====================

class TestRemoveClient(TestBase):
    """Tests for _remove_client on the WebSocket server (api-key/tier client model)."""

    def _make_server(self):
        from vanta_api.websocket_server import WebSocketServer
        server = object.__new__(WebSocketServer)
        server._clients = {}
        server._api_key_client_ids = defaultdict(list)
        server.api_key_to_alias = {}
        server._event_loop = None
        return server

    def test_remove_api_key_client(self):
        """Removing a client drops it from _clients and its api-key client list."""
        server = self._make_server()
        client = MagicMock()
        client.api_key = "key123"
        client.websocket.state = "CLOSED"  # != State.OPEN, so no async close is scheduled
        server._clients["c1"] = client
        server._api_key_client_ids["key123"] = ["c1"]

        server._remove_client("c1")

        self.assertNotIn("c1", server._clients)
        self.assertNotIn("c1", server._api_key_client_ids["key123"])

    def test_remove_client_preserves_others(self):
        """Removing one client leaves the api-key's other clients registered."""
        server = self._make_server()
        for cid in ["c1", "c2"]:
            client = MagicMock()
            client.api_key = "key123"
            client.websocket.state = "CLOSED"
            server._clients[cid] = client
        server._api_key_client_ids["key123"] = ["c1", "c2"]

        server._remove_client("c1")

        self.assertNotIn("c1", server._clients)
        self.assertIn("c2", server._clients)
        self.assertEqual(server._api_key_client_ids["key123"], ["c2"])

    def test_remove_unknown_client_is_noop(self):
        """Removing an unknown client id does not raise."""
        server = self._make_server()
        server._remove_client("does-not-exist")
        self.assertEqual(len(server._clients), 0)


# ==================== Notify New Subaccount RPC Tests ====================

class TestNotifyNewSubaccount(TestBase):
    """Tests for notify_new_subaccount_rpc on the WebSocket server."""

    def _make_server(self):
        from vanta_api.websocket_server import WebSocketServer
        server = object.__new__(WebSocketServer)
        server.api_key_to_alias = {}
        server._api_key_client_ids = defaultdict(list)
        server._clients = {}
        server._event_loop = None  # skips the async subscription-status send
        server.broadcast_subaccount_dashboard_rpc = MagicMock()
        return server

    def test_notify_subscribes_connected_clients(self):
        """Connected entity clients are dashboard-subscribed to the new subaccount."""
        server = self._make_server()
        entity = "5Entity"
        api_key = "key-1"
        server.api_key_to_alias[api_key] = entity
        server._api_key_client_ids[api_key] = ["c1", "c2"]
        for cid in ["c1", "c2"]:
            client = MagicMock()
            client.dashboard_subscriptions = {}
            server._clients[cid] = client

        result = server.notify_new_subaccount_rpc(entity, f"{entity}_5")

        self.assertTrue(result)
        self.assertIn(f"{entity}_5", server._clients["c1"].dashboard_subscriptions)
        self.assertIn(f"{entity}_5", server._clients["c2"].dashboard_subscriptions)
        server.broadcast_subaccount_dashboard_rpc.assert_called_once_with(f"{entity}_5")

    def test_notify_no_connected_clients(self):
        """Returns True (no-op) and does not broadcast when the entity has no clients."""
        server = self._make_server()
        server.api_key_to_alias["key-off"] = "5Offline"
        server._api_key_client_ids["key-off"] = []

        result = server.notify_new_subaccount_rpc("5Offline", "5Offline_0")

        self.assertTrue(result)
        server.broadcast_subaccount_dashboard_rpc.assert_not_called()

    def test_notify_unknown_entity(self):
        """Returns True (no-op) when the entity has no api-key mapping yet."""
        server = self._make_server()
        result = server.notify_new_subaccount_rpc("5Unmapped", "5Unmapped_0")
        self.assertTrue(result)
        server.broadcast_subaccount_dashboard_rpc.assert_not_called()


# ==================== WebSocketNotifierClient Tests ====================

class TestWebSocketNotifierClientNewMethod(TestBase):
    """Tests for the notify_new_subaccount method on WebSocketNotifierClient."""

    def test_notify_new_subaccount_delegates_to_server(self):
        """notify_new_subaccount calls the RPC method on the server."""
        from vanta_api.websocket_notifier import WebSocketNotifierClient
        from vali_objects.vali_config import RPCConnectionMode

        client = WebSocketNotifierClient(connection_mode=RPCConnectionMode.LOCAL)
        mock_server = MagicMock()
        mock_server.notify_new_subaccount_rpc.return_value = True
        client.set_direct_server(mock_server)

        result = client.notify_new_subaccount("5Entity", "5Entity_0")

        self.assertTrue(result)
        mock_server.notify_new_subaccount_rpc.assert_called_once_with("5Entity", "5Entity_0")

    def test_notify_new_subaccount_handles_exception(self):
        """notify_new_subaccount returns False on exception."""
        from vanta_api.websocket_notifier import WebSocketNotifierClient
        from vali_objects.vali_config import RPCConnectionMode

        client = WebSocketNotifierClient(connection_mode=RPCConnectionMode.LOCAL)
        mock_server = MagicMock()
        mock_server.notify_new_subaccount_rpc.side_effect = Exception("RPC down")
        client.set_direct_server(mock_server)

        result = client.notify_new_subaccount("5Entity", "5Entity_0")

        self.assertFalse(result)


# ==================== HyperliquidTracker Rejection Broadcast Tests ====================

class TestHLTrackerRejectionBroadcasts(TestBase):
    """Tests for rejection/acceptance broadcast calls in HyperliquidTracker."""

    def _make_tracker(self, ws_notifier_client="__default__"):
        """Create a HyperliquidTracker with all dependencies mocked."""
        from entity_management.hyperliquid_tracker import HyperliquidTracker
        from vali_objects.vali_config import RPCConnectionMode

        notifier = MagicMock() if ws_notifier_client == "__default__" else ws_notifier_client
        return HyperliquidTracker(
            entity_client=MagicMock(),
            price_fetcher_client=MagicMock(),
            order_processor=MagicMock(),
            ws_notifier_client=notifier,
            connection_mode=RPCConnectionMode.LOCAL,
        )

    def _make_fill(self, coin="BTC", side="B", sz="1.0", px="50000"):
        return {"coin": coin, "side": side, "sz": sz, "px": px}

    def _setup_order_path(self, tracker, synthetic="5Entity_0"):
        """Mock every dependency so a fill reaches _order_processor.validate / _dispatch_order."""
        tracker._entity_client.get_synthetic_hotkey_for_hl_address.return_value = synthetic
        tracker._entity_client.get_subaccount_info_for_synthetic.return_value = {"account_size": 100_000}
        tracker._rate_limiter = MagicMock()
        tracker._rate_limiter.is_allowed.return_value = (True, 0)
        tracker._fetch_hl_account_state = MagicMock(return_value={
            "total_portfolio_value": 100_000,
            "positions": {"BTC": {"weight": 0.3}},
        })
        tracker._position_client = MagicMock()
        tracker._position_client.get_open_position_for_trade_pair.return_value = None
        tracker._miner_account_client = MagicMock()
        tracker._miner_account_client.get_balance.return_value = 100_000
        price = MagicMock()
        price.close = 50_000.0
        tracker._price_fetcher_client.get_sorted_price_sources_for_trade_pair.return_value = [price]
        tracker._compute_fill_price = MagicMock(return_value=50_000.0)

    def test_broadcast_rejection_calls_notifier(self):
        """_broadcast_rejection broadcasts the subaccount dashboard."""
        tracker = self._make_tracker()
        tracker._broadcast_rejection("5Entity_0", "Test error message")
        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once_with("5Entity_0")

    def test_broadcast_rejection_no_notifier(self):
        """_broadcast_rejection is a no-op when ws_notifier_client is None."""
        tracker = self._make_tracker(ws_notifier_client=None)
        tracker._broadcast_rejection("5Entity_0", "No crash")  # must not raise

    def test_broadcast_accepted_fill_calls_notifier(self):
        """Accepted fills broadcast the subaccount dashboard."""
        tracker = self._make_tracker()
        tracker._broadcast_accepted_fill(
            synthetic_hotkey="5Entity_0",
            trade_pair="BTCUSD",
            order_type="LONG",
            fill_hash="0xabc123",
        )
        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once_with("5Entity_0")

    def test_rate_limit_rejection_broadcasts(self):
        """A rate-limited fill triggers a rejection broadcast."""
        tracker = self._make_tracker()
        tracker._entity_client.get_synthetic_hotkey_for_hl_address.return_value = "5Entity_0"
        tracker._rate_limiter = MagicMock()
        tracker._rate_limiter.is_allowed.return_value = (False, 5.0)

        tracker._process_fill("0xaddr", self._make_fill())

        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once()

    def test_validate_rejection_broadcasts(self):
        """A fill rejected by the order processor's pre-trade validation broadcasts."""
        tracker = self._make_tracker()
        self._setup_order_path(tracker)
        tracker._order_processor.validate.return_value = (False, "Miner eliminated", None)

        tracker._process_fill("0xaddr", self._make_fill())

        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once()

    def test_signal_exception_rejection_broadcasts(self):
        """A SignalException raised while dispatching the order broadcasts a rejection."""
        from vali_objects.exceptions.signal_exception import SignalException

        tracker = self._make_tracker()
        self._setup_order_path(tracker)
        tracker._order_processor.validate.return_value = (True, "", None)
        tracker._order_processor.process_hyperliquid_order.side_effect = SignalException("Leverage too high")

        tracker._process_fill("0xaddr", self._make_fill())

        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once()

    def test_unexpected_exception_rejection_broadcasts(self):
        """An unexpected exception while dispatching the order also broadcasts a rejection."""
        tracker = self._make_tracker()
        self._setup_order_path(tracker)
        tracker._order_processor.validate.return_value = (True, "", None)
        tracker._order_processor.process_hyperliquid_order.side_effect = ValueError("boom")

        tracker._process_fill("0xaddr", self._make_fill())

        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once()


# ==================== EntityMinerRestServer WS Message Handling Tests ====================

class TestEntityMinerRestServerMessageHandling(TestBase):
    """Tests for EntityMinerRestServer._handle_ws_message."""

    def _make_gateway(self):
        """Create a minimal EntityMinerRestServer for testing message handling."""
        from vanta_api.entity_miner_rest_server import EntityMinerRestServer, OrderEventStore

        gw = object.__new__(EntityMinerRestServer)
        gw._event_store = OrderEventStore()
        gw._dashboard_cache = {}
        gw._hl_to_synthetic = {"0xHL1": "5Entity_0"}
        gw._synthetic_to_hl = {"5Entity_0": "0xHL1"}
        gw._dashboard_cache_updated_ms = {}
        gw._mapping_last_refresh_ms = {}
        gw._sse_subscribers = {}
        gw._sse_lock = __import__('threading').Lock()
        gw._validator_url = None
        gw._hotkey = None
        return gw

    def test_handle_dashboard_message(self):
        """Dashboard messages update cache."""
        gw = self._make_gateway()
        gw._push_sse = MagicMock()

        gw._handle_ws_message({
            "type": "subaccount_dashboard",
            "synthetic_hotkey": "5Entity_0",
            "timestamp": 1700000000000,
            "data": {"pnl": 500.0, "positions": []}
        })

        self.assertIn("0xHL1", gw._dashboard_cache)
        cached = gw._dashboard_cache["0xHL1"]
        self.assertEqual(cached["pnl"], 500.0)
        self.assertEqual(cached["hl_address"], "0xHL1")
        self.assertEqual(cached["synthetic_hotkey"], "5Entity_0")
        gw._push_sse.assert_called_once()

    def test_handle_error_message(self):
        """Error messages create OrderEvents and update SSE."""
        gw = self._make_gateway()
        gw._push_sse = MagicMock()

        gw._handle_ws_message({
            "type": "error",
            "synthetic_hotkey": "5Entity_0",
            "timestamp": 1700000000000,
            "data": {"error_msg": "Market is closed for BTCUSD."}
        })

        events = gw._event_store.get_events("0xHL1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "rejected")
        self.assertEqual(events[0]["error_message"], "Market is closed for BTCUSD.")
        gw._push_sse.assert_called_once()

    def test_handle_accepted_order_event_message(self):
        """Accepted order_event payloads create OrderEvents and SSE updates."""
        gw = self._make_gateway()
        gw._push_sse = MagicMock()

        gw._handle_ws_message({
            "type": "subaccount_dashboard",
            "synthetic_hotkey": "5Entity_0",
            "timestamp": 1700000000000,
            "data": {
                "order_event": {
                    "status": "accepted",
                    "trade_pair": "BTCUSD",
                    "order_type": "LONG",
                    "fill_hash": "0xfill",
                }
            }
        })

        events = gw._event_store.get_events("0xHL1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "accepted")
        self.assertEqual(events[0]["trade_pair"], "BTCUSD")
        self.assertEqual(events[0]["order_type"], "LONG")
        self.assertEqual(events[0]["fill_hash"], "0xfill")
        self.assertNotIn("0xHL1", gw._dashboard_cache)
        gw._push_sse.assert_called_once()

    def test_handle_message_unknown_synthetic(self):
        """Messages for unknown synthetic hotkeys are silently dropped."""
        gw = self._make_gateway()
        gw._push_sse = MagicMock()

        gw._handle_ws_message({
            "type": "subaccount_dashboard",
            "synthetic_hotkey": "5Unknown_99",
            "timestamp": 1700000000000,
            "data": {}
        })

        self.assertEqual(len(gw._dashboard_cache), 0)
        gw._push_sse.assert_not_called()

    def test_handle_message_no_synthetic(self):
        """Messages without synthetic_hotkey are silently dropped."""
        gw = self._make_gateway()
        gw._push_sse = MagicMock()

        gw._handle_ws_message({"type": "pong"})
        gw._push_sse.assert_not_called()

    def test_handle_new_subaccount_reloads_mappings(self):
        """new_subaccount_subscribed action triggers mapping reload."""
        gw = self._make_gateway()
        gw._load_hl_mappings = MagicMock()
        gw._push_sse = MagicMock()

        gw._handle_ws_message({
            "type": "subscription_status",
            "action": "new_subaccount_subscribed",
            "synthetic_hotkey": "5Entity_1",
            "entity_hotkey": "5Entity"
        })

        gw._load_hl_mappings.assert_called_once()


class TestEntityMinerDashboardCacheReconciliation(TestBase):
    """Regression tests for HL mapping reassignment and stale dashboard cache behavior."""

    def _make_gateway(self):
        from vanta_api.entity_miner_rest_server import EntityMinerRestServer, OrderEventStore
        import threading
        try:
            from flask import Flask
        except ModuleNotFoundError:
            self.skipTest("flask not installed")

        gw = object.__new__(EntityMinerRestServer)
        gw.app = Flask(__name__)
        gw._event_store = OrderEventStore()
        gw._dashboard_cache = {}
        gw._dashboard_cache_updated_ms = {}
        gw._hl_to_synthetic = {}
        gw._synthetic_to_hl = {}
        gw._mapping_last_refresh_ms = {}
        gw._sse_subscribers = {}
        gw._sse_lock = threading.Lock()
        gw._validator_url = "http://validator"
        gw.DASHBOARD_CACHE_TTL_MS = 10_000
        gw.MAPPING_REFRESH_TTL_MS = 5_000
        return gw

    def test_set_hl_mapping_reassignment_evicts_dashboard(self):
        gw = self._make_gateway()
        hl = "0xabc"

        gw._dashboard_cache[hl] = {"synthetic_hotkey": "entity_409", "timestamp_ms": 1}
        gw._dashboard_cache_updated_ms[hl] = 1
        gw._hl_to_synthetic[hl] = "entity_409"
        gw._synthetic_to_hl["entity_409"] = hl
        gw._save_hl_mappings = MagicMock()

        gw._set_hl_mapping(hl, "entity_443", source="test")

        self.assertEqual(gw._hl_to_synthetic[hl], "entity_443")
        self.assertEqual(gw._synthetic_to_hl["entity_443"], hl)
        self.assertNotIn("entity_409", gw._synthetic_to_hl)
        self.assertNotIn(hl, gw._dashboard_cache)

    def test_dashboard_endpoint_refreshes_when_mapping_changes(self):
        gw = self._make_gateway()
        hl = "0x2d26b7339a624e84634cde1d1fb5128eb02e4b0e"

        # Stale cache points to old synthetic.
        gw._dashboard_cache[hl] = {
            "timestamp_ms": 1000,
            "synthetic_hotkey": "entity_409",
            "hl_address": hl,
            "balance": 100000.0,
            "total_realized_pnl": 0.0,
        }
        gw._dashboard_cache_updated_ms[hl] = int(time.time() * 1000)
        gw._hl_to_synthetic[hl] = "entity_443"

        # Validator returns canonical snapshot for new synthetic.
        gw._fetch_validator_hl_trader = MagicMock(return_value={
            "status": "success",
            "timestamp": 2000,
            "dashboard": {
                "subaccount_info": {
                    "synthetic_hotkey": "entity_443",
                    "balance": 99844.9234,
                    "total_realized_pnl": -71.8062,
                }
            },
        })

        with gw.app.test_request_context(f"/api/hl/{hl}/dashboard"):
            response, status_code = gw.dashboard_endpoint(hl)

        payload = response.get_json()
        self.assertEqual(status_code, 200)
        self.assertEqual(payload["synthetic_hotkey"], "entity_443")
        self.assertAlmostEqual(payload["balance"], 99844.9234)
        self.assertAlmostEqual(payload["total_realized_pnl"], -71.8062)

    def test_events_endpoint_filters_out_old_synthetic_events(self):
        from vanta_api.entity_miner_rest_server import OrderEvent

        gw = self._make_gateway()
        hl = "0x2d26b7339a624e84634cde1d1fb5128eb02e4b0e"
        gw._hl_to_synthetic[hl] = "entity_443"
        gw._mapping_last_refresh_ms[hl] = int(time.time() * 1000)

        gw._event_store.add(OrderEvent(
            timestamp_ms=1000,
            hl_address=hl,
            trade_pair="BTCUSD",
            order_type="LONG",
            status="accepted",
            synthetic_hotkey="entity_409",
        ))
        gw._event_store.add(OrderEvent(
            timestamp_ms=2000,
            hl_address=hl,
            trade_pair="BTCUSD",
            order_type="LONG",
            status="accepted",
            synthetic_hotkey="entity_443",
        ))

        with gw.app.test_request_context(f"/api/hl/{hl}/events"):
            response, status_code = gw.events_endpoint(hl)

        payload = response.get_json()
        self.assertEqual(status_code, 200)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["events"][0]["synthetic_hotkey"], "entity_443")


# ==================== SSE Tests ====================

class TestSSESubscription(TestBase):
    """Tests for SSE subscribe/unsubscribe/push."""

    def _make_gateway(self):
        from vanta_api.entity_miner_rest_server import EntityMinerRestServer
        import threading

        gw = object.__new__(EntityMinerRestServer)
        gw._sse_subscribers = {}
        gw._sse_lock = threading.Lock()
        return gw

    def test_subscribe_creates_queue(self):
        """Subscribing returns a queue and registers it."""
        gw = self._make_gateway()
        q = gw._subscribe_sse("0xHL1")

        self.assertIsNotNone(q)
        self.assertIn(q, gw._sse_subscribers["0xHL1"])

    def test_unsubscribe_removes_queue(self):
        """Unsubscribing removes the queue from tracking."""
        gw = self._make_gateway()
        q = gw._subscribe_sse("0xHL1")
        gw._unsubscribe_sse("0xHL1", q)

        self.assertNotIn("0xHL1", gw._sse_subscribers)

    def test_push_delivers_to_subscribers(self):
        """Push delivers data to all subscribers for the address."""
        gw = self._make_gateway()
        q1 = gw._subscribe_sse("0xHL1")
        q2 = gw._subscribe_sse("0xHL1")

        gw._push_sse("0xHL1", {"type": "test", "data": "hello"})

        self.assertEqual(q1.get_nowait()["data"], "hello")
        self.assertEqual(q2.get_nowait()["data"], "hello")

    def test_push_different_address_no_crosstalk(self):
        """Push to one address doesn't affect other addresses."""
        gw = self._make_gateway()
        q1 = gw._subscribe_sse("0xHL1")
        q2 = gw._subscribe_sse("0xHL2")

        gw._push_sse("0xHL1", {"data": "for_hl1"})

        self.assertFalse(q2.empty() is False)  # q2 should be empty
        self.assertEqual(q1.get_nowait()["data"], "for_hl1")

    def test_push_full_queue_drops(self):
        """Full queues don't block push — events are silently dropped."""

        gw = self._make_gateway()
        q = gw._subscribe_sse("0xHL1")

        # Fill the queue (maxsize=50)
        for i in range(50):
            q.put({"i": i})

        # This should not block or raise
        gw._push_sse("0xHL1", {"data": "overflow"})

        # Queue is still at 50 (the overflow was dropped)
        self.assertEqual(q.qsize(), 50)


# ==================== Entity Manager Notification Tests ====================

class TestEntityManagerNotification(TestBase):
    """Tests that EntityManager calls notify_new_subaccount on subaccount creation."""

    def test_create_subaccount_admin_notifies(self):
        """Admin subaccount creation calls notify_new_subaccount."""
        from entity_management.entity_manager import EntityManager
        from vali_objects.vali_config import RPCConnectionMode

        # Create a minimal manager in test mode
        manager = EntityManager(
            running_unit_tests=True,
            connection_mode=RPCConnectionMode.LOCAL
        )
        manager._websocket_client = MagicMock()
        manager._websocket_client.notify_new_subaccount = MagicMock(return_value=True)
        manager._entity_collateral_client = MagicMock()
        # Admin create still reads cached collateral for the result message (slashing is skipped).
        manager._entity_collateral_client.get_cached_collateral.return_value = 1_000_000.0
        manager._entity_collateral_client.compute_entity_required_collateral.return_value = 0.0

        # Mock RPC clients that are called during register/create
        manager._position_client = MagicMock()
        manager._position_client.get_positions_for_one_hotkey.return_value = []
        manager._challenge_period_client = MagicMock()
        manager._asset_selection_client = MagicMock()
        manager._asset_selection_client.process_asset_selection_request.return_value = {
            'successfully_processed': True
        }
        manager._miner_account_client = MagicMock()
        manager._miner_account_client.set_miner_account_size.return_value = True

        entity_hotkey = "5TestEntity"

        # Register entity
        manager.register_entity(entity_hotkey=entity_hotkey)

        # The WS-notification path runs only outside unit-test mode; enable it for
        # this create call while stubbing the disk write and dashboard broadcasts so
        # the create stays fully in-memory.
        manager._write_entities_from_memory_to_disk = MagicMock()
        manager.broadcast_subaccount_registration = MagicMock()
        manager.broadcast_subaccount_dashboard = MagicMock()
        manager.running_unit_tests = False

        # Create admin subaccount (skips slashing, synchronous path)
        success, info, msg = manager.create_subaccount(
            entity_hotkey=entity_hotkey,
            account_size=10_000,
            asset_class="crypto",
            admin=True
        )

        self.assertTrue(success)
        synthetic = info.synthetic_hotkey
        manager._websocket_client.notify_new_subaccount.assert_called_once_with(
            entity_hotkey, synthetic
        )


if __name__ == "__main__":
    unittest.main()
