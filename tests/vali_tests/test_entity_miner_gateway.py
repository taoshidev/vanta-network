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
from collections import defaultdict, deque
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

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


# ==================== WebSocket Server Entity Auth Tests ====================

class TestWebSocketEntityAuth(unittest.IsolatedAsyncioTestCase):
    """Tests for the WebSocket server entity authentication flow."""

    def _make_server(self):
        """Create a minimal WebSocketServer-like object for testing auth methods."""
        from vanta_api.websocket_server import (
            WebSocketServer, MAX_N_WS_PER_ENTITY,
            ENTITY_AUTH_TIMESTAMP_TTL_MS, ENTITY_AUTH_MAX_NONCES
        )

        server = object.__new__(WebSocketServer)
        # Initialize only the state we need
        server.connected_clients = {}
        server.client_auth = {}
        server.api_key_clients = defaultdict(deque)
        server.entity_clients = defaultdict(deque)
        server._entity_auth_nonces = defaultdict(dict)
        server._entity_nonce_lock = asyncio.Lock()
        server.subscribed_clients = set()
        server.subaccount_subscriptions = defaultdict(set)
        server.subaccount_last_broadcast_ms = {}
        server._subaccount_poll_tasks = {}
        server.sequence_number = 0
        server.loop = asyncio.get_event_loop()
        server._entity_client = MagicMock()
        server.api_key_to_alias = {}
        return server

    def _mock_websocket(self):
        """Create a mock WebSocket with async send/close."""
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        return ws

    # --- Nonce Cleanup ---

    def test_cleanup_expired_nonces(self):
        """Expired nonces are removed during cleanup."""
        from vanta_api.websocket_server import ENTITY_AUTH_TIMESTAMP_TTL_MS

        server = self._make_server()
        entity = "5EntityHotkey"
        now = 10_000_000

        # Add nonces: some expired, some valid
        server._entity_auth_nonces[entity] = {
            "old1": now - ENTITY_AUTH_TIMESTAMP_TTL_MS - 1000,  # expired
            "old2": now - ENTITY_AUTH_TIMESTAMP_TTL_MS - 500,   # expired
            "new1": now - 1000,                                  # valid
            "new2": now - 500,                                   # valid
        }

        server._cleanup_expired_nonces(entity, now)

        self.assertEqual(len(server._entity_auth_nonces[entity]), 2)
        self.assertIn("new1", server._entity_auth_nonces[entity])
        self.assertIn("new2", server._entity_auth_nonces[entity])
        self.assertNotIn("old1", server._entity_auth_nonces[entity])

    def test_cleanup_nonces_bounded(self):
        """Nonce set is bounded to ENTITY_AUTH_MAX_NONCES."""
        from vanta_api.websocket_server import ENTITY_AUTH_MAX_NONCES

        server = self._make_server()
        entity = "5EntityHotkey"
        now = 10_000_000

        # Add more than max nonces, all valid
        for i in range(ENTITY_AUTH_MAX_NONCES + 100):
            server._entity_auth_nonces[entity][f"nonce_{i}"] = now - i

        server._cleanup_expired_nonces(entity, now)

        self.assertEqual(len(server._entity_auth_nonces[entity]), ENTITY_AUTH_MAX_NONCES)

    def test_cleanup_nonces_empty(self):
        """Cleanup on empty nonce set is a no-op."""
        server = self._make_server()
        server._cleanup_expired_nonces("nonexistent", 10000)
        # Should not raise

    # --- _authenticate_entity ---

    async def test_entity_auth_missing_fields(self):
        """Auth fails when required fields are missing."""
        server = self._make_server()
        ws = self._mock_websocket()

        result = await server._authenticate_entity("c1", ws, {
            "entity_hotkey": "5abc",
            # Missing timestamp, nonce, signature
        })

        self.assertFalse(result)
        ws.send.assert_called_once()
        msg = json.loads(ws.send.call_args[0][0])
        self.assertEqual(msg["status"], "error")
        self.assertIn("requires", msg["message"])

    async def test_entity_auth_expired_timestamp(self):
        """Auth fails when timestamp is outside the TTL window."""
        server = self._make_server()
        ws = self._mock_websocket()

        old_timestamp = int(time.time() * 1000) - 10 * 60 * 1000  # 10 min ago

        result = await server._authenticate_entity("c1", ws, {
            "entity_hotkey": "5abc",
            "timestamp": old_timestamp,
            "nonce": "unique-nonce",
            "signature": "deadbeef"
        })

        self.assertFalse(result)
        msg = json.loads(ws.send.call_args[0][0])
        self.assertIn("expired", msg["message"].lower())

    async def test_entity_auth_nonce_replay(self):
        """Auth fails when nonce has already been used."""
        server = self._make_server()
        ws = self._mock_websocket()
        entity = "5EntityKey"
        now_ms = int(time.time() * 1000)

        # Pre-register a nonce
        server._entity_auth_nonces[entity]["used-nonce"] = now_ms

        result = await server._authenticate_entity("c1", ws, {
            "entity_hotkey": entity,
            "timestamp": now_ms,
            "nonce": "used-nonce",
            "signature": "deadbeef"
        })

        self.assertFalse(result)
        msg = json.loads(ws.send.call_args[0][0])
        self.assertIn("Nonce already used", msg["message"])

    async def test_entity_auth_unregistered_entity(self):
        """Auth fails when entity is not registered."""
        server = self._make_server()
        ws = self._mock_websocket()
        now_ms = int(time.time() * 1000)

        server._entity_client.get_entity_data.return_value = None

        result = await server._authenticate_entity("c1", ws, {
            "entity_hotkey": "5Unregistered",
            "timestamp": now_ms,
            "nonce": "fresh-nonce",
            "signature": "deadbeef"
        })

        self.assertFalse(result)
        msg = json.loads(ws.send.call_args[0][0])
        self.assertIn("not registered", msg["message"])

    async def test_entity_auth_invalid_signature(self):
        """Auth fails when signature doesn't verify."""
        server = self._make_server()
        ws = self._mock_websocket()
        now_ms = int(time.time() * 1000)

        server._entity_client.get_entity_data.return_value = {"subaccounts": {}}

        # Use a valid-looking but wrong signature
        result = await server._authenticate_entity("c1", ws, {
            "entity_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "timestamp": now_ms,
            "nonce": "fresh-nonce",
            "signature": "00" * 64  # Wrong signature
        })

        self.assertFalse(result)
        # Should fail at signature verification
        msg = json.loads(ws.send.call_args[0][0])
        self.assertEqual(msg["status"], "error")

    async def test_entity_auth_success(self):
        """Full entity auth succeeds with valid signature."""
        server = self._make_server()
        ws = self._mock_websocket()

        try:
            from bittensor_wallet import Keypair
        except ImportError:
            self.skipTest("bittensor_wallet not installed")

        # Generate a real keypair for testing
        keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        entity_hotkey = keypair.ss58_address
        now_ms = int(time.time() * 1000)
        nonce = "test-nonce-123"

        # Sign the auth message
        message_dict = {
            "entity_hotkey": entity_hotkey,
            "nonce": nonce,
            "timestamp": now_ms
        }
        message_bytes = json.dumps(message_dict, sort_keys=True).encode('utf-8')
        signature = keypair.sign(message_bytes).hex()

        # Mock entity data with one active subaccount
        server._entity_client.get_entity_data.return_value = {
            "subaccounts": {
                "0": {
                    "status": "active",
                    "synthetic_hotkey": f"{entity_hotkey}_0"
                }
            }
        }

        # Mock _ensure_subaccount_poll_task to avoid event loop issues
        server._ensure_subaccount_poll_task = MagicMock()

        result = await server._authenticate_entity("c1", ws, {
            "entity_hotkey": entity_hotkey,
            "timestamp": now_ms,
            "nonce": nonce,
            "signature": signature
        })

        self.assertTrue(result)
        # Client should be registered
        self.assertIn("c1", server.connected_clients)
        self.assertEqual(server.client_auth["c1"]["auth_type"], "entity")
        self.assertEqual(server.client_auth["c1"]["entity_hotkey"], entity_hotkey)
        # Should be in entity_clients
        self.assertIn("c1", server.entity_clients[entity_hotkey])
        # Should be auto-subscribed to the subaccount
        self.assertIn("c1", server.subaccount_subscriptions[f"{entity_hotkey}_0"])

        # Auth success message sent
        last_send = ws.send.call_args_list[-1]
        msg = json.loads(last_send[0][0])
        self.assertEqual(msg["status"], "success")
        self.assertEqual(msg["auth_type"], "entity")
        self.assertEqual(msg["subscribed_subaccounts"], 1)

    async def test_entity_auth_fifo_eviction(self):
        """Oldest entity client is evicted when MAX_N_WS_PER_ENTITY is reached."""
        from vanta_api.websocket_server import MAX_N_WS_PER_ENTITY

        server = self._make_server()
        entity = "5EntityKey"

        try:
            from bittensor_wallet import Keypair
        except ImportError:
            self.skipTest("bittensor_wallet not installed")

        keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        entity = keypair.ss58_address

        server._entity_client.get_entity_data.return_value = {"subaccounts": {}}
        server._ensure_subaccount_poll_task = MagicMock()

        # Fill up to max connections
        for i in range(MAX_N_WS_PER_ENTITY):
            client_id = f"client_{i}"
            mock_ws = self._mock_websocket()
            server.connected_clients[client_id] = mock_ws
            server.client_auth[client_id] = {"auth_type": "entity", "entity_hotkey": entity}
            server.entity_clients[entity].append(client_id)

        self.assertEqual(len(server.entity_clients[entity]), MAX_N_WS_PER_ENTITY)

        # Now authenticate a new client — should evict client_0
        new_ws = self._mock_websocket()
        now_ms = int(time.time() * 1000)
        nonce = "eviction-nonce"
        msg_dict = {"entity_hotkey": entity, "nonce": nonce, "timestamp": now_ms}
        sig = keypair.sign(json.dumps(msg_dict, sort_keys=True).encode()).hex()

        result = await server._authenticate_entity("new_client", new_ws, {
            "entity_hotkey": entity, "timestamp": now_ms, "nonce": nonce, "signature": sig
        })

        self.assertTrue(result)
        self.assertNotIn("client_0", server.connected_clients)
        self.assertIn("new_client", server.entity_clients[entity])
        self.assertEqual(len(server.entity_clients[entity]), MAX_N_WS_PER_ENTITY)


# ==================== Auto-Subscribe Tests ====================

class TestAutoSubscribeEntitySubaccounts(unittest.IsolatedAsyncioTestCase):
    """Tests for _auto_subscribe_entity_subaccounts."""

    def _make_server(self):
        from vanta_api.websocket_server import WebSocketServer
        server = object.__new__(WebSocketServer)
        server.subaccount_subscriptions = defaultdict(set)
        server._subaccount_poll_tasks = {}
        server._ensure_subaccount_poll_task = MagicMock()
        return server

    async def test_subscribe_active_subaccounts(self):
        """Auto-subscribes to active and admin subaccounts."""
        server = self._make_server()
        entity = "5Entity"
        entity_data = {
            "subaccounts": {
                "0": {"status": "active", "synthetic_hotkey": f"{entity}_0"},
                "1": {"status": "admin", "synthetic_hotkey": f"{entity}_1"},
                "2": {"status": "eliminated", "synthetic_hotkey": f"{entity}_2"},
                "3": {"status": "failed", "synthetic_hotkey": f"{entity}_3"},
            }
        }

        count = await server._auto_subscribe_entity_subaccounts("c1", entity, entity_data)

        self.assertEqual(count, 2)  # only active + admin
        self.assertIn("c1", server.subaccount_subscriptions[f"{entity}_0"])
        self.assertIn("c1", server.subaccount_subscriptions[f"{entity}_1"])
        self.assertNotIn(f"{entity}_2", server.subaccount_subscriptions)
        self.assertNotIn(f"{entity}_3", server.subaccount_subscriptions)
        self.assertEqual(server._ensure_subaccount_poll_task.call_count, 2)

    async def test_subscribe_no_subaccounts(self):
        """Zero subscriptions when entity has no subaccounts."""
        server = self._make_server()
        count = await server._auto_subscribe_entity_subaccounts("c1", "5Entity", {"subaccounts": {}})
        self.assertEqual(count, 0)

    async def test_subscribe_generates_synthetic_hotkey(self):
        """Generates synthetic_hotkey from entity+id if not in subaccount data."""
        server = self._make_server()
        entity = "5Entity"
        entity_data = {
            "subaccounts": {
                "0": {"status": "active"}  # No synthetic_hotkey field
            }
        }

        count = await server._auto_subscribe_entity_subaccounts("c1", entity, entity_data)

        self.assertEqual(count, 1)
        self.assertIn("c1", server.subaccount_subscriptions[f"{entity}_0"])


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


# ==================== Remove Client Entity Cleanup Tests ====================

class TestRemoveClientEntityCleanup(TestBase):
    """Tests for _remove_client cleaning up entity tracking."""

    def _make_server(self):
        from vanta_api.websocket_server import WebSocketServer
        server = object.__new__(WebSocketServer)
        server.connected_clients = {}
        server.client_auth = {}
        server.api_key_clients = defaultdict(deque)
        server.entity_clients = defaultdict(deque)
        server.subscribed_clients = set()
        server.subaccount_subscriptions = defaultdict(set)
        server._subaccount_poll_tasks = {}
        server.api_key_to_alias = {}
        server._cancel_subaccount_poll_task = MagicMock()
        return server

    def test_remove_entity_client(self):
        """Removing an entity client cleans up entity_clients tracking."""
        server = self._make_server()
        entity = "5EntityKey"
        ws = MagicMock()

        # Register client
        server.connected_clients["c1"] = ws
        server.client_auth["c1"] = {"auth_type": "entity", "entity_hotkey": entity}
        server.entity_clients[entity].append("c1")
        server.subaccount_subscriptions[f"{entity}_0"].add("c1")

        # Remove
        server._remove_client("c1")

        self.assertNotIn("c1", server.connected_clients)
        self.assertNotIn("c1", server.client_auth)
        self.assertNotIn(entity, server.entity_clients)  # Empty deque removed
        self.assertNotIn("c1", server.subaccount_subscriptions.get(f"{entity}_0", set()))

    def test_remove_entity_client_preserves_others(self):
        """Removing one entity client doesn't affect other clients for the same entity."""
        server = self._make_server()
        entity = "5EntityKey"

        # Register two clients
        for cid in ["c1", "c2"]:
            server.connected_clients[cid] = MagicMock()
            server.client_auth[cid] = {"auth_type": "entity", "entity_hotkey": entity}
            server.entity_clients[entity].append(cid)

        server._remove_client("c1")

        self.assertNotIn("c1", server.connected_clients)
        self.assertIn("c2", server.connected_clients)
        self.assertEqual(len(server.entity_clients[entity]), 1)
        self.assertIn("c2", server.entity_clients[entity])

    def test_remove_api_key_client_no_entity_side_effects(self):
        """Removing an API key client doesn't touch entity_clients."""
        server = self._make_server()

        server.connected_clients["c1"] = MagicMock()
        server.client_auth["c1"] = {"auth_type": "api_key", "api_key": "key123", "tier": 100}
        server.api_key_clients["key123"].append("c1")

        server._remove_client("c1")

        self.assertNotIn("c1", server.connected_clients)
        # entity_clients should be untouched (empty)
        self.assertEqual(len(server.entity_clients), 0)


# ==================== Notify New Subaccount RPC Tests ====================

class TestNotifyNewSubaccount(TestBase):
    """Tests for notify_new_subaccount_rpc on the WebSocket server."""

    def _make_server(self):
        from vanta_api.websocket_server import WebSocketServer
        server = object.__new__(WebSocketServer)
        server.connected_clients = {}
        server.entity_clients = defaultdict(deque)
        server.subaccount_subscriptions = defaultdict(set)
        server.subaccount_last_broadcast_ms = {}
        server._subaccount_poll_tasks = {}
        server.sequence_number = 0
        server.loop = None  # Will prevent actual async send
        server._ensure_subaccount_poll_task = MagicMock()
        return server

    def test_notify_subscribes_connected_clients(self):
        """Connected entity clients are auto-subscribed to new subaccount."""
        server = self._make_server()
        entity = "5Entity"

        # Register two clients for this entity
        server.entity_clients[entity] = deque(["c1", "c2"])

        result = server.notify_new_subaccount_rpc(entity, f"{entity}_5")

        self.assertTrue(result)
        self.assertIn("c1", server.subaccount_subscriptions[f"{entity}_5"])
        self.assertIn("c2", server.subaccount_subscriptions[f"{entity}_5"])
        server._ensure_subaccount_poll_task.assert_called_once_with(f"{entity}_5")

    def test_notify_no_connected_clients(self):
        """Returns True (no-op) when entity has no connected clients."""
        server = self._make_server()
        result = server.notify_new_subaccount_rpc("5Offline", "5Offline_0")
        self.assertTrue(result)
        self.assertEqual(len(server.subaccount_subscriptions), 0)


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
    """Tests for rejection broadcast calls in HyperliquidTracker._process_fill."""

    def _make_tracker(self):
        """Create a HyperliquidTracker with all dependencies mocked."""
        from entity_management.hyperliquid_tracker import HyperliquidTracker

        tracker = HyperliquidTracker(
            entity_client=MagicMock(),
            elimination_client=MagicMock(),
            price_fetcher_client=MagicMock(),
            asset_selection_client=MagicMock(),
            market_order_manager=MagicMock(),
            limit_order_client=MagicMock(),
            uuid_tracker=MagicMock(),
            ws_notifier_client=MagicMock(),
        )
        return tracker

    def _make_fill(self, coin="BTC", side="B", sz="1.0", px="50000"):
        return {"coin": coin, "side": side, "sz": sz, "px": px}

    def _setup_valid_fill_path(self, tracker, synthetic="5Entity_0"):
        """Configure mocks so fill passes coin/hotkey resolution but hits fail-early checks."""
        from vali_objects.vali_config import DynamicTradePair

        # Coin resolves to a valid trade pair via dynamic registry
        tracker._hl_universe = {
            "BTC": DynamicTradePair(
                trade_pair_id="BTCUSDC", trade_pair="BTC/USDC", hl_coin="BTC", max_leverage=0.5
            )
        }
        tracker._entity_client.get_synthetic_hotkey_for_hl_address.return_value = synthetic
        tracker._entity_client.get_subaccount_info_for_synthetic.return_value = {
            "account_size": 100_000
        }

    def test_broadcast_rejection_calls_notifier(self):
        """_broadcast_rejection sends error via ws_notifier_client."""
        tracker = self._make_tracker()

        tracker._broadcast_rejection("5Entity_0", "Test error message")

        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once_with(
            "5Entity_0"
        )

    def test_broadcast_rejection_no_notifier(self):
        """_broadcast_rejection is a no-op when ws_notifier_client is None."""
        from entity_management.hyperliquid_tracker import HyperliquidTracker

        tracker = HyperliquidTracker(
            entity_client=MagicMock(),
            elimination_client=MagicMock(),
            price_fetcher_client=MagicMock(),
            asset_selection_client=MagicMock(),
            market_order_manager=MagicMock(),
            limit_order_client=MagicMock(),
            uuid_tracker=MagicMock(),
            ws_notifier_client=None,
        )
        # Should not raise
        tracker._broadcast_rejection("5Entity_0", "No crash")

    def test_broadcast_accepted_fill_calls_notifier(self):
        """Accepted fills are broadcast as order_event payloads."""
        tracker = self._make_tracker()

        tracker._broadcast_accepted_fill(
            synthetic_hotkey="5Entity_0",
            trade_pair="BTCUSD",
            order_type="LONG",
            fill_hash="0xabc123",
        )

        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once_with(
            "5Entity_0"
        )

    def test_rate_limit_rejection_broadcasts(self):
        """Rate limiting rejection triggers broadcast with wait time."""
        tracker = self._make_tracker()
        self._setup_valid_fill_path(tracker)

        # Rate limiter rejects
        tracker._rate_limiter = MagicMock()
        tracker._rate_limiter.is_allowed.return_value = (False, 5.0)

        tracker._process_fill("0xaddr", self._make_fill())

        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once()

    def test_elimination_rejection_broadcasts(self):
        """Eliminated miner rejection triggers broadcast."""
        tracker = self._make_tracker()
        self._setup_valid_fill_path(tracker)

        # Rate limiter allows
        tracker._rate_limiter = MagicMock()
        tracker._rate_limiter.is_allowed.return_value = (True, 0)

        # Elimination check returns data (eliminated)
        tracker._elimination_client.get_elimination_local_cache.return_value = {"reason": "mdd"}

        tracker._process_fill("0xaddr", self._make_fill())

        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once()

    def test_signal_exception_rejection_broadcasts(self):
        """SignalException during order processing triggers broadcast."""
        from vali_objects.exceptions.signal_exception import SignalException

        tracker = self._make_tracker()
        self._setup_valid_fill_path(tracker)

        tracker._rate_limiter = MagicMock()
        tracker._rate_limiter.is_allowed.return_value = (True, 0)
        tracker._elimination_client.get_elimination_local_cache.return_value = None
        tracker._entity_client.validate_hotkey_for_orders.return_value = {"is_valid": True}

        tracker._price_fetcher_client.is_market_open.return_value = True
        tracker._fetch_hl_account_state = MagicMock(return_value={
            "total_portfolio_value": 100_000,
            "positions": {"BTC": {"weight": 0.3}},
        })
        tracker._position_client = MagicMock()
        tracker._position_client.get_open_position_for_trade_pair.return_value = None

        with patch('entity_management.hyperliquid_tracker.OrderProcessor') as mock_op:
            mock_op.process_order.side_effect = SignalException("Leverage too high")

            tracker._process_fill("0xaddr", self._make_fill())

        tracker._ws_notifier_client.broadcast_subaccount_dashboard.assert_called_once()

    def test_unexpected_exception_rejection_broadcasts(self):
        """Unexpected exceptions during order processing also trigger rejection broadcast."""
        tracker = self._make_tracker()
        self._setup_valid_fill_path(tracker)

        tracker._rate_limiter = MagicMock()
        tracker._rate_limiter.is_allowed.return_value = (True, 0)
        tracker._elimination_client.get_elimination_local_cache.return_value = None
        tracker._entity_client.validate_hotkey_for_orders.return_value = {"is_valid": True}

        tracker._price_fetcher_client.is_market_open.return_value = True
        tracker._fetch_hl_account_state = MagicMock(return_value={
            "total_portfolio_value": 100_000,
            "positions": {"BTC": {"weight": 0.3}},
        })
        tracker._position_client = MagicMock()
        tracker._position_client.get_open_position_for_trade_pair.return_value = None

        with patch('entity_management.hyperliquid_tracker.OrderProcessor') as mock_op:
            mock_op.process_order.side_effect = ValueError("Position at max $1000.00 (limit: $1000.00)")

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
        import queue

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
