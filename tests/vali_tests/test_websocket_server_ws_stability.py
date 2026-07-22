# developer: lp
# Copyright (c) 2026 Taoshi Inc
"""
Unit tests for the PTN WebSocket server stability fixes:
  A1  _remove_client works under websockets 15 (no fail_connection / no
      websockets.protocol.OPEN) and never raises during teardown.
  A2  Failed/empty dashboard builds back off exponentially instead of being
      re-hammered by the 0.1s staleness scanner.
  A6  Frame serialization happens once, off the event loop, with a monotonic
      per-client sequence number.

These construct the server via object.__new__ and set only the attributes the
methods under test touch — the same pattern the existing gateway tests use —
so no RPC servers or api-keys file are needed.
"""
import unittest
from collections import defaultdict, deque
from unittest.mock import MagicMock, patch

from websockets.protocol import State

from vanta_api.websocket_server import (
    WebSocketServer,
    WebSocketServerClient,
    DashboardSubscription,
    SUBACCOUNT_RETRY_BASE_MS,
    SUBACCOUNT_RETRY_MAX_MS,
)


def _bare_server():
    server = object.__new__(WebSocketServer)
    server._clients = {}
    server._api_key_client_ids = defaultdict(deque)
    server.api_key_to_alias = {}
    server._event_loop = MagicMock()
    server._event_loop.is_closed.return_value = False
    server._entity_client = MagicMock()
    for attr in (
        "_challenge_period_client", "_elimination_client", "_miner_account_client",
        "_position_client", "_limit_order_client", "_debt_ledger_client",
        "_statistics_client",
    ):
        setattr(server, attr, MagicMock())
    return server


class TestBuildBackoff(unittest.TestCase):
    def test_backoff_increments_and_caps(self):
        sub = DashboardSubscription()
        prev = 0
        for i in range(1, 12):
            WebSocketServer._apply_build_backoff(sub)
            self.assertEqual(sub.failure_count, i)
            delay = sub.next_attempt_ms  # relative to now; monotonic growth then cap
            self.assertGreater(delay, prev - 1)
            prev = delay
        # After many failures the delay is capped at SUBACCOUNT_RETRY_MAX_MS.
        # failure_count is large here, so the raw exp far exceeds the cap.
        import time as _t
        now = int(_t.time() * 1000)
        self.assertLessEqual(sub.next_attempt_ms - now, SUBACCOUNT_RETRY_MAX_MS + 50)

    def test_first_backoff_is_base(self):
        sub = DashboardSubscription()
        import time as _t
        before = int(_t.time() * 1000)
        WebSocketServer._apply_build_backoff(sub)
        # first failure → base delay
        self.assertGreaterEqual(sub.next_attempt_ms, before + SUBACCOUNT_RETRY_BASE_MS - 50)
        self.assertLessEqual(sub.next_attempt_ms, before + SUBACCOUNT_RETRY_BASE_MS + 100)


class TestSendDashboardUpdateBackoff(unittest.TestCase):
    def test_none_dashboard_backs_off_and_does_not_send(self):
        server = _bare_server()
        server._entity_client.get_subaccount_dashboard.return_value = None
        server._send_message = MagicMock()
        sub = DashboardSubscription()

        with patch("vanta_api.websocket_server.asyncio.run_coroutine_threadsafe") as rcs:
            server._send_dashboard_update("5Entity_1", MagicMock(), sub)

        self.assertEqual(sub.failure_count, 1)
        self.assertGreater(sub.next_attempt_ms, 0)
        self.assertEqual(sub.last_update_time_ms, 0)  # NOT advanced → scanner won't re-hammer
        rcs.assert_not_called()

    def test_success_resets_backoff_and_sends(self):
        server = _bare_server()
        server._entity_client.get_subaccount_dashboard.return_value = {"ok": True}
        server._send_serialized = MagicMock(return_value=MagicMock())  # not a real coroutine
        sub = DashboardSubscription(failure_count=4, next_attempt_ms=999999999999)
        client = MagicMock()
        client.serialize.return_value = '{"sequence":0}'

        with patch("vanta_api.websocket_server.create_subaccount_dashboard",
                   return_value={"subaccount_info": {}}), \
             patch("vanta_api.websocket_server.asyncio.run_coroutine_threadsafe") as rcs:
            server._send_dashboard_update("5Entity_1", client, sub)

        self.assertEqual(sub.failure_count, 0)
        self.assertEqual(sub.next_attempt_ms, 0)
        self.assertGreater(sub.last_update_time_ms, 0)  # update_times stamped it
        # Serialized in the worker (off the loop), then raw send scheduled.
        client.serialize.assert_called_once_with({"dashboard": {"subaccount_info": {}}})
        rcs.assert_called_once()

    def test_build_exception_backs_off(self):
        server = _bare_server()
        server._entity_client.get_subaccount_dashboard.return_value = {"ok": True}
        sub = DashboardSubscription()

        with patch("vanta_api.websocket_server.create_subaccount_dashboard",
                   side_effect=RuntimeError("boom")), \
             patch("vanta_api.websocket_server.asyncio.run_coroutine_threadsafe") as rcs:
            server._send_dashboard_update("5Entity_1", MagicMock(), sub)

        self.assertEqual(sub.failure_count, 1)
        self.assertEqual(sub.last_update_time_ms, 0)
        rcs.assert_not_called()


class TestClientSerialize(unittest.TestCase):
    def _client(self):
        ws = MagicMock()
        return WebSocketServerClient(client_id=1, websocket=ws, api_key="k", tier=200)

    def test_serialize_shape_and_sequence(self):
        import json as _json
        client = self._client()
        s0 = _json.loads(client.serialize({"dashboard": {"a": 1}}))
        s1 = _json.loads(client.serialize({"dashboard": {"a": 2}}))
        self.assertEqual(s0["sequence"], 0)
        self.assertEqual(s1["sequence"], 1)
        self.assertEqual(s0["data"], {"dashboard": {"a": 1}})
        self.assertIn("timestamp", s0)

    def test_sequence_monotonic_under_concurrency(self):
        import json as _json
        from concurrent.futures import ThreadPoolExecutor
        client = self._client()
        N = 500

        def one(_):
            return _json.loads(client.serialize({"x": 1}))["sequence"]

        with ThreadPoolExecutor(max_workers=16) as pool:
            seqs = list(pool.map(one, range(N)))

        # No duplicates and no gaps despite concurrent worker-thread serialization.
        self.assertEqual(sorted(seqs), list(range(N)))
        self.assertEqual(client.sequence_number, N)


class TestInitialSnapshot(unittest.TestCase):
    def test_submits_build_when_subscribed(self):
        server = _bare_server()
        server._thread_pool = MagicMock()
        client = MagicMock()
        sub = DashboardSubscription()
        client.dashboard_subscriptions = {"5Entity_1": sub}

        server._submit_initial_snapshot("5Entity_1", client)

        server._thread_pool.submit.assert_called_once_with(
            server._send_dashboard_update, "5Entity_1", client, sub)

    def test_noop_when_not_subscribed(self):
        server = _bare_server()
        server._thread_pool = MagicMock()
        client = MagicMock()
        client.dashboard_subscriptions = {}

        server._submit_initial_snapshot("5Entity_1", client)

        server._thread_pool.submit.assert_not_called()


class TestRemoveClient(unittest.TestCase):
    def _client(self, state=State.OPEN):
        ws = MagicMock()
        ws.state = state
        return WebSocketServerClient(client_id=1, websocket=ws, api_key="k", tier=200)

    def test_open_client_schedules_close_and_removes(self):
        server = _bare_server()
        client = self._client(State.OPEN)
        server._clients[1] = client
        server._api_key_client_ids["k"].append(1)

        with patch("vanta_api.websocket_server.asyncio.run_coroutine_threadsafe") as rcs:
            server._remove_client(1)  # must not raise

        self.assertNotIn(1, server._clients)
        rcs.assert_called_once()

    def test_no_loop_does_not_raise_or_schedule(self):
        server = _bare_server()
        server._event_loop = None
        client = self._client(State.OPEN)
        server._clients[1] = client
        server._api_key_client_ids["k"].append(1)

        with patch("vanta_api.websocket_server.asyncio.run_coroutine_threadsafe") as rcs:
            server._remove_client(1)  # must not raise

        self.assertNotIn(1, server._clients)
        rcs.assert_not_called()

    def test_closed_client_no_close_scheduled(self):
        server = _bare_server()
        client = self._client(State.CLOSED)
        server._clients[1] = client

        with patch("vanta_api.websocket_server.asyncio.run_coroutine_threadsafe") as rcs:
            server._remove_client(1)

        self.assertNotIn(1, server._clients)
        rcs.assert_not_called()

    def test_unknown_client_is_noop(self):
        server = _bare_server()
        server._remove_client(999)  # must not raise


if __name__ == "__main__":
    unittest.main()
