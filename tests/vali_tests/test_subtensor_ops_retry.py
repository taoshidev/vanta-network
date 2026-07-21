# developer: lp
# Copyright (c) 2026 Taoshi Inc
"""
Regression tests for metagraph-update failure accounting and connection
recreation on the LIVE daemon path (SubtensorOpsServer.run_daemon_iteration ->
SubtensorOpsManager.update_metagraph).

Bug fixed by these tests' subject: the RPC-daemon path never incremented
manager.consecutive_failures, so the connection-recreation block in
update_metagraph() (and round-robin endpoint rotation on mainnet networks) was
unreachable after runtime failures — a validator that drew a sick backend from
the chain endpoint's load balancer retried the same dead websocket forever.
Observed on testnet, where wss://test.finney.opentensor.ai balances across
backends at wildly different states.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from shared_objects.subtensor_ops.subtensor_ops import SubtensorOpsManager
from shared_objects.subtensor_ops.subtensor_ops_server import SubtensorOpsServer

TEST_HOTKEY = "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw"


def _mock_config(netuid=116, network="test"):
    """Mirror test_metagraph_updater's mock config (testnet defaults here)."""
    config = Mock()
    config.netuid = netuid
    config.subtensor = Mock()
    config.subtensor.network = network
    config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443"
    config.wallet = Mock()
    config.wallet.name = "test_wallet"
    config.wallet.hotkey = "test_hotkey"
    config.wallet.path = "~/.bittensor/wallets"
    config.logging = Mock()
    config.logging.debug = False
    config.logging.trace = False
    config.logging.logging_dir = "~/.bittensor/miners"
    return config


def _make_manager():
    """Real manager in unit-test mode (mock subtensor), miner mode to keep
    dependencies minimal; the accounting under test is mode-independent."""
    manager = SubtensorOpsManager(
        config=_mock_config(),
        hotkey=TEST_HOTKEY,
        is_miner=True,
        running_unit_tests=True,
    )
    # Wire a minimal metagraph client (normally done by the orchestrator).
    manager._metagraph_client = Mock(
        get_hotkeys=Mock(return_value=[]),
        update_metagraph=Mock(),
    )
    return manager


def _run_daemon_iteration(manager):
    """Invoke the real wrapper logic against a minimal host object, so the
    accounting is tested without spawning RPCServerBase machinery."""
    host = SimpleNamespace(manager=manager, _is_shutdown=lambda: False)
    return SubtensorOpsServer.run_daemon_iteration(host)


class TestDaemonFailureAccounting(unittest.TestCase):
    """The wrapper must record failures on the manager and re-raise."""

    def test_failure_increments_manager_counter_and_reraises(self):
        manager = Mock()
        manager.consecutive_failures = 0
        manager.update_metagraph = Mock(side_effect=RuntimeError("boom"))
        host = SimpleNamespace(manager=manager, _is_shutdown=lambda: False)

        with self.assertRaises(RuntimeError):
            SubtensorOpsServer.run_daemon_iteration(host)
        self.assertEqual(manager.consecutive_failures, 1)

        with self.assertRaises(RuntimeError):
            SubtensorOpsServer.run_daemon_iteration(host)
        self.assertEqual(manager.consecutive_failures, 2)

    def test_shutdown_skips_update_entirely(self):
        manager = Mock()
        manager.update_metagraph = Mock()
        host = SimpleNamespace(manager=manager, _is_shutdown=lambda: True)

        SubtensorOpsServer.run_daemon_iteration(host)
        manager.update_metagraph.assert_not_called()


class TestReconnectEngagement(unittest.TestCase):
    """With failures pending, the next update must recreate the connection —
    even when round-robin is disabled (network='test')."""

    def test_pending_failure_recreates_connection_without_round_robin(self):
        manager = _make_manager()
        self.assertFalse(manager.round_robin_enabled)
        old_subtensor = manager.subtensor

        manager.consecutive_failures = 1  # as recorded by run_daemon_iteration
        manager.update_metagraph()

        # Fresh connection swapped in; old one cleaned up; endpoint untouched.
        self.assertIsNot(manager.subtensor, old_subtensor)
        old_subtensor.substrate.close.assert_called_once()
        self.assertEqual(manager.config.subtensor.network, "test")
        # Fully successful sync resets the accounting.
        self.assertEqual(manager.consecutive_failures, 0)

    def test_no_recreation_when_no_failures_pending(self):
        manager = _make_manager()
        old_subtensor = manager.subtensor

        manager.update_metagraph()

        self.assertIs(manager.subtensor, old_subtensor)
        self.assertEqual(manager.consecutive_failures, 0)

    def test_full_daemon_cycle_fail_then_heal(self):
        """End-to-end through the real wrapper: failure arms the reconnect,
        the next iteration recreates and resets."""
        manager = _make_manager()
        old_subtensor = manager.subtensor

        # Iteration 1: sick backend — sync raises out of the real manager.
        manager.subtensor.metagraph = Mock(side_effect=Exception("Internal error"))
        with self.assertRaises(Exception):
            _run_daemon_iteration(manager)
        self.assertEqual(manager.consecutive_failures, 1)

        # Iteration 2: reconnect engages, mock 'backend' now healthy.
        _run_daemon_iteration(manager)
        self.assertIsNot(manager.subtensor, old_subtensor)
        self.assertEqual(manager.consecutive_failures, 0)

    def test_anomalous_metagraph_early_return_does_not_reset_counter(self):
        """The anomalous-loss early return is not a success: pending failure
        accounting must survive it (reset happens only after a full sync)."""
        manager = _make_manager()
        # Client currently knows many hotkeys; the (mock) chain returns none —
        # a 100% loss is rejected as anomalous and the update early-returns.
        # (Anomaly needs BOTH an absolute count above the minimum AND a high
        # percentage — 30 lost of 30 clears both thresholds.)
        manager._metagraph_client.get_hotkeys = Mock(
            return_value=[f"hk{i}" for i in range(30)]
        )
        manager.consecutive_failures = 2

        manager.update_metagraph()  # recreates connection, then rejects sync

        self.assertEqual(manager.consecutive_failures, 2)


if __name__ == "__main__":
    unittest.main()
