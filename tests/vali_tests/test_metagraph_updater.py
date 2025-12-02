# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Test suite for MetagraphUpdater that verifies both miner and validator modes.

Tests metagraph syncing, caching, and validator-specific weight setting with
mocked network connections (handled internally by MetagraphUpdater when running_unit_tests=True).
"""
import unittest
from unittest.mock import Mock
from dataclasses import dataclass

from shared_objects.metagraph_updater import MetagraphUpdater
from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase

from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig


# Simple picklable data structures for testing
@dataclass
class SimpleAxonInfo:
    """Simple picklable axon info for testing."""
    ip: str
    port: int


@dataclass
class SimpleNeuron:
    """Simple picklable neuron for testing."""
    uid: int
    hotkey: str
    incentive: float
    validator_trust: float
    axon_info: SimpleAxonInfo


class TestMetagraphUpdater(TestBase):
    """
    Integration tests for MetagraphUpdater using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).

    Tests both miner and validator modes with mocked subtensor connections.
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    metagraph_client = None
    live_price_fetcher_client = None

    # Test hotkeys
    TEST_VALIDATOR_HOTKEY = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    TEST_MINER_HOTKEY = "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw"

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
        # Get the singleton orchestrator and start all required servers
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator (servers guaranteed ready, no connection delays)
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.live_price_fetcher_client = cls.orchestrator.get_client('live_price_fetcher')

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No action needed.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # NOTE: Skip super().setUp() to avoid killing ports (servers already running)

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    # ==================== Helper Methods ====================

    def _create_mock_config(self, netuid=8, network="finney"):
        """Create a mock config for MetagraphUpdater tests."""
        config = Mock()
        config.netuid = netuid
        config.subtensor = Mock()
        config.subtensor.network = network
        config.subtensor.chain_endpoint = f"wss://entrypoint-{network}.opentensor.ai:443"

        # Mock wallet config
        config.wallet = Mock()
        config.wallet.name = "test_wallet"
        config.wallet.hotkey = "test_hotkey"
        config.wallet.path = "~/.bittensor/wallets"

        # Mock logging config
        config.logging = Mock()
        config.logging.debug = False
        config.logging.trace = False
        config.logging.logging_dir = "~/.bittensor/miners"

        return config

    def _create_mock_neuron(self, uid, hotkey, incentive=0.0, validator_trust=0.0):
        """Create a simple picklable neuron object for testing."""
        axon_info = SimpleAxonInfo(ip="192.168.1.1", port=8091)
        return SimpleNeuron(
            uid=uid,
            hotkey=hotkey,
            incentive=incentive,
            validator_trust=validator_trust,
            axon_info=axon_info
        )

    def _create_mock_metagraph(self, hotkeys_list):
        """Create a mock metagraph with specified hotkeys."""
        # NOTE: This is only used for helper methods now - the actual mocking
        # is done inside MetagraphUpdater via set_mock_metagraph_data()
        mock_metagraph = Mock()
        mock_metagraph.hotkeys = hotkeys_list
        mock_metagraph.uids = list(range(len(hotkeys_list)))
        mock_metagraph.block_at_registration = [1000] * len(hotkeys_list)
        mock_metagraph.emission = [1.0] * len(hotkeys_list)

        # Create simple picklable neurons
        neurons = [
            self._create_mock_neuron(i, hk, incentive=0.1, validator_trust=0.1)
            for i, hk in enumerate(hotkeys_list)
        ]
        mock_metagraph.neurons = neurons

        # Use simple picklable axons
        mock_metagraph.axons = [n.axon_info for n in neurons]

        # Mock pool data (for validators)
        mock_metagraph.pool = Mock()
        mock_metagraph.pool.tao_in = 1000.0  # 1000 TAO
        mock_metagraph.pool.alpha_in = 5000.0  # 5000 ALPHA

        return mock_metagraph

    def _create_mock_subtensor(self, hotkeys_list):
        """Create a mock subtensor that returns a mock metagraph."""
        mock_subtensor = Mock()
        mock_subtensor.metagraph = Mock(return_value=self._create_mock_metagraph(hotkeys_list))

        # Mock set_weights method
        mock_subtensor.set_weights = Mock(return_value=(True, None))

        # Mock substrate connection for cleanup
        mock_subtensor.substrate = Mock()
        mock_subtensor.substrate.close = Mock()

        return mock_subtensor

    def _create_mock_wallet(self, hotkey):
        """Create a mock wallet for weight setting tests."""
        mock_wallet = Mock()
        mock_wallet.hotkey = Mock()
        mock_wallet.hotkey.ss58_address = hotkey
        return mock_wallet

    def _create_mock_position_inspector(self):
        """Create a mock position inspector for miner tests."""
        mock_inspector = Mock()
        mock_inspector.get_recently_acked_validators = Mock(return_value=[])
        return mock_inspector

    # ==================== Validator Mode Tests ====================

    def test_validator_initialization(self):
        """Test MetagraphUpdater initialization in validator mode."""
        # Create validator MetagraphUpdater (mocking is handled internally)
        config = self._create_mock_config()
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_VALIDATOR_HOTKEY,
            is_miner=False,
            running_unit_tests=True
        )

        # Verify validator-specific initialization
        self.assertFalse(updater.is_miner)
        self.assertTrue(updater.is_validator)
        self.assertIsNotNone(updater.live_price_fetcher)
        self.assertIsNotNone(updater.weight_failure_tracker)
        self.assertEqual(
            updater.interval_wait_time_ms,
            ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_VALIDATOR_MS
        )
        # Verify mock subtensor was created
        self.assertIsNotNone(updater.subtensor)

    def test_validator_metagraph_update(self):
        """Test metagraph update in validator mode."""
        # Setup test data
        hotkeys = [self.TEST_VALIDATOR_HOTKEY, self.TEST_MINER_HOTKEY]
        config = self._create_mock_config()

        # Create validator MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_VALIDATOR_HOTKEY,
            is_miner=False,
            running_unit_tests=True
        )

        # Set mock metagraph data
        updater.set_mock_metagraph_data(hotkeys)


        # Perform metagraph update
        updater.update_metagraph()

        # Verify metagraph data was updated
        updated_hotkeys = self.metagraph_client.get_hotkeys()
        self.assertEqual(len(updated_hotkeys), 2)
        self.assertIn(self.TEST_VALIDATOR_HOTKEY, updated_hotkeys)
        self.assertIn(self.TEST_MINER_HOTKEY, updated_hotkeys)

    def test_validator_hotkey_cache(self):
        """Test hotkey cache updates correctly in validator mode."""
        # Setup test data
        initial_hotkeys = [self.TEST_VALIDATOR_HOTKEY, self.TEST_MINER_HOTKEY]
        config = self._create_mock_config()

        # Create validator MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_VALIDATOR_HOTKEY,
            is_miner=False,
            running_unit_tests=True
        )

        # Set mock metagraph data before updating
        updater.set_mock_metagraph_data(initial_hotkeys)

        # Perform initial update
        updater.update_metagraph()

        # Verify cache is populated
        self.assertTrue(updater.is_hotkey_registered_cached(self.TEST_VALIDATOR_HOTKEY))
        self.assertTrue(updater.is_hotkey_registered_cached(self.TEST_MINER_HOTKEY))

        # Add a new hotkey to the metagraph
        new_hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        updated_hotkeys = initial_hotkeys + [new_hotkey]
        updater.set_mock_metagraph_data(updated_hotkeys)

        # Perform another update
        updater.update_metagraph()

        # Verify cache is updated
        self.assertTrue(updater.is_hotkey_registered_cached(new_hotkey))

    def test_validator_weight_setting_rpc(self):
        """Test weight setting via RPC in validator mode."""
        # Setup test data
        hotkeys = [self.TEST_VALIDATOR_HOTKEY, self.TEST_MINER_HOTKEY]
        config = self._create_mock_config()

        # Create validator MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_VALIDATOR_HOTKEY,
            is_miner=False,
            running_unit_tests=True
        )
        updater.set_mock_metagraph_data(hotkeys)

        # Call set_weights_rpc directly (simulating SubtensorWeightCalculator)
        uids = [0, 1]
        weights = [0.6, 0.4]
        version_key = 200

        result = updater.set_weights_rpc(uids, weights, version_key)

        # Verify result (mock subtensor always returns success)
        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])

    def test_validator_weight_setting_failure_tracking(self):
        """Test weight failure tracking in validator mode."""
        # Setup test data
        hotkeys = [self.TEST_VALIDATOR_HOTKEY, self.TEST_MINER_HOTKEY]
        config = self._create_mock_config()

        # Create validator MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_VALIDATOR_HOTKEY,
            is_miner=False,
            running_unit_tests=True
        )
        updater.set_mock_metagraph_data(hotkeys)

        # Mock set_weights to fail
        error_msg = "Subtensor returned: Invalid transaction"
        updater.subtensor.set_weights = Mock(return_value=(False, error_msg))

        # Call set_weights_rpc (should fail)
        result = updater.set_weights_rpc([0, 1], [0.6, 0.4], 200)

        # Verify failure was tracked
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertEqual(updater.weight_failure_tracker.consecutive_failures, 1)

        # Classify the failure
        failure_type = updater.weight_failure_tracker.classify_failure(error_msg)
        self.assertEqual(failure_type, "critical")

    # ==================== Miner Mode Tests ====================

    def test_miner_initialization(self):
        """Test MetagraphUpdater initialization in miner mode."""
        # Setup test data
        config = self._create_mock_config()
        mock_position_inspector = self._create_mock_position_inspector()

        # Create miner MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_MINER_HOTKEY,
            is_miner=True,
            position_inspector=mock_position_inspector,
            running_unit_tests=True
        )

        # Verify miner-specific initialization
        self.assertTrue(updater.is_miner)
        self.assertFalse(updater.is_validator)
        self.assertIsNone(updater.live_price_fetcher)
        self.assertIsNone(updater.weight_failure_tracker)
        self.assertEqual(
            updater.interval_wait_time_ms,
            ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_MINER_MS
        )
        # Verify mock subtensor was created
        self.assertIsNotNone(updater.subtensor)

    def test_miner_metagraph_update(self):
        """Test metagraph update in miner mode."""
        # Setup test data
        hotkeys = [self.TEST_VALIDATOR_HOTKEY, self.TEST_MINER_HOTKEY]
        config = self._create_mock_config()
        mock_position_inspector = self._create_mock_position_inspector()

        # Create miner MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_MINER_HOTKEY,
            is_miner=True,
            position_inspector=mock_position_inspector,
            running_unit_tests=True
        )
        updater.set_mock_metagraph_data(hotkeys)

        # Perform metagraph update
        updater.update_metagraph()

        # Verify metagraph data was updated
        updated_hotkeys = self.metagraph_client.get_hotkeys()
        self.assertEqual(len(updated_hotkeys), 2)
        self.assertIn(self.TEST_VALIDATOR_HOTKEY, updated_hotkeys)
        self.assertIn(self.TEST_MINER_HOTKEY, updated_hotkeys)

    def test_miner_hotkey_cache(self):
        """Test hotkey cache updates correctly in miner mode."""
        # Setup test data
        initial_hotkeys = [self.TEST_VALIDATOR_HOTKEY, self.TEST_MINER_HOTKEY]
        config = self._create_mock_config()
        mock_position_inspector = self._create_mock_position_inspector()

        # Create miner MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_MINER_HOTKEY,
            is_miner=True,
            position_inspector=mock_position_inspector,
            running_unit_tests=True
        )
        updater.set_mock_metagraph_data(initial_hotkeys)

        # Perform initial update
        updater.update_metagraph()

        # Verify cache is populated
        self.assertTrue(updater.is_hotkey_registered_cached(self.TEST_VALIDATOR_HOTKEY))
        self.assertTrue(updater.is_hotkey_registered_cached(self.TEST_MINER_HOTKEY))

        # Verify unregistered hotkey returns False
        unregistered_hotkey = "5FakeHotkeyNotInMetagraph"
        self.assertFalse(updater.is_hotkey_registered_cached(unregistered_hotkey))

    def test_miner_validator_estimation(self):
        """Test likely validator estimation in miner mode."""
        # Setup test data with different validator_trust values
        hotkeys = [self.TEST_VALIDATOR_HOTKEY, self.TEST_MINER_HOTKEY]
        config = self._create_mock_config()
        mock_position_inspector = self._create_mock_position_inspector()

        # Create neurons with different validator_trust values
        validator_neuron = self._create_mock_neuron(0, self.TEST_VALIDATOR_HOTKEY, incentive=0.1, validator_trust=0.8)
        miner_neuron = self._create_mock_neuron(1, self.TEST_MINER_HOTKEY, incentive=0.1, validator_trust=0.0)
        neurons = [validator_neuron, miner_neuron]

        # Create miner MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_MINER_HOTKEY,
            is_miner=True,
            position_inspector=mock_position_inspector,
            running_unit_tests=True
        )
        updater.set_mock_metagraph_data(hotkeys, neurons=neurons)

        # Perform metagraph update
        updater.update_metagraph()

        # Estimate validators
        n_validators = updater.estimate_number_of_validators()
        self.assertGreaterEqual(n_validators, 1)  # At least one validator

    # ==================== Common Tests (Both Modes) ====================

    def test_anomalous_hotkey_loss_detection(self):
        """Test that anomalous hotkey losses are detected and rejected."""
        # Setup test data with many hotkeys
        initial_hotkeys = [f"5Hotkey{i:04d}" for i in range(100)]
        config = self._create_mock_config()
        mock_position_inspector = self._create_mock_position_inspector()

        # Create miner MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_MINER_HOTKEY,
            is_miner=True,
            position_inspector=mock_position_inspector,
            running_unit_tests=True
        )
        updater.set_mock_metagraph_data(initial_hotkeys)

        # Perform initial update
        updater.update_metagraph()

        # Verify initial state
        self.assertEqual(len(self.metagraph_client.get_hotkeys()), 100)

        # Simulate anomalous loss (50% of hotkeys lost)
        remaining_hotkeys = initial_hotkeys[:50]
        updater.set_mock_metagraph_data(remaining_hotkeys)

        # Perform update (should be rejected)
        updater.update_metagraph()

        # Verify metagraph was NOT updated (still has 100 hotkeys)
        self.assertEqual(len(self.metagraph_client.get_hotkeys()), 100)

    def test_normal_hotkey_changes(self):
        """Test that normal hotkey additions/removals are accepted."""
        # Setup test data
        initial_hotkeys = [self.TEST_VALIDATOR_HOTKEY, self.TEST_MINER_HOTKEY]
        config = self._create_mock_config()
        mock_position_inspector = self._create_mock_position_inspector()

        # Create miner MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_MINER_HOTKEY,
            is_miner=True,
            position_inspector=mock_position_inspector,
            running_unit_tests=True
        )
        updater.set_mock_metagraph_data(initial_hotkeys)

        # Perform initial update
        updater.update_metagraph()
        self.assertEqual(len(self.metagraph_client.get_hotkeys()), 2)

        # Add a new hotkey (normal change)
        new_hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        updated_hotkeys = initial_hotkeys + [new_hotkey]
        updater.set_mock_metagraph_data(updated_hotkeys)

        # Perform update (should be accepted)
        updater.update_metagraph()

        # Verify metagraph was updated
        self.assertEqual(len(self.metagraph_client.get_hotkeys()), 3)
        self.assertIn(new_hotkey, self.metagraph_client.get_hotkeys())

    def test_round_robin_network_switching(self):
        """Test round-robin network switching on failures."""
        # Setup test data with round-robin enabled
        hotkeys = [self.TEST_VALIDATOR_HOTKEY, self.TEST_MINER_HOTKEY]
        config = self._create_mock_config(network="finney")  # Enable round-robin
        mock_position_inspector = self._create_mock_position_inspector()

        # Create miner MetagraphUpdater (mocking handled internally)
        updater = MetagraphUpdater(
            config=config,
            hotkey=self.TEST_MINER_HOTKEY,
            is_miner=True,
            position_inspector=mock_position_inspector,
            running_unit_tests=True
        )
        updater.set_mock_metagraph_data(hotkeys)

        # Verify round-robin is enabled
        self.assertTrue(updater.round_robin_enabled)
        self.assertEqual(updater.current_round_robin_index, 0)  # finney index

        # Simulate network switch
        initial_network = updater.config.subtensor.network
        updater._switch_to_next_network(cleanup_connection=False, create_new_subtensor=False)

        # Verify network was switched
        self.assertNotEqual(updater.config.subtensor.network, initial_network)
        self.assertEqual(updater.current_round_robin_index, 1)  # subvortex index


if __name__ == '__main__':
    unittest.main()
