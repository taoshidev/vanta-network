# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Integration tests for ValidatorBroadcastBase and all inheriting classes.

Tests end-to-end broadcast scenarios using real server infrastructure:
- EntityManager: SubaccountRegistration broadcasts
- AssetSelectionManager: AssetSelection broadcasts
- ValidatorContractManager: CollateralRecord broadcasts

Pattern follows test_challengeperiod_integration.py with ServerOrchestrator.
"""
import unittest
from copy import deepcopy
from types import SimpleNamespace
import bittensor as bt

from time_util.time_util import TimeUtil
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.vali_config import ValiConfig, TradePairCategory
from vali_objects.utils.vali_utils import ValiUtils
from entitiy_management.entity_manager import EntityManager, EntityData, SubaccountInfo
from vali_objects.utils.asset_selection.asset_selection_manager import AssetSelectionManager
from vali_objects.contract.validator_contract_manager import ValidatorContractManager, CollateralRecord


class TestBroadcastIntegration(TestBase):
    """
    Integration tests for validator broadcast functionality using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    entity_client = None
    asset_selection_client = None
    contract_client = None
    metagraph_client = None
    subtensor_ops_client = None

    # Class-level constants
    MOTHERSHIP_HOTKEY = "test_mothership_hotkey"
    NON_MOTHERSHIP_HOTKEY = "test_non_mothership_hotkey"
    TEST_ENTITY_HOTKEY = "test_entity_hotkey"
    TEST_MINER_HOTKEY = "test_miner_hotkey"

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
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.asset_selection_client = cls.orchestrator.get_client('asset_selection')
        cls.contract_client = cls.orchestrator.get_client('contract')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.subtensor_ops_client = cls.orchestrator.get_client('subtensor_ops')

        bt.logging.info("[BROADCAST_INTEGRATION] Servers started and clients initialized")

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
        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Set up test metagraph
        self.metagraph_client.set_hotkeys([
            self.MOTHERSHIP_HOTKEY,
            self.NON_MOTHERSHIP_HOTKEY,
            self.TEST_MINER_HOTKEY
        ])

        bt.logging.info("[BROADCAST_INTEGRATION] Test setup complete")

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    # ==================== EntityManager Broadcast Tests ====================

    def test_entity_manager_subaccount_broadcast_mothership_to_receiver(self):
        """
        Test SubaccountRegistration broadcast from mothership to other validators.

        Flow:
        1. Mothership creates a subaccount (triggers broadcast)
        2. Non-mothership validator receives broadcast
        3. Non-mothership validator processes and persists the subaccount

        NOTE: In unit test mode, actual network broadcast is skipped (running_unit_tests=True).
        This test validates the broadcast/receive methods work correctly when called directly.
        """
        # Create a mock config for the mothership
        mothership_config = SimpleNamespace(
            netuid=116,  # testnet
            wallet=SimpleNamespace(hotkey=self.MOTHERSHIP_HOTKEY),
            subtensor=SimpleNamespace(network="test")
        )

        # Create mothership manager with running_unit_tests=True
        # IMPORTANT: This prevents actual network calls in ValidatorBroadcastBase._broadcast_to_validators
        mothership_manager = EntityManager(
            running_unit_tests=True,
            config=mothership_config,
            is_backtesting=False
        )

        # Manually set is_mothership to True for testing
        # (In production, this is derived from ValiUtils.is_mothership_wallet)
        mothership_manager.is_mothership = True

        # 1. Mothership registers an entity
        success, msg = mothership_manager.register_entity(
            entity_hotkey=self.TEST_ENTITY_HOTKEY,
            max_subaccounts=5
        )
        self.assertTrue(success, f"Entity registration failed: {msg}")

        # 2. Mothership creates a subaccount
        success, subaccount_info, msg = mothership_manager.create_subaccount(
            entity_hotkey=self.TEST_ENTITY_HOTKEY
        )
        self.assertTrue(success, f"Subaccount creation failed: {msg}")
        self.assertIsNotNone(subaccount_info)
        self.assertEqual(subaccount_info.subaccount_id, 0)
        self.assertEqual(subaccount_info.synthetic_hotkey, f"{self.TEST_ENTITY_HOTKEY}_0")

        # 3. Simulate broadcast reception on non-mothership validator
        # Create non-mothership manager
        non_mothership_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey=self.NON_MOTHERSHIP_HOTKEY),
            subtensor=SimpleNamespace(network="test")
        )

        non_mothership_manager = EntityManager(
            running_unit_tests=True,
            config=non_mothership_config,
            is_backtesting=False
        )
        non_mothership_manager.is_mothership = False

        # Prepare broadcast data
        subaccount_data = {
            "entity_hotkey": self.TEST_ENTITY_HOTKEY,
            "subaccount_id": subaccount_info.subaccount_id,
            "subaccount_uuid": subaccount_info.subaccount_uuid,
            "synthetic_hotkey": subaccount_info.synthetic_hotkey
        }

        # IMPORTANT: Must temporarily set MOTHERSHIP_HOTKEY in ValiConfig for sender verification
        original_mothership_hotkey = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = self.MOTHERSHIP_HOTKEY

        try:
            # 4. Non-mothership receives broadcast
            result = non_mothership_manager.receive_subaccount_registration_update(
                subaccount_data=subaccount_data,
                sender_hotkey=self.MOTHERSHIP_HOTKEY  # Sender is mothership
            )

            self.assertTrue(result, "Broadcast reception failed")

            # 5. Verify subaccount was persisted on non-mothership
            entity_data = non_mothership_manager.get_entity_data(self.TEST_ENTITY_HOTKEY)
            self.assertIsNotNone(entity_data, "Entity data not found on non-mothership")
            self.assertEqual(len(entity_data.subaccounts), 1)
            self.assertIn(0, entity_data.subaccounts)

            received_subaccount = entity_data.subaccounts[0]
            self.assertEqual(received_subaccount.subaccount_uuid, subaccount_info.subaccount_uuid)
            self.assertEqual(received_subaccount.synthetic_hotkey, subaccount_info.synthetic_hotkey)
            self.assertEqual(received_subaccount.status, "active")

            bt.logging.success(
                f"✓ SubaccountRegistration broadcast test passed:\n"
                f"  - Mothership created subaccount: {subaccount_info.synthetic_hotkey}\n"
                f"  - Non-mothership received and persisted subaccount\n"
                f"  - Subaccount status: {received_subaccount.status}"
            )
        finally:
            # Restore original MOTHERSHIP_HOTKEY
            ValiConfig.MOTHERSHIP_HOTKEY = original_mothership_hotkey

    def test_entity_manager_reject_unauthorized_broadcast(self):
        """
        Test that non-mothership broadcasts are rejected by verify_broadcast_sender.

        Security test: Only mothership should be able to send broadcasts.
        """
        # Create non-mothership manager
        non_mothership_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey=self.NON_MOTHERSHIP_HOTKEY),
            subtensor=SimpleNamespace(network="test")
        )

        receiver_manager = EntityManager(
            running_unit_tests=True,
            config=non_mothership_config,
            is_backtesting=False
        )
        receiver_manager.is_mothership = False

        # Prepare fake broadcast from non-mothership
        subaccount_data = {
            "entity_hotkey": self.TEST_ENTITY_HOTKEY,
            "subaccount_id": 0,
            "subaccount_uuid": "fake-uuid",
            "synthetic_hotkey": f"{self.TEST_ENTITY_HOTKEY}_0"
        }

        # IMPORTANT: Set MOTHERSHIP_HOTKEY for verification
        original_mothership_hotkey = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = self.MOTHERSHIP_HOTKEY

        try:
            # Attempt to receive broadcast from unauthorized sender
            result = receiver_manager.receive_subaccount_registration_update(
                subaccount_data=subaccount_data,
                sender_hotkey=self.NON_MOTHERSHIP_HOTKEY  # UNAUTHORIZED SENDER
            )

            # Should be rejected
            self.assertFalse(result, "Unauthorized broadcast should be rejected")

            # Verify no data was persisted
            entity_data = receiver_manager.get_entity_data(self.TEST_ENTITY_HOTKEY)
            self.assertIsNone(entity_data, "Entity should not exist after rejected broadcast")

            bt.logging.success(
                "✓ Unauthorized broadcast rejection test passed:\n"
                f"  - Broadcast from {self.NON_MOTHERSHIP_HOTKEY} was rejected\n"
                f"  - Expected mothership: {self.MOTHERSHIP_HOTKEY}"
            )
        finally:
            # Restore original MOTHERSHIP_HOTKEY
            ValiConfig.MOTHERSHIP_HOTKEY = original_mothership_hotkey

    # ==================== AssetSelectionManager Broadcast Tests ====================

    def test_asset_selection_manager_broadcast_mothership_to_receiver(self):
        """
        Test AssetSelection broadcast from mothership to other validators.

        Flow:
        1. Mothership processes asset selection (triggers broadcast)
        2. Non-mothership validator receives broadcast
        3. Non-mothership validator processes and persists the selection
        """
        # Create mothership manager
        mothership_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey=self.MOTHERSHIP_HOTKEY),
            subtensor=SimpleNamespace(network="test")
        )

        mothership_manager = AssetSelectionManager(
            running_unit_tests=True,
            config=mothership_config
        )
        mothership_manager.is_mothership = True

        # 1. Mothership processes asset selection
        result = mothership_manager.process_asset_selection_request(
            asset_selection="crypto",
            miner=self.TEST_MINER_HOTKEY
        )
        self.assertTrue(result['successfully_processed'])
        asset_class = result['asset_class']

        # 2. Create non-mothership manager
        non_mothership_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey=self.NON_MOTHERSHIP_HOTKEY),
            subtensor=SimpleNamespace(network="test")
        )

        non_mothership_manager = AssetSelectionManager(
            running_unit_tests=True,
            config=non_mothership_config
        )
        non_mothership_manager.is_mothership = False

        # Prepare broadcast data
        asset_selection_data = {
            "hotkey": self.TEST_MINER_HOTKEY,
            "asset_selection": asset_class.value
        }

        # Set MOTHERSHIP_HOTKEY for verification
        original_mothership_hotkey = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = self.MOTHERSHIP_HOTKEY

        try:
            # 3. Non-mothership receives broadcast
            result = non_mothership_manager.receive_asset_selection_update(
                asset_selection_data=asset_selection_data,
                sender_hotkey=self.MOTHERSHIP_HOTKEY
            )

            self.assertTrue(result, "Broadcast reception failed")

            # 4. Verify asset selection was persisted on non-mothership
            received_selection = non_mothership_manager.get_asset_selection(self.TEST_MINER_HOTKEY)
            self.assertIsNotNone(received_selection)
            self.assertEqual(received_selection, TradePairCategory.CRYPTO)

            bt.logging.success(
                f"✓ AssetSelection broadcast test passed:\n"
                f"  - Mothership selected asset class: {asset_class.value}\n"
                f"  - Non-mothership received and persisted selection\n"
                f"  - Miner: {self.TEST_MINER_HOTKEY}"
            )
        finally:
            ValiConfig.MOTHERSHIP_HOTKEY = original_mothership_hotkey

    def test_asset_selection_manager_reject_unauthorized_broadcast(self):
        """
        Test that non-mothership asset selection broadcasts are rejected.
        """
        # Create receiver manager
        receiver_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey=self.NON_MOTHERSHIP_HOTKEY),
            subtensor=SimpleNamespace(network="test")
        )

        receiver_manager = AssetSelectionManager(
            running_unit_tests=True,
            config=receiver_config
        )
        receiver_manager.is_mothership = False

        # Fake broadcast data
        asset_selection_data = {
            "hotkey": self.TEST_MINER_HOTKEY,
            "asset_selection": "crypto"
        }

        # Set MOTHERSHIP_HOTKEY
        original_mothership_hotkey = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = self.MOTHERSHIP_HOTKEY

        try:
            # Attempt unauthorized broadcast
            result = receiver_manager.receive_asset_selection_update(
                asset_selection_data=asset_selection_data,
                sender_hotkey=self.NON_MOTHERSHIP_HOTKEY  # UNAUTHORIZED
            )

            # Should be rejected
            self.assertFalse(result)

            # Verify no data persisted
            selection = receiver_manager.get_asset_selection(self.TEST_MINER_HOTKEY)
            self.assertIsNone(selection, "Asset selection should not exist after rejected broadcast")

            bt.logging.success("✓ Unauthorized AssetSelection broadcast rejected")
        finally:
            ValiConfig.MOTHERSHIP_HOTKEY = original_mothership_hotkey

    # ==================== ValidatorContractManager Broadcast Tests ====================

    def test_contract_manager_collateral_record_broadcast_mothership_to_receiver(self):
        """
        Test CollateralRecord broadcast from mothership to other validators.

        Flow:
        1. Mothership sets miner account size (triggers broadcast)
        2. Non-mothership validator receives broadcast
        3. Non-mothership validator processes and persists the record

        NOTE: Uses test collateral balance injection to avoid blockchain calls.
        """
        # Create mothership manager
        mothership_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey=self.MOTHERSHIP_HOTKEY),
            subtensor=SimpleNamespace(network="test")
        )

        mothership_manager = ValidatorContractManager(
            config=mothership_config,
            running_unit_tests=True,
            is_backtesting=False
        )
        mothership_manager.is_mothership = True

        # Inject test collateral balance to avoid blockchain call
        # 1000 theta = 1000 * 10^9 rao
        test_balance_rao = 1000 * 10**9
        mothership_manager.set_test_collateral_balance(self.TEST_MINER_HOTKEY, test_balance_rao)

        # 1. Mothership sets account size
        timestamp_ms = TimeUtil.now_in_millis()
        success = mothership_manager.set_miner_account_size(
            hotkey=self.TEST_MINER_HOTKEY,
            timestamp_ms=timestamp_ms
        )
        self.assertTrue(success, "Failed to set account size")

        # Get the collateral record from mothership
        account_size = mothership_manager.get_miner_account_size(
            hotkey=self.TEST_MINER_HOTKEY,
            most_recent=True
        )
        self.assertIsNotNone(account_size)

        # 2. Create non-mothership manager
        non_mothership_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey=self.NON_MOTHERSHIP_HOTKEY),
            subtensor=SimpleNamespace(network="test")
        )

        non_mothership_manager = ValidatorContractManager(
            config=non_mothership_config,
            running_unit_tests=True,
            is_backtesting=False
        )
        non_mothership_manager.is_mothership = False

        # Prepare broadcast data
        collateral_balance_theta = mothership_manager.to_theta(test_balance_rao)
        expected_account_size = min(
            ValiConfig.MAX_COLLATERAL_BALANCE_THETA,
            collateral_balance_theta
        ) * ValiConfig.COST_PER_THETA

        collateral_record_data = {
            "hotkey": self.TEST_MINER_HOTKEY,
            "account_size": expected_account_size,
            "account_size_theta": collateral_balance_theta,
            "update_time_ms": timestamp_ms
        }

        # Set MOTHERSHIP_HOTKEY
        original_mothership_hotkey = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = self.MOTHERSHIP_HOTKEY

        try:
            # 3. Non-mothership receives broadcast
            result = non_mothership_manager.receive_collateral_record_update(
                collateral_record_data=collateral_record_data,
                sender_hotkey=self.MOTHERSHIP_HOTKEY
            )

            self.assertTrue(result, "Broadcast reception failed")

            # 4. Verify collateral record was persisted on non-mothership
            received_account_size = non_mothership_manager.get_miner_account_size(
                hotkey=self.TEST_MINER_HOTKEY,
                most_recent=True
            )
            self.assertIsNotNone(received_account_size)
            self.assertEqual(received_account_size, expected_account_size)

            bt.logging.success(
                f"✓ CollateralRecord broadcast test passed:\n"
                f"  - Mothership set account size: ${expected_account_size:,.2f}\n"
                f"  - Non-mothership received and persisted record\n"
                f"  - Miner: {self.TEST_MINER_HOTKEY}"
            )
        finally:
            ValiConfig.MOTHERSHIP_HOTKEY = original_mothership_hotkey

    def test_contract_manager_reject_unauthorized_broadcast(self):
        """
        Test that non-mothership collateral record broadcasts are rejected.
        """
        # Create receiver manager
        receiver_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey=self.NON_MOTHERSHIP_HOTKEY),
            subtensor=SimpleNamespace(network="test")
        )

        receiver_manager = ValidatorContractManager(
            config=receiver_config,
            running_unit_tests=True,
            is_backtesting=False
        )
        receiver_manager.is_mothership = False

        # Fake broadcast data
        timestamp_ms = TimeUtil.now_in_millis()
        collateral_record_data = {
            "hotkey": self.TEST_MINER_HOTKEY,
            "account_size": 100000.0,
            "account_size_theta": 1000.0,
            "update_time_ms": timestamp_ms
        }

        # Set MOTHERSHIP_HOTKEY
        original_mothership_hotkey = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = self.MOTHERSHIP_HOTKEY

        try:
            # Attempt unauthorized broadcast
            result = receiver_manager.receive_collateral_record_update(
                collateral_record_data=collateral_record_data,
                sender_hotkey=self.NON_MOTHERSHIP_HOTKEY  # UNAUTHORIZED
            )

            # Should be rejected
            self.assertFalse(result)

            # Verify no data persisted
            account_size = receiver_manager.get_miner_account_size(
                hotkey=self.TEST_MINER_HOTKEY,
                most_recent=True
            )
            self.assertIsNone(account_size, "Account size should not exist after rejected broadcast")

            bt.logging.success("✓ Unauthorized CollateralRecord broadcast rejected")
        finally:
            ValiConfig.MOTHERSHIP_HOTKEY = original_mothership_hotkey

    # ==================== Cross-Manager Integration Tests ====================

    def test_all_managers_broadcast_in_sequence(self):
        """
        Test that all three managers can broadcast in sequence without conflicts.

        This validates that the shared ValidatorBroadcastBase infrastructure
        works correctly when multiple managers broadcast different synapse types.
        """
        # Set MOTHERSHIP_HOTKEY
        original_mothership_hotkey = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = self.MOTHERSHIP_HOTKEY

        try:
            # Create mothership config
            mothership_config = SimpleNamespace(
                netuid=116,
                wallet=SimpleNamespace(hotkey=self.MOTHERSHIP_HOTKEY),
                subtensor=SimpleNamespace(network="test")
            )

            # Create non-mothership config
            non_mothership_config = SimpleNamespace(
                netuid=116,
                wallet=SimpleNamespace(hotkey=self.NON_MOTHERSHIP_HOTKEY),
                subtensor=SimpleNamespace(network="test")
            )

            # 1. EntityManager broadcast
            entity_mothership = EntityManager(
                running_unit_tests=True,
                config=mothership_config,
                is_backtesting=False
            )
            entity_mothership.is_mothership = True
            entity_mothership.register_entity(self.TEST_ENTITY_HOTKEY, max_subaccounts=5)
            success, subaccount_info, msg = entity_mothership.create_subaccount(self.TEST_ENTITY_HOTKEY)
            self.assertTrue(success)

            entity_receiver = EntityManager(
                running_unit_tests=True,
                config=non_mothership_config,
                is_backtesting=False
            )
            entity_receiver.is_mothership = False
            result = entity_receiver.receive_subaccount_registration_update(
                {
                    "entity_hotkey": self.TEST_ENTITY_HOTKEY,
                    "subaccount_id": 0,
                    "subaccount_uuid": subaccount_info.subaccount_uuid,
                    "synthetic_hotkey": subaccount_info.synthetic_hotkey
                },
                sender_hotkey=self.MOTHERSHIP_HOTKEY
            )
            self.assertTrue(result)

            # 2. AssetSelectionManager broadcast
            asset_mothership = AssetSelectionManager(
                running_unit_tests=True,
                config=mothership_config
            )
            asset_mothership.is_mothership = True
            result = asset_mothership.process_asset_selection_request("crypto", self.TEST_MINER_HOTKEY)
            self.assertTrue(result['successfully_processed'])

            asset_receiver = AssetSelectionManager(
                running_unit_tests=True,
                config=non_mothership_config
            )
            asset_receiver.is_mothership = False
            result = asset_receiver.receive_asset_selection_update(
                {
                    "hotkey": self.TEST_MINER_HOTKEY,
                    "asset_selection": "crypto"
                },
                sender_hotkey=self.MOTHERSHIP_HOTKEY
            )
            self.assertTrue(result)

            # 3. ValidatorContractManager broadcast
            contract_mothership = ValidatorContractManager(
                config=mothership_config,
                running_unit_tests=True,
                is_backtesting=False
            )
            contract_mothership.is_mothership = True
            test_balance_rao = 1000 * 10**9
            contract_mothership.set_test_collateral_balance(self.TEST_MINER_HOTKEY, test_balance_rao)
            timestamp_ms = TimeUtil.now_in_millis()
            success = contract_mothership.set_miner_account_size(self.TEST_MINER_HOTKEY, timestamp_ms)
            self.assertTrue(success)

            contract_receiver = ValidatorContractManager(
                config=non_mothership_config,
                running_unit_tests=True,
                is_backtesting=False
            )
            contract_receiver.is_mothership = False
            collateral_balance_theta = contract_mothership.to_theta(test_balance_rao)
            account_size = min(
                ValiConfig.MAX_COLLATERAL_BALANCE_THETA,
                collateral_balance_theta
            ) * ValiConfig.COST_PER_THETA
            result = contract_receiver.receive_collateral_record_update(
                {
                    "hotkey": self.TEST_MINER_HOTKEY,
                    "account_size": account_size,
                    "account_size_theta": collateral_balance_theta,
                    "update_time_ms": timestamp_ms
                },
                sender_hotkey=self.MOTHERSHIP_HOTKEY
            )
            self.assertTrue(result)

            # Verify all data persisted correctly
            entity_data = entity_receiver.get_entity_data(self.TEST_ENTITY_HOTKEY)
            self.assertIsNotNone(entity_data)
            self.assertEqual(len(entity_data.subaccounts), 1)

            asset_selection = asset_receiver.get_asset_selection(self.TEST_MINER_HOTKEY)
            self.assertEqual(asset_selection, TradePairCategory.CRYPTO)

            received_account_size = contract_receiver.get_miner_account_size(
                self.TEST_MINER_HOTKEY,
                most_recent=True
            )
            self.assertEqual(received_account_size, account_size)

            bt.logging.success(
                "✓ All managers broadcast test passed:\n"
                "  - EntityManager: SubaccountRegistration broadcast successful\n"
                "  - AssetSelectionManager: AssetSelection broadcast successful\n"
                "  - ValidatorContractManager: CollateralRecord broadcast successful"
            )
        finally:
            ValiConfig.MOTHERSHIP_HOTKEY = original_mothership_hotkey


if __name__ == '__main__':
    unittest.main()
