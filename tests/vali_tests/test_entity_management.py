# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Entity Management unit tests using the new client/server architecture.

This test file validates the core entity management functionality including:
- Entity registration
- Subaccount creation and tracking
- Synthetic hotkey validation
- Subaccount elimination
- Metagraph integration
"""
import unittest
from types import SimpleNamespace

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtCheckpoint, DebtLedger
from vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger import SettledSegment
from time_util.time_util import MS_IN_WEEK, TimeUtil
from entity_management.entity_utils import is_synthetic_hotkey, parse_synthetic_hotkey
from vali_objects.enums.miner_bucket_enum import MinerBucket


class TestEntityManagement(TestBase):
    """
    Entity Management unit tests using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    entity_client = None
    metagraph_client = None
    challenge_period_client = None

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
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')

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

        # Set up test entities (avoid pattern {text}_{number} to prevent synthetic hotkey collision)
        self.ENTITY_HOTKEY_1 = "entity_alpha"
        self.ENTITY_HOTKEY_2 = "entity_beta"
        self.ENTITY_HOTKEY_3 = "entity_gamma"

        # Initialize metagraph with test entities
        self.metagraph_client.set_hotkeys([
            self.ENTITY_HOTKEY_1,
            self.ENTITY_HOTKEY_2,
            self.ENTITY_HOTKEY_3
        ])

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    # ==================== Entity Registration Tests ====================

    def test_register_entity_success(self):
        """Test successful entity registration."""
        success, message = self.entity_client.register_entity(
            entity_hotkey=self.ENTITY_HOTKEY_1
        )

        self.assertTrue(success, f"Entity registration failed: {message}")

        # Verify entity exists
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertIsNotNone(entity_data)
        self.assertEqual(entity_data['entity_hotkey'], self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data['subaccounts']), 0)

    def test_register_entity_duplicate(self):
        """Test that registering the same entity twice fails."""
        # Register first time
        success, _ = self.entity_client.register_entity(
            entity_hotkey=self.ENTITY_HOTKEY_1
        )
        self.assertTrue(success)

        # Try to register again
        success, message = self.entity_client.register_entity(
            entity_hotkey=self.ENTITY_HOTKEY_1
        )
        self.assertFalse(success)
        self.assertIn("already registered", message.lower())

    def test_register_entity_default_values(self):
        """Test entity registration with default values."""
        success, _ = self.entity_client.register_entity(
            entity_hotkey=self.ENTITY_HOTKEY_1
        )
        self.assertTrue(success)

        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)

    # ==================== Subaccount Creation Tests ====================

    def test_create_subaccount_success(self):
        """Test successful subaccount creation."""
        # Register entity first
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        # Create subaccount
        success, subaccount_info, message = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=100_000,
            asset_class="crypto"
        )

        self.assertTrue(success, f"Subaccount creation failed: {message}")
        self.assertIsNotNone(subaccount_info)
        self.assertEqual(subaccount_info['subaccount_id'], 0)
        self.assertEqual(subaccount_info['status'], 'active')

        # Verify synthetic hotkey format
        synthetic_hotkey = subaccount_info['synthetic_hotkey']
        self.assertEqual(synthetic_hotkey, f"{self.ENTITY_HOTKEY_1}_0")

    def test_create_subaccount_is_always_standard(self):
        """Every newly created subaccount starts on the standard track."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        success, subaccount_info, message = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=100_000,
            asset_class="crypto"
        )

        self.assertTrue(success, f"Subaccount creation failed: {message}")
        self.assertEqual(subaccount_info['account_type'], 'standard')
        bucket = self.challenge_period_client.get_miner_bucket(subaccount_info['synthetic_hotkey'])
        self.assertEqual(bucket, MinerBucket.SUBACCOUNT_CHALLENGE)

    def test_create_hl_subaccount_is_always_standard(self):
        """Hyperliquid subaccounts have no pro tier - they always start on the standard track."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        success, subaccount_info, message = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=100_000,
            hl_address="0x" + "a" * 40,
        )

        self.assertTrue(success, f"Subaccount creation failed: {message}")
        self.assertEqual(subaccount_info['account_type'], 'standard')
        bucket = self.challenge_period_client.get_miner_bucket(subaccount_info['synthetic_hotkey'])
        self.assertEqual(bucket, MinerBucket.SUBACCOUNT_CHALLENGE)

    def test_create_multiple_subaccounts(self):
        """Test creating multiple subaccounts for an entity."""
        # Register entity
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        # Create 3 subaccounts
        subaccount_ids = []
        for i in range(3):
            success, subaccount_info, _ = self.entity_client.create_subaccount(
                entity_hotkey=self.ENTITY_HOTKEY_1,
                account_size=100_000,
                asset_class="crypto"
            )
            self.assertTrue(success)
            subaccount_ids.append(subaccount_info['subaccount_id'])

        # Verify sequential IDs (0, 1, 2)
        self.assertEqual(subaccount_ids, [0, 1, 2])

        # Verify entity data
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data['subaccounts']), 3)

    # def test_create_subaccount_max_limit(self):
    #     """Test that subaccount creation fails when max limit is reached."""
    #     # TODO: mock override max_subaccounts
    #     # Register entity
    #     self.entity_client.register_entity(
    #         entity_hotkey=self.ENTITY_HOTKEY_1
    #     )

    #     # Create 2 subaccounts (should succeed)
    #     for i in range(2):
    #         success, _, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")
    #         self.assertTrue(success)

    #     # Try to create 3rd subaccount (should fail)
    #     success, subaccount_info, message = self.entity_client.create_subaccount(
    #         self.ENTITY_HOTKEY_1,
    #         account_size=100_000,
    #         asset_class="crypto"
    #     )
    #     self.assertFalse(success)
    #     self.assertIsNone(subaccount_info)
    #     self.assertIn("maximum", message.lower())

    def test_create_subaccount_unregistered_entity(self):
        """Test that subaccount creation fails for unregistered entity."""
        success, subaccount_info, message = self.entity_client.create_subaccount(
            entity_hotkey="unregistered_entity",
            account_size=100_000,
            asset_class="crypto"
        )

        self.assertFalse(success)
        self.assertIsNone(subaccount_info)
        self.assertIn("not registered", message.lower())

    # ==================== Synthetic Hotkey Tests ====================

    def test_is_synthetic_hotkey_valid(self):
        """Test synthetic hotkey detection using entity_utils directly."""
        # Valid synthetic hotkeys
        self.assertTrue(is_synthetic_hotkey("entity_123"))
        self.assertTrue(is_synthetic_hotkey("my_entity_0"))
        self.assertTrue(is_synthetic_hotkey("foo_bar_99"))

        # Invalid synthetic hotkeys (no underscore + integer)
        self.assertFalse(is_synthetic_hotkey("regular_hotkey"))
        self.assertFalse(is_synthetic_hotkey("no_number_"))
        self.assertFalse(is_synthetic_hotkey("just_text"))

    def test_parse_synthetic_hotkey_valid(self):
        """Test parsing valid synthetic hotkeys using entity_utils directly."""
        entity_hotkey, subaccount_id = parse_synthetic_hotkey(
            "my_entity_5"
        )
        self.assertEqual(entity_hotkey, "my_entity")
        self.assertEqual(subaccount_id, 5)

        # Test with entity hotkey containing underscores
        entity_hotkey, subaccount_id = parse_synthetic_hotkey(
            "entity_with_underscores_123"
        )
        self.assertEqual(entity_hotkey, "entity_with_underscores")
        self.assertEqual(subaccount_id, 123)

    def test_parse_synthetic_hotkey_invalid(self):
        """Test parsing invalid synthetic hotkeys using entity_utils directly."""
        entity_hotkey, subaccount_id = parse_synthetic_hotkey(
            "invalid_hotkey"
        )
        self.assertIsNone(entity_hotkey)
        self.assertIsNone(subaccount_id)

    # ==================== Subaccount Status Tests ====================

    def test_get_subaccount_status_active(self):
        """Test getting status of an active subaccount."""
        # Register entity and create subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Get status
        found, status, returned_hotkey = self.entity_client.get_subaccount_status(
            synthetic_hotkey
        )

        self.assertTrue(found)
        self.assertEqual(status, 'active')
        self.assertEqual(returned_hotkey, synthetic_hotkey)

    def test_get_subaccount_status_eliminated(self):
        """Test getting status of an eliminated subaccount."""
        # Register entity and create subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Eliminate subaccount
        self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            subaccount_id=0,
            reason="test_elimination"
        )

        # Get status
        found, status, returned_hotkey = self.entity_client.get_subaccount_status(
            synthetic_hotkey
        )

        self.assertTrue(found)
        self.assertEqual(status, 'eliminated')
        self.assertEqual(returned_hotkey, synthetic_hotkey)

    def test_get_subaccount_status_not_found(self):
        """Test getting status of non-existent subaccount."""
        found, status, returned_hotkey = self.entity_client.get_subaccount_status(
            "nonexistent_entity_0"
        )

        self.assertFalse(found)
        self.assertIsNone(status)

    # ==================== Subaccount Elimination Tests ====================

    def test_eliminate_subaccount_success(self):
        """Test successful subaccount elimination."""
        # Register entity and create subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")

        # Eliminate subaccount
        success, message = self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            subaccount_id=0,
            reason="test_elimination"
        )

        self.assertTrue(success, f"Subaccount elimination failed: {message}")

        # Verify status changed to eliminated
        found, status, _ = self.entity_client.get_subaccount_status(
            f"{self.ENTITY_HOTKEY_1}_0"
        )
        self.assertTrue(found)
        self.assertEqual(status, 'eliminated')

    def test_eliminate_subaccount_nonexistent(self):
        """Test eliminating a non-existent subaccount."""
        # Register entity without creating subaccounts
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        # Try to eliminate non-existent subaccount
        success, message = self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            subaccount_id=999,
            reason="test"
        )

        self.assertFalse(success)
        self.assertIn("not found", message.lower())

    def test_eliminate_already_eliminated_subaccount(self):
        """Test eliminating an already eliminated subaccount."""
        # Register entity and create subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")

        # Eliminate subaccount first time
        success, _ = self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            subaccount_id=0,
            reason="first_elimination"
        )
        self.assertTrue(success)

        # Try to eliminate again
        success, message = self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            subaccount_id=0,
            reason="second_elimination"
        )

        # Should still succeed (idempotent)
        self.assertTrue(success)

    # ==================== Metagraph Integration Tests ====================

    def test_metagraph_has_hotkey_entity(self):
        """Test that regular entity hotkeys are recognized by metagraph."""
        # Entity hotkey should be in metagraph (set in setUp)
        self.assertTrue(self.metagraph_client.has_hotkey(self.ENTITY_HOTKEY_1))

    def test_metagraph_has_hotkey_synthetic_active(self):
        """Test that active synthetic hotkeys are recognized by metagraph."""
        # Register entity and create subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Synthetic hotkey should be recognized (entity in metagraph + subaccount active)
        self.assertTrue(self.metagraph_client.has_hotkey(synthetic_hotkey))

    def test_metagraph_has_hotkey_synthetic_eliminated(self):
        """Test that eliminated synthetic hotkeys are NOT recognized by metagraph."""
        # Register entity and create subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Eliminate subaccount
        self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            subaccount_id=0,
            reason="test"
        )

        # Synthetic hotkey should NOT be recognized (eliminated)
        self.assertFalse(self.metagraph_client.has_hotkey(synthetic_hotkey))

    def test_metagraph_has_hotkey_synthetic_entity_not_in_metagraph(self):
        """Test that synthetic hotkeys fail if entity not in metagraph."""
        # Register entity that's NOT in metagraph
        unregistered_entity = "entity_not_in_metagraph"
        self.entity_client.register_entity(entity_hotkey=unregistered_entity)
        _, subaccount_info, _ = self.entity_client.create_subaccount(unregistered_entity, account_size=100_000, asset_class="crypto")
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Synthetic hotkey should NOT be recognized (entity not in metagraph)
        self.assertFalse(self.metagraph_client.has_hotkey(synthetic_hotkey))

    # ==================== Query Tests ====================

    def test_get_all_entities(self):
        """Test getting all entities."""
        # Register multiple entities
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_2)
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_3)

        # Get all entities
        all_entities = self.entity_client.get_all_entities()

        self.assertEqual(len(all_entities), 3)
        self.assertIn(self.ENTITY_HOTKEY_1, all_entities)
        self.assertIn(self.ENTITY_HOTKEY_2, all_entities)
        self.assertIn(self.ENTITY_HOTKEY_3, all_entities)

    def test_get_entity_data_nonexistent(self):
        """Test getting data for non-existent entity."""
        entity_data = self.entity_client.get_entity_data("nonexistent_entity")
        self.assertIsNone(entity_data)

    # ==================== Validator Order Placement Logic Tests ====================
    # These tests verify the behavior expected by validator.py's should_fail_early()
    # method for entity hotkey validation (lines 482-506 in neurons/validator.py).

    def test_validator_entity_hotkey_detection(self):
        """
        Test that entity hotkeys can be detected for order rejection.

        Validator logic:
        - Entity hotkeys (non-synthetic) should be rejected
        - Only synthetic hotkeys can place orders
        """
        # Register an entity
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        # Verify entity hotkey is NOT synthetic (should be rejected for orders)
        hotkey_is_synthetic = is_synthetic_hotkey(self.ENTITY_HOTKEY_1)
        self.assertFalse(hotkey_is_synthetic, "Entity hotkey should not be synthetic")

        # Verify entity data exists (allows validator to detect and reject)
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertIsNotNone(entity_data, "Entity data should exist for rejection check")

    def test_validator_synthetic_hotkey_active_acceptance(self):
        """
        Test that active synthetic hotkeys are accepted for orders.

        Validator logic:
        - Synthetic hotkeys with status='active' should be accepted
        """
        # Register entity and create active subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Verify hotkey is synthetic
        hotkey_is_synthetic = is_synthetic_hotkey(synthetic_hotkey)
        self.assertTrue(hotkey_is_synthetic, "Subaccount hotkey should be synthetic")

        # Verify status is active (should be accepted for orders)
        found, status, _ = self.entity_client.get_subaccount_status(synthetic_hotkey)
        self.assertTrue(found)
        self.assertEqual(status, 'active', "Active subaccount should be accepted for orders")

    def test_validator_synthetic_hotkey_eliminated_rejection(self):
        """
        Test that eliminated synthetic hotkeys are rejected for orders.

        Validator logic:
        - Synthetic hotkeys with status='eliminated' should be rejected
        """
        # Register entity and create subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Eliminate the subaccount
        self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            subaccount_id=0,
            reason="test_elimination"
        )

        # Verify hotkey is synthetic
        hotkey_is_synthetic = is_synthetic_hotkey(synthetic_hotkey)
        self.assertTrue(hotkey_is_synthetic, "Subaccount hotkey should be synthetic")

        # Verify status is eliminated (should be rejected for orders)
        found, status, _ = self.entity_client.get_subaccount_status(synthetic_hotkey)
        self.assertTrue(found)
        self.assertEqual(status, 'eliminated', "Eliminated subaccount should be rejected for orders")

    def test_validator_non_entity_regular_hotkey_acceptance(self):
        """
        Test that regular miner hotkeys (non-entity, non-synthetic) are accepted.

        Validator logic:
        - Regular hotkeys that are neither entity nor synthetic should pass through
        """
        regular_hotkey = "regular_miner_hotkey"

        # Verify it's not synthetic
        hotkey_is_synthetic = is_synthetic_hotkey(regular_hotkey)
        self.assertFalse(hotkey_is_synthetic, "Regular hotkey should not be synthetic")

        # Verify it's not an entity
        entity_data = self.entity_client.get_entity_data(regular_hotkey)
        self.assertIsNone(entity_data, "Regular hotkey should not be an entity")

    # ==================== Entity Sync Tests (Auto-Sync Integration) ====================

    def test_sync_entity_data_new_entity(self):
        """Test syncing a new entity from checkpoint."""
        # Create checkpoint dict with new entity
        checkpoint_dict = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'test-uuid-0',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': TimeUtil.now_in_millis(),
                        'eliminated_at_ms': None,
                        'account_size': 100_000,
                        'asset_class': 'crypto'
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': TimeUtil.now_in_millis()
            }
        }

        # Sync entity data
        stats = self.entity_client.sync_entity_data(checkpoint_dict)

        # Verify stats
        self.assertEqual(stats['entities_added'], 1)
        self.assertEqual(stats['subaccounts_added'], 1)
        self.assertEqual(stats['subaccounts_updated'], 0)

        # Verify entity exists
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertIsNotNone(entity_data)
        self.assertEqual(len(entity_data['subaccounts']), 1)
        self.assertEqual(entity_data['next_subaccount_id'], 1)

    def test_sync_entity_data_new_subaccount(self):
        """Test syncing new subaccounts to existing entity."""
        # Register entity locally with 1 subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")

        # Create checkpoint dict with additional subaccounts (0, 1, 2)
        checkpoint_dict = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-0',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': TimeUtil.now_in_millis(),
                        'eliminated_at_ms': None,
                        'account_size': 100_000,
                        'asset_class': 'crypto'
                    },
                    '1': {
                        'subaccount_id': 1,
                        'subaccount_uuid': 'uuid-1',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_1',
                        'status': 'active',
                        'created_at_ms': TimeUtil.now_in_millis(),
                        'eliminated_at_ms': None,
                        'account_size': 100_000,
                        'asset_class': 'crypto'
                    },
                    '2': {
                        'subaccount_id': 2,
                        'subaccount_uuid': 'uuid-2',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_2',
                        'status': 'active',
                        'created_at_ms': TimeUtil.now_in_millis(),
                        'eliminated_at_ms': None,
                        'account_size': 100_000,
                        'asset_class': 'crypto'
                    }
                },
                'next_subaccount_id': 3,
                'registered_at_ms': TimeUtil.now_in_millis()
            }
        }

        # Sync entity data
        stats = self.entity_client.sync_entity_data(checkpoint_dict)

        # Verify stats (entity exists, so 2 new subaccounts added)
        self.assertEqual(stats['entities_added'], 0)
        self.assertEqual(stats['subaccounts_added'], 2)

        # Verify all 3 subaccounts exist
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data['subaccounts']), 3)
        self.assertEqual(entity_data['next_subaccount_id'], 3)

    def test_sync_entity_data_status_update(self):
        """Test syncing subaccount status changes (active -> eliminated)."""
        # Register entity and create active subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")

        # Verify initially active
        found, status, _ = self.entity_client.get_subaccount_status(f'{self.ENTITY_HOTKEY_1}_0')
        self.assertTrue(found)
        self.assertEqual(status, 'active')

        # Create checkpoint dict with eliminated subaccount
        checkpoint_dict = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-0',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'eliminated',
                        'created_at_ms': TimeUtil.now_in_millis(),
                        'eliminated_at_ms': TimeUtil.now_in_millis(),
                        'account_size': 100_000,
                        'asset_class': 'crypto'
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': TimeUtil.now_in_millis()
            }
        }

        # Sync entity data
        stats = self.entity_client.sync_entity_data(checkpoint_dict)

        # Verify stats (1 subaccount updated)
        self.assertEqual(stats['subaccounts_updated'], 1)

        # Verify status changed to eliminated
        found, status, _ = self.entity_client.get_subaccount_status(f'{self.ENTITY_HOTKEY_1}_0')
        self.assertTrue(found)
        self.assertEqual(status, 'eliminated')

    def test_sync_entity_data_collision_prevention(self):
        """Test that next_subaccount_id is updated to prevent ID collisions."""
        # Register entity locally with next_subaccount_id = 1
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")

        # Get current next_subaccount_id (should be 1)
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(entity_data['next_subaccount_id'], 1)

        # Create checkpoint dict with higher next_subaccount_id (5)
        checkpoint_dict = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-0',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': TimeUtil.now_in_millis(),
                        'eliminated_at_ms': None,
                        'account_size': 100_000,
                        'asset_class': 'crypto'
                    }
                },
                'next_subaccount_id': 5,
                'registered_at_ms': TimeUtil.now_in_millis()
            }
        }

        # Sync entity data
        self.entity_client.sync_entity_data(checkpoint_dict)

        # Verify next_subaccount_id updated to prevent collisions
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(entity_data['next_subaccount_id'], 5)

    def test_sync_entity_data_invalid_input(self):
        """Test that sync handles invalid input gracefully."""
        # Test with None
        stats = self.entity_client.sync_entity_data(None)
        self.assertEqual(stats['entities_added'], 0)
        self.assertEqual(stats['subaccounts_added'], 0)

        # Test with empty dict
        stats = self.entity_client.sync_entity_data({})
        self.assertEqual(stats['entities_added'], 0)
        self.assertEqual(stats['subaccounts_added'], 0)

        # Test with non-dict type (should return empty stats)
        stats = self.entity_client.sync_entity_data("invalid_string")
        self.assertEqual(stats['entities_added'], 0)
        self.assertEqual(stats['subaccounts_added'], 0)

    def test_sync_entity_data_multiple_entities(self):
        """Test syncing multiple entities in one operation."""
        # Create checkpoint dict with 3 entities
        now_ms = TimeUtil.now_in_millis()
        checkpoint_dict = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-1-0',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 100_000,
                        'asset_class': 'crypto'
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            },
            self.ENTITY_HOTKEY_2: {
                'entity_hotkey': self.ENTITY_HOTKEY_2,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-2-0',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_2}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 100_000,
                        'asset_class': 'crypto'
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            },
            self.ENTITY_HOTKEY_3: {
                'entity_hotkey': self.ENTITY_HOTKEY_3,
                'subaccounts': {},
                'next_subaccount_id': 0,
                'registered_at_ms': now_ms
            }
        }

        # Sync all entities
        stats = self.entity_client.sync_entity_data(checkpoint_dict)

        # Verify stats
        self.assertEqual(stats['entities_added'], 3)
        self.assertEqual(stats['subaccounts_added'], 2)

        # Verify all entities exist
        all_entities = self.entity_client.get_all_entities()
        self.assertEqual(len(all_entities), 3)
        self.assertIn(self.ENTITY_HOTKEY_1, all_entities)
        self.assertIn(self.ENTITY_HOTKEY_2, all_entities)
        self.assertIn(self.ENTITY_HOTKEY_3, all_entities)


class TestSubaccountPayoutWeeklyPenalty(TestBase):
    """A blocked payout week defers the subaccount's USDC payout for that week only, and the
    escrow is either released on the next clean pro week or forfeited when the account leaves
    the pro track."""

    ENTITY_HOTKEY = "entity"
    SUBACCOUNT_HOTKEY = "entity_1"
    SUBACCOUNT_UUID = "uuid-1"
    CP_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    def _payout_result(self, blocked_checkpoint_indices=(), week_buckets=None,
                       current_bucket=MinerBucket.PRO_FUNDED, payout_scale=1.0,
                       sealed_weeks=None, settled_segments=(), orders=None,
                       checkpoint_buckets=None, has_perf_ledger=True):
        """Two payout weeks of 12h checkpoints. `week_buckets` maps a payout-week index (0 or 1) to
        the bucket stamped on that week's checkpoints (default PRO_FUNDED); `checkpoint_buckets`
        overrides individual checkpoints, which is how a bucket change lands mid-week;
        `current_bucket` is the bucket at end_time_ms; `payout_scale` is standard_account_size /
        pro_account_size; `settled_segments` are stretches settled early by an account switch;
        `orders` overrides the default one-order-per-cell history (pass [] for an account that has
        not traded since); `has_perf_ledger=False` is the state a switch leaves behind, where the
        perf ledger bundle is dropped along with the positions."""
        from entity_management.entity_manager import EntityManager

        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        end_time_ms = week_0_start + 2 * MS_IN_WEEK
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        week_buckets = week_buckets or {}
        checkpoint_buckets = checkpoint_buckets or {}

        # One order realizing 10 USD per 12h cell across two weeks
        orders = orders if orders is not None else [
            SimpleNamespace(
                processed_ms=week_0_start + i * self.CP_DURATION_MS + 1,
                realized_pnl=10.0,
                to_python_dict=lambda: {},
            )
            for i in range(2 * MS_IN_WEEK // self.CP_DURATION_MS)
        ]
        debt_checkpoints = [
            DebtCheckpoint(
                timestamp_ms=week_0_start + (i + 1) * self.CP_DURATION_MS,
                weekly_penalty=0.0 if i in blocked_checkpoint_indices else 1.0,
                challenge_period_status=checkpoint_buckets.get(
                    i, week_buckets.get(i // cps_per_week, MinerBucket.PRO_FUNDED)
                ).value,
            )
            for i in range(2 * MS_IN_WEEK // self.CP_DURATION_MS)
        ]

        manager = object.__new__(EntityManager)
        manager.running_unit_tests = True
        manager.get_synthetic_hotkey_from_uuid = lambda _uuid: self.SUBACCOUNT_HOTKEY
        manager.get_entity_data = lambda _hk: SimpleNamespace(subaccounts={1: {'id': 1}})
        manager.get_payout_scale = lambda _hk: payout_scale
        manager._debt_ledger_client = SimpleNamespace(
            get_ledger=lambda _hk: DebtLedger(self.SUBACCOUNT_HOTKEY, checkpoints=debt_checkpoints),
            # Nothing sealed by default: every week is recomputed, which is what most cases exercise
            get_sealed_weeks=lambda _hk: sealed_weeks or {},
            # Nothing settled early by default: no account switch has wound this subaccount down
            get_settled_segments=lambda _hk: list(settled_segments),
        )
        manager._perf_ledger_client = SimpleNamespace(
            get_perf_ledger_for_hotkey=lambda hk: {
                hk: SimpleNamespace(get_checkpoint_at_time=lambda *_a: None)
            } if has_perf_ledger else {},
            get_frozen_ledgers=lambda: {},
        )
        manager._challenge_period_client = SimpleNamespace(
            get_miner_bucket=lambda *_a: current_bucket
        )
        manager._position_client = SimpleNamespace(
            get_positions_for_one_hotkey=lambda *_a, **_k: [
                SimpleNamespace(orders=orders, fee_history=[], unrealized_pnl=0.0)
            ]
        )

        return manager.calculate_subaccount_payout(self.SUBACCOUNT_UUID, week_0_start, end_time_ms)

    def _payouts_by_week(self, blocked_checkpoint_indices=()):
        result = self._payout_result(blocked_checkpoint_indices)
        return [w['payout'] for w in result['weekly_settlements']], result['payout']

    def test_unblocked_weeks_pay_out(self):
        per_week, total = self._payouts_by_week()
        self.assertEqual(per_week, [140.0, 140.0])
        self.assertAlmostEqual(total, 280.0)

    def test_single_breach_defers_that_week_until_the_next_clean_week(self):
        # Breach stamped on one mid-week checkpoint withholds all of week 0; the clean week 1
        # pays its own 140 plus the released 140
        per_week, total = self._payouts_by_week(blocked_checkpoint_indices=(8,))
        self.assertEqual(per_week, [0.0, 280.0])
        self.assertAlmostEqual(total, 280.0)

    def test_clean_pro_week_releases_escrow_and_forfeits_nothing(self):
        # Week 0 withheld; week 1 clean and still PRO_FUNDED: the escrow settles, nothing is forfeited
        result = self._payout_result(blocked_checkpoint_indices=(8,))
        week_0, week_1 = result['weekly_settlements']
        self.assertAlmostEqual(week_0['deferred'], 140.0)
        self.assertAlmostEqual(week_0['deferred_balance'], 140.0)
        self.assertEqual(week_0['deferred_forfeited'], 0.0)
        self.assertAlmostEqual(week_1['deferred_released'], 140.0)
        self.assertEqual(week_1['deferred_forfeited'], 0.0)
        self.assertEqual(week_1['deferred_balance'], 0.0)
        self.assertAlmostEqual(week_1['payout'], 280.0)
        self.assertEqual(result['deferred_forfeited'], 0.0)
        self.assertEqual(result['deferred_balance'], 0.0)
        self.assertFalse(result['off_track'])

    def test_leaving_pro_track_with_balance_forfeits_it(self):
        # Week 0 withheld in PRO_FUNDED; week 1 the account is off the track: the escrow is dropped
        # in that week and reported as forfeited, while the week's own earnings still pay
        result = self._payout_result(
            blocked_checkpoint_indices=(8,),
            week_buckets={1: MinerBucket.SUBACCOUNT_FUNDED},
            current_bucket=MinerBucket.SUBACCOUNT_FUNDED,
        )
        week_0, week_1 = result['weekly_settlements']
        self.assertAlmostEqual(week_0['deferred_balance'], 140.0)
        self.assertEqual(week_0['deferred_forfeited'], 0.0)
        self.assertEqual(week_1['deferred_released'], 0.0)
        self.assertAlmostEqual(week_1['deferred_forfeited'], 140.0)
        self.assertEqual(week_1['deferred_balance'], 0.0)
        self.assertAlmostEqual(week_1['payout'], 140.0)
        self.assertAlmostEqual(result['deferred_forfeited'], 140.0)
        self.assertEqual(result['deferred_balance'], 0.0)
        self.assertAlmostEqual(result['payout'], 140.0)
        self.assertTrue(result['off_track'])

    def test_leaving_pro_track_with_zero_balance_forfeits_nothing(self):
        # Nothing was ever withheld, so leaving the track reports no phantom forfeiture
        result = self._payout_result(
            week_buckets={1: MinerBucket.SUBACCOUNT_FUNDED},
            current_bucket=MinerBucket.SUBACCOUNT_FUNDED,
        )
        for week in result['weekly_settlements']:
            self.assertEqual(week['deferred_forfeited'], 0.0)
            self.assertEqual(week['deferred_balance'], 0.0)
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [140.0, 140.0])
        self.assertEqual(result['deferred_forfeited'], 0.0)
        self.assertEqual(result['deferred_balance'], 0.0)
        self.assertTrue(result['off_track'])

    def test_non_earning_bucket_returns_empty_settlement_with_deferral_fields(self):
        result = self._payout_result(current_bucket=MinerBucket.SUBACCOUNT_CHALLENGE)
        self.assertEqual(result['weekly_settlements'], [])
        self.assertEqual(result['deferred_balance'], 0.0)
        self.assertEqual(result['deferred_forfeited'], 0.0)
        self.assertTrue(result['off_track'])

    def test_pro_challenge_direct_gains_are_never_paid_after_promotion(self):
        """A pro challenge run directly on the pro account earns nothing.

        Passing it keeps the account - balance, positions and ledgers all carry over - so the
        payout basis has to rebase at the promotion instead. Week 0 is PRO_CHALLENGE_DIRECT and
        realizes 140 USD; only week 1's 140 is payable.
        """
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_DIRECT, 1: MinerBucket.PRO_FUNDED}
        )
        per_week = [w['payout'] for w in result['weekly_settlements']]
        self.assertEqual(per_week, [0.0, 140.0])
        self.assertAlmostEqual(result['payout'], 140.0)

    def test_pro_challenge_from_standard_is_paid_on_the_standard_account_size(self):
        """PRO_CHALLENGE_FROM_STANDARD trades the pro account but is paid on the standard one.

        With a 100k standard account inside a 500k pro account the scale is 0.2, so week 0's
        140 USD of gross PnL pays 28. On promotion the scale goes to 1.0 and the gross high water
        mark carries, so week 1's next 140 of gross gain pays 140 - the trader is never paid twice
        for the same dollars, and never at pro scale for pre-promotion gains.
        """
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_FROM_STANDARD, 1: MinerBucket.PRO_FUNDED},
            payout_scale=0.2,
        )
        per_week = [w['payout'] for w in result['weekly_settlements']]
        self.assertEqual([w['payout_scale'] for w in result['weekly_settlements']], [0.2, 1.0])
        self.assertEqual(per_week, [28.0, 140.0])
        self.assertAlmostEqual(result['payout'], 168.0)

    def test_a_breach_after_a_mid_week_promotion_still_pays_the_pre_promotion_stretch(self):
        """A soft breach is a pro rule, so it cannot reach back past the promotion that imposed it.

        The account promotes halfway through week 0 and breaches after. The week's penalty is
        widened across the whole week, but the gate is the segment's own bucket: the
        PRO_CHALLENGE_FROM_STANDARD half is paid on Monday at its standard scale (70 gross at 0.2),
        and only the PRO_FUNDED half is withheld. Week 1 is clean, so the 70 it deferred settles
        on top of its own 140.
        """
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        promoted_at = cps_per_week // 2
        result = self._payout_result(
            checkpoint_buckets={
                i: (MinerBucket.PRO_CHALLENGE_FROM_STANDARD if i < promoted_at
                    else MinerBucket.PRO_FUNDED)
                for i in range(2 * cps_per_week)
            },
            # Breached after the promotion, so the pro rules did govern the account at the time
            blocked_checkpoint_indices=(promoted_at + 3,),
            payout_scale=0.2,
        )
        rows = result['weekly_settlements']
        self.assertEqual([r['bucket'] for r in rows], [
            MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value,
            MinerBucket.PRO_FUNDED.value,
            MinerBucket.PRO_FUNDED.value,
        ])
        self.assertEqual([r['weekly_penalty'] for r in rows], [1.0, 0.0, 1.0])
        self.assertEqual([r['payout_scale'] for r in rows], [0.2, 1.0, 1.0])
        self.assertEqual([r['payout'] for r in rows], [14.0, 0.0, 210.0])
        # Only the post-promotion half was ever withheld
        self.assertEqual([r['deferred'] for r in rows], [0.0, 70.0, 0.0])
        self.assertEqual(rows[2]['deferred_released'], 70.0)
        self.assertAlmostEqual(result['payout'], 224.0)
        self.assertEqual(result['deferred_forfeited'], 0.0)

    def test_a_breach_before_the_promotion_cannot_withhold_anything(self):
        """The breach latch never fires outside PRO_FUNDED, but the week-level min would carry a
        stale one anyway. Gating on the segment's bucket means a penalty stamped on the challenge
        half withholds nothing - there and in the funded half, which the pro rules did not govern
        at the time it was stamped."""
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        promoted_at = cps_per_week // 2
        result = self._payout_result(
            checkpoint_buckets={
                i: (MinerBucket.PRO_CHALLENGE_FROM_STANDARD if i < promoted_at
                    else MinerBucket.PRO_FUNDED)
                for i in range(2 * cps_per_week)
            },
            blocked_checkpoint_indices=(promoted_at - 3,),
            payout_scale=0.2,
        )
        rows = result['weekly_settlements']
        # The challenge half is ungated; the funded half still answers to the week's penalty
        self.assertEqual([r['weekly_penalty'] for r in rows], [1.0, 0.0, 1.0])
        self.assertEqual([r['payout'] for r in rows], [14.0, 0.0, 210.0])

    def test_standard_subaccount_payouts_are_unchanged(self):
        """A standard funded subaccount is untouched by any of the pro machinery."""
        result = self._payout_result(
            week_buckets={0: MinerBucket.SUBACCOUNT_FUNDED, 1: MinerBucket.SUBACCOUNT_FUNDED},
            current_bucket=MinerBucket.SUBACCOUNT_FUNDED,
        )
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [140.0, 140.0])
        self.assertEqual([w['payout_scale'] for w in result['weekly_settlements']], [1.0, 1.0])
        self.assertEqual([w['weekly_penalty'] for w in result['weekly_settlements']], [1.0, 1.0])
        self.assertAlmostEqual(result['payout'], 280.0)
        self.assertEqual(result['deferred_balance'], 0.0)
        self.assertEqual(result['deferred_forfeited'], 0.0)

    def test_sealed_week_survives_a_rebuild_that_would_have_changed_it(self):
        """A settled week replays as settled even when the ledgers now say otherwise.

        Week 0 is breached in the ledger, but it was sealed clean. The sealed record wins, so the
        rebuild cannot retroactively withhold a week that was already paid.
        """
        from vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger import SealedWeek

        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        sealed = {week_0_start: SealedWeek(
            week_start_ms=week_0_start,
            weekly_penalty=1.0,
            payout_scale=1.0,
            track='ON_TRACK',
            first_earning_ms=None,
            sealed_ms=0,
        )}
        result = self._payout_result(blocked_checkpoint_indices=(0,), sealed_weeks=sealed)
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [140.0, 140.0])
        self.assertEqual(result['deferred_balance'], 0.0)

    def test_a_sealed_week_keeps_its_payout_scale_after_the_account_is_resized(self):
        """Account sizes are read live, so only the seal stops a resize repricing a paid week.

        Week 0 settled at a 0.2 ratio. The pro account has since been resized to put the live
        ratio at 0.5, but the sealed week is still paid at 0.2.
        """
        from vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger import SealedWeek

        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        sealed = {week_0_start: SealedWeek(
            week_start_ms=week_0_start,
            weekly_penalty=1.0,
            payout_scale=0.2,
            track='OFF_TRACK',
            first_earning_ms=None,
            sealed_ms=0,
        )}
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_FROM_STANDARD, 1: MinerBucket.PRO_FUNDED},
            payout_scale=0.5,
            sealed_weeks=sealed,
        )
        self.assertEqual([w['payout_scale'] for w in result['weekly_settlements']], [0.2, 1.0])
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [28.0, 140.0])

    def test_a_settled_segment_pays_when_the_promoted_account_has_not_traded(self):
        """The promotion archives every position, so the payout path has nothing left to compute
        from. The dollars settled at the switch are the only thing that pays that stretch."""
        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        segment = SettledSegment(
            week_start_ms=week_0_start,
            segment_start_ms=week_0_start,
            segment_end_ms=week_0_start + MS_IN_WEEK // 2,
            bucket=MinerBucket.PRO_CHALLENGE_TRANSITION.value,
            payout_usd=1843.20,
            gross_payout_usd=1843.20,
            weekly_penalty=1.0,
            payout_scale=1.0,
            recorded_ms=0,
        )
        result = self._payout_result(
            current_bucket=MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
            settled_segments=[segment],
            orders=[],
        )
        self.assertEqual(result['payout'], 1843.20)
        rows = result['weekly_settlements']
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['settled_segment'])
        self.assertEqual(rows[0]['bucket'], MinerBucket.PRO_CHALLENGE_TRANSITION.value)

    def test_a_settled_segment_pays_when_the_promotion_left_no_perf_ledger(self):
        """The switch archives the positions, and the perf ledger goes with them. The settled
        money still has to be reported rather than reading as a subaccount that does not exist."""
        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        segment = SettledSegment(
            week_start_ms=week_0_start,
            segment_start_ms=week_0_start,
            segment_end_ms=week_0_start + MS_IN_WEEK // 2,
            bucket=MinerBucket.PRO_CHALLENGE_TRANSITION.value,
            payout_usd=1843.20,
            gross_payout_usd=1843.20,
            weekly_penalty=1.0,
            payout_scale=1.0,
            recorded_ms=0,
        )
        result = self._payout_result(
            current_bucket=MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
            settled_segments=[segment],
            orders=[],
            has_perf_ledger=False,
        )
        self.assertEqual(result['payout'], 1843.20)
        self.assertEqual(len(result['weekly_settlements']), 1)
        self.assertTrue(result['weekly_settlements'][0]['settled_segment'])

    def test_no_perf_ledger_and_nothing_settled_is_still_not_found(self):
        """Nothing settled means there is nothing to report without a ledger."""
        self.assertIsNone(self._payout_result(has_perf_ledger=False))

    def test_a_settled_segment_is_added_to_the_weeks_computed_around_it(self):
        """The settled stretch and the weeks the account went on to trade are both paid, once."""
        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        segment = SettledSegment(
            week_start_ms=week_0_start,
            segment_start_ms=week_0_start,
            segment_end_ms=week_0_start + MS_IN_WEEK // 2,
            bucket=MinerBucket.PRO_CHALLENGE_TRANSITION.value,
            payout_usd=1843.20,
            gross_payout_usd=1843.20,
            weekly_penalty=1.0,
            payout_scale=1.0,
            recorded_ms=0,
        )
        baseline = self._payout_result()
        result = self._payout_result(settled_segments=[segment])
        self.assertEqual(result['payout'], baseline['payout'] + 1843.20)
        self.assertEqual(
            sum(1 for w in result['weekly_settlements'] if w['settled_segment']), 1
        )
        # The computed weeks pay exactly what they pay with nothing settled at all
        computed = [w for w in result['weekly_settlements'] if not w['settled_segment']]
        self.assertEqual([w['payout'] for w in computed],
                         [w['payout'] for w in baseline['weekly_settlements']])
        # ... and no two rows report overlapping windows
        rows = result['weekly_settlements']
        for earlier, later in zip(rows, rows[1:]):
            self.assertLessEqual(earlier['end_ms'], later['start_ms'])

    def test_the_wound_down_segment_is_settled_at_the_standard_size(self):
        """_switch_account resizes the account to pro *before* settling, so get_payout_scale is
        already the pro ratio by the time the segment is captured. Only the bucket gate keeps the
        pre-promotion money whole - a regression here is a silent underpayment, not a crash."""
        from entity_management.entity_manager import EntityManager

        now = TimeUtil.now_in_millis()
        monday = TimeUtil.ms_at_start_of_week(now)
        cells = max(1, (now - monday) // self.CP_DURATION_MS)
        orders = [SimpleNamespace(processed_ms=monday + i * self.CP_DURATION_MS + 1,
                                  realized_pnl=100.0, to_python_dict=lambda: {})
                  for i in range(cells)]
        checkpoints = [DebtCheckpoint(
            timestamp_ms=monday + (i + 1) * self.CP_DURATION_MS,
            challenge_period_status=MinerBucket.PRO_CHALLENGE_TRANSITION.value,
        ) for i in range(cells)]

        recorded = {}
        manager = object.__new__(EntityManager)
        manager.running_unit_tests = True
        manager.get_entity_data = lambda _hk: SimpleNamespace(subaccounts={1: {'id': 1}})
        # The live scale the account already carries after apply_bucket_account_size
        manager.get_payout_scale = lambda _hk: 2.0 * 100_000 / 500_000
        manager._debt_ledger_client = SimpleNamespace(
            get_ledger=lambda _hk: DebtLedger(self.SUBACCOUNT_HOTKEY, checkpoints=checkpoints),
            get_sealed_weeks=lambda _hk: {},
            get_settled_segments=lambda _hk: [],
            record_settled_segment=lambda *a: recorded.update(
                payout_usd=a[5], gross=a[6], scale=a[8], bucket=a[4]) or True,
        )
        manager._perf_ledger_client = SimpleNamespace(
            get_perf_ledger_for_hotkey=lambda hk: {
                hk: SimpleNamespace(get_checkpoint_at_time=lambda *_a: None)})
        manager._position_client = SimpleNamespace(
            get_positions_for_one_hotkey=lambda *_a, **_k: [
                SimpleNamespace(orders=orders, fee_history=[], unrealized_pnl=0.0)])

        self.assertTrue(manager.settle_wound_down_segment(
            self.SUBACCOUNT_HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION.value, now))
        self.assertEqual(recorded['payout_usd'], 100.0 * cells)   # NOT scaled by 0.4
        self.assertEqual(recorded['scale'], 1.0)
        self.assertEqual(recorded['bucket'], MinerBucket.PRO_CHALLENGE_TRANSITION.value)

    def test_no_settled_segment_leaves_the_weekly_windows_untouched(self):
        """The promotion handling must be inert for every account that never promoted."""
        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        result = self._payout_result()
        self.assertEqual(
            [(w['start_ms'], w['end_ms']) for w in result['weekly_settlements']],
            [(week_0_start, week_0_start + MS_IN_WEEK),
             (week_0_start + MS_IN_WEEK, week_0_start + 2 * MS_IN_WEEK)],
        )
        self.assertFalse(any(w['settled_segment'] for w in result['weekly_settlements']))

    def test_a_settled_segment_opens_its_week_at_the_switch(self):
        """The settled stretch and the week the account went on to trade must not overlap."""
        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        switch_ms = week_0_start + MS_IN_WEEK // 2
        segment = SettledSegment(
            week_start_ms=week_0_start,
            segment_start_ms=week_0_start,
            segment_end_ms=switch_ms,
            bucket=MinerBucket.PRO_CHALLENGE_TRANSITION.value,
            payout_usd=1843.20,
            gross_payout_usd=1843.20,
            weekly_penalty=1.0,
            payout_scale=1.0,
            recorded_ms=0,
        )
        result = self._payout_result(settled_segments=[segment])
        self.assertEqual(
            [(w['start_ms'], w['end_ms']) for w in result['weekly_settlements']],
            [(week_0_start, switch_ms),
             (switch_ms, week_0_start + MS_IN_WEEK),
             (week_0_start + MS_IN_WEEK, week_0_start + 2 * MS_IN_WEEK)],
        )
        # Only the promotion week's window moved; the week after it is untouched
        self.assertEqual(result['weekly_settlements'][2]['payout'],
                         self._payout_result()['weekly_settlements'][1]['payout'])


if __name__ == '__main__':
    unittest.main()
