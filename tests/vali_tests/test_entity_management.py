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

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.vali_utils import ValiUtils
from time_util.time_util import TimeUtil


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

        # Set up test entities
        self.ENTITY_HOTKEY_1 = "entity_hotkey_1"
        self.ENTITY_HOTKEY_2 = "entity_hotkey_2"
        self.ENTITY_HOTKEY_3 = "entity_hotkey_3"

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
            entity_hotkey=self.ENTITY_HOTKEY_1,
            collateral_amount=1000.0,
            max_subaccounts=5
        )

        self.assertTrue(success, f"Entity registration failed: {message}")

        # Verify entity exists
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertIsNotNone(entity_data)
        self.assertEqual(entity_data['entity_hotkey'], self.ENTITY_HOTKEY_1)
        self.assertEqual(entity_data['collateral_amount'], 1000.0)
        self.assertEqual(entity_data['max_subaccounts'], 5)
        self.assertEqual(len(entity_data['subaccounts']), 0)

    def test_register_entity_duplicate(self):
        """Test that registering the same entity twice fails."""
        # Register first time
        success, _ = self.entity_client.register_entity(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            collateral_amount=1000.0
        )
        self.assertTrue(success)

        # Try to register again
        success, message = self.entity_client.register_entity(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            collateral_amount=2000.0
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
        self.assertEqual(entity_data['collateral_amount'], 0.0)
        self.assertEqual(entity_data['max_subaccounts'], 10)  # ValiConfig default

    # ==================== Subaccount Creation Tests ====================

    def test_create_subaccount_success(self):
        """Test successful subaccount creation."""
        # Register entity first
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        # Create subaccount
        success, subaccount_info, message = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1
        )

        self.assertTrue(success, f"Subaccount creation failed: {message}")
        self.assertIsNotNone(subaccount_info)
        self.assertEqual(subaccount_info['subaccount_id'], 0)
        self.assertEqual(subaccount_info['status'], 'active')

        # Verify synthetic hotkey format
        synthetic_hotkey = subaccount_info['synthetic_hotkey']
        self.assertEqual(synthetic_hotkey, f"{self.ENTITY_HOTKEY_1}_0")

    def test_create_multiple_subaccounts(self):
        """Test creating multiple subaccounts for an entity."""
        # Register entity
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        # Create 3 subaccounts
        subaccount_ids = []
        for i in range(3):
            success, subaccount_info, _ = self.entity_client.create_subaccount(
                entity_hotkey=self.ENTITY_HOTKEY_1
            )
            self.assertTrue(success)
            subaccount_ids.append(subaccount_info['subaccount_id'])

        # Verify sequential IDs (0, 1, 2)
        self.assertEqual(subaccount_ids, [0, 1, 2])

        # Verify entity data
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data['subaccounts']), 3)

    def test_create_subaccount_max_limit(self):
        """Test that subaccount creation fails when max limit is reached."""
        # Register entity with max_subaccounts=2
        self.entity_client.register_entity(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            max_subaccounts=2
        )

        # Create 2 subaccounts (should succeed)
        for i in range(2):
            success, _, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1)
            self.assertTrue(success)

        # Try to create 3rd subaccount (should fail)
        success, subaccount_info, message = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1
        )
        self.assertFalse(success)
        self.assertIsNone(subaccount_info)
        self.assertIn("maximum", message.lower())

    def test_create_subaccount_unregistered_entity(self):
        """Test that subaccount creation fails for unregistered entity."""
        success, subaccount_info, message = self.entity_client.create_subaccount(
            entity_hotkey="unregistered_entity"
        )

        self.assertFalse(success)
        self.assertIsNone(subaccount_info)
        self.assertIn("not registered", message.lower())

    # ==================== Synthetic Hotkey Tests ====================

    def test_is_synthetic_hotkey_valid(self):
        """Test synthetic hotkey detection."""
        # Valid synthetic hotkeys
        self.assertTrue(self.entity_client.is_synthetic_hotkey("entity_123"))
        self.assertTrue(self.entity_client.is_synthetic_hotkey("my_entity_0"))
        self.assertTrue(self.entity_client.is_synthetic_hotkey("foo_bar_99"))

        # Invalid synthetic hotkeys (no underscore + integer)
        self.assertFalse(self.entity_client.is_synthetic_hotkey("regular_hotkey"))
        self.assertFalse(self.entity_client.is_synthetic_hotkey("no_number_"))
        self.assertFalse(self.entity_client.is_synthetic_hotkey("just_text"))

    def test_parse_synthetic_hotkey_valid(self):
        """Test parsing valid synthetic hotkeys."""
        entity_hotkey, subaccount_id = self.entity_client.parse_synthetic_hotkey(
            "my_entity_5"
        )
        self.assertEqual(entity_hotkey, "my_entity")
        self.assertEqual(subaccount_id, 5)

        # Test with entity hotkey containing underscores
        entity_hotkey, subaccount_id = self.entity_client.parse_synthetic_hotkey(
            "entity_with_underscores_123"
        )
        self.assertEqual(entity_hotkey, "entity_with_underscores")
        self.assertEqual(subaccount_id, 123)

    def test_parse_synthetic_hotkey_invalid(self):
        """Test parsing invalid synthetic hotkeys."""
        entity_hotkey, subaccount_id = self.entity_client.parse_synthetic_hotkey(
            "invalid_hotkey"
        )
        self.assertIsNone(entity_hotkey)
        self.assertIsNone(subaccount_id)

    # ==================== Subaccount Status Tests ====================

    def test_get_subaccount_status_active(self):
        """Test getting status of an active subaccount."""
        # Register entity and create subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1)
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
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1)
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
        self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1)

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
        self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1)

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
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1)
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Synthetic hotkey should be recognized (entity in metagraph + subaccount active)
        self.assertTrue(self.metagraph_client.has_hotkey(synthetic_hotkey))

    def test_metagraph_has_hotkey_synthetic_eliminated(self):
        """Test that eliminated synthetic hotkeys are NOT recognized by metagraph."""
        # Register entity and create subaccount
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1)
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
        _, subaccount_info, _ = self.entity_client.create_subaccount(unregistered_entity)
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

    def test_update_collateral(self):
        """Test updating collateral for an entity."""
        # Register entity with initial collateral
        self.entity_client.register_entity(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            collateral_amount=1000.0
        )

        # Update collateral
        success, message = self.entity_client.update_collateral(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            collateral_amount=2000.0
        )

        self.assertTrue(success, f"Collateral update failed: {message}")

        # Verify updated collateral
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(entity_data['collateral_amount'], 2000.0)

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
        is_synthetic = self.entity_client.is_synthetic_hotkey(self.ENTITY_HOTKEY_1)
        self.assertFalse(is_synthetic, "Entity hotkey should not be synthetic")

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
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1)
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Verify hotkey is synthetic
        is_synthetic = self.entity_client.is_synthetic_hotkey(synthetic_hotkey)
        self.assertTrue(is_synthetic, "Subaccount hotkey should be synthetic")

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
        _, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY_1)
        synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Eliminate the subaccount
        self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            subaccount_id=0,
            reason="test_elimination"
        )

        # Verify hotkey is synthetic
        is_synthetic = self.entity_client.is_synthetic_hotkey(synthetic_hotkey)
        self.assertTrue(is_synthetic, "Subaccount hotkey should be synthetic")

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
        is_synthetic = self.entity_client.is_synthetic_hotkey(regular_hotkey)
        self.assertFalse(is_synthetic, "Regular hotkey should not be synthetic")

        # Verify it's not an entity
        entity_data = self.entity_client.get_entity_data(regular_hotkey)
        self.assertIsNone(entity_data, "Regular hotkey should not be an entity")


if __name__ == '__main__':
    unittest.main()
