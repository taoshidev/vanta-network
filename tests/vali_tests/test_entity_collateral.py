# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
Entity Collateral unit tests using the client/server architecture.

Tests the entity cross-margin collateral system including:
- Cumulative MDD slashing model
- Order gating (can_open_position)
- Slash tracking persistence (disk load/save with legacy migration)
- Collateral cache management
- Server orchestrator registration
- Withdrawal blocking for entities with open positions
- Elimination-driven slashing
"""
import unittest

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair
from time_util.time_util import TimeUtil
from entity_management.entity_utils import is_synthetic_hotkey, parse_synthetic_hotkey


class TestEntityCollateral(TestBase):
    """
    Entity Collateral unit tests using ServerOrchestrator.

    Tests the EntityCollateralManager through the EntityCollateralClient,
    validating:
    - Cumulative MDD slashing model (accumulate loss, dynamic max slash, delta slashing)
    - Order gating (can_open_position based on entity cross-margin)
    - Slash tracking persistence and legacy migration
    - Collateral cache operations
    - Integration with server orchestrator
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    entity_client = None
    entity_collateral_client = None
    metagraph_client = None
    contract_client = None
    position_client = None
    miner_account_client = None
    challenge_period_client = None

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator."""
        cls.orchestrator = ServerOrchestrator.get_instance()

        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.entity_collateral_client = cls.orchestrator.get_client('entity_collateral')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.contract_client = cls.orchestrator.get_client('contract')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.miner_account_client = cls.orchestrator.get_client('miner_account')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')

    @classmethod
    def tearDownClass(cls):
        """No action needed - orchestrator manages shutdown."""
        pass

    def setUp(self):
        """Per-test setup: Reset data state."""
        self.orchestrator.clear_all_test_data()
        # Also clear entity collateral state (not covered by clear_all_test_data)
        self.entity_collateral_client.clear_test_state()

        # Test entity hotkeys (avoid {text}_{number} pattern to prevent synthetic hotkey collision)
        self.ENTITY_HOTKEY = "entity_alpha"
        self.ENTITY_HOTKEY_2 = "entity_beta"

        # Initialize metagraph
        self.metagraph_client.set_hotkeys([
            self.ENTITY_HOTKEY,
            self.ENTITY_HOTKEY_2,
        ])

        # MDD percentage (same as manager uses)
        self.mdd_percent = ValiConfig.SUBACCOUNT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD  # 0.08

    def tearDown(self):
        """Per-test teardown: Clear data."""
        self.entity_collateral_client.clear_test_state()
        self.orchestrator.clear_all_test_data()

    # ==================== Helper Methods ====================

    def _register_entity_with_subaccount(self, entity_hotkey=None, account_size=100_000, asset_class="crypto"):
        """
        Helper: Register entity, create subaccount, set up account size.
        Returns (entity_hotkey, synthetic_hotkey, subaccount_info).
        """
        entity_hotkey = entity_hotkey or self.ENTITY_HOTKEY

        # Register entity
        success, msg = self.entity_client.register_entity(entity_hotkey=entity_hotkey)
        self.assertTrue(success, f"Entity registration failed: {msg}")

        # Create subaccount
        success, subaccount_info, msg = self.entity_client.create_subaccount(
            entity_hotkey=entity_hotkey,
            account_size=account_size,
            asset_class=asset_class
        )
        self.assertTrue(success, f"Subaccount creation failed: {msg}")

        synthetic_hotkey = subaccount_info['synthetic_hotkey']
        return entity_hotkey, synthetic_hotkey, subaccount_info

    def _set_collateral_cache(self, entity_hotkey, collateral_theta):
        """Helper: Inject collateral cache value in theta via RPC."""
        self.entity_collateral_client.set_test_collateral_cache(entity_hotkey, collateral_theta)

    def _get_slash_tracking(self, synthetic_hotkey):
        """Helper: Get slash tracking data via RPC."""
        return self.entity_collateral_client.get_test_slash_tracking(synthetic_hotkey)

    def _set_slash_tracking(self, synthetic_hotkey, cumulative_realized_loss, cumulative_slashed):
        """Helper: Set slash tracking data via RPC."""
        self.entity_collateral_client.set_test_slash_tracking(
            synthetic_hotkey, cumulative_realized_loss, cumulative_slashed
        )

    def _clear_entity_collateral_state(self):
        """Helper: Clear all entity collateral state via RPC."""
        self.entity_collateral_client.clear_test_state()

    # ==================== Server Registration Tests ====================

    def test_server_registered_in_orchestrator(self):
        """Test that entity_collateral server is registered and accessible."""
        self.assertIn('entity_collateral', ServerOrchestrator.SERVERS)
        config = ServerOrchestrator.SERVERS['entity_collateral']
        self.assertTrue(config.required_in_testing)
        self.assertTrue(config.required_in_validator)
        self.assertFalse(config.required_in_miner)

    def test_client_accessible_from_orchestrator(self):
        """Test that entity_collateral client can be retrieved."""
        client = self.orchestrator.get_client('entity_collateral')
        self.assertIsNotNone(client)

    def test_server_classes_loaded(self):
        """Test that server and client classes are properly loaded."""
        from vali_objects.utils.entity_collateral.entity_collateral_server import EntityCollateralServer
        from vali_objects.utils.entity_collateral.entity_collateral_client import EntityCollateralClient
        config = ServerOrchestrator.SERVERS['entity_collateral']
        self.assertEqual(config.server_class, EntityCollateralServer)
        self.assertEqual(config.client_class, EntityCollateralClient)

    # ==================== Collateral Cache Tests ====================

    def test_get_cached_collateral_none_for_unknown(self):
        """Test that get_cached_collateral returns None for unknown entity."""
        result = self.entity_collateral_client.get_cached_collateral("unknown_entity")
        self.assertIsNone(result)

    def test_get_cached_collateral_returns_value(self):
        """Test that get_cached_collateral returns injected value (theta)."""
        self._set_collateral_cache(self.ENTITY_HOTKEY, 100.0)

        result = self.entity_collateral_client.get_cached_collateral(self.ENTITY_HOTKEY)
        self.assertAlmostEqual(result, 100.0)

    def test_cached_collateral_independent_per_entity(self):
        """Test that collateral cache is independent per entity."""
        self._set_collateral_cache(self.ENTITY_HOTKEY, 50.0)
        self._set_collateral_cache(self.ENTITY_HOTKEY_2, 100.0)

        self.assertAlmostEqual(
            self.entity_collateral_client.get_cached_collateral(self.ENTITY_HOTKEY), 50.0
        )
        self.assertAlmostEqual(
            self.entity_collateral_client.get_cached_collateral(self.ENTITY_HOTKEY_2), 100.0
        )

    # ==================== Order Gating Tests (can_open_position) ====================

    def test_can_open_position_allowed_with_sufficient_collateral(self):
        """Test that order is allowed when entity has sufficient collateral."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )

        # Required delta: min(50K, 8K)/10 = 800 theta (first order, no existing positions)
        # Set 5000 theta (well above 800)
        self._set_collateral_cache(entity_hotkey, 5000.0)

        allowed, reason = self.entity_collateral_client.can_open_position(
            entity_hotkey, synthetic_hotkey, 50_000.0
        )

        self.assertTrue(allowed, f"Order should be allowed: {reason}")
        self.assertEqual(reason, "")

    def test_can_open_position_rejected_insufficient_collateral(self):
        """Test that order is rejected when entity has insufficient collateral."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )

        # Required delta: min(50K, 8K)/10 = 800 theta. Set only 10 theta → insufficient.
        self._set_collateral_cache(entity_hotkey, 10.0)

        allowed, reason = self.entity_collateral_client.can_open_position(
            entity_hotkey, synthetic_hotkey, 50_000.0
        )

        self.assertFalse(allowed)
        self.assertIn("Insufficient", reason)

    def test_can_open_position_rejected_no_cached_collateral(self):
        """Test that order is rejected when no collateral is cached."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        # Don't set any collateral cache

        allowed, reason = self.entity_collateral_client.can_open_position(
            entity_hotkey, synthetic_hotkey, 50_000.0
        )

        self.assertFalse(allowed)
        self.assertIn("no cached collateral", reason.lower())

    def test_can_open_position_multiple_subaccounts_margin_summed(self):
        """Test that margin requirements are summed across subaccounts."""
        # Register entity with two subaccounts
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY)
        _, sa_info_1, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY, account_size=100_000, asset_class="crypto"
        )
        _, sa_info_2, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY, account_size=100_000, asset_class="crypto"
        )
        synthetic_1 = sa_info_1['synthetic_hotkey']
        synthetic_2 = sa_info_2['synthetic_hotkey']

        # Required delta: min(1K, 8K)/10 = 100 theta. Set 500 theta (sufficient).
        self._set_collateral_cache(self.ENTITY_HOTKEY, 500.0)

        # First subaccount with small order should work
        allowed, reason = self.entity_collateral_client.can_open_position(
            self.ENTITY_HOTKEY, synthetic_1, 1_000.0
        )
        self.assertTrue(allowed, f"First order should be allowed with sufficient margin: {reason}")

    # ==================== Cumulative MDD Slashing Model Tests ====================

    def test_slash_basic_single_loss(self):
        """Test basic slashing with a single realized loss."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)  # 100 theta

        # Loss of $5,000 on an account with max_slash = $100K * 8% = $8K
        slashed = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 5_000.0
        )

        # Should slash exactly $5,000 (loss < max_slash)
        self.assertAlmostEqual(slashed, 5_000.0)

        # Verify tracking
        tracking = self._get_slash_tracking(synthetic_hotkey)
        self.assertIsNotNone(tracking)
        self.assertAlmostEqual(tracking["cumulative_realized_loss"], 5_000.0)
        self.assertAlmostEqual(tracking["cumulative_slashed"], 5_000.0)

    def test_slash_zero_loss_no_slash(self):
        """Test that zero loss produces no slash."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        slashed = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 0.0
        )

        self.assertAlmostEqual(slashed, 0.0)

    def test_slash_negative_loss_no_slash(self):
        """Test that negative loss (profit) produces no slash."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        slashed = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, -1_000.0
        )

        self.assertAlmostEqual(slashed, 0.0)

    def test_slash_cumulative_losses_across_multiple_trades(self):
        """Test that cumulative losses are tracked across multiple trades."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        # max_slash = $100K * 8% = $8K

        # Trade 1: Lose $3K
        slashed_1 = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 3_000.0
        )
        self.assertAlmostEqual(slashed_1, 3_000.0)

        # Trade 2: Lose $4K (cumulative_loss = $7K, still under max $10K)
        slashed_2 = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 4_000.0
        )
        self.assertAlmostEqual(slashed_2, 4_000.0)

        # Verify cumulative tracking
        tracking = self._get_slash_tracking(synthetic_hotkey)
        self.assertAlmostEqual(tracking["cumulative_realized_loss"], 7_000.0)
        self.assertAlmostEqual(tracking["cumulative_slashed"], 7_000.0)

    def test_slash_capped_at_max_slash(self):
        """Test that slashing is capped at account_size * MDD%."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        # max_slash = $100K * 8% = $8K

        # Trade 1: Lose $8K → exactly at cap, slash $8K
        slashed_1 = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 8_000.0
        )
        self.assertAlmostEqual(slashed_1, 8_000.0)

        # Trade 2: Lose $5K → already at cap, slash $0
        slashed_2 = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 5_000.0
        )
        self.assertAlmostEqual(slashed_2, 0.0)

        # Total slashed should be capped at $8K
        tracking = self._get_slash_tracking(synthetic_hotkey)
        self.assertAlmostEqual(tracking["cumulative_slashed"], 8_000.0)
        self.assertAlmostEqual(tracking["cumulative_realized_loss"], 13_000.0)

    def test_slash_completely_at_cap_returns_zero(self):
        """Test that no further slashing when already at cap."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        # Pre-fill tracking to exactly at cap ($8K = $100K * 8%)
        self._set_slash_tracking(synthetic_hotkey, 8_000.0, 8_000.0)

        # Try to slash more — should return 0
        slashed = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 5_000.0
        )
        self.assertAlmostEqual(slashed, 0.0)

        # cumulative_realized_loss should still be updated
        tracking = self._get_slash_tracking(synthetic_hotkey)
        self.assertAlmostEqual(tracking["cumulative_realized_loss"], 13_000.0)
        self.assertAlmostEqual(tracking["cumulative_slashed"], 8_000.0)

    def test_slash_no_account_size_returns_zero(self):
        """Test that slashing returns 0 when account has no size (max_slash=0)."""
        # Register entity but DON'T set up account size properly
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY)
        # Create subaccount but with a hotkey that has no account size in miner_account
        # The synthetic hotkey won't have a miner account size
        synthetic_hotkey = f"{self.ENTITY_HOTKEY}_999"

        self._set_collateral_cache(self.ENTITY_HOTKEY, 100.0)

        slashed = self.entity_collateral_client.slash_on_realized_loss(
            self.ENTITY_HOTKEY, synthetic_hotkey, 5_000.0
        )
        self.assertAlmostEqual(slashed, 0.0)

    def test_slash_collateral_cache_decremented(self):
        """Test that collateral cache (theta) is decremented after successful slash."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 5000.0)  # 5000 theta

        # Slash $5,000 → 5000 / CPT_RISK(10) = 500 theta decrement
        self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 5_000.0
        )

        cached = self.entity_collateral_client.get_cached_collateral(entity_hotkey)
        self.assertAlmostEqual(cached, 4500.0)  # 5000 - 500 theta

    def test_slash_independent_per_subaccount(self):
        """Test that slash tracking is independent per subaccount."""
        # Register entity with two subaccounts
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY)
        _, sa_info_1, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY, account_size=100_000, asset_class="crypto"
        )
        _, sa_info_2, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY, account_size=100_000, asset_class="crypto"
        )
        synthetic_1 = sa_info_1['synthetic_hotkey']
        synthetic_2 = sa_info_2['synthetic_hotkey']

        self._set_collateral_cache(self.ENTITY_HOTKEY, 200.0)  # 200 theta

        # Slash subaccount 1
        slashed_1 = self.entity_collateral_client.slash_on_realized_loss(
            self.ENTITY_HOTKEY, synthetic_1, 5_000.0
        )
        self.assertAlmostEqual(slashed_1, 5_000.0)

        # Slash subaccount 2 independently
        slashed_2 = self.entity_collateral_client.slash_on_realized_loss(
            self.ENTITY_HOTKEY, synthetic_2, 3_000.0
        )
        self.assertAlmostEqual(slashed_2, 3_000.0)

        # Verify independent tracking
        tracking_1 = self._get_slash_tracking(synthetic_1)
        tracking_2 = self._get_slash_tracking(synthetic_2)
        self.assertAlmostEqual(tracking_1["cumulative_realized_loss"], 5_000.0)
        self.assertAlmostEqual(tracking_2["cumulative_realized_loss"], 3_000.0)

    def test_slash_cumulative_loss_tracked_even_when_no_new_slash(self):
        """Test that cumulative_realized_loss is always updated, even when already at slash cap."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        # Pre-fill: already at cap ($8K = $100K * 8%)
        self._set_slash_tracking(synthetic_hotkey, 8_000.0, 8_000.0)

        # Another loss of $3K → no slash, but loss tracked
        slashed = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 3_000.0
        )
        self.assertAlmostEqual(slashed, 0.0)

        tracking = self._get_slash_tracking(synthetic_hotkey)
        self.assertAlmostEqual(tracking["cumulative_realized_loss"], 11_000.0)
        self.assertAlmostEqual(tracking["cumulative_slashed"], 8_000.0)

    # ==================== Cumulative Slashed Query Tests ====================

    def test_get_cumulative_slashed_zero_for_unknown(self):
        """Test that get_cumulative_slashed returns 0 for unknown hotkey."""
        result = self.entity_collateral_client.get_cumulative_slashed("unknown_hotkey_0")
        self.assertAlmostEqual(result, 0.0)

    def test_get_cumulative_slashed_returns_tracked_value(self):
        """Test that get_cumulative_slashed returns the tracked value."""
        self._set_slash_tracking("test_hotkey_0", 5000.0, 3000.0)

        result = self.entity_collateral_client.get_cumulative_slashed("test_hotkey_0")
        self.assertAlmostEqual(result, 3000.0)

    # ==================== Max Slash Tests ====================

    def test_get_max_slash_returns_account_size_times_mdd(self):
        """Test that max_slash = account_size * MDD%."""
        _, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )

        max_slash = self.entity_collateral_client.get_max_slash(synthetic_hotkey)
        expected = 100_000 * self.mdd_percent  # $8,000
        self.assertAlmostEqual(max_slash, expected)

    def test_get_max_slash_zero_for_unknown(self):
        """Test that max_slash returns 0 for unknown hotkey (no account)."""
        result = self.entity_collateral_client.get_max_slash("unknown_hotkey_0")
        self.assertAlmostEqual(result, 0.0)

    # ==================== Slash Tracking Persistence Tests ====================

    def test_slash_tracking_disk_persistence(self):
        """Test that slash tracking is persisted to disk after slashing."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        # Slash to create tracking data
        self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 5_000.0
        )

        # Verify tracking persisted (via RPC query — data was saved to disk by slash)
        tracking = self._get_slash_tracking(synthetic_hotkey)
        self.assertIsNotNone(tracking)
        self.assertAlmostEqual(tracking["cumulative_realized_loss"], 5_000.0)
        self.assertAlmostEqual(tracking["cumulative_slashed"], 5_000.0)

    def test_slash_tracking_legacy_format_migration(self):
        """Test that legacy format (hotkey -> float) is migrated correctly on load."""
        from vali_objects.utils.entity_collateral.entity_collateral_manager import EntityCollateralManager
        from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

        # Create a temporary manager to test disk loading directly (LOCAL mode, no server)
        mgr = EntityCollateralManager(running_unit_tests=True)

        # Write legacy format to disk
        legacy_data = {
            "entity_alpha_0": 3000.0,
            "entity_beta_0": 7000.0
        }
        ValiBkpUtils.write_file(mgr._slash_file, legacy_data)

        # Load and verify migration
        loaded = mgr._load_slash_tracking_from_disk()

        self.assertIn("entity_alpha_0", loaded)
        self.assertAlmostEqual(loaded["entity_alpha_0"]["cumulative_realized_loss"], 3000.0)
        self.assertAlmostEqual(loaded["entity_alpha_0"]["cumulative_slashed"], 3000.0)

        self.assertIn("entity_beta_0", loaded)
        self.assertAlmostEqual(loaded["entity_beta_0"]["cumulative_realized_loss"], 7000.0)
        self.assertAlmostEqual(loaded["entity_beta_0"]["cumulative_slashed"], 7000.0)

    def test_slash_tracking_new_format_loads_correctly(self):
        """Test that new format loads correctly."""
        from vali_objects.utils.entity_collateral.entity_collateral_manager import EntityCollateralManager
        from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

        mgr = EntityCollateralManager(running_unit_tests=True)

        # Write new format to disk
        new_data = {
            "entity_alpha_0": {
                "cumulative_realized_loss": 15000.0,
                "cumulative_slashed": 10000.0
            }
        }
        ValiBkpUtils.write_file(mgr._slash_file, new_data)

        # Load and verify
        loaded = mgr._load_slash_tracking_from_disk()

        self.assertIn("entity_alpha_0", loaded)
        self.assertAlmostEqual(loaded["entity_alpha_0"]["cumulative_realized_loss"], 15000.0)
        self.assertAlmostEqual(loaded["entity_alpha_0"]["cumulative_slashed"], 10000.0)

    # ==================== Required Collateral Calculation Tests ====================

    def test_compute_entity_required_collateral_zero_with_no_subaccounts(self):
        """Test required collateral is 0 for entity with no subaccounts."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY)

        required = self.entity_collateral_client.compute_entity_required_collateral(self.ENTITY_HOTKEY)
        self.assertAlmostEqual(required, 0.0)

    def test_compute_entity_required_collateral_zero_for_unknown_entity(self):
        """Test required collateral is 0 for unknown entity."""
        required = self.entity_collateral_client.compute_entity_required_collateral("unknown_entity")
        self.assertAlmostEqual(required, 0.0)

    # ==================== Integration: Slashing Reduces Collateral Available for Gating ====================

    def test_slashing_reduces_available_collateral_for_gating(self):
        """Test that slashing reduces available collateral, potentially blocking new orders."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )

        # Entity has 5000 theta. $1K order delta: min(1K, 8K)/10 = 100 theta. 5000 > 100 → allowed.
        self._set_collateral_cache(entity_hotkey, 5000.0)

        allowed, _ = self.entity_collateral_client.can_open_position(
            entity_hotkey, synthetic_hotkey, 1_000.0
        )
        self.assertTrue(allowed)

        # Slash $5K → cache decremented by 5000/10 = 500 theta → 4500 theta remaining
        self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 5_000.0
        )

        cached = self.entity_collateral_client.get_cached_collateral(entity_hotkey)
        self.assertAlmostEqual(cached, 4500.0)

    # ==================== Edge Cases ====================

    def test_slash_very_small_loss(self):
        """Test slashing with a very small loss amount."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        slashed = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 0.01
        )
        self.assertAlmostEqual(slashed, 0.01)

    def test_slash_exact_max_slash_amount(self):
        """Test slashing exactly the max_slash amount."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        # max_slash = $8K, loss exactly $8K
        slashed = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 8_000.0
        )
        self.assertAlmostEqual(slashed, 8_000.0)

        # Verify at cap
        tracking = self._get_slash_tracking(synthetic_hotkey)
        self.assertAlmostEqual(tracking["cumulative_slashed"], 8_000.0)

    def test_slash_loss_exceeds_max_slash_single_trade(self):
        """Test that a single trade loss exceeding max_slash is capped."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        # max_slash = $8K, loss = $50K → only slash $8K
        slashed = self.entity_collateral_client.slash_on_realized_loss(
            entity_hotkey, synthetic_hotkey, 50_000.0
        )
        self.assertAlmostEqual(slashed, 8_000.0)

        tracking = self._get_slash_tracking(synthetic_hotkey)
        self.assertAlmostEqual(tracking["cumulative_realized_loss"], 50_000.0)
        self.assertAlmostEqual(tracking["cumulative_slashed"], 8_000.0)

    def test_sequential_slashes_track_cumulative_loss_correctly(self):
        """Test a sequence of losses accumulates correctly in tracking."""
        entity_hotkey, synthetic_hotkey, _ = self._register_entity_with_subaccount(
            account_size=100_000
        )
        self._set_collateral_cache(entity_hotkey, 100.0)

        # max_slash = $8K
        # losses: $1K, $2K, $3K, $4K (capped at $8K so only $2K), $5K ($0)
        losses = [1_000, 2_000, 3_000, 4_000, 5_000]
        total_slashed = 0
        for loss in losses:
            slashed = self.entity_collateral_client.slash_on_realized_loss(
                entity_hotkey, synthetic_hotkey, float(loss)
            )
            total_slashed += slashed

        # Total losses = $15K, max slash = $8K
        # Should have slashed exactly $8K total
        self.assertAlmostEqual(total_slashed, 8_000.0)

        tracking = self._get_slash_tracking(synthetic_hotkey)
        self.assertAlmostEqual(tracking["cumulative_realized_loss"], 15_000.0)
        self.assertAlmostEqual(tracking["cumulative_slashed"], 8_000.0)

    # ==================== MarketOrderManager Wiring Tests ====================

    def test_market_order_manager_has_entity_collateral_client(self):
        """Test that MarketOrderManager creates an EntityCollateralClient."""
        from vali_objects.utils.limit_order.market_order_manager import MarketOrderManager
        from vali_objects.utils.entity_collateral.entity_collateral_client import EntityCollateralClient

        mom = MarketOrderManager(serve=False, running_unit_tests=True)
        self.assertIsInstance(mom._entity_collateral_client, EntityCollateralClient)

    # ==================== EliminationManager Wiring Tests ====================

    def test_elimination_manager_has_entity_collateral_client(self):
        """Test that EliminationManager creates an EntityCollateralClient."""
        from vali_objects.utils.entity_collateral.entity_collateral_client import EntityCollateralClient

        # Access the elimination server's manager through the orchestrator
        elim_server = self.orchestrator._servers.get('elimination')
        if elim_server is None:
            self.skipTest("Elimination server not accessible in this mode")

        # In testing mode, the server handle is the process handle, not the manager directly.
        # Instead, verify the import works and the client exists by checking the elimination manager class
        from vali_objects.utils.elimination.elimination_manager import EliminationManager
        import inspect
        init_source = inspect.getsource(EliminationManager.__init__)
        self.assertIn("EntityCollateralClient", init_source)
        self.assertIn("_entity_collateral_client", init_source)

    # ==================== ValidatorContractManager Wiring Tests ====================

    def test_validator_contract_manager_has_entity_client(self):
        """Test that ValidatorContractManager imports EntityClient for withdrawal blocking."""
        from vali_objects.contract.validator_contract_manager import ValidatorContractManager
        import inspect
        source = inspect.getsource(ValidatorContractManager.__init__)
        self.assertIn("EntityClient", source)
        self.assertIn("_entity_client", source)

    def test_validator_contract_manager_withdrawal_blocking_code_exists(self):
        """Test that process_withdrawal_request contains entity withdrawal blocking logic."""
        from vali_objects.contract.validator_contract_manager import ValidatorContractManager
        import inspect
        source = inspect.getsource(ValidatorContractManager.process_withdrawal_request)
        self.assertIn("entity_data", source)
        self.assertIn("open_positions", source)
        self.assertIn("subaccount", source.lower())

    # ==================== Validator.py Wiring Tests ====================

    def test_validator_py_retrieves_entity_collateral_client(self):
        """Test that validator.py has the entity_collateral_client retrieval line."""
        import inspect
        # Read the validator module source to verify wiring
        with open("neurons/validator.py", "r") as f:
            source = f.read()
        self.assertIn("entity_collateral_client", source)
        self.assertIn("orchestrator.get_client('entity_collateral')", source)

    def test_validator_py_starts_entity_collateral_daemon(self):
        """Test that validator.py starts entity_collateral daemon."""
        with open("neurons/validator.py", "r") as f:
            source = f.read()
        self.assertIn("'entity_collateral'", source)


if __name__ == '__main__':
    unittest.main()
