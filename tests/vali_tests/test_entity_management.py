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
from time_util.time_util import MS_IN_WEEK, TimeUtil
from entity_management.entity_utils import is_synthetic_hotkey, parse_synthetic_hotkey
from vali_objects.enums.miner_bucket_enum import BucketPhase, MinerBucket



# PRO_CHALLENGE_FUNDED in its transition week, and the ledger status its checkpoints carry
TRANSITION = BucketPhase(MinerBucket.SUBACCOUNT_FUNDED, in_transition=True)
TRANSITION_STATUS = TRANSITION.status


def _phase(bucket):
    if bucket is None or isinstance(bucket, BucketPhase):
        return bucket
    return bucket.phase()


def _status(bucket):
    """The challenge_period_status a checkpoint stamped in this bucket (or phase) carries."""
    return _phase(bucket).status

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

    def test_create_subaccount_with_intraday_drawdown_threshold(self):
        """A chosen intraday drawdown threshold is stored as a fraction and applies from the first bucket."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        success, subaccount_info, message = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=100_000,
            asset_class="crypto",
            drawdown_criteria="static",
            intraday_drawdown_threshold=0.03,
        )

        self.assertTrue(success, f"Subaccount creation failed: {message}")
        self.assertEqual(subaccount_info['intraday_drawdown_threshold'], 0.03)
        stats = self.challenge_period_client.get_drawdown_stats(subaccount_info['synthetic_hotkey'])
        self.assertEqual(stats['intraday_drawdown_threshold'], 0.03)

    def test_create_hl_subaccount_with_intraday_drawdown_threshold(self):
        """Hyperliquid subaccounts may choose an intraday drawdown threshold too."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        success, subaccount_info, message = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=100_000,
            hl_address="0x" + "b" * 40,
            intraday_drawdown_threshold=0.03,
        )

        self.assertTrue(success, f"Subaccount creation failed: {message}")
        self.assertEqual(subaccount_info['intraday_drawdown_threshold'], 0.03)
        stats = self.challenge_period_client.get_drawdown_stats(subaccount_info['synthetic_hotkey'])
        self.assertEqual(stats['intraday_drawdown_threshold'], 0.03)

    def test_create_subaccount_without_intraday_drawdown_threshold_keeps_bucket_default(self):
        """Omitting the intraday drawdown threshold keeps today's bucket threshold."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        success, subaccount_info, message = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=100_000,
            asset_class="crypto",
        )

        self.assertTrue(success, f"Subaccount creation failed: {message}")
        self.assertIsNone(subaccount_info['intraday_drawdown_threshold'])
        stats = self.challenge_period_client.get_drawdown_stats(subaccount_info['synthetic_hotkey'])
        self.assertEqual(stats['intraday_drawdown_threshold'], ValiConfig.CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD)

    def test_create_subaccount_rejects_invalid_intraday_drawdown_threshold(self):
        """Values outside SUBACCOUNT_INTRADAY_DRAWDOWN_VALUES are rejected before anything is created."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        for value in (0.04, 0.07 - 0.04, 0.02, 0.06, 3, True):
            success, subaccount_info, message = self.entity_client.create_subaccount(
                entity_hotkey=self.ENTITY_HOTKEY_1,
                account_size=100_000,
                asset_class="crypto",
                intraday_drawdown_threshold=value,
            )
            self.assertFalse(success, f"intraday_drawdown_threshold={value!r} should be rejected")
            self.assertIsNone(subaccount_info)
            self.assertIn("intraday_drawdown_threshold", message)

        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data['subaccounts']), 0)
        self.assertEqual(entity_data['next_subaccount_id'], 0)

    # ==================== Creation bucket ====================

    def _create(self, **kwargs):
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        return self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto", **kwargs)

    def test_create_subaccount_defaults_to_the_challenge_bucket(self):
        success, subaccount_info, message = self._create()
        self.assertTrue(success, message)
        self.assertEqual(subaccount_info['initial_bucket'], MinerBucket.SUBACCOUNT_CHALLENGE.value)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(subaccount_info['synthetic_hotkey']),
                         MinerBucket.SUBACCOUNT_CHALLENGE)

    def test_create_subaccount_directly_into_funded(self):
        success, subaccount_info, message = self._create(bucket=MinerBucket.SUBACCOUNT_FUNDED.value)
        self.assertTrue(success, message)
        hotkey = subaccount_info['synthetic_hotkey']
        self.assertEqual(self.challenge_period_client.get_miner_bucket(hotkey), MinerBucket.SUBACCOUNT_FUNDED)
        self.assertEqual(subaccount_info['account_type'], 'standard')
        self.assertEqual(subaccount_info['account_size'], 100_000)

    def test_create_subaccount_directly_into_pro_challenge_funded(self):
        """Starts on the pro account, with the fixed intraday limit, the chosen EOD limit, and a 1x payout scale."""
        success, subaccount_info, message = self._create(
            bucket=MinerBucket.PRO_CHALLENGE_FUNDED.value, pro_account_size=500_000,
            eod_hwm_threshold=0.05)
        self.assertTrue(success, message)
        # The promotion fee of a 3% daily loss limit account: 500K / 5,000 - 100K / 2,500
        self.assertAlmostEqual(subaccount_info['pro_fee_theta'], 60.0)
        self.assertIn(f"{subaccount_info['reg_fee_theta'] + subaccount_info['pro_fee_theta']} theta", message)
        self.assertIn("pro fee", message)
        hotkey = subaccount_info['synthetic_hotkey']

        # Instant Funded starts on the pro account, never in the transition
        self.assertEqual(self.challenge_period_client.get_miner_phase(hotkey), MinerBucket.PRO_CHALLENGE_FUNDED.phase())
        self.assertEqual(subaccount_info['account_type'], 'pro')
        self.assertEqual(subaccount_info['account_size'], 500_000)
        self.assertEqual(subaccount_info['standard_account_size'], 100_000)
        self.assertEqual(subaccount_info['pro_account_size'], 500_000)
        self.assertEqual(subaccount_info['asset_class'], 'all_markets')
        self.assertEqual(subaccount_info['payout_scale'], ValiConfig.GROW_DIRECT_CREATION_PAYOUT_MULTIPLIER)

        stats = self.challenge_period_client.get_drawdown_stats(hotkey)
        self.assertEqual(stats['intraday_drawdown_threshold'],
                         ValiConfig.INSTANT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD)
        self.assertEqual(stats['eod_drawdown_threshold'], 0.05)
        self.assertAlmostEqual(self.entity_client.get_payout_scale(hotkey),
                               ValiConfig.GROW_DIRECT_CREATION_PAYOUT_MULTIPLIER * 100_000 / 500_000)

    def test_create_subaccount_directly_into_pro_challenge_direct(self):
        success, subaccount_info, message = self._create(
            bucket=MinerBucket.PRO_CHALLENGE_DIRECT.value, pro_account_size=250_000)
        self.assertTrue(success, message)
        # 250K / 3,000 - 100K / 1,500 at the default 5% daily loss limit
        self.assertAlmostEqual(subaccount_info['pro_fee_theta'], 250_000 / 3000 - 100_000 / 1500)
        hotkey = subaccount_info['synthetic_hotkey']
        self.assertEqual(self.challenge_period_client.get_miner_bucket(hotkey), MinerBucket.PRO_CHALLENGE_DIRECT)
        self.assertEqual(subaccount_info['account_size'], 250_000)
        stats = self.challenge_period_client.get_drawdown_stats(hotkey)
        self.assertEqual(stats['intraday_drawdown_threshold'], ValiConfig.PRO_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD)
        self.assertEqual(stats['eod_drawdown_threshold'], ValiConfig.PRO_CHALLENGE_EOD_DRAWDOWN_THRESHOLD)

    def test_promotion_to_pro_funded_restores_the_pro_drawdown_defaults(self):
        success, subaccount_info, message = self._create(
            bucket=MinerBucket.PRO_CHALLENGE_FUNDED.value, pro_account_size=500_000, eod_hwm_threshold=0.05)
        self.assertTrue(success, message)
        hotkey = subaccount_info['synthetic_hotkey']

        self.challenge_period_client.set_miner_bucket(hotkey, MinerBucket.PRO_FUNDED, TimeUtil.now_in_millis())
        stats = self.challenge_period_client.get_drawdown_stats(hotkey)
        self.assertEqual(stats['intraday_drawdown_threshold'], ValiConfig.PRO_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD)
        self.assertEqual(stats['eod_drawdown_threshold'], ValiConfig.PRO_FUNDED_EOD_DRAWDOWN_THRESHOLD)

    def test_create_hl_subaccount_directly_into_funded(self):
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        success, subaccount_info, message = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1, account_size=100_000, hl_address="0x" + "d" * 40,
            bucket=MinerBucket.SUBACCOUNT_FUNDED.value)
        self.assertTrue(success, message)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(subaccount_info['synthetic_hotkey']),
                         MinerBucket.SUBACCOUNT_FUNDED)

    def test_registration_fee_by_creation_bucket(self):
        """Standard buckets pay the standard CPT for their daily loss limit; Instant Funded buckets pay the
        Instant Funded CPT for their EOD threshold (omitted is 8%)."""
        from_standard = MinerBucket.PRO_CHALLENGE_FUNDED.value
        cases = (
            ({}, 100_000 / 1500),
            ({"intraday_drawdown_threshold": 0.03}, 100_000 / 2500),
            ({"bucket": MinerBucket.PRO_CHALLENGE_DIRECT.value, "pro_account_size": 500_000}, 100_000 / 1500),
            ({"bucket": MinerBucket.SUBACCOUNT_FUNDED.value}, 100_000 / 300),
            ({"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "eod_hwm_threshold": 0.05}, 100_000 / 400),
            ({"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "eod_hwm_threshold": 0.08}, 100_000 / 300),
            ({"bucket": from_standard, "pro_account_size": 500_000, "eod_hwm_threshold": 0.05}, 100_000 / 400),
        )
        for kwargs, expected_theta in cases:
            with self.subTest(**kwargs):
                success, subaccount_info, message = self._create(**kwargs)
                self.assertTrue(success, message)
                self.assertAlmostEqual(subaccount_info['reg_fee_theta'], expected_theta)

    def test_small_accounts_register_at_half_the_cpt(self):
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        for bucket, expected_theta in ((MinerBucket.SUBACCOUNT_CHALLENGE, 10_000 / 750),
                                       (MinerBucket.SUBACCOUNT_FUNDED, 10_000 / 150)):
            with self.subTest(bucket=bucket):
                success, subaccount_info, message = self.entity_client.create_subaccount(
                    entity_hotkey=self.ENTITY_HOTKEY_1, account_size=10_000, asset_class="crypto",
                    bucket=bucket.value)
                self.assertTrue(success, message)
                self.assertAlmostEqual(subaccount_info['reg_fee_theta'], expected_theta)

    def test_instant_funded_into_funded_has_the_fixed_intraday_and_chosen_eod_limits(self):
        success, subaccount_info, message = self._create(
            bucket=MinerBucket.SUBACCOUNT_FUNDED.value, eod_hwm_threshold=0.05)
        self.assertTrue(success, message)
        self.assertEqual(subaccount_info['intraday_drawdown_threshold'],
                         ValiConfig.INSTANT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD)
        self.assertEqual(subaccount_info['eod_hwm_threshold'], 0.05)
        self.assertIsNone(subaccount_info['payout_scale'])
        stats = self.challenge_period_client.get_drawdown_stats(subaccount_info['synthetic_hotkey'])
        self.assertEqual(stats['intraday_drawdown_threshold'], ValiConfig.INSTANT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD)
        self.assertEqual(stats['eod_drawdown_threshold'], 0.05)

    def test_create_hl_subaccount_is_priced_like_a_standard_subaccount(self):
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        cases = (
            ({}, 100_000 / 1500, None),
            ({"intraday_drawdown_threshold": 0.03}, 100_000 / 2500, 0.03),
            ({"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "eod_hwm_threshold": 0.05}, 100_000 / 400, 0.03),
        )
        for i, (kwargs, expected_theta, expected_intraday) in enumerate(cases):
            with self.subTest(**kwargs):
                success, subaccount_info, message = self.entity_client.create_hl_subaccount(
                    entity_hotkey=self.ENTITY_HOTKEY_1, account_size=100_000, hl_address="0x" + f"{i}" * 40,
                    **kwargs)
                self.assertTrue(success, message)
                self.assertAlmostEqual(subaccount_info['reg_fee_theta'], expected_theta)
                self.assertEqual(subaccount_info['intraday_drawdown_threshold'], expected_intraday)

    def test_create_hl_subaccount_rejects_eod_outside_funded(self):
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        success, subaccount_info, message = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1, account_size=100_000, hl_address="0x" + "e" * 40,
            eod_hwm_threshold=0.05)
        self.assertFalse(success)
        self.assertIn("eod_hwm_threshold", message)
        self.assertEqual(len(self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)['subaccounts']), 0)

    def test_create_hl_subaccount_rejects_pro_buckets(self):
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        for bucket in (MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT):
            with self.subTest(bucket=bucket):
                success, subaccount_info, message = self.entity_client.create_hl_subaccount(
                    entity_hotkey=self.ENTITY_HOTKEY_1, account_size=100_000, hl_address="0x" + "d" * 40,
                    bucket=bucket.value)
                self.assertFalse(success)
                self.assertIn("bucket", message)
        self.assertEqual(len(self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)['subaccounts']), 0)

    def test_create_subaccount_rejects_invalid_bucket_options(self):
        """Every invalid combination is rejected before anything is created."""
        from_standard = MinerBucket.PRO_CHALLENGE_FUNDED.value
        cases = (
            ({"bucket": "PRO_FUNDED"}, "bucket"),
            ({"bucket": "ELIMINATED"}, "bucket"),
            ({"bucket": "NOT_A_BUCKET"}, "bucket"),
            ({"bucket": from_standard}, "pro_account_size"),
            ({"bucket": MinerBucket.PRO_CHALLENGE_DIRECT.value}, "pro_account_size"),
            ({"bucket": from_standard, "pro_account_size": 50_000}, "below"),
            ({"bucket": from_standard, "pro_account_size": 2_000_000}, "at most"),
            ({"pro_account_size": 500_000}, "pro_account_size"),
            ({"eod_hwm_threshold": 0.05}, "eod_hwm_threshold"),
            ({"bucket": MinerBucket.PRO_CHALLENGE_DIRECT.value, "pro_account_size": 500_000,
              "eod_hwm_threshold": 0.05}, "eod_hwm_threshold"),
            ({"bucket": from_standard, "pro_account_size": 500_000, "eod_hwm_threshold": 0.06}, "eod_hwm_threshold"),
            ({"bucket": from_standard, "pro_account_size": 500_000, "intraday_drawdown_threshold": 0.05},
             "intraday_drawdown_threshold"),
            ({"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "intraday_drawdown_threshold": 0.05},
             "intraday_drawdown_threshold"),
            ({"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "eod_hwm_threshold": 0.06}, "eod_hwm_threshold"),
            ({"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "pro_account_size": 500_000}, "pro_account_size"),
            # Static rules never check the EOD high-water mark an Instant Funded account is priced on
            ({"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "drawdown_criteria": "static"}, "drawdown_criteria"),
            ({"bucket": from_standard, "pro_account_size": 500_000, "drawdown_criteria": "static"},
             "drawdown_criteria"),
        )
        for kwargs, fragment in cases:
            with self.subTest(**kwargs):
                success, subaccount_info, message = self._create(**kwargs)
                self.assertFalse(success, f"{kwargs} should be rejected")
                self.assertIsNone(subaccount_info)
                self.assertIn(fragment, message)

        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data['subaccounts']), 0)
        self.assertEqual(entity_data['next_subaccount_id'], 0)

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

    # ==================== client_ref Idempotency Tests ====================

    def test_client_ref_dedupe_returns_existing(self):
        """Same (entity, client_ref) twice => one subaccount, 2nd flagged duplicate."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        ok1, s1, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto",
            client_ref="acct-abc-123")
        self.assertTrue(ok1)
        self.assertIsNone(s1.get("duplicate"))  # first create is not a duplicate

        ok2, s2, msg2 = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto",
            client_ref="acct-abc-123")
        self.assertTrue(ok2)
        self.assertTrue(s2.get("duplicate"), "retry must be flagged duplicate")
        self.assertEqual(s2["subaccount_id"], s1["subaccount_id"])
        self.assertEqual(s2["subaccount_uuid"], s1["subaccount_uuid"])

        # Exactly one subaccount exists, and the id counter advanced only once.
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data["subaccounts"]), 1)
        self.assertEqual(entity_data["next_subaccount_id"], 1)
        self.assertIn("duplicate", msg2.lower())

    def test_client_ref_persisted_on_subaccount(self):
        """The normalized client_ref is stored on the created subaccount."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, s, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=50_000, asset_class="crypto",
            client_ref="acct-XYZ-9")
        # normalized to lowercase
        self.assertEqual(s["client_ref"], "acct-xyz-9")

    def test_client_ref_normalization_collides(self):
        """' ABC ' and 'abc' normalize to the same key and dedupe together."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, s1, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=25_000, asset_class="crypto",
            client_ref="  ABC  ")
        _, s2, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=25_000, asset_class="crypto",
            client_ref="abc")
        self.assertTrue(s2.get("duplicate"))
        self.assertEqual(s2["subaccount_id"], s1["subaccount_id"])
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data["subaccounts"]), 1)

    def test_without_client_ref_unchanged(self):
        """No client_ref => every call creates a new subaccount (today's behavior)."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        for _ in range(3):
            ok, s, _ = self.entity_client.create_subaccount(
                self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto")
            self.assertTrue(ok)
            self.assertIsNone(s.get("duplicate"))
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data["subaccounts"]), 3)

    def test_client_ref_scoped_per_entity(self):
        """Same client_ref under two different entities => two subaccounts."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_2)
        _, s1, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto",
            client_ref="shared-ref")
        ok2, s2, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_2, account_size=100_000, asset_class="crypto",
            client_ref="shared-ref")
        self.assertTrue(ok2)
        self.assertIsNone(s2.get("duplicate"), "ref is namespaced per entity")
        self.assertNotEqual(s1["subaccount_uuid"], s2["subaccount_uuid"])

    def test_different_client_ref_distinct_subaccounts(self):
        """Different refs, identical params => two subaccounts (legit 2nd purchase)."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, s1, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto",
            client_ref="ref-one")
        ok2, s2, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto",
            client_ref="ref-two")
        self.assertTrue(ok2)
        self.assertIsNone(s2.get("duplicate"))
        self.assertNotEqual(s1["subaccount_id"], s2["subaccount_id"])
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data["subaccounts"]), 2)

    def test_malformed_client_ref_disables_dedupe(self):
        """A ref that fails normalization (>64 chars) is treated as absent, so
        each call creates a new subaccount rather than silently claiming a slot."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        bad = "x" * 65
        _, s1, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto",
            client_ref=bad)
        _, s2, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto",
            client_ref=bad)
        self.assertIsNone(s1.get("client_ref"))
        self.assertIsNone(s2.get("duplicate"))
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data["subaccounts"]), 2)

    def test_client_ref_dedupe_after_elimination(self):
        """A retry under a ref whose subaccount was eliminated returns the
        eliminated subaccount and creates nothing (a claim is never released)."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        _, s1, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto",
            client_ref="elim-ref")
        self.entity_client.eliminate_subaccount(
            self.ENTITY_HOTKEY_1, s1["subaccount_id"])
        ok2, s2, _ = self.entity_client.create_subaccount(
            self.ENTITY_HOTKEY_1, account_size=100_000, asset_class="crypto",
            client_ref="elim-ref")
        self.assertTrue(ok2)
        self.assertTrue(s2.get("duplicate"))
        self.assertEqual(s2["subaccount_id"], s1["subaccount_id"])
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY_1)
        self.assertEqual(len(entity_data["subaccounts"]), 1)

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
    the pro track. Every bucket segment is paid on its own: closed earning segments from the payout
    record pinned when they closed, the live one from its own positions."""

    ENTITY_HOTKEY = "entity"
    SUBACCOUNT_HOTKEY = "entity_1"
    SUBACCOUNT_UUID = "uuid-1"
    CP_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    def _week_0_start(self):
        return TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK

    def _manager(self, blocked_checkpoint_indices=(), week_buckets=None, payout_scale=1.0,
                 orders=None, checkpoint_buckets=None, has_perf_ledger=True, subaccount=None,
                 has_debt_ledger=True, segment_payout_scale=None, has_bucket_history=True,
                 current_bucket=None):
        """Two payout weeks of 12h checkpoints. `week_buckets` maps a payout-week index (0 or 1) to
        the bucket of that week (default PRO_FUNDED); `checkpoint_buckets` overrides individual
        checkpoints, which is how a bucket change lands mid-week. The bucket segments follow the
        checkpoints, each closed earning segment carrying the payout record its hop would have
        written, and each segment trading one position holding its own orders. `payout_scale` is the
        live standard/pro ratio; `segment_payout_scale` the one captured when each segment opened.
        `orders` overrides the default one order realizing 10 USD per 12h cell."""
        from entity_management.entity_manager import EntityManager

        week_0_start = self._week_0_start()
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        n_cps = 2 * cps_per_week
        week_buckets = week_buckets or {}
        checkpoint_buckets = checkpoint_buckets or {}
        phases = [_phase(checkpoint_buckets.get(i, week_buckets.get(i // cps_per_week, MinerBucket.PRO_FUNDED)))
                  for i in range(n_cps)]
        orders = orders if orders is not None else [
            SimpleNamespace(processed_ms=week_0_start + i * self.CP_DURATION_MS + 1, realized_pnl=10.0,
                            to_python_dict=lambda: {})
            for i in range(n_cps)
        ]
        debt_checkpoints = [
            DebtCheckpoint(
                timestamp_ms=week_0_start + (i + 1) * self.CP_DURATION_MS,
                weekly_penalty=0.0 if i in blocked_checkpoint_indices else 1.0,
                challenge_period_status=phases[i].status,
            )
            for i in range(n_cps)
        ]

        # One segment per run of a bucket; the transition is a range inside SUBACCOUNT_FUNDED
        segments = []
        for i, phase in enumerate(phases):
            boundary_ms = week_0_start + i * self.CP_DURATION_MS
            if not segments or segments[-1]['bucket'] != phase.bucket.value:
                if segments:
                    segments[-1]['end_ms'] = boundary_ms
                segments.append({'bucket': phase.bucket.value, 'start_ms': boundary_ms, 'end_ms': None,
                                 'transition_start_ms': None, 'transition_end_ms': None,
                                 'payout_scale': segment_payout_scale, 'payout_record': None})
            if phase.in_transition and segments[-1]['transition_start_ms'] is None:
                segments[-1]['transition_start_ms'] = boundary_ms
        positions = [
            SimpleNamespace(
                open_ms=seg['start_ms'], fee_history=[], unrealized_pnl=0.0,
                orders=[o for o in orders if seg['start_ms'] <= o.processed_ms
                        and (seg['end_ms'] is None or o.processed_ms < seg['end_ms'])],
            )
            for seg in segments
        ]

        manager = object.__new__(EntityManager)
        manager.running_unit_tests = True
        manager.get_synthetic_hotkey_from_uuid = lambda _uuid: self.SUBACCOUNT_HOTKEY
        manager.get_entity_data = lambda _hk: SimpleNamespace(subaccounts={1: {'id': 1}})
        if subaccount is not None:
            manager.get_subaccount_info_for_synthetic = lambda _hk: subaccount
        else:
            manager.get_payout_scale = lambda _hk: payout_scale
        manager._debt_ledger_client = SimpleNamespace(
            get_ledger=lambda _hk: (DebtLedger(self.SUBACCOUNT_HOTKEY, checkpoints=debt_checkpoints)
                                   if has_debt_ledger else None),
        )
        manager._perf_ledger_client = SimpleNamespace(
            get_perf_ledger_for_hotkey=lambda hk: {
                hk: SimpleNamespace(get_checkpoint_at_time=lambda *_a: None)
            } if has_perf_ledger else {},
            get_frozen_ledgers=lambda: {},
        )
        current = _phase(current_bucket) if current_bucket is not None else phases[-1]
        manager._challenge_period_client = SimpleNamespace(
            get_miner_phase=lambda *_a: current,
            get_miner_bucket=lambda *_a: current.bucket,
            get_bucket_segments=lambda _hk: segments if has_bucket_history else [],
        )
        manager._position_client = SimpleNamespace(
            get_positions_for_one_hotkey=lambda *_a, **_k: positions
        )

        for segment, position in zip(segments, positions):
            if segment['end_ms'] is not None and BucketPhase.from_status(segment['bucket']).is_subaccount_earning:
                segment['payout_record'] = manager.calculate_segment_payout_record(
                    self.SUBACCOUNT_HOTKEY, segment, [position], segment['end_ms'])
        manager.segments = segments
        return manager

    def _payout_result(self, **kwargs):
        manager = kwargs.pop('manager', None) or self._manager(**kwargs)
        week_0_start = self._week_0_start()
        return manager.calculate_subaccount_payout(self.SUBACCOUNT_UUID, week_0_start, week_0_start + 2 * MS_IN_WEEK)

    def _payouts_by_week(self, blocked_checkpoint_indices=()):
        result = self._payout_result(blocked_checkpoint_indices=blocked_checkpoint_indices)
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
        """Week 0 withheld in PRO_FUNDED, then the account moves to SUBACCOUNT_FUNDED. The record
        PRO_FUNDED closes with forfeits the escrow, and the new segment pays its own week."""
        result = self._payout_result(
            blocked_checkpoint_indices=(8,),
            week_buckets={1: MinerBucket.SUBACCOUNT_FUNDED},
        )
        week_0, week_1 = result['weekly_settlements']
        self.assertTrue(week_0['payout_record'])
        self.assertEqual(week_0['payout'], 0.0)
        self.assertAlmostEqual(week_0['deferred_forfeited'], 140.0)
        self.assertEqual(week_0['deferred_balance'], 0.0)
        self.assertFalse(week_1['payout_record'])
        self.assertAlmostEqual(week_1['payout'], 140.0)
        self.assertAlmostEqual(result['deferred_forfeited'], 140.0)
        self.assertEqual(result['deferred_balance'], 0.0)
        self.assertAlmostEqual(result['payout'], 140.0)
        self.assertTrue(result['off_track'])

    def test_leaving_pro_track_with_zero_balance_forfeits_nothing(self):
        result = self._payout_result(week_buckets={1: MinerBucket.SUBACCOUNT_FUNDED})
        for week in result['weekly_settlements']:
            self.assertEqual(week['deferred_forfeited'], 0.0)
            self.assertEqual(week['deferred_balance'], 0.0)
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [140.0, 140.0])
        self.assertTrue(result['off_track'])

    def test_non_earning_bucket_returns_empty_settlement_with_deferral_fields(self):
        result = self._payout_result(week_buckets={0: MinerBucket.SUBACCOUNT_CHALLENGE,
                                                   1: MinerBucket.SUBACCOUNT_CHALLENGE})
        self.assertEqual(result['weekly_settlements'], [])
        self.assertEqual(result['deferred_balance'], 0.0)
        self.assertEqual(result['deferred_forfeited'], 0.0)
        self.assertTrue(result['off_track'])

    def test_elimination_forfeits_the_live_segment_but_keeps_earlier_records(self):
        """No record is written when an earning segment is eliminated, so nothing it earned is paid.
        The record of the segment before it still is."""
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_FUNDED, 1: MinerBucket.PRO_FUNDED},
            current_bucket=MinerBucket.ELIMINATED,
            payout_scale=0.2,
        )
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [28.0])
        self.assertTrue(result['weekly_settlements'][0]['payout_record'])

    def test_pro_challenge_direct_gains_are_never_paid_after_promotion(self):
        """A pro challenge run directly on the pro account earns nothing, and PRO_FUNDED is only
        paid on what it trades itself."""
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_DIRECT, 1: MinerBucket.PRO_FUNDED}
        )
        self.assertEqual([w['bucket'] for w in result['weekly_settlements']], [MinerBucket.PRO_FUNDED.value])
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [140.0])
        self.assertAlmostEqual(result['payout'], 140.0)

    def test_pro_challenge_funded_is_paid_on_the_standard_account_size(self):
        """PRO_CHALLENGE_FUNDED trades the pro account but is paid on the standard one: with a 100k
        standard account inside a 500k pro account its 140 pays 28, pinned in its payout record.
        PRO_FUNDED starts from zero and pays its own 140 unscaled."""
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_FUNDED, 1: MinerBucket.PRO_FUNDED},
            payout_scale=0.2,
        )
        rows = result['weekly_settlements']
        self.assertEqual([w['payout_record'] for w in rows], [True, False])
        self.assertEqual([w['payout_scale'] for w in rows], [0.2, 1.0])
        self.assertEqual([w['payout'] for w in rows], [28.0, 140.0])
        self.assertAlmostEqual(result['payout'], 168.0)

    def test_pro_funded_is_paid_only_on_what_it_makes_after_promotion(self):
        """Promoted at +30K on a 500K pro account, then up 1K: PRO_FUNDED pays 1K, not 31K. The
        30K is paid once, at standard scale, from the PRO_CHALLENGE_FUNDED record."""
        week_0_start = self._week_0_start()
        orders = [
            SimpleNamespace(processed_ms=week_0_start + 1, realized_pnl=30_000.0, to_python_dict=lambda: {}),
            SimpleNamespace(processed_ms=week_0_start + MS_IN_WEEK + 1, realized_pnl=1_000.0, to_python_dict=lambda: {}),
        ]
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_FUNDED, 1: MinerBucket.PRO_FUNDED},
            orders=orders, payout_scale=0.2,
        )
        record_row, funded_row = result['weekly_settlements']
        self.assertAlmostEqual(record_row['payout'], 6_000.0)
        self.assertAlmostEqual(funded_row['payout'], 1_000.0)
        self.assertAlmostEqual(funded_row['eow_balance'], 1_000.0)

    def test_the_transition_week_is_paid_unscaled_before_the_pro_phase_is_scaled(self):
        result = self._payout_result(
            week_buckets={0: TRANSITION, 1: MinerBucket.PRO_CHALLENGE_FUNDED},
            payout_scale=0.2,
        )
        rows = result['weekly_settlements']
        self.assertEqual([r['bucket'] for r in rows],
                         [MinerBucket.SUBACCOUNT_FUNDED.value, MinerBucket.PRO_CHALLENGE_FUNDED.value])
        self.assertEqual([r['in_transition'] for r in rows], [True, False])
        self.assertEqual([r['payout_scale'] for r in rows], [1.0, 0.2])
        self.assertEqual([r['payout'] for r in rows], [140.0, 28.0])

    def test_the_transition_splits_the_funded_segment_where_it_starts(self):
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        result = self._payout_result(
            checkpoint_buckets={i: TRANSITION for i in range(cps_per_week // 2, cps_per_week)},
            week_buckets={0: MinerBucket.SUBACCOUNT_FUNDED, 1: MinerBucket.PRO_CHALLENGE_FUNDED},
            payout_scale=0.2,
        )
        rows = result['weekly_settlements']
        self.assertEqual([r['in_transition'] for r in rows], [False, True, False])
        self.assertEqual([r['payout'] for r in rows], [70.0, 70.0, 28.0])

    def test_the_chosen_payout_scale_prices_the_subaccount_payout(self):
        subaccount = SimpleNamespace(standard_account_size=100_000.0, pro_account_size=500_000.0, payout_scale=1.0)
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_FUNDED, 1: MinerBucket.PRO_FUNDED},
            subaccount=subaccount,
        )
        self.assertEqual([w['payout_scale'] for w in result['weekly_settlements']], [0.2, 1.0])
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [28.0, 140.0])

    def test_no_chosen_payout_scale_keeps_the_network_multiplier_in_the_payout(self):
        subaccount = SimpleNamespace(standard_account_size=100_000.0, pro_account_size=500_000.0, payout_scale=None)
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_FUNDED, 1: MinerBucket.PRO_CHALLENGE_FUNDED},
            subaccount=subaccount,
        )
        scale = ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER * 0.2
        self.assertAlmostEqual(result['payout'], 280.0 * scale)

    def test_a_resize_after_the_segment_opened_does_not_reprice_it(self):
        """The scale is the one captured when the segment opened, not the live one."""
        result = self._payout_result(
            week_buckets={0: MinerBucket.PRO_CHALLENGE_FUNDED, 1: MinerBucket.PRO_CHALLENGE_FUNDED},
            segment_payout_scale=0.2, payout_scale=0.5,
        )
        self.assertEqual([w['payout_scale'] for w in result['weekly_settlements']], [0.2, 0.2])
        self.assertAlmostEqual(result['payout'], 56.0)

    def test_a_breach_after_a_mid_week_promotion_still_pays_the_pre_promotion_stretch(self):
        """The PRO_CHALLENGE_FUNDED half of week 0 is paid from its record at its standard scale
        (70 gross at 0.2); only the PRO_FUNDED half is withheld, and week 1 releases it."""
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        promoted_at = cps_per_week // 2
        result = self._payout_result(
            checkpoint_buckets={
                i: (MinerBucket.PRO_CHALLENGE_FUNDED if i < promoted_at else MinerBucket.PRO_FUNDED)
                for i in range(2 * cps_per_week)
            },
            blocked_checkpoint_indices=(promoted_at + 3,),
            payout_scale=0.2,
        )
        rows = result['weekly_settlements']
        self.assertEqual([r['bucket'] for r in rows], [
            MinerBucket.PRO_CHALLENGE_FUNDED.value,
            MinerBucket.PRO_FUNDED.value,
            MinerBucket.PRO_FUNDED.value,
        ])
        self.assertEqual([r['weekly_penalty'] for r in rows], [1.0, 0.0, 1.0])
        self.assertEqual([r['payout_scale'] for r in rows], [0.2, 1.0, 1.0])
        self.assertEqual([r['payout'] for r in rows], [14.0, 0.0, 210.0])
        self.assertEqual([r['deferred'] for r in rows], [0.0, 70.0, 0.0])
        self.assertEqual(rows[2]['deferred_released'], 70.0)
        self.assertAlmostEqual(result['payout'], 224.0)
        self.assertEqual(result['deferred_forfeited'], 0.0)

    def test_a_breach_before_the_promotion_cannot_withhold_the_challenge_half(self):
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        promoted_at = cps_per_week // 2
        result = self._payout_result(
            checkpoint_buckets={
                i: (MinerBucket.PRO_CHALLENGE_FUNDED if i < promoted_at else MinerBucket.PRO_FUNDED)
                for i in range(2 * cps_per_week)
            },
            blocked_checkpoint_indices=(promoted_at - 3,),
            payout_scale=0.2,
        )
        rows = result['weekly_settlements']
        self.assertEqual([r['weekly_penalty'] for r in rows], [1.0, 0.0, 1.0])
        self.assertEqual([r['payout'] for r in rows], [14.0, 0.0, 210.0])

    def test_standard_subaccount_payouts_are_unchanged(self):
        result = self._payout_result(
            week_buckets={0: MinerBucket.SUBACCOUNT_FUNDED, 1: MinerBucket.SUBACCOUNT_FUNDED},
        )
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [140.0, 140.0])
        self.assertEqual([w['payout_scale'] for w in result['weekly_settlements']], [1.0, 1.0])
        self.assertEqual([w['weekly_penalty'] for w in result['weekly_settlements']], [1.0, 1.0])
        self.assertAlmostEqual(result['payout'], 280.0)
        self.assertEqual(result['deferred_balance'], 0.0)
        self.assertEqual(result['deferred_forfeited'], 0.0)

    def test_a_pinned_record_is_paid_as_recorded_not_recomputed(self):
        """An amended record pays its amended figure; the segment's positions are not read again."""
        manager = self._manager(week_buckets={0: MinerBucket.PRO_CHALLENGE_FUNDED, 1: MinerBucket.PRO_FUNDED},
                                payout_scale=0.2)
        manager.segments[0]['payout_record']['settlements'][0]['payout'] = 999.0
        result = self._payout_result(manager=manager)
        self.assertEqual([w['payout'] for w in result['weekly_settlements']], [999.0, 140.0])

    def test_a_record_pays_when_the_promoted_account_has_not_traded(self):
        """The transition record is all there is to pay; the new account has no orders yet."""
        week_0_start = self._week_0_start()
        orders = [SimpleNamespace(processed_ms=week_0_start + i * self.CP_DURATION_MS + 1, realized_pnl=10.0,
                                  to_python_dict=lambda: {}) for i in range(MS_IN_WEEK // self.CP_DURATION_MS)]
        result = self._payout_result(
            week_buckets={0: TRANSITION, 1: MinerBucket.PRO_CHALLENGE_FUNDED}, orders=orders)
        rows = result['weekly_settlements']
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['payout_record'])
        self.assertIs(rows[0]['in_transition'], True)
        self.assertEqual(result['payout'], 140.0)

    def test_the_record_forfeits_held_escrow_and_drops_the_orders(self):
        manager = self._manager(blocked_checkpoint_indices=(8,), week_buckets={1: MinerBucket.SUBACCOUNT_FUNDED})
        record = manager.segments[0]['payout_record']
        self.assertEqual(record['payout_usd'], 0.0)
        self.assertEqual(record['gross_payout_usd'], 140.0)
        self.assertEqual(record['settlements'][-1]['deferred_forfeited'], 140.0)
        self.assertEqual(record['settlements'][-1]['orders'], [])

    def test_the_record_reads_positions_passed_as_dicts(self):
        """The challenge period manager passes the positions read back from the segment directory."""
        from vali_objects.enums.order_type_enum import OrderType
        from vali_objects.vali_config import TradePair
        from vali_objects.vali_dataclasses.order import Order
        from vali_objects.vali_dataclasses.position import Position

        manager = self._manager(week_buckets={0: MinerBucket.SUBACCOUNT_FUNDED, 1: MinerBucket.SUBACCOUNT_FUNDED})
        week_0_start = self._week_0_start()
        position = Position(miner_hotkey=self.SUBACCOUNT_HOTKEY, position_uuid="p1", open_ms=week_0_start + 1,
                            trade_pair=TradePair.EURUSD, position_type=OrderType.FLAT, is_closed_position=True,
                            close_ms=week_0_start + 2, orders=[
                                Order(order_type=OrderType.LONG, leverage=1.0, price=1.0, processed_ms=week_0_start + 1,
                                      order_uuid="o1", trade_pair=TradePair.EURUSD),
                                Order(order_type=OrderType.FLAT, leverage=0.0, price=1.0, processed_ms=week_0_start + 2,
                                      order_uuid="o2", trade_pair=TradePair.EURUSD, realized_pnl=50.0),
                            ])
        segment = {**manager.segments[0], 'end_ms': week_0_start + MS_IN_WEEK}
        record = manager.calculate_segment_payout_record(
            self.SUBACCOUNT_HOTKEY, segment, [position.model_dump()], segment['end_ms'])
        self.assertEqual(record['payout_usd'], 50.0)

    def test_no_debt_ledger_pays_the_segment_unpenalized(self):
        """Without a debt ledger the weeks carry no penalty, and the scale still only applies in
        PRO_CHALLENGE_FUNDED."""
        for bucket, expected in (
            (MinerBucket.PRO_CHALLENGE_FUNDED, 140.0),
            (TRANSITION, 280.0),
            (MinerBucket.PRO_FUNDED, 280.0),
        ):
            with self.subTest(bucket=bucket):
                result = self._payout_result(has_debt_ledger=False, week_buckets={0: bucket, 1: bucket},
                                             payout_scale=0.5)
                self.assertAlmostEqual(result['payout'], expected)
                for week in result['weekly_settlements']:
                    self.assertEqual(week['bucket'], _phase(bucket).value)
                    self.assertEqual(week['in_transition'], _phase(bucket).in_transition)

    def test_no_perf_ledger_and_no_snapshot_is_not_found(self):
        self.assertIsNone(self._payout_result(has_perf_ledger=False))

    def test_no_bucket_history_is_one_segment_in_the_current_bucket(self):
        week_0_start = self._week_0_start()
        result = self._payout_result(has_bucket_history=False)
        self.assertEqual(
            [(w['start_ms'], w['end_ms']) for w in result['weekly_settlements']],
            [(week_0_start, week_0_start + MS_IN_WEEK), (week_0_start + MS_IN_WEEK, week_0_start + 2 * MS_IN_WEEK)],
        )
        self.assertAlmostEqual(result['payout'], 280.0)

    def test_rows_split_at_mondays_and_at_the_segment_boundary(self):
        week_0_start = self._week_0_start()
        switch_ms = week_0_start + MS_IN_WEEK // 2
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        result = self._payout_result(checkpoint_buckets={
            i: TRANSITION if i < cps_per_week // 2 else MinerBucket.PRO_CHALLENGE_FUNDED
            for i in range(2 * cps_per_week)
        })
        self.assertEqual(
            [(w['start_ms'], w['end_ms']) for w in result['weekly_settlements']],
            [(week_0_start, switch_ms), (switch_ms, week_0_start + MS_IN_WEEK),
             (week_0_start + MS_IN_WEEK, week_0_start + 2 * MS_IN_WEEK)],
        )


class TestEntityRegistrationMothershipOnly(TestBase):
    """Non-mothership slashes are dry runs, so entity registration must stay mothership-only."""

    def test_non_mothership_rejects_registration(self):
        import threading
        from unittest.mock import MagicMock
        from entity_management.entity_manager import EntityManager

        m = object.__new__(EntityManager)
        m.running_unit_tests = False
        m.is_mothership = False
        m._entities_lock = threading.RLock()
        m.entities = {}
        m._position_client = MagicMock()
        m._position_client.get_positions_for_one_hotkey.return_value = []
        m._contract_client = MagicMock()

        success, message = m.register_entity("5Entity")
        self.assertFalse(success)
        self.assertIn("mothership", message)
        m._contract_client.slash_miner_collateral.assert_not_called()
        self.assertNotIn("5Entity", m.entities)



class TestSubaccountCreationRollback(unittest.TestCase):
    """create_subaccount_ex run in-process (the orchestrator servers above cannot be made to fail), so
    a failed challenge period registration and the broadcast order can be checked directly."""

    ENTITY = "entity_alpha"

    def setUp(self):
        from unittest.mock import MagicMock
        from entity_management.entity_manager import EntityData
        from tests.vali_tests.test_pro_account_size import _bare_manager

        self.manager = _bare_manager()
        self.manager._uuid_to_hotkey = {}
        self.manager._entity_collateral_client = MagicMock()
        self.manager._entity_collateral_client.get_cached_collateral.return_value = 1_000_000.0
        self.manager._entity_collateral_client.compute_entity_required_collateral.return_value = 0.0
        self.manager.entities[self.ENTITY] = EntityData(entity_hotkey=self.ENTITY, registered_at_ms=0)

    def _net_cache_offset(self) -> float:
        offsets = self.manager._entity_collateral_client.offset_collateral_cache.call_args_list
        return sum(call.args[1] for call in offsets)

    def test_a_failed_challenge_period_registration_undoes_the_creation(self):
        """The record, its id, both fees and any challenge period state go; a retry reuses the id."""
        cases = (
            {},
            {"bucket": MinerBucket.SUBACCOUNT_FUNDED.value},
            {"bucket": MinerBucket.PRO_CHALLENGE_FUNDED.value, "pro_account_size": 500_000},
        )
        for options in cases:
            with self.subTest(**options):
                self.setUp()
                challenge_period = self.manager._challenge_period_client
                challenge_period.set_miner_bucket.side_effect = RuntimeError("rpc down")

                success, info, message, _ = self.manager.create_subaccount_ex(self.ENTITY, 100_000, "crypto", **options)

                self.assertFalse(success)
                self.assertIsNone(info)
                self.assertIn("challenge period", message)
                entity = self.manager.entities[self.ENTITY]
                self.assertEqual(entity.subaccounts, {})
                self.assertEqual(entity.next_subaccount_id, 0)
                self.assertAlmostEqual(self._net_cache_offset(), 0.0)
                challenge_period.remove_miners.assert_called_once_with([f"{self.ENTITY}_0"])

                challenge_period.set_miner_bucket.side_effect = None
                success, info, message, _ = self.manager.create_subaccount_ex(self.ENTITY, 100_000, "crypto", **options)
                self.assertTrue(success, message)
                self.assertEqual(info.synthetic_hotkey, f"{self.ENTITY}_0")
                # The successful retry holds its fees: registration, plus the pro fee for a pro bucket
                self.assertAlmostEqual(self._net_cache_offset(), -(info.reg_fee_theta + info.pro_fee_theta))

    def test_a_pro_creation_broadcasts_once_after_registration(self):
        """apply_bucket_account_size does not broadcast during creation; peers get the finished record once."""
        from unittest.mock import MagicMock

        self.manager.running_unit_tests = False
        self.manager._websocket_client = MagicMock()
        self.manager.broadcast_subaccount_dashboard = MagicMock()
        order = []
        self.manager._challenge_period_client.set_miner_bucket.side_effect = lambda *a, **k: order.append("registered")
        self.manager.broadcast_subaccount_registration = MagicMock(side_effect=lambda *a, **k: order.append("broadcast"))

        success, info, message, _ = self.manager.create_subaccount_ex(
            self.ENTITY, 100_000, "crypto",
            bucket=MinerBucket.PRO_CHALLENGE_FUNDED.value, pro_account_size=500_000)

        self.assertTrue(success, message)
        self.assertEqual(order, ["registered", "broadcast"])
        broadcast_info = self.manager.broadcast_subaccount_registration.call_args.args[1]
        self.assertEqual(broadcast_info.account_size, 500_000)
        self.assertEqual(broadcast_info.asset_class, "all_markets")


class TestAdminEliminateSubaccount(unittest.TestCase):
    """POST /admin/eliminate/<hotkey> on the validator, with mocked RPC clients."""

    def test_an_entitys_first_subaccount_can_be_eliminated(self):
        """Subaccount ids start at 0: id 0 is a valid subaccount, not a malformed hotkey."""
        from unittest.mock import MagicMock
        from flask import Flask
        from vanta_api.validator_rest_server import ValidatorRestServer

        server = object.__new__(ValidatorRestServer)
        server._get_api_key_safe = MagicMock(return_value="key")
        server.is_valid_api_key = MagicMock(return_value=True)
        server.can_access_tier = MagicMock(return_value=True)
        for client in ("_entity_client", "_elimination_client", "_challenge_period_client", "_miner_account_client"):
            setattr(server, client, MagicMock())
        server._entity_client.eliminate_subaccount.return_value = (True, "eliminated")
        app = Flask(__name__)
        app.route("/admin/eliminate/<hotkey>", methods=["POST"])(server.eliminate_hotkey)

        response = app.test_client().post("/admin/eliminate/entity_alpha_0", json={})

        self.assertEqual(response.status_code, 200, response.get_json())
        server._entity_client.eliminate_subaccount.assert_called_once_with(
            entity_hotkey="entity_alpha", subaccount_id=0, reason="DEREGISTERED")
        server._elimination_client.append_elimination_row.assert_called_once()


class TestSubaccountPayout(unittest.TestCase):
    """The payout calculation on a bare EntityManager, with mocked RPC clients."""

    def test_an_entitys_first_subaccount_is_paid_like_any_other(self):
        """Subaccount ids start at 0: the payout calculation reads the same trades the same way on 0 and 1."""
        from unittest.mock import MagicMock, patch
        from tests.vali_tests.test_challengeperiod_pro import MONDAY_MS
        from tests.vali_tests.test_pro_account_size import _add_standard, _bare_manager

        entity_manager = _bare_manager()
        entity_manager._challenge_period_client.get_miner_phase.return_value = BucketPhase(MinerBucket.SUBACCOUNT_FUNDED)
        entity_manager._challenge_period_client.get_bucket_segments.return_value = []
        entity_manager._perf_ledger_client = MagicMock()
        order = SimpleNamespace(processed_ms=MONDAY_MS - 2 * MS_IN_WEEK + 3_600_000, realized_pnl=12_000.0,
                                to_python_dict=lambda: {})
        entity_manager._position_client.get_positions_for_one_hotkey.return_value = [
            SimpleNamespace(orders=[order], fee_history=[], unrealized_pnl=0.0, open_ms=order.processed_ms)]
        checkpoint = SimpleNamespace(unrealized_pnl=0.0)

        payouts = {}
        for subaccount_id in (0, 1):
            hotkey = _add_standard(entity_manager, subaccount_id=subaccount_id)
            entity_manager._perf_ledger_client.get_perf_ledger_for_hotkey.return_value = {
                hotkey: SimpleNamespace(get_checkpoint_at_time=lambda *_: checkpoint)}
            with patch("entity_management.entity_manager.read_all_snapshots", return_value=[]):
                result = entity_manager._calculate_payout_for_hotkey(
                    hotkey, 0, MONDAY_MS)
            payouts[subaccount_id] = [(row['start_ms'], row['payout']) for row in result['weekly_settlements']]

        self.assertEqual(payouts[0], payouts[1])
        self.assertEqual(sum(payout for _, payout in payouts[0]), 12_000.0)


if __name__ == '__main__':
    unittest.main()
