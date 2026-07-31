# developer: trdougherty, jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Integration tests for ChallengePeriodManager.

Organised into two tiers:

  Tier 1 — RPC round-trips (Sections 1-2)
    Uses the real spawned server via ChallengePeriodClient.
    Verifies that the RPC layer correctly serialises/deserialises all client calls
    and that checkpoint persistence round-trips survive the process boundary.

  Tier 2 — Manager logic (Sections 3-6)
    Creates a ChallengePeriodManager in-process with mocked sub-clients
    (the same pattern as the unit tests but with more elaborate scenarios).
    Focuses on multi-step state flows that would be tedious to exercise with
    only the exported RPC surface.

The two tiers are complementary:
  - Unit tests check individual static methods.
  - Tier-2 integration tests check multi-method flows (sync → refresh → eliminate).
  - Tier-1 integration tests check the RPC boundary and cross-process state.
"""
import contextlib
from types import SimpleNamespace
from unittest.mock import patch

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.challenge_period.challengeperiod_manager import (
    ChallengePeriodManager,
    DrawdownStats,
    MinerBucketState,
)
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePairCategory, ValiConfig
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger, PerfCheckpoint

# ── Constants ─────────────────────────────────────────────────────────────────

DAILY_MS = ValiConfig.DAILY_MS
MIN_CHALLENGE_MS = ValiConfig.CHALLENGE_PERIOD_MINIMUM_MS
MAX_CHALLENGE_MS = ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS
THRESHOLD = ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD_DEFAULT
RANK_LIMIT = ValiConfig.PROMOTION_THRESHOLD_RANK
INTRADAY_DD_PCT = ValiConfig.CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD * 100
EOD_DD_PCT = ValiConfig.CHALLENGE_EOD_DRAWDOWN_THRESHOLD * 100
STATIC_DD_PCT = ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD * 100
STATIC_EOD_DD_PCT = ValiConfig.SUBACCOUNT_STATIC_EOD_DRAWDOWN_THRESHOLD * 100

_CLIENT_PATHS = [
    "vali_objects.challenge_period.challengeperiod_manager.PerfLedgerClient",
    "vali_objects.challenge_period.challengeperiod_manager.PositionManagerClient",
    "vali_objects.challenge_period.challengeperiod_manager.EliminationClient",
    "vali_objects.challenge_period.challengeperiod_manager.PlagiarismClient",
    "vali_objects.challenge_period.challengeperiod_manager.MinerAccountClient",
    "vali_objects.challenge_period.challengeperiod_manager.CommonDataClient",
    "vali_objects.challenge_period.challengeperiod_manager.AssetSelectionClient",
    "vali_objects.challenge_period.challengeperiod_manager.DebtLedgerClient",
]


def _make_state(hotkey: str, bucket: MinerBucket, start_ms: int) -> MinerBucketState:
    return MinerBucketState(hotkey, [BucketEntry(bucket, start_ms)])


def _local_manager() -> ChallengePeriodManager:
    """Create an in-process manager with all RPC sub-clients mocked."""
    stack = contextlib.ExitStack()
    for path in _CLIENT_PATHS:
        stack.enter_context(patch(path))
    mgr = ChallengePeriodManager(is_backtesting=True)
    return mgr, stack


def _wire_refresh_clients(mgr, hk: str, now: int, elapsed_ms: int = 0):
    """Minimal stub wiring so refresh() doesn't crash on missing client calls."""
    mgr._position_client.filtered_positions_for_scoring.return_value = (
        {hk: []}, {hk: now - elapsed_ms}
    )
    mgr._position_client.get_all_hotkeys.return_value = [hk]
    mgr._position_client.get_positions_for_hotkeys.return_value = {hk: []}
    mgr._elimination_client.get_eliminated_hotkeys.return_value = []
    mgr._plagiarism_client.get_plagiarism_miners.return_value = []
    mgr._miner_account_client.get_accounts.return_value = {}
    mgr._perf_ledger_client.filtered_ledger_for_scoring.return_value = {}
    mgr._asset_selection_client.get_asset_selections.return_value = {
        hk: TradePairCategory.CRYPTO
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — RPC round-trips
# ═══════════════════════════════════════════════════════════════════════════════

class TestChallengePeriodRPC(TestBase):
    """
    RPC round-trip tests — all assertions go through the client → server boundary.

    These tests start the full server infrastructure once (shared with other test
    classes via the ServerOrchestrator singleton) and use clear_all_test_data()
    for per-test isolation.
    """

    orchestrator = None
    challenge_period_client = None

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.orchestrator.clear_all_test_data()

    def tearDown(self):
        self.orchestrator.clear_all_test_data()

    # ── Section 1: Bucket CRUD ────────────────────────────────────────────────

    def test_rpc_set_and_get_miner_bucket(self):
        """set_miner_bucket → get_miner_bucket serialises correctly over RPC."""
        now = TimeUtil.now_in_millis()
        self.challenge_period_client.set_miner_bucket("hk1", MinerBucket.CHALLENGE, now)

        self.assertTrue(self.challenge_period_client.has_miner("hk1"))
        self.assertEqual(self.challenge_period_client.get_miner_bucket("hk1"), MinerBucket.CHALLENGE)
        self.assertEqual(self.challenge_period_client.get_miner_start_time("hk1"), now)

    def test_rpc_has_miner_absent(self):
        self.assertFalse(self.challenge_period_client.has_miner("nonexistent"))

    def test_rpc_get_hotkeys_by_bucket_filters_correctly(self):
        now = TimeUtil.now_in_millis()
        self.challenge_period_client.set_miner_bucket("hk_challenge", MinerBucket.CHALLENGE, now)
        self.challenge_period_client.set_miner_bucket("hk_maincomp", MinerBucket.MAINCOMP, now)
        self.challenge_period_client.set_miner_bucket("hk_probation", MinerBucket.PROBATION, now)

        challenge_hks = self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE)
        self.assertIn("hk_challenge", challenge_hks)
        self.assertNotIn("hk_maincomp", challenge_hks)

        maincomp_hks = self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        self.assertIn("hk_maincomp", maincomp_hks)
        self.assertNotIn("hk_challenge", maincomp_hks)

    def test_rpc_get_testing_miners(self):
        now = TimeUtil.now_in_millis()
        self.challenge_period_client.set_miner_bucket("hk_c", MinerBucket.CHALLENGE, now)
        self.challenge_period_client.set_miner_bucket("hk_m", MinerBucket.MAINCOMP, now)

        testing = self.challenge_period_client.get_testing_miners()
        self.assertIn("hk_c", testing)
        self.assertNotIn("hk_m", testing)

    def test_rpc_get_success_miners(self):
        now = TimeUtil.now_in_millis()
        self.challenge_period_client.set_miner_bucket("hk_m", MinerBucket.MAINCOMP, now)
        self.challenge_period_client.set_miner_bucket("hk_c", MinerBucket.CHALLENGE, now)

        success = self.challenge_period_client.get_success_miners()
        self.assertIn("hk_m", success)
        self.assertNotIn("hk_c", success)

    def test_rpc_remove_miners(self):
        now = TimeUtil.now_in_millis()
        self.challenge_period_client.set_miner_bucket("hk1", MinerBucket.CHALLENGE, now)
        self.challenge_period_client.set_miner_bucket("hk2", MinerBucket.MAINCOMP, now)

        result = self.challenge_period_client.remove_miners(["hk1", "hk2"])
        self.assertTrue(result)
        self.assertFalse(self.challenge_period_client.has_miner("hk1"))
        self.assertFalse(self.challenge_period_client.has_miner("hk2"))

    def test_rpc_get_all_miner_hotkeys(self):
        now = TimeUtil.now_in_millis()
        for hk in ["a", "b", "c"]:
            self.challenge_period_client.set_miner_bucket(hk, MinerBucket.CHALLENGE, now)

        all_hks = self.challenge_period_client.get_all_miner_hotkeys()
        for hk in ["a", "b", "c"]:
            self.assertIn(hk, all_hks)

    def test_rpc_get_dashboard_returns_bucket_entry(self):
        now = TimeUtil.now_in_millis()
        self.challenge_period_client.set_miner_bucket("hk_dash", MinerBucket.CHALLENGE, now)

        dashboard = self.challenge_period_client.get_dashboard("hk_dash")
        self.assertIsNotNone(dashboard)
        self.assertEqual(dashboard["bucket"], MinerBucket.CHALLENGE.value)

    def test_rpc_get_dashboard_missing_returns_none(self):
        self.assertIsNone(self.challenge_period_client.get_dashboard("no_such_miner"))

    # ── Section 2: Checkpoint persistence ────────────────────────────────────

    def test_sync_challenge_period_data_round_trip(self):
        """sync_challenge_period_data → get_miner_bucket preserves bucket and start time."""
        now = TimeUtil.now_in_millis()
        states = {
            "hk_a": MinerBucketState("hk_a", [BucketEntry(MinerBucket.CHALLENGE, now)]).to_json(),
            "hk_b": MinerBucketState("hk_b", [BucketEntry(MinerBucket.MAINCOMP, now - DAILY_MS)]).to_json(),
        }
        self.challenge_period_client.sync_challenge_period_data(states)

        self.assertEqual(self.challenge_period_client.get_miner_bucket("hk_a"), MinerBucket.CHALLENGE)
        self.assertEqual(self.challenge_period_client.get_miner_bucket("hk_b"), MinerBucket.MAINCOMP)
        self.assertEqual(self.challenge_period_client.get_miner_start_time("hk_a"), now)

    def test_to_checkpoint_dict_captures_all_miners(self):
        """to_checkpoint_dict serialises every tracked miner and can be parsed back."""
        now = TimeUtil.now_in_millis()
        for hk, bucket in [("c1", MinerBucket.CHALLENGE), ("m1", MinerBucket.MAINCOMP), ("p1", MinerBucket.PROBATION)]:
            self.challenge_period_client.set_miner_bucket(hk, bucket, now)

        checkpoint = self.challenge_period_client.to_checkpoint_dict()
        parsed = ChallengePeriodManager.parse_checkpoint_dict(checkpoint)

        self.assertEqual(parsed["c1"].current_bucket, MinerBucket.CHALLENGE)
        self.assertEqual(parsed["m1"].current_bucket, MinerBucket.MAINCOMP)
        self.assertEqual(parsed["p1"].current_bucket, MinerBucket.PROBATION)

    def test_checkpoint_preserves_multi_entry_bucket_history(self):
        """Multi-entry bucket history survives the RPC boundary and parse round-trip."""
        now = TimeUtil.now_in_millis()
        state = MinerBucketState("hk_hist", [
            BucketEntry(MinerBucket.CHALLENGE, now - DAILY_MS * 70),
            BucketEntry(MinerBucket.MAINCOMP, now - DAILY_MS * 5),
        ])
        self.challenge_period_client.sync_challenge_period_data({"hk_hist": state.to_json()})

        checkpoint = self.challenge_period_client.to_checkpoint_dict()
        parsed = ChallengePeriodManager.parse_checkpoint_dict(checkpoint)

        self.assertEqual(len(parsed["hk_hist"].entries), 2)
        self.assertEqual(parsed["hk_hist"].entries[0].bucket, MinerBucket.CHALLENGE)
        self.assertEqual(parsed["hk_hist"].entries[1].bucket, MinerBucket.MAINCOMP)

    def test_clear_test_state_removes_all_miners(self):
        """clear_test_state wipes all miner state so tests don't leak into each other."""
        now = TimeUtil.now_in_millis()
        for hk in ["hk1", "hk2", "hk3"]:
            self.challenge_period_client.set_miner_bucket(hk, MinerBucket.CHALLENGE, now)

        self.challenge_period_client.clear_test_state()

        for hk in ["hk1", "hk2", "hk3"]:
            self.assertFalse(self.challenge_period_client.has_miner(hk))

    def test_orchestrator_clear_all_test_data_resets_challenge_period(self):
        """orchestrator.clear_all_test_data() resets the challenge period server."""
        now = TimeUtil.now_in_millis()
        self.challenge_period_client.set_miner_bucket("hk_orphan", MinerBucket.MAINCOMP, now)
        self.assertTrue(self.challenge_period_client.has_miner("hk_orphan"))

        self.orchestrator.clear_all_test_data()
        self.assertFalse(self.challenge_period_client.has_miner("hk_orphan"))


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Manager logic (in-process, mocked sub-clients)
# ═══════════════════════════════════════════════════════════════════════════════

class TestChallengePeriodManagerLogic(TestBase):
    """
    In-process integration tests for ChallengePeriodManager multi-step flows.

    A fresh manager with mocked RPC sub-clients is created per test.
    This tier covers scenarios that require calling multiple manager methods
    in sequence (e.g., sync → refresh → check state) and that cannot be
    driven through the exported RPC surface alone.
    """

    # No server infrastructure needed for these tests
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def _make_manager(self):
        stack = contextlib.ExitStack()
        for path in _CLIENT_PATHS:
            stack.enter_context(patch(path))
        mgr = ChallengePeriodManager(is_backtesting=True)
        return mgr, stack

    # ── Section 3: Sync method flows ─────────────────────────────────────────

    def test_sync_plagiarism_miners_adds_and_removes_bucket(self):
        """PLAGIARISM bucket is pushed on match and popped when cleared."""
        now = TimeUtil.now_in_millis()
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket("hk_plag", MinerBucket.CHALLENGE, now)

            mgr.sync_plagiarism_miners(["hk_plag"], now)
            self.assertEqual(mgr.get_miner_bucket("hk_plag"), MinerBucket.PLAGIARISM)

            # Cleared from plagiarism list → pops PLAGIARISM, back to CHALLENGE
            mgr.sync_plagiarism_miners([], now)
            self.assertEqual(mgr.get_miner_bucket("hk_plag"), MinerBucket.CHALLENGE)

    def test_sync_plagiarism_miners_only_affects_known_miners(self):
        """Unknown hotkeys in the plagiarism list are silently ignored."""
        now = TimeUtil.now_in_millis()
        mgr, stack = self._make_manager()
        with stack:
            changed = mgr.sync_plagiarism_miners(["unknown_hk"], now)
            self.assertFalse(changed)

    def test_sync_elimination_miners_removes_targeted_hotkey(self):
        """Eliminated miners are removed from state; unaffected miners remain."""
        now = TimeUtil.now_in_millis()
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket("hk_elim", MinerBucket.CHALLENGE, now)
            mgr.set_miner_bucket("hk_safe", MinerBucket.MAINCOMP, now)

            changed = mgr.sync_elimination_miners(["hk_elim"], now)
            self.assertTrue(changed)
            self.assertFalse(mgr.has_miner("hk_elim"))
            self.assertTrue(mgr.has_miner("hk_safe"))

    def test_sync_elimination_miners_empty_list_is_noop(self):
        now = TimeUtil.now_in_millis()
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket("hk1", MinerBucket.CHALLENGE, now)
            changed = mgr.sync_elimination_miners([], now)
            self.assertFalse(changed)
            self.assertTrue(mgr.has_miner("hk1"))

    def test_prune_hotkeys_no_positions_removes_challenge_only(self):
        """Miners absent from position data are pruned unless in ENTITY/SUBACCOUNT_FUNDED."""
        now = TimeUtil.now_in_millis()
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket("hk_gone", MinerBucket.CHALLENGE, now)
            mgr.set_miner_bucket("hk_entity", MinerBucket.ENTITY, now)
            mgr.set_miner_bucket("hk_funded", MinerBucket.SUBACCOUNT_FUNDED, now)
            mgr._position_client.get_all_hotkeys.return_value = []

            changed = mgr._prune_hotkeys_no_positions()
            self.assertTrue(changed)
            self.assertFalse(mgr.has_miner("hk_gone"))
            self.assertTrue(mgr.has_miner("hk_entity"))
            self.assertTrue(mgr.has_miner("hk_funded"))

    # ── Section 4: Refresh flows ──────────────────────────────────────────────

    def test_refresh_promotes_challenge_miner_after_minimum_time(self):
        """CHALLENGE miner with sufficient time and returns is promoted to MAINCOMP."""
        now = TimeUtil.now_in_millis()
        hk = "hk_promote"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - MIN_CHALLENGE_MS - DAILY_MS)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                current_equity=1.0 + THRESHOLD + 0.05,
                current_balance=1.0 + THRESHOLD + 0.05,
            )
            mgr.miner_states[hk].rank = 1
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=MIN_CHALLENGE_MS + DAILY_MS)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk'),
                patch.object(mgr, '_sync_buckets_to_accounts'),
            ):
                mgr.refresh(current_time_ms=now)

        self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.MAINCOMP)

    def test_refresh_does_not_promote_before_minimum_time(self):
        """CHALLENGE miner below minimum time is NOT promoted regardless of returns."""
        now = TimeUtil.now_in_millis()
        hk = "hk_early"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS)  # only 1 day old
            mgr.miner_states[hk].drawdown = DrawdownStats(
                current_equity=1.0 + THRESHOLD + 0.05,
                current_balance=1.0 + THRESHOLD + 0.05,
            )
            mgr.miner_states[hk].rank = 1
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk'),
                patch.object(mgr, '_sync_buckets_to_accounts'),
            ):
                mgr.refresh(current_time_ms=now)

        self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.CHALLENGE)

    def test_refresh_eliminates_time_expired_challenge_miner(self):
        """CHALLENGE miner past max duration is eliminated."""
        now = TimeUtil.now_in_millis()
        hk = "hk_timeout"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - MAX_CHALLENGE_MS - DAILY_MS)
            mgr.miner_states[hk].drawdown = DrawdownStats()
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=MAX_CHALLENGE_MS + DAILY_MS)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk'),
                patch.object(mgr, '_sync_buckets_to_accounts'),
            ):
                mgr.refresh(current_time_ms=now)

        self.assertFalse(mgr.has_miner(hk))

    def test_refresh_demotes_maincomp_miner_with_bad_rank(self):
        """MAINCOMP miner with rank beyond threshold is demoted to PROBATION."""
        now = TimeUtil.now_in_millis()
        hk = "hk_demote"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.MAINCOMP, now - DAILY_MS * 30)
            mgr.miner_states[hk].drawdown = DrawdownStats(current_equity=1.5, current_balance=1.5)
            mgr.miner_states[hk].rank = RANK_LIMIT + 1
            _wire_refresh_clients(mgr, hk, now)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk'),
                patch.object(mgr, '_sync_buckets_to_accounts'),
            ):
                mgr.refresh(current_time_ms=now)

        self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.PROBATION)

    def test_refresh_promotes_probation_miner_with_good_rank(self):
        """PROBATION miner with rank within threshold and sufficient returns is promoted to MAINCOMP."""
        now = TimeUtil.now_in_millis()
        hk = "hk_probation_promote"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.PROBATION, now - DAILY_MS * 5)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                current_equity=1.0 + THRESHOLD + 0.05,
                current_balance=1.0 + THRESHOLD + 0.05,
            )
            mgr.miner_states[hk].rank = RANK_LIMIT
            _wire_refresh_clients(mgr, hk, now)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk'),
                patch.object(mgr, '_sync_buckets_to_accounts'),
            ):
                mgr.refresh(current_time_ms=now)

        self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.MAINCOMP)

    def test_refresh_eliminates_intraday_drawdown(self):
        """Miner whose intraday drawdown exceeds the threshold is eliminated during refresh."""
        now = TimeUtil.now_in_millis()
        hk = "hk_intraday"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS * 3)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                intraday_drawdown_pct=INTRADAY_DD_PCT + 1.0,
                current_equity=1.0 - (INTRADAY_DD_PCT + 1.0) / 100,
                current_balance=1.0,
                daily_open_equity=1.0,
            )
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 3)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk'),
                patch.object(mgr, '_sync_buckets_to_accounts'),
            ):
                mgr.refresh(current_time_ms=now)

        self.assertFalse(mgr.has_miner(hk))

    def test_refresh_eliminates_eod_drawdown(self):
        """Miner whose EOD trailing drawdown exceeds the threshold is eliminated during refresh."""
        now = TimeUtil.now_in_millis()
        hk = "hk_eod"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS * 5)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                eod_drawdown_pct=EOD_DD_PCT + 1.0,
                eod_hwm=1.1,
                last_eod_equity=1.1 * (1 - (EOD_DD_PCT + 1.0) / 100),
            )
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 5)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk'),
                patch.object(mgr, '_sync_buckets_to_accounts'),
            ):
                mgr.refresh(current_time_ms=now)

        self.assertFalse(mgr.has_miner(hk))

    def test_refresh_healthy_miner_unchanged_no_disk_write(self):
        """Brand-new miner with no issues triggers no state change and no disk write."""
        now = TimeUtil.now_in_millis()
        hk = "hk_healthy"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS)
            mgr.miner_states[hk].drawdown = DrawdownStats()
            mgr.miner_states[hk].rank = 1
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk') as mock_save,
                patch.object(mgr, '_sync_buckets_to_accounts') as mock_sync,
            ):
                mgr.refresh(current_time_ms=now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.CHALLENGE)
            mock_save.assert_not_called()
            mock_sync.assert_not_called()

    def test_demotion_then_re_promotion_full_cycle(self):
        """MAINCOMP → PROBATION → MAINCOMP bucket progression works end-to-end."""
        now = TimeUtil.now_in_millis()
        hk = "hk_cycle"
        mgr, stack = self._make_manager()
        with stack:
            # Phase 1: MAINCOMP with bad rank → demote
            mgr.set_miner_bucket(hk, MinerBucket.MAINCOMP, now - DAILY_MS * 30)
            mgr.miner_states[hk].drawdown = DrawdownStats(current_equity=1.5, current_balance=1.5)
            mgr.miner_states[hk].rank = RANK_LIMIT + 5
            _wire_refresh_clients(mgr, hk, now)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk'),
                patch.object(mgr, '_sync_buckets_to_accounts'),
            ):
                mgr.refresh(current_time_ms=now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.PROBATION)

            # Phase 2: PROBATION with good rank → promote back
            mgr.miner_states[hk].drawdown = DrawdownStats(
                current_equity=1.0 + THRESHOLD + 0.05,
                current_balance=1.0 + THRESHOLD + 0.05,
            )
            mgr.miner_states[hk].rank = RANK_LIMIT - 5
            _wire_refresh_clients(mgr, hk, now)

            with (
                patch.object(mgr, '_refresh_drawdown_cache'),
                patch.object(mgr, '_refresh_rank_cache'),
                patch.object(mgr, '_save_to_disk'),
                patch.object(mgr, '_sync_buckets_to_accounts'),
            ):
                mgr.refresh(current_time_ms=now)

        self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.MAINCOMP)
        self.assertEqual(len(mgr.miner_states[hk].entries), 3)  # MAINCOMP → PROBATION → MAINCOMP

    # ── Section 5: Drawdown cache computation ────────────────────────────────

    def test_refresh_drawdown_cache_intraday_drop(self):
        """_refresh_drawdown_cache correctly computes intraday drawdown from account and ledger."""
        now = TimeUtil.now_in_millis()
        hk = "hk_dd_cache"
        account_size = 100_000.0

        today_midnight_ms = (now // 86400000) * 86400000
        yesterday_midnight_ms = today_midnight_ms - 86400000

        ledger = PerfLedger(cps=[
            PerfCheckpoint(last_update_ms=yesterday_midnight_ms, prev_portfolio_ret=1.0,
                           accum_ms=DAILY_MS, equity_ret=1.0, gain=0.0, loss=0.0, mdd=1.0),
            PerfCheckpoint(last_update_ms=today_midnight_ms, prev_portfolio_ret=1.0,
                           accum_ms=DAILY_MS, equity_ret=1.0, gain=0.0, loss=0.0, mdd=1.0),
        ])

        intraday_drop = 0.07
        accounts = {hk: {'account_size': account_size, 'balance': account_size * (1 - intraday_drop)}}
        positions = {hk: []}  # no open positions → equity = balance / account_size

        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS * 5)
            mgr._refresh_drawdown_cache([hk], accounts, {hk: ledger}, positions, now)

            dd = mgr.miner_states[hk].drawdown
            self.assertAlmostEqual(dd.intraday_drawdown_pct, intraday_drop * 100, delta=0.1)

    def test_refresh_drawdown_cache_eod_drop_from_hwm(self):
        """_refresh_drawdown_cache correctly computes EOD trailing drawdown from highest-ever EOD."""
        now = TimeUtil.now_in_millis()
        hk = "hk_eod_cache"
        account_size = 100_000.0

        today_midnight_ms = (now // 86400000) * 86400000
        # HWM was 1.10 five days ago, last EOD at 1.04 → expected EOD dd ≈ 5.45 %
        ledger = PerfLedger(cps=[
            PerfCheckpoint(last_update_ms=today_midnight_ms - 5 * 86400000, prev_portfolio_ret=1.0,
                           accum_ms=DAILY_MS, equity_ret=1.10, gain=0.1, loss=0.0, mdd=1.0),
            PerfCheckpoint(last_update_ms=today_midnight_ms - 86400000, prev_portfolio_ret=1.0,
                           accum_ms=DAILY_MS, equity_ret=1.04, gain=0.0, loss=-0.06, mdd=0.95),
            PerfCheckpoint(last_update_ms=today_midnight_ms, prev_portfolio_ret=1.0,
                           accum_ms=DAILY_MS, equity_ret=1.04, gain=0.0, loss=0.0, mdd=1.0),
        ])

        accounts = {hk: {'account_size': account_size, 'balance': account_size * 1.04}}
        positions = {hk: []}

        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS * 10)
            mgr._refresh_drawdown_cache([hk], accounts, {hk: ledger}, positions, now)

            dd = mgr.miner_states[hk].drawdown
            expected_eod_dd_pct = (1.0 - 1.04 / 1.10) * 100
            self.assertAlmostEqual(dd.eod_drawdown_pct, expected_eod_dd_pct, delta=0.1)

    def test_refresh_drawdown_cache_skips_missing_ledger(self):
        """Miner with no ledger is silently skipped; drawdown remains at default."""
        now = TimeUtil.now_in_millis()
        hk = "hk_no_ledger"
        account_size = 100_000.0
        accounts = {hk: {'account_size': account_size, 'balance': account_size}}

        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS * 3)
            default_dd = DrawdownStats()
            mgr._refresh_drawdown_cache([hk], accounts, {}, {hk: []}, now)

            dd = mgr.miner_states[hk].drawdown
            self.assertEqual(dd.intraday_drawdown_pct, default_dd.intraday_drawdown_pct)
            self.assertEqual(dd.eod_drawdown_pct, default_dd.eod_drawdown_pct)

    # ── Section 6: Subaccount static drawdown rules ───────────────────────────

    def _refresh_with_patched_caches(self, mgr, now):
        with (
            patch.object(mgr, '_refresh_drawdown_cache'),
            patch.object(mgr, '_refresh_rank_cache'),
            patch.object(mgr, '_save_to_disk'),
            patch.object(mgr, '_sync_buckets_to_accounts'),
        ):
            mgr.refresh(current_time_ms=now)

    def test_refresh_eliminates_subaccount_funded_static_drawdown(self):
        """SUBACCOUNT_FUNDED with balance more than 5% below starting balance is eliminated."""
        now = TimeUtil.now_in_millis()
        hk = "hk_static_funded"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.SUBACCOUNT_FUNDED, now - DAILY_MS * 3)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                current_balance=1.0 - (STATIC_DD_PCT + 1.0) / 100,
                static_drawdown_pct=STATIC_DD_PCT + 1.0,
            )
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 3)
            self._refresh_with_patched_caches(mgr, now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.ELIMINATED)
            kwargs = mgr._elimination_client.append_elimination_row.call_args.kwargs
            self.assertEqual(kwargs["reason"], EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN)
            self.assertAlmostEqual(kwargs["elimination_drawdown_pct"], STATIC_DD_PCT + 1.0)

    def test_refresh_eliminates_subaccount_challenge_static_drawdown(self):
        """SUBACCOUNT_CHALLENGE gets the challenge-period static reason."""
        now = TimeUtil.now_in_millis()
        hk = "hk_static_challenge"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.SUBACCOUNT_CHALLENGE, now - DAILY_MS * 3)
            mgr.miner_states[hk].drawdown = DrawdownStats(static_drawdown_pct=STATIC_DD_PCT + 1.0)
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 3)
            self._refresh_with_patched_caches(mgr, now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.ELIMINATED)
            kwargs = mgr._elimination_client.append_elimination_row.call_args.kwargs
            self.assertEqual(kwargs["reason"], EliminationReason.FAILED_CHALLENGE_PERIOD_STATIC_DRAWDOWN)

    def test_refresh_eliminates_subaccount_static_eod_drawdown(self):
        """SUBACCOUNT_FUNDED with midnight equity more than 5% below starting balance is
        eliminated at the EOD check, with the elimination timestamped at that midnight."""
        now = TimeUtil.now_in_millis()
        midnight_ms = (now // MS_IN_24_HOURS) * MS_IN_24_HOURS
        hk = "hk_static_eod"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.SUBACCOUNT_FUNDED, now - DAILY_MS * 3)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                last_eod_equity=1.0 - (STATIC_EOD_DD_PCT + 1.0) / 100,
                static_eod_drawdown_pct=STATIC_EOD_DD_PCT + 1.0,
                last_eod_checked_ms=midnight_ms,
            )
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 3)
            self._refresh_with_patched_caches(mgr, now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.ELIMINATED)
            kwargs = mgr._elimination_client.append_elimination_row.call_args.kwargs
            self.assertEqual(kwargs["reason"], EliminationReason.FAILED_FUNDED_PERIOD_STATIC_EOD_DRAWDOWN)
            self.assertAlmostEqual(kwargs["elimination_drawdown_pct"], STATIC_EOD_DD_PCT + 1.0)
            self.assertEqual(kwargs["elimination_time_ms"], midnight_ms)

    def test_refresh_subaccount_survives_deep_intraday_equity_dip(self):
        """Key behavior change: a deep unrealized intraday dip (would have died under the
        legacy intraday/EOD-trailing rules) no longer eliminates a subaccount as long as
        balance and midnight equity stay within the static limits."""
        now = TimeUtil.now_in_millis()
        hk = "hk_deep_dip"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.SUBACCOUNT_FUNDED, now - DAILY_MS * 3)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                current_equity=0.88,
                current_balance=0.99,
                intraday_drawdown_pct=12.0,
                eod_drawdown_pct=9.0,
                static_drawdown_pct=1.0,
                static_eod_drawdown_pct=2.0,
            )
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 3)
            self._refresh_with_patched_caches(mgr, now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.SUBACCOUNT_FUNDED)
            mgr._elimination_client.append_elimination_row.assert_not_called()

    def test_refresh_subaccount_survives_at_exact_static_thresholds(self):
        """Exactly 5% down on both static measures survives (strict > comparison)."""
        now = TimeUtil.now_in_millis()
        hk = "hk_boundary"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.SUBACCOUNT_FUNDED, now - DAILY_MS * 3)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                current_balance=1.0 - STATIC_DD_PCT / 100,
                last_eod_equity=1.0 - STATIC_EOD_DD_PCT / 100,
                static_drawdown_pct=STATIC_DD_PCT,
                static_eod_drawdown_pct=STATIC_EOD_DD_PCT,
            )
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 3)
            self._refresh_with_patched_caches(mgr, now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.SUBACCOUNT_FUNDED)
            mgr._elimination_client.append_elimination_row.assert_not_called()

    def test_refresh_regular_miner_ignores_static_stats(self):
        """Regular miners are never evaluated against the static subaccount rules."""
        now = ChallengePeriodManager.DRAWDOWN_ACTIVATION_MS + DAILY_MS
        hk = "hk_regular_static"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS * 3)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                static_drawdown_pct=50.0,
                static_eod_drawdown_pct=50.0,
            )
            mgr.miner_states[hk].rank = 1
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 3)
            self._refresh_with_patched_caches(mgr, now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.CHALLENGE)
            mgr._elimination_client.append_elimination_row.assert_not_called()

    def test_refresh_regular_miner_intraday_inactive_before_activation(self):
        """Before DRAWDOWN_ACTIVATION_MS a regular miner tripping the intraday rule is not eliminated."""
        now = ChallengePeriodManager.DRAWDOWN_ACTIVATION_MS - DAILY_MS
        hk = "hk_pre_activation"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS * 3)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                intraday_drawdown_pct=INTRADAY_DD_PCT + 1.0,
                daily_open_equity=1.0,
            )
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 3)
            self._refresh_with_patched_caches(mgr, now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.CHALLENGE)
            mgr._elimination_client.append_elimination_row.assert_not_called()

    def test_refresh_regular_miner_intraday_eliminates_after_activation(self):
        """After DRAWDOWN_ACTIVATION_MS the legacy intraday rule still eliminates regular miners."""
        now = ChallengePeriodManager.DRAWDOWN_ACTIVATION_MS + DAILY_MS
        hk = "hk_post_activation"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.CHALLENGE, now - DAILY_MS * 3)
            mgr.miner_states[hk].drawdown = DrawdownStats(
                intraday_drawdown_pct=INTRADAY_DD_PCT + 1.0,
                daily_open_equity=1.0,
            )
            _wire_refresh_clients(mgr, hk, now, elapsed_ms=DAILY_MS * 3)
            self._refresh_with_patched_caches(mgr, now)

            self.assertEqual(mgr.get_miner_bucket(hk), MinerBucket.ELIMINATED)
            kwargs = mgr._elimination_client.append_elimination_row.call_args.kwargs
            self.assertEqual(kwargs["reason"], EliminationReason.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN)

    def test_refresh_drawdown_cache_computes_static_pcts(self):
        """_refresh_drawdown_cache derives both static percentages from account balance
        and the latest midnight equity checkpoint."""
        now = TimeUtil.now_in_millis()
        hk = "hk_static_cache"
        account_size = 100_000.0
        today_midnight_ms = (now // MS_IN_24_HOURS) * MS_IN_24_HOURS

        ledger = PerfLedger(cps=[
            PerfCheckpoint(last_update_ms=today_midnight_ms, prev_portfolio_ret=1.0,
                           accum_ms=DAILY_MS, equity_ret=0.93, gain=0.0, loss=-0.07, mdd=1.0),
        ])
        account = SimpleNamespace(
            account_size=account_size,
            balance=account_size * 0.96,
            equity=account_size * 0.90,
            daily_open_snapshot=None,
        )

        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.SUBACCOUNT_FUNDED, now - DAILY_MS * 5)
            mgr._refresh_drawdown_cache([hk], {hk: account}, {hk: ledger}, {hk: []}, now)

            dd = mgr.miner_states[hk].drawdown
            self.assertAlmostEqual(dd.static_drawdown_pct, 4.0, delta=1e-6)
            self.assertAlmostEqual(dd.static_eod_drawdown_pct, 7.0, delta=1e-6)

    def test_get_drawdown_stats_includes_static_fields(self):
        """Dashboard payload carries the static percentages and thresholds."""
        now = TimeUtil.now_in_millis()
        hk = "hk_dd_stats"
        mgr, stack = self._make_manager()
        with stack:
            mgr.set_miner_bucket(hk, MinerBucket.SUBACCOUNT_FUNDED, now)
            mgr.miner_states[hk].drawdown = DrawdownStats(static_drawdown_pct=1.5, static_eod_drawdown_pct=2.5)

            stats = mgr.get_drawdown_stats(hk)
            self.assertEqual(stats["static_drawdown_pct"], 1.5)
            self.assertEqual(stats["static_eod_drawdown_pct"], 2.5)
            self.assertEqual(stats["static_drawdown_threshold"], ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD)
            self.assertEqual(stats["static_eod_drawdown_threshold"], ValiConfig.SUBACCOUNT_STATIC_EOD_DRAWDOWN_THRESHOLD)
