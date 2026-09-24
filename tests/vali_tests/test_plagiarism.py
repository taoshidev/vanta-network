# Copyright (c) 2024 Taoshi Inc
"""
Plagiarism tests, organised in the same two tiers as test_challengeperiod_integration.py.

Since the challenge-period/elimination refactor, plagiarism handling is split across
two services, so the tests are split the same way:

  Tier 1 — PlagiarismServer over RPC (TestPlagiarismService)
    The plagiarism service owns *who* is flagged. It computes newly-flagged and
    whitelisted hotkeys against the list the caller currently holds, and reports
    which flagged miners have outlived PLAGIARISM_REVIEW_PERIOD_MS.

  Tier 2 — Bucket transitions in-process (TestPlagiarismBucketTransitions)
    ChallengePeriodManager.sync_plagiarism_miners() applies that list to miner
    buckets by pushing a PLAGIARISM entry on demotion and popping it on whitelist,
    and _check_time() eliminates miners whose PLAGIARISM entry has expired.
    This is the path refresh() drives in production (challengeperiod_manager.py).
"""
import contextlib
import unittest
from unittest.mock import patch

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig

REVIEW_PERIOD_MS = ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS

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


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Plagiarism service over RPC
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlagiarismService(TestBase):
    """
    Plagiarism service tests using ServerOrchestrator for shared server infrastructure.

    Servers start once (via singleton orchestrator) and are shared across all test classes.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    metagraph_client = None
    challenge_period_client = None
    plagiarism_client = None

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.plagiarism_client = cls.orchestrator.get_client('plagiarism')

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No action needed.

        Servers and clients are managed by the ServerOrchestrator singleton and shared
        across all test classes. They are shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        self.orchestrator.clear_all_test_data()

        # Test miner hotkeys
        self.MINER_HOTKEY1 = "test_miner1"
        self.MINER_HOTKEY2 = "test_miner2"
        self.MINER_HOTKEY3 = "test_miner3"
        self.PLAGIARISM_HOTKEY = "plagiarism_miner"
        self.current_time = TimeUtil.now_in_millis()

        self.metagraph_client.set_hotkeys([
            self.MINER_HOTKEY1,
            self.MINER_HOTKEY2,
            self.MINER_HOTKEY3,
            self.PLAGIARISM_HOTKEY
        ])

        self.challenge_period_client.set_miner_bucket(self.MINER_HOTKEY1, MinerBucket.MAINCOMP, self.current_time)
        self.challenge_period_client.set_miner_bucket(self.MINER_HOTKEY2, MinerBucket.PROBATION, self.current_time)
        self.challenge_period_client.set_miner_bucket(self.MINER_HOTKEY3, MinerBucket.CHALLENGE, self.current_time)

        # PLAGIARISM is stacked on top of the miner's previous bucket, so seed both entries.
        self.challenge_period_client.set_miner_bucket(
            self.PLAGIARISM_HOTKEY, MinerBucket.PROBATION, self.current_time - REVIEW_PERIOD_MS
        )
        self.challenge_period_client.set_miner_bucket(
            self.PLAGIARISM_HOTKEY, MinerBucket.PLAGIARISM, self.current_time
        )

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def test_update_plagiarism_miners_new_plagiarists(self):
        """Miners newly flagged by the service are reported as new plagiarists."""
        self.plagiarism_client.set_plagiarism_miners_for_test(
            [self.MINER_HOTKEY1, self.MINER_HOTKEY2], self.current_time
        )

        new_plagiarists, whitelisted = self.plagiarism_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners=[]
        )

        self.assertEqual(sorted(new_plagiarists), sorted([self.MINER_HOTKEY1, self.MINER_HOTKEY2]))
        self.assertEqual(whitelisted, [])

    def test_update_plagiarism_miners_whitelisted_promotion(self):
        """A miner the service no longer flags is reported as whitelisted."""
        # Empty flag list = no longer a plagiarist
        self.plagiarism_client.set_plagiarism_miners_for_test([], self.current_time)

        new_plagiarists, whitelisted = self.plagiarism_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners=[self.PLAGIARISM_HOTKEY]
        )

        self.assertEqual(new_plagiarists, [])
        self.assertEqual(whitelisted, [self.PLAGIARISM_HOTKEY])

    def test_update_plagiarism_miners_no_change(self):
        """A miner already known to be flagged is neither new nor whitelisted."""
        self.plagiarism_client.set_plagiarism_miners_for_test([self.PLAGIARISM_HOTKEY], self.current_time)

        new_plagiarists, whitelisted = self.plagiarism_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners=[self.PLAGIARISM_HOTKEY]
        )

        self.assertEqual(new_plagiarists, [])
        self.assertEqual(whitelisted, [])

    def test_plagiarism_miners_to_eliminate(self):
        """Miners flagged for longer than the review period are eligible for elimination."""
        old_time = self.current_time - REVIEW_PERIOD_MS - 1000
        self.plagiarism_client.set_plagiarism_miners_for_test([self.PLAGIARISM_HOTKEY], old_time)

        result = self.plagiarism_client.plagiarism_miners_to_eliminate(self.current_time)

        self.assertEqual(result, {self.PLAGIARISM_HOTKEY: self.current_time})

    def test_plagiarism_miners_to_eliminate_within_review_period(self):
        """Miners flagged inside the review period are not yet eligible for elimination."""
        recent_time = self.current_time - REVIEW_PERIOD_MS + 1000
        self.plagiarism_client.set_plagiarism_miners_for_test([self.PLAGIARISM_HOTKEY], recent_time)

        result = self.plagiarism_client.plagiarism_miners_to_eliminate(self.current_time)

        self.assertEqual(result, {})

    def test_get_plagiarism_miners_round_trip(self):
        """Hotkeys injected into the service are readable back through the client."""
        self.plagiarism_client.set_plagiarism_miners_for_test(
            [self.MINER_HOTKEY1, self.PLAGIARISM_HOTKEY], self.current_time
        )

        flagged = self.plagiarism_client.get_plagiarism_miners()

        self.assertEqual(sorted(flagged), sorted([self.MINER_HOTKEY1, self.PLAGIARISM_HOTKEY]))

    def test_slack_notifications_disabled_during_tests(self):
        """Notification calls are no-ops when the server runs with running_unit_tests=True."""
        # Each returns early inside the server; reaching the end means no Slack call was attempted.
        self.plagiarism_client.send_plagiarism_demotion_notification(self.MINER_HOTKEY1)
        self.plagiarism_client.send_plagiarism_promotion_notification(self.MINER_HOTKEY1)
        self.plagiarism_client.send_plagiarism_elimination_notification(self.MINER_HOTKEY1)

    def test_get_bucket_methods(self):
        """Test helper methods for getting miners by bucket"""
        plagiarism_miners = self.challenge_period_client.get_plagiarism_miners()
        expected_plagiarism = {self.PLAGIARISM_HOTKEY: self.current_time}
        self.assertEqual(plagiarism_miners, expected_plagiarism)

        maincomp_miners = self.challenge_period_client.get_success_miners()
        expected_maincomp = {self.MINER_HOTKEY1: self.current_time}
        self.assertEqual(maincomp_miners, expected_maincomp)

        probation_miners = self.challenge_period_client.get_probation_miners()
        expected_probation = {self.MINER_HOTKEY2: self.current_time}
        self.assertEqual(probation_miners, expected_probation)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Bucket transitions (in-process, mocked sub-clients)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlagiarismBucketTransitions(TestBase):
    """
    In-process tests for the challenge-period side of plagiarism handling.

    sync_plagiarism_miners() is not on the exported RPC surface — refresh() calls it
    internally with the hotkeys read from the plagiarism service — so these drive a
    manager with mocked sub-clients, the same pattern as TestChallengePeriodManagerLogic.
    """

    # No server infrastructure needed for these tests
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.MINER_HOTKEY1 = "test_miner1"
        self.MINER_HOTKEY2 = "test_miner2"
        self.MINER_HOTKEY3 = "test_miner3"
        self.PLAGIARISM_HOTKEY = "plagiarism_miner"
        self.current_time = TimeUtil.now_in_millis()

    def _make_manager(self):
        stack = contextlib.ExitStack()
        for path in _CLIENT_PATHS:
            stack.enter_context(patch(path))
        mgr = ChallengePeriodManager(is_backtesting=True)
        return mgr, stack

    def _seed_miners(self, mgr):
        """MAINCOMP / PROBATION / CHALLENGE miners plus one already in PLAGIARISM."""
        mgr.set_miner_bucket(self.MINER_HOTKEY1, MinerBucket.MAINCOMP, self.current_time)
        mgr.set_miner_bucket(self.MINER_HOTKEY2, MinerBucket.PROBATION, self.current_time)
        mgr.set_miner_bucket(self.MINER_HOTKEY3, MinerBucket.CHALLENGE, self.current_time)
        mgr.set_miner_bucket(self.PLAGIARISM_HOTKEY, MinerBucket.PROBATION, self.current_time - REVIEW_PERIOD_MS)
        mgr.set_miner_bucket(self.PLAGIARISM_HOTKEY, MinerBucket.PLAGIARISM, self.current_time)

    def test_demotion_from_multiple_buckets(self):
        """Newly flagged miners are demoted to PLAGIARISM regardless of their current bucket."""
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)
            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.MAINCOMP)
            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PROBATION)

            mgr.sync_plagiarism_miners(
                [self.MINER_HOTKEY1, self.MINER_HOTKEY2, self.PLAGIARISM_HOTKEY], self.current_time
            )

            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.PLAGIARISM)
            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PLAGIARISM)

    def test_demotion_stamps_current_time(self):
        """Demotion starts the PLAGIARISM entry at the sync time."""
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)
            demotion_time = self.current_time + 5000

            mgr.sync_plagiarism_miners(
                [self.MINER_HOTKEY1, self.MINER_HOTKEY2, self.PLAGIARISM_HOTKEY], demotion_time
            )

            self.assertEqual(mgr.get_miner_start_time(self.MINER_HOTKEY1), demotion_time)
            self.assertEqual(mgr.get_miner_start_time(self.MINER_HOTKEY2), demotion_time)

    def test_promotion_restores_previous_bucket_and_start_time(self):
        """Whitelisting pops the PLAGIARISM entry, restoring the prior bucket and its original start time."""
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)
            self.assertEqual(mgr.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PLAGIARISM)
            self.assertEqual(mgr.get_miner_start_time(self.PLAGIARISM_HOTKEY), self.current_time)

            mgr.sync_plagiarism_miners([], self.current_time)

            self.assertEqual(mgr.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PROBATION)
            # The popped entry reveals the original PROBATION start, not the whitelist time.
            self.assertEqual(
                mgr.get_miner_start_time(self.PLAGIARISM_HOTKEY), self.current_time - REVIEW_PERIOD_MS
            )

    def test_no_flags_leaves_other_buckets_unchanged(self):
        """An empty flag list demotes nobody."""
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)

            mgr.sync_plagiarism_miners([], self.current_time)

            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.MAINCOMP)
            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PROBATION)
            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.CHALLENGE)

    def test_still_flagged_miner_is_not_restacked(self):
        """Re-syncing a miner who is already in PLAGIARISM is a no-op, not a second entry."""
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)

            changed = mgr.sync_plagiarism_miners([self.PLAGIARISM_HOTKEY], self.current_time + 5000)

            self.assertFalse(changed)
            self.assertEqual(mgr.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PLAGIARISM)
            # Start time unchanged, so the review-period clock is not reset by a repeat flag.
            self.assertEqual(mgr.get_miner_start_time(self.PLAGIARISM_HOTKEY), self.current_time)

    def test_sync_ignores_unknown_hotkeys(self):
        """A flagged hotkey the challenge period has never seen is silently ignored."""
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)

            changed = mgr.sync_plagiarism_miners(
                ["non_existant", self.PLAGIARISM_HOTKEY], self.current_time
            )

            self.assertFalse(changed)
            self.assertFalse(mgr.has_miner("non_existant"))
            self.assertIsNone(mgr.get_miner_bucket("non_existant"))

    def test_whitelisting_unknown_hotkey_is_safe(self):
        """Whitelisting a hotkey that is not in the challenge period does not raise.

        This can happen if an already-eliminated miner is dropped from the plagiarism
        service's list.
        """
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)

            mgr.sync_plagiarism_miners(["non_existant"], self.current_time)

            self.assertIsNone(mgr.get_miner_bucket("non_existant"))
            # The real plagiarism miner is whitelisted by its absence from the list.
            self.assertEqual(mgr.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PROBATION)

    def test_expired_review_period_yields_plagiarism_elimination(self):
        """A PLAGIARISM entry older than the review period elimination-checks as PLAGIARISM."""
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)
            expired_time = self.current_time + REVIEW_PERIOD_MS + 1000

            reason = mgr._check_time(mgr.miner_states[self.PLAGIARISM_HOTKEY], expired_time)

            self.assertEqual(reason, EliminationReason.PLAGIARISM)

    def test_unexpired_review_period_is_not_eliminated(self):
        """A PLAGIARISM entry inside the review period is not eliminated."""
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)
            within_review = self.current_time + REVIEW_PERIOD_MS - 1000

            reason = mgr._check_time(mgr.miner_states[self.PLAGIARISM_HOTKEY], within_review)

            self.assertIsNone(reason)

    def test_integration_full_plagiarism_flow(self):
        """Complete plagiarism flow: demotion -> promotion -> demotion -> elimination"""
        mgr, stack = self._make_manager()
        with stack:
            self._seed_miners(mgr)

            # Step 1: Demotion (new plagiarist detected)
            mgr.sync_plagiarism_miners([self.MINER_HOTKEY3, self.PLAGIARISM_HOTKEY], self.current_time)
            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.PLAGIARISM)

            # Step 2: Promotion (plagiarist is whitelisted) back to its original bucket
            mgr.sync_plagiarism_miners([self.PLAGIARISM_HOTKEY], self.current_time)
            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.CHALLENGE)

            # Step 3: Demote back to plagiarism for the elimination leg
            flag_time = self.current_time + 1000
            mgr.sync_plagiarism_miners([self.MINER_HOTKEY3, self.PLAGIARISM_HOTKEY], flag_time)
            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.PLAGIARISM)

            # Step 4: Review period expires -> elimination
            expired_time = flag_time + REVIEW_PERIOD_MS + 1000
            reason = mgr._check_time(mgr.miner_states[self.MINER_HOTKEY3], expired_time)
            self.assertEqual(reason, EliminationReason.PLAGIARISM)

            mgr.eliminate_hotkeys({self.MINER_HOTKEY3: reason}, expired_time)

            self.assertTrue(mgr.miner_states[self.MINER_HOTKEY3].is_eliminated)
            self.assertEqual(mgr.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.ELIMINATED)
            mgr._elimination_client.append_elimination_row.assert_called_once()


if __name__ == '__main__':
    unittest.main()
