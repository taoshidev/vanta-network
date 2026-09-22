"""
The admin-set pro account size.

There is no network default and no fixed minimum: the admin sets the size when offering the pro track, and
the network enforces only that it is numeric, finite, positive, at most ValiConfig.MAX_PRO_ACCOUNT_SIZE, and
never below the subaccount's own standard account size.

Covers:
  * ValiConfig: MAX_PRO_ACCOUNT_SIZE is $1M, there is no MIN_PRO_ACCOUNT_SIZE, and the old network default
    PRO_ACCOUNT_SIZE is gone.
  * pro_account_size_error: the shared check behind every entry point (cap, positivity, the standard-size
    floor when the caller knows it, NaN / Infinity, bool, str).
  * EntityManager.apply_bucket_account_size: a size is required to enter the pro track, including a re-offer
    after a demotion; a size below the subaccount's standard account size is refused; a move within the track
    uses an explicit size or the recorded one; PRO_FUNDED can re-set the size; a standard target never records
    a size; a rejected or failed move changes nothing; the log line names the size source.

This file also owns the shared pro fixtures (_bare_manager, _add_standard / _add_pro / _add_demoted,
INVALID_SIZES) that test_challengeperiod_pro.py and test_pro_endpoints.py build on. The bucket moves that
call apply_bucket_account_size are in test_challengeperiod_pro.py; the payloads that report the granted size
are in test_pro_endpoints.py.

EntityManager is built with object.__new__ and only the attributes these methods touch, so no RPC servers
are started.
"""
import json
import threading
import unittest
from unittest.mock import MagicMock, patch

import vali_objects.vali_config as vali_config_module
from entity_management.entity_manager import EntityData, EntityManager, SubaccountInfo
from entity_management.entity_utils import pro_account_size_error, pro_payout_scale
from vali_objects.enums.account_type_enum import AccountType
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig

NOW_MS = 1_748_000_000_000
ENTITY_HOTKEY = "entity_alpha"
STANDARD_SIZE = 100_000.0
GRANTED_SIZE = 500_000.0
PAYOUT_MULTIPLIER = ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER
REQUIRED = "pro_account_size is required to enter the pro track"

PRO_BUCKETS = (MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
               MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_FUNDED)
STANDARD_BUCKETS = (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA)

# Over the cap or non-positive, NaN / Infinity (which json.loads, so Flask's get_json, accepts), and wrong
# types. Each is (size, fragment of the rejection message). These are refused on shape alone, so every entry
# point rejects them whether or not it can see the subaccount's standard account size.
OUT_OF_RANGE = "must be positive and at most"
BELOW_STANDARD = "is below the subaccount's standard account size"
NOT_FINITE = "must be a finite number"
NOT_A_NUMBER = "must be a number"
INVALID_SIZES = (
    (1_000_001, OUT_OF_RANGE),
    (1_000_000.01, OUT_OF_RANGE),
    (0, OUT_OF_RANGE),
    (-500_000, OUT_OF_RANGE),
    (10 ** 400, OUT_OF_RANGE),
    (float("nan"), NOT_FINITE),
    (float("inf"), NOT_FINITE),
    (float("-inf"), NOT_FINITE),
    (True, NOT_A_NUMBER),
    (False, NOT_A_NUMBER),
    ("500000", NOT_A_NUMBER),
    ([500_000], NOT_A_NUMBER),
)


def _bare_manager():
    """An EntityManager with in-memory entities, a mocked account client, and no disk writes."""
    manager = object.__new__(EntityManager)
    manager.is_backtesting = True
    manager.running_unit_tests = True
    manager.entities = {}
    manager._entities_lock = threading.RLock()
    manager._entity_locks = {}
    manager._miner_account_client = MagicMock()
    manager._miner_account_client.set_miner_account_size.return_value = {"account_size": "set"}
    manager._miner_account_client.get_account.return_value = None
    manager._challenge_period_client = MagicMock()
    manager._challenge_period_client.has_miner.return_value = False
    manager._challenge_period_client.get_drawdown_stats.return_value = None
    manager._debt_ledger_client = MagicMock()
    manager._debt_ledger_client.get_ledger.return_value = None
    manager._position_client = MagicMock()
    manager._position_client.get_positions_for_one_hotkey.return_value = []
    manager._limit_order_client = MagicMock()
    manager._limit_order_client.to_dashboard_dict.return_value = None
    manager._statistics_client = MagicMock()
    manager._statistics_client.get_miner_statistics_for_hotkey.return_value = None
    manager._elimination_client = MagicMock()
    manager._elimination_client.get_elimination.return_value = None
    manager._asset_selection_client = MagicMock()
    manager._asset_selection_client.process_asset_selection_request.return_value = {
        "successfully_processed": True
    }
    return manager


def _add_standard(manager, subaccount_id=0, account_size=STANDARD_SIZE):
    info = SubaccountInfo(
        subaccount_id=subaccount_id,
        subaccount_uuid=f"uuid-{subaccount_id}",
        synthetic_hotkey=f"{ENTITY_HOTKEY}_{subaccount_id}",
        created_at_ms=NOW_MS,
        account_size=account_size,
        asset_class="forex",
    )
    entity = manager.entities.setdefault(
        ENTITY_HOTKEY, EntityData(entity_hotkey=ENTITY_HOTKEY, registered_at_ms=NOW_MS)
    )
    entity.subaccounts[subaccount_id] = info
    return info.synthetic_hotkey


def _add_pro(manager, subaccount_id=1, pro_size=GRANTED_SIZE):
    """A subaccount already on the pro track, trading a pro account of pro_size."""
    hotkey = _add_standard(manager, subaccount_id)
    info = manager.get_subaccount_info_for_synthetic(hotkey)
    info.standard_account_size = STANDARD_SIZE
    info.pro_account_size = pro_size
    info.account_size = pro_size
    info.account_type = AccountType.PRO.value
    return hotkey


def _add_demoted(manager, subaccount_id=2, pro_size=GRANTED_SIZE):
    """A subaccount demoted back to the standard track: standard again, but still carrying the pro size
    recorded on its earlier pro journey (a standard bucket never clears it)."""
    hotkey = _add_pro(manager, subaccount_id, pro_size)
    info = manager.get_subaccount_info_for_synthetic(hotkey)
    info.account_size = STANDARD_SIZE
    info.account_type = AccountType.STANDARD.value
    return hotkey


# ═══════════════════════════════════════════════════════════════════════════════
# ValiConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestProAccountSizeConfig(unittest.TestCase):

    def test_the_cap_is_1m(self):
        self.assertEqual(ValiConfig.MAX_PRO_ACCOUNT_SIZE, 1_000_000)

    def test_there_is_no_network_default_size(self):
        self.assertFalse(hasattr(ValiConfig, "PRO_ACCOUNT_SIZE"))

    def test_there_is_no_fixed_minimum_size(self):
        """The floor is the subaccount's own standard account size, not a network-wide constant."""
        self.assertFalse(hasattr(ValiConfig, "MIN_PRO_ACCOUNT_SIZE"))
        with open(vali_config_module.__file__) as f:
            self.assertNotIn("MIN_PRO_ACCOUNT_SIZE", f.read())

    def test_the_promotion_fee_is_never_negative(self):
        """A pro size at or below the standard size grants no dollars, so nothing is owed at the
        registration rate and the fee is the premium alone - never a credit back."""
        premium = (ValiConfig.PRO_PROMOTION_PREMIUM_RATE
                   * ValiConfig.PRO_FUNDED_EOD_DRAWDOWN_THRESHOLD * STANDARD_SIZE
                   / ValiConfig.THETA_USD_PRICE)
        for pro_size in (STANDARD_SIZE, STANDARD_SIZE - 1, 1.0, 0.0):
            with self.subTest(pro_account_size=pro_size):
                fee = ValiConfig.pro_promotion_fee_theta(pro_size, STANDARD_SIZE)
                self.assertGreaterEqual(fee, 0.0)
                self.assertAlmostEqual(fee, premium)
        self.assertGreaterEqual(ValiConfig.pro_promotion_fee_theta(0.0, 0.0), 0.0)


class TestAdoptProSizing(unittest.TestCase):
    """A promotion runs on one validator; the sizing every validator builds the payout scale from
    has to reach the rest. Both merge paths (checkpoint sync, broadcast) call this."""

    def _pair(self):
        manager = _bare_manager()
        local = manager.get_subaccount_info_for_synthetic(_add_standard(manager))
        promoted = manager.get_subaccount_info_for_synthetic(_add_pro(manager))
        return local, promoted

    def test_a_promotion_is_adopted(self):
        local, promoted = self._pair()
        self.assertTrue(EntityManager.adopt_pro_sizing(local, promoted))
        self.assertEqual(local.pro_account_size, GRANTED_SIZE)
        self.assertEqual(local.standard_account_size, STANDARD_SIZE)
        self.assertEqual(local.account_size, GRANTED_SIZE)
        self.assertEqual(local.account_type, AccountType.PRO.value)
        self.assertAlmostEqual(pro_payout_scale(local.standard_account_size, local.pro_account_size),
                               PAYOUT_MULTIPLIER * STANDARD_SIZE / GRANTED_SIZE)
        # idempotent: a re-sync of the same record is not a change
        self.assertFalse(EntityManager.adopt_pro_sizing(local, promoted))

    def test_a_stale_checkpoint_cannot_undo_a_promotion(self):
        """Auto-sync snapshots can predate the promotion, so they fill the gap but never downgrade.
        account_size and account_type are never None, so a field-by-field merge would half-revert."""
        stale, promoted = self._pair()
        self.assertFalse(EntityManager.adopt_pro_sizing(promoted, stale))
        self.assertEqual(promoted.pro_account_size, GRANTED_SIZE)
        self.assertEqual(promoted.account_size, GRANTED_SIZE)
        self.assertEqual(promoted.account_type, AccountType.PRO.value)

    def test_a_broadcast_reverts_a_rolled_back_promotion(self):
        """The broadcast is the validator that just made the change, so it is allowed to clear."""
        rolled_back, promoted = self._pair()
        self.assertTrue(EntityManager.adopt_pro_sizing(promoted, rolled_back, allow_clear=True))
        self.assertIsNone(promoted.pro_account_size)
        self.assertIsNone(promoted.standard_account_size)
        self.assertEqual(promoted.account_size, STANDARD_SIZE)
        self.assertEqual(promoted.account_type, AccountType.STANDARD.value)
        self.assertEqual(pro_payout_scale(promoted.standard_account_size, promoted.pro_account_size), 1.0)


class TestProPayoutScale(unittest.TestCase):
    """PnL on the pro account pays the standard account, uplifted by the transition multiplier."""

    def test_scaled_to_a_multiple_of_the_standard_account(self):
        # $5K on a $500K pro account pays a $100K standard account 2 * (5/500) * 100K = $2K
        self.assertAlmostEqual(pro_payout_scale(STANDARD_SIZE, GRANTED_SIZE),
                               PAYOUT_MULTIPLIER * STANDARD_SIZE / GRANTED_SIZE)
        self.assertAlmostEqual(5_000 * pro_payout_scale(STANDARD_SIZE, GRANTED_SIZE), 2_000)

    def test_a_subaccount_off_the_pro_track_is_unscaled(self):
        for standard, pro in ((None, GRANTED_SIZE), (STANDARD_SIZE, None), (None, None), (0, 0)):
            with self.subTest(standard=standard, pro=pro):
                self.assertEqual(pro_payout_scale(standard, pro), 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# pro_account_size_error
# ═══════════════════════════════════════════════════════════════════════════════

class TestProAccountSizeError(unittest.TestCase):

    def test_accepts_any_positive_size_up_to_the_cap(self):
        for size in (0.01, 1, 50_000, 200_000, 250_000, 500_000.5, 999_999.99, 1_000_000, 1_000_000.0):
            with self.subTest(size=size):
                self.assertIsNone(pro_account_size_error(size))

    def test_the_standard_account_size_is_the_floor_when_the_caller_knows_it(self):
        """A promotion grants size; it never shrinks the account the subaccount already trades."""
        for size in (STANDARD_SIZE - 0.01, STANDARD_SIZE / 2, 1):
            with self.subTest(size=size):
                error = pro_account_size_error(size, STANDARD_SIZE)
                self.assertIn(BELOW_STANDARD, error)
                self.assertIn("pro_account_size", error)
        for size in (STANDARD_SIZE, STANDARD_SIZE + 0.01, GRANTED_SIZE, 1_000_000):
            with self.subTest(size=size):
                self.assertIsNone(pro_account_size_error(size, STANDARD_SIZE))

    def test_no_floor_is_applied_when_the_standard_size_is_unknown(self):
        self.assertIsNone(pro_account_size_error(1))
        self.assertIsNone(pro_account_size_error(1, None))

    def test_the_cap_still_wins_over_the_floor(self):
        """An oversized request is refused on the cap even when it clears the standard account size."""
        self.assertIn(OUT_OF_RANGE, pro_account_size_error(1_000_001, STANDARD_SIZE))

    def test_rejects_out_of_range_non_finite_and_non_numeric(self):
        for size, reason in INVALID_SIZES + ((None, NOT_A_NUMBER), ({"size": 500_000}, NOT_A_NUMBER)):
            with self.subTest(size=size):
                error = pro_account_size_error(size)
                self.assertIsNotNone(error)
                self.assertIn(reason, error)
                self.assertIn("pro_account_size", error)

    def test_the_json_literals_flask_accepts_are_rejected(self):
        """json.loads (and so Flask's request.get_json) turns these literals into non-finite floats, and NaN
        passes a plain "< MIN" / "> MAX" range check."""
        for literal in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(literal=literal):
                value = json.loads(literal)
                self.assertIsInstance(value, float)
                self.assertIn(NOT_FINITE, pro_account_size_error(value))

    def test_the_cap_is_read_from_config_at_call_time(self):
        with patch.object(ValiConfig, "MAX_PRO_ACCOUNT_SIZE", 300_000):
            self.assertIn(OUT_OF_RANGE, pro_account_size_error(300_001))
            self.assertIsNone(pro_account_size_error(300_000))


# ═══════════════════════════════════════════════════════════════════════════════
# EntityManager.apply_bucket_account_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyBucketAccountSize(unittest.TestCase):

    def setUp(self):
        self.manager = _bare_manager()
        self.standard = _add_standard(self.manager)
        self.set_size = self.manager._miner_account_client.set_miner_account_size

    def _info(self, hotkey):
        return self.manager.get_subaccount_info_for_synthetic(hotkey)

    def _snapshot(self, hotkey):
        return self._info(hotkey).model_dump()

    def _assert_rejected_unchanged(self, hotkey, bucket, pro_account_size, reason):
        before = self._snapshot(hotkey)
        self.set_size.reset_mock()
        with patch.object(self.manager, "_write_entities_from_memory_to_disk") as write:
            success, message = self.manager.apply_bucket_account_size(hotkey, bucket, pro_account_size)
        self.assertFalse(success, message)
        self.assertIn(reason, message)
        self.assertEqual(self._snapshot(hotkey), before)
        self.set_size.assert_not_called()
        write.assert_not_called()

    def test_an_unknown_subaccount_pays_nothing_rather_than_at_the_best_ratio(self):
        """The scale multiplies real money and 1.0 is the most generous value it can take, so an
        account we know nothing about must fail closed."""
        self.assertEqual(self.manager.get_payout_scale("entity_does_not_exist"), 0.0)
        # A known standard subaccount is unaffected - it is still paid unscaled
        self.assertEqual(self.manager.get_payout_scale(self.standard), 1.0)

    # ==================== entering the pro track ====================

    def test_entering_the_track_requires_a_size(self):
        for bucket in PRO_BUCKETS:
            with self.subTest(bucket=bucket):
                self._assert_rejected_unchanged(self.standard, bucket, None, REQUIRED)
        self.assertEqual(self.manager.get_payout_scale(self.standard), 1.0)

    def test_transition_with_a_size_records_it_and_keeps_the_standard_account(self):
        success, message = self.manager.apply_bucket_account_size(
            self.standard, MinerBucket.PRO_CHALLENGE_TRANSITION, GRANTED_SIZE
        )

        self.assertTrue(success, message)
        info = self._info(self.standard)
        self.assertEqual(info.pro_account_size, GRANTED_SIZE)
        self.assertEqual(info.standard_account_size, STANDARD_SIZE)
        self.assertEqual(info.account_type, AccountType.PRO.value)
        self.assertEqual(info.account_size, STANDARD_SIZE)
        self.set_size.assert_not_called()

    def test_range_bounds_are_inclusive(self):
        """The floor is the subaccount's own standard size and the cap is MAX_PRO_ACCOUNT_SIZE; both inclusive."""
        for size, reason in ((STANDARD_SIZE - 1, BELOW_STANDARD), (STANDARD_SIZE, None), (200_000, None),
                             (1_000_000, None), (1_000_001, OUT_OF_RANGE)):
            with self.subTest(pro_account_size=size):
                manager = self.manager = _bare_manager()
                self.set_size = manager._miner_account_client.set_miner_account_size
                hotkey = _add_standard(manager)
                if reason is not None:
                    self._assert_rejected_unchanged(hotkey, MinerBucket.PRO_CHALLENGE_DIRECT, size, reason)
                    continue
                success, message = manager.apply_bucket_account_size(hotkey, MinerBucket.PRO_CHALLENGE_DIRECT, size)
                self.assertTrue(success, message)
                info = self._info(hotkey)
                self.assertEqual(info.pro_account_size, size)
                self.assertEqual(info.account_size, size)
                if size == STANDARD_SIZE:
                    # Already trading that size, so there is nothing to resize
                    self.set_size.assert_not_called()
                    continue
                self.assertEqual(self.set_size.call_args.kwargs["account_size"], size)
                self.assertEqual(self.set_size.call_args.kwargs["collateral_balance_theta"],
                                 size / ValiConfig.ENTITY_COST_PER_THETA)

    def test_invalid_explicit_sizes_are_rejected_for_every_target_and_change_nothing(self):
        pro = _add_pro(self.manager)
        for hotkey in (self.standard, pro):
            for bucket in PRO_BUCKETS + STANDARD_BUCKETS:
                for size, reason in INVALID_SIZES:
                    with self.subTest(hotkey=hotkey, bucket=bucket, pro_account_size=size):
                        self._assert_rejected_unchanged(hotkey, bucket, size, reason)
        self.assertIsNone(self._info(self.standard).pro_account_size)
        self.assertEqual(self._info(pro).pro_account_size, GRANTED_SIZE)

    def test_a_size_below_the_standard_account_is_rejected_for_every_target(self):
        """Promoting grants size. A subaccount trading $100K cannot be put on a smaller pro account,
        whether the standard size is the one it trades now or the one snapshotted on entry."""
        pro = _add_pro(self.manager)
        for hotkey in (self.standard, pro):
            for bucket in PRO_BUCKETS + STANDARD_BUCKETS:
                for size in (STANDARD_SIZE - 0.01, STANDARD_SIZE / 2, 1):
                    with self.subTest(hotkey=hotkey, bucket=bucket, pro_account_size=size):
                        self._assert_rejected_unchanged(hotkey, bucket, size, BELOW_STANDARD)
        self.assertIsNone(self._info(self.standard).pro_account_size)
        self.assertEqual(self._info(pro).pro_account_size, GRANTED_SIZE)

    def test_a_pro_size_equal_to_the_standard_account_is_allowed(self):
        success, message = self.manager.apply_bucket_account_size(
            self.standard, MinerBucket.PRO_CHALLENGE_DIRECT, STANDARD_SIZE
        )
        self.assertTrue(success, message)
        info = self._info(self.standard)
        self.assertEqual(info.pro_account_size, STANDARD_SIZE)
        self.assertEqual(info.standard_account_size, STANDARD_SIZE)
        self.assertEqual(info.account_size, STANDARD_SIZE)
        self.assertEqual(info.account_type, AccountType.PRO.value)
        # Nothing to resize: the pro account is the size it was already trading
        self.set_size.assert_not_called()
        self.assertAlmostEqual(self.manager.get_payout_scale(self.standard), PAYOUT_MULTIPLIER)

    def test_reoffer_after_demotion_requires_a_size_again(self):
        """The size recorded on an earlier pro journey is never silently reused when re-entering the track."""
        demoted = _add_demoted(self.manager)
        for bucket in PRO_BUCKETS:
            with self.subTest(bucket=bucket):
                self._assert_rejected_unchanged(demoted, bucket, None, REQUIRED)
        self.assertEqual(self._info(demoted).pro_account_size, GRANTED_SIZE)

        success, message = self.manager.apply_bucket_account_size(
            demoted, MinerBucket.PRO_CHALLENGE_TRANSITION, 300_000
        )
        self.assertTrue(success, message)
        info = self._info(demoted)
        self.assertEqual(info.pro_account_size, 300_000)
        self.assertEqual(info.account_type, AccountType.PRO.value)
        self.assertEqual(info.account_size, STANDARD_SIZE)

    def test_full_journey_offer_demote_reoffer(self):
        """Offer at 500K, start pro, demote, re-offer: the re-offer needs a size and trades the new one."""
        hotkey = self.standard
        apply = self.manager.apply_bucket_account_size
        self.assertTrue(apply(hotkey, MinerBucket.PRO_CHALLENGE_TRANSITION, GRANTED_SIZE)[0])
        self.assertTrue(apply(hotkey, MinerBucket.PRO_CHALLENGE_FROM_STANDARD)[0])
        self.assertEqual(self._info(hotkey).account_size, GRANTED_SIZE)
        self.assertTrue(apply(hotkey, MinerBucket.SUBACCOUNT_FUNDED)[0])
        self.assertEqual(self._info(hotkey).account_size, STANDARD_SIZE)

        self._assert_rejected_unchanged(hotkey, MinerBucket.PRO_CHALLENGE_TRANSITION, None, REQUIRED)

        self.assertTrue(apply(hotkey, MinerBucket.PRO_CHALLENGE_TRANSITION, 250_000)[0])
        self.assertTrue(apply(hotkey, MinerBucket.PRO_CHALLENGE_FROM_STANDARD)[0])
        info = self._info(hotkey)
        self.assertEqual(info.pro_account_size, 250_000)
        self.assertEqual(info.account_size, 250_000)
        self.assertEqual(info.standard_account_size, STANDARD_SIZE)
        self.assertAlmostEqual(self.manager.get_payout_scale(hotkey),
                               PAYOUT_MULTIPLIER * STANDARD_SIZE / 250_000)

    # ==================== within the pro track ====================

    def test_move_within_the_track_uses_the_recorded_size(self):
        pro = _add_pro(self.manager)
        for bucket in (MinerBucket.PRO_FUNDED, MinerBucket.PRO_CHALLENGE_FROM_STANDARD):
            with self.subTest(bucket=bucket):
                success, message = self.manager.apply_bucket_account_size(pro, bucket)
                self.assertTrue(success, message)
                info = self._info(pro)
                self.assertEqual(info.pro_account_size, GRANTED_SIZE)
                self.assertEqual(info.account_size, GRANTED_SIZE)
                self.assertEqual(info.standard_account_size, STANDARD_SIZE)

    def test_start_pro_now_trades_the_size_set_at_transition(self):
        self.assertTrue(self.manager.apply_bucket_account_size(
            self.standard, MinerBucket.PRO_CHALLENGE_TRANSITION, GRANTED_SIZE)[0])

        success, message = self.manager.apply_bucket_account_size(
            self.standard, MinerBucket.PRO_CHALLENGE_FROM_STANDARD
        )

        self.assertTrue(success, message)
        info = self._info(self.standard)
        self.assertEqual(info.account_size, GRANTED_SIZE)
        self.assertEqual(self.set_size.call_args.kwargs["account_size"], GRANTED_SIZE)
        self.assertAlmostEqual(self.manager.get_payout_scale(self.standard),
                               PAYOUT_MULTIPLIER * STANDARD_SIZE / GRANTED_SIZE)

    def test_move_within_the_track_with_no_recorded_size_is_rejected(self):
        pro = _add_pro(self.manager)
        self._info(pro).pro_account_size = None
        for bucket in PRO_BUCKETS:
            with self.subTest(bucket=bucket):
                self._assert_rejected_unchanged(pro, bucket, None, REQUIRED)

    def test_pro_funded_can_re_set_the_size_within_range(self):
        pro = _add_pro(self.manager)

        self._assert_rejected_unchanged(pro, MinerBucket.PRO_FUNDED, 1_000_001, OUT_OF_RANGE)
        self._assert_rejected_unchanged(pro, MinerBucket.PRO_FUNDED, STANDARD_SIZE - 1, BELOW_STANDARD)

        success, message = self.manager.apply_bucket_account_size(pro, MinerBucket.PRO_FUNDED, 750_000)
        self.assertTrue(success, message)
        info = self._info(pro)
        self.assertEqual(info.pro_account_size, 750_000)
        self.assertEqual(info.account_size, 750_000)
        self.assertEqual(info.standard_account_size, STANDARD_SIZE)

    def test_recorded_size_is_not_rechecked_against_the_range(self):
        """A size that was valid when granted keeps working if the cap later moves past it."""
        pro = _add_pro(self.manager)
        with patch.object(ValiConfig, "MAX_PRO_ACCOUNT_SIZE", 400_000):
            success, message = self.manager.apply_bucket_account_size(pro, MinerBucket.PRO_FUNDED)
        self.assertTrue(success, message)
        self.assertEqual(self._info(pro).account_size, GRANTED_SIZE)

    def test_corrupt_recorded_size_is_rejected_and_changes_nothing(self):
        for recorded in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(recorded=recorded):
                manager = self.manager = _bare_manager()
                self.set_size = manager._miner_account_client.set_miner_account_size
                pro = _add_pro(manager)
                self._info(pro).pro_account_size = recorded
                self._assert_rejected_unchanged(pro, MinerBucket.PRO_FUNDED, None, "Recorded pro_account_size")

    def test_explicit_size_replaces_a_corrupt_recorded_one(self):
        pro = _add_pro(self.manager)
        self._info(pro).pro_account_size = float("nan")

        success, message = self.manager.apply_bucket_account_size(pro, MinerBucket.PRO_FUNDED, 600_000)

        self.assertTrue(success, message)
        self.assertEqual(self._info(pro).pro_account_size, 600_000)
        self.assertEqual(self._info(pro).account_size, 600_000)

    # ==================== standard buckets ====================

    def test_standard_buckets_never_record_a_pro_size(self):
        for bucket in STANDARD_BUCKETS:
            with self.subTest(bucket=bucket):
                success, message = self.manager.apply_bucket_account_size(self.standard, bucket, GRANTED_SIZE)
                self.assertTrue(success, message)
                info = self._info(self.standard)
                self.assertIsNone(info.pro_account_size)
                self.assertIsNone(info.standard_account_size)
                self.assertEqual(info.account_type, AccountType.STANDARD.value)
                self.assertEqual(info.account_size, STANDARD_SIZE)
                self.assertEqual(self.manager.get_payout_scale(self.standard), 1.0)
        self.set_size.assert_not_called()

    def test_demotion_restores_the_standard_size_and_ignores_a_sent_size(self):
        pro = _add_pro(self.manager)

        success, message = self.manager.apply_bucket_account_size(pro, MinerBucket.SUBACCOUNT_FUNDED, 750_000)

        self.assertTrue(success, message)
        info = self._info(pro)
        self.assertEqual(info.account_size, STANDARD_SIZE)
        self.assertEqual(info.account_type, AccountType.STANDARD.value)
        self.assertEqual(info.pro_account_size, GRANTED_SIZE)

    # ==================== failed resize ====================

    def test_failed_account_resize_leaves_the_subaccount_untouched(self):
        self.set_size.return_value = None
        pro = _add_pro(self.manager)
        for hotkey, bucket, size in (
            (self.standard, MinerBucket.PRO_CHALLENGE_DIRECT, GRANTED_SIZE),
            (pro, MinerBucket.PRO_FUNDED, 750_000),
            (pro, MinerBucket.SUBACCOUNT_FUNDED, None),
        ):
            with self.subTest(hotkey=hotkey, bucket=bucket):
                before = self._snapshot(hotkey)
                with patch.object(self.manager, "_write_entities_from_memory_to_disk") as write:
                    success, message = self.manager.apply_bucket_account_size(hotkey, bucket, size)
                self.assertFalse(success)
                self.assertIn("Failed to set account size", message)
                self.assertEqual(self._snapshot(hotkey), before)
                write.assert_not_called()
        self.assertEqual(self.manager.get_payout_scale(self.standard), 1.0)


class TestApplyBucketAccountSizeLog(unittest.TestCase):
    """The size-change log line says whether a pro size was explicit or recorded."""

    def setUp(self):
        self.manager = _bare_manager()
        self.standard = _add_standard(self.manager)

    def _logged(self, hotkey, bucket, pro_account_size=None):
        with patch("entity_management.entity_manager.logger") as logger:
            success, message = self.manager.apply_bucket_account_size(hotkey, bucket, pro_account_size)
        self.assertTrue(success, message)
        # The asset-class switch onto all_markets logs its own lines; only the sizing line is under test
        lines = [call.args[0] for call in logger.info.call_args_list if "account_size=$" in call.args[0]]
        self.assertEqual(len(lines), 1, lines)
        return lines[0]

    def test_the_line_names_the_size_and_where_it_came_from(self):
        """Which of the two branches ran is the thing an operator cannot otherwise tell apart when
        a subaccount ends up on the wrong size."""
        pro = _add_pro(self.manager)
        for hotkey, bucket, size, source, logged_size in (
            (self.standard, MinerBucket.PRO_CHALLENGE_TRANSITION, GRANTED_SIZE, "explicit", GRANTED_SIZE),
            (pro, MinerBucket.PRO_FUNDED, None, "recorded", GRANTED_SIZE),
            (pro, MinerBucket.PRO_FUNDED, 750_000, "explicit", 750_000),
        ):
            with self.subTest(bucket=bucket, pro_account_size=size):
                line = self._logged(hotkey, bucket, size)
                self.assertIn(f"pro size source: {source}", line)
                self.assertIn(f"pro=${logged_size}", line)

    def test_standard_bucket_names_no_source(self):
        pro = _add_pro(self.manager)
        for hotkey, bucket, size in ((pro, MinerBucket.SUBACCOUNT_FUNDED, None),
                                     (self.standard, MinerBucket.SUBACCOUNT_CHALLENGE, GRANTED_SIZE)):
            with self.subTest(bucket=bucket):
                self.assertNotIn("source", self._logged(hotkey, bucket, size))


if __name__ == "__main__":
    unittest.main()
