"""
The admin-set pro account size.

There is no network default: the admin sets the size when offering the pro track, and the network enforces
only the inclusive range [ValiConfig.MIN_PRO_ACCOUNT_SIZE, ValiConfig.MAX_PRO_ACCOUNT_SIZE] plus finite and
numeric.

Covers:
  * ValiConfig: MIN_PRO_ACCOUNT_SIZE / MAX_PRO_ACCOUNT_SIZE are $200K / $1M, checked at import with a raise, and
    the old network default PRO_ACCOUNT_SIZE is gone.
  * pro_account_size_error: the shared check behind every entry point (range bounds, NaN / Infinity, bool, str).
  * EntityManager.apply_bucket_account_size: a size is required to enter the pro track, including a re-offer
    after a demotion; a move within the track uses an explicit size or the recorded one; PRO_FUNDED can re-set
    the size within range; a standard target never records a size; a rejected or failed move changes nothing;
    the log line names the size source.
  * Dashboards: subaccount_info no longer carries default_pro_account_size in the v1, v2, websocket or
    hl-traders payloads.

EntityManager is built with object.__new__ and only the attributes these methods touch, so no RPC servers
are started.
"""
import json
import re
import threading
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

import vali_objects.vali_config as vali_config_module
from entity_management.entity_manager import EntityData, EntityManager, SubaccountInfo
from entity_management.entity_utils import create_subaccount_dashboard, pro_account_size_error
from vali_objects.enums.account_type_enum import AccountType
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig

NOW_MS = 1_748_000_000_000
ENTITY_HOTKEY = "entity_alpha"
STANDARD_SIZE = 100_000.0
GRANTED_SIZE = 500_000.0
REMOVED_FIELD = "default_pro_account_size"
REQUIRED = "pro_account_size is required to enter the pro track"

PRO_BUCKETS = (MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
               MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_FUNDED)
STANDARD_BUCKETS = (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA)

# Just outside the range, NaN / Infinity (which json.loads, so Flask's get_json, accepts), and wrong types.
# Each is (size, fragment of the rejection message).
OUT_OF_RANGE = "outside the allowed range"
NOT_FINITE = "must be a finite number"
NOT_A_NUMBER = "must be a number"
INVALID_SIZES = (
    (199_999, OUT_OF_RANGE),
    (199_999.99, OUT_OF_RANGE),
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


def _no_section_clients():
    """Dashboard section clients that have nothing to report, so only subaccount_info is built."""
    clients = {name: MagicMock() for name in (
        "challenge_period", "elimination", "miner_account", "position", "limit_order", "debt_ledger",
        "statistics",
    )}
    clients["challenge_period"].get_dashboard.return_value = None
    clients["challenge_period"].get_drawdown_stats.return_value = None
    clients["challenge_period"].get_pro_stats.return_value = None
    for name in ("elimination", "miner_account", "position", "limit_order", "debt_ledger", "statistics"):
        clients[name].get_dashboard.return_value = None
    return clients


# ═══════════════════════════════════════════════════════════════════════════════
# ValiConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestProAccountSizeConfig(unittest.TestCase):

    def test_range_is_200k_to_1m(self):
        self.assertEqual(ValiConfig.MIN_PRO_ACCOUNT_SIZE, 200_000)
        self.assertEqual(ValiConfig.MAX_PRO_ACCOUNT_SIZE, 1_000_000)

    def test_there_is_no_network_default_size(self):
        self.assertFalse(hasattr(ValiConfig, "PRO_ACCOUNT_SIZE"))

    def test_import_raises_for_an_invalid_range(self):
        """MIN above MAX, or a non-positive MIN, breaks import with a ValueError (a raise, not an assert that
        python -O would strip)."""
        path = vali_config_module.__file__
        with open(path) as f:
            source = f.read()
        assignment = re.compile(r"^(    MIN_PRO_ACCOUNT_SIZE = ).*$", re.M)
        self.assertEqual(len(assignment.findall(source)), 1)

        for bad in ("MAX_PRO_ACCOUNT_SIZE + 1", "0", "-200_000"):
            with self.subTest(MIN_PRO_ACCOUNT_SIZE=bad):
                tampered = assignment.sub(lambda m: m.group(1) + bad, source)
                with self.assertRaisesRegex(ValueError, "MIN_PRO_ACCOUNT_SIZE"):
                    exec(compile(tampered, path, "exec"), {"__name__": "vali_config_probe", "__file__": path})

        # MIN == MAX is a valid (single-size) range
        equal = assignment.sub(lambda m: m.group(1) + "MAX_PRO_ACCOUNT_SIZE", source)
        exec(compile(equal, path, "exec"), {"__name__": "vali_config_probe", "__file__": path})

        # The untouched source imports cleanly, so the failures above come from the guard
        namespace = {"__name__": "vali_config_probe", "__file__": path}
        exec(compile(source, path, "exec"), namespace)
        self.assertEqual(namespace["ValiConfig"].MIN_PRO_ACCOUNT_SIZE, ValiConfig.MIN_PRO_ACCOUNT_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
# pro_account_size_error
# ═══════════════════════════════════════════════════════════════════════════════

class TestProAccountSizeError(unittest.TestCase):

    def test_accepts_the_inclusive_range(self):
        for size in (200_000, 200_000.0, 250_000, 500_000.5, 999_999.99, 1_000_000, 1_000_000.0):
            with self.subTest(size=size):
                self.assertIsNone(pro_account_size_error(size))

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

    def test_range_is_read_from_config_at_call_time(self):
        with patch.object(ValiConfig, "MIN_PRO_ACCOUNT_SIZE", 300_000):
            self.assertIn(OUT_OF_RANGE, pro_account_size_error(250_000))
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
        for size, accepted in ((199_999, False), (200_000, True), (1_000_000, True), (1_000_001, False)):
            with self.subTest(pro_account_size=size):
                manager = self.manager = _bare_manager()
                self.set_size = manager._miner_account_client.set_miner_account_size
                hotkey = _add_standard(manager)
                if not accepted:
                    self._assert_rejected_unchanged(hotkey, MinerBucket.PRO_CHALLENGE_DIRECT, size, OUT_OF_RANGE)
                    continue
                success, message = manager.apply_bucket_account_size(hotkey, MinerBucket.PRO_CHALLENGE_DIRECT, size)
                self.assertTrue(success, message)
                info = self._info(hotkey)
                self.assertEqual(info.pro_account_size, size)
                self.assertEqual(info.account_size, size)
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
        self.assertAlmostEqual(self.manager.get_payout_scale(hotkey), STANDARD_SIZE / 250_000)

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
        self.assertAlmostEqual(self.manager.get_payout_scale(self.standard), STANDARD_SIZE / GRANTED_SIZE)

    def test_move_within_the_track_with_no_recorded_size_is_rejected(self):
        pro = _add_pro(self.manager)
        self._info(pro).pro_account_size = None
        for bucket in PRO_BUCKETS:
            with self.subTest(bucket=bucket):
                self._assert_rejected_unchanged(pro, bucket, None, REQUIRED)

    def test_pro_funded_can_re_set_the_size_within_range(self):
        pro = _add_pro(self.manager)

        self._assert_rejected_unchanged(pro, MinerBucket.PRO_FUNDED, 1_000_001, OUT_OF_RANGE)
        self._assert_rejected_unchanged(pro, MinerBucket.PRO_FUNDED, 199_999, OUT_OF_RANGE)

        success, message = self.manager.apply_bucket_account_size(pro, MinerBucket.PRO_FUNDED, 750_000)
        self.assertTrue(success, message)
        info = self._info(pro)
        self.assertEqual(info.pro_account_size, 750_000)
        self.assertEqual(info.account_size, 750_000)
        self.assertEqual(info.standard_account_size, STANDARD_SIZE)

    def test_recorded_size_is_not_rechecked_against_the_range(self):
        """A size that was valid when granted keeps working if the range later moves past it."""
        pro = _add_pro(self.manager)
        with patch.object(ValiConfig, "MAX_PRO_ACCOUNT_SIZE", 400_000), \
                patch.object(ValiConfig, "MIN_PRO_ACCOUNT_SIZE", 300_000):
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
        lines = [call.args[0] for call in logger.info.call_args_list if "[ENTITY_MANAGER]" in call.args[0]]
        self.assertEqual(len(lines), 1, lines)
        return lines[0]

    def test_explicit_size(self):
        line = self._logged(self.standard, MinerBucket.PRO_CHALLENGE_TRANSITION, GRANTED_SIZE)
        self.assertIn("pro size source: explicit", line)
        self.assertIn(f"pro=${GRANTED_SIZE}", line)

    def test_recorded_size(self):
        pro = _add_pro(self.manager)
        line = self._logged(pro, MinerBucket.PRO_FUNDED)
        self.assertIn("pro size source: recorded", line)
        self.assertIn(f"pro=${GRANTED_SIZE}", line)

    def test_explicit_re_set_within_the_track(self):
        pro = _add_pro(self.manager)
        line = self._logged(pro, MinerBucket.PRO_FUNDED, 750_000)
        self.assertIn("pro size source: explicit", line)

    def test_standard_bucket_names_no_source(self):
        pro = _add_pro(self.manager)
        for hotkey, bucket, size in ((pro, MinerBucket.SUBACCOUNT_FUNDED, None),
                                     (self.standard, MinerBucket.SUBACCOUNT_CHALLENGE, GRANTED_SIZE)):
            with self.subTest(bucket=bucket):
                self.assertNotIn("source", self._logged(hotkey, bucket, size))


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboards: no default_pro_account_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestDashboardsCarryNoDefaultProAccountSize(unittest.TestCase):

    def setUp(self):
        self.manager = _bare_manager()
        self.standard = _add_standard(self.manager)
        self.pro = _add_pro(self.manager)
        self.hl = _add_standard(self.manager, subaccount_id=2)
        self.manager.get_subaccount_info_for_synthetic(self.hl).hl_address = "0x" + "ab" * 20

    def _v1_info(self, hotkey):
        return self.manager.get_subaccount_dashboard_data(hotkey)["subaccount_info"]

    def _v2_info(self, hotkey):
        clients = _no_section_clients()
        dashboard = create_subaccount_dashboard(
            hotkey,
            self.manager.get_subaccount_dashboard(hotkey),
            clients["challenge_period"], clients["elimination"], clients["miner_account"],
            clients["position"], clients["limit_order"], clients["debt_ledger"], clients["statistics"],
            0, 0, 0, 0,
        )
        return dashboard["subaccount_info"]

    def test_v1_subaccount_info(self):
        for hotkey in (self.standard, self.pro, self.hl):
            with self.subTest(hotkey=hotkey):
                self.assertNotIn(REMOVED_FIELD, self._v1_info(hotkey))

    def test_v2_subaccount_info_keeps_the_granted_size(self):
        for hotkey, granted in ((self.standard, None), (self.pro, GRANTED_SIZE), (self.hl, None)):
            with self.subTest(hotkey=hotkey):
                info = self._v2_info(hotkey)
                self.assertNotIn(REMOVED_FIELD, info)
                self.assertEqual(info["pro_account_size"], granted)


class TestDashboardEndpointsCarryNoDefaultProAccountSize(unittest.TestCase):
    """On the wire, through the real validator REST handlers and websocket frame builder."""

    def setUp(self):
        from vanta_api.validator_rest_server import ValidatorRestServer

        self.manager = _bare_manager()
        self.standard = _add_standard(self.manager)
        self.pro = _add_pro(self.manager)

        server = object.__new__(ValidatorRestServer)
        server._get_api_key_safe = MagicMock(return_value="key")
        server.is_valid_api_key = MagicMock(return_value=True)
        server.can_access_tier = MagicMock(return_value=True)
        server._entity_client = MagicMock()
        server._entity_client.get_subaccount_dashboard.side_effect = self.manager.get_subaccount_dashboard
        server._entity_client.get_subaccount_dashboard_data.side_effect = self.manager.get_subaccount_dashboard_data
        clients = _no_section_clients()
        server._challenge_period_client = clients["challenge_period"]
        server._elimination_client = clients["elimination"]
        server._miner_account_client = clients["miner_account"]
        server._position_client = clients["position"]
        server._limit_order_client = clients["limit_order"]
        server._debt_ledger_client = clients["debt_ledger"]
        server._statistics_client = clients["statistics"]
        self.server = server
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.route("/entity/subaccount/<synthetic_hotkey>", methods=["GET"])(server.get_subaccount_dashboard)
        app.route("/v2/entity/subaccount/<synthetic_hotkey>", methods=["GET"])(server.v2_get_subaccount_dashboard)
        app.route("/hl-traders/<hl_address>", methods=["GET"])(server.get_hl_trader)
        self.client = app.test_client()

    def _get(self, path):
        resp = self.client.get(path)
        self.assertEqual(resp.status_code, 200, resp.data)
        return json.loads(resp.data)["dashboard"]["subaccount_info"]

    def test_v1_endpoint(self):
        for hotkey in (self.standard, self.pro):
            with self.subTest(hotkey=hotkey):
                self.assertNotIn(REMOVED_FIELD, self._get(f"/entity/subaccount/{hotkey}"))

    def test_v2_endpoint(self):
        for hotkey, granted in ((self.standard, None), (self.pro, GRANTED_SIZE)):
            with self.subTest(hotkey=hotkey):
                info = self._get(f"/v2/entity/subaccount/{hotkey}")
                self.assertNotIn(REMOVED_FIELD, info)
                self.assertEqual(info["pro_account_size"], granted)

    def test_hl_traders_endpoint(self):
        self.server._entity_client.get_synthetic_hotkey_for_hl_address.return_value = self.standard
        info = self._get("/hl-traders/0x" + "ab" * 20)
        self.assertEqual(info["synthetic_hotkey"], self.standard)
        self.assertNotIn(REMOVED_FIELD, info)

    def test_websocket_frames(self):
        """The websocket rebuilds subaccount_info in full on each frame, including incremental ones."""
        from vanta_api.websocket_server import DashboardSubscription, WebSocketServer, WebSocketServerClient

        server = object.__new__(WebSocketServer)
        server._event_loop = MagicMock()
        server._entity_client = MagicMock()
        server._entity_client.get_subaccount_dashboard.side_effect = self.manager.get_subaccount_dashboard
        clients = _no_section_clients()
        server._challenge_period_client = clients["challenge_period"]
        server._elimination_client = clients["elimination"]
        server._miner_account_client = clients["miner_account"]
        server._position_client = clients["position"]
        server._limit_order_client = clients["limit_order"]
        server._debt_ledger_client = clients["debt_ledger"]
        server._statistics_client = clients["statistics"]
        server._send_serialized = MagicMock()

        for hotkey, granted in ((self.standard, None), (self.pro, GRANTED_SIZE)):
            ws_client = WebSocketServerClient(client_id=1, websocket=MagicMock(), api_key="k", tier=200)
            subscription = DashboardSubscription()
            for frame in ("initial", "incremental"):
                with self.subTest(hotkey=hotkey, frame=frame), \
                        patch("vanta_api.websocket_server.asyncio.run_coroutine_threadsafe"):
                    if frame == "incremental":
                        subscription.positions_time_ms = NOW_MS
                        subscription.checkpoints_time_ms = NOW_MS
                    server._send_serialized.reset_mock()
                    server._send_dashboard_update(hotkey, ws_client, subscription)
                    _client, serialized = server._send_serialized.call_args.args
                    info = json.loads(serialized)["data"]["dashboard"]["subaccount_info"]
                    self.assertNotIn(REMOVED_FIELD, info)
                    self.assertEqual(info["pro_account_size"], granted)


if __name__ == "__main__":
    unittest.main()
