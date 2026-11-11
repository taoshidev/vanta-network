"""
A standard miner's weeks on the pro track, priced as a subaccount's: a breached pro funded week is withheld until a
clean week and lost on leaving the track, and a closed week keeps the scale and penalty it was sealed with.
"""
from unittest.mock import MagicMock, patch

from tests.vali_tests.base_objects.test_base import TestBase
from tests.vali_tests.test_standard_miner_payout import (CP, GROW, GROW_SCALE, HOTKEY, HOUR, PRO, PRO_SIZE, WEEK,
                                                         debt_ledger, order, payouts, perf_ledger, positions, record)
from time_util.time_util import MS_IN_WEEK, TimeUtil
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_manager import DebtLedgerManager
from vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger import SealedWeek

PER_WEEK = MS_IN_WEEK // CP  # 12 hour checkpoints in a week


def breached(ledger, week_index):
    """Stamp a breach on every checkpoint of one week of the ledger (checkpoints start at the week's first boundary)."""
    for cp in ledger.checkpoints[week_index * PER_WEEK:(week_index + 1) * PER_WEEK]:
        cp.weekly_penalty = 0.0
    return ledger


def settle(ledger, pos, scale=GROW_SCALE, weeks=3, sealed=None):
    return payouts(DebtBasedScoring.generate_weekly_settlements(
        pos, perf_ledger(), ledger, start_time_ms=WEEK, end_time_ms=WEEK + weeks * MS_IN_WEEK, payout_scale=scale,
        week_context=ledger.weekly_payout_context(scale, sealed=sealed)))


class TestWithholding(TestBase):

    THREE_WEEKS = positions(order(WEEK + 6 * HOUR, 100.0), order(WEEK + MS_IN_WEEK + 6 * HOUR, 50.0),
                            order(WEEK + 2 * MS_IN_WEEK + 6 * HOUR, 30.0))

    def test_a_breached_funded_week_waits_for_a_clean_one(self):
        ledger = breached(debt_ledger([PRO] * (3 * PER_WEEK)), 0)
        # The 100 of the breached week is held; the next clean week pays its own 50 and releases it
        self.assertEqual(settle(ledger, self.THREE_WEEKS),
                         [(WEEK, 100.0, 0.0), (WEEK + MS_IN_WEEK, 50.0, 150.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)])
        # Two breached weeks in a row are both released by the third
        self.assertEqual(settle(breached(ledger, 1), self.THREE_WEEKS),
                         [(WEEK, 100.0, 0.0), (WEEK + MS_IN_WEEK, 50.0, 0.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 180.0)])

    def test_leaving_the_pro_track_loses_what_was_held(self):
        ledger = breached(debt_ledger([PRO] * PER_WEEK + [MinerBucket.ELIMINATED] * (2 * PER_WEEK)), 0)
        self.assertEqual(settle(ledger, self.THREE_WEEKS),
                         [(WEEK, 100.0, 0.0), (WEEK + MS_IN_WEEK, 50.0, 50.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)])

    def test_the_week_of_the_pass_withholds_the_funded_stretch_only(self):
        # Two Grow checkpoints, then funded with a breach that week: Grow's 100 pays 60 now, the funded 50 waits
        ledger = breached(debt_ledger([GROW, GROW] + [PRO] * (3 * PER_WEEK - 2)), 0)
        for cp in ledger.checkpoints[:2]:
            cp.weekly_penalty = 1.0
        pos = positions(order(WEEK + 6 * HOUR, 100.0), order(WEEK + 30 * HOUR, 50.0),
                        order(WEEK + MS_IN_WEEK + 6 * HOUR, 20.0))
        self.assertEqual(settle(ledger, pos), [(WEEK, 110.0, 60.0), (WEEK + MS_IN_WEEK, 20.0, 70.0),
                                               (WEEK + 2 * MS_IN_WEEK, 0.0, 0.0)])

    def test_a_sealed_week_keeps_the_scale_and_penalty_it_settled_with(self):
        ledger = breached(debt_ledger([GROW] * (3 * PER_WEEK)), 1)  # a breach stamped in Grow withholds nothing
        sealed = {WEEK: SealedWeek(week_start_ms=WEEK, weekly_penalty=1.0, payout_scale=GROW_SCALE, track="OFF_TRACK",
                                   first_earning_ms=WEEK + CP, sealed_ms=WEEK + MS_IN_WEEK)}
        # The multiplier drops to 1.2 after the first week closed: that week stays at 0.6, the next are priced live
        live = 1.2 * 50_000.0 / PRO_SIZE
        self.assertEqual(settle(ledger, self.THREE_WEEKS, scale=live, sealed=sealed),
                         [(WEEK, 60.0, 60.0), (WEEK + MS_IN_WEEK, 24.0, 24.0), (WEEK + 2 * MS_IN_WEEK, 14.4, 14.4)])
        self.assertEqual(settle(ledger, self.THREE_WEEKS, scale=live),
                         [(WEEK, 48.0, 48.0), (WEEK + MS_IN_WEEK, 24.0, 24.0), (WEEK + 2 * MS_IN_WEEK, 14.4, 14.4)])

    def test_a_sealed_week_overrides_the_live_penalty_and_track(self):
        paid = [(WEEK, 100.0, 100.0), (WEEK + MS_IN_WEEK, 50.0, 50.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)]
        # Sealed clean, the breach a rebuild now reads is ignored and the week stays paid
        clean = {WEEK: SealedWeek(week_start_ms=WEEK, weekly_penalty=1.0, payout_scale=1.0, track="ON_TRACK",
                                  first_earning_ms=WEEK + CP, sealed_ms=WEEK + MS_IN_WEEK)}
        self.assertEqual(settle(breached(debt_ledger([PRO] * (3 * PER_WEEK)), 0), self.THREE_WEEKS, sealed=clean), paid)
        # Sealed breached and off the track, a rebuild that now reads a clean on-track week brings nothing back
        lost = {WEEK: SealedWeek(week_start_ms=WEEK, weekly_penalty=0.0, payout_scale=1.0, track="OFF_TRACK",
                                 first_earning_ms=WEEK + CP, sealed_ms=WEEK + MS_IN_WEEK)}
        self.assertEqual(settle(debt_ledger([PRO] * (3 * PER_WEEK)), self.THREE_WEEKS, sealed=lost),
                         [(WEEK, 100.0, 0.0), (WEEK + MS_IN_WEEK, 50.0, 50.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)])

    def test_a_partial_breach_holds_its_share_until_a_clean_week(self):
        ledger = debt_ledger([PRO] * (3 * PER_WEEK))
        for cp in ledger.checkpoints[:PER_WEEK]:
            cp.weekly_penalty = 0.5
        self.assertEqual(settle(ledger, self.THREE_WEEKS),
                         [(WEEK, 100.0, 50.0), (WEEK + MS_IN_WEEK, 50.0, 100.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)])

    def test_a_week_without_checkpoints_holds_the_escrow(self):
        # The ledger stops after the breached week: the later weeks neither release nor lose what it held
        ledger = breached(debt_ledger([PRO] * PER_WEEK), 0)
        self.assertEqual(settle(ledger, self.THREE_WEEKS),
                         [(WEEK, 100.0, 0.0), (WEEK + MS_IN_WEEK, 50.0, 50.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)])

    def test_the_per_hotkey_path_withholds_for_a_standard_miner_only(self):
        ledger = breached(debt_ledger([PRO] * (3 * PER_WEEK)), 0)
        held = [(WEEK, 100.0, 0.0), (WEEK + MS_IN_WEEK, 50.0, 150.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)]
        paid = [(WEEK, 100.0, 100.0), (WEEK + MS_IN_WEEK, 50.0, 50.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)]
        clean = {WEEK: SealedWeek(week_start_ms=WEEK, weekly_penalty=1.0, payout_scale=1.0, track="ON_TRACK",
                                  first_earning_ms=WEEK + CP, sealed_ms=WEEK + MS_IN_WEEK)}

        def through(**extra):
            return payouts(DebtBasedScoring.generate_payout_settlements(
                [HOTKEY], {HOTKEY: self.THREE_WEEKS}, {HOTKEY: perf_ledger()}, {HOTKEY: ledger},
                start_time_ms=WEEK, end_time_ms=WEEK + 3 * MS_IN_WEEK + HOUR, **extra)[HOTKEY])

        self.assertEqual(through(payout_scales={HOTKEY: 1.0}), held)
        self.assertEqual(through(payout_scales={HOTKEY: 1.0}, sealed_weeks={HOTKEY: clean}), paid)
        # Not a standard miner: nothing withheld, and the window before the first checkpoint is not payable
        self.assertEqual(through(), [(WEEK, 0.0, 0.0), (WEEK + MS_IN_WEEK, 50.0, 50.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)])

        def direct(standard_miner):
            return payouts(DebtBasedScoring.payout_settlements_for_hotkey(
                self.THREE_WEEKS, perf_ledger(), ledger, WEEK, WEEK + 3 * MS_IN_WEEK, 1.0, first_window_payable=True,
                standard_miner=standard_miner))

        self.assertEqual(direct(True), held)
        self.assertEqual(direct(False), paid)

    def test_any_other_miner_is_untouched(self):
        ledger = breached(debt_ledger([MinerBucket.MAINCOMP] * (3 * PER_WEEK)), 0)
        before = payouts(DebtBasedScoring.generate_weekly_settlements(
            self.THREE_WEEKS, perf_ledger(), ledger, start_time_ms=WEEK + CP, end_time_ms=WEEK + 3 * MS_IN_WEEK))
        # The order before the first checkpoint is not payable for a legacy miner, as before
        self.assertEqual(before, [(WEEK, 0.0, 0.0), (WEEK + MS_IN_WEEK, 50.0, 50.0), (WEEK + 2 * MS_IN_WEEK, 30.0, 30.0)])
        self.assertEqual(payouts(DebtBasedScoring.generate_payout_settlements(
            ["legacy"], {"legacy": self.THREE_WEEKS}, {"legacy": perf_ledger()}, {"legacy": ledger},
            start_time_ms=WEEK, end_time_ms=WEEK + 3 * MS_IN_WEEK + HOUR)["legacy"]), before)


class TestSealing(TestBase):
    """The ledger build seals a standard miner's closed weeks as the aggregation seals a subaccount's."""

    def setUp(self):
        super().setUp()
        self.manager = object.__new__(DebtLedgerManager)
        self.manager.running_unit_tests = True
        self.manager.weekly_seal_ledger.clear_for_test()
        self.addCleanup(self.manager.weekly_seal_ledger.clear_for_test)
        self.manager._miner_account_client = MagicMock()

    def test_closed_weeks_are_sealed_with_the_scale_and_penalty_of_the_moment(self):
        now_week = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis())
        ledger = breached(debt_ledger([PRO] * (2 * PER_WEEK) + [PRO] * 2, now_week - 2 * MS_IN_WEEK + CP), 0)
        self.manager.debt_ledgers = {HOTKEY: ledger, "legacy": debt_ledger([MinerBucket.MAINCOMP] * 4)}
        self.manager._miner_account_client.get_standard_accounts.return_value = {HOTKEY: record(pro_account_size=PRO_SIZE)}
        self.manager._seal_standard_miner_weeks()
        sealed = self.manager.weekly_seal_ledger.get_sealed(HOTKEY)
        self.assertEqual(sorted(sealed), [now_week - 2 * MS_IN_WEEK, now_week - MS_IN_WEEK])  # the open week is not sealed
        self.assertEqual((sealed[now_week - 2 * MS_IN_WEEK].weekly_penalty, sealed[now_week - 2 * MS_IN_WEEK].track), (0.0, "ON_TRACK"))
        self.assertAlmostEqual(sealed[now_week - MS_IN_WEEK].payout_scale, GROW_SCALE)
        self.assertEqual(self.manager.weekly_seal_ledger.get_sealed("legacy"), {})
        self.assertEqual(self.manager.get_sealed_weeks_for([HOTKEY, "legacy"]), {HOTKEY: sealed})
        # A second build leaves the seals as they are and disputes a changed penalty, not a changed multiplier
        with patch("vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger.logger") as log:
            self.manager._miner_account_client.get_standard_accounts.return_value = {HOTKEY: record(pro_account_size=2 * PRO_SIZE)}
            self.manager._seal_standard_miner_weeks()
            log.warning.assert_not_called()
            for cp in ledger.checkpoints:
                cp.weekly_penalty = 1.0
            self.manager._seal_standard_miner_weeks()
            log.warning.assert_called_once()
        sealed = self.manager.weekly_seal_ledger.get_sealed(HOTKEY)
        self.assertEqual(sealed[now_week - 2 * MS_IN_WEEK].weekly_penalty, 0.0)
        self.assertAlmostEqual(sealed[now_week - MS_IN_WEEK].payout_scale, GROW_SCALE)

    def test_a_sealing_failure_does_not_fail_the_build(self):
        self.manager.debt_ledgers = {HOTKEY: debt_ledger([PRO] * (2 * PER_WEEK),
                                                         TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - MS_IN_WEEK + CP)}
        self.manager._miner_account_client.get_standard_accounts.side_effect = RuntimeError("records server down")
        self.manager._seal_standard_miner_weeks()
        self.manager._miner_account_client.get_standard_accounts.side_effect = None
        self.manager._miner_account_client.get_standard_accounts.return_value = {HOTKEY: record(pro_account_size=PRO_SIZE)}
        with patch.object(self.manager.weekly_seal_ledger, "save_to_disk", side_effect=OSError("disk full")):
            self.manager._seal_standard_miner_weeks()
        self.assertEqual(len(self.manager.weekly_seal_ledger.get_sealed(HOTKEY)), 1)
