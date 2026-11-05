"""
The payout accounting of a standard miner on the per-hotkey path: the Grow multiplier on what the pro challenge
earns, priced stretch by stretch where the multiplier starts or stops applying, the week's earnings pinned at an
account switch and paid back in the week they were earned, and the inputs the weight calculator gathers for it.
"""
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from tests.vali_tests.test_standard_miner_pipeline import CLIENT_PATHS
from time_util.time_util import MS_IN_WEEK, TimeUtil
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.standard_miner import StandardMinerAccount
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring, PayoutSettlement
from vali_objects.scoring.weight_calculator_manager import WeightCalculatorManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtCheckpoint, DebtLedger
from vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger import SettledSegment
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint, PerfLedger

HOTKEY = "standard_miner_hotkey"
LEGACY = "legacy_miner_hotkey"
SUBACCOUNT = "entity_hotkey_1"
NOW_MS = 1_700_000_000_000
WEEK = TimeUtil.ms_at_start_of_week(NOW_MS)  # a Monday 00:00 UTC, on a checkpoint boundary
HOUR = 60 * 60 * 1000
CP = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
SIZE = 50_000.0
PRO_SIZE = 125_000.0
GROW_SCALE = 1.5 * SIZE / PRO_SIZE  # 0.6
GROW, PRO, TRANSITION = (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_FUNDED,
                         MinerBucket.PRO_CHALLENGE_TRANSITION)
OPENING = dict(opened_at_ms=NOW_MS - 30 * ValiConfig.DAILY_MS, opening_leverage_tier=2, asset_class="crypto",
               account_size=SIZE, locked_theta=70.0, opening_price_usd=7.857)


def record(**overrides) -> StandardMinerAccount:
    return StandardMinerAccount(**{**OPENING, **overrides})


def order(at_ms: int, realized_pnl: float):
    return SimpleNamespace(processed_ms=at_ms, realized_pnl=realized_pnl)


def positions(*orders, unrealized_pnl=0.0, fees=()):
    """One position holding the orders, as the settlements read them."""
    return [SimpleNamespace(orders=list(orders), fee_history=[SimpleNamespace(time_ms=t, amount=a) for t, a in fees],
                            unrealized_pnl=unrealized_pnl)]


def debt_ledger(buckets_by_checkpoint, first_ms=WEEK + CP) -> DebtLedger:
    """Contiguous 12 hour checkpoints from first_ms, each stamped with its bucket."""
    return DebtLedger(HOTKEY, [DebtCheckpoint(timestamp_ms=first_ms + index * CP, challenge_period_status=bucket.value,
                                              total_penalty=1.0) for index, bucket in enumerate(buckets_by_checkpoint)])


def perf_ledger(unrealized_by_checkpoint=(), first_ms=WEEK + CP) -> PerfLedger:
    return PerfLedger(initialization_time_ms=WEEK, cps=[
        PerfCheckpoint(last_update_ms=first_ms + index * CP, prev_portfolio_ret=1.0, unrealized_pnl=unrealized)
        for index, unrealized in enumerate(unrealized_by_checkpoint)])


def settled(week_start_ms, payout_usd, gross_payout_usd=None, end_ms=None, bucket=TRANSITION) -> SettledSegment:
    return SettledSegment(week_start_ms=week_start_ms, segment_start_ms=week_start_ms,
                          segment_end_ms=end_ms or week_start_ms + 3 * HOUR, bucket=bucket.value,
                          payout_usd=payout_usd, gross_payout_usd=gross_payout_usd or payout_usd,
                          weekly_penalty=1.0, payout_scale=GROW_SCALE, recorded_ms=NOW_MS)


def payouts(settlements):
    return [(s.start_ms, round(s.payout_raw, 6), round(s.payout_penalized, 6)) for s in settlements]


class TestWeeklySettlementsWithTheGrowMultiplier(TestBase):
    """generate_weekly_settlements prices what the pro challenge earns at the multiplier, and nothing else."""

    THREE_WEEKS = positions(order(WEEK + 6 * HOUR, 100.0), order(WEEK + 30 * HOUR, 50.0),
                            order(WEEK + MS_IN_WEEK + 6 * HOUR, -30.0), order(WEEK + MS_IN_WEEK + 30 * HOUR, 10.0),
                            order(WEEK + 2 * MS_IN_WEEK + 6 * HOUR, 40.0))
    END = WEEK + 3 * MS_IN_WEEK

    def _settle(self, ledger, scale, pos=None, perf=None, end_ms=END):
        return DebtBasedScoring.generate_weekly_settlements(pos or self.THREE_WEEKS, perf or perf_ledger(), ledger,
                                                            start_time_ms=WEEK, end_time_ms=end_ms, payout_scale=scale)

    def test_a_legacy_miner_pays_as_before_whatever_multiplier_is_passed(self):
        legacy = debt_ledger([MinerBucket.CHALLENGE] * 10 + [MinerBucket.MAINCOMP] * 32)
        expected = [(WEEK, 150.0, 150.0), (WEEK + MS_IN_WEEK, 0.0, 0.0), (WEEK + 2 * MS_IN_WEEK, 20.0, 20.0)]
        self.assertEqual(payouts(self._settle(legacy, 1.0)), expected)
        self.assertEqual(payouts(self._settle(legacy, GROW_SCALE)), expected)
        self.assertEqual(payouts(self._settle(None, 1.0)), expected)
        self.assertEqual(legacy.payout_scale_change_times(), [])

    def test_the_pro_challenge_is_paid_at_the_multiplier_above_the_unscaled_high_water_mark(self):
        grow = debt_ledger([GROW] * 42)
        self.assertEqual(payouts(self._settle(grow, GROW_SCALE)),
                         [(WEEK, 90.0, 90.0), (WEEK + MS_IN_WEEK, 0.0, 0.0), (WEEK + 2 * MS_IN_WEEK, 12.0, 12.0)])
        self.assertEqual(grow.payout_scale_change_times(), [])

    def test_the_week_of_the_pass_is_priced_stretch_by_stretch(self):
        # The checkpoint at T closes the window ending at T: GROW through Tuesday 00:00, PRO_FUNDED from then on
        passed = debt_ledger([GROW, GROW] + [PRO] * 40)
        self.assertEqual(passed.payout_scale_change_times(), [WEEK + 24 * HOUR])
        # +100 in Grow pays 60 and lifts the mark to 100; +50 funded pays 50 in full
        self.assertEqual(payouts(self._settle(passed, GROW_SCALE))[0], (WEEK, 110.0, 110.0))
        # A loss in Grow is carried in full into the funded stretch: the account is up 50, so 50 is paid
        loss_then_gain = positions(order(WEEK + 6 * HOUR, -100.0), order(WEEK + 30 * HOUR, 150.0))
        self.assertEqual(payouts(self._settle(passed, GROW_SCALE, loss_then_gain, end_ms=WEEK + MS_IN_WEEK)),
                         [(WEEK, 50.0, 50.0)])
        # An unrealized loss at the boundary counts against the Grow stretch, as at any week's end
        underwater = perf_ledger([0.0, -20.0] + [0.0] * 40)
        self.assertEqual(payouts(self._settle(passed, GROW_SCALE, perf=underwater, end_ms=WEEK + MS_IN_WEEK)),
                         [(WEEK, 98.0, 98.0)])
        # Without a multiplier the week is one stretch, priced as any other
        self.assertEqual(payouts(self._settle(passed, 1.0, end_ms=WEEK + MS_IN_WEEK)), [(WEEK, 150.0, 150.0)])
        # The mark set by the Grow stretch stands: a loss after the pass and its recovery next week pay nothing more
        dip = positions(order(WEEK + 6 * HOUR, 100.0), order(WEEK + 30 * HOUR, -100.0),
                        order(WEEK + MS_IN_WEEK + 6 * HOUR, 100.0))
        self.assertEqual(payouts(self._settle(passed, GROW_SCALE, dip, end_ms=WEEK + 2 * MS_IN_WEEK)),
                         [(WEEK, 60.0, 60.0), (WEEK + MS_IN_WEEK, 0.0, 0.0)])

    def test_without_a_ledger_the_scale_given_applies_throughout(self):
        pos = positions(order(WEEK + 6 * HOUR, 100.0), order(WEEK + 30 * HOUR, 50.0))
        settlements = DebtBasedScoring.generate_weekly_settlements(pos, None, None, start_time_ms=0,
                                                                   end_time_ms=WEEK + MS_IN_WEEK, payout_scale=GROW_SCALE)
        self.assertEqual(payouts(settlements), [(WEEK, 90.0, 90.0)])

    def test_fees_and_penalties_still_apply_inside_a_stretch(self):
        grow = debt_ledger([GROW] * 14)
        grow.checkpoints[1].total_penalty = 0.5  # the window Monday 12:00 to Tuesday 00:00
        with_fee = positions(order(WEEK + 6 * HOUR, 100.0), order(WEEK + 20 * HOUR, 40.0), fees=((WEEK + 8 * HOUR, 10.0),))
        settlements = self._settle(grow, GROW_SCALE, with_fee, end_ms=WEEK + MS_IN_WEEK)
        # Raw: (100 + 40 - 10) x 0.6; penalized: (100 + 20 - 10) x 0.6
        self.assertEqual(payouts(settlements), [(WEEK, 78.0, 66.0)])
        self.assertEqual(settlements[0].balance, 130.0)


class TestPayoutSettlementsWithSettledSegments(TestBase):
    """generate_payout_settlements pays what an earlier account settled at a switch in the week it was earned."""

    def _generate(self, pos=None, perf=None, ledger=None, scales=None, segments=None, hotkey=HOTKEY):
        return DebtBasedScoring.generate_payout_settlements(
            [hotkey], {hotkey: pos} if pos else {}, {hotkey: perf} if perf else {}, {hotkey: ledger} if ledger else {},
            start_time_ms=WEEK, end_time_ms=WEEK + 2 * MS_IN_WEEK + HOUR, payout_scales=scales,
            settled_segments=segments)[hotkey]

    def test_an_account_wiped_by_a_switch_still_pays_what_it_settled(self):
        segments = {HOTKEY: [settled(WEEK, 80.0, 100.0), settled(WEEK + MS_IN_WEEK, 30.0)]}
        self.assertEqual(payouts(self._generate(segments=segments)),
                         [(WEEK, 100.0, 80.0), (WEEK + MS_IN_WEEK, 30.0, 30.0)])
        self.assertEqual(self._generate(), [])
        # A settled week older than the last closed one is not read as the week just closed: the list ends on the
        # last closed week, empty
        self.assertEqual(payouts(self._generate(segments={HOTKEY: [settled(WEEK, 80.0, 100.0)]})),
                         [(WEEK, 100.0, 80.0), (WEEK + MS_IN_WEEK, 0.0, 0.0)])

    def test_a_settled_segment_joins_the_week_the_new_account_traded_in(self):
        pos = positions(order(WEEK + MS_IN_WEEK + 6 * HOUR, 40.0))
        ledger = debt_ledger([GROW] * 28)
        segments = {HOTKEY: [settled(WEEK + MS_IN_WEEK, 30.0, 50.0), settled(WEEK, 20.0),
                             settled(WEEK + 2 * MS_IN_WEEK, 999.0)]}  # the open week is not paid yet
        settlements = self._generate(pos, perf_ledger(), ledger, {HOTKEY: GROW_SCALE}, segments)
        self.assertEqual(payouts(settlements), [(WEEK, 20.0, 20.0), (WEEK + MS_IN_WEEK, 74.0, 54.0)])
        self.assertEqual([len(s.orders) for s in settlements], [0, 1])
        # The same account with nothing settled pays its own earnings alone, at the multiplier
        self.assertEqual(payouts(self._generate(pos, perf_ledger(), ledger, {HOTKEY: GROW_SCALE})),
                         [(WEEK + MS_IN_WEEK, 24.0, 24.0)])

    def test_nothing_traded_up_to_a_settlement_is_paid_again(self):
        # The positions of the account wound down are back (a switch retried, or a sync from before it): what was
        # traded at or before the settlement is in the segment and only that
        pos = positions(order(WEEK + 6 * HOUR, 100.0), order(WEEK + MS_IN_WEEK + 6 * HOUR, 40.0))
        ledger = debt_ledger([GROW] * 28)
        segments = {HOTKEY: [settled(WEEK, 20.0, end_ms=WEEK + 10 * HOUR)]}
        self.assertEqual(payouts(self._generate(pos, perf_ledger(), ledger, {HOTKEY: GROW_SCALE}, segments)),
                         [(WEEK, 20.0, 20.0), (WEEK + MS_IN_WEEK, 24.0, 24.0)])

    def test_a_standard_miner_is_paid_from_the_window_before_its_first_checkpoint(self):
        # An account opened by a switch starts earning before its first checkpoint lands
        pos = positions(order(WEEK + 6 * HOUR, 100.0))
        ledger = debt_ledger([GROW] * 28)
        self.assertEqual(payouts(self._generate(pos, perf_ledger(), ledger, {HOTKEY: GROW_SCALE})),
                         [(WEEK, 60.0, 60.0), (WEEK + MS_IN_WEEK, 0.0, 0.0)])
        self.assertEqual(payouts(self._generate(pos, perf_ledger(), ledger, {HOTKEY: 1.0})),
                         [(WEEK, 100.0, 100.0), (WEEK + MS_IN_WEEK, 0.0, 0.0)])
        # Not when the ledger started elsewhere: the window before the first earning checkpoint stays unpaid
        funded_later = debt_ledger([MinerBucket.SUBACCOUNT_CHALLENGE, GROW] + [GROW] * 26)
        self.assertEqual(payouts(self._generate(pos, perf_ledger(), funded_later, {HOTKEY: GROW_SCALE})),
                         [(WEEK, 0.0, 0.0), (WEEK + MS_IN_WEEK, 0.0, 0.0)])

    def test_a_legacy_miner_is_untouched(self):
        pos = positions(order(WEEK + 6 * HOUR, 100.0), order(WEEK + MS_IN_WEEK + 6 * HOUR, 40.0))
        ledger = debt_ledger([MinerBucket.MAINCOMP] * 28)
        before = DebtBasedScoring.generate_weekly_settlements(pos, perf_ledger(), ledger, start_time_ms=WEEK + CP,
                                                              end_time_ms=WEEK + 2 * MS_IN_WEEK)
        self.assertEqual(payouts(self._generate(pos, perf_ledger(), ledger, hotkey=LEGACY)), payouts(before))
        # The order before the first earning checkpoint is not payable, as before
        self.assertEqual(payouts(before), [(WEEK, 0.0, 0.0), (WEEK + MS_IN_WEEK, 40.0, 40.0)])


class TestSettlingAtTheSwitch(TestBase):
    """_switch_account pins what a standard miner's account earned this payout week, as the entity does for a
    subaccount, with every client mocked."""

    def setUp(self):
        super().setUp()
        stack = contextlib.ExitStack()
        for path in CLIENT_PATHS:
            stack.enter_context(patch(path))
        self.addCleanup(stack.close)
        self.manager = ChallengePeriodManager(is_backtesting=True)
        stack.enter_context(patch.object(self.manager, "_sync_buckets_to_accounts"))
        stack.enter_context(patch.object(self.manager, "_save_to_disk"))
        self.accounts = self.manager._miner_account_client
        self.entity = self.manager._entity_client
        self.records = {HOTKEY: record(pro_account_size=PRO_SIZE)}
        self.accounts.get_standard_account.side_effect = self.records.get
        self.accounts.set_standard_account.side_effect = self.records.__setitem__
        self.entity.get_subaccount_info_for_synthetic.side_effect = (
            lambda hotkey: {"initial_bucket": None} if hotkey == SUBACCOUNT else None)
        self.entity.apply_bucket_account_size.return_value = (True, "account size set")
        self.entity.is_mothership.return_value = True
        self.week_start = TimeUtil.ms_at_start_of_week(NOW_MS)
        self.last_week = self.week_start - MS_IN_WEEK
        # Up 100 last week and 50 this week on the account being wound down
        self.manager._position_client.get_positions_for_one_hotkey.return_value = positions(
            order(self.last_week + 6 * HOUR, 100.0), order(self.week_start + 6 * HOUR, 50.0))
        self.manager._perf_ledger_client.get_perf_ledger_for_hotkey.side_effect = lambda hotkey: {hotkey: perf_ledger()}
        # A ledger opened by the switch into the funded bucket last week
        self.manager._debt_ledger_client.get_ledger.return_value = debt_ledger([TRANSITION] * 28, self.last_week + CP)
        self.manager._debt_ledger_client.get_settled_segments.return_value = []
        self.manager._debt_ledger_client.record_settled_segment.return_value = True
        self.settle = self.manager._debt_ledger_client.record_settled_segment

    def _leave_the_transition(self, hotkey=HOTKEY, switched=True):
        self.manager.set_miner_bucket(hotkey, TRANSITION, NOW_MS - 10 * ValiConfig.DAILY_MS)
        self.manager.promote_hotkeys({hotkey: TRANSITION}, NOW_MS)
        self.assertEqual(self.manager.miner_states[hotkey].current_bucket, GROW if switched else TRANSITION)

    def _pinned(self):
        return [(c.args[1], c.args[4], c.args[5], c.args[6], c.args[8]) for c in self.settle.call_args_list]

    def test_every_week_still_being_paid_is_settled_before_the_ledgers_go(self):
        self._leave_the_transition()
        # Last week, paid out this week, and this week, both unscaled: the transition is paid on the standard size
        self.assertEqual(self._pinned(), [(self.last_week, TRANSITION.value, 100.0, 100.0, 1.0),
                                          (self.week_start, TRANSITION.value, 50.0, 50.0, 1.0)])
        self.assertTrue(all(c.args[3] == NOW_MS for c in self.settle.call_args_list))
        self.entity.settle_wound_down_segment.assert_not_called()
        self.manager._debt_ledger_client.delete_debt_ledger.assert_called_once_with(HOTKEY)
        self.assertEqual(self.records[HOTKEY], record(pro_account_size=PRO_SIZE, trading_account_size=PRO_SIZE))

    def test_a_week_already_pinned_by_an_earlier_attempt_is_skipped(self):
        self.manager._debt_ledger_client.get_settled_segments.return_value = [
            settled(self.last_week, 100.0, end_ms=self.last_week + HOUR)]
        self._leave_the_transition()
        self.assertEqual(self._pinned(), [(self.week_start, TRANSITION.value, 50.0, 50.0, 1.0)])

    def test_what_an_earlier_switch_settled_is_not_pinned_again_under_another_bucket(self):
        # A move out of the funded bucket settled up to Wednesday last week and then failed; the promotion later
        # settles only what was traded after it
        self.manager._debt_ledger_client.get_settled_segments.return_value = [
            settled(self.last_week, 100.0, end_ms=self.last_week + 2 * ValiConfig.DAILY_MS, bucket=MinerBucket.SUBACCOUNT_FUNDED)]
        self._leave_the_transition()
        self.assertEqual(self._pinned(), [(self.week_start, TRANSITION.value, 50.0, 50.0, 1.0)])

    def test_a_ledger_without_an_earning_checkpoint_pins_nothing(self):
        self.manager._debt_ledger_client.get_ledger.return_value = debt_ledger(
            [MinerBucket.SUBACCOUNT_CHALLENGE] * 28, self.last_week + CP)
        self._leave_the_transition()
        self.settle.assert_not_called()

    def test_a_missing_ledger_on_an_account_that_traded_for_days_holds_the_switch(self):
        self.manager._debt_ledger_client.get_ledger.return_value = None
        self._leave_the_transition(switched=False)
        self.settle.assert_not_called()

    def test_nothing_is_settled_for_weeks_that_earned_nothing(self):
        self.manager._position_client.get_positions_for_one_hotkey.return_value = positions(
            order(self.week_start + 6 * HOUR, -100.0))
        self._leave_the_transition()
        self.settle.assert_not_called()

    def test_an_account_too_young_for_a_checkpoint_is_settled_on_its_positions(self):
        self.manager._debt_ledger_client.get_ledger.return_value = None
        self.manager._position_client.get_positions_for_one_hotkey.return_value = positions(
            order(NOW_MS - 5 * HOUR, 100.0), order(NOW_MS - 2 * HOUR, 50.0))
        self._leave_the_transition()
        self.assertEqual(self._pinned(), [(self.week_start, TRANSITION.value, 150.0, 150.0, 1.0)])
        # Leaving the pro challenge itself, its earnings are priced at the multiplier
        self.settle.reset_mock()
        self.records[HOTKEY] = record(pro_account_size=PRO_SIZE, trading_account_size=PRO_SIZE)
        self.manager.set_miner_bucket(HOTKEY, GROW, NOW_MS - 10 * ValiConfig.DAILY_MS)
        moved, message = self.manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS)
        self.assertTrue(moved, message)
        self.assertEqual(self._pinned(), [(self.week_start, GROW.value, 90.0, 90.0, GROW_SCALE)])

    def test_a_settlement_that_fails_holds_the_switch_for_a_retry(self):
        for failure in ("raises", "not written"):
            with self.subTest(failure=failure):
                self.settle.reset_mock()
                self.accounts.set_standard_account.reset_mock()
                if failure == "raises":
                    self.manager._position_client.get_positions_for_one_hotkey.side_effect = RuntimeError("position server down")
                else:
                    self.settle.return_value = False
                self._leave_the_transition(switched=False)
                self.accounts.set_standard_account.assert_not_called()
                self.manager._debt_ledger_client.delete_debt_ledger.assert_not_called()
                self.assertEqual(self.records[HOTKEY], record(pro_account_size=PRO_SIZE))

    def test_a_subaccount_still_settles_through_its_entity(self):
        self.manager.set_miner_bucket(SUBACCOUNT, TRANSITION, NOW_MS - 10 * ValiConfig.DAILY_MS)
        self.manager.promote_hotkeys({SUBACCOUNT: TRANSITION}, NOW_MS)
        self.entity.settle_wound_down_segment.assert_called_once_with(SUBACCOUNT, TRANSITION.value, NOW_MS)
        self.settle.assert_not_called()


class TestWeightCalculatorInputs(TestBase):

    def test_only_the_standard_miners_being_paid_get_a_multiplier_and_their_settled_segments(self):
        calculator = object.__new__(WeightCalculatorManager)
        calculator._miner_account_client = MagicMock()
        calculator._debt_ledger_client = MagicMock()
        calculator._miner_account_client.get_standard_accounts.return_value = {
            HOTKEY: record(pro_account_size=PRO_SIZE), "other": record()}
        calculator._debt_ledger_client.get_all_settled_segments.return_value = {
            HOTKEY: [settled(WEEK, 10.0)], "other": [settled(WEEK, 5.0)]}
        scales, segments = calculator._standard_miner_payout_inputs([HOTKEY, LEGACY])
        self.assertAlmostEqual(scales[HOTKEY], GROW_SCALE)
        self.assertEqual(set(scales), {HOTKEY})
        self.assertEqual(segments, {HOTKEY: [settled(WEEK, 10.0)]})
        calculator._debt_ledger_client.get_all_settled_segments.assert_called_once_with()
        # Before the pro size is granted the multiplier is 1, and with no standard miner paid nothing is fetched
        calculator._miner_account_client.get_standard_accounts.return_value = {HOTKEY: record()}
        calculator._debt_ledger_client.get_all_settled_segments.return_value = {}
        self.assertEqual(calculator._standard_miner_payout_inputs([HOTKEY]), ({HOTKEY: 1.0}, {HOTKEY: []}))
        calculator._debt_ledger_client.get_all_settled_segments.reset_mock()
        self.assertEqual(calculator._standard_miner_payout_inputs([LEGACY]), ({}, {}))
        calculator._debt_ledger_client.get_all_settled_segments.assert_not_called()


class TestThroughTheServers(TestBase):

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.account_client = cls.orchestrator.get_client('miner_account')
        cls.debt_ledger_client = cls.orchestrator.get_client('debt_ledger')

    def setUp(self):
        super().setUp()
        self.orchestrator.clear_all_test_data()
        self.debt_ledger_client.clear_weekly_seals_for_test()

    def tearDown(self):
        self.debt_ledger_client.clear_weekly_seals_for_test()
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    def test_a_real_hotkey_records_and_reads_back_its_settled_segments_and_records(self):
        self.assertTrue(self.debt_ledger_client.record_settled_segment(
            HOTKEY, WEEK, WEEK, WEEK + 3 * HOUR, TRANSITION.value, 80.0, 100.0, 1.0, GROW_SCALE))
        # A retried switch writes the same week once
        self.assertFalse(self.debt_ledger_client.record_settled_segment(
            HOTKEY, WEEK, WEEK, WEEK + 4 * HOUR, TRANSITION.value, 80.0, 100.0, 1.0, GROW_SCALE))
        segments = self.debt_ledger_client.get_settled_segments(HOTKEY)
        self.assertEqual([(s.week_start_ms, s.payout_usd, s.gross_payout_usd) for s in segments], [(WEEK, 80.0, 100.0)])
        self.assertEqual(self.debt_ledger_client.get_all_settled_segments(), {HOTKEY: segments})
        self.account_client.open_standard_account(HOTKEY, record())
        self.assertEqual(self.account_client.get_standard_accounts(), {HOTKEY: record()})
