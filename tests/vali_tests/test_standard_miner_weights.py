"""
The weights of standard miners: the buckets the weight calculator lists them in, a weight of exactly what they are
owed when emissions cover everyone (paid out over the week, whatever the projection got wrong on the way), the same
pro rata cut as everyone when they do not, and no dust anywhere.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import MS_IN_WEEK, TimeUtil
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.standard_miner import StandardMinerAccount
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring, PayoutSettlement
from vali_objects.scoring.weight_calculator_manager import WeightCalculatorManager
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtCheckpoint, DebtLedger

STANDARD, OTHER_STANDARD, LEGACY, BIG_LEGACY = "standard_miner", "other_standard_miner", "legacy_miner", "big_legacy_miner"
DUST = ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
DAY_MS = 24 * 60 * 60 * 1000


def raw_weights(owed, projected, capped=None, capped_targets=None, emissions_cover_owed=None):
    weights = DebtBasedScoring._compute_raw_weights_from_payouts(owed, projected, capped_hotkeys=capped,
                                                                 capped_targets=capped_targets,
                                                                 emissions_cover_owed=emissions_cover_owed)
    return {hotkey: round(weight, 9) for hotkey, weight in weights.items()}


class TestWeightsOneToOne(TestBase):

    OWED = {STANDARD: 100.0, LEGACY: 100.0, BIG_LEGACY: 300.0}

    def test_with_emissions_to_spare_a_standard_miner_gets_what_it_is_owed_and_no_more(self):
        # 1000 a day projected against 500 owed: the standard miner's 100 is a tenth of the emissions; the other
        # 90% is split 1:3 between the others, surplus included
        self.assertEqual(raw_weights(self.OWED, 1000.0, {STANDARD}), {STANDARD: 0.1, LEGACY: 0.225, BIG_LEGACY: 0.675})
        # Two standard miners are each capped at their own share
        owed = {**self.OWED, OTHER_STANDARD: 50.0}
        self.assertEqual(raw_weights(owed, 1000.0, {STANDARD, OTHER_STANDARD}),
                         {STANDARD: 0.1, OTHER_STANDARD: 0.05, LEGACY: 0.2125, BIG_LEGACY: 0.6375})
        # Emissions exactly covering what is owed is the pro rata split
        self.assertEqual(raw_weights(self.OWED, 500.0, {STANDARD}), {STANDARD: 0.2, LEGACY: 0.2, BIG_LEGACY: 0.6})
        # What the week already paid comes off the standard miner's target, the others' split stays by owed
        self.assertEqual(raw_weights(self.OWED, 1000.0, {STANDARD}, {STANDARD: 40.0}),
                         {STANDARD: 0.04, LEGACY: 0.24, BIG_LEGACY: 0.72})

    def test_when_emissions_fall_short_everyone_is_cut_pro_rata(self):
        self.assertEqual(raw_weights(self.OWED, 400.0, {STANDARD}), {STANDARD: 0.2, LEGACY: 0.2, BIG_LEGACY: 0.6})
        self.assertEqual(raw_weights(self.OWED, 400.0, {STANDARD}, {STANDARD: 40.0}),
                         {STANDARD: 0.2, LEGACY: 0.2, BIG_LEGACY: 0.6})
        # No projection to measure against: see test_without_a_projection_a_standard_miner_takes_nothing...

    def test_the_surplus_over_standard_miners_alone_goes_unassigned(self):
        # Nothing but standard miners owed: their weights sum below one and the rest is left for the burn address
        self.assertEqual(raw_weights({STANDARD: 100.0, OTHER_STANDARD: 300.0}, 1000.0, {STANDARD, OTHER_STANDARD}),
                         {STANDARD: 0.1, OTHER_STANDARD: 0.3})
        # A legacy miner owed nothing shares nothing either
        self.assertEqual(raw_weights({STANDARD: 100.0, LEGACY: 0.0}, 1000.0, {STANDARD}), {STANDARD: 0.1, LEGACY: 0.0})

    def test_without_standard_miners_the_split_is_what_it_was(self):
        for capped in (None, set(), {"someone_not_paid"}):
            with self.subTest(capped=capped):
                self.assertEqual(raw_weights(self.OWED, 1000.0, capped), {STANDARD: 0.2, LEGACY: 0.2, BIG_LEGACY: 0.6})
        self.assertEqual(raw_weights({LEGACY: 0.0}, 1000.0, None), {LEGACY: 0.0})

    def test_the_dust_of_the_others_never_shaves_a_capped_weight(self):
        weights = {STANDARD: 0.1, LEGACY: 0.9, "idle_1": 2 * DUST, "idle_2": 2 * DUST}
        raw = {STANDARD: 0.1, LEGACY: 0.9}
        held = DebtBasedScoring._hold_capped_weights(weights, {STANDARD}, raw)
        self.assertEqual(held[STANDARD], 0.1)
        self.assertAlmostEqual(sum(held.values()), 1.0)
        self.assertAlmostEqual(held["idle_1"] / held[LEGACY], 2 * DUST / 0.9)
        # Below one, or without capped miners, nothing moves
        self.assertEqual(DebtBasedScoring._hold_capped_weights({STANDARD: 0.1, LEGACY: 0.5}, {STANDARD}, raw),
                         {STANDARD: 0.1, LEGACY: 0.5})
        self.assertEqual(DebtBasedScoring._hold_capped_weights(weights, None, raw), weights)
        # Capped weights leaving no room for the others' floors are left to the normalization, which shaves everyone
        crowded = {STANDARD: 1.0, "idle_1": 2 * DUST, "idle_2": 2 * DUST}
        self.assertEqual(DebtBasedScoring._hold_capped_weights(crowded, {STANDARD}, {STANDARD: 1.0}), crowded)
        # And the cap itself never asks for more than everything
        self.assertEqual(raw_weights({STANDARD: 1500.0, LEGACY: 0.0}, 1000.0, {STANDARD}, emissions_cover_owed=True),
                         {STANDARD: 1.0, LEGACY: 0.0})

    def test_the_projection_counts_the_miners_emission_only(self):
        # Subnet 8 at block 9,486,004 (alpha per tempo): the two miners holding nearly all the incentive also hold
        # stake (12,207 and 1,820), so stake cannot tell them from the validators earning dividends
        metagraph = MagicMock()
        metagraph.get_emission.return_value = [137.3, 42.4, 0.3, 92.2, 57.2, 21.2, 9.4]  # 180 incentive, 180 dividends
        metagraph.get_stake_weight.return_value = [12_207.5, 1_820.0, 0.0, 1_722_573.5, 1_065_128.1, 397_995.1, 104_000.0]
        per_day = DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK / 360
        alpha = DebtBasedScoring._estimate_alpha_emissions_until_target(metagraph, 5)
        self.assertAlmostEqual(alpha, 180.0 * 5 * per_day)  # 0.5 alpha a block over 36,000 blocks
        self.assertAlmostEqual(alpha, 18_000.0)
        # Stake weights play no part, nor does any UID's being the burn address
        metagraph.get_stake_weight.return_value = None
        self.assertAlmostEqual(DebtBasedScoring._estimate_alpha_emissions_until_target(metagraph, 5), alpha)

    def test_without_a_projection_a_standard_miner_takes_nothing_rather_than_a_share_of_the_surplus(self):
        # Reserves unread (projection 0) or missing: nothing prices the cap, so the standard miner waits for the next
        # weight setting and the others split pro rata among themselves
        for projected in (0.0, None):
            self.assertEqual(raw_weights(self.OWED, projected, {STANDARD}, emissions_cover_owed=False),
                             {STANDARD: 0.0, LEGACY: 0.25, BIG_LEGACY: 0.75})
        # Alone, it takes nothing at all
        self.assertEqual(raw_weights({STANDARD: 10.0}, 0.0, {STANDARD}), {STANDARD: 0.0})
        # With no standard miner the split is the plain pro rata one, as before
        self.assertEqual(raw_weights({LEGACY: 100.0, BIG_LEGACY: 300.0}, 0.0),
                         {LEGACY: 0.25, BIG_LEGACY: 0.75})

    def test_the_cap_is_decided_for_the_week_not_the_day(self):
        # Monday: 1000 a day against 700 owed over 7 days passes either test; told the week does not cover it, the
        # split is pro rata, told it does, the cap holds whatever the day's figures say
        self.assertEqual(raw_weights(self.OWED, 1000.0, {STANDARD}, emissions_cover_owed=False),
                         {STANDARD: 0.2, LEGACY: 0.2, BIG_LEGACY: 0.6})
        self.assertEqual(raw_weights({STANDARD: 700.0, LEGACY: 700.0}, 1000.0, {STANDARD}, emissions_cover_owed=True),
                         {STANDARD: 0.7, LEGACY: 0.3})


class TestNoDust(TestBase):

    def _minimums(self, raw, buckets, no_dust):
        challenge_period = MagicMock()
        challenge_period.get_miner_bucket.side_effect = buckets.get
        return DebtBasedScoring._apply_minimum_weights(
            eligible_hotkeys=list(buckets), raw_debt_weights=raw, ledger_dict={}, challengeperiod_client=challenge_period,
            miner_account_client=MagicMock(), current_time_ms=None, no_dust_hotkeys=no_dust)

    def test_a_standard_miner_gets_no_dust_in_any_bucket(self):
        buckets = {STANDARD: MinerBucket.SUBACCOUNT_FUNDED, OTHER_STANDARD: MinerBucket.PLAGIARISM,
                   LEGACY: MinerBucket.MAINCOMP, BIG_LEGACY: MinerBucket.MAINCOMP}
        weights = self._minimums({STANDARD: 0.0, LEGACY: 0.0, BIG_LEGACY: 0.4}, buckets, {STANDARD, OTHER_STANDARD})
        self.assertEqual(weights, {STANDARD: 0.0, OTHER_STANDARD: 0.0, LEGACY: 3 * DUST, BIG_LEGACY: 0.4})
        # What it is owed, it keeps
        self.assertEqual(self._minimums({STANDARD: 0.3}, buckets, {STANDARD})[STANDARD], 0.3)
        # Without the set, the floors are what they were
        self.assertEqual(self._minimums({}, buckets, None),
                         {STANDARD: DUST, OTHER_STANDARD: DUST, LEGACY: 3 * DUST, BIG_LEGACY: 3 * DUST})


class TestTheBucketsListed(TestBase):

    def test_the_pipeline_buckets_that_pay_are_weighted_and_paid_and_the_challenge_is_not(self):
        earning = {b for b in MinerBucket if b.is_subaccount_earning}
        self.assertEqual(set(WeightCalculatorManager.WEIGHTED_BUCKETS),
                         {MinerBucket.CHALLENGE, MinerBucket.PROBATION, MinerBucket.MAINCOMP, MinerBucket.ENTITY} | earning)
        self.assertEqual(set(WeightCalculatorManager.PAID_BUCKETS), {MinerBucket.PROBATION, MinerBucket.MAINCOMP} | earning)
        for buckets in (WeightCalculatorManager.WEIGHTED_BUCKETS, WeightCalculatorManager.PAID_BUCKETS):
            self.assertNotIn(MinerBucket.SUBACCOUNT_CHALLENGE, buckets)
            self.assertNotIn(MinerBucket.PLAGIARISM, buckets)

    def test_the_pipeline_buckets_pay_real_hotkeys_only(self):
        calculator = object.__new__(WeightCalculatorManager)
        calculator._challengeperiod_client = MagicMock()
        calculator._challengeperiod_client.get_hotkeys_by_bucket.side_effect = lambda buckets: (
            ["legacy_1", LEGACY] if MinerBucket.MAINCOMP in buckets else ["entity_hotkey_1", STANDARD])
        self.assertEqual(calculator._paid_hotkeys(), (["legacy_1", LEGACY, STANDARD], {STANDARD}))


class TestAWeekOfWeights(TestBase):
    """_compute_miner_weights day by day through a week: a standard miner owed 700 for the week that closed is paid
    700 over the seven days, the legacy miner takes everything else, and nothing is paid twice."""

    E_DAILY = 10_000.0
    WEEK = TimeUtil.ms_at_start_of_week(1_700_000_000_000)

    def setUp(self):
        super().setUp()
        c = object.__new__(WeightCalculatorManager)
        c.is_mainnet = True
        c._last_verbose_ms = 0
        c._debt_ledger_client = MagicMock()
        c._position_client = MagicMock()
        c._perf_ledger_client = MagicMock()
        c._challengeperiod_client = MagicMock()
        c._miner_account_client = MagicMock()
        c._metagraph_client = MagicMock()
        c._position_client.get_positions_for_all_miners.return_value = {}
        c._perf_ledger_client.get_perf_ledgers.return_value = {}
        c._debt_ledger_client.get_all_debt_ledgers.return_value = {
            hotkey: DebtLedger(hotkey, [DebtCheckpoint(timestamp_ms=self.WEEK, challenge_period_status=bucket.value)])
            for hotkey, bucket in ((STANDARD, MinerBucket.SUBACCOUNT_FUNDED), (LEGACY, MinerBucket.MAINCOMP))}
        c._debt_ledger_client.get_all_settled_segments.return_value = {}
        c._miner_account_client.get_standard_accounts.return_value = {STANDARD: StandardMinerAccount(
            opened_at_ms=self.WEEK, opening_leverage_tier=2, asset_class="crypto", account_size=50_000.0,
            locked_theta=70.0, opening_price_usd=7.857)}
        buckets = {STANDARD: MinerBucket.SUBACCOUNT_FUNDED, LEGACY: MinerBucket.MAINCOMP}
        c._challengeperiod_client.get_miner_bucket.side_effect = buckets.get
        c._challengeperiod_client.get_hotkeys_by_bucket.side_effect = lambda b: [
            hotkey for hotkey, bucket in buckets.items() if bucket in (b if isinstance(b, list) else [b])]
        # (chunk_start_ms, usd) paid out so far; the ledgers start two weeks back, as a miner's would
        self.received = {STANDARD: [(self.WEEK - 2 * MS_IN_WEEK, 0.0)], LEGACY: [(self.WEEK - 2 * MS_IN_WEEK, 0.0)]}
        c._debt_ledger_client.get_all_emissions_ledgers.side_effect = lambda: {
            hotkey: SimpleNamespace(checkpoints=[SimpleNamespace(chunk_start_ms=start, chunk_emissions_usd=usd)
                                                 for start, usd in chunks]) for hotkey, chunks in self.received.items()}
        self.calculator = c
        self.settlements = {
            STANDARD: [PayoutSettlement(start_ms=self.WEEK - MS_IN_WEEK, end_ms=self.WEEK, balance=700.0,
                                        unrealized_pnl=0.0, payout_raw=700.0, payout_penalized=700.0)],
            LEGACY: [PayoutSettlement(start_ms=self.WEEK - MS_IN_WEEK, end_ms=self.WEEK, balance=300.0,
                                      unrealized_pnl=0.0, payout_raw=300.0, payout_penalized=300.0)]}

    def _settle(self, hotkey, week_start_ms, owed):
        self.settlements.setdefault(hotkey, []).append(PayoutSettlement(
            start_ms=week_start_ms, end_ms=week_start_ms + MS_IN_WEEK, balance=owed, unrealized_pnl=0.0,
            payout_raw=owed, payout_penalized=owed))

    def _run_week(self, first_day=0, paid_fraction=1.0, e_daily=None):
        """Compute the weights each day of a week and feed back what was paid; returns the week's payouts."""
        e_daily = e_daily or self.E_DAILY
        paid = {STANDARD: 0.0, LEGACY: 0.0}
        for day in range(first_day, first_day + 7):
            weights, now_ms = self._weights_on(day, e_daily)
            self.assertAlmostEqual(weights.get("burn", 0.0), 0.0, places=9)  # nothing but a rounding crumb is burned
            for hotkey in paid:
                usd = weights.get(hotkey, 0.0) * e_daily * paid_fraction
                paid[hotkey] += usd
                self.received[hotkey].append((now_ms, usd))
        return paid

    def _weights_on(self, day, e_daily=None):
        self.E_DAILY = e_daily or self.E_DAILY
        now_ms = self.WEEK + day * DAY_MS + 60 * 60 * 1000
        with patch.object(TimeUtil, "now_in_millis", return_value=now_ms), \
                patch.object(DebtBasedScoring, "generate_payout_settlements", return_value=self.settlements), \
                patch.object(DebtBasedScoring, "_estimate_alpha_emissions_until_target",
                             side_effect=lambda metagraph_client, days_until_target, **kwargs: self.E_DAILY * days_until_target), \
                patch.object(DebtBasedScoring, "_convert_alpha_to_usd", side_effect=lambda alpha_amount, **kwargs: alpha_amount), \
                patch.object(DebtBasedScoring, "log_projections"), \
                patch.object(DebtBasedScoring, "_get_burn_address_hotkey", return_value="burn"):
            _, results = self.calculator._compute_miner_weights([STANDARD, LEGACY], {STANDARD: 0, LEGACY: 1, "burn": 2}, now_ms)
        return dict(results), now_ms

    def test_a_standard_miner_is_paid_its_week_over_the_week_and_the_legacy_miner_the_rest(self):
        for day in range(7):
            weights, _ = self._weights_on(day)
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
            self.assertAlmostEqual(weights[STANDARD] * self.E_DAILY, 100.0, places=6)
            self.received[STANDARD].append((self.WEEK + day * DAY_MS, 100.0))
            self.received[LEGACY].append((self.WEEK + day * DAY_MS, self.E_DAILY - 100.0))
        paid = self._run_week()  # the same week once more, now already paid: nothing further
        self.assertAlmostEqual(paid[STANDARD], 0.0, places=6)

    def test_owed_more_than_a_day_of_emissions_the_cap_holds_all_week(self):
        # 35,000 owed against 10,000 a day: covered over the week, so 5,000 a day, never the day-by-day test that
        # would hand the surplus over from Friday on
        self.settlements[STANDARD][0].payout_penalized = 35_000.0
        paid = self._run_week()
        self.assertAlmostEqual(paid[STANDARD], 35_000.0, places=6)
        self.assertAlmostEqual(paid[LEGACY], 7 * self.E_DAILY - 35_000.0, places=6)

    def test_a_projection_that_runs_high_is_made_up_within_the_week_and_the_rest_carried(self):
        # The emissions actually paid are half the projection: each day's target grows with what is still owed
        paid = self._run_week(paid_fraction=0.5)
        self.assertGreater(paid[STANDARD], 500.0)
        self.assertLess(paid[STANDARD], 700.0)
        # The next week owes 700 more; the balance still unpaid is paid along with it, and nothing twice
        self._settle(STANDARD, self.WEEK, 700.0)
        self._settle(LEGACY, self.WEEK, 300.0)
        next_week = self._run_week(first_day=7)
        self.assertAlmostEqual(next_week[STANDARD], 700.0 + 700.0 - paid[STANDARD], places=6)
        # A week that overpaid (receipts landing after the week's last run) is taken back the week after
        self.received[STANDARD].append((self.WEEK + 2 * MS_IN_WEEK - 1, 50.0))
        self._settle(STANDARD, self.WEEK + MS_IN_WEEK, 700.0)
        self._settle(LEGACY, self.WEEK + MS_IN_WEEK, 300.0)
        self.assertAlmostEqual(self._run_week(first_day=14)[STANDARD], 650.0, places=6)

    def test_a_balance_caught_up_on_fast_never_wipes_the_others_dust(self):
        # Owed 35,000, projected 10,000 a day but paid 5,000: the balance climbs until the capped weight would take
        # everything; the legacy miner keeps a floor on every day and the weights never exceed one
        self.settlements[STANDARD][0].payout_penalized = 35_000.0
        self.settlements[LEGACY][0].payout_penalized = 0.0
        for day in range(7):
            weights, now_ms = self._weights_on(day)
            self.assertGreater(weights[LEGACY], 0.0)
            self.assertLessEqual(sum(weights.values()), 1.0 + 1e-9)
            self.received[STANDARD].append((now_ms, weights[STANDARD] * self.E_DAILY / 2))

    def test_without_an_emissions_ledger_the_day_figure_stands(self):
        # An earlier week on record would be counted as unpaid without a ledger to say otherwise: it is not
        self.settlements[STANDARD].insert(0, PayoutSettlement(
            start_ms=self.WEEK - 2 * MS_IN_WEEK, end_ms=self.WEEK - MS_IN_WEEK, balance=500.0, unrealized_pnl=0.0,
            payout_raw=500.0, payout_penalized=500.0))
        self.received = {LEGACY: self.received[LEGACY]}
        weights, _ = self._weights_on(0)
        self.assertAlmostEqual(weights[STANDARD] * self.E_DAILY, 100.0, places=6)

    def test_a_shortfall_week_leaves_the_others_their_dust(self):
        # Emissions far short of what the standard miner alone is owed: pro rata gives it everything owed and the
        # legacy miner, owed nothing, keeps its floor rather than being scaled to zero
        self.settlements[LEGACY][0].payout_penalized = 0.0
        weights, _ = self._weights_on(0, e_daily=50.0)
        self.assertGreater(weights[LEGACY], 0.0)
        self.assertAlmostEqual(weights[STANDARD], 1.0, places=3)
