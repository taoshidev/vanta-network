"""
The payout accounting of a standard miner: the weight calculator tells it apart by its record and pays it from its
bucket segments' payout table, as a subaccount (test_pro_funded_payout_history). The seal ledger it was paid through
before still records and reads back, for the migration.
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
from vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger import SealedWeek, SettledSegment
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
GROW_MULTIPLIER = 1.5
GROW_SCALE = GROW_MULTIPLIER * SIZE / PRO_SIZE  # 0.6
# FUNDED is the standard funded account the hop onto the pro account winds down
GROW, PRO, FUNDED = (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_FUNDED, MinerBucket.SUBACCOUNT_FUNDED)
OPENING = dict(opened_at_ms=NOW_MS - 30 * ValiConfig.DAILY_MS, opening_leverage_tier=2, asset_class="crypto",
               account_size=SIZE, locked_theta=70.0, opening_price_usd=7.857)


def record(**overrides) -> StandardMinerAccount:
    if "pro_account_size" in overrides:  # a granted pro size carries the multiplier fixed with it
        overrides.setdefault("payout_scale", GROW_MULTIPLIER)
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


def settled(week_start_ms, payout_usd, gross_payout_usd=None, end_ms=None, bucket=FUNDED) -> SettledSegment:
    return SettledSegment(week_start_ms=week_start_ms, segment_start_ms=week_start_ms,
                          segment_end_ms=end_ms or week_start_ms + 3 * HOUR, bucket=bucket.value,
                          payout_usd=payout_usd, gross_payout_usd=gross_payout_usd or payout_usd,
                          weekly_penalty=1.0, payout_scale=GROW_SCALE, recorded_ms=NOW_MS)


def payouts(settlements):
    return [(s.start_ms, round(s.payout_raw, 6), round(s.payout_penalized, 6)) for s in settlements]


class TestWeightCalculatorInputs(TestBase):

    def test_the_standard_miners_are_told_apart_by_their_records(self):
        """A standard miner is paid from its payout table, a regular miner on the existing rules from its positions."""
        calculator = object.__new__(WeightCalculatorManager)
        calculator._miner_account_client = MagicMock()
        calculator._miner_account_client.get_standard_accounts.return_value = {
            HOTKEY: record(pro_account_size=PRO_SIZE), "other": record()}
        self.assertEqual(calculator._standard_miners_among([HOTKEY, LEGACY]), {HOTKEY})
        self.assertEqual(calculator._standard_miners_among([LEGACY]), set())


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
            HOTKEY, WEEK, WEEK, WEEK + 3 * HOUR, FUNDED.value, 80.0, 100.0, 1.0, GROW_SCALE))
        # A retried switch writes the same week once
        self.assertFalse(self.debt_ledger_client.record_settled_segment(
            HOTKEY, WEEK, WEEK, WEEK + 4 * HOUR, FUNDED.value, 80.0, 100.0, 1.0, GROW_SCALE))
        segments = self.debt_ledger_client.get_settled_segments(HOTKEY)
        self.assertEqual([(s.week_start_ms, s.payout_usd, s.gross_payout_usd) for s in segments], [(WEEK, 80.0, 100.0)])
        self.assertEqual(self.debt_ledger_client.get_all_settled_segments(), {HOTKEY: segments})
        self.account_client.open_standard_account(HOTKEY, record())
        self.assertEqual(self.account_client.get_standard_accounts(), {HOTKEY: record()})
        # The sealed weeks of the hotkeys asked for, through the servers
        sealed = SealedWeek(week_start_ms=WEEK, weekly_penalty=0.0, payout_scale=GROW_SCALE, track="ON_TRACK",
                            first_earning_ms=WEEK + CP, sealed_ms=WEEK + MS_IN_WEEK)
        self.assertEqual(self.debt_ledger_client.get_sealed_weeks_for([HOTKEY, "other"]), {})
        self.debt_ledger_client.sync_weekly_seals({"sealed": {HOTKEY: {str(WEEK): sealed.to_dict()}}, "settled": {}})
        self.assertEqual(self.debt_ledger_client.get_sealed_weeks_for([HOTKEY, "other"]), {HOTKEY: {WEEK: sealed}})
