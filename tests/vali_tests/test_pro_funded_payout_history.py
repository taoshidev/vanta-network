"""
PRO_FUNDED payout history, end to end: a real ChallengePeriodManager moves the subaccount between
buckets, latches soft breaches and freezes closed payout weeks, and a real EntityManager prices
every segment. Only the RPC clients are faked (positions in memory, a debt ledger stamped from
the live bucket entries and latched breach weeks, a 0.2 standard/pro payout scale).

What every scenario holds to:
  * A breached week stays breached and a clean week stays clean once frozen, whatever the ledger
    says later.
  * Settlement rows never overlap and leave no gap inside a segment, and a segment's gross payouts
    add up to its high water mark, so no PnL is paid twice.
  * PRO_FUNDED keeps the closed positions of its PRO_CHALLENGE_FUNDED stint, but pays at full
    scale only on what it makes after the promotion. The challenge stint is paid once, at the
    standard scale, from its payout record.
  * Nothing accrues while eliminated. A reverted elimination resumes the payout as though the
    account had stayed active and simply not traded.

Tests marked xfail(strict=True) assert the intended behaviour where the code does not deliver it
yet; each reason says what goes wrong.
"""
import contextlib
import os
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from entity_management.entity_manager import EntityManager
from tests.vali_tests.mock_utils import fake_account_transitions
from tests.vali_tests.test_bucket_history import _CP_CLIENT_PATHS, FakePositions
from time_util.time_util import TimeUtil
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtCheckpoint, DebtLedger
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.position import Position

DAY = ValiConfig.DAILY_MS
WEEK = 7 * DAY
CP_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
MON = 1_767_571_200_000  # 2026-01-05 00:00:00 UTC, a Monday
HK = "pro_payout_entity_1"
SCALE = 0.2  # 100K standard inside a 500K pro account

BREACHED = MinerBucket.PRO_FUNDED.calmar_threshold - 1.0
CLEAN = MinerBucket.PRO_FUNDED.calmar_threshold + 1.0


def W(n: int, days: float = 0) -> int:
    """Monday 00:00 of payout week n, plus days."""
    return int(MON + n * WEEK + days * DAY)


class ClosingPositions(FakePositions):
    """FakePositions whose close-all realizes a set PnL on each open position, as the real force
    close does: a FLAT order stamped at the close time."""

    def __init__(self):
        super().__init__()
        self.close_pnl: dict[str, float] = {}

    def close_all_positions(self, hotkey, close_time_ms, order_source=None):
        for position in self.live.get(hotkey, []):
            if position.is_open_position:
                position.orders.append(_order(OrderType.FLAT, close_time_ms, self.close_pnl.get(position.position_uuid, 0.0),
                                              f"{position.position_uuid}-close"))
                position.is_closed_position = True
                position.close_ms = close_time_ms


def _order(kind: OrderType, ms: int, pnl: float, uuid: str) -> Order:
    return Order(order_type=kind, leverage=1.0 if kind == OrderType.LONG else 0.0, price=1.0, processed_ms=ms,
                 order_uuid=uuid, trade_pair=TradePair.EURUSD, realized_pnl=pnl)


class Scenario:
    """One subaccount's life, driven through the challenge period manager one refresh at a time."""

    def __init__(self, cpm: ChallengePeriodManager, positions: ClosingPositions):
        self.cpm = cpm
        self.positions = positions
        self.now_ms = MON
        self.calmar = CLEAN
        # (after_ms, through_ms, status): checkpoints stamped with a status other than the entries say
        self.stamp_overrides: list[tuple[int, int, str]] = []
        self._n = 0

        em = object.__new__(EntityManager)
        em.running_unit_tests = True
        em.get_entity_data = lambda _hk: SimpleNamespace(subaccounts={1: {'id': 1}})
        em.get_payout_scale = lambda _hk: SCALE
        em._debt_ledger_client = SimpleNamespace(get_ledger=lambda _hk: self.debt_ledger())
        em._perf_ledger_client = SimpleNamespace(
            get_perf_ledger_for_hotkey=lambda hk: {hk: SimpleNamespace(get_checkpoint_at_time=lambda *_a: None)},
            get_frozen_ledgers=lambda: {},
        )
        em._challenge_period_client = SimpleNamespace(get_miner_bucket=cpm.get_miner_bucket,
                                                      get_bucket_segments=cpm.get_bucket_segments)
        em._position_client = positions
        self.em = em

        entity = cpm._entity_client
        entity.calculate_segment_payout_record.side_effect = em.calculate_segment_payout_record
        entity.calculate_segment_closed_weeks.side_effect = em.calculate_segment_closed_weeks
        entity.get_payout_scale.side_effect = em.get_payout_scale
        entity.apply_bucket_account_size.return_value = (True, "")
        entity.is_instant_funded.return_value = False

    @property
    def state(self):
        return self.cpm.miner_states[HK]

    # ── The fake debt ledger ──────────────────────────────────────────────────

    def debt_ledger(self) -> DebtLedger:
        """12h checkpoints from the first bucket entry through now, each stamped the way the penalty
        ledger stamps it: the bucket in force at its end (reverted entries skipped), and a zero weekly
        penalty in a soft-breach bucket when the week has a latched breach day."""
        weeks = self.state.pro_stats.soft_breach_weeks
        checkpoints = []
        cp_ms = self.state.entries[0].start_time_ms + CP_MS
        while cp_ms <= self.now_ms:
            status = self.cpm.get_miner_bucket(HK, cp_ms)
            for after_ms, through_ms, override in self.stamp_overrides:
                if after_ms < cp_ms <= through_ms:
                    status = MinerBucket(override)
            breached = status.soft_breach_applies and bool(weeks.get(TimeUtil.ms_at_start_of_week(cp_ms - 1)))
            checkpoints.append(DebtCheckpoint(timestamp_ms=cp_ms, weekly_penalty=0.0 if breached else 1.0,
                                              challenge_period_status=status.value))
            cp_ms += CP_MS
        return DebtLedger(HK, checkpoints=checkpoints)

    # ── Driving the subaccount ────────────────────────────────────────────────

    def refresh(self, t: int, calmar: float | None = None) -> None:
        """The parts of a refresh that feed the payout: latch a soft breach, capture the payout
        scale, freeze the payout weeks that have closed."""
        self.now_ms = t
        if calmar is not None:
            self.calmar = calmar
        self.state.pro_stats.calmar = self.calmar
        self.cpm._latch_soft_breaches([HK], t)
        self.cpm._record_bucket_stats([HK], t)
        self.cpm._freeze_closed_weeks([HK], t)

    def run_until(self, t: int, breach_days: tuple[int, ...] = ()) -> None:
        """Refresh at noon of every UTC day from the next one through t. A day in breach_days (day
        starts) is refreshed in breach; every other day is clean."""
        day = TimeUtil.get_start_of_day_ms(self.now_ms) + DAY
        while day + DAY // 2 <= t:
            self.refresh(day + DAY // 2, BREACHED if day in breach_days else CLEAN)
            day += DAY
        self.now_ms = max(self.now_ms, t)

    def trade(self, open_ms: int, close_ms: int, pnl: float) -> Position:
        self._n += 1
        uuid = f"p{self._n}"
        return self.positions.add(Position(
            miner_hotkey=HK, position_uuid=uuid, open_ms=open_ms, trade_pair=TradePair.EURUSD,
            position_type=OrderType.FLAT, is_closed_position=True, close_ms=close_ms,
            orders=[_order(OrderType.LONG, open_ms, 0.0, f"{uuid}-o"), _order(OrderType.FLAT, close_ms, pnl, f"{uuid}-c")],
        ))

    def open(self, open_ms: int, pnl_when_closed: float) -> Position:
        self._n += 1
        uuid = f"p{self._n}"
        self.positions.close_pnl[uuid] = pnl_when_closed
        return self.positions.add(Position(
            miner_hotkey=HK, position_uuid=uuid, open_ms=open_ms, trade_pair=TradePair.EURUSD,
            position_type=OrderType.LONG, orders=[_order(OrderType.LONG, open_ms, 0.0, f"{uuid}-o")],
        ))

    def promote(self, t: int, target: MinerBucket | None = None) -> None:
        self.run_until(t)
        bucket = self.state.current_bucket
        assert self.cpm.promote_hotkeys({HK: bucket}, t)
        assert self.state.current_bucket == (target or bucket.next_bucket)

    def eliminate(self, t: int) -> None:
        """Eliminate as the elimination manager does: open positions are flattened at t first."""
        self.run_until(t)
        self.positions.close_all_positions(HK, t)
        self.cpm.eliminate_hotkeys({HK: EliminationReason.FAILED_PRO_FUNDED_PERIOD_INTRADAY_DRAWDOWN}, t)
        assert self.state.current_bucket == MinerBucket.ELIMINATED

    def revert(self, t: int) -> None:
        self.run_until(t)
        with patch.object(TimeUtil, "now_in_millis", return_value=t):
            assert self.cpm.revert_elimination(HK)

    # ── Reading the payout ────────────────────────────────────────────────────

    def payout(self, end_ms: int | None = None, start_ms: int = 0) -> dict:
        end_ms = self.now_ms if end_ms is None else end_ms
        result = self.em._calculate_payout_for_hotkey(HK, start_ms, end_ms)
        assert result is not None
        assert_no_double_payout_or_holes(result, self.cpm.get_bucket_segments(HK))
        return result

    def segment(self, bucket: MinerBucket, start_ms: int):
        segment = self.cpm._bucket_history.get_segment(HK, start_ms, bucket.value)
        assert segment is not None, f"no {bucket.value} segment at {start_ms}"
        return segment


def spans(result) -> list[tuple]:
    """(bucket, start, end, payout) per settlement row, start and end as (week, day) offsets."""
    def at(ms):
        return divmod(round((ms - MON) / DAY, 3), 7)
    return [(r['bucket'], at(r['start_ms']), at(r['end_ms']), round(r['payout'], 6))
            for r in result['weekly_settlements']]


def assert_no_double_payout_or_holes(result: dict, segments: list[dict]) -> None:
    rows = result['weekly_settlements']
    for a, b in zip(rows, rows[1:]):
        assert a['end_ms'] <= b['start_ms'], f"overlapping rows pay a stretch twice: {a} / {b}"
    for segment in segments:
        own = [r for r in rows if segment['start_ms'] <= r['start_ms']
               and (segment['end_ms'] is None or r['start_ms'] < segment['end_ms'])]
        if not own:
            continue
        assert all(r['bucket'] == segment['bucket'] for r in own), f"a row priced in the wrong bucket: {own}"
        for a, b in zip(own, own[1:]):
            assert a['end_ms'] == b['start_ms'], f"hole inside {segment['bucket']}: {a['end_ms']} -> {b['start_ms']}"
        assert sum(r['gross_payout'] for r in own) == pytest.approx(own[-1]['payout_hwm']), \
            f"gross payouts of {segment['bucket']} exceed its high water mark"
        owed = sum(r['gross_payout'] * r['payout_scale'] for r in own)
        settled = sum(r['payout'] + r['deferred_forfeited'] for r in own) + own[-1]['deferred_balance']
        assert settled == pytest.approx(owed), f"escrow of {segment['bucket']} does not balance"


@pytest.fixture
def scenario(request):
    """Starts in PRO_CHALLENGE_FUNDED at MON, or in the bucket passed by indirect parametrization."""
    with contextlib.ExitStack() as stack:
        for path in _CP_CLIENT_PATHS:
            stack.enter_context(patch(path))
        cpm = ChallengePeriodManager(is_backtesting=True, running_unit_tests=True)
        cpm._bucket_history.clear_for_test()
        fake_account_transitions(cpm)
        miner_dir = f"{ValiBkpUtils.get_miner_dir(running_unit_tests=True)}{HK}"
        os.makedirs(miner_dir, exist_ok=True)
        positions = ClosingPositions()
        cpm._position_client = positions
        cpm._miner_account_client.get_accounts.side_effect = lambda hks: {
            hk: SimpleNamespace(balance=500_000.0, equity=500_000.0, get_account_size=lambda: 500_000.0) for hk in hks}
        stack.enter_context(patch.object(cpm, "_sync_buckets_to_accounts"))
        stack.enter_context(patch("entity_management.entity_manager.read_all_snapshots", return_value=[]))
        cpm.set_miner_bucket(HK, getattr(request, "param", MinerBucket.PRO_CHALLENGE_FUNDED), MON)
        yield Scenario(cpm, positions)
        cpm._bucket_history.clear_for_test()
        shutil.rmtree(miner_dir, ignore_errors=True)


def pro_funded_from_week_1(s: Scenario, challenge_pnl: float = 1_000.0) -> None:
    """A week of PRO_CHALLENGE_FUNDED earning challenge_pnl, promoted at Monday 00:00 of week 1."""
    s.trade(W(0, 1), W(0, 2), challenge_pnl)
    s.promote(W(1))


# ── Promotion: PRO_FUNDED keeps the challenge positions, pays only its own PnL ────

def test_pro_funded_keeps_the_challenge_positions_but_pays_full_scale_only_after_promotion(scenario):
    s = scenario
    challenge = [s.trade(W(0, 1), W(0, 2), 1_000.0), s.trade(W(1, 0.5), W(1, 1), 500.0)]
    s.promote(W(1, 2))
    funded = [s.trade(W(1, 3), W(1, 4), 300.0), s.trade(W(2, 1), W(2, 2), 200.0)]
    s.run_until(W(3, 0.5))

    # Nothing was archived: the challenge positions are still on the live account
    assert {p.position_uuid for p in s.positions.get_positions_for_one_hotkey(HK)} == \
        {p.position_uuid for p in challenge + funded}
    assert s.positions.archived == {}
    # ...and kept in the challenge segment's directory
    copied = {p["position_uuid"] for p in s.cpm._bucket_history.read_positions(
        s.segment(MinerBucket.PRO_CHALLENGE_FUNDED, MON))}
    assert copied == {p.position_uuid for p in challenge}
    assert s.segment(MinerBucket.PRO_FUNDED, W(1, 2)).account_start_ms == MON

    result = s.payout(W(3))
    assert spans(result) == [
        ("PRO_CHALLENGE_FROM_STANDARD", (0, 0), (1, 0), 200.0),   # 1000 at 0.2
        ("PRO_CHALLENGE_FROM_STANDARD", (1, 0), (1, 2), 100.0),   # 500 at 0.2, cut at the promotion
        ("PRO_FUNDED", (1, 2), (2, 0), 300.0),             # full scale from the promotion on
        ("PRO_FUNDED", (2, 0), (3, 0), 200.0),
    ]
    assert [r['payout_record'] for r in result['weekly_settlements']] == [True, True, False, False]
    assert result['payout'] == pytest.approx(800.0)


def test_pro_funded_starts_its_high_water_mark_at_zero_on_the_promotion(scenario):
    """Promoted up 30K, PRO_FUNDED loses 1K and wins back 3K: it is paid the 2K above where it
    started, never the challenge's 30K a second time."""
    s = scenario
    pro_funded_from_week_1(s, challenge_pnl=30_000.0)
    s.trade(W(1, 1), W(1, 2), -1_000.0)
    s.trade(W(2, 1), W(2, 2), 3_000.0)
    s.run_until(W(3, 0.5))

    result = s.payout(W(3))
    assert spans(result) == [
        ("PRO_CHALLENGE_FROM_STANDARD", (0, 0), (1, 0), 6_000.0),
        ("PRO_FUNDED", (1, 0), (2, 0), 0.0),
        ("PRO_FUNDED", (2, 0), (3, 0), 2_000.0),
    ]


def test_a_position_force_closed_by_the_promotion_is_paid_once_at_the_standard_scale(scenario):
    """The close is stamped 1ms before the promotion: the challenge record pays it, and
    PRO_FUNDED, which keeps the closed position on the account, does not pay it again."""
    s = scenario
    s.open(W(0, 1), pnl_when_closed=400.0)
    s.promote(W(0, 3))
    s.trade(W(0, 4), W(0, 5), 100.0)
    s.run_until(W(1, 0.5))

    result = s.payout(W(1))
    assert spans(result) == [
        ("PRO_CHALLENGE_FROM_STANDARD", (0, 0), (0, 3), 80.0),
        ("PRO_FUNDED", (0, 3), (1, 0), 100.0),
    ]
    record = s.segment(MinerBucket.PRO_CHALLENGE_FUNDED, MON).payout_record
    assert (record['payout_usd'], record['gross_payout_usd']) == (80.0, 400.0)


@pytest.mark.parametrize("scenario", [MinerBucket.SUBACCOUNT_FUNDED], indirect=True)
def test_a_position_force_closed_by_an_account_switch_is_paid_by_the_segment_it_leaves(scenario):
    """Leaving the transition week switches account: the open position is closed at the switch,
    paid unscaled in the transition, and archived, so PRO_CHALLENGE_FUNDED starts from nothing."""
    s = scenario
    s.open(W(0, 1), pnl_when_closed=400.0)
    s.run_until(W(0, 2))
    assert s.cpm.admin_set_bucket(HK, MinerBucket.SUBACCOUNT_FUNDED, W(0, 2), in_transition=True)[0]
    s.promote(W(0, 4), target=MinerBucket.PRO_CHALLENGE_FUNDED)
    assert s.positions.get_positions_for_one_hotkey(HK) == []
    s.trade(W(0, 5), W(0, 6), 100.0)
    s.run_until(W(1, 0.5))

    result = s.payout(W(1))
    assert spans(result) == [
        ("SUBACCOUNT_FUNDED", (0, 0), (0, 2), 0.0),
        ("SUBACCOUNT_FUNDED", (0, 2), (0, 4), 400.0),
        ("PRO_CHALLENGE_FROM_STANDARD", (0, 4), (1, 0), 20.0),
    ]
    assert [r['in_transition'] for r in result['weekly_settlements']] == [False, True, False]


# ── Soft breaches ─────────────────────────────────────────────────────────────

def test_a_soft_breach_withholds_the_week_and_the_next_clean_week_releases_it(scenario):
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.trade(W(2, 1), W(2, 2), 100.0)
    s.run_until(W(3, 0.5), breach_days=(W(1, 3),))

    result = s.payout(W(3))
    week_1, week_2 = result['weekly_settlements'][1:]
    assert (week_1['payout'], week_1['deferred'], week_1['weekly_penalty']) == (0.0, 500.0, 0.0)
    assert (week_2['payout'], week_2['deferred_released']) == (600.0, 500.0)
    assert result['deferred_forfeited'] == 0.0


def test_consecutive_breached_weeks_hold_the_escrow_until_a_clean_one(scenario):
    s = scenario
    pro_funded_from_week_1(s)
    for week in (1, 2, 3):
        s.trade(W(week, 1), W(week, 2), 100.0)
    s.run_until(W(4, 0.5), breach_days=(W(1, 4), W(2, 6)))

    rows = s.payout(W(4))['weekly_settlements'][1:]
    assert [r['payout'] for r in rows] == [0.0, 0.0, 300.0]
    assert [r['deferred_balance'] for r in rows] == [100.0, 200.0, 0.0]


def test_a_frozen_breached_week_stays_breached_after_the_breach_is_cleared(scenario):
    """Week 1 froze withheld. Clearing its latched breach afterwards restates the ledger, but the
    frozen week still pays nothing and the escrow is still only released by week 2."""
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.run_until(W(2, 0.5), breach_days=(W(1, 3),))
    assert s.segment(MinerBucket.PRO_FUNDED, W(1)).closed_weeks[0]['payout'] == 0.0
    before = s.payout(W(2))

    assert s.cpm.admin_set_soft_breach_week(HK, W(1), [])[0]
    assert s.debt_ledger().weekly_payout_context()[W(1)].weekly_penalty == 1.0  # the ledger says clean now
    assert spans(s.payout(W(2))) == spans(before)
    assert before['weekly_settlements'][-1]['payout'] == 0.0

    s.trade(W(2, 1), W(2, 2), 100.0)
    s.run_until(W(3, 0.5))
    assert [r['payout'] for r in s.payout(W(3))['weekly_settlements'][1:]] == [0.0, 600.0]


def test_a_frozen_clean_week_stays_clean_after_a_breach_is_latched_into_it(scenario):
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.run_until(W(2, 0.5))
    before = s.payout(W(2))

    assert s.cpm.admin_set_soft_breach_week(HK, W(1), [W(1, 3)])[0]
    assert s.debt_ledger().weekly_payout_context()[W(1)].weekly_penalty == 0.0  # the ledger says breached now
    assert spans(s.payout(W(2))) == spans(before)
    assert before['weekly_settlements'][-1]['payout'] == 500.0


def test_the_open_week_follows_the_ledger_until_it_freezes(scenario):
    """Only closed weeks are frozen: a breach latched on the week still running withholds it."""
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.run_until(W(1, 3.5))
    assert s.payout(W(1, 4))['weekly_settlements'][-1]['payout'] == 500.0

    s.run_until(W(1, 4.5), breach_days=(W(1, 4),))
    assert s.payout(W(1, 5))['weekly_settlements'][-1]['payout'] == 0.0


def test_closed_weeks_are_never_restated_as_later_weeks_close(scenario):
    s = scenario
    pro_funded_from_week_1(s)
    for week in range(1, 6):
        s.trade(W(week, 1), W(week, 2), 100.0 * week)
    breach_days = (W(2, 2), W(4, 5))
    paid = {}
    for week in range(2, 7):
        s.run_until(W(week, 0.5), breach_days=breach_days)
        for row in s.payout(W(week))['weekly_settlements']:
            key = (row['start_ms'], row['end_ms'])
            assert paid.setdefault(key, row['payout']) == row['payout'], f"week {key} restated"
    assert [paid[(W(w), W(w + 1))] for w in range(1, 6)] == [100.0, 0.0, 500.0, 0.0, 900.0]


def test_a_breach_latched_in_the_challenge_never_withholds_pro_funded(scenario):
    """Breaches only latch in a soft-breach bucket, so the challenge week pays at its scale and
    PRO_FUNDED pays its own clean week."""
    s = scenario
    s.trade(W(0, 1), W(0, 2), 1_000.0)
    s.run_until(W(0, 6.5), breach_days=(W(0, 3),))
    s.promote(W(1))
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.run_until(W(2, 0.5))

    assert s.state.pro_stats.soft_breach_weeks.get(W(0)) is None
    assert [r['payout'] for r in s.payout(W(2))['weekly_settlements']] == [200.0, 500.0]


# ── Elimination and its revert ────────────────────────────────────────────────

def test_nothing_accrues_while_eliminated(scenario):
    """Weeks frozen before the elimination are still paid; the partial week it cut short and
    everything after are not, and the payout does not move while the miner stays eliminated."""
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.trade(W(2, 0.5), W(2, 1), 200.0)
    s.open(W(2, 1), pnl_when_closed=100.0)
    s.eliminate(W(2, 2))
    frozen = list(s.segment(MinerBucket.PRO_FUNDED, W(1)).closed_weeks)

    eliminated_then = s.payout(W(3))
    s.run_until(W(6, 0.5))
    assert spans(s.payout(W(6))) == spans(eliminated_then) == [
        ("PRO_CHALLENGE_FROM_STANDARD", (0, 0), (1, 0), 200.0),
        ("PRO_FUNDED", (1, 0), (2, 0), 500.0),
    ]
    assert s.segment(MinerBucket.PRO_FUNDED, W(1)).closed_weeks == frozen
    assert s.segment(MinerBucket.PRO_FUNDED, W(1)).payout_record is None


def test_a_reverted_pro_funded_elimination_resumes_as_if_it_had_not_traded(scenario):
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.trade(W(2, 0.5), W(2, 1), 200.0)
    s.open(W(2, 1), pnl_when_closed=100.0)
    s.eliminate(W(2, 2))
    s.revert(W(4, 1))
    s.trade(W(4, 2), W(4, 3), 300.0)
    s.run_until(W(5, 0.5))

    assert s.state.current_bucket == MinerBucket.PRO_FUNDED
    funded = s.segment(MinerBucket.PRO_FUNDED, W(1))
    assert funded.is_live and funded.end_ms is None
    assert s.segment(MinerBucket.ELIMINATED, W(2, 2)).reverted_ms == W(4, 1)

    result = s.payout(W(5))
    assert spans(result) == [
        ("PRO_CHALLENGE_FROM_STANDARD", (0, 0), (1, 0), 200.0),
        ("PRO_FUNDED", (1, 0), (2, 0), 500.0),
        ("PRO_FUNDED", (2, 0), (3, 0), 300.0),   # the elimination flattened the open position at +100
        ("PRO_FUNDED", (3, 0), (4, 0), 0.0),     # eliminated the whole week: nothing traded
        ("PRO_FUNDED", (4, 0), (5, 0), 300.0),
    ]
    assert result['payout'] == pytest.approx(1_300.0)


def test_a_drawdown_carried_through_a_reverted_elimination_is_not_paid_twice(scenario):
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 1_000.0)
    s.trade(W(2, 0.5), W(2, 1), -400.0)
    s.eliminate(W(2, 2))
    s.revert(W(3, 1))
    s.trade(W(3, 2), W(3, 3), 600.0)
    s.run_until(W(4, 0.5))

    assert [r['payout'] for r in s.payout(W(4))['weekly_settlements'][1:]] == [1_000.0, 0.0, 200.0]


def test_a_breached_week_stays_breached_through_an_elimination_and_its_revert(scenario):
    """The latched breach days survive the elimination, so the withheld week stays withheld and
    its escrow is released by the first clean week after the revert, not forfeited."""
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.run_until(W(2, 0.5), breach_days=(W(1, 3),))
    s.eliminate(W(2, 0.75))
    assert s.state.pro_stats.breach_days(W(1)) == [W(1, 3)]
    s.revert(W(3, 1))
    s.trade(W(3, 2), W(3, 3), 100.0)
    s.run_until(W(4, 0.5))

    assert s.state.pro_stats.breach_days(W(1)) == [W(1, 3)]
    result = s.payout(W(4))
    rows = result['weekly_settlements'][1:]
    assert [r['payout'] for r in rows] == [0.0, 500.0, 100.0]
    assert rows[1]['deferred_released'] == 500.0
    assert result['deferred_forfeited'] == 0.0


def test_a_clean_week_stays_clean_through_an_elimination_and_its_revert(scenario):
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.run_until(W(2, 0.5))
    s.eliminate(W(2, 1))
    s.revert(W(3, 1))
    s.run_until(W(4, 0.5), breach_days=(W(3, 4),))

    rows = s.payout(W(4))['weekly_settlements'][1:]
    assert [(r['payout'], r['weekly_penalty']) for r in rows] == [(500.0, 1.0), (0.0, 1.0), (0.0, 0.0)]


@pytest.mark.xfail(strict=True, reason=(
    "A checkpoint built while the miner was eliminated keeps its ELIMINATED stamp after the revert "
    "(PenaltyLedgerManager.build_penalty_ledgers copies the old status on a full rebuild, and a "
    "delta update never revisits it). weekly_payout_context reads such a week as OFF_TRACK, and "
    "apply_deferral then forfeits the escrow of an elimination that was undone."))
def test_escrow_survives_checkpoints_stamped_while_eliminated(scenario):
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.run_until(W(2, 0.5), breach_days=(W(1, 3),))
    s.eliminate(W(2, 1))
    s.revert(W(3, 1))
    s.stamp_overrides.append((W(2, 1), W(3, 1), MinerBucket.ELIMINATED.value))
    s.trade(W(3, 2), W(3, 3), 100.0)
    s.run_until(W(4, 0.5))

    result = s.payout(W(4))
    assert result['deferred_forfeited'] == 0.0
    assert result['payout'] == pytest.approx(200.0 + 600.0)


def test_a_payout_queried_for_the_eliminated_stretch_after_the_revert_reads_it_as_pro_funded(scenario):
    s = scenario
    pro_funded_from_week_1(s)
    s.trade(W(1, 1), W(1, 2), 500.0)
    s.trade(W(2, 0.5), W(2, 1), 200.0)
    s.eliminate(W(2, 2))
    s.revert(W(4, 1))
    s.run_until(W(5, 0.5))

    assert s.cpm.get_miner_bucket(HK, W(3)) == MinerBucket.PRO_FUNDED
    assert spans(s.payout(W(3, 3)))[-2:] == [
        ("PRO_FUNDED", (2, 0), (3, 0), 200.0),
        ("PRO_FUNDED", (3, 0), (3, 3), 0.0),
    ]


@pytest.mark.xfail(strict=True, reason=(
    "revert_elimination only marks the ELIMINATED entry reverted for PRO_FUNDED. Anywhere else the "
    "restored bucket opens a new segment at the revert time (PRO_CHALLENGE_FUNDED switches_account, "
    "so it also starts a new account), which drops the PnL of the week cut short by the elimination "
    "and restarts the high water mark at zero."))
def test_a_reverted_pro_challenge_funded_elimination_resumes_its_segment(scenario):
    s = scenario
    s.trade(W(0, 1), W(0, 2), 1_000.0)
    s.trade(W(1, 0.5), W(1, 1), 500.0)
    s.eliminate(W(1, 2))
    s.revert(W(2, 1))
    s.trade(W(2, 2), W(2, 3), 100.0)
    s.run_until(W(3, 0.5))

    assert [r['payout'] for r in s.payout(W(3))['weekly_settlements']] == [200.0, 100.0, 20.0]
