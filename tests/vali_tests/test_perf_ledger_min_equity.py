"""PerfCheckpoint.min_equity_ret: the lowest equity_ret reached inside each checkpoint."""
import pytest

from vali_objects.enums.misc import TradePairReturnStatus
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint, PerfLedger

ACCOUNT_SIZE = 100_000.0
MINUTE_MS = 60_000
HALF_DAY_MS = 12 * 60 * MINUTE_MS
DAY_START_MS = 20_000 * 2 * HALF_DAY_MS  # a UTC midnight


def _update(ledger: PerfLedger, t_ms: int, unrealized: float = 0.0, realized: float = 0.0, fees: float = 0.0,
            tracked: bool = True) -> None:
    """One ledger update. tracked=False is an update without an account size, as for a regular miner
    or the shortcut path."""
    ledger.update_pl(
        1.0 + (realized + unrealized) / ACCOUNT_SIZE, t_ms, "hk", TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
        realized, unrealized,
        account_size=ACCOUNT_SIZE if tracked else None,
        cumulative_fees_usd=fees if tracked else None,
    )


def _equity(unrealized: float = 0.0, realized: float = 0.0, fees: float = 0.0) -> float:
    return (ACCOUNT_SIZE + realized - fees + unrealized) / ACCOUNT_SIZE


def test_tracks_the_intraday_low():
    ledger = PerfLedger()
    _update(ledger, DAY_START_MS + MINUTE_MS)
    _update(ledger, DAY_START_MS + 2 * MINUTE_MS, unrealized=-3_000)
    _update(ledger, DAY_START_MS + 3 * MINUTE_MS, unrealized=1_000)

    [cp] = ledger.cps
    assert cp.min_equity_ret == pytest.approx(0.97)
    assert cp.equity_ret == pytest.approx(1.01)


def test_counts_realized_pnl_and_fees():
    ledger = PerfLedger()
    _update(ledger, DAY_START_MS + MINUTE_MS)
    _update(ledger, DAY_START_MS + 2 * MINUTE_MS, unrealized=-4_000, realized=2_000, fees=500)

    assert ledger.cps[-1].min_equity_ret == pytest.approx(_equity(unrealized=-4_000, realized=2_000, fees=500))


def test_new_checkpoint_opens_at_the_carried_equity():
    ledger = PerfLedger()
    _update(ledger, DAY_START_MS + MINUTE_MS)
    _update(ledger, DAY_START_MS + 2 * MINUTE_MS, unrealized=-3_000)
    _update(ledger, DAY_START_MS + HALF_DAY_MS - MINUTE_MS, unrealized=-1_000)
    _update(ledger, DAY_START_MS + HALF_DAY_MS + MINUTE_MS, unrealized=500)

    morning, afternoon = ledger.cps
    assert morning.min_equity_ret == pytest.approx(0.97)
    # The equity at the boundary is the last one the morning closed on
    assert afternoon.min_equity_ret == pytest.approx(0.99)
    assert afternoon.equity_ret == pytest.approx(1.005)


def test_void_checkpoints_hold_the_carried_equity():
    ledger = PerfLedger()
    _update(ledger, DAY_START_MS + MINUTE_MS, unrealized=-2_000)
    _update(ledger, DAY_START_MS + 2 * 2 * HALF_DAY_MS + MINUTE_MS, unrealized=-500)

    void_cps = ledger.cps[1:-1]
    assert len(void_cps) == 3
    for cp in void_cps:
        assert cp.min_equity_ret == cp.equity_ret == pytest.approx(0.98)
    assert ledger.cps[-1].min_equity_ret == pytest.approx(0.98)
    assert ledger.cps[-1].equity_ret == pytest.approx(0.995)


def test_matches_the_lowest_sampled_equity_of_each_window():
    """Every checkpoint's low is the lowest equity sampled in its window, or the equity it opened on."""
    ledger = PerfLedger()
    samples = []
    for minute in range(1, 2 * 24 * 60, 7):
        t_ms = DAY_START_MS + minute * MINUTE_MS
        unrealized = ((minute * 37) % 1_000 - 600) * 5.0
        _update(ledger, t_ms, unrealized=unrealized)
        samples.append((t_ms, _equity(unrealized=unrealized)))

    opening = None
    for cp in ledger.cps:
        window = [equity for t_ms, equity in samples if cp.last_update_ms - cp.accum_ms < t_ms <= cp.last_update_ms]
        expected = min(window + ([opening] if opening is not None else []))
        assert cp.min_equity_ret == pytest.approx(expected)
        assert cp.min_equity_ret <= cp.equity_ret
        opening = cp.equity_ret


def test_untracked_ledger_leaves_it_none():
    ledger = PerfLedger()
    _update(ledger, DAY_START_MS + MINUTE_MS, unrealized=-2_000, tracked=False)
    _update(ledger, DAY_START_MS + 2 * 2 * HALF_DAY_MS + MINUTE_MS, unrealized=-500, tracked=False)

    assert len(ledger.cps) > 1
    assert all(cp.min_equity_ret is None for cp in ledger.cps)


def test_checkpoint_started_before_the_field_keeps_an_unknown_low():
    """A checkpoint loaded mid-window from an older ledger has already tracked equity without its low:
    later updates must not record a low covering only the rest of the window."""
    legacy_cp = PerfCheckpoint(last_update_ms=DAY_START_MS + MINUTE_MS, prev_portfolio_ret=1.0,
                               accum_ms=MINUTE_MS, equity_ret=0.95, mpv=1.0)
    ledger = PerfLedger(cps=[legacy_cp])
    _update(ledger, DAY_START_MS + 2 * MINUTE_MS, unrealized=-1_000)
    assert ledger.cps[0].min_equity_ret is None

    # The next checkpoint opens tracked, at the equity carried over
    _update(ledger, DAY_START_MS + HALF_DAY_MS + MINUTE_MS, unrealized=-2_000)
    assert ledger.cps[1].min_equity_ret == pytest.approx(0.98)


def test_round_trips_and_loads_legacy_checkpoints_as_none():
    ledger = PerfLedger()
    _update(ledger, DAY_START_MS + MINUTE_MS, unrealized=-2_000)

    restored = PerfLedger.from_dict(ledger.to_dict())
    assert restored.cps == ledger.cps
    assert restored.cps[0].min_equity_ret == pytest.approx(0.98)

    legacy = ledger.cps[0].to_dict()
    del legacy['min_equity_ret']
    assert PerfCheckpoint(**legacy).min_equity_ret is None
