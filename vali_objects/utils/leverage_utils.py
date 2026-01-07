LEVERAGE_BOUNDS_V2_START_TIME_MS = 1722018483000
LEVERAGE_BOUNDS_V3_START_TIME_MS = 1739937600000
PORTFOLIO_LEVERAGE_BOUNDS_START_TIME_MS = 1727161200000
INDICES_SOFT_CUTOFF_MS = 1731700800000
EQUITIES_METALS_SOFT_CUTOFF_MS = 1744700399000

from vali_objects.vali_config import TradePair, ValiConfig  # noqa: E402

def positional_leverage_limit_v1(trade_pair: TradePair) -> int:
    if trade_pair.is_crypto:
        return 20
    elif trade_pair.is_forex or trade_pair.is_indices:
        return 500
    else:
        raise ValueError(f"Unknown trade pair type {trade_pair.trade_pair_id}")

def positional_leverage_limit_v2(trade_pair: TradePair) -> float:
    if trade_pair.is_crypto:
        return 0.5
    elif trade_pair.is_forex or trade_pair.is_equities:
        return 5
    else:
        raise ValueError(f"Unknown trade pair type {trade_pair.trade_pair_id}")

def get_position_leverage_bounds(trade_pair: TradePair, t_ms: int) -> (float, float):
    is_leverage_v3 = t_ms >= LEVERAGE_BOUNDS_V3_START_TIME_MS
    is_leverage_v2 = t_ms >= LEVERAGE_BOUNDS_V2_START_TIME_MS
    if is_leverage_v3:
        max_position_leverage = trade_pair.max_leverage
    elif is_leverage_v2:
        max_position_leverage = positional_leverage_limit_v2(trade_pair)
    else:
        max_position_leverage = positional_leverage_limit_v1(trade_pair)
    if (trade_pair.is_indices and t_ms > INDICES_SOFT_CUTOFF_MS):
        # do not allow trade_pair orders after a soft cutoff date
        max_position_leverage = 0.0
    min_position_leverage = trade_pair.min_leverage if is_leverage_v2 else 0.001  # clamping from below not needed in v1
    return min_position_leverage, max_position_leverage

def get_portfolio_leverage_cap(t_ms: int) -> float:
    is_portfolio_cap= t_ms >= PORTFOLIO_LEVERAGE_BOUNDS_START_TIME_MS
    max_portfolio_leverage = ValiConfig.PORTFOLIO_LEVERAGE_CAP if is_portfolio_cap else float('inf')
    return max_portfolio_leverage
