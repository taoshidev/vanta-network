from vali_objects.vali_config import TradePair, TradePairCategory, ValiConfig  # noqa: E402


def get_position_leverage_bounds(trade_pair: TradePair) -> (float, float):
    return trade_pair.min_leverage, trade_pair.max_leverage

def get_portfolio_leverage_cap(trade_pair_category: TradePairCategory) -> float:
    return ValiConfig.PORTFOLIO_LEVERAGE_CAP[trade_pair_category]
