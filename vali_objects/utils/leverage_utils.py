from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import TradePair, TradePairCategory, TradePairSource, ValiConfig  # noqa: E402


# Legacy positional caps for XAUUSD/XAGUSD (FOREX-tagged commodity pairs). These pairs will
# be deprecated as the HL-sourced commodity lineup (GOLDUSDC, SILVERUSDC, etc.) takes over
# the commodities category; this block goes away with them.
_LEGACY_XAU_XAG_TIER_POSITIONAL = {1: 1.0, 2: 1.0, 3: 1.5, 4: 2.0}


def get_order_leverage_bounds() -> tuple[float, float]:
    return ValiConfig.ORDER_MIN_LEVERAGE, ValiConfig.ORDER_MAX_LEVERAGE


def get_position_leverage_bounds(trade_pair: TradePair) -> tuple[float, float]:
    return trade_pair.min_leverage, trade_pair.max_leverage


def get_portfolio_leverage_cap(trade_pair_category: TradePairCategory) -> float:
    return ValiConfig.PORTFOLIO_LEVERAGE_CAP[trade_pair_category]


def get_leverage_tier(miner_bucket, account_size: float) -> int:
    """Return leverage tier (1-4) for an entity subaccount.

    Tier 1: SUBACCOUNT_CHALLENGE (any size)
    Tier 2: non-challenge, account_size < $200K
    Tier 3: non-challenge, $200K <= account_size < $1M
    Tier 4: non-challenge, account_size >= $1M
    """
    if miner_bucket == MinerBucket.SUBACCOUNT_CHALLENGE:
        return 1
    if account_size >= ValiConfig.LEVERAGE_TIER4_MIN_ACCOUNT_SIZE:
        return 4
    if account_size >= ValiConfig.LEVERAGE_TIER3_MIN_ACCOUNT_SIZE:
        return 3
    return 2


def get_tier_positional_leverage(tier: int, trade_pair: TradePair) -> float:
    """Return the positional leverage limit for a given tier and trade pair.

    XAUUSD/XAGUSD are FOREX-categorized but draw from a dedicated legacy column
    (_LEGACY_XAU_XAG_TIER_POSITIONAL); they will be deprecated. HL trade pairs use
    CRYPTO limits for now regardless of their category (redirect to be removed in a
    follow-up commit).
    """
    if trade_pair.src == TradePairSource.HYPERLIQUID:
        return ValiConfig.TIER_POSITIONAL_LEVERAGE[tier][TradePairCategory.CRYPTO]
    if trade_pair.trade_pair_id in ("XAUUSD", "XAGUSD"):
        return _LEGACY_XAU_XAG_TIER_POSITIONAL[tier]
    return ValiConfig.TIER_POSITIONAL_LEVERAGE[tier][trade_pair.trade_pair_category]
