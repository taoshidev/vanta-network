from typing import TYPE_CHECKING

from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import InstrumentType, TradePair, TradePairCategory, ValiConfig  # noqa: E402

if TYPE_CHECKING:
    # Avoid circular import at runtime — MinerAccount already imports from this module.
    from vali_objects.miner_account.miner_account_manager import MinerAccount


# Legacy positional caps for XAUUSD/XAGUSD (FOREX-tagged commodity pairs). These pairs will
# be deprecated as the HL-sourced commodity lineup (GOLDUSDC, SILVERUSDC, etc.) takes over
# the commodities category; this block goes away with them.
_LEGACY_XAU_XAG_TIER_POSITIONAL = {1: 1.0, 2: 1.0, 3: 1.5, 4: 2.0}

# Reg T overnight margin caps equity SPOT at 2x regardless of subaccount tier.
REG_T_OVERNIGHT_EQUITY_SPOT_CAP = 2.0


def get_order_leverage_bounds() -> tuple[float, float]:
    return ValiConfig.ORDER_MIN_LEVERAGE, ValiConfig.ORDER_MAX_LEVERAGE


def get_position_leverage_bounds(trade_pair: TradePair) -> tuple[float, float]:
    return trade_pair.min_leverage, trade_pair.max_leverage


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


def get_portfolio_caps(account: 'MinerAccount', trade_pair: TradePair) -> tuple[float, float]:
    """Return (per_class_cap_multiplier, overall_cap_multiplier) for subaccount portfolio caps.

    For multi-class subaccounts (see MULTI_CLASS_CATEGORIES, today only HL_ALL), the two
    values differ:
      - per_class_cap_multiplier limits exposure within `trade_pair`'s asset class
      - overall_cap_multiplier limits total subaccount exposure across all classes; this
        is designed to be strictly tighter than the sum of per-class caps

    For single-class subaccounts, both return values equal the same per-class multiplier so
    the caller's overall-cap check is a no-op (it can apply the same two-gate logic blindly).

    Multipliers are returned, not USD amounts; the caller multiplies by balance to get the cap.
    """
    tier = get_leverage_tier(account.miner_bucket, account.get_account_size())

    if account.is_multi_class():
        per_class_cap = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[tier].get(trade_pair.trade_pair_category, 1.0)
        overall_cap = ValiConfig.TIER_MULTI_CLASS_OVERALL_CAP[tier]
    else:
        single_cap = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[tier].get(account.asset_class, 1.0)
        per_class_cap = single_cap
        overall_cap = single_cap

    return per_class_cap, overall_cap


def get_tier_positional_leverage(tier: int, trade_pair: TradePair) -> float:
    """Per-pair positional leverage for the subaccount path.

    Linear scaling: pair.subaccount_tier_base_leverage * tier (tier ∈ {1, 2, 3, 4} maps to 1x-4x base).
    If the tier curve ever needs to be non-linear, replace the multiplication with a
    {tier: multiplier} dict in ValiConfig and update this helper.

    Two exceptions:
      - XAUUSD/XAGUSD bypass via the legacy mini-dict (non-linear, retained until external
        deprecation of XAU/XAG completes).
      - EQUITIES SPOT hard-capped at the Reg T overnight margin (2x).
    """
    if trade_pair.trade_pair_id in ("XAUUSD", "XAGUSD"):
        return _LEGACY_XAU_XAG_TIER_POSITIONAL[tier]
    scaled = trade_pair.subaccount_tier_base_leverage * tier
    if trade_pair.trade_pair_category == TradePairCategory.EQUITIES and trade_pair.instrument_type == InstrumentType.SPOT:
        scaled = min(scaled, REG_T_OVERNIGHT_EQUITY_SPOT_CAP)  # Reg T overnight equity-margin cap
    return scaled
