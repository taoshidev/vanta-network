from typing import Optional

from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.miner_account.miner_account_manager import MinerAccount
from vali_objects.vali_config import ValiConfig
from vali_objects.trade_pair import InstrumentType, TradePair, TradePairCategory
from vali_objects.vali_dataclasses.position import Position


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


def get_portfolio_caps(
    subaccount_asset_class: MinerAssetClass,
    miner_bucket: MinerBucket,
    account_size: float,
    trade_pair_category: TradePairCategory,
) -> tuple[float, float]:
    """Return (per_class_cap_multiplier, overall_cap_multiplier) for subaccount portfolio caps.

    For multi-class subaccounts (HL_ALL, ALL_MARKETS), the two
    values differ:
      - per_class_cap_multiplier limits exposure within `trade_pair_category`
      - overall_cap_multiplier limits total subaccount exposure across all classes; this
        is designed to be strictly tighter than the sum of per-class caps

    For single-class subaccounts, both return values equal the same per-class multiplier so
    the caller's overall-cap check is a no-op (it can apply the same two-gate logic blindly).

    Multipliers are returned, not USD amounts; the caller multiplies by balance to get the cap.
    Takes primitives (not a MinerAccount object) so it can be called from the order-entry path
    where the account is materialized as an RPC dict, not the live MinerAccount.
    """
    tier = get_leverage_tier(miner_bucket, account_size)
    per_class_cap = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[tier].get(trade_pair_category, 1.0)
    overall_cap = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[tier].get(subaccount_asset_class, 1.0)
    return per_class_cap, overall_cap

def get_max_order_size(
    account: MinerAccount,
    position: Position,
) -> tuple[float, str]:
    """Return (max_usd_value, max_value_reason) for this position.

    Three-dimensional cap:
      1. Per-pair positional leverage cap × balance
      2. Per-asset-class portfolio cap (subaccounts only)
      3. Overall portfolio cap (subaccounts only)

    Returns the remaining room after subtracting current exposure:
      max(0, min(position_cap, effective_buying_power) - abs(position.net_value))
    """
    trade_pair = position.trade_pair
    _subaccount_buckets = {MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA}
    if account.miner_bucket in _subaccount_buckets:
        tier = get_leverage_tier(account.miner_bucket, account.account_size)
        max_position_leverage = get_tier_positional_leverage(tier, trade_pair)
    else:
        max_position_leverage = trade_pair.max_leverage
    max_position_value = account.balance * max_position_leverage

    effective_buying_power = account.buying_power
    max_value_reason = "overall portfolio cap"
    if account.miner_bucket in _subaccount_buckets:
        if not account.asset_class:
            raise ValueError("asset_class must be selected for trading")
        per_class_cap, overall_cap = get_portfolio_caps(
            account.asset_class, account.miner_bucket, account.account_size, trade_pair.trade_pair_category
        )
        per_class_used = account.capital_used_by_class.get(trade_pair.trade_pair_category, 0.0)
        per_class_room = account.balance * per_class_cap - per_class_used
        overall_room = account.balance * overall_cap - account.capital_used
        effective_buying_power = min(account.buying_power, per_class_room, overall_room)
        if effective_buying_power == per_class_room:
            max_value_reason = f"per class cap ({trade_pair.trade_pair_category})"
        elif effective_buying_power == overall_room:
            max_value_reason = "overall portfolio cap"

    combined = min(max_position_value, effective_buying_power)
    if combined == max_position_value:
        max_value_reason = f"per pair cap ({max_position_leverage}x)"
    max_value = combined - abs(position.net_value)
    if max_value * (1 + trade_pair.taker_fee_rate * account.multiplier) > effective_buying_power:
        max_value = max_value / (1 + trade_pair.taker_fee_rate * account.multiplier)

    return max(0.0, max_value), max_value_reason


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
