
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

    Tier 1: SUBACCOUNT_CHALLENGE / SUBACCOUNT_PRO_CHALLENGE (any size)
    Tier 2: non-challenge, account_size < $200K
    Tier 3: non-challenge, $200K <= account_size < $1M
    Tier 4: non-challenge, account_size >= $1M
    """
    if miner_bucket and miner_bucket.is_subaccount_challenge:
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


# Correlation group key prefixes. Groups span trade pair categories (the US index group holds both
# INDICES perps and EQUITIES ETFs), so keys are namespaced strings rather than a single enum.
_CURRENCY_GROUP_PREFIX = "currency"
_SECTOR_GROUP_PREFIX = "sector"
_US_INDEX_GROUP = "index:us"


def get_correlation_legs(trade_pair: TradePair) -> tuple[tuple[str, float], ...]:
    """Return the (group_key, direction) legs a LONG position in `trade_pair` contributes to.

    A long EURUSD is long EUR and short USD, so it returns both legs with opposite directions.
    Equities contribute a single sector leg, index pairs and broad US ETFs a single index leg.
    Pairs in no group return an empty tuple.
    """
    if trade_pair.trade_pair_id in ValiConfig.PRO_US_INDEX_TRADE_PAIR_IDS:
        return ((_US_INDEX_GROUP, 1.0),)

    if trade_pair.is_forex:
        legs = []
        for currency, direction in ((trade_pair.base, 1.0), (trade_pair.quote, -1.0)):
            if currency in ValiConfig.PRO_CURRENCY_EXPOSURE_LIMITS:
                legs.append((f"{_CURRENCY_GROUP_PREFIX}:{currency}", direction))
        return tuple(legs)

    exposure_group = trade_pair.exposure_group
    if exposure_group is not None:
        return ((f"{_SECTOR_GROUP_PREFIX}:{exposure_group.value}", 1.0),)

    return ()


def get_correlation_group_limit(group_key: str) -> float:
    """Net exposure limit for a group, as a multiple of account balance."""
    prefix, _, name = group_key.partition(":")
    if prefix == _CURRENCY_GROUP_PREFIX:
        return ValiConfig.PRO_CURRENCY_EXPOSURE_LIMITS[name]
    if prefix == _SECTOR_GROUP_PREFIX:
        return ValiConfig.PRO_SECTOR_EXPOSURE_LIMIT
    return ValiConfig.PRO_US_INDEX_EXPOSURE_LIMIT


def compute_correlated_exposures(open_positions: list[Position]) -> dict[str, float]:
    """Net signed USD exposure per correlation group across all open positions."""
    exposures: dict[str, float] = {}
    for position in open_positions:
        for group_key, direction in get_correlation_legs(position.trade_pair):
            exposures[group_key] = exposures.get(group_key, 0.0) + direction * position.net_value
    return exposures


def get_max_correlated_order_size(
    trade_pair: TradePair,
    open_positions: list[Position],
    balance: float,
    value_sign: float,
) -> tuple[float, str | None]:
    """Return (max_usd_value, binding_group_label) allowed by correlated-exposure limits.

    `value_sign` is the sign of the order's change to the position's net_value. A reducing order
    can still grow a group: long EURUSD + long USDJPY nets USD short, and selling EURUSD pushes
    net USD further positive.
    """
    legs = get_correlation_legs(trade_pair)
    if not legs:
        return float("inf"), None

    exposures = compute_correlated_exposures(open_positions)
    max_value, binding_group = float("inf"), None
    for group_key, direction in legs:
        limit = get_correlation_group_limit(group_key)
        # |direction * value_sign| == 1, so this is the distance to whichever bound we move toward.
        room = limit * balance - (direction * value_sign) * exposures.get(group_key, 0.0)
        if room < max_value:
            max_value, binding_group = room, f"{group_key} exposure cap {limit}x"

    return max(0.0, max_value), binding_group


def get_max_order_size(
    account: MinerAccount,
    position: Position,
    open_positions: list[Position] | None = None,
    is_buy: bool = True,
    value_sign: float = 1.0,
) -> tuple[float, str]:
    """Return (max_usd_value, binding_cap_label) for this position.

    Computes remaining room as min across all applicable caps:
      - per_pair_room:   max position size for this pair minus current exposure  (buys only)
      - per_class_room:  per-asset-class portfolio cap minus class exposure  (subaccounts, buys)
      - overall_room:    overall portfolio cap minus total exposure           (subaccounts, buys)
      - correlated_room: net exposure cap across correlated pairs             (pro accounts only)

    Orders that reduce a position (`is_buy=False`) consume no buying power, so only the
    correlated cap applies to them. `open_positions` must include this position and is only
    consulted for pro accounts.
    """
    trade_pair = position.trade_pair
    limits = []

    if is_buy:
        if account.miner_bucket and account.miner_bucket.is_subaccount:
            tier = get_leverage_tier(account.miner_bucket, account.account_size)
            max_position_leverage = get_tier_positional_leverage(tier, trade_pair)
        else:
            max_position_leverage = trade_pair.max_leverage

        per_pair_room = account.balance * max_position_leverage - abs(position.net_value)
        portfolio_room = account.buying_power
        limits += [
            (per_pair_room,  f"per pair cap {max_position_leverage}x"),
            (portfolio_room, f"overall portfolio cap {account.multiplier}x"),
        ]

        if account.miner_bucket and account.miner_bucket.is_subaccount:
            if not account.asset_class:
                raise ValueError("asset_class must be selected for trading")
            per_class_cap, overall_cap = get_portfolio_caps(
                account.asset_class, account.miner_bucket, account.account_size, trade_pair.trade_pair_category
            )
            per_class_used = account.capital_used_by_class.get(trade_pair.trade_pair_category, 0.0)
            per_class_room = account.balance * per_class_cap - per_class_used
            limits.append((per_class_room, f"per class cap {trade_pair.trade_pair_category.value} {per_class_cap}x"))

    if account.miner_bucket and account.miner_bucket.is_pro and open_positions is not None:
        correlated_room, correlated_label = get_max_correlated_order_size(
            trade_pair, open_positions, account.balance, value_sign
        )
        if correlated_label:
            limits.append((correlated_room, correlated_label))

    if not limits:
        return float("inf"), "uncapped"

    max_value, binding_cap = min(limits, key=lambda x: x[0])

    if is_buy:
        transaction_fee_rate = trade_pair.transaction_fee_rate()
        if max_value * (1 + transaction_fee_rate * account.multiplier) > account.buying_power:
            max_value = max_value / (1 + transaction_fee_rate * account.multiplier)

    return max(0.0, max_value), binding_cap


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
