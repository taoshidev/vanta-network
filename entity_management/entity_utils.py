# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Entity utility functions for synthetic hotkey parsing and validation.

These are static utility functions that can be called without RPC overhead.
"""
import math
from typing import Tuple, Optional
from shared_objects.log import logger
from vali_objects.vali_config import ValiConfig


def is_synthetic_hotkey(hotkey: str) -> bool:
    """
    Check if a hotkey is synthetic (contains underscore with integer suffix).

    This is a static utility function that does not require RPC calls.
    Synthetic hotkeys follow the pattern: {entity_hotkey}_{subaccount_id}

    Edge case: If an entity hotkey itself contains an underscore, we check
    if the part after the last underscore is a valid integer to distinguish
    synthetic hotkeys from entity hotkeys with underscores.

    Args:
        hotkey: The hotkey to check

    Returns:
        True if synthetic (format: base_123), False otherwise

    Examples:
        is_synthetic_hotkey("entity_123")
            True
        is_synthetic_hotkey("my_entity_0")
            True
        is_synthetic_hotkey("foo_bar_99")
            True
        is_synthetic_hotkey("regular_hotkey")
            False
        is_synthetic_hotkey("no_number_")
            False
        is_synthetic_hotkey("just_text")
            False
    """
    if "_" not in hotkey:
        return False

    # Try to parse as synthetic hotkey
    parts = hotkey.rsplit("_", 1)
    if len(parts) != 2:
        return False

    try:
        int(parts[1])  # Check if last part is a valid integer
        return True
    except ValueError:
        return False


def parse_synthetic_hotkey(synthetic_hotkey: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse a synthetic hotkey into entity_hotkey and subaccount_id.

    This is a static utility function that does not require RPC calls.

    Args:
        synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

    Returns:
        (entity_hotkey, subaccount_id) or (None, None) if invalid

    Examples:
        parse_synthetic_hotkey("entity_123")
            ("entity", 123)
        parse_synthetic_hotkey("my_entity_0")
            ("my_entity", 0)
        parse_synthetic_hotkey("foo_bar_99")
            ("foo_bar", 99)
        parse_synthetic_hotkey("invalid")
            (None, None)
    """
    if not is_synthetic_hotkey(synthetic_hotkey):
        return None, None

    parts = synthetic_hotkey.rsplit("_", 1)
    entity_hotkey = parts[0]
    try:
        subaccount_id = int(parts[1])
        return entity_hotkey, subaccount_id
    except ValueError:
        return None, None


def pro_account_size_error(pro_account_size, standard_account_size=None) -> Optional[str]:
    """
    Why pro_account_size cannot be used as a pro account size, or None when it can.

    A pro account size is an int or float (never a bool) that is finite, positive and at most
    ValiConfig.MAX_PRO_ACCOUNT_SIZE. The size presets offered by the Command Center are a UI concern.

    `standard_account_size`, when known, is the floor

    Sizes arrive as parsed JSON, and Python's JSON parser (so Flask's request.get_json) accepts the
    literals NaN, Infinity and -Infinity. NaN fails every ordered comparison, so a plain range check
    lets it through: finiteness is checked first. Integers are always finite (and may be too large to
    convert to float), so only floats go through math.isfinite.
    """
    if isinstance(pro_account_size, bool) or not isinstance(pro_account_size, (int, float)):
        return f"pro_account_size must be a number, got {type(pro_account_size).__name__}"
    if isinstance(pro_account_size, float) and not math.isfinite(pro_account_size):
        return f"pro_account_size must be a finite number, got {pro_account_size}"
    if not 0 < pro_account_size <= ValiConfig.MAX_PRO_ACCOUNT_SIZE:
        return (
            f"pro_account_size ${pro_account_size} must be positive and at most "
            f"${ValiConfig.MAX_PRO_ACCOUNT_SIZE:,}"
        )
    if standard_account_size is not None and pro_account_size < standard_account_size:
        return (
            f"pro_account_size ${pro_account_size:,} is below the subaccount's standard account size "
            f"${standard_account_size:,}"
        )
    return None


def pro_payout_scale(standard_account_size, pro_account_size) -> float:
    """
    Multiplier applied to a pro account's PnL when the subaccount is paid on its standard size.

    A trader running the pro challenge after passing the standard challenge trades the larger pro
    account but is paid on their standard account, uplifted by
    ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER. Returns 1.0 when either size is missing, so a
    subaccount that never entered the pro track is paid on its own PnL unchanged.

    Both payout paths (EntityManager.get_payout_scale and the debt-ledger aggregation) read this,
    so the number a trader is quoted is the number the weight calculator pays.
    """
    if not standard_account_size or not pro_account_size:
        return 1.0
    return ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER * standard_account_size / pro_account_size


def attach_correlated_exposure_report(account_size_data: dict | None) -> None:
    """Expand a pro account's stored correlated exposure into limits and remaining room.

    The account carries raw gross [long, short] per group; a client also needs the cap and the
    headroom, which are a function of the account's balance. Non-pro accounts get nothing, and a
    pro account with no exposure gets an empty `groups` so a client can tell the two apart.
    """
    if not account_size_data or not account_size_data.get("is_pro"):
        return
    # Local import: leverage_utils -> miner_account_manager -> this module.
    from vali_objects.utils.leverage_utils import build_correlated_exposure_report

    exposures = {
        group_key: (sides[0], sides[1])
        for group_key, sides in (account_size_data.get("correlated_exposure_by_group") or {}).items()
    }
    account_size_data["correlated_exposures"] = build_correlated_exposure_report(
        exposures, account_size_data.get("balance", 0.0)
    )


def create_subaccount_dashboard(
    synthetic_hotkey: str,
    subaccount_dashboard: dict | None,
    challenge_period_client,
    elimination_client,
    miner_account_client,
    position_client,
    limit_order_client,
    debt_ledger_client,
    statistics_client,
    positions_time_ms: int,
    limit_orders_time_ms: int,
    checkpoints_time_ms: int,
    daily_returns_time_ms: int,
) -> dict:
    """
    Aggregates the per-section dashboards for a hotkey. `subaccount_dashboard`
    is entity/subaccount-specific info; pass None for a regular miner hotkey
    to omit the "subaccount_info" section.
    """
    dashboard = {} if subaccount_dashboard is None else {"subaccount_info": subaccount_dashboard}

    # Fail gracefully if other services are not available
    def add_to_dashboard(section, function, *args, **kwargs):
        try:
            # Assume the first parameter is the synthetic_hotkey
            section_data = function(synthetic_hotkey, *args, **kwargs)
            if section_data is not None:
                dashboard[section] = section_data
        except Exception as ex:
            logger.error(f"Error retrieving {section} for {synthetic_hotkey}: {ex}")

    add_to_dashboard("challenge_period", challenge_period_client.get_dashboard)
    add_to_dashboard("drawdown", challenge_period_client.get_drawdown_stats)
    add_to_dashboard("pro_stats", challenge_period_client.get_pro_stats)
    add_to_dashboard("elimination", elimination_client.get_dashboard)
    add_to_dashboard("account_size_data", miner_account_client.get_dashboard)
    attach_correlated_exposure_report(dashboard.get("account_size_data"))
    add_to_dashboard("positions", position_client.get_dashboard, positions_time_ms)
    add_to_dashboard("limit_orders", limit_order_client.get_dashboard, limit_orders_time_ms)
    add_to_dashboard("ledger", debt_ledger_client.get_dashboard, checkpoints_time_ms)
    add_to_dashboard("statistics", statistics_client.get_dashboard, daily_returns_time_ms)

    return dashboard