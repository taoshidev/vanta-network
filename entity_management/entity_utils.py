# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Entity utility functions for synthetic hotkey parsing and validation.

These are static utility functions that can be called without RPC overhead.
"""
import bittensor as bt
from typing import Tuple, Optional


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


def create_subaccount_dashboard(
    synthetic_hotkey: str,
    subaccount_dashboard: dict,
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
    dashboard = {"subaccount_info": subaccount_dashboard}

    # Fail gracefully if other services are not available
    def add_to_dashboard(section, function, *args, **kwargs):
        try:
            # Assume the first parameter is the synthetic_hotkey
            section_data = function(synthetic_hotkey, *args, **kwargs)
            if section_data is not None:
                dashboard[section] = section_data
        except Exception as ex:
            bt.logging.error(f"Error retrieving {section} for {synthetic_hotkey}: {ex}")

    add_to_dashboard("challenge_period", challenge_period_client.get_dashboard)
    add_to_dashboard("drawdown", challenge_period_client.get_drawdown_stats)
    add_to_dashboard("elimination", elimination_client.get_dashboard)
    add_to_dashboard("account_size_data", miner_account_client.get_dashboard)
    add_to_dashboard("positions", position_client.get_dashboard, positions_time_ms)
    add_to_dashboard("limit_orders", limit_order_client.get_dashboard, limit_orders_time_ms)
    add_to_dashboard("ledger", debt_ledger_client.get_dashboard, checkpoints_time_ms)
    add_to_dashboard("statistics", statistics_client.get_dashboard, daily_returns_time_ms)

    return dashboard