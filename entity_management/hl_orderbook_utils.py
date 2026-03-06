"""
Utility for simulating order fills against an L2 orderbook.
Extracted from hl_vanta_translation/calc_slippage.py for reuse by HyperliquidTracker.
"""


def simulate_fill(levels, size, unit="usd"):
    """Walk through orderbook levels and fill the order.

    Args:
        levels: List of orderbook level dicts with 'px' and 'sz' keys.
        size: Order size (in coins or USD depending on unit).
        unit: 'coins' or 'usd'.

    Returns:
        Tuple of (fills, remaining) where fills is a list of (price, filled_coins, filled_usd)
        per level and remaining is the unfilled amount.
    """
    remaining = size
    fills = []
    for level in levels:
        if remaining <= 0:
            break
        px = float(level["px"])
        sz = float(level["sz"])
        if unit == "coins":
            fill_coins = min(sz, remaining)
            fill_usd = fill_coins * px
            remaining -= fill_coins
        else:  # usd
            max_usd = sz * px
            fill_usd = min(max_usd, remaining)
            fill_coins = fill_usd / px
            remaining -= fill_usd
        fills.append((px, fill_coins, fill_usd))
    return fills, remaining
