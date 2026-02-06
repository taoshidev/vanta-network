from enum import IntEnum


class OrderSource(IntEnum):
    """Enum representing the source/origin of an order."""
    ORGANIC = 0                        # order generated from a miner's signal
    ELIMINATION_FLAT = 1               # order inserted when a miner is eliminated (0 used for price. DEPRECATED)
    DEPRECATION_FLAT = 2               # order inserted when a trade pair is removed (0 used for price)
    PRICE_FILLED_ELIMINATION_FLAT = 3  # order inserted when a miner is eliminated but we price fill it accurately.
    MAX_ORDERS_PER_POSITION_CLOSE = 4  # order inserted when position hits max orders limit and needs to be closed
    LIMIT_UNFILLED = 5                 # limit order created but not yet filled
    LIMIT_FILLED = 6                   # limit order that was filled
    LIMIT_CANCELLED = 7                # limit order that was cancelled
    BRACKET_UNFILLED = 8               # bracket order (stop loss/take profit) created but not yet filled
    BRACKET_FILLED = 9                 # bracket order (stop loss/take profit) that was filled
    BRACKET_CANCELLED = 10             # bracket order (stop loss/take profit) that was cancelled
    FLAT_ALL_CLOSE = 11                # order inserted when miner sends FLAT_ALL to close all positions
    SUBACCOUNT_PROMOTION = 12          # order inserted when subaccount promoted from challenge to funded

    @staticmethod
    def get_fill(order_src):
        if order_src == OrderSource.LIMIT_UNFILLED:
            return OrderSource.LIMIT_FILLED
        elif order_src == OrderSource.BRACKET_UNFILLED:
            return OrderSource.BRACKET_FILLED
        elif order_src == OrderSource.ORGANIC:
            return OrderSource.ORGANIC
        else:
            return None

    @staticmethod
    def get_cancel(order_src):
        if order_src in [OrderSource.LIMIT_UNFILLED, OrderSource.LIMIT_FILLED]:
            return OrderSource.LIMIT_CANCELLED
        elif order_src in [OrderSource.BRACKET_UNFILLED, OrderSource.BRACKET_FILLED]:
            return OrderSource.BRACKET_CANCELLED
        elif order_src == OrderSource.ORGANIC:
            return OrderSource.ORGANIC
        else:
            return None

    @staticmethod
    def is_open(src) -> bool:
        return src in {
            OrderSource.LIMIT_UNFILLED,
            OrderSource.BRACKET_UNFILLED,
        }

    @staticmethod
    def is_closed(src) -> bool:
        return src in {
            OrderSource.LIMIT_FILLED,
            OrderSource.LIMIT_CANCELLED,
            OrderSource.BRACKET_FILLED,
            OrderSource.BRACKET_CANCELLED
        }

    @staticmethod
    def is_cancelled(src) -> bool:
        return src in {
            OrderSource.LIMIT_CANCELLED,
            OrderSource.BRACKET_CANCELLED,
        }

    @staticmethod
    def status(order_src) -> str:
        if order_src in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
            return "UNFILLED"
        elif order_src in [OrderSource.LIMIT_FILLED, OrderSource.BRACKET_FILLED]:
            return "FILLED"
        elif order_src in [OrderSource.LIMIT_CANCELLED, OrderSource.BRACKET_CANCELLED]:
            return "CANCELLED"
        else:
            return OrderSource(order_src).name
