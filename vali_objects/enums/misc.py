from enum import Enum, auto


class OrderStatus(Enum):
    OPEN = auto()
    CLOSED = auto()
    ALL = auto()  # Represents both or neither, depending on your logic


class SynapseMethod(Enum):
    POSITION_INSPECTOR = "GetPositions"
    SIGNAL = "SendSignal"
    CHECKPOINT = "SendCheckpoint"


class TradePairReturnStatus(Enum):
    TP_NO_OPEN_POSITIONS = 0
    TP_MARKET_NOT_OPEN = 1
    TP_MARKET_OPEN_NO_PRICE_CHANGE = 2
    TP_MARKET_OPEN_PRICE_CHANGE = 3

    # Define greater than oeprator for TradePairReturnStatus
    def __gt__(self, other):
        return self.value > other.value


class ShortcutReason(Enum):
    NO_SHORTCUT = 0
    NO_OPEN_POSITIONS = 1
    OUTSIDE_WINDOW = 2


class PenaltyInputType(Enum):
    LEDGER = auto()
    POSITIONS = auto()
    PSEUDO_POSITIONS = auto()
    COLLATERAL = auto()


class PositionSyncResult(Enum):
    NOTHING = 0
    UPDATED = 1
    DELETED = 2
    INSERTED = 3
