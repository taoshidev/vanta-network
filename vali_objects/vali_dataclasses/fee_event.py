from enum import Enum

from pydantic import BaseModel


class FeeType(str, Enum):
    """Types of fees that can accrue on a position."""
    CARRY = "carry"
    HL_FUNDING = "hl_funding"
    BORROW = "borrow"
    INTEREST = "interest"
    TRANSACTION = "transaction"
    DIVIDEND_LIABILITY = "dividend_liability"


class FeeEvent(BaseModel):
    """Position-level record of a fee applied to a position."""
    fee_type: FeeType
    amount: float
    time_ms: int
