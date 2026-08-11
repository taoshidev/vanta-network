from pydantic import BaseModel


class FeeEvent(BaseModel):
    """Position-level record of a fee applied to a position."""
    fee_type: str  # e.g. "carry", "interest", "borrow", "spread", "slippage"
    amount: float
    time_ms: int
