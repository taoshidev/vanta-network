from pydantic import BaseModel


class DividendEvent(BaseModel):
    """Market-level dividend event from corporate actions data."""
    gross_dividend: float
    ex_date: str        # YYYY-MM-DD
    payment_date: str   # YYYY-MM-DD


class DividendHistoryEntry(BaseModel):
    """Position-level record of a dividend application."""
    type: str           # "long_credit" | "short_debit"
    gross_dividend: float
    quantity: float
    amount: float
    ex_date: str
    payment_date: str
    time_ms: int
    applied: bool = False


class CorporateActions(BaseModel):
    """All corporate actions on a given date, keyed by symbol."""
    splits: dict[str, float]              # symbol -> ratio_new / ratio_old
    dividends: dict[str, DividendEvent]   # symbol -> DividendEvent
