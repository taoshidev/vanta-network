from enum import Enum


class DrawdownCriteria(str, Enum):
    TRAILING = "trailing"   # intraday + EOD trailing (regular miners and pre-effective subaccounts)
    STATIC = "static"       # static balance + static EOD (post-effective subaccounts, non-HL)
