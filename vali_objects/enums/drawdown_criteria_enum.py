from enum import Enum


class DrawdownCriteria(str, Enum):
    """Which drawdown rule set applies to a miner. Regular miners always use TRAILING. For entity
    subaccounts, this is chosen once at subaccount creation (SubaccountInfo.drawdown_criteria) and
    is immutable afterward"""
    TRAILING = "trailing"   # intraday + EOD trailing (regular miners and subaccounts opted into legacy rules)
    STATIC = "static"        # static balance + static EOD (subaccounts opted into static rules at creation)
