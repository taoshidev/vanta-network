from enum import Enum


class DrawdownCriteria(str, Enum):
    """Which drawdown rule set applies to a miner. Miners on the existing rules use TRAILING unless an admin
    sets otherwise, and standard miners use STATIC. For entity subaccounts, this is chosen once at subaccount
    creation (SubaccountInfo.drawdown_criteria) and is immutable afterward"""
    TRAILING = "trailing"   # intraday + EOD trailing (miners on the existing rules and subaccounts opted into legacy rules)
    STATIC = "static"        # static equity vs starting balance + intraday drawdown vs day-open equity (standard miners, and subaccounts opted in at creation)
