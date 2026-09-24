from enum import Enum


class AccountType(str, Enum):
    """Which account tier an entity subaccount belongs to. Every subaccount is created as
    STANDARD; PRO is set only by admin promotion into the pro bucket track, and determines
    the subaccount's fee schedule, challenge period rules, and permitted trade pairs."""
    STANDARD = "standard"  # SUBACCOUNT_CHALLENGE -> SUBACCOUNT_FUNDED
    PRO = "pro"            # PRO_CHALLENGE_* -> PRO_FUNDED

    @staticmethod
    def is_valid(account_type: str) -> bool:
        """True if `account_type` (case-insensitive) is a valid AccountType value."""
        if not isinstance(account_type, str):
            return False
        return account_type.lower() in {t.value for t in AccountType}
