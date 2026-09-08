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

    @property
    def challenge_bucket(self):
        """The bucket a newly created subaccount starts in. Pro is not reachable at creation."""
        # Deferred import: miner_bucket_enum pulls in ValiConfig.
        from vali_objects.enums.miner_bucket_enum import MinerBucket
        return MinerBucket.SUBACCOUNT_CHALLENGE
