from enum import Enum


class AccountType(str, Enum):
    """Which account tier an entity subaccount belongs to. Chosen once at subaccount creation
    (SubaccountInfo.account_type) and immutable afterward. Determines the subaccount's bucket
    track, fee schedule, challenge period rules, and permitted trade pairs."""
    STANDARD = "standard"  # SUBACCOUNT_CHALLENGE -> SUBACCOUNT_FUNDED
    PRO = "pro"            # SUBACCOUNT_PRO_CHALLENGE -> SUBACCOUNT_PRO_FUNDED

    @staticmethod
    def is_valid(account_type: str) -> bool:
        """True if `account_type` (case-insensitive) is a valid AccountType value."""
        if not isinstance(account_type, str):
            return False
        return account_type.lower() in {t.value for t in AccountType}

    @property
    def challenge_bucket(self):
        """The bucket a newly created subaccount of this type starts in."""
        # Deferred import: miner_bucket_enum pulls in ValiConfig.
        from vali_objects.enums.miner_bucket_enum import MinerBucket
        if self == AccountType.PRO:
            return MinerBucket.SUBACCOUNT_PRO_CHALLENGE
        return MinerBucket.SUBACCOUNT_CHALLENGE
