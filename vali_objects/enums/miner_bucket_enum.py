from dataclasses import dataclass
from enum import Enum

from vali_objects.vali_config import ValiConfig

# Registration cutoffs for versioned SUBACCOUNT_FUNDED thresholds (ms)
_FUNDED_V0_CUTOFF_MS = 1773532799000  # Mar 14, 2026 23:59:59 UTC
_FUNDED_V1_CUTOFF_MS = 1778828399000  # May 15, 2026 06:59:59 UTC

class MinerBucket(Enum):
    MAINCOMP = "MAINCOMP"
    CHALLENGE = "CHALLENGE"
    PROBATION = "PROBATION"
    PLAGIARISM = "PLAGIARISM"
    UNKNOWN = "unknown"
    # Entity system buckets
    ENTITY = "ENTITY"
    SUBACCOUNT_CHALLENGE = "SUBACCOUNT_CHALLENGE"
    SUBACCOUNT_FUNDED = "SUBACCOUNT_FUNDED"
    SUBACCOUNT_ALPHA = "SUBACCOUNT_ALPHA"


    def intraday_drawdown_threshold(self, time_ms: int | None = None) -> float:
        """
        Returns the intraday drawdown threshold for this bucket from ValiConfig.
        For SUBACCOUNT_FUNDED, time_ms is the miner's SUBACCOUNT_CHALLENGE registration
        timestamp and determines which versioned threshold applies.
        """
        if self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.CHALLENGE):
            return ValiConfig.CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD

        if self == MinerBucket.SUBACCOUNT_FUNDED:
            if time_ms is not None and time_ms < _FUNDED_V0_CUTOFF_MS:
                return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V0
            if time_ms is not None and time_ms < _FUNDED_V1_CUTOFF_MS:
                return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V1
            return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.MAINCOMP, MinerBucket.PROBATION, MinerBucket.SUBACCOUNT_ALPHA):
            return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        raise ValueError(f"No intraday drawdown threshold defined for bucket {self}")

    def eod_drawdown_threshold(self, time_ms: int | None = None) -> float:
        """
        Returns the intraday drawdown threshold for this bucket from ValiConfig.
        For SUBACCOUNT_FUNDED, time_ms is the miner's SUBACCOUNT_CHALLENGE registration
        timestamp and determines which versioned threshold applies.
        """
        if self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.CHALLENGE):
            return ValiConfig.CHALLENGE_EOD_DRAWDOWN_THRESHOLD

        if self == MinerBucket.SUBACCOUNT_FUNDED:
            if time_ms is not None and time_ms < _FUNDED_V0_CUTOFF_MS:
                return ValiConfig.FUNDED_EOD_DRAWDOWN_THRESHOLD_V0
            return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.MAINCOMP, MinerBucket.PROBATION, MinerBucket.SUBACCOUNT_ALPHA):
            return ValiConfig.FUNDED_EOD_DRAWDOWN_THRESHOLD

        raise ValueError(f"No intraday drawdown threshold defined for bucket {self}")

    @property
    def next_bucket(self) -> "MinerBucket | None":
        if self == MinerBucket.CHALLENGE:
            return MinerBucket.MAINCOMP
        elif self == MinerBucket.PROBATION:
            return MinerBucket.MAINCOMP
        elif self == MinerBucket.SUBACCOUNT_CHALLENGE:
            return MinerBucket.SUBACCOUNT_FUNDED
        # TODO determine if we need alpha or keep subaccounts as funded
        # elif self == MinerBucket.SUBACCOUNT_FUNDED:
        #     return MinerBucket.SUBACCOUNT_ALPHA
        return None

    @property
    def time_limit_ms(self) -> int | None:
        if self == MinerBucket.CHALLENGE:
            return ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS
        elif self == MinerBucket.PROBATION:
            return ValiConfig.PROBATION_MAXIMUM_MS
        else:
            return None

    @property
    def is_rank_based(self):
        return self in (
                MinerBucket.CHALLENGE,
                MinerBucket.MAINCOMP,
                MinerBucket.PROBATION,
                MinerBucket.SUBACCOUNT_ALPHA
                )

    @property
    def is_evaluation_eligible(self):
        return self in (
                MinerBucket.CHALLENGE,
                MinerBucket.MAINCOMP,
                MinerBucket.PROBATION,
                MinerBucket.SUBACCOUNT_CHALLENGE,
                MinerBucket.SUBACCOUNT_FUNDED,
                MinerBucket.SUBACCOUNT_ALPHA
                )
    @property
    def is_elimination_eligible(self):
        return self in (
                MinerBucket.CHALLENGE,
                MinerBucket.MAINCOMP,
                MinerBucket.PROBATION,
                MinerBucket.PLAGIARISM,
                MinerBucket.SUBACCOUNT_CHALLENGE,
                MinerBucket.SUBACCOUNT_FUNDED,
                MinerBucket.SUBACCOUNT_ALPHA
                )


@dataclass
class BucketEntry:
    bucket: MinerBucket
    start_time_ms: int

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            'bucket': self.bucket.value,
            'start_time_ms': self.start_time_ms
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BucketEntry':
        """Create from dict for deserialization."""
        bucket = d.get('bucket')
        if isinstance(bucket, str):
            bucket = MinerBucket(bucket)
        return cls(
            bucket=bucket,
            start_time_ms=d.get('start_time_ms', 0)
        )
