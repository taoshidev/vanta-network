from dataclasses import dataclass
from enum import Enum

from vali_objects.vali_config import ValiConfig


class MinerBucket(Enum):
    MAINCOMP = "MAINCOMP"
    CHALLENGE = "CHALLENGE"
    PROBATION = "PROBATION"
    PLAGIARISM = "PLAGIARISM"
    ELIMINATED = "ELIMINATED"
    UNKNOWN = "unknown"
    # Entity system buckets
    ENTITY = "ENTITY"
    SUBACCOUNT_CHALLENGE = "SUBACCOUNT_CHALLENGE"
    SUBACCOUNT_FUNDED = "SUBACCOUNT_FUNDED"
    SUBACCOUNT_ALPHA = "SUBACCOUNT_ALPHA"
    # Pro account buckets
    PRO_CHALLENGE_TRANSITION = "PRO_CHALLENGE_TRANSITION"
    PRO_CHALLENGE_FROM_STANDARD = "PRO_CHALLENGE_FROM_STANDARD"
    PRO_CHALLENGE_DIRECT = "PRO_CHALLENGE_DIRECT"
    PRO_FUNDED = "PRO_FUNDED"

    def intraday_drawdown_threshold(self, time_ms: int | None = None) -> float:
        """
        Returns the intraday drawdown threshold for this bucket from ValiConfig.
        For SUBACCOUNT_FUNDED and PRO_CHALLENGE_TRANSITION, time_ms is the miner's
        SUBACCOUNT_CHALLENGE registration timestamp and determines which versioned threshold
        applies. Pro buckets are not versioned - they have no legacy tier.
        """
        if self in (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_CHALLENGE_DIRECT):
            return ValiConfig.PRO_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD

        if self == MinerBucket.PRO_FUNDED:
            return ValiConfig.PRO_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.CHALLENGE):
            return ValiConfig.CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD

        # TRANSITION is still trading the standard funded account, so it keeps funded rules
        if self in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.PRO_CHALLENGE_TRANSITION):
            if time_ms is not None and time_ms < ValiConfig.FUNDED_V0_CUTOFF_MS:
                return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V0
            if time_ms is not None and time_ms < ValiConfig.FUNDED_V1_CUTOFF_MS:
                return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V1
            return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.MAINCOMP, MinerBucket.PROBATION, MinerBucket.SUBACCOUNT_ALPHA, MinerBucket.PLAGIARISM):
            return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        raise ValueError(f"No intraday drawdown threshold defined for bucket {self}")

    def eod_drawdown_threshold(self, time_ms: int | None = None) -> float:
        """
        Returns the intraday drawdown threshold for this bucket from ValiConfig.
        For SUBACCOUNT_FUNDED and PRO_CHALLENGE_TRANSITION, time_ms is the miner's
        SUBACCOUNT_CHALLENGE registration timestamp and determines which versioned threshold
        applies.
        """
        if self in (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_CHALLENGE_DIRECT):
            return ValiConfig.PRO_CHALLENGE_EOD_DRAWDOWN_THRESHOLD

        if self == MinerBucket.PRO_FUNDED:
            return ValiConfig.PRO_FUNDED_EOD_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.CHALLENGE):
            return ValiConfig.CHALLENGE_EOD_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.PRO_CHALLENGE_TRANSITION):
            if time_ms is not None and time_ms < ValiConfig.FUNDED_V0_CUTOFF_MS:
                return ValiConfig.FUNDED_EOD_DRAWDOWN_THRESHOLD_V0
            return ValiConfig.FUNDED_EOD_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.MAINCOMP, MinerBucket.PROBATION, MinerBucket.SUBACCOUNT_ALPHA, MinerBucket.PLAGIARISM):
            return ValiConfig.FUNDED_EOD_DRAWDOWN_THRESHOLD

        raise ValueError(f"No intraday drawdown threshold defined for bucket {self}")

    @property
    def is_regular_miner(self) -> bool:
        return self in (MinerBucket.CHALLENGE, MinerBucket.PROBATION, MinerBucket.MAINCOMP)

    @property
    def is_subaccount(self) -> bool:
        return self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA,
                        MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
                        MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_FUNDED)

    @property
    def is_pro(self) -> bool:
        """True for buckets trading the pro account. Excludes TRANSITION, which is still on the
        standard account and so keeps standard fees, leverage, and permitted trade pairs."""
        return self in (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_CHALLENGE_DIRECT,
                        MinerBucket.PRO_FUNDED)

    @property
    def is_pro_track(self) -> bool:
        """True for every bucket on the pro journey, including the transition week."""
        return self.is_pro or self == MinerBucket.PRO_CHALLENGE_TRANSITION

    @property
    def calmar_threshold(self) -> float | None:
        """Minimum all-time calmar required for promotion. None for buckets with no calmar requirement."""
        return ValiConfig.PRO_CHALLENGE_CALMAR_THRESHOLD if self.is_pro else None

    @property
    def daily_consistency_threshold(self) -> float | None:
        """Maximum return consistency allowed for promotion. None for buckets with no consistency requirement."""
        return ValiConfig.PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD if self.is_pro else None

    def returns_threshold(self, asset_class) -> float:
        """Return required for promotion out of this bucket, by the miner's asset class."""
        if self.is_pro:
            return ValiConfig.PRO_CHALLENGE_RETURNS_THRESHOLD.get(
                asset_class, ValiConfig.PRO_CHALLENGE_RETURNS_THRESHOLD_DEFAULT)
        return ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD.get(
            asset_class, ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD_DEFAULT)

    @property
    def soft_breach_applies(self) -> bool:
        """True for buckets where a pro-rule breach withholds the week's payout.
        Miners who already passed the standard challenge keep earning through a soft breach."""
        return self in (MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_FUNDED)

    @property
    def payout_scale_applies(self) -> bool:
        """True for buckets whose PnL is scaled by standard_account_size / pro_account_size."""
        return self == MinerBucket.PRO_CHALLENGE_FROM_STANDARD

    @property
    def is_subaccount_challenge(self) -> bool:
        """True for a subaccount's challenge bucket on either track."""
        return self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
                        MinerBucket.PRO_CHALLENGE_DIRECT)

    @property
    def is_subaccount_funded(self) -> bool:
        """True for either subaccount track's funded bucket (the promotion target)."""
        return self in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.PRO_FUNDED)

    @property
    def is_subaccount_earning(self) -> bool:
        """True for subaccount buckets that earn payouts, carry margin requirements, and are
        subject to entity collateral slashing."""
        return self.is_subaccount_funded or self in (
            MinerBucket.SUBACCOUNT_ALPHA,
            MinerBucket.PRO_CHALLENGE_TRANSITION,
            MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
        )

    @property
    def next_bucket(self) -> "MinerBucket | None":
        if self == MinerBucket.CHALLENGE:
            return MinerBucket.MAINCOMP
        elif self == MinerBucket.PROBATION:
            return MinerBucket.MAINCOMP
        elif self == MinerBucket.SUBACCOUNT_CHALLENGE:
            return MinerBucket.SUBACCOUNT_FUNDED
        elif self == MinerBucket.PRO_CHALLENGE_TRANSITION:
            return MinerBucket.PRO_CHALLENGE_FROM_STANDARD
        elif self in (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_CHALLENGE_DIRECT):
            return MinerBucket.PRO_FUNDED
        # TODO determine if we need alpha or keep subaccounts as funded
        # elif self == MinerBucket.SUBACCOUNT_FUNDED:
        #     return MinerBucket.SUBACCOUNT_ALPHA
        return None

    @property
    def demotion_bucket(self) -> "MinerBucket | None":
        """Where a miner lands when they fail out of this bucket instead of being eliminated.
        A pro challenge failure returns the miner to the standard track they came from."""
        if self == MinerBucket.PRO_CHALLENGE_FROM_STANDARD:
            return MinerBucket.SUBACCOUNT_FUNDED
        elif self == MinerBucket.PRO_CHALLENGE_DIRECT:
            return MinerBucket.SUBACCOUNT_CHALLENGE
        return None

    @property
    def switches_account(self) -> bool:
        """True when moving *into* this bucket changes the account size, which requires closing
        positions, cancelling limit orders, and restarting the ledgers."""
        return self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED,
                        MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_CHALLENGE_DIRECT)

    @property
    def max_time_ms(self) -> int | None:
        if self == MinerBucket.CHALLENGE:
            return ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS
        elif self == MinerBucket.PROBATION:
            return ValiConfig.PROBATION_MAXIMUM_MS
        elif self == MinerBucket.PLAGIARISM:
            return   ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS
        else:
            return None

    @property
    def grace_period_ms(self) -> int | None:
        """Time in this bucket before the miner is advanced to next_bucket automatically."""
        if self == MinerBucket.PRO_CHALLENGE_TRANSITION:
            return ValiConfig.PRO_TRANSITION_GRACE_PERIOD_MS
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
    def is_active(self):
        return self in (
                MinerBucket.CHALLENGE,
                MinerBucket.MAINCOMP,
                MinerBucket.PROBATION,
                MinerBucket.PLAGIARISM,
                MinerBucket.SUBACCOUNT_CHALLENGE,
                MinerBucket.SUBACCOUNT_FUNDED,
                MinerBucket.SUBACCOUNT_ALPHA,
                MinerBucket.PRO_CHALLENGE_TRANSITION,
                MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
                MinerBucket.PRO_CHALLENGE_DIRECT,
                MinerBucket.PRO_FUNDED
                )

@dataclass
class BucketEntry:
    bucket: MinerBucket
    start_time_ms: int

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            'bucket': self.bucket.value,
            'start_time_ms': self.start_time_ms,
            'bucket_start_time': self.start_time_ms  # for backwards compatibility TODO remove
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BucketEntry':
        """Create from dict for deserialization."""
        bucket = d.get('bucket', MinerBucket.UNKNOWN)
        if isinstance(bucket, str):
            bucket = MinerBucket(bucket)
        return cls(
            bucket=bucket,
            start_time_ms=d.get('start_time_ms') or d.get('bucket_start_time', 0)
        )
