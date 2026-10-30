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
    # A funded standard subaccount. Moving onto the pro track starts its transition week
    # (MinerAccount.transition_start_ms): a wind-down still trading the standard account, which
    # ends in PRO_CHALLENGE_FUNDED.
    SUBACCOUNT_FUNDED = "SUBACCOUNT_FUNDED"
    SUBACCOUNT_ALPHA = "SUBACCOUNT_ALPHA"
    # Pro account buckets
    # A funded standard subaccount that has left its transition week: trading the pro account, paid
    # on the standard size while passing the pro challenge. Keeps its pre-merge value on the wire.
    PRO_CHALLENGE_FUNDED = "PRO_CHALLENGE_FROM_STANDARD"
    PRO_CHALLENGE_FROM_STANDARD = "PRO_CHALLENGE_FROM_STANDARD"  # alias of PRO_CHALLENGE_FUNDED
    PRO_CHALLENGE_DIRECT = "PRO_CHALLENGE_DIRECT"
    PRO_FUNDED = "PRO_FUNDED"

    @classmethod
    def _missing_(cls, value):
        """Parse the retired bucket names."""
        if value in LEGACY_BUCKET_NAMES:
            return cls(LEGACY_BUCKET_NAMES[value])
        return None

    @classmethod
    def from_status(cls, status: str | None) -> "MinerBucket":
        """Parse a stored status string, tolerating unknown values."""
        try:
            return cls(status)
        except ValueError:
            return cls.UNKNOWN

    @property
    def statuses(self) -> set[str]:
        """Every status string this bucket may have been stored as, retired names included."""
        return {self.value, *(name for name, value in LEGACY_BUCKET_NAMES.items() if value == self.value)}

    def intraday_drawdown_threshold(self, time_ms: int | None = None) -> float:
        """
        Returns the intraday drawdown threshold for this bucket from ValiConfig.
        For SUBACCOUNT_FUNDED (in or out of its transition week), time_ms is the miner's
        SUBACCOUNT_CHALLENGE registration timestamp and determines which versioned threshold applies. Pro buckets are not versioned - they have no
        legacy tier.
        """
        if self in (MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT):
            return ValiConfig.PRO_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD

        if self == MinerBucket.PRO_FUNDED:
            return ValiConfig.PRO_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.CHALLENGE):
            return ValiConfig.CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD

        if self == MinerBucket.SUBACCOUNT_FUNDED:
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
        For SUBACCOUNT_FUNDED (in or out of its transition week), time_ms is the miner's
        SUBACCOUNT_CHALLENGE registration timestamp and determines which versioned threshold
        applies.
        """
        if self in (MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT):
            return ValiConfig.PRO_CHALLENGE_EOD_DRAWDOWN_THRESHOLD

        if self == MinerBucket.PRO_FUNDED:
            return ValiConfig.PRO_FUNDED_EOD_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.CHALLENGE):
            return ValiConfig.CHALLENGE_EOD_DRAWDOWN_THRESHOLD

        if self == MinerBucket.SUBACCOUNT_FUNDED:
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
                        MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT,
                        MinerBucket.PRO_FUNDED)

    @property
    def is_pro(self) -> bool:
        """True for buckets trading the pro account. The SUBACCOUNT_FUNDED transition week is still
        on the standard account and so is not pro."""
        return self in (MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT,
                        MinerBucket.PRO_FUNDED)

    @property
    def is_pro_track(self) -> bool:
        """True for every bucket on the pro account. The SUBACCOUNT_FUNDED transition week is on the
        pro journey too; check MinerAccount.in_transition for it."""
        return self.is_pro

    @property
    def calmar_threshold(self) -> float | None:
        """Minimum all-time calmar required for promotion. None for buckets with no calmar requirement."""
        return ValiConfig.PRO_CHALLENGE_CALMAR_THRESHOLD if self.is_pro else None

    @property
    def daily_consistency_threshold(self) -> float | None:
        """Maximum return consistency allowed for promotion. None for buckets with no consistency requirement."""
        return ValiConfig.PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD if self.is_pro else None

    @property
    def minimum_trading_days(self) -> int | None:
        """Full days of tracked returns required before promotion. None for buckets with no day requirement."""
        return ValiConfig.PRO_CHALLENGE_MINIMUM_DAYS if self.is_pro else None

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
        return self == MinerBucket.PRO_FUNDED

    @property
    def daily_loss_from_balance(self) -> bool:
        """True for the standard buckets whose daily loss limit is measured from the day-open balance
        instead of the day-open equity, for subaccounts created through SUBACCOUNT_CHALLENGE that are
        not HL-linked. SUBACCOUNT_FUNDED includes its transition week, still on the standard account."""
        return self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED)

    @property
    def payout_scale_applies(self) -> bool:
        """True for buckets whose PnL is scaled by standard_account_size / pro_account_size."""
        return self == MinerBucket.PRO_CHALLENGE_FUNDED

    @property
    def is_subaccount_challenge(self) -> bool:
        """True for a subaccount's challenge bucket on either track."""
        return self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.PRO_CHALLENGE_FUNDED,
                        MinerBucket.PRO_CHALLENGE_DIRECT)

    @property
    def is_subaccount_funded(self) -> bool:
        """True for either subaccount track's funded buckets."""
        return self in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.PRO_FUNDED)

    @property
    def is_subaccount_earning(self) -> bool:
        """True for subaccount buckets that earn payouts, carry margin requirements, and are
        subject to entity collateral slashing."""
        return self.is_subaccount_funded or self in (
            MinerBucket.SUBACCOUNT_ALPHA,
            MinerBucket.PRO_CHALLENGE_FUNDED,
        )

    @property
    def next_bucket(self) -> "MinerBucket | None":
        if self == MinerBucket.CHALLENGE:
            return MinerBucket.MAINCOMP
        elif self == MinerBucket.PROBATION:
            return MinerBucket.MAINCOMP
        elif self == MinerBucket.SUBACCOUNT_CHALLENGE:
            return MinerBucket.SUBACCOUNT_FUNDED
        elif self in (MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT):
            return MinerBucket.PRO_FUNDED
        # TODO determine if we need alpha or keep subaccounts as funded
        # elif self == MinerBucket.SUBACCOUNT_FUNDED:
        #     return MinerBucket.SUBACCOUNT_ALPHA
        return None

    @property
    def promotion_target(self) -> "MinerBucket | None":
        """Where a miner may promote itself with its own signed request (POST /entity/subaccount/promote).

        These hops onto the pro track are the only self-service bucket moves on the network; every
        other bucket returns None and cannot be promoted this way. SUBACCOUNT_FUNDED first enters its
        transition week, which stays in SUBACCOUNT_FUNDED, and leaves it for PRO_CHALLENGE_FUNDED
        (ChallengePeriodManager.promote_subaccount).
        """
        if self == MinerBucket.SUBACCOUNT_CHALLENGE:
            return MinerBucket.PRO_CHALLENGE_DIRECT
        elif self == MinerBucket.SUBACCOUNT_FUNDED:
            return MinerBucket.PRO_CHALLENGE_FUNDED
        return None

    @property
    def switches_account(self) -> bool:
        """True when moving *into* this bucket requires closing
        positions, cancelling limit orders, and restarting the ledgers.

        PRO_FUNDED is deliberately absent: passing the pro challenge keeps the same account, so
        balance, equity and the ledgers carry over and all-time calmar keeps the ratio it passed
        with. Its positions are still closed (closes_positions_from). Nothing earned before the promotion is paid at pro scale - the payout paths
        exclude it by reading each checkpoint's own bucket instead of wiping the history.

        Entering the SUBACCOUNT_FUNDED transition week is not a bucket move and keeps the
        standard account.
        """
        return self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED,
                        MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT)

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
                MinerBucket.PRO_CHALLENGE_FUNDED,
                MinerBucket.PRO_CHALLENGE_DIRECT,
                MinerBucket.PRO_FUNDED
                )

    def closes_positions_from(self, from_bucket: "MinerBucket") -> bool:
        """True when entering this bucket from from_bucket closes the open positions: every move into
        or out of an earning bucket, so each earning bucket is only paid on what it trades itself.
        An elimination forfeits instead."""
        return ((from_bucket.is_subaccount_earning or self.is_subaccount_earning)
                and self != from_bucket
                and self != MinerBucket.ELIMINATED)


# Bucket names that are not a bucket value, mapped to the value they parse as. The transition week
# became MinerAccount.transition_start_ms on SUBACCOUNT_FUNDED (still reported by name through
# api_bucket_name), and PRO_CHALLENGE_FUNDED is accepted as a spelling of its wire value.
LEGACY_BUCKET_NAMES = {
    "PRO_CHALLENGE_TRANSITION": "SUBACCOUNT_FUNDED",
    "PRO_CHALLENGE_FUNDED": "PRO_CHALLENGE_FROM_STANDARD",
}
LEGACY_TRANSITION_BUCKET = "PRO_CHALLENGE_TRANSITION"


def api_bucket_name(bucket: "MinerBucket | str | None", in_transition: bool = False) -> str | None:
    """Bucket name for API output and ledger checkpoint statuses: the SUBACCOUNT_FUNDED transition
    week reads as the retired PRO_CHALLENGE_TRANSITION while ValiConfig.API_LEGACY_BUCKET_NAMES is on."""
    if bucket is None:
        return None
    value = bucket.value if isinstance(bucket, MinerBucket) else bucket
    if in_transition and value == MinerBucket.SUBACCOUNT_FUNDED.value and ValiConfig.API_LEGACY_BUCKET_NAMES:
        return LEGACY_TRANSITION_BUCKET
    return value


@dataclass
class BucketEntry:
    bucket: MinerBucket
    start_time_ms: int
    # When set, this span was undone (an elimination that was reverted). The entry is kept as the
    # record that it happened, but point-in-time lookups skip it: bucket history is replayed to
    # stamp historical checkpoints, so leaving a reverted span visible would permanently reclassify
    # every checkpoint inside it and change payout weeks that were already settled.
    reverted_ms: int | None = None

    @property
    def is_reverted(self) -> bool:
        return self.reverted_ms is not None

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            'bucket': self.bucket.value,
            'start_time_ms': self.start_time_ms,
            'bucket_start_time': self.start_time_ms,  # for backwards compatibility TODO remove
            'reverted_ms': self.reverted_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BucketEntry':
        """Create from dict for deserialization. Retired bucket names map to their new bucket."""
        bucket = d.get('bucket', MinerBucket.UNKNOWN)
        return cls(
            bucket=MinerBucket(bucket) if isinstance(bucket, str) else bucket,
            start_time_ms=d.get('start_time_ms') or d.get('bucket_start_time', 0),
            reverted_ms=d.get('reverted_ms'),
        )
