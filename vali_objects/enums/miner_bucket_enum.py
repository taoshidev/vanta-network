from dataclasses import dataclass
from enum import Enum


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


@dataclass
class BucketEntry:
    bucket: MinerBucket
    start_time_ms: int
