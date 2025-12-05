from enum import Enum


class MinerBucket(Enum):
    MAINCOMP = "MAINCOMP"
    CHALLENGE = "CHALLENGE"
    PROBATION = "PROBATION"
    PLAGIARISM = "PLAGIARISM"
    UNKNOWN = "unknown"
