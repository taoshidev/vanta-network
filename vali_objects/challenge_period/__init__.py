# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc

"""Challenge period package - management of testing/production miner buckets.

Note: Imports are lazy to avoid circular import issues.
Use explicit imports from submodules:
    from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
    from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
    from vali_objects.challenge_period.challengeperiod_server import ChallengePeriodServer
"""

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == 'ChallengePeriodManager':
        from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
        return ChallengePeriodManager
    elif name == 'ChallengePeriodClient':
        from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
        return ChallengePeriodClient
    elif name == 'ChallengePeriodServer':
        from vali_objects.challenge_period.challengeperiod_server import ChallengePeriodServer
        return ChallengePeriodServer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['ChallengePeriodManager', 'ChallengePeriodClient', 'ChallengePeriodServer']
