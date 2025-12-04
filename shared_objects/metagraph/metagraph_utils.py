# developer: jbonilla
# Copyright © 2024 Taoshi Inc

"""
Shared utilities for metagraph analysis and anomaly detection.
"""

# Constants for anomaly detection
ANOMALY_DETECTION_MIN_LOST = 10  # Minimum number of lost hotkeys to trigger anomaly detection
ANOMALY_DETECTION_PERCENT_THRESHOLD = 25  # Percentage threshold for anomaly detection


def is_anomalous_hotkey_loss(lost_hotkeys: set, total_hotkeys_before: int) -> tuple[bool, float]:
    """
    Detect anomalous drops in hotkey counts to avoid false positives from network issues.

    This function identifies when too many hotkeys disappear at once, which likely indicates
    a network connectivity issue rather than legitimate de-registrations. Both the absolute
    count and percentage must exceed thresholds to trigger anomaly detection.

    Args:
        lost_hotkeys: Set of hotkeys that were lost in the metagraph update
        total_hotkeys_before: Total number of hotkeys before the change

    Returns:
        tuple[bool, float]: (is_anomalous, percent_lost)
            - is_anomalous: True if the change is anomalous (likely a network issue), False otherwise
            - percent_lost: Percentage of hotkeys lost (0-100)

    Examples:
        >>> # Normal case: 5 hotkeys lost out of 100 (5%)
        >>> is_anomalous_hotkey_loss({1, 2, 3, 4, 5}, 100)
        (False, 5.0)

        >>> # Anomalous case: 30 hotkeys lost out of 100 (30%)
        >>> is_anomalous_hotkey_loss(set(range(30)), 100)
        (True, 30.0)

        >>> # Edge case: 15 hotkeys lost out of 40 (37.5% - high percentage but meets both thresholds)
        >>> is_anomalous_hotkey_loss(set(range(15)), 40)
        (True, 37.5)

        >>> # Edge case: 11 hotkeys lost out of 100 (11% - above min count but below percent threshold)
        >>> is_anomalous_hotkey_loss(set(range(11)), 100)
        (False, 11.0)
    """
    # Handle edge cases
    if not lost_hotkeys or total_hotkeys_before == 0:
        return False, 0.0

    num_lost = len(lost_hotkeys)
    percent_lost = 100.0 * num_lost / total_hotkeys_before

    # Anomaly if we lost more than MIN_LOST hotkeys AND >= PERCENT_THRESHOLD of total
    # Both conditions must be true to avoid false positives
    is_anomalous = (
        num_lost > ANOMALY_DETECTION_MIN_LOST and
        percent_lost >= ANOMALY_DETECTION_PERCENT_THRESHOLD
    )

    return is_anomalous, percent_lost
