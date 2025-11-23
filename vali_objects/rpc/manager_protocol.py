# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Shared RPC protocol definitions for manager-to-manager communication.

This module defines the message format for communication between:
- EliminationServer <-> ChallengePeriodManager
- Other future manager interactions

Key principle: NO imports of actual manager classes.
Only primitives, lists, dicts, and dataclasses defined here.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


# ==================== Elimination Manager Protocol ====================

@dataclass
class EliminationInfo:
    """Serializable elimination information"""
    hotkey: str
    reason: str
    dd: float
    elimination_initiated_time_ms: int
    price_info: Optional[Dict] = None
    return_info: Optional[Dict] = None


@dataclass
class GetEliminationsRequest:
    """Request all eliminations from EliminationServer"""
    pass


@dataclass
class GetEliminationsResponse:
    """Response with list of eliminations"""
    eliminations: List[Dict]  # List of elimination dicts


# ==================== ChallengePeriod Manager Protocol ====================

@dataclass
class HasEliminationReasonsRequest:
    """Check if CP manager has any elimination reasons"""
    pass


@dataclass
class HasEliminationReasonsResponse:
    """Response with boolean"""
    has_reasons: bool


@dataclass
class GetAllEliminationReasonsRequest:
    """Get all elimination reasons from CP manager"""
    pass


@dataclass
class GetAllEliminationReasonsResponse:
    """Response with elimination reasons dict"""
    reasons: Dict[str, Tuple[str, float]]  # {hotkey: (reason, drawdown)}


@dataclass
class PopEliminationReasonRequest:
    """Atomically get and remove an elimination reason"""
    hotkey: str


@dataclass
class PopEliminationReasonResponse:
    """Response with elimination reason or None"""
    reason: Optional[Tuple[str, float]]  # (reason, drawdown) or None
