"""
Debt-Based Scoring

This module computes miner weights based on debt ledger information.
The algorithm pays miners based on their previous month's performance (PnL scaled by penalties),
proportionally distributing emissions to cover remaining debt by day 25 of the current month.

Key Concepts:
- "Needed payout" = What miners earned in previous month (PnL in USD * penalties)
- "Actual payout" = What they've been paid so far in current month (emissions in USD)
- "Remaining payout" = needed_payout_usd - actual_payout_usd (in USD)
- "Projected emissions" = Estimated total ALPHA available, converted to USD for comparison
- Weights = Proportional to remaining_payout_usd, with warning if insufficient emissions

Algorithm Flow:
1. Calculate needed_payout_usd from previous month's performance (only MAINCOMP/PROBATION checkpoints)
2. Calculate actual_payout_usd from current month's emissions (only MAINCOMP/PROBATION checkpoints)
3. Calculate remaining_payout_usd for each miner (in USD)
4. Query real-time TAO emission rate from subtensor
5. Convert to ALPHA, then convert ALPHA to USD using current conversion rates
6. Apply aggressive payout strategy (early month = 4-day horizon, late month = actual remaining)
7. Project total USD value available over aggressive timeline
8. Set weights proportional to remaining_payout_usd
9. Warn if sum(remaining_payouts_usd) > projected_usd_emissions
10. Enforce minimum weights based on challenge period status:
    - CHALLENGE/PLAGIARISM: 1x dust
    - PROBATION: 2x dust
    - MAINCOMP: 3x dust
    - UNKNOWN: 0x dust (no weight)
11. Normalize weights with burn address logic:
    - If sum < 1.0: assign (1.0 - sum) to burn address (uid 229 mainnet / uid 5 testnet)
    - If sum >= 1.0: normalize to 1.0, burn address gets 0

Aggressive Payout Strategy:
- Day 1-20: Target completion in 4 days (aggressive, creates urgency)
- Day 21-24: Target completion in actual remaining days (tapers off)
- Day 25: Final deadline
- This front-loads emissions early in the month while respecting the hard deadline

Important Notes:
- Debt-based scoring activates December 2025 (pays for November 2025 performance)
- Before December 2025, miners only receive minimum dust weights
- Excess weight (when sum < 1.0) goes to burn address (uid 229 mainnet, uid 220 testnet)
- Hard deadline: day 25 of each month
- Checkpoints are 12-hour intervals (2 per day)
- Uses real-time subtensor queries for emission rate estimation
"""

import bittensor as bt
from datetime import datetime, timezone
from typing import List, Tuple
from calendar import monthrange
import statistics

from time_util.time_util import TimeUtil
from vali_objects.utils.contract_server import ContractClient
from vali_objects.vali_dataclasses.debt_ledger import DebtLedger
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.scoring.scoring import Scoring
from collections import defaultdict


class DebtBasedScoring:
    """
    Debt-based scoring system that pays miners proportionally to their previous month's
    performance, targeting payout completion by day 25 of each month.

    Uses real-time subtensor queries to estimate emission rates and project available ALPHA.
    """

    # Activation: First payouts in December 2025 for November 2025 performance
    # (Previous month must be >= November 2025 to activate debt-based payouts)
    ACTIVATION_YEAR = 2025
    ACTIVATION_MONTH = 11

    # Target payout completion by day 25
    PAYOUT_TARGET_DAY = 25

    # Aggressive payout buffer: aim to complete this many days from now (minimum)
    # This makes early-month payouts more aggressive (day 1 targets 4-day completion)
    # while tapering to actual remaining days as we approach the deadline
    AGGRESSIVE_PAYOUT_BUFFER_DAYS = 4

    # Bittensor network parameters (approximate, for fallback)
    BLOCKS_PER_DAY_FALLBACK = 7200  # ~12 seconds per block
    RAO_PER_TOKEN = 1e9

    # Burn address UIDs (receives excess weight when sum < 1.0)
    BURN_UID_MAINNET = 229
    BURN_UID_TESTNET = 220

    @staticmethod
    def get_burn_uid(is_testnet: bool = False) -> int:
        """
        Get the correct burn UID based on network (testnet vs mainnet).

        Args:
            is_testnet: True for testnet (netuid 116), False for mainnet (netuid 8)

        Returns:
            229 for mainnet, 220 for testnet
        """
        return DebtBasedScoring.BURN_UID_TESTNET if is_testnet else DebtBasedScoring.BURN_UID_MAINNET

    @staticmethod
    def _safe_get_reserve_value(reserve_obj) -> float:
        """
        Safely extract reserve value from metagraph reserve object.

        Handles both manager.Value() objects (with .value attribute) and
        plain numeric values. Returns 0.0 if object is None or invalid.

        Args:
            reserve_obj: Reserve object from metagraph (tao_reserve_rao or alpha_reserve_rao)

        Returns:
            Reserve value as float, or 0.0 if invalid/missing
        """
        if reserve_obj is None:
            return 0.0

        # Try to access .value attribute (manager.Value() objects)
        if hasattr(reserve_obj, 'value'):
            try:
                return float(reserve_obj.value)
            except (TypeError, ValueError):
                return 0.0

        # Try to convert directly to float
        try:
            return float(reserve_obj)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def calculate_dynamic_dust(
        metagraph: 'bt.metagraph_handle',
        target_daily_usd: float = 0.01,
        verbose: bool = False
    ) -> float:
        """
        DEPRECATED: This function is no longer used. Dust is now a static value.

        Dust weight is set to ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT (static value).
        This function remains for reference but is not called in the scoring system.

        Historical Purpose:
        This function previously calculated dynamic dust weight that yielded target daily USD earnings.
        The calculation ensured that a miner receiving only dust weight would earn
        approximately target_daily_usd per day in ALPHA emissions, providing market-responsive
        minimum rewards that automatically adjusted as TAO/USD price, ALPHA/TAO conversion rate,
        and total subnet emission rate changed.

        Args:
            metagraph: Shared IPC metagraph with emission data and substrate reserves
            target_daily_usd: Target daily USD earnings for dust weight (default: $0.01)
            verbose: Enable detailed logging

        Returns:
            Static dust weight from ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
            (This function always returns the static fallback value)

        Note:
            This function always falls back to ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT.
            For the current static dust value, use ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT directly.
        """
        try:
            # Fallback detection: Check if metagraph has emission data
            emission = metagraph.get_emission()
            if emission is None:
                bt.logging.warning(
                    "Metagraph missing 'emission' attribute. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Step 1: Calculate total ALPHA emissions per day
            try:
                total_tao_per_tempo = sum(emission)  # TAO per tempo (360 blocks)
            except (TypeError, AttributeError) as e:
                bt.logging.warning(
                    f"Failed to sum metagraph.emission: {e}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Fallback detection: Check for zero/negative emissions
            if total_tao_per_tempo <= 0:
                bt.logging.warning(
                    f"Total TAO per tempo is non-positive: {total_tao_per_tempo}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            total_tao_per_block = total_tao_per_tempo / 360
            total_tao_per_day = total_tao_per_block * DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK

            if verbose:
                bt.logging.info(f"Total subnet emissions: {total_tao_per_day:.6f} TAO/day")

            # Step 2: Get conversion rates from metagraph with comprehensive fallback detection
            tao_reserve_obj = getattr(metagraph, 'tao_reserve_rao', None)
            alpha_reserve_obj = getattr(metagraph, 'alpha_reserve_rao', None)

            # Fallback detection: Check for missing reserve attributes
            if tao_reserve_obj is None or alpha_reserve_obj is None:
                bt.logging.warning(
                    f"Substrate reserve attributes not found in metagraph "
                    f"(tao_reserve_rao={tao_reserve_obj is not None}, "
                    f"alpha_reserve_rao={alpha_reserve_obj is not None}). "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Extract values using safe helper function
            tao_reserve_rao = DebtBasedScoring._safe_get_reserve_value(tao_reserve_obj)
            alpha_reserve_rao = DebtBasedScoring._safe_get_reserve_value(alpha_reserve_obj)

            # Fallback detection: Check for zero/negative reserves
            if tao_reserve_rao <= 0 or alpha_reserve_rao <= 0:
                bt.logging.warning(
                    f"Substrate reserve data not available or invalid for dynamic dust calculation "
                    f"(TAO={tao_reserve_rao} RAO, ALPHA={alpha_reserve_rao} RAO). "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Calculate ALPHA-to-TAO rate
            alpha_to_tao_rate = tao_reserve_rao / alpha_reserve_rao

            # Fallback detection: Sanity check on conversion rate
            if alpha_to_tao_rate <= 0 or alpha_to_tao_rate > 1.0:
                bt.logging.warning(
                    f"ALPHA-to-TAO rate outside expected range (0, 1.0]: {alpha_to_tao_rate}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Convert TAO/day to ALPHA/day
            total_alpha_per_day = total_tao_per_day / alpha_to_tao_rate

            if verbose:
                bt.logging.info(
                    f"Total subnet emissions: {total_alpha_per_day:.2f} ALPHA/day "
                    f"(conversion rate: {alpha_to_tao_rate:.6f} TAO/ALPHA)"
                )

            # Step 3: Get TAO/USD price with fallback detection
            tao_to_usd_rate_raw = getattr(metagraph, 'tao_to_usd_rate', None)

            # Fallback detection: Check for missing TAO/USD price
            if tao_to_usd_rate_raw is None:
                bt.logging.warning(
                    "TAO/USD price not available in metagraph for dynamic dust calculation. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Fallback detection: Validate TAO/USD price type and value
            try:
                tao_to_usd_rate = float(tao_to_usd_rate_raw)
            except (TypeError, ValueError) as e:
                bt.logging.warning(
                    f"TAO/USD price has invalid type: {type(tao_to_usd_rate_raw).__name__}, error: {e}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            if tao_to_usd_rate <= 0:
                bt.logging.warning(
                    f"TAO/USD price is non-positive: {tao_to_usd_rate}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Fallback detection: Sanity check on TAO price (should be between $1 and $10,000)
            if tao_to_usd_rate < 1.0 or tao_to_usd_rate > 10000.0:
                bt.logging.warning(
                    f"TAO/USD price outside reasonable range [$1, $10,000]: ${tao_to_usd_rate}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Step 4: Calculate ALPHA equivalent of target USD amount
            target_in_tao = target_daily_usd / tao_to_usd_rate
            target_in_alpha = target_in_tao / alpha_to_tao_rate

            if verbose:
                bt.logging.info(
                    f"${target_daily_usd:.2f} USD = {target_in_tao:.6f} TAO = "
                    f"{target_in_alpha:.6f} ALPHA"
                )

            # Step 5: Calculate dust weight as proportion of daily emissions
            # Fallback detection: Check for zero/negative total emissions
            if total_alpha_per_day <= 0:
                bt.logging.warning(
                    f"Total ALPHA per day is non-positive: {total_alpha_per_day}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            dust_weight = target_in_alpha / total_alpha_per_day

            if verbose:
                bt.logging.info(
                    f"Dynamic dust weight: {dust_weight:.8f} "
                    f"(yields ${target_daily_usd:.2f}/day at current emission rates)"
                )

            # Fallback detection: Sanity check on dust weight range
            # Should be small but not zero (typical range: 1e-8 to 1e-3)
            if dust_weight <= 0:
                bt.logging.warning(
                    f"Dynamic dust weight is non-positive: {dust_weight}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            if dust_weight > 0.001:
                bt.logging.warning(
                    f"Dynamic dust weight ({dust_weight:.8f}) exceeds reasonable maximum (0.001). "
                    f"This suggests anomalous market conditions. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Success! Return dynamic dust weight
            return dust_weight

        except Exception as e:
            # Fallback detection: Catch-all for any unexpected errors
            bt.logging.error(
                f"Unexpected error calculating dynamic dust: {e}. "
                f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}",
                exc_info=True
            )
            return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

    @staticmethod
    def log_projections(metagraph, days_until_target, verbose, total_remaining_payout_usd):
        """
        Log emission projections and compare to remaining payout needs.

        Args:
            metagraph: Bittensor metagraph with emission data
            days_until_target: Number of days until payout deadline
            verbose: Enable detailed logging
            total_remaining_payout_usd: Total remaining payout needed (must be > 0)
        """
        # Validate input to prevent division by zero
        if total_remaining_payout_usd <= 0:
            bt.logging.warning(
                f"total_remaining_payout_usd must be > 0, got {total_remaining_payout_usd}. "
                "Skipping projection log."
            )
            return

        # Query current emission rate and project availability
        # Get projected ALPHA emissions
        projected_alpha_available = DebtBasedScoring._estimate_alpha_emissions_until_target(
            metagraph=metagraph,
            days_until_target=days_until_target,
            verbose=verbose
        )

        # Convert projected ALPHA to USD for comparison
        projected_usd_available = DebtBasedScoring._convert_alpha_to_usd(
            alpha_amount=projected_alpha_available,
            metagraph=metagraph,
            verbose=verbose
        )

        if verbose:
            bt.logging.info(
                f"Projected emissions: {projected_alpha_available:.2f} ALPHA "
                f"≈ ${projected_usd_available:.2f} USD"
            )

        # Check if projected emissions (in USD) are sufficient
        if projected_usd_available < total_remaining_payout_usd:
            shortage_pct = ((total_remaining_payout_usd - projected_usd_available) / total_remaining_payout_usd) * 100
            bt.logging.warning(
                f"⚠️  INSUFFICIENT EMISSIONS: Projected USD value in next {days_until_target} days "
                f"(${projected_usd_available:.2f}) is less than total remaining payout needed "
                f"(${total_remaining_payout_usd:.2f}). Shortage: {shortage_pct:.1f}%. "
                f"Using aggressive {days_until_target}-day payout strategy (target day {DebtBasedScoring.PAYOUT_TARGET_DAY}). "
                f"Miners will receive proportional payouts."
            )
        else:
            surplus_pct = ((projected_usd_available - total_remaining_payout_usd) / total_remaining_payout_usd) * 100
            bt.logging.info(
                f"✓ Projected USD value in next {days_until_target} days (${projected_usd_available:.2f}) exceeds "
                f"total remaining payout needed (${total_remaining_payout_usd:.2f}). "
                f"Surplus: {surplus_pct:.1f}%. "
                f"Using aggressive {days_until_target}-day payout strategy (actual deadline: day {DebtBasedScoring.PAYOUT_TARGET_DAY})."
            )

    @staticmethod
    def compute_results(
        ledger_dict: dict[str, DebtLedger],
        metagraph: 'MetagraphClient',
        challengeperiod_client: 'ChallengePeriodClient',
        contract_client: 'ContractClient',
        current_time_ms: int = None,
        verbose: bool = False,
        is_testnet: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Compute miner weights based on debt ledger information with real-time emission projections.

        The algorithm works as follows:
        1. Check if we're in activation period (>= December 2025)
        2. For each miner, calculate their "needed payout" from previous month's performance
        3. Calculate "actual payout" given so far in current month
        4. Calculate "remaining payout" to be distributed
        5. Query real-time TAO emission rate from metagraph
        6. Convert to ALPHA using reserve data from shared metagraph (TAO/ALPHA ratio)
        7. Project total ALPHA available from now until day 25
        8. Set weights proportional to remaining_payout
        9. Warn if sum(remaining_payouts) > projected_emissions
        10. Enforce minimum weights with static dust (performance-scaled by 30-day PnL within buckets)
        11. Normalize weights with burn address logic (sum < 1.0 → burn gets excess)

        Args:
            ledger_dict: Dict of {hotkey: DebtLedger} containing debt ledger data
            metagraph: Shared IPC metagraph with emission data and substrate reserves
            challengeperiod_client: Client for querying current challenge period status (required)
            contract_client: Client for querying miner collateral balances (required)
            current_time_ms: Current timestamp in milliseconds (defaults to now)
            verbose: Enable detailed logging
            is_testnet: True for testnet (netuid 116), False for mainnet (netuid 8)

        Returns:
            List of (hotkey, weight) tuples sorted by weight (descending)
            Includes burn address (uid 229 mainnet / uid 220 testnet) if sum of weights < 1.0

        Note:
            Dust is a static value from ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT.
            Performance-based scaling is always enabled. Miners receive dust weights scaled by
            30-day penalty-adjusted PnL within their bucket:
            floor = bucket dust multiplier × static dust, ceiling = floor + static dust
        """
        if current_time_ms is None:
            current_time_ms = TimeUtil.now_in_millis()

        # Handle edge cases
        if not ledger_dict:
            bt.logging.info("No debt ledgers provided, setting burn address weight to 1.0")
            burn_hotkey = DebtBasedScoring._get_burn_address_hotkey(metagraph, is_testnet)
            return [(burn_hotkey, 1.0)]

        # Step 1: Get current month and year
        current_dt = TimeUtil.millis_to_datetime(current_time_ms)
        current_year = current_dt.year
        current_month = current_dt.month

        if verbose:
            bt.logging.info(
                f"Computing debt-based weights for {current_dt.strftime('%B %Y')} "
                f"({len(ledger_dict)} miners)"
            )

        # Step 2: Check if previous month is before November 2025
        # Calculate previous month
        if current_month == 1:
            prev_month = 12
            prev_year = current_year - 1
        else:
            prev_month = current_month - 1
            prev_year = current_year

        if verbose:
            bt.logging.info(f"Previous month: {prev_year}-{prev_month:02d}")

        # Check activation date: prev_month must be >= November 2025 for debt-based payouts
        # This means first debt-based payouts occur in December 2025 (for Nov 2025 performance)
        if (prev_year < DebtBasedScoring.ACTIVATION_YEAR or
            (prev_year == DebtBasedScoring.ACTIVATION_YEAR and
             prev_month < DebtBasedScoring.ACTIVATION_MONTH)):
            bt.logging.info(
                f"Previous month ({prev_year}-{prev_month:02d}) is before activation "
                f"({DebtBasedScoring.ACTIVATION_YEAR}-{DebtBasedScoring.ACTIVATION_MONTH:02d}). "
                f"Applying only minimum dust weights, excess goes to burn address."
            )
            # Before activation: apply minimum dust weights only, burn the rest
            return DebtBasedScoring._apply_pre_activation_weights(
                ledger_dict=ledger_dict,
                metagraph=metagraph,
                challengeperiod_client=challengeperiod_client,
                contract_client=contract_client,
                current_time_ms=current_time_ms,
                is_testnet=is_testnet,
                verbose=verbose
            )

        # Step 3: Calculate month boundaries
        # Previous month: full month
        prev_month_start_dt = datetime(prev_year, prev_month, 1, 0, 0, 0, tzinfo=timezone.utc)
        prev_month_days = monthrange(prev_year, prev_month)[1]  # Number of days in previous month
        prev_month_end_dt = datetime(prev_year, prev_month, prev_month_days, 23, 59, 59, 999999, tzinfo=timezone.utc)

        prev_month_start_ms = int(prev_month_start_dt.timestamp() * 1000)
        prev_month_end_ms = int(prev_month_end_dt.timestamp() * 1000)

        # Current month: from start of month to now
        current_month_start_dt = datetime(current_year, current_month, 1, 0, 0, 0, tzinfo=timezone.utc)
        current_month_start_ms = int(current_month_start_dt.timestamp() * 1000)

        if verbose:
            bt.logging.info(
                f"Previous month window: {prev_month_start_dt.strftime('%Y-%m-%d')} to "
                f"{prev_month_end_dt.strftime('%Y-%m-%d')}"
            )
            bt.logging.info(
                f"Current month elapsed: {current_month_start_dt.strftime('%Y-%m-%d')} to "
                f"{current_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # Calculate days until target payout day (day 25)
        current_day = current_dt.day

        if current_day > DebtBasedScoring.PAYOUT_TARGET_DAY:
            # Past target day, treat as 0 days remaining (will warn about insufficient time)
            actual_days_until_target = 0
        else:
            actual_days_until_target = DebtBasedScoring.PAYOUT_TARGET_DAY - current_day + 1  # +1 to include today

        # Apply aggressive payout strategy:
        # Early in month: Use shorter time horizon (e.g., 4 days) to be more aggressive
        # Late in month: Use actual remaining days as we approach deadline
        # This creates urgency early while respecting the hard deadline
        days_until_target = min(actual_days_until_target, DebtBasedScoring.AGGRESSIVE_PAYOUT_BUFFER_DAYS)

        # Ensure at least 1 day if we haven't reached deadline yet
        if actual_days_until_target > 0 and days_until_target == 0:
            days_until_target = 1

        if verbose:
            bt.logging.info(
                f"Current day: {current_day}, "
                f"target day: {DebtBasedScoring.PAYOUT_TARGET_DAY}, "
                f"actual days until target: {actual_days_until_target}, "
                f"aggressive days until target: {days_until_target}"
            )

        # Step 4-6: Process each miner to calculate remaining payouts (in USD)
        miner_remaining_payouts_usd = {}
        miner_actual_payouts_usd = {}  # Track what's been paid so far this month
        miner_penalty_loss_usd = {}  # Track how much was lost to penalties

        # Pre-fetch source ledgers for diagnostics (create clients once, fetch all ledgers once)
        all_emissions_ledgers = {}
        all_penalty_ledgers = {}
        all_perf_ledgers = {}
        if verbose:
            try:
                from vali_objects.vali_dataclasses.debt_ledger_server import DebtLedgerClient
                from vali_objects.vali_dataclasses.perf_ledger_server import PerfLedgerClient

                # Create clients once
                debt_ledger_client = DebtLedgerClient(connection_mode=RPCConnectionMode.RPC, connect_immediately=True)
                perf_ledger_client = PerfLedgerClient(connection_mode=RPCConnectionMode.RPC, connect_immediately=True)

                # Fetch ALL ledgers at once
                all_emissions_ledgers = debt_ledger_client.get_all_emissions_ledgers() or {}
                all_penalty_ledgers = debt_ledger_client.get_all_penalty_ledgers() or {}
                # Get all perf ledgers (portfolio_only=False returns per-trade-pair bundles)
                all_perf_ledgers = perf_ledger_client.get_perf_ledgers(portfolio_only=False, from_disk=False) or {}

                bt.logging.debug(
                    f"Pre-fetched source ledgers for diagnostics: "
                    f"{len(all_emissions_ledgers)} emissions, "
                    f"{len(all_penalty_ledgers)} penalty, "
                    f"{len(all_perf_ledgers)} perf"
                )

                # Debug: Sample one perf ledger to see its structure
                if all_perf_ledgers:
                    sample_hotkey = next(iter(all_perf_ledgers.keys()))
                    sample_perf_dict = all_perf_ledgers[sample_hotkey]
                    from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
                    bt.logging.debug(
                        f"Sample perf ledger for {sample_hotkey[:16]}...{sample_hotkey[-8:]}: "
                        f"type={type(sample_perf_dict).__name__}, "
                        f"keys={list(sample_perf_dict.keys()) if isinstance(sample_perf_dict, dict) else 'N/A'}, "
                        f"has_portfolio={TP_ID_PORTFOLIO in sample_perf_dict if isinstance(sample_perf_dict, dict) else False}"
                    )
            except Exception as e:
                bt.logging.debug(f"Could not pre-fetch source ledgers for diagnostics: {e}")

        for hotkey, debt_ledger in ledger_dict.items():
            if not debt_ledger.checkpoints:
                if verbose:
                    bt.logging.debug(f"Skipping {hotkey}: no checkpoints")
                miner_remaining_payouts_usd[hotkey] = 0.0
                miner_actual_payouts_usd[hotkey] = 0.0
                continue

            # Extract ALL checkpoints for previous month (for diagnostic purposes)
            all_prev_month_checkpoints = [
                cp for cp in debt_ledger.checkpoints
                if prev_month_start_ms <= cp.timestamp_ms <= prev_month_end_ms
            ]

            # Diagnostic: Log checkpoint coverage for miners with data
            if verbose and all_prev_month_checkpoints:
                # Show total checkpoints and date range
                total_cps = len(debt_ledger.checkpoints)
                if total_cps > 0:
                    first_cp_date = TimeUtil.millis_to_formatted_date_str(debt_ledger.checkpoints[0].timestamp_ms)
                    last_cp_date = TimeUtil.millis_to_formatted_date_str(debt_ledger.checkpoints[-1].timestamp_ms)

                    # Show November checkpoint date range
                    first_nov_cp_date = TimeUtil.millis_to_formatted_date_str(all_prev_month_checkpoints[0].timestamp_ms)
                    last_nov_cp_date = TimeUtil.millis_to_formatted_date_str(all_prev_month_checkpoints[-1].timestamp_ms)

                    # Calculate days covered in November
                    nov_start_day = TimeUtil.millis_to_datetime(all_prev_month_checkpoints[0].timestamp_ms).day
                    nov_end_day = TimeUtil.millis_to_datetime(all_prev_month_checkpoints[-1].timestamp_ms).day
                    nov_days_covered = nov_end_day - nov_start_day + 1
                    expected_cps = nov_days_covered * 2  # 2 checkpoints per day

                    bt.logging.debug(
                        f"{hotkey[:16]}...{hotkey[-8:]}: Total debt checkpoints: {total_cps} "
                        f"(range: {first_cp_date} to {last_cp_date}). "
                        f"Nov checkpoints: {len(all_prev_month_checkpoints)}/60 possible, "
                        f"Nov date range: {first_nov_cp_date} to {last_nov_cp_date} "
                        f"({nov_days_covered} days, expected ~{expected_cps} checkpoints)"
                    )

            # Extract checkpoints for previous month
            # Only include checkpoints where status is MAINCOMP or PROBATION (earning periods)
            prev_month_checkpoints = [
                cp for cp in all_prev_month_checkpoints
                if cp.challenge_period_status in (MinerBucket.MAINCOMP.value, MinerBucket.PROBATION.value)
            ]

            # Extract checkpoints for current month (up to now)
            # Only include checkpoints where status is MAINCOMP or PROBATION (earning periods)
            current_month_checkpoints = [
                cp for cp in debt_ledger.checkpoints
                if current_month_start_ms <= cp.timestamp_ms <= current_time_ms
                and cp.challenge_period_status in (MinerBucket.MAINCOMP.value, MinerBucket.PROBATION.value)
            ]

            # Diagnostic logging: Show sample checkpoint data
            if verbose and prev_month_checkpoints:
                # Sample first 3 checkpoints from previous month
                sample_cps = prev_month_checkpoints[:3]
                sample_info = []
                for cp in sample_cps:
                    cp_date = TimeUtil.millis_to_formatted_date_str(cp.timestamp_ms)
                    sample_info.append(
                        f"{cp_date}: realized_pnl=${cp.realized_pnl:.2f}, "
                        f"unrealized_pnl=${cp.unrealized_pnl:.2f}, penalty={cp.total_penalty:.4f}"
                    )

                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"{len(prev_month_checkpoints)} prev month checkpoints, "
                    f"{len(current_month_checkpoints)} current month checkpoints. "
                    f"Sample Nov checkpoints: {'; '.join(sample_info)}"
                )
            elif verbose:
                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"{len(prev_month_checkpoints)} prev month checkpoints, "
                    f"{len(current_month_checkpoints)} current month checkpoints"
                )

            # Step 4: Calculate needed payout from previous month (in USD)
            # "needed payout" = sum of (realized_pnl * total_penalty) across all prev month checkpoints
            #                   and (unrealized_pnl * total_penalty) of the last checkpoint
            # NOTE: realized_pnl and unrealized_pnl are in USD, per-checkpoint values (NOT cumulative)
            needed_payout_usd = 0.0
            penalty_loss_usd = 0.0
            if prev_month_checkpoints:
                # Sum penalty-adjusted PnL across all checkpoints in the month
                # Each checkpoint has its own PnL (for that 12-hour period) and its own penalty
                last_checkpoint = prev_month_checkpoints[-1]
                realized_component = sum(cp.realized_pnl * cp.total_penalty for cp in prev_month_checkpoints)
                unrealized_component = min(0.0, last_checkpoint.unrealized_pnl) * last_checkpoint.total_penalty
                needed_payout_usd = realized_component + unrealized_component

                # Calculate penalty loss: what would have been earned WITHOUT penalties
                payout_without_penalties = sum(cp.realized_pnl for cp in prev_month_checkpoints)
                payout_without_penalties += min(0.0, last_checkpoint.unrealized_pnl)
                penalty_loss_usd = payout_without_penalties - needed_payout_usd

                # Diagnostic: Show comprehensive breakdown for all miners with earning checkpoints
                if verbose:
                    total_realized = sum(cp.realized_pnl for cp in prev_month_checkpoints)
                    avg_penalty = sum(cp.total_penalty for cp in prev_month_checkpoints) / len(prev_month_checkpoints)
                    last_cp_date = TimeUtil.millis_to_formatted_date_str(last_checkpoint.timestamp_ms)

                    # Analyze all November checkpoints to understand the gap
                    status_breakdown = {}
                    for cp in all_prev_month_checkpoints:
                        status = cp.challenge_period_status
                        if status not in status_breakdown:
                            status_breakdown[status] = []
                        status_breakdown[status].append(cp)

                    # Find the absolute last checkpoint in November (any status)
                    absolute_last_cp = all_prev_month_checkpoints[-1] if all_prev_month_checkpoints else None
                    absolute_last_cp_date = TimeUtil.millis_to_formatted_date_str(absolute_last_cp.timestamp_ms) if absolute_last_cp else "N/A"

                    # Build status breakdown string
                    status_info = ", ".join([f"{status}={len(cps)}" for status, cps in sorted(status_breakdown.items())])

                    # NEW: Analyze penalty breakdown to understand why avg_penalty is 0
                    # Sample 3 earning checkpoints (first, middle, last) to show penalty components
                    sample_indices = []
                    if len(prev_month_checkpoints) >= 3:
                        sample_indices = [0, len(prev_month_checkpoints) // 2, -1]
                    elif len(prev_month_checkpoints) > 0:
                        sample_indices = list(range(len(prev_month_checkpoints)))

                    penalty_samples = []
                    for idx in sample_indices:
                        cp = prev_month_checkpoints[idx]
                        cp_date = TimeUtil.millis_to_formatted_date_str(cp.timestamp_ms)
                        penalty_samples.append(
                            f"{cp_date}: total_penalty={cp.total_penalty:.4f} "
                            f"(dd={cp.drawdown_penalty:.4f}, rp={cp.risk_profile_penalty:.4f}, "
                            f"mc={cp.min_collateral_penalty:.4f}, rap={cp.risk_adjusted_performance_penalty:.4f}), "
                            f"pnl={cp.realized_pnl:.2f}, emissions=${cp.chunk_emissions_usd:.2f}"
                        )

                    # Count how many checkpoints have zero penalties for each component
                    zero_penalties = {
                        'total': sum(1 for cp in prev_month_checkpoints if cp.total_penalty == 0.0),
                        'drawdown': sum(1 for cp in prev_month_checkpoints if cp.drawdown_penalty == 0.0),
                        'risk_profile': sum(1 for cp in prev_month_checkpoints if cp.risk_profile_penalty == 0.0),
                        'min_collateral': sum(1 for cp in prev_month_checkpoints if cp.min_collateral_penalty == 0.0),
                        'risk_adjusted_perf': sum(1 for cp in prev_month_checkpoints if cp.risk_adjusted_performance_penalty == 0.0),
                    }

                    # NEW: Access source ledgers to show Nov checkpoint counts from each ledger type
                    # Use pre-fetched ledgers to avoid creating new clients for each miner
                    source_ledger_info = ""
                    try:
                        # Get source ledgers for this hotkey from pre-fetched dicts
                        emissions_ledger = all_emissions_ledgers.get(hotkey)
                        penalty_ledger = all_penalty_ledgers.get(hotkey)
                        perf_ledgers_dict = all_perf_ledgers.get(hotkey)

                        # Count Nov checkpoints in each source ledger
                        emissions_nov_count = 0
                        penalty_nov_count = 0
                        perf_nov_count = 0

                        if emissions_ledger and emissions_ledger.checkpoints:
                            emissions_nov_count = sum(1 for cp in emissions_ledger.checkpoints
                                                     if prev_month_start_ms <= cp.chunk_end_ms <= prev_month_end_ms)

                        if penalty_ledger and penalty_ledger.checkpoints:
                            penalty_nov_count = sum(1 for cp in penalty_ledger.checkpoints
                                                   if prev_month_start_ms <= cp.last_processed_ms <= prev_month_end_ms)

                        if perf_ledgers_dict:
                            from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
                            portfolio_ledger = perf_ledgers_dict.get(TP_ID_PORTFOLIO)
                            if portfolio_ledger and portfolio_ledger.cps:
                                perf_nov_count = sum(1 for cp in portfolio_ledger.cps
                                                    if prev_month_start_ms <= cp.last_update_ms <= prev_month_end_ms)
                                # Debug: Log perf ledger checkpoint info when count is 0
                                if perf_nov_count == 0 and len(portfolio_ledger.cps) > 0:
                                    first_cp_ts = portfolio_ledger.cps[0].last_update_ms
                                    last_cp_ts = portfolio_ledger.cps[-1].last_update_ms
                                    first_cp_date = TimeUtil.millis_to_formatted_date_str(first_cp_ts)
                                    last_cp_date = TimeUtil.millis_to_formatted_date_str(last_cp_ts)
                                    bt.logging.debug(
                                        f"{hotkey[:16]}...{hotkey[-8:]}: perf_nov_count=0 but has {len(portfolio_ledger.cps)} total cps. "
                                        f"Perf checkpoint range: {first_cp_date} to {last_cp_date}. "
                                        f"Nov range: {TimeUtil.millis_to_formatted_date_str(prev_month_start_ms)} to "
                                        f"{TimeUtil.millis_to_formatted_date_str(prev_month_end_ms)}"
                                    )

                        source_ledger_info = f"Source ledger Nov checkpoints: debt={len(all_prev_month_checkpoints)}, emissions={emissions_nov_count}, penalty={penalty_nov_count}, perf={perf_nov_count}. "
                    except Exception as e:
                        bt.logging.debug(f"Could not access source ledgers for diagnostic: {e}")

                    # Use WARNING for zero payout (problematic), INFO for non-zero payout (normal)
                    log_level = bt.logging.warning if needed_payout_usd == 0.0 else bt.logging.info
                    payout_status = "ZERO needed_payout" if needed_payout_usd == 0.0 else f"needed_payout=${needed_payout_usd:.2f}"

                    log_level(
                        f"{hotkey[:16]}...{hotkey[-8:]}: {payout_status} despite {len(prev_month_checkpoints)} earning checkpoints! "
                        f"Total Nov checkpoints: {len(all_prev_month_checkpoints)} (by status: {status_info}). "
                        f"{source_ledger_info}"
                        f"Total realized_pnl=${total_realized:.2f}, avg_penalty={avg_penalty:.4f}, "
                        f"realized_component=${realized_component:.2f}, unrealized_component=${unrealized_component:.2f}. "
                        f"Last earning checkpoint ({last_cp_date}): realized_pnl=${last_checkpoint.realized_pnl:.2f}, "
                        f"unrealized_pnl=${last_checkpoint.unrealized_pnl:.2f}, penalty={last_checkpoint.total_penalty:.4f}, "
                        f"status={last_checkpoint.challenge_period_status}. "
                        f"Absolute last Nov checkpoint: {absolute_last_cp_date}, status={absolute_last_cp.challenge_period_status if absolute_last_cp else 'N/A'}. "
                        f"Zero penalty counts (of {len(prev_month_checkpoints)}): total={zero_penalties['total']}, "
                        f"drawdown={zero_penalties['drawdown']}, risk_profile={zero_penalties['risk_profile']}, "
                        f"min_collateral={zero_penalties['min_collateral']}, risk_adj_perf={zero_penalties['risk_adjusted_perf']}. "
                        f"Sample checkpoints: {'; '.join(penalty_samples)}"
                    )

            # Step 5: Calculate actual payout (in USD)
            # Special case for December 2025 (first activation month):
            #   Include both November + December payouts to avoid double-paying
            #   (miners may have received emissions in November using old scoring)
            # For all other months: Only include current month payouts
            if (current_year == DebtBasedScoring.ACTIVATION_YEAR and
                current_month == DebtBasedScoring.ACTIVATION_MONTH + 1):
                # December 2025: Count November + December payouts
                actual_payout_usd = sum(cp.chunk_emissions_usd for cp in prev_month_checkpoints)
                actual_payout_usd += sum(cp.chunk_emissions_usd for cp in current_month_checkpoints)
            else:
                # All other months: Only current month payouts
                actual_payout_usd = sum(cp.chunk_emissions_usd for cp in current_month_checkpoints)

            # Step 6: Calculate remaining payout (in USD)
            remaining_payout_usd = needed_payout_usd - actual_payout_usd

            # Clamp to zero if negative (over-paid or negative performance)
            if remaining_payout_usd < 0:
                remaining_payout_usd = 0.0

            miner_remaining_payouts_usd[hotkey] = remaining_payout_usd
            miner_actual_payouts_usd[hotkey] = actual_payout_usd
            miner_penalty_loss_usd[hotkey] = penalty_loss_usd

            # Debug: Log when needed_payout > 0 but remaining_payout = 0 (already fully paid)
            if verbose and needed_payout_usd > 0 and remaining_payout_usd == 0.0:
                # Calculate raw payouts without any filters for diagnostics
                raw_pnl_sum = sum(cp.realized_pnl for cp in all_prev_month_checkpoints)
                raw_pnl_with_penalty = sum(cp.realized_pnl * cp.total_penalty for cp in all_prev_month_checkpoints)
                raw_emissions_sum = sum(cp.chunk_emissions_usd for cp in all_prev_month_checkpoints)

                # Get unrealized PnL from last checkpoint (all statuses)
                last_cp_all = all_prev_month_checkpoints[-1] if all_prev_month_checkpoints else None
                unrealized_pnl_all = last_cp_all.unrealized_pnl if last_cp_all else 0.0
                unrealized_with_penalty_all = unrealized_pnl_all * last_cp_all.total_penalty if last_cp_all else 0.0

                # Split actual_payout into November vs December
                nov_emissions = sum(cp.chunk_emissions_usd for cp in prev_month_checkpoints)
                dec_emissions = sum(cp.chunk_emissions_usd for cp in current_month_checkpoints)

                # Break down by status
                status_breakdown_detailed = {}
                for cp in all_prev_month_checkpoints:
                    status = cp.challenge_period_status
                    if status not in status_breakdown_detailed:
                        status_breakdown_detailed[status] = {
                            'count': 0,
                            'pnl': 0.0,
                            'pnl_with_penalty': 0.0,
                            'emissions': 0.0
                        }
                    status_breakdown_detailed[status]['count'] += 1
                    status_breakdown_detailed[status]['pnl'] += cp.realized_pnl
                    status_breakdown_detailed[status]['pnl_with_penalty'] += cp.realized_pnl * cp.total_penalty
                    status_breakdown_detailed[status]['emissions'] += cp.chunk_emissions_usd

                # Find top 5 checkpoints by realized PnL
                sorted_checkpoints = sorted(all_prev_month_checkpoints, key=lambda cp: cp.realized_pnl, reverse=True)
                top_5_cps = sorted_checkpoints[:5]
                top_5_info = []
                for cp in top_5_cps:
                    cp_date = TimeUtil.millis_to_formatted_date_str(cp.timestamp_ms)
                    top_5_info.append(
                        f"{cp_date}({cp.challenge_period_status}): pnl=${cp.realized_pnl:.2f}, "
                        f"penalty={cp.total_penalty:.4f}, emissions=${cp.chunk_emissions_usd:.2f}"
                    )

                # Calculate what-if: if ALL checkpoints were MAINCOMP
                what_if_all_maincomp = sum(cp.realized_pnl * cp.total_penalty for cp in all_prev_month_checkpoints)
                what_if_all_maincomp += min(0.0, unrealized_pnl_all) * last_cp_all.total_penalty if last_cp_all else 0.0

                # Format status breakdown
                status_details = []
                for status, data in sorted(status_breakdown_detailed.items()):
                    status_details.append(
                        f"{status}(n={data['count']}, pnl=${data['pnl']:.2f}, "
                        f"pnl_w_penalty=${data['pnl_with_penalty']:.2f}, emissions=${data['emissions']:.2f})"
                    )

                bt.logging.info(
                    f"{hotkey[:16]}...{hotkey[-8:]}: ALREADY FULLY PAID - "
                    f"needed_payout=${needed_payout_usd:.2f}, actual_payout=${actual_payout_usd:.2f} "
                    f"(Nov=${nov_emissions:.2f} + Dec=${dec_emissions:.2f}), remaining=${remaining_payout_usd:.2f}"
                )
                bt.logging.info(
                    f"{hotkey[:16]}...{hotkey[-8:]}: RAW NOVEMBER TOTALS - "
                    f"all_checkpoints_pnl=${raw_pnl_sum:.2f} (no penalty, no filter), "
                    f"all_checkpoints_pnl_w_penalty=${raw_pnl_with_penalty:.2f}, "
                    f"all_checkpoints_emissions=${raw_emissions_sum:.2f}, "
                    f"unrealized_pnl=${unrealized_pnl_all:.2f} (w_penalty=${unrealized_with_penalty_all:.2f})"
                )
                bt.logging.info(
                    f"{hotkey[:16]}...{hotkey[-8:]}: STATUS BREAKDOWN - {'; '.join(status_details)}"
                )
                bt.logging.info(
                    f"{hotkey[:16]}...{hotkey[-8:]}: TOP 5 CHECKPOINTS BY PNL - {'; '.join(top_5_info)}"
                )
                bt.logging.info(
                    f"{hotkey[:16]}...{hotkey[-8:]}: WHAT-IF ANALYSIS - "
                    f"if_all_60_were_MAINCOMP=${what_if_all_maincomp:.2f}, "
                    f"actual_needed_payout=${needed_payout_usd:.2f}, "
                    f"gap=${what_if_all_maincomp - needed_payout_usd:.2f}"
                )

            if verbose:
                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"needed_payout_usd=${needed_payout_usd:.2f}, "
                    f"actual_payout_usd=${actual_payout_usd:.2f}, "
                    f"remaining_usd=${remaining_payout_usd:.2f}, "
                    f"penalty_loss_usd=${penalty_loss_usd:.2f}"
                )

        # Step 7-9: Query real-time emissions and project availability (in USD)
        total_remaining_payout_usd = sum(miner_remaining_payouts_usd.values())

        # Diagnostic summary: Show aggregate statistics
        if verbose:
            miners_with_nov_data = sum(1 for hotkey in ledger_dict.keys()
                                      if any(prev_month_start_ms <= cp.timestamp_ms <= prev_month_end_ms
                                            for cp in ledger_dict[hotkey].checkpoints))
            miners_with_zero_needed = sum(1 for needed in miner_remaining_payouts_usd.values() if needed == 0.0)
            miners_with_nonzero_needed = len(miner_remaining_payouts_usd) - miners_with_zero_needed

            bt.logging.info(
                f"November data summary: {miners_with_nov_data}/{len(ledger_dict)} miners have Nov checkpoints, "
                f"{miners_with_nonzero_needed} have non-zero needed_payout, "
                f"{miners_with_zero_needed} have zero needed_payout, "
                f"total_remaining=${total_remaining_payout_usd:.2f}"
            )

            # DISTRIBUTION ANALYSIS: Comprehensive analysis of payouts and realized/unrealized PnL
            bt.logging.info("=" * 80)
            bt.logging.info("DISTRIBUTION ANALYSIS: Payouts and PnL across all miners")
            bt.logging.info("=" * 80)

            # Collect all checkpoint data from miners with November data
            all_checkpoints = []
            miner_stats = []  # List of per-miner statistics

            from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO

            for hotkey in ledger_dict.keys():
                debt_ledger = ledger_dict[hotkey]

                # Get November checkpoints (ALL statuses for comprehensive analysis)
                nov_checkpoints = [
                    cp for cp in debt_ledger.checkpoints
                    if prev_month_start_ms <= cp.timestamp_ms <= prev_month_end_ms
                ]

                if not nov_checkpoints:
                    continue

                all_checkpoints.extend(nov_checkpoints)

                # Calculate per-miner statistics
                miner_realized_pnl = sum(cp.realized_pnl for cp in nov_checkpoints)
                miner_emissions = sum(cp.chunk_emissions_usd for cp in nov_checkpoints)
                miner_avg_penalty = statistics.mean(cp.total_penalty for cp in nov_checkpoints)

                # Get unrealized PnL from perf ledger (most recent)
                miner_unrealized_pnl = 0.0
                miner_perf_cp_count = 0
                try:
                    perf_ledgers_dict = all_perf_ledgers.get(hotkey)
                    if perf_ledgers_dict:
                        portfolio_ledger = perf_ledgers_dict.get(TP_ID_PORTFOLIO)
                        if portfolio_ledger and portfolio_ledger.cps:
                            # Count November perf checkpoints
                            nov_perf_cps = [
                                cp for cp in portfolio_ledger.cps
                                if prev_month_start_ms <= cp.last_update_ms <= prev_month_end_ms
                            ]
                            miner_perf_cp_count = len(nov_perf_cps)
                            # Get last November unrealized PnL
                            if nov_perf_cps:
                                miner_unrealized_pnl = nov_perf_cps[-1].unrealized_pnl
                except Exception:
                    pass

                miner_stats.append({
                    'hotkey': hotkey,
                    'checkpoint_count': len(nov_checkpoints),
                    'perf_checkpoint_count': miner_perf_cp_count,
                    'realized_pnl': miner_realized_pnl,
                    'unrealized_pnl': miner_unrealized_pnl,
                    'emissions': miner_emissions,
                    'avg_penalty': miner_avg_penalty,
                    'needed_payout': miner_remaining_payouts_usd.get(hotkey, 0.0) + miner_actual_payouts_usd.get(hotkey, 0.0),
                    'actual_payout': miner_actual_payouts_usd.get(hotkey, 0.0)
                })

            if not all_checkpoints:
                bt.logging.info("No November checkpoints found for distribution analysis")
            else:
                # 1. OVERALL STATISTICS ACROSS ALL MINERS
                bt.logging.info(f"\n1. OVERALL STATISTICS ({len(miner_stats)} miners, {len(all_checkpoints)} checkpoints)")
                bt.logging.info("-" * 80)

                # Per-miner aggregates
                realized_pnls = [m['realized_pnl'] for m in miner_stats]
                unrealized_pnls = [m['unrealized_pnl'] for m in miner_stats]
                emissions = [m['emissions'] for m in miner_stats]
                needed_payouts = [m['needed_payout'] for m in miner_stats]
                checkpoint_counts = [m['checkpoint_count'] for m in miner_stats]
                perf_checkpoint_counts = [m['perf_checkpoint_count'] for m in miner_stats]

                bt.logging.info(
                    f"Realized PnL per miner: min=${min(realized_pnls):.2f}, max=${max(realized_pnls):.2f}, "
                    f"mean=${statistics.mean(realized_pnls):.2f}, median=${statistics.median(realized_pnls):.2f}, "
                    f"stdev=${statistics.stdev(realized_pnls) if len(realized_pnls) > 1 else 0:.2f}"
                )
                bt.logging.info(
                    f"Unrealized PnL per miner: min=${min(unrealized_pnls):.2f}, max=${max(unrealized_pnls):.2f}, "
                    f"mean=${statistics.mean(unrealized_pnls):.2f}, median=${statistics.median(unrealized_pnls):.2f}, "
                    f"stdev=${statistics.stdev(unrealized_pnls) if len(unrealized_pnls) > 1 else 0:.2f}"
                )
                bt.logging.info(
                    f"Emissions per miner: min=${min(emissions):.2f}, max=${max(emissions):.2f}, "
                    f"mean=${statistics.mean(emissions):.2f}, median=${statistics.median(emissions):.2f}, "
                    f"stdev=${statistics.stdev(emissions) if len(emissions) > 1 else 0:.2f}"
                )
                bt.logging.info(
                    f"Needed payout per miner: min=${min(needed_payouts):.2f}, max=${max(needed_payouts):.2f}, "
                    f"mean=${statistics.mean(needed_payouts):.2f}, median=${statistics.median(needed_payouts):.2f}, "
                    f"stdev=${statistics.stdev(needed_payouts) if len(needed_payouts) > 1 else 0:.2f}"
                )
                bt.logging.info(
                    f"Debt checkpoints per miner: min={min(checkpoint_counts)}, max={max(checkpoint_counts)}, "
                    f"mean={statistics.mean(checkpoint_counts):.1f}, median={statistics.median(checkpoint_counts):.1f}"
                )
                bt.logging.info(
                    f"Perf checkpoints per miner: min={min(perf_checkpoint_counts)}, max={max(perf_checkpoint_counts)}, "
                    f"mean={statistics.mean(perf_checkpoint_counts):.1f}, median={statistics.median(perf_checkpoint_counts):.1f}"
                )

                # 2. PER-CHECKPOINT ANALYSIS
                bt.logging.info(f"\n2. PER-CHECKPOINT ANALYSIS ({len(all_checkpoints)} total checkpoints)")
                bt.logging.info("-" * 80)

                # Realized PnL distribution
                checkpoint_realized_pnls = [cp.realized_pnl for cp in all_checkpoints]
                zero_pnl_count = sum(1 for pnl in checkpoint_realized_pnls if pnl == 0.0)
                positive_pnl_count = sum(1 for pnl in checkpoint_realized_pnls if pnl > 0.0)
                negative_pnl_count = sum(1 for pnl in checkpoint_realized_pnls if pnl < 0.0)

                bt.logging.info(
                    f"Realized PnL per checkpoint: {zero_pnl_count} zero ({zero_pnl_count/len(all_checkpoints)*100:.1f}%), "
                    f"{positive_pnl_count} positive ({positive_pnl_count/len(all_checkpoints)*100:.1f}%), "
                    f"{negative_pnl_count} negative ({negative_pnl_count/len(all_checkpoints)*100:.1f}%)"
                )

                if positive_pnl_count > 0:
                    positive_pnls = [pnl for pnl in checkpoint_realized_pnls if pnl > 0.0]
                    bt.logging.info(
                        f"Positive PnL distribution: min=${min(positive_pnls):.2f}, max=${max(positive_pnls):.2f}, "
                        f"mean=${statistics.mean(positive_pnls):.2f}, median=${statistics.median(positive_pnls):.2f}"
                    )

                if negative_pnl_count > 0:
                    negative_pnls = [pnl for pnl in checkpoint_realized_pnls if pnl < 0.0]
                    bt.logging.info(
                        f"Negative PnL distribution: min=${min(negative_pnls):.2f}, max=${max(negative_pnls):.2f}, "
                        f"mean=${statistics.mean(negative_pnls):.2f}, median=${statistics.median(negative_pnls):.2f}"
                    )

                # Emissions distribution per checkpoint
                checkpoint_emissions = [cp.chunk_emissions_usd for cp in all_checkpoints]
                bt.logging.info(
                    f"Emissions per checkpoint: min=${min(checkpoint_emissions):.2f}, max=${max(checkpoint_emissions):.2f}, "
                    f"mean=${statistics.mean(checkpoint_emissions):.2f}, median=${statistics.median(checkpoint_emissions):.2f}"
                )

                # Penalty distribution per checkpoint
                checkpoint_penalties = [cp.total_penalty for cp in all_checkpoints]
                bt.logging.info(
                    f"Penalty per checkpoint: min={min(checkpoint_penalties):.4f}, max={max(checkpoint_penalties):.4f}, "
                    f"mean={statistics.mean(checkpoint_penalties):.4f}, median={statistics.median(checkpoint_penalties):.4f}"
                )

                # 3. STATUS-BASED DISTRIBUTION
                bt.logging.info("\n3. STATUS-BASED DISTRIBUTION")
                bt.logging.info("-" * 80)

                status_groups = defaultdict(list)
                for cp in all_checkpoints:
                    status_groups[cp.challenge_period_status].append(cp)

                for status, checkpoints in sorted(status_groups.items()):
                    realized_pnls_status = [cp.realized_pnl for cp in checkpoints]
                    emissions_status = [cp.chunk_emissions_usd for cp in checkpoints]
                    penalties_status = [cp.total_penalty for cp in checkpoints]

                    zero_count = sum(1 for pnl in realized_pnls_status if pnl == 0.0)
                    positive_count = sum(1 for pnl in realized_pnls_status if pnl > 0.0)
                    negative_count = sum(1 for pnl in realized_pnls_status if pnl < 0.0)

                    bt.logging.info(
                        f"Status {status}: {len(checkpoints)} checkpoints ({len(checkpoints)/len(all_checkpoints)*100:.1f}%), "
                        f"PnL: zero={zero_count}, pos={positive_count}, neg={negative_count}, "
                        f"total_pnl=${sum(realized_pnls_status):.2f}, "
                        f"mean_pnl=${statistics.mean(realized_pnls_status):.2f}, "
                        f"total_emissions=${sum(emissions_status):.2f}, "
                        f"mean_penalty={statistics.mean(penalties_status):.4f}"
                    )

                # 4. TEMPORAL DISTRIBUTION
                bt.logging.info("\n4. TEMPORAL DISTRIBUTION")
                bt.logging.info("-" * 80)

                # Group by date (YYYY-MM-DD)
                date_groups = defaultdict(list)
                for cp in all_checkpoints:
                    date_str = TimeUtil.millis_to_formatted_date_str(cp.timestamp_ms).split()[0]  # Get just the date part
                    date_groups[date_str].append(cp)

                # Show top 5 dates by total PnL
                date_totals = [(date, sum(cp.realized_pnl for cp in cps)) for date, cps in date_groups.items()]
                date_totals_sorted = sorted(date_totals, key=lambda x: abs(x[1]), reverse=True)[:5]

                bt.logging.info("Top 5 dates by absolute PnL:")
                for date, total_pnl in date_totals_sorted:
                    cp_count = len(date_groups[date])
                    avg_pnl = total_pnl / cp_count if cp_count > 0 else 0.0
                    bt.logging.info(f"  {date}: total_pnl=${total_pnl:.2f}, checkpoints={cp_count}, avg_pnl=${avg_pnl:.2f}")

                # Show temporal breakdown by week
                # Divide November into weeks
                week_groups = defaultdict(list)
                for cp in all_checkpoints:
                    cp_date = TimeUtil.millis_to_datetime(cp.timestamp_ms)
                    week_num = (cp_date.day - 1) // 7 + 1  # Week 1-5
                    week_groups[week_num].append(cp)

                bt.logging.info("\nPnL distribution by week of November:")
                for week in sorted(week_groups.keys()):
                    checkpoints = week_groups[week]
                    total_pnl = sum(cp.realized_pnl for cp in checkpoints)
                    avg_pnl = total_pnl / len(checkpoints) if checkpoints else 0.0
                    zero_count = sum(1 for cp in checkpoints if cp.realized_pnl == 0.0)
                    bt.logging.info(
                        f"  Week {week}: {len(checkpoints)} checkpoints, total_pnl=${total_pnl:.2f}, "
                        f"avg_pnl=${avg_pnl:.2f}, zero_pnl_count={zero_count} ({zero_count/len(checkpoints)*100:.1f}%)"
                    )

                # 5. PNL CONCENTRATION ANALYSIS
                bt.logging.info("\n5. PNL CONCENTRATION ANALYSIS")
                bt.logging.info("-" * 80)

                # Analyze how concentrated PnL is (do a few checkpoints account for most PnL?)
                # For each miner, find what % of their PnL came from their top checkpoint
                concentration_ratios = []
                for miner in miner_stats:
                    hotkey = miner['hotkey']
                    debt_ledger = ledger_dict[hotkey]
                    nov_checkpoints = [
                        cp for cp in debt_ledger.checkpoints
                        if prev_month_start_ms <= cp.timestamp_ms <= prev_month_end_ms
                    ]

                    if not nov_checkpoints or miner['realized_pnl'] == 0.0:
                        continue

                    # Find max absolute PnL checkpoint
                    max_pnl = max(abs(cp.realized_pnl) for cp in nov_checkpoints)
                    concentration = max_pnl / abs(miner['realized_pnl']) if miner['realized_pnl'] != 0 else 0.0
                    concentration_ratios.append(concentration)

                if concentration_ratios:
                    bt.logging.info(
                        f"PnL concentration (top checkpoint / total PnL per miner): "
                        f"mean={statistics.mean(concentration_ratios):.2%}, "
                        f"median={statistics.median(concentration_ratios):.2%}, "
                        f"max={max(concentration_ratios):.2%}"
                    )

                    # Count how many miners have >80% of PnL from single checkpoint
                    high_concentration = sum(1 for r in concentration_ratios if r > 0.8)
                    bt.logging.info(
                        f"Miners with >80% PnL from single checkpoint: {high_concentration}/{len(concentration_ratios)} "
                        f"({high_concentration/len(concentration_ratios)*100:.1f}%)"
                    )

                bt.logging.info("=" * 80)

        # Step 9a: Calculate projected emissions (needed for weight normalization)
        # Get projected ALPHA emissions
        projected_alpha_available = DebtBasedScoring._estimate_alpha_emissions_until_target(
            metagraph=metagraph,
            days_until_target=days_until_target,
            verbose=verbose
        )

        # Convert projected ALPHA to USD for comparison
        projected_usd_available = DebtBasedScoring._convert_alpha_to_usd(
            alpha_amount=projected_alpha_available,
            metagraph=metagraph,
            verbose=verbose
        )

        if total_remaining_payout_usd > 0 and days_until_target > 0:
            DebtBasedScoring.log_projections(metagraph, days_until_target, verbose, total_remaining_payout_usd)
        else:
            bt.logging.info(
                f"No remaining payouts needed {total_remaining_payout_usd} or no days until target "
                f"{days_until_target}, skipping projection log"
            )

        # Step 9b: Calculate daily target payouts using aggressive payout strategy
        # Instead of paying the entire remaining amount at once, spread it over days_until_target
        # This implements the aggressive payout strategy correctly
        miner_daily_target_payouts_usd = {}
        for hotkey, remaining_payout_usd in miner_remaining_payouts_usd.items():
            if days_until_target > 0:
                daily_target = remaining_payout_usd / days_until_target
            else:
                # Past deadline or exactly at deadline, pay everything today
                daily_target = remaining_payout_usd

            miner_daily_target_payouts_usd[hotkey] = daily_target

            if verbose:
                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"remaining_usd=${remaining_payout_usd:.2f}, "
                    f"daily_target_usd=${daily_target:.2f} "
                    f"(over {days_until_target} days)"
                )

        # Step 10: Enforce minimum weights based on challenge period status
        # All miners get minimum "dust" weights based on their current status
        # Dust is a static value from ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
        # Weights are performance-scaled by 30-day PnL within each bucket
        # NOTE: Weights are unitless proportions, normalized against projected daily emissions
        # Calculate projected daily emissions in USD
        if days_until_target > 0:
            projected_daily_usd = projected_usd_available / days_until_target
        else:
            # Past deadline, use full remaining emissions for today
            projected_daily_usd = projected_usd_available

        miner_weights_with_minimums = DebtBasedScoring._apply_minimum_weights(
            ledger_dict=ledger_dict,
            miner_remaining_payouts_usd=miner_daily_target_payouts_usd,
            challengeperiod_client=challengeperiod_client,
            contract_client=contract_client,
            metagraph=metagraph,
            current_time_ms=current_time_ms,
            projected_daily_emissions_usd=projected_daily_usd,
            verbose=verbose
        )

        # Step 11: Normalize weights with special burn address logic
        # If sum < 1.0: assign remaining weight to burn address (uid 229 / uid 5)
        # If sum >= 1.0: normalize to 1.0, burn address gets 0
        result = DebtBasedScoring._normalize_with_burn_address(
            weights=miner_weights_with_minimums,
            metagraph=metagraph,
            is_testnet=is_testnet,
            verbose=verbose
        )

        if verbose:
            bt.logging.info(f"Debt-based weights computed for {len(result)} miners")
            if result:
                # Filter for miners with non-zero needed_payout (dynamic count)
                miners_with_payout = [(hk, w) for hk, w in result if miner_remaining_payouts_usd.get(hk, 0.0) > 0]
                bt.logging.info(f"Miners with non-zero needed_payout ({len(miners_with_payout)}):")
                for hotkey, weight in miners_with_payout:
                    daily_target = miner_daily_target_payouts_usd.get(hotkey, 0.0)
                    monthly_target = miner_remaining_payouts_usd.get(hotkey, 0.0)
                    actual_paid = miner_actual_payouts_usd.get(hotkey, 0.0)
                    penalty_loss = miner_penalty_loss_usd.get(hotkey, 0.0)

                    # Get perf ledger checkpoint range
                    perf_range_str = "N/A"
                    try:
                        perf_ledgers_dict = all_perf_ledgers.get(hotkey)
                        if perf_ledgers_dict:
                            from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
                            portfolio_ledger = perf_ledgers_dict.get(TP_ID_PORTFOLIO)
                            if portfolio_ledger and portfolio_ledger.cps:
                                first_cp_date = TimeUtil.millis_to_formatted_date_str(portfolio_ledger.cps[0].last_update_ms)
                                last_cp_date = TimeUtil.millis_to_formatted_date_str(portfolio_ledger.cps[-1].last_update_ms)
                                perf_range_str = f"{first_cp_date} to {last_cp_date} ({len(portfolio_ledger.cps)} cps)"
                    except Exception:
                        pass

                    bt.logging.info(
                        f"  {hotkey[:16]}...{hotkey[-8:]}: weight={weight:.6f}, "
                        f"daily_target_usd=${daily_target:.2f}, monthly_target_usd=${monthly_target:.2f}, "
                        f"paid_this_month_usd=${actual_paid:.2f}, penalty_loss_usd=${penalty_loss:.2f}, "
                        f"perf_range=[{perf_range_str}]"
                    )

        return result

    @staticmethod
    def _estimate_alpha_emissions_until_target(
        metagraph: 'MetagraphClient',
        days_until_target: int,
        verbose: bool = False
    ) -> float:
        """
        Estimate total ALPHA emissions available from now until target day.

        Uses real-time metagraph data to get current TAO emission rate,
        then converts to ALPHA using reserve data from shared metagraph.

        Args:
            metagraph: Shared IPC metagraph with emission data and substrate reserves
            days_until_target: Number of days until target payout day
            verbose: Enable detailed logging

        Returns:
            Estimated total ALPHA emissions available (float)
        """
        try:
            # Get total TAO emission per block for the subnet (sum across all miners)
            # metagraph.emission is already in TAO (not RAO), but per tempo (360 blocks)
            # Need to convert: per-tempo → per-block (÷360)
            total_tao_per_tempo = sum(metagraph.get_emission())
            total_tao_per_block = total_tao_per_tempo / 360

            if verbose:
                bt.logging.info(f"Current subnet emission rate: {total_tao_per_block:.6f} TAO/block")

            # Estimate blocks until target day
            # Use approximate 12 seconds per block (7200 blocks/day)
            blocks_until_target = days_until_target * DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK

            # Calculate total TAO emissions until target
            total_tao_until_target = total_tao_per_block * blocks_until_target

            if verbose:
                bt.logging.info(
                    f"Estimated blocks until day {DebtBasedScoring.PAYOUT_TARGET_DAY}: {blocks_until_target}, "
                    f"total TAO: {total_tao_until_target:.2f}"
                )

            # Get substrate reserves from shared metagraph (refreshed by MetagraphUpdater)
            # Use safe helper to extract values from manager.Value() objects or plain numerics
            tao_reserve_obj = getattr(metagraph, 'tao_reserve_rao', None)
            alpha_reserve_obj = getattr(metagraph, 'alpha_reserve_rao', None)

            tao_reserve_rao = DebtBasedScoring._safe_get_reserve_value(tao_reserve_obj)
            alpha_reserve_rao = DebtBasedScoring._safe_get_reserve_value(alpha_reserve_obj)

            if tao_reserve_rao == 0 or alpha_reserve_rao == 0:
                bt.logging.warning(
                    "Substrate reserve data not available in shared metagraph "
                    f"(TAO={tao_reserve_rao} RAO, ALPHA={alpha_reserve_rao} RAO). "
                    "Cannot calculate ALPHA conversion rate."
                )
                return 0.0

            # Calculate ALPHA-to-TAO conversion rate from reserve data
            # alpha_to_tao_rate = tao_reserve / alpha_reserve (both in RAO, ratio is unitless)
            # (How much TAO per ALPHA)
            alpha_to_tao_rate = tao_reserve_rao / alpha_reserve_rao

            if verbose:
                bt.logging.info(
                    f"Substrate reserves: TAO={tao_reserve_rao / 1e9:.2f} TAO ({tao_reserve_rao:.0f} RAO), "
                    f"ALPHA={alpha_reserve_rao / 1e9:.2f} ALPHA ({alpha_reserve_rao:.0f} RAO), "
                    f"rate={alpha_to_tao_rate:.6f} TAO/ALPHA"
                )

            # Convert TAO to ALPHA
            # If ALPHA costs X TAO per ALPHA, then Y TAO buys Y/X ALPHA
            if alpha_to_tao_rate > 0:
                total_alpha_until_target = total_tao_until_target / alpha_to_tao_rate
            else:
                bt.logging.warning("ALPHA-to-TAO rate is zero, cannot convert")
                return 0.0

            if verbose:
                bt.logging.info(f"Projected ALPHA available until target: {total_alpha_until_target:.2f}")

            return total_alpha_until_target

        except Exception as e:
            bt.logging.error(f"Error estimating ALPHA emissions: {e}")
            raise

    @staticmethod
    def _convert_alpha_to_usd(
        alpha_amount: float,
        metagraph: 'bt.metagraph_handle',
        verbose: bool = False
    ) -> float:
        """
        Convert ALPHA amount to USD value using current market rates.

        Uses reserve data from shared metagraph to calculate conversion rate:
        ALPHA → TAO (via reserves) → USD (via TAO price oracle)

        Args:
            alpha_amount: Amount of ALPHA tokens to convert
            metagraph: Shared IPC metagraph with substrate reserves
            verbose: Enable detailed logging

        Returns:
            USD value of the ALPHA amount (float)
        """
        if alpha_amount == 0:
            return 0.0

        # Get substrate reserves from shared metagraph
        # Use safe helper to extract values from manager.Value() objects or plain numerics
        tao_reserve_obj = getattr(metagraph, 'tao_reserve_rao', None)
        alpha_reserve_obj = getattr(metagraph, 'alpha_reserve_rao', None)

        tao_reserve_rao = DebtBasedScoring._safe_get_reserve_value(tao_reserve_obj)
        alpha_reserve_rao = DebtBasedScoring._safe_get_reserve_value(alpha_reserve_obj)

        if tao_reserve_rao == 0 or alpha_reserve_rao == 0:
            bt.logging.warning(
                "Substrate reserve data not available for ALPHA→USD conversion. "
                f"(TAO={tao_reserve_rao} RAO, ALPHA={alpha_reserve_rao} RAO)"
            )
            return 0.0

        # Calculate ALPHA→TAO conversion rate
        # alpha_to_tao_rate = how much TAO per ALPHA
        alpha_to_tao_rate = tao_reserve_rao / alpha_reserve_rao

        # Convert ALPHA to TAO
        tao_amount = alpha_amount * alpha_to_tao_rate

        # Get TAO→USD price from metagraph
        # This is set by MetagraphUpdater via live_price_fetcher.get_close_at_date(TradePair.TAOUSD)
        tao_to_usd_rate_raw = getattr(metagraph, 'tao_to_usd_rate', None)

        # Validate that we have a valid TAO/USD rate
        if tao_to_usd_rate_raw is None:
            raise ValueError(
                "TAO/USD price not available in metagraph. "
                "MetagraphUpdater must set metagraph.tao_to_usd_rate via live_price_fetcher."
            )

        if not isinstance(tao_to_usd_rate_raw, (int, float)) or tao_to_usd_rate_raw <= 0:
            raise ValueError(
                f"Invalid TAO/USD price in metagraph: {tao_to_usd_rate_raw}. "
                f"Expected positive number, got {type(tao_to_usd_rate_raw).__name__}."
            )

        tao_to_usd_rate = float(tao_to_usd_rate_raw)

        # Convert TAO to USD
        usd_amount = tao_amount * tao_to_usd_rate

        if verbose:
            bt.logging.debug(
                f"ALPHA→USD conversion: {alpha_amount:.2f} ALPHA "
                f"→ {tao_amount:.6f} TAO "
                f"→ ${usd_amount:.2f} USD "
                f"(rates: {alpha_to_tao_rate:.6f} TAO/ALPHA, ${tao_to_usd_rate:.2f}/TAO)"
            )

        return usd_amount



    @staticmethod
    def _calculate_penalty_adjusted_pnl(
        ledger: DebtLedger,
        start_time_ms: int,
        end_time_ms: int,
        earning_statuses: set[str] = None
    ) -> float:
        """
        Calculate penalty-adjusted PnL for a time period (in USD).

        This is the SINGLE SOURCE OF TRUTH for PnL calculations,
        used by both main scoring and dynamic dust weight calculations.

        NOTE: realized_pnl and unrealized_pnl in checkpoints are in USD (performance value),
        so the return value is also in USD.

        Args:
            ledger: Miner's debt ledger
            start_time_ms: Period start (inclusive)
            end_time_ms: Period end (inclusive)
            earning_statuses: Set of statuses to include (default: MAINCOMP, PROBATION)

        Returns:
            Penalty-adjusted PnL for the period in USD (sum of realized_pnl * total_penalty
            across all checkpoints plus unrealized_pnl * total_penalty for the last checkpoint)
        """
        # Default to earning statuses
        if earning_statuses is None:
            earning_statuses = {
                MinerBucket.MAINCOMP.value,
                MinerBucket.PROBATION.value
            }

        if not ledger.checkpoints:
            return 0.0

        # Filter checkpoints within time range and matching statuses
        relevant_checkpoints = [
            cp for cp in ledger.checkpoints
            if start_time_ms <= cp.timestamp_ms <= end_time_ms
            and cp.challenge_period_status in earning_statuses
        ]

        if not relevant_checkpoints:
            return 0.0

        # Sum penalty-adjusted PnL across all checkpoints in the time range
        # NOTE: realized_pnl/unrealized_pnl are per-checkpoint values (NOT cumulative), so we must sum
        # Each checkpoint has its own PnL (for that 12-hour period) and its own penalty
        last_checkpoint = relevant_checkpoints[-1]
        penalty_adjusted_pnl = sum(cp.realized_pnl * cp.total_penalty for cp in relevant_checkpoints)
        penalty_adjusted_pnl += min(0.0, last_checkpoint.unrealized_pnl) * last_checkpoint.total_penalty

        return penalty_adjusted_pnl

    @staticmethod
    def _calculate_pnl_scores_for_bucket(
        miners: list[tuple[str, DebtLedger]],
        lookback_start_ms: int,
        current_time_ms: int
    ) -> dict[str, float]:
        """
        Calculate 30-day penalty-adjusted PnL scores for miners in a bucket.

        Args:
            miners: List of (hotkey, ledger) tuples
            lookback_start_ms: Start of 30-day lookback window
            current_time_ms: Current timestamp

        Returns:
            Dict mapping hotkey -> PnL score (floored at 0)
        """
        pnl_scores = {}
        all_statuses = {b.value for b in MinerBucket}

        for hotkey, ledger in miners:
            pnl = DebtBasedScoring._calculate_penalty_adjusted_pnl(
                ledger,
                start_time_ms=lookback_start_ms,
                end_time_ms=current_time_ms,
                earning_statuses=all_statuses  # Consider all recent performance
            )
            # Floor at 0 (negative PnL doesn't reduce dust below floor)
            pnl_scores[hotkey] = max(0.0, pnl)

        return pnl_scores

    @staticmethod
    def _calculate_collateral_priority_scores(
        pnl_scores: dict[str, float],
        collateral_balances: dict[str, float],
        min_collateral_threshold: float = None
    ) -> dict[str, tuple[int, float]]:
        """
        Calculate priority scores for CHALLENGE miners based on collateral + PnL.

        Priority tiers (lower number = higher priority for 0 weight/dereg):
        - Tier 0: Zero collateral (ALWAYS get 0 weight, no cap)
        - Tier 1: Below MIN_COLLATERAL_VALUE (prioritized for 0 weight)
        - Tier 2: Adequate collateral (use PnL as tiebreaker)

        Args:
            pnl_scores: Dict of hotkey -> PnL score (in USD)
            collateral_balances: Dict of hotkey -> collateral balance (in USD)
            min_collateral_threshold: Minimum required collateral (default: ValiConfig.MIN_COLLATERAL_VALUE)

        Returns:
            Dict mapping hotkey -> (priority_tier, pnl_score)
            Lower priority_tier = higher priority for elimination
        """
        if min_collateral_threshold is None:
            min_collateral_threshold = ValiConfig.MIN_COLLATERAL_VALUE

        priority_scores = {}

        for hotkey, pnl in pnl_scores.items():
            collateral = collateral_balances.get(hotkey, 0.0)

            if collateral == 0:
                # Tier 0: Zero collateral - ALWAYS eliminate
                priority_tier = 0
            elif collateral < min_collateral_threshold:
                # Tier 1: Below minimum - prioritize for elimination
                priority_tier = 1
            else:
                # Tier 2: Adequate collateral - use PnL ranking
                priority_tier = 2

            priority_scores[hotkey] = (priority_tier, pnl)

        return priority_scores

    @staticmethod
    def _calculate_challenge_zero_weight_miners(
        pnl_scores: dict[str, float],
        contract_client: 'ContractClient',
        percentile: float = 0.25,
        max_zero_weight_miners: int = 10
    ) -> set[str]:
        """
        Determine which CHALLENGE miners should get 0 weight (dereg candidates).

        Prioritization logic:
        1. ALL miners with 0 collateral get 0 weight (no cap)
        2. Miners below MIN_COLLATERAL fill remaining slots (up to max_zero_weight_miners)
        3. If slots remain, worst PnL performers with adequate collateral fill them

        Args:
            pnl_scores: Dict of hotkey -> PnL score
            contract_client: Contract client for collateral queries
            percentile: Target percentile for 0 weight (0.25 = 25%)
            max_zero_weight_miners: Maximum total miners to assign 0 weight

        Returns:
            Set of hotkeys that should receive 0 weight
        """
        if len(pnl_scores) <= 1:
            return set()

        # Get cached collateral balances (in USD) for all miners
        # Use cached data to avoid rate limiting on-chain queries
        collateral_balances = {}
        for hotkey in pnl_scores.keys():
            collateral_usd = contract_client.get_miner_account_size(hotkey, most_recent=True)
            # Handle None or negative values
            if collateral_usd is None or collateral_usd <= 0:
                collateral_usd = 0.0
            collateral_balances[hotkey] = collateral_usd

        # Calculate priority scores (tier, pnl) for each miner
        priority_scores = DebtBasedScoring._calculate_collateral_priority_scores(
            pnl_scores=pnl_scores,
            collateral_balances=collateral_balances
        )

        # Sort by priority: (tier ASC, pnl ASC)
        # Lower tier = higher priority for elimination
        # Within same tier, lower PnL = higher priority for elimination
        sorted_miners = sorted(
            priority_scores.items(),
            key=lambda x: (x[1][0], x[1][1])  # Sort by (tier, pnl)
        )

        zero_weight_miners = set()

        # Calculate target count: percentile of total miners, capped at max
        target_zero_weight_count = min(int(len(pnl_scores) * percentile), max_zero_weight_miners)

        # Phase 1: ALL zero-collateral miners get 0 weight (no cap)
        zero_collateral_miners = [hk for hk, (tier, _) in sorted_miners if tier == 0]
        zero_weight_miners.update(zero_collateral_miners)

        if zero_collateral_miners:
            bt.logging.warning(
                f"Found {len(zero_collateral_miners)} CHALLENGE miners with ZERO collateral. "
                f"All will receive 0 weight (priority dereg): {[hk[:16] for hk in zero_collateral_miners]}"
            )

        # Phase 2: Fill remaining slots (up to target_zero_weight_count total)
        remaining_slots = target_zero_weight_count - len(zero_weight_miners)

        if remaining_slots > 0:
            # Get miners not yet assigned 0 weight, sorted by priority
            remaining_miners = [hk for hk, _ in sorted_miners if hk not in zero_weight_miners]

            # Fill slots with next-worst miners (low collateral first, then bad PnL)
            additional_zero_weight = remaining_miners[:remaining_slots]
            zero_weight_miners.update(additional_zero_weight)

            if additional_zero_weight:
                low_collateral_count = sum(
                    1 for hk in additional_zero_weight
                    if priority_scores[hk][0] == 1  # Tier 1 = below MIN_COLLATERAL
                )
                bt.logging.info(
                    f"Assigned 0 weight to {len(additional_zero_weight)} additional CHALLENGE miners: "
                    f"{low_collateral_count} with low collateral, "
                    f"{len(additional_zero_weight) - low_collateral_count} with poor PnL"
                )

        return zero_weight_miners

    @staticmethod
    def _calculate_challenge_percentile_threshold(
        pnl_scores: dict[str, float],
        percentile: float = 0.25,
        max_zero_weight_miners: int = 10
    ) -> float:
        """
        DEPRECATED: Use _calculate_challenge_zero_weight_miners instead for collateral-aware selection.

        Calculate percentile threshold for CHALLENGE bucket miners.

        The threshold is calculated such that the bottom percentile get 0 weight,
        but capped at max_zero_weight_miners.

        Args:
            pnl_scores: Dict of hotkey -> PnL score
            percentile: Percentile to calculate (0.25 = 25th percentile)
            max_zero_weight_miners: Maximum number of miners to assign 0 weight (default: 10)

        Returns:
            PnL threshold value, or None if too few miners
        """
        if len(pnl_scores) <= 1:
            return None

        # Calculate how many miners should get 0 weight (bottom 25%, capped at 10)
        num_zero_weight = min(int(len(pnl_scores) * percentile), max_zero_weight_miners)

        if num_zero_weight == 0:
            return None

        # Sort PnL values and find the threshold
        sorted_pnls = sorted(pnl_scores.values())
        # Miners with PnL < threshold will get 0 weight
        return sorted_pnls[num_zero_weight]

    @staticmethod
    def _assign_weights_with_performance_scaling(
        pnl_scores: dict[str, float],
        bucket: int,
        floor: float,
        ceiling: float,
        zero_weight_miners: set[str] = None,
        verbose: bool = False
    ) -> dict[str, float]:
        """
        Assign weights to miners based on PnL scores with performance scaling.

        For CHALLENGE bucket, miners in zero_weight_miners set get 0 weight (collateral-aware).
        Others are scaled from floor to ceiling based on normalized PnL.

        Args:
            pnl_scores: Dict of hotkey -> PnL score
            bucket: Bucket type (MinerBucket enum value)
            floor: Minimum weight for this bucket
            ceiling: Maximum weight for this bucket
            zero_weight_miners: Set of miners that should receive 0 weight (collateral-aware)
            verbose: Enable detailed logging

        Returns:
            Dict mapping hotkey -> assigned weight
        """
        weights = {}
        max_pnl = max(pnl_scores.values()) if pnl_scores else 0.0

        if zero_weight_miners is None:
            zero_weight_miners = set()

        if max_pnl > 0:
            # Scale each miner's PnL to [0, 1] then map to [floor, ceiling]
            for hotkey, pnl in pnl_scores.items():
                # CHALLENGE bucket: miners in zero_weight_miners set get 0 weight
                if bucket == MinerBucket.CHALLENGE.value and hotkey in zero_weight_miners:
                    weights[hotkey] = 0.0
                    if verbose:
                        bt.logging.debug(
                            f"  {hotkey[:16]}...{hotkey[-8:]}: "
                            f"pnl_usd=${pnl:.2f} (collateral-aware 0 weight)"
                        )
                else:
                    normalized = pnl / max_pnl
                    # Scale to [floor, ceiling]
                    weights[hotkey] = floor + (normalized * (ceiling - floor))

                    if verbose:
                        bt.logging.debug(
                            f"  {hotkey[:16]}...{hotkey[-8:]}: "
                            f"pnl_usd=${pnl:.2f}, norm={normalized:.4f}, "
                            f"weight={weights[hotkey]:.8f}"
                        )
        else:
            # All miners have 0 PnL
            weights = DebtBasedScoring._handle_zero_pnl_weights(
                pnl_scores=pnl_scores,
                bucket=bucket,
                floor=floor,
                zero_weight_miners=zero_weight_miners,
                verbose=verbose
            )

        return weights

    @staticmethod
    def _handle_zero_pnl_weights(
        pnl_scores: dict[str, float],
        bucket: int,
        floor: float,
        zero_weight_miners: set[str] = None,
        verbose: bool = False
    ) -> dict[str, float]:
        """
        Handle weight assignment when all miners in a bucket have 0 PnL.

        For CHALLENGE bucket with multiple miners, uses zero_weight_miners set (collateral-aware)
        to determine who gets 0 weight. If zero_weight_miners is not provided, falls back to
        lexicographic order. Other buckets get floor weight.

        Args:
            pnl_scores: Dict of hotkey -> PnL score (all should be 0)
            bucket: Bucket type (MinerBucket enum value)
            floor: Minimum weight for this bucket
            zero_weight_miners: Set of miners that should receive 0 weight (collateral-aware)
            verbose: Enable detailed logging

        Returns:
            Dict mapping hotkey -> assigned weight
        """
        weights = {}

        if zero_weight_miners is None:
            zero_weight_miners = set()

        # CHALLENGE bucket: use zero_weight_miners set when all have 0 PnL
        # Only apply this penalty if there are multiple miners to compare
        if bucket == MinerBucket.CHALLENGE.value and len(pnl_scores) > 1:
            for hotkey in pnl_scores.keys():
                if hotkey in zero_weight_miners:
                    weights[hotkey] = 0.0
                else:
                    weights[hotkey] = floor

            if verbose:
                bt.logging.debug(
                    f"  All CHALLENGE miners have 0 PnL, assigning 0 weight to {len(zero_weight_miners)} "
                    f"miners (collateral-aware), floor weight to others"
                )
        else:
            # Other buckets or single CHALLENGE miner: all get floor weight
            for hotkey in pnl_scores.keys():
                weights[hotkey] = floor
            if verbose:
                bt.logging.debug(f"  All miners have 0 PnL, assigning floor weight")

        return weights

    @staticmethod
    def _calculate_dynamic_dust_weights(
        ledger_dict: dict[str, DebtLedger],
        challengeperiod_client: 'ChallengePeriodClient',
        contract_client: 'ContractClient',
        current_time_ms: int,
        base_dust: float,
        verbose: bool = False
    ) -> dict[str, float]:
        """
        Calculate performance-scaled dust weights for all miners.

        NOTE: Despite the function name, dust is a static value from ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT.
        This function scales the static dust based on 30-day performance within each bucket.

        Process:
        1. Group miners by bucket
        2. For each bucket, calculate 30-day penalty-adjusted PnL (in USD) for all miners
        3. Normalize PnL within bucket to [0, 1] range
        4. Scale to [floor, ceiling] where:
           - floor = bucket multiplier × static dust (e.g., 3× for MAINCOMP)
           - ceiling = floor + static dust amount

        This incentivizes recent performance while maintaining bucket hierarchy.

        NOTE: PnL values are in USD as calculated by _calculate_penalty_adjusted_pnl.

        Args:
            ledger_dict: All miner ledgers
            challengeperiod_client: Client for querying bucket status
            contract_client: Client for querying miner collateral balances (required)
            current_time_ms: Current timestamp
            base_dust: Static dust value from ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
            verbose: Enable detailed logging

        Returns:
            Dict mapping hotkey -> performance_scaled_dust_weight (unitless proportion)
        """
        # Original dust floor multipliers (respecting existing system)
        BUCKET_DUST_FLOORS = {
            MinerBucket.CHALLENGE.value: 1,      # 1x dust floor
            MinerBucket.PROBATION.value: 2,      # 2x dust floor
            MinerBucket.MAINCOMP.value: 3,       # 3x dust floor
            MinerBucket.UNKNOWN.value: 0,        # 0x dust (no weight for unknown status)
            MinerBucket.PLAGIARISM.value: 1,     # 1x dust floor
        }

        dynamic_weights = {}
        thirty_days_ms = 30 * 24 * 60 * 60 * 1000
        lookback_start = current_time_ms - thirty_days_ms

        # Group miners by current bucket
        bucket_groups = defaultdict(list)
        for hotkey, ledger in ledger_dict.items():
            bucket = challengeperiod_client.get_miner_bucket(hotkey)
            # Handle None case - use UNKNOWN as default
            if bucket is None:
                bt.logging.warning(
                    f"get_miner_bucket returned None for hotkey {hotkey[:16]}...{hotkey[-8:]} in dust calculation. "
                    f"Using {MinerBucket.UNKNOWN.value} as default bucket."
                )
                bucket_value = MinerBucket.UNKNOWN.value
            else:
                bucket_value = bucket.value
            bucket_groups[bucket_value].append((hotkey, ledger))

        if verbose:
            bt.logging.info(
                f"Performance-scaled dust: Processing {len(ledger_dict)} miners across "
                f"{len(bucket_groups)} buckets (30-day lookback, static dust={base_dust:.8f})"
            )

        # Process each bucket independently
        for bucket, miners in bucket_groups.items():
            floor_multiplier = BUCKET_DUST_FLOORS.get(bucket, 1)
            floor = floor_multiplier * base_dust
            ceiling = floor + base_dust  # +1 DUST range above floor

            if verbose:
                bucket_name = MinerBucket(bucket).name if bucket in [b.value for b in MinerBucket] else "UNKNOWN"
                bt.logging.debug(
                    f"Performance-scaled dust bucket {bucket_name}: {len(miners)} miners, "
                    f"floor={floor:.8f}, ceiling={ceiling:.8f}"
                )

            # Calculate 30-day PnL for all miners in bucket
            pnl_scores = DebtBasedScoring._calculate_pnl_scores_for_bucket(
                miners=miners,
                lookback_start_ms=lookback_start,
                current_time_ms=current_time_ms
            )

            # Calculate zero-weight miners for CHALLENGE bucket (collateral-aware)
            zero_weight_miners = set()
            if bucket == MinerBucket.CHALLENGE.value:
                zero_weight_miners = DebtBasedScoring._calculate_challenge_zero_weight_miners(
                    pnl_scores=pnl_scores,
                    contract_client=contract_client,
                    percentile=0.25,
                    max_zero_weight_miners=10
                )
                if zero_weight_miners and verbose:
                    bt.logging.info(
                        f"CHALLENGE bucket: {len(pnl_scores)} miners, "
                        f"{len(zero_weight_miners)} miners get 0 weight (collateral-aware prioritization)"
                    )

            # Assign weights based on PnL scores with performance scaling
            if pnl_scores:
                bucket_weights = DebtBasedScoring._assign_weights_with_performance_scaling(
                    pnl_scores=pnl_scores,
                    bucket=bucket,
                    floor=floor,
                    ceiling=ceiling,
                    zero_weight_miners=zero_weight_miners,
                    verbose=verbose
                )
                dynamic_weights.update(bucket_weights)

        if verbose:
            bt.logging.info(f"Performance-scaled dust weights calculated for {len(dynamic_weights)} miners")

        return dynamic_weights

    @staticmethod
    def _apply_minimum_weights(
        ledger_dict: dict[str, DebtLedger],
        miner_remaining_payouts_usd: dict[str, float],
        challengeperiod_client: 'ChallengePeriodClient',
        contract_client: 'ContractClient',
        metagraph: 'bt.metagraph_handle',
        current_time_ms: int = None,
        projected_daily_emissions_usd: float = None,
        verbose: bool = False
    ) -> dict[str, float]:
        """
        Enforce minimum weights based on challenge period status with performance scaling.

        All miners receive minimum "dust" weights based on their current status:
        - CHALLENGE/PLAGIARISM: 1x dust
        - PROBATION: 2x dust
        - MAINCOMP: 3x dust
        - UNKNOWN: 0x dust (no weight)

        Dust value is a static constant taken from ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT.

        Performance scaling is always enabled: miners are scaled within bucket based on 30-day
        penalty-adjusted PnL (in USD), with range [floor, floor+1 DUST], where floor is
        the bucket's static dust multiplier and ceiling is floor + base dust amount.

        IMPORTANT: Weights are normalized against projected daily emissions (NOT total payouts).
        This ensures excess emissions are burned when we have surplus capacity.

        Args:
            ledger_dict: Dict of {hotkey: DebtLedger}
            miner_remaining_payouts_usd: Dict of {hotkey: remaining_payout_usd} in USD (daily targets)
            challengeperiod_client: Client for querying current challenge period status (required)
            contract_client: Client for querying miner collateral balances (required)
            metagraph: Shared IPC metagraph (not used for dust calculation)
            current_time_ms: Current timestamp (required for performance scaling)
            projected_daily_emissions_usd: Projected daily emissions in USD (for normalization)
            verbose: Enable detailed logging

        Returns:
            Dict of {hotkey: weight} with minimums applied (weights are unitless proportions)
        """
        # Use static dust weight from config
        DUST = ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

        # Calculate dynamic dust weights (always enabled)
        if current_time_ms is None:
            bt.logging.warning(
                "current_time_ms not provided. Falling back to static dust weights."
            )
            dynamic_dust_weights = None
        else:
            try:
                dynamic_dust_weights = DebtBasedScoring._calculate_dynamic_dust_weights(
                    ledger_dict=ledger_dict,
                    challengeperiod_client=challengeperiod_client,
                    contract_client=contract_client,
                    current_time_ms=current_time_ms,
                    base_dust=DUST,
                    verbose=verbose
                )
                if verbose:
                    bt.logging.info("Using performance-scaled dust weights (30-day PnL scaling within buckets)")
            except Exception as e:
                bt.logging.error(f"Error calculating performance-scaled dust weights: {e}. Falling back to static floor values.")
                dynamic_dust_weights = None

        # Static minimum weights (fallback)
        status_to_minimum_weight = {
            MinerBucket.CHALLENGE.value: 1 * DUST,
            MinerBucket.PLAGIARISM.value: 1 * DUST,
            MinerBucket.UNKNOWN.value: 0 * DUST,  # 0x dust (no weight for unknown status)
            MinerBucket.PROBATION.value: 2 * DUST,
            MinerBucket.MAINCOMP.value: 3 * DUST,
        }

        # Batch read all statuses in one IPC call to minimize overhead
        miner_statuses = {}
        for hotkey in ledger_dict.keys():
            bucket = challengeperiod_client.get_miner_bucket(hotkey)
            # Handle None case - use UNKNOWN as default
            if bucket is None:
                bt.logging.warning(
                    f"get_miner_bucket returned None for hotkey {hotkey[:16]}...{hotkey[-8:]}. "
                    f"Using {MinerBucket.UNKNOWN.value} as default status."
                )
                miner_statuses[hotkey] = MinerBucket.UNKNOWN.value
            else:
                miner_statuses[hotkey] = bucket.value

        # Step 1: Convert daily target payouts to weights based on projected daily emissions
        # CRITICAL FIX: Normalize against projected emissions (NOT total payouts!)
        # This ensures excess emissions are burned when we have surplus capacity.
        # Example: If daily targets = $30k but emissions = $1.7M, weights sum to 0.0175 (1.75%)
        # and burn address gets 0.9825 (98.25%)
        total_daily_target_payout = sum(miner_remaining_payouts_usd.values())

        if projected_daily_emissions_usd is None or projected_daily_emissions_usd <= 0:
            # Fallback to old behavior (normalize to 1.0) if projected emissions not provided
            bt.logging.warning(
                "projected_daily_emissions_usd not provided or invalid. "
                "Falling back to normalizing against total payouts (may not burn excess emissions)."
            )
            if total_daily_target_payout > 0:
                normalized_debt_weights = {
                    hotkey: (payout_usd / total_daily_target_payout)
                    for hotkey, payout_usd in miner_remaining_payouts_usd.items()
                }
            else:
                normalized_debt_weights = {hotkey: 0.0 for hotkey in ledger_dict.keys()}
        else:
            # NEW: Normalize against projected daily emissions (enables burning surplus)
            if verbose:
                bt.logging.info(
                    f"Normalizing weights against projected daily emissions: "
                    f"total_daily_target=${total_daily_target_payout:.2f}, "
                    f"projected_daily_emissions=${projected_daily_emissions_usd:.2f}, "
                    f"payout_fraction={total_daily_target_payout/projected_daily_emissions_usd:.4f}"
                )

            normalized_debt_weights = {
                hotkey: (payout_usd / projected_daily_emissions_usd)
                for hotkey, payout_usd in miner_remaining_payouts_usd.items()
            }

        # Step 2: Apply minimum weights (now both are in 0-1 range)
        miner_weights_with_minimums = {}

        for hotkey, debt_ledger in ledger_dict.items():
            # Get normalized debt-based weight (proportional, 0-1 range)
            debt_weight = normalized_debt_weights.get(hotkey, 0.0)

            # Get current status from batch-loaded statuses
            current_status = miner_statuses.get(hotkey, MinerBucket.UNKNOWN.value)

            # Get minimum weight (dynamic or static)
            if dynamic_dust_weights is not None and hotkey in dynamic_dust_weights:
                minimum_weight = dynamic_dust_weights[hotkey]
            else:
                # Fallback to static dust weight
                minimum_weight = status_to_minimum_weight.get(current_status, 1 * DUST)

            # Apply max(debt_weight, minimum_weight) - now both are in same scale!
            final_weight = max(debt_weight, minimum_weight)

            miner_weights_with_minimums[hotkey] = final_weight

            if verbose:
                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"status={current_status}, "
                    f"debt_weight={debt_weight:.8f}, "
                    f"minimum={minimum_weight:.8f}, "
                    f"final={final_weight:.8f}"
                )

        return miner_weights_with_minimums

    @staticmethod
    def _get_burn_address_hotkey(
        metagraph: 'bt.metagraph_handle',
        is_testnet: bool = False
    ) -> str:
        """
        Get the hotkey for the burn address.

        Args:
            metagraph: Bittensor metagraph for accessing hotkeys
            is_testnet: True for testnet (uid 220), False for mainnet (uid 229)

        Returns:
            Hotkey string for burn address (uid 229 mainnet / uid 220 testnet)
        """
        burn_uid = DebtBasedScoring.get_burn_uid(is_testnet)

        # Get hotkey for burn UID
        hotkeys = metagraph.get_hotkeys()
        if burn_uid < len(hotkeys):
            return hotkeys[burn_uid]
        else:
            bt.logging.warning(
                f"Burn UID {burn_uid} not found in metagraph "
                f"(only {len(hotkeys)} UIDs). Using placeholder."
            )
            return f"burn_uid_{burn_uid}"

    @staticmethod
    def _normalize_with_burn_address(
        weights: dict[str, float],
        metagraph: 'bt.metagraph_handle',
        is_testnet: bool = False,
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Normalize weights with special burn address logic.

        If sum of weights < 1.0:
            - Assign remaining weight (1.0 - sum) to burn address (uid 229 mainnet / uid 220 testnet)
        If sum of weights >= 1.0:
            - Normalize all weights to sum to 1.0
            - Burn address gets 0 (not included)

        Args:
            weights: Dict of {hotkey: weight}
            metagraph: Bittensor metagraph for accessing hotkeys
            is_testnet: True for testnet (uid 220), False for mainnet (uid 229)
            verbose: Enable detailed logging

        Returns:
            List of (hotkey, weight) tuples sorted by weight (descending)
        """
        if not weights:
            bt.logging.info("No weights to normalize, returning empty list")
            return []

        sum_weights = sum(weights.values())

        if verbose:
            bt.logging.info(f"Sum of weights before normalization: {sum_weights:.6f}")

        burn_uid = DebtBasedScoring.get_burn_uid(is_testnet)

        if sum_weights < 1.0:
            # Excess weight goes to burn address
            burn_weight = 1.0 - sum_weights

            # Get burn address hotkey
            burn_hotkey = DebtBasedScoring._get_burn_address_hotkey(metagraph, is_testnet)

            bt.logging.info(
                f"Sum of weights ({sum_weights:.6f}) < 1.0. "
                f"Assigning {burn_weight:.6f} to burn address (uid {burn_uid})"
            )

            # Create result with original weights + burn address
            result = [(hotkey, weight) for hotkey, weight in weights.items()]
            result.append((burn_hotkey, burn_weight))

        else:
            # Sum >= 1.0: normalize to exactly 1.0
            bt.logging.info(
                f"Sum of weights ({sum_weights:.6f}) >= 1.0. "
                f"Normalizing to 1.0, burn address gets 0."
            )

            # Use standard normalization
            normalized_weights = Scoring.normalize_scores(weights)
            result = [(hotkey, weight) for hotkey, weight in normalized_weights.items()]

        # Sort by weight descending
        result = sorted(result, key=lambda x: x[1], reverse=True)

        return result

    @staticmethod
    def _apply_pre_activation_weights(
        ledger_dict: dict[str, DebtLedger],
        metagraph: 'bt.metagraph_handle',
        challengeperiod_client: 'ChallengePeriodClient',
        contract_client: 'ContractClient',
        current_time_ms: int = None,
        is_testnet: bool = False,
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Apply weights for pre-activation period (before December 2025).

        During pre-activation, miners only receive minimum dust weights based on
        their challenge period status. Excess weight goes to burn address.
        Performance-based scaling within buckets is always enabled (using static dust value).

        Args:
            ledger_dict: Dict of {hotkey: DebtLedger}
            metagraph: Bittensor metagraph for accessing hotkeys
            challengeperiod_client: Client for querying current challenge period status (required)
            contract_client: Client for querying miner collateral balances (required)
            current_time_ms: Current timestamp (required for performance-scaled dust calculation)
            is_testnet: True for testnet (uid 220), False for mainnet (uid 229)
            verbose: Enable detailed logging

        Returns:
            List of (hotkey, weight) tuples with dust weights + burn address
        """
        # Apply minimum dust weights only (no debt-based earnings)
        miner_dust_weights = DebtBasedScoring._apply_minimum_weights(
            ledger_dict=ledger_dict,
            miner_remaining_payouts_usd={hotkey: 0.0 for hotkey in ledger_dict.keys()},  # No debt earnings
            challengeperiod_client=challengeperiod_client,
            contract_client=contract_client,
            metagraph=metagraph,
            current_time_ms=current_time_ms,
            verbose=verbose
        )

        # Apply burn address normalization
        result = DebtBasedScoring._normalize_with_burn_address(
            weights=miner_dust_weights,
            metagraph=metagraph,
            is_testnet=is_testnet,
            verbose=verbose
        )

        if verbose:
            bt.logging.info(
                f"Pre-activation weights: {len(ledger_dict)} miners with dust weights, "
                f"excess to burn address"
            )

        return result
