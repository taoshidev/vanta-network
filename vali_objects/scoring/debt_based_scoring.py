"""
Debt-Based Scoring

This module computes miner weights based on debt ledger information.
The algorithm pays miners based on their previous payout period's performance (PnL scaled by penalties),
proportionally distributing emissions to cover remaining debt by end of the current week.

Key Concepts:
- Activation began 2025-11-01, and was originally on a monthly basis
- Payout periods are now weekly, starting and ending at midnight on Sunday, 00:00:00 UTC
- "Needed payout" = Cumulative earnings from activation through end of previous pay period (PnL in USD * penalties)
- "Actual payout" = Cumulative emissions paid from activation through current time (emissions in USD)
- "Remaining payout" = needed_payout_usd - actual_payout_usd (in USD)
- "Projected emissions" = Estimated total ALPHA available, converted to USD for comparison
- Weights = Proportional to remaining_payout_usd, with warning if insufficient emissions
- Cumulative tracking allows negative PnL to carry forward and offset future gains

Algorithm Flow:
- Calculate needed_payout_usd from activation through end of previous pay period (only MAINCOMP/PROBATION checkpoints)
- Calculate actual_payout_usd from activation through current time (only MAINCOMP/PROBATION checkpoints)
- Calculate remaining_payout_usd for each miner (in USD)
- Query real-time TAO emission rate from subtensor
- Convert to ALPHA, then convert ALPHA to USD using current conversion rates
- Project total USD value available over current pay period
- Set weights proportional to remaining_payout_usd
- Warn if sum(remaining_payouts_usd) > projected_usd_emissions
- Enforce minimum weights based on challenge period status:
    - CHALLENGE/PLAGIARISM: 1x dust
    - PROBATION: 2x dust
    - MAINCOMP: 3x dust
    - UNKNOWN: 0x dust (no weight)
- Normalize weights with burn address logic:
    - If sum < 1.0: assign (1.0 - sum) to burn address (uid 229 mainnet / uid 5 testnet)
    - If sum >= 1.0: normalize to 1.0, burn address gets 0

Important Notes:
- Debt-based scoring activated December 2025 (on a monthly basis, paid for November 2025)
- Before December 2025, miners only received minimum dust weights
- Excess weight (when sum < 1.0) goes to burn address (uid 229 mainnet, uid 220 testnet)
- Checkpoints are 12-hour intervals (2 per day)
- Uses real-time subtensor queries for emission rate estimation
"""

import bittensor as bt
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from shared_objects.rpc.metagraph_client import MetagraphClient
from time_util.time_util import TimeUtil
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtLedger, DebtCheckpoint
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig
from vali_objects.scoring.scoring import Scoring
from collections import defaultdict


class DebtBasedScoring:
    """
    Debt-based scoring system that pays miners proportionally to their cumulative performance
    from activation through the previous pay period, targeting payout completion by the
    beginning of the next pay period.

    Uses cumulative tracking to allow negative PnL to carry forward and offset future gains.
    Uses real-time subtensor queries to estimate emission rates and project available ALPHA.
    """

    # Activation: First payouts in December 2025 for November 2025 performance
    ACTIVATION_YEAR = 2025
    ACTIVATION_MONTH = 11

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
    def log_projections(metagraph_client, days_until_target, verbose, total_remaining_payout_usd):
        """
        Log emission projections and compare to remaining payout needs.

        Args:
            metagraph_client: Bittensor metagraph with emission data
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
            metagraph_client=metagraph_client,
            days_until_target=days_until_target,
            verbose=verbose
        )

        # Convert projected ALPHA to USD for comparison
        projected_usd_available = DebtBasedScoring._convert_alpha_to_usd(
            alpha_amount=projected_alpha_available,
            metagraph_client=metagraph_client,
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
                f"Miners will receive proportional payouts."
            )
        else:
            surplus_pct = ((projected_usd_available - total_remaining_payout_usd) / total_remaining_payout_usd) * 100
            bt.logging.info(
                f"✓ Projected USD value in next {days_until_target} days (${projected_usd_available:.2f}) exceeds "
                f"total remaining payout needed (${total_remaining_payout_usd:.2f}). "
                f"Surplus: {surplus_pct:.1f}%. "
            )

    @staticmethod
    def calculate_payout_from_checkpoints(
        checkpoints: List[DebtCheckpoint]
    ) -> float:
        """
        Calculate payout from a list of debt checkpoints using HWM-gated scoring.

        Only pays for incremental gains above the prior realized PnL high water mark.
        This ensures miners only earn emissions when making new cumulative highs,
        not when recovering from drawdowns.

        NOTE: realized_pnl and unrealized_pnl are in USD, per-checkpoint values (NOT cumulative)

        Formula:
        - Track cumulative_realized = running sum of realized_pnl across checkpoints
        - Track realized_hwm = highest cumulative_realized seen so far
        - For each checkpoint where cumulative_realized > realized_hwm:
            realized_component += (cumulative_realized - realized_hwm) * cp.total_penalty
            realized_hwm = cumulative_realized
        - unrealized_component = min(0.0, last_cp.unrealized_pnl) * last_cp.total_penalty
        - payout = realized_component + unrealized_component

        Args:
            checkpoints: List of DebtCheckpoint objects (should be in chronological order)

        Returns:
            Calculated payout in USD (can be negative if unrealized losses exceed gains)
        """
        if not checkpoints:
            return 0.0

        # HWM-gated realized component: only pay the delta above prior cumulative peak
        cumulative_realized = 0.0
        realized_hwm = 0.0
        realized_component = 0.0

        for cp in checkpoints:
            cumulative_realized += cp.realized_pnl
            if cumulative_realized > realized_hwm:
                delta = cumulative_realized - realized_hwm
                realized_component += delta * cp.total_penalty
                realized_hwm = cumulative_realized

        # Unrealized component: min(0, unrealized_pnl) * penalty of last checkpoint
        # (only count unrealized losses, not gains)
        last_checkpoint = checkpoints[-1]
        unrealized_component = min(0.0, last_checkpoint.unrealized_pnl) * last_checkpoint.total_penalty

        payout = realized_component + unrealized_component
        return payout

    @staticmethod
    def compute_results(
        ledger_dict: dict[str, DebtLedger],
        metagraph_client: 'MetagraphClient',
        challengeperiod_client: 'ChallengePeriodClient',
        miner_account_client: 'MinerAccountClient',
        current_time_ms: int = None,
        verbose: bool = False,
        is_testnet: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Compute miner weights based on debt ledger information with real-time emission projections.

        The algorithm works as follows:
        - For each miner, calculate their "needed payout" from activation through end of previous pay period (cumulative)
        - Calculate "actual payout" from activation through current time (cumulative)
        - Calculate "remaining payout" to be distributed (allows negative PnL to carry forward)
        - Query real-time TAO emission rate from metagraph
        - Convert to ALPHA using reserve data from shared metagraph (TAO/ALPHA ratio)
        - Project total ALPHA available from now until day 25
        - Set weights proportional to remaining_payout
        - Warn if sum(remaining_payouts) > projected_emissions
        - Enforce minimum weights with static dust (performance-scaled by 30-day PnL within buckets)
        - Normalize weights with burn address logic (sum < 1.0 → burn gets excess)

        Args:
            ledger_dict: Dict of {hotkey: DebtLedger} containing debt ledger data
            metagraph_client: Shared IPC metagraph with emission data and substrate reserves
            challengeperiod_client: Client for querying current challenge period status (required)
            miner_account_client: Client for querying miner account sizes (required)
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
            burn_hotkey = DebtBasedScoring._get_burn_address_hotkey(metagraph_client, is_testnet)
            return [(burn_hotkey, 1.0)]

        # Get current datetime
        current_dt = TimeUtil.millis_to_datetime(current_time_ms)

        if verbose:
            bt.logging.info(
                f"Computing debt-based weights for {current_dt.strftime('%B %Y')} "
                f"({len(ledger_dict)} miners)"
            )

        # Calculate boundaries
        # Needed payout calculation: Sum from activation through end of previous
        # week (considered midnight on Sunday 00:00:00)
        # This allows negative PnL to carry across weeks and offset future gains
        payout_calc_start_dt = datetime(
            DebtBasedScoring.ACTIVATION_YEAR,
            DebtBasedScoring.ACTIVATION_MONTH,
            1, 0, 0, 0,
            tzinfo=timezone.utc
        )
        payout_calc_start_ms = int(payout_calc_start_dt.timestamp() * 1000)

        current_weekday = current_dt.weekday()
        prev_target_day_offset = (current_weekday + 1) % 7
        days_until_target = 7 - prev_target_day_offset
        prev_target_dt = current_dt - timedelta(days=prev_target_day_offset)
        prev_target_end_dt = datetime.combine(prev_target_dt, datetime.min.time())
        prev_target_end_ms = int(prev_target_end_dt.timestamp() * 1000)

        if verbose:
            bt.logging.info(
                f"Needed payout window (cumulative): {payout_calc_start_dt.strftime('%Y-%m-%d')} to "
                f"{prev_target_end_dt.strftime('%Y-%m-%d')} "
                f"(allows negative PnL to carry across weeks)"
            )

        # Process each miner to calculate remaining payouts (in USD)
        miner_remaining_payouts_usd = {}
        miner_actual_payouts_usd = {}  # Track what's been paid so far this pay period
        miner_penalty_loss_usd = {}  # Track how much was lost to penalties

        for hotkey, debt_ledger in ledger_dict.items():
            if not debt_ledger.checkpoints:
                if verbose:
                    bt.logging.debug(f"Skipping {hotkey}: no checkpoints")
                miner_remaining_payouts_usd[hotkey] = 0.0
                miner_actual_payouts_usd[hotkey] = 0.0
                continue

            # Extract checkpoints from activation through end of previous pay period (cumulative)
            # This allows negative PnL to accumulate and offset future gains
            cumulative_checkpoints = [
                cp for cp in debt_ledger.checkpoints
                if payout_calc_start_ms <= cp.timestamp_ms <= prev_target_end_ms
            ]

            # Only include checkpoints where status is MAINCOMP or PROBATION (earning periods)
            earning_checkpoints = [
                cp for cp in cumulative_checkpoints
                if cp.challenge_period_status in (
                    MinerBucket.MAINCOMP.value,
                    MinerBucket.PROBATION.value,
                    MinerBucket.SUBACCOUNT_FUNDED.value,
                    MinerBucket.SUBACCOUNT_ALPHA.value
                )
            ]

            # Calculate needed payout from activation through end of previous pay period (in USD)
            # "needed payout" = sum of (realized_pnl * total_penalty) across all earning checkpoints
            #                   and (unrealized_pnl * total_penalty) of the last checkpoint
            # NOTE:
            # realized_pnl and unrealized_pnl are both in USD. unrealized_pnl is cumulative.
            # realized_pnl is a per-checkpoint value (NOT cumulative).
            # This cumulative approach allows negative PnL to carry forward and offset future gains.
            needed_payout_usd = 0.0
            penalty_loss_usd = 0.0
            if earning_checkpoints:
                # Sum penalty-adjusted PnL across all checkpoints from activation to end of prev pay period
                # Each checkpoint has its own PnL (for that 12-hour period) and its own penalty
                needed_payout_usd = DebtBasedScoring.calculate_payout_from_checkpoints(
                    earning_checkpoints
                )

                last_checkpoint = earning_checkpoints[-1]
                # Calculate penalty loss: what would have been earned WITHOUT penalties
                payout_without_penalties = sum(cp.realized_pnl for cp in earning_checkpoints)
                payout_without_penalties += min(0.0, last_checkpoint.unrealized_pnl)
                penalty_loss_usd = payout_without_penalties - needed_payout_usd

            # Calculate actual payout (in USD)
            # Use cumulative approach: sum all emissions from activation through current time
            # This matches the cumulative needed payout calculation
            cumulative_payout_checkpoints = [
                cp for cp in debt_ledger.checkpoints
                if payout_calc_start_ms <= cp.timestamp_ms <= current_time_ms
                and cp.challenge_period_status in (
                    MinerBucket.MAINCOMP.value,
                    MinerBucket.PROBATION.value,
                    MinerBucket.SUBACCOUNT_FUNDED.value,
                    MinerBucket.SUBACCOUNT_ALPHA.value
                )
            ]
            actual_payout_usd = sum(cp.chunk_emissions_usd for cp in cumulative_payout_checkpoints)

            # Calculate remaining payout (in USD)
            remaining_payout_usd = needed_payout_usd - actual_payout_usd

            # Log debt calculation details
            bt.logging.info(
                f"[PAYOUT_DEBUG] DEBT CALC [{hotkey}]: total_needed_payout=${needed_payout_usd:.2f}\t"
                f"total_cumulative_emissions=${actual_payout_usd:.2f}, remaining=${remaining_payout_usd:.2f}, "
                f"penalty_loss=${penalty_loss_usd:.2f}, earning_cps={len(earning_checkpoints)}"
            )

            # Clamp to zero if negative (over-paid or negative performance)
            if remaining_payout_usd < 0:
                remaining_payout_usd = 0.0

            miner_remaining_payouts_usd[hotkey] = remaining_payout_usd
            miner_actual_payouts_usd[hotkey] = actual_payout_usd
            miner_penalty_loss_usd[hotkey] = penalty_loss_usd

        # Query real-time emissions and project availability (in USD)
        total_remaining_payout_usd = sum(miner_remaining_payouts_usd.values())
        total_actual_payout_usd = sum(miner_actual_payouts_usd.values())
        total_needed_payout_usd = total_remaining_payout_usd + total_actual_payout_usd

        bt.logging.info(
            f"[PAYOUT_DEBUG] PAYOUT TOTALS: needs=${total_needed_payout_usd:.2f}, "
            f"paid_so_far=${total_actual_payout_usd:.2f}, remaining=${total_remaining_payout_usd:.2f}"
        )

        # Calculate projected emissions (needed for weight normalization)
        # Get projected ALPHA emissions
        projected_alpha_available = DebtBasedScoring._estimate_alpha_emissions_until_target(
            metagraph_client=metagraph_client,
            days_until_target=days_until_target,
            verbose=verbose
        )

        # Convert projected ALPHA to USD for comparison
        projected_usd_available = DebtBasedScoring._convert_alpha_to_usd(
            alpha_amount=projected_alpha_available,
            metagraph_client=metagraph_client,
            verbose=verbose
        )

        bt.logging.info(
            f"[PAYOUT_DEBUG] PROJECTED EMISSIONS: {projected_alpha_available:.2f} ALPHA = ${projected_usd_available:.2f} USD "
            f"over {days_until_target} days (${projected_usd_available / days_until_target:.2f}/day)"
        )

        if total_remaining_payout_usd > 0:
            DebtBasedScoring.log_projections(metagraph_client, days_until_target, verbose, total_remaining_payout_usd)
        else:
            bt.logging.info(
                f"No remaining payouts needed {total_remaining_payout_usd} or no days until target "
                f"{days_until_target}, skipping projection log"
            )

        # Calculate daily target payouts
        # Instead of paying the entire remaining amount at once, spread it over days_until_target
        miner_daily_target_payouts_usd = {}
        for hotkey, remaining_payout_usd in miner_remaining_payouts_usd.items():
            daily_target = remaining_payout_usd / days_until_target
            miner_daily_target_payouts_usd[hotkey] = daily_target

        # Enforce minimum weights based on challenge period status
        # All miners get minimum "dust" weights based on their current status
        # Dust is a static value from ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
        # Weights are performance-scaled by 30-day PnL within each bucket
        # NOTE: Weights are unitless proportions, normalized against projected daily emissions
        # Calculate projected daily emissions in USD
        projected_daily_usd = projected_usd_available / days_until_target

        miner_weights_with_minimums = DebtBasedScoring._apply_minimum_weights(
            ledger_dict=ledger_dict,
            miner_remaining_payouts_usd=miner_daily_target_payouts_usd,
            challengeperiod_client=challengeperiod_client,
            miner_account_client=miner_account_client,
            current_time_ms=current_time_ms,
            projected_daily_emissions_usd=projected_daily_usd,
            verbose=verbose
        )

        # Log weight summary before normalization
        bt.logging.info(
            f"[PAYOUT_DEBUG] WEIGHT SUMMARY: {len(miner_weights_with_minimums)} miners, "
            f"total_remaining_payout=${total_remaining_payout_usd:.2f}, "
            f"projected_daily_usd=${projected_daily_usd:.2f}, "
            f"days_until_target={days_until_target}"
        )
        for hk, w in sorted(miner_weights_with_minimums.items(), key=lambda x: -x[1])[:10]:
            daily_target = miner_daily_target_payouts_usd.get(hk, 0.0)
            bt.logging.info(
                f"[PAYOUT_DEBUG] TOP WEIGHT [{hk}]: weight={w:.8f}, daily_target=${daily_target:.2f}"
            )

        # Normalize weights with special burn address logic
        # If sum < 1.0: assign remaining weight to burn address (uid 229 / uid 5)
        # If sum >= 1.0: normalize to 1.0, burn address gets 0
        result = DebtBasedScoring._normalize_with_burn_address(
            weights=miner_weights_with_minimums,
            metagraph_client=metagraph_client,
            is_testnet=is_testnet,
            verbose=verbose
        )

        return result

    @staticmethod
    def _estimate_alpha_emissions_until_target(
        metagraph_client: 'MetagraphClient',
        days_until_target: int,
        verbose: bool = False
    ) -> float:
        """
        Estimate total ALPHA emissions available from now until target day.

        Uses real-time metagraph data to get current TAO emission rate,
        then converts to ALPHA using reserve data from shared metagraph.

        Args:
            metagraph_client: Shared IPC metagraph with emission data and substrate reserves
            days_until_target: Number of days until target payout day
            verbose: Enable detailed logging

        Returns:
            Estimated total ALPHA emissions available (float)
        """
        try:
            total_alpha_per_tempo = sum(metagraph_client.get_emission())
            total_alpha_per_block = total_alpha_per_tempo / 360
            if verbose:
                bt.logging.info(f"Current subnet emission rate: {total_alpha_per_block:.6f} alpha/block")

            # Estimate blocks until target day
            # Use approximate 12 seconds per block (7200 blocks/day)
            blocks_until_target = days_until_target * DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK

            # Calculate total TAO emissions until target
            total_alpha_until_target = total_alpha_per_block * blocks_until_target
            if verbose:
                bt.logging.info(f"Projected ALPHA available until target: {total_alpha_until_target:.2f}")
            return total_alpha_until_target

            # # Get total TAO emission per block for the subnet (sum across all miners)
            # # metagraph.emission is already in TAO (not RAO), but per tempo (360 blocks)
            # # Need to convert: per-tempo → per-block (÷360)
            # total_tao_per_tempo = sum(metagraph_client.get_emission())
            # total_tao_per_block = total_tao_per_tempo / 360
            #
            # if verbose:
            #     bt.logging.info(f"Current subnet emission rate: {total_tao_per_block:.6f} TAO/block")
            #
            # # Estimate blocks until target day
            # # Use approximate 12 seconds per block (7200 blocks/day)
            # blocks_until_target = days_until_target * DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK
            #
            # # Calculate total TAO emissions until target
            # total_tao_until_target = total_tao_per_block * blocks_until_target
            #
            # if verbose:
            #     bt.logging.info(
            #         f"Estimated blocks until day {DebtBasedScoring.PAYOUT_TARGET_DAY}: {blocks_until_target}, "
            #         f"total TAO: {total_tao_until_target:.2f}"
            #     )
            #
            # # Get substrate reserves from shared metagraph (refreshed by SubtensorOpsManager)
            # # Use safe helper to extract values from manager.Value() objects or plain numerics
            # tao_reserve_obj = getattr(metagraph_client, 'tao_reserve_rao', None)
            # alpha_reserve_obj = getattr(metagraph_client, 'alpha_reserve_rao', None)
            #
            # tao_reserve_rao = DebtBasedScoring._safe_get_reserve_value(tao_reserve_obj)
            # alpha_reserve_rao = DebtBasedScoring._safe_get_reserve_value(alpha_reserve_obj)
            #
            # if tao_reserve_rao == 0 or alpha_reserve_rao == 0:
            #     bt.logging.warning(
            #         "Substrate reserve data not available in shared metagraph "
            #         f"(TAO={tao_reserve_rao} RAO, ALPHA={alpha_reserve_rao} RAO). "
            #         "Cannot calculate ALPHA conversion rate."
            #     )
            #     return 0.0
            #
            # # Calculate ALPHA-to-TAO conversion rate from reserve data
            # # alpha_to_tao_rate = tao_reserve / alpha_reserve (both in RAO, ratio is unitless)
            # # (How much TAO per ALPHA)
            # alpha_to_tao_rate = tao_reserve_rao / alpha_reserve_rao
            #
            # if verbose:
            #     bt.logging.info(
            #         f"Substrate reserves: TAO={tao_reserve_rao / 1e9:.2f} TAO ({tao_reserve_rao:.0f} RAO), "
            #         f"ALPHA={alpha_reserve_rao / 1e9:.2f} ALPHA ({alpha_reserve_rao:.0f} RAO), "
            #         f"rate={alpha_to_tao_rate:.6f} TAO/ALPHA"
            #     )
            #
            # # Convert TAO to ALPHA
            # # If ALPHA costs X TAO per ALPHA, then Y TAO buys Y/X ALPHA
            # if alpha_to_tao_rate > 0:
            #     total_alpha_until_target = total_tao_until_target / alpha_to_tao_rate
            # else:
            #     bt.logging.warning("ALPHA-to-TAO rate is zero, cannot convert")
            #     return 0.0
            #
            # if verbose:
            #     bt.logging.info(f"Projected ALPHA available until target: {total_alpha_until_target:.2f}")
            #
            # return total_alpha_until_target

        except Exception as e:
            bt.logging.error(f"Error estimating ALPHA emissions: {e}")
            raise

    @staticmethod
    def _convert_alpha_to_usd(
        alpha_amount: float,
        metagraph_client: 'MetagraphClient',
        verbose: bool = False
    ) -> float:
        """
        Convert ALPHA amount to USD value using current market rates.

        Uses reserve data from shared metagraph to calculate conversion rate:
        ALPHA → TAO (via reserves) → USD (via TAO price oracle)

        Args:
            alpha_amount: Amount of ALPHA tokens to convert
            metagraph_client: Shared IPC metagraph with substrate reserves
            verbose: Enable detailed logging

        Returns:
            USD value of the ALPHA amount (float)
        """
        if alpha_amount == 0:
            return 0.0

        # Get substrate reserves from shared metagraph
        # Use safe helper to extract values from manager.Value() objects or plain numerics
        tao_reserve_obj = getattr(metagraph_client, 'tao_reserve_rao', None)
        alpha_reserve_obj = getattr(metagraph_client, 'alpha_reserve_rao', None)

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
        # This is set by SubtensorOpsManager via live_price_fetcher.get_close_at_date(TradePair.TAOUSD)
        tao_to_usd_rate_raw = getattr(metagraph_client, 'tao_to_usd_rate', None)

        # Validate that we have a valid TAO/USD rate
        if tao_to_usd_rate_raw is None:
            raise ValueError(
                "TAO/USD price not available in metagraph. "
                "SubtensorOpsManager must set metagraph.tao_to_usd_rate via live_price_fetcher."
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
                MinerBucket.PROBATION.value,
                MinerBucket.SUBACCOUNT_FUNDED.value,
                MinerBucket.SUBACCOUNT_ALPHA.value
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

        # HWM-gated realized component: only pay the delta above prior cumulative peak
        # Each checkpoint has its own PnL (for that 12-hour period) and its own penalty
        cumulative_realized = 0.0
        realized_hwm = 0.0
        penalty_adjusted_pnl = 0.0

        for cp in relevant_checkpoints:
            cumulative_realized += cp.realized_pnl
            if cumulative_realized > realized_hwm:
                delta = cumulative_realized - realized_hwm
                penalty_adjusted_pnl += delta * cp.total_penalty
                realized_hwm = cumulative_realized

        last_checkpoint = relevant_checkpoints[-1]
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
        miner_account_client: 'MinerAccountClient',
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
            miner_account_client: MinerAccount client for account size queries
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
            collateral_usd = miner_account_client.get_miner_account_size(hotkey, most_recent=True)
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
    ) -> float | None:
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
        miner_account_client: 'MinerAccountClient',
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
            miner_account_client: Client for querying miner account sizes (required)
            current_time_ms: Current timestamp
            base_dust: Static dust value from ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
            verbose: Enable detailed logging

        Returns:
            Dict mapping hotkey -> performance_scaled_dust_weight (unitless proportion)
        """
        # Original dust floor multipliers (respecting existing system)
        BUCKET_DUST_FLOORS = {
            MinerBucket.CHALLENGE.value: 1,  # 1x dust floor
            MinerBucket.PROBATION.value: 2,  # 2x dust floor
            MinerBucket.MAINCOMP.value: 3,  # 3x dust floor
            MinerBucket.UNKNOWN.value: 0,  # 0x dust (no weight for unknown status)
            MinerBucket.PLAGIARISM.value: 1,  # 1x dust floor
            # Entity bucket (synthetic hotkeys don't need dust - not in metagraph)
            MinerBucket.ENTITY.value: 4,  # 4x dust floor
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
                    miner_account_client=miner_account_client,
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
        miner_account_client: 'MinerAccountClient',
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
            miner_account_client: Client for querying miner account sizes (required)
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
                    miner_account_client=miner_account_client,
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
            # Entity bucket (synthetic hotkeys don't need dust - not in metagraph)
            MinerBucket.ENTITY.value: 4 * DUST,
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
                    f"payout_fraction={total_daily_target_payout / projected_daily_emissions_usd:.4f}"
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
        metagraph_client: 'MetagraphClient',
        is_testnet: bool = False
    ) -> str:
        """
        Get the hotkey for the burn address.

        Args:
            metagraph_client: Metagraph client for accessing hotkeys
            is_testnet: True for testnet (uid 220), False for mainnet (uid 229)

        Returns:
            Hotkey string for burn address (uid 229 mainnet / uid 220 testnet)
        """
        burn_uid = DebtBasedScoring.get_burn_uid(is_testnet)

        # Get hotkey for burn UID
        hotkeys = metagraph_client.get_hotkeys()
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
        metagraph_client: 'MetagraphClient',
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
            metagraph_client: Client for accessing hotkeys
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
            burn_hotkey = DebtBasedScoring._get_burn_address_hotkey(metagraph_client, is_testnet)

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
        metagraph_client: 'MetagraphClient',
        challengeperiod_client: 'ChallengePeriodClient',
        miner_account_client: 'MinerAccountClient',
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
            metagraph_client: Bittensor metagraph for accessing hotkeys
            challengeperiod_client: Client for querying current challenge period status (required)
            miner_account_client: Client for querying miner account sizes (required)
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
            miner_account_client=miner_account_client,
            current_time_ms=current_time_ms,
            verbose=verbose
        )

        # Apply burn address normalization
        result = DebtBasedScoring._normalize_with_burn_address(
            weights=miner_dust_weights,
            metagraph_client=metagraph_client,
            is_testnet=is_testnet,
            verbose=verbose
        )

        if verbose:
            bt.logging.info(
                f"Pre-activation weights: {len(ledger_dict)} miners with dust weights, "
                f"excess to burn address"
            )

        return result
