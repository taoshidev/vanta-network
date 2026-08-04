# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
WeightCalculatorManager - Core business logic for weight calculation and setting.

This manager handles all heavy logic for weight calculation operations.
WeightCalculatorServer wraps this and exposes methods via RPC.

This follows the same pattern as ChallengePeriodManager.
"""
from datetime import datetime, timedelta, timezone
import traceback
import threading
from typing import List, Tuple


from shared_objects.cache_controller import CacheController
from shared_objects.error_utils import ErrorUtils
from shared_objects.slack_notifier import SlackNotifier
from time_util.time_util import MS_IN_WEEK, TimeUtil
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from shared_objects.log import logger


class WeightCalculatorManager(CacheController):
    """
    Weight Calculator Manager - Contains all business logic for weight calculation.

    This manager is wrapped by WeightCalculatorServer which exposes methods via RPC.
    All heavy logic resides here - server delegates to this manager.

    Pattern:
    - Server holds a `self._manager` instance
    - Server delegates all RPC methods to manager methods
    - Manager creates its own clients internally (forward compatibility)
    """

    def __init__(
        self,
        *,
        is_backtesting=False,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        config=None,
        hotkey=None,
        is_mainnet=True,
        slack_notifier=None
    ):
        """
        Initialize WeightCalculatorManager.

        Args:
            is_backtesting: Whether running in backtesting mode
            running_unit_tests: Whether running in test mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
            config: Validator config (for slack webhook URLs)
            hotkey: Validator hotkey
            is_mainnet: Whether running on mainnet
            slack_notifier: Optional external slack notifier
        """
        super().__init__(
            running_unit_tests=running_unit_tests,
            is_backtesting=is_backtesting,
            connection_mode=connection_mode
        )

        self.running_unit_tests = running_unit_tests
        self.connection_mode = connection_mode
        self.config = config
        self.hotkey = hotkey
        self.is_mainnet = is_mainnet
        self.subnet_version = 200

        # Create clients internally (forward compatibility - no parameter passing)
        from shared_objects.rpc.common_data_client import CommonDataClient
        self._common_data_client = CommonDataClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        from shared_objects.rpc.metagraph_client import MetagraphClient
        self._metagraph_client = MetagraphClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        from vali_objects.position_management.position_manager_client import PositionManagerClient
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            running_unit_tests=running_unit_tests,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
        self._challengeperiod_client = ChallengePeriodClient(
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode
        )

        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        self._miner_account_client = MinerAccountClient()
        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient

        self._perf_ledger_client = PerfLedgerClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        self._debt_ledger_client = DebtLedgerClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        from shared_objects.subtensor_ops.subtensor_ops_client import SubtensorOpsClient
        self._subtensor_ops_client = SubtensorOpsClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False
        )

        # Slack notifier (lazy initialization)
        self._external_slack_notifier = slack_notifier
        self._slack_notifier = None

        # Verbose logging cooldown
        self._last_verbose_ms: int = 0

        # Store results for external access
        self.checkpoint_results: List[Tuple[str, float]] = []
        self.transformed_list: List[Tuple[int, float]] = []
        self._results_lock = threading.Lock()

        logger.info("[WC_MANAGER] WeightCalculatorManager initialized")

    # ==================== Properties ====================

    @property
    def slack_notifier(self):
        """Get slack notifier (lazy initialization)."""
        if self._external_slack_notifier:
            return self._external_slack_notifier

        if self._slack_notifier is None and self.config and self.hotkey:
            self._slack_notifier = SlackNotifier(
                hotkey=self.hotkey,
                webhook_url=getattr(self.config, 'slack_webhook_url', None),
                error_webhook_url=getattr(self.config, 'slack_error_webhook_url', None),
                is_miner=False
            )
        return self._slack_notifier

    # ==================== Core Business Logic ====================

    def compute_weights(self, current_time: int = None) -> Tuple[List[Tuple[str, float]], List[Tuple[int, float]]]:
        """
        Compute weights for all miners using debt-based scoring.

        This is the main entry point for weight calculation.

        Args:
            current_time: Current time in milliseconds. If None, uses TimeUtil.now_in_millis().

        Returns:
            Tuple of (checkpoint_results, transformed_list)
            - checkpoint_results: List of (hotkey, score) tuples
            - transformed_list: List of (uid, weight) tuples
        """
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        logger.info("Computing weights for all miners")

        try:
            # Compute weights
            checkpoint_results, transformed_list = self.compute_weights_default(current_time)

            # Store results (thread-safe)
            with self._results_lock:
                self.checkpoint_results = checkpoint_results
                self.transformed_list = transformed_list

            if transformed_list:
                # Send weight setting request via RPC
                self._send_weight_request(transformed_list)
            else:
                logger.warning(
                    "No weights computed (debt ledgers may still be initializing). "
                    "Will retry later..."
                )

            return checkpoint_results, transformed_list

        except Exception as e:
            logger.error(f"Error computing weights: {e}")
            logger.error(traceback.format_exc())

            if self.slack_notifier:
                compact_trace = ErrorUtils.get_compact_stacktrace(e)
                self.slack_notifier.send_message(
                    f"Weight computation error!\n"
                    f"Error: {str(e)}\n"
                    f"Trace: {compact_trace}",
                    level="error"
                )
            raise

    def compute_weights_default(self, current_time: int) -> Tuple[List[Tuple[str, float]], List[Tuple[int, float]]]:
        """
        Compute weights for all miners using debt-based scoring.

        Args:
            current_time: Current time in milliseconds

        Returns:
            Tuple of (checkpoint_results, transformed_list)
            - checkpoint_results: List of (hotkey, score) tuples
            - transformed_list: List of (uid, weight) tuples
        """
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        # Collect metagraph hotkeys to ensure we are only setting weights for miners in the metagraph
        metagraph_hotkeys = list(self._metagraph_client.get_hotkeys())
        metagraph_hotkeys_set = set(metagraph_hotkeys)
        hotkey_to_idx = {hotkey: idx for idx, hotkey in enumerate(metagraph_hotkeys)}

        # Get all miners from all buckets
        # TODO discuss whether to include plagiarism miners in dust emissions
        buckets = [MinerBucket.CHALLENGE, MinerBucket.PROBATION, MinerBucket.MAINCOMP, MinerBucket.ENTITY]
        all_hotkeys = self._challengeperiod_client.get_hotkeys_by_bucket(buckets)

        # Filter out zombie miners (miners in buckets but not in metagraph)
        all_hotkeys_before_filter = len(all_hotkeys)
        all_hotkeys = [hk for hk in all_hotkeys if hk in metagraph_hotkeys_set]
        zombies_filtered = all_hotkeys_before_filter - len(all_hotkeys)

        if zombies_filtered > 0:
            logger.info(f"Filtered out {zombies_filtered} zombie miners (not in metagraph)")

        logger.info(
            f"Computing weights for {len(all_hotkeys)} miners: "
            f"({zombies_filtered} zombies filtered)"
        )

        # Compute weights for all miners using debt-based scoring
        checkpoint_netuid_weights, checkpoint_results = self._compute_miner_weights(
            all_hotkeys, hotkey_to_idx, current_time
        )

        if checkpoint_netuid_weights is None or len(checkpoint_netuid_weights) == 0:
            logger.info("No weights computed. Do nothing for now.")
            return [], []

        transformed_list = checkpoint_netuid_weights
        logger.info(f"transformed list: {transformed_list}")

        return checkpoint_results, transformed_list

    def _compute_miner_weights(
        self,
        hotkeys_to_compute_weights_for: List[str],
        hotkey_to_idx: dict,
        current_time: int
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[str, float]]]:
        """
        Compute weights for specified miners using debt-based scoring.

        Args:
            hotkeys_to_compute_weights_for: List of miner hotkeys
            hotkey_to_idx: Mapping of hotkey to metagraph index
            current_time: Current time in milliseconds

        Returns:
            Tuple of (netuid_weights, checkpoint_results)
        """
        if len(hotkeys_to_compute_weights_for) == 0:
            return [], []

        logger.info("Calculating new subtensor weights using debt-based scoring...")

        # Get debt ledgers for the specified miners via RPC
        all_debt_ledgers = self._debt_ledger_client.get_all_debt_ledgers()
        filtered_debt_ledgers = {
            hotkey: ledger
            for hotkey, ledger in all_debt_ledgers.items()
            if hotkey in hotkeys_to_compute_weights_for
        }
        all_emissions_ledgers = self._debt_ledger_client.get_all_emissions_ledgers()
        if len(filtered_debt_ledgers) == 0:
            total_ledgers = len(all_debt_ledgers)
            if total_ledgers == 0:
                logger.info(
                    f"No debt ledgers loaded yet. "
                    f"Requested {len(hotkeys_to_compute_weights_for)} hotkeys. "
                    f"Debt ledger daemon likely still building initial data (120s delay + build time). "
                    f"Will retry in 5 minutes."
                )
            else:
                logger.warning(
                    f"No debt ledgers found. "
                    f"Requested {len(hotkeys_to_compute_weights_for)} hotkeys, "
                    f"debt_ledger_client has {total_ledgers} ledgers loaded."
                )
            return [], []

        # Use debt-based scoring with shared metagraph
        verbose = current_time - self._last_verbose_ms >= 30 * 60 * 1000
        if verbose:
            self._last_verbose_ms = current_time

        current_time_ms = TimeUtil.now_in_millis()
        # Get current datetime
        current_dt = TimeUtil.millis_to_datetime(current_time_ms)

        if verbose:
            logger.info(
                f"Computing debt-based weights for {current_dt.strftime('%B %Y')} "
                f"({len(filtered_debt_ledgers)} miners)"
            )

        # Calculate boundaries
        # Needed payout calculation: Sum from activation through end of previous week
        # Payout activation start Nov 3, 2025 Monday 00:00 UTC
        # This allows negative PnL to carry across weeks and offset future gains
        payout_activation_start_dt = datetime(
            DebtBasedScoring.ACTIVATION_YEAR,
            DebtBasedScoring.ACTIVATION_MONTH,
            3, 0, 0, 0,
            tzinfo=timezone.utc
        ) - timedelta(days=7)
        payout_activation_start_ms = int(payout_activation_start_dt.timestamp() * 1000)
        prev_target_end_ms = TimeUtil.ms_at_start_of_week(current_time_ms)

        all_positions = self._position_client.get_positions_for_all_miners()
        all_perf_ledgers = self._perf_ledger_client.get_perf_ledgers()

        # Calculate regular hotkey payouts
        funded_hotkeys = self._challengeperiod_client.get_hotkeys_by_bucket(
            [MinerBucket.PROBATION, MinerBucket.MAINCOMP]
        )

        # Payout settlements are generated starting the first full week of each miner's debt ledger
        miner_required_payout_settlements = DebtBasedScoring.generate_payout_settlements(
            hotkeys=funded_hotkeys,
            miner_positions=all_positions,
            perf_ledgers=all_perf_ledgers,
            debt_ledgers=filtered_debt_ledgers,
            start_time_ms=payout_activation_start_ms,
            end_time_ms=prev_target_end_ms
        )
        miner_required_payouts = {
            hotkey: sum(payout.payout_penalized for payout in payouts)
            for hotkey, payouts in miner_required_payout_settlements.items()
        }
        miner_required_payouts_raw = {
            hotkey: sum(payout.payout_raw for payout in payouts)
            for hotkey, payouts in miner_required_payout_settlements.items()
        }
        miner_required_payouts_raw = {
            hotkey: sum(payout.payout_raw for payout in payouts)
            for hotkey, payouts in miner_required_payout_settlements.items()
        }

        # Emissions should be calculated starting from the full week after the first weekly settlement
        miner_emissions_cumulative = {}
        for hotkey, weekly_settlements in miner_required_payout_settlements.items():
            if not weekly_settlements:
                miner_emissions_cumulative[hotkey] = 0
                continue

            debt_ledger = all_debt_ledgers.get(hotkey)
            if not debt_ledger:
                miner_emissions_cumulative[hotkey] = 0
                continue

            emissions_start_ms = weekly_settlements[0].end_ms
            emissions_end_ms = weekly_settlements[-1].end_ms + MS_IN_WEEK
            miner_emissions_cumulative[hotkey] = sum(
                cp.chunk_emissions_usd
                for cp in debt_ledger.checkpoints
                if emissions_start_ms <= cp.timestamp_ms <= emissions_end_ms
            )

        # Use regular checkopint calculations for entity miners for frozen ledgers
        # Means we have to use the full cumulative emissions of the debt ledger
        # Debt ledger of entity miners should have full history from first funded miner's perf ledger
        entity_hotkeys = self._challengeperiod_client.get_hotkeys_by_bucket(MinerBucket.ENTITY)
        entity_miner_required_payouts = {
            hotkey: DebtBasedScoring.calculate_payout_from_checkpoints(
                all_debt_ledgers[hotkey].checkpoints if hotkey in all_debt_ledgers else [])
            for hotkey in entity_hotkeys
        }
        entity_miner_emissions_cumulative = {
            hotkey: sum(cp.chunk_emissions_usd for cp in all_emissions_ledgers[hotkey].checkpoints)
            for hotkey in entity_hotkeys if hotkey in all_emissions_ledgers
        }

        # Combine miner and entity payouts/emissions into unified dicts
        all_required_payouts = {**miner_required_payouts, **entity_miner_required_payouts}
        all_emissions_cumulative = {**miner_emissions_cumulative, **entity_miner_emissions_cumulative}

        # Calculate remaining payouts: needed minus already paid, clamped to zero
        miner_remaining_payouts_usd = {}
        for hotkey in all_required_payouts:
            payout_penalized = all_required_payouts.get(hotkey, 0.0)
            actual = all_emissions_cumulative.get(hotkey, 0.0)
            miner_remaining_payouts_usd[hotkey] = max(0.0, payout_penalized)
            # miner_remaining_payouts_usd[hotkey] = max(0.0, payout_penalized - actual)

        total_remaining_payout_usd = sum(miner_remaining_payouts_usd.values())
        total_actual_payout_usd = sum(all_emissions_cumulative.values())
        total_needed_payout_usd = total_remaining_payout_usd
        # total_needed_payout_usd = total_remaining_payout_usd + total_actual_payout_usd
        logger.info(
            f"[PAYOUT] SUMMARY ({len(miner_remaining_payouts_usd)} miners): "
            f"remaining_payouts=${total_remaining_payout_usd:.2f}, "
            f"required_payouts=${total_needed_payout_usd:.2f}, "
            f"paid_out=${total_actual_payout_usd:.2f} (accumulated until {TimeUtil.millis_to_short_date_str(prev_target_end_ms)}"
        )

        all_time_emissions = {
            hotkey: sum(cp.chunk_emissions_usd for cp in ledger.checkpoints)
            for hotkey, ledger in all_emissions_ledgers.items()
        }

        if verbose:
            _payouts_sorted = sorted(all_required_payouts.keys(), key=lambda hk: miner_remaining_payouts_usd.get(hk, 0.0), reverse=True)
            for hotkey in _payouts_sorted:
                remaining = miner_remaining_payouts_usd.get(hotkey, 0.0)
                actual = all_emissions_cumulative.get(hotkey, 0.0)
                payout_penalized = all_required_payouts.get(hotkey, 0.0)
                payout_raw = miner_required_payouts_raw.get(hotkey, 0.0)
                all_time = all_time_emissions.get(hotkey, 0.0)
                settlements = miner_required_payout_settlements.get(hotkey, [])
                if settlements:
                    first_week = TimeUtil.millis_to_datetime(settlements[0].start_ms).strftime('%Y-%m-%d')
                    last_week = TimeUtil.millis_to_datetime(settlements[-1].end_ms).strftime('%Y-%m-%d')
                    week_range = f" ({first_week} - {last_week})"
                else:
                    week_range = ""
                logger.info(
                    f"[PAYOUT] [{hotkey}] remaining=${remaining:.2f}, paid_out=${actual:.2f}, "
                    f"payout_penalized=${payout_penalized:.2f}, payout_raw=${payout_raw:.2f}{week_range}, all_time_emissions=${all_time:.2f}"
                )

        days_until_target = 7 - current_dt.weekday()
        # Estimate projected emissions in USD for weight normalization
        projected_alpha_available = DebtBasedScoring._estimate_alpha_emissions_until_target(
            metagraph_client=self._metagraph_client,
            days_until_target=days_until_target,
            verbose=verbose
        )
        projected_usd_available = DebtBasedScoring._convert_alpha_to_usd(
            alpha_amount=projected_alpha_available,
            metagraph_client=self._metagraph_client,
            verbose=verbose
        )

        if verbose:
            logger.info(
                f"[PAYOUT_DEBUG] PROJECTED EMISSIONS: {projected_alpha_available:.2f} ALPHA = "
                f"${projected_usd_available:.2f} USD over {days_until_target} days "
                f"(${projected_usd_available / days_until_target:.2f}/day)"
            )

        if total_remaining_payout_usd > 0:
            DebtBasedScoring.log_projections(self._metagraph_client, days_until_target, verbose, total_remaining_payout_usd)

        # Spread remaining payouts over days until target
        miner_daily_target_payouts_usd = {
            hotkey: remaining / days_until_target
            for hotkey, remaining in miner_remaining_payouts_usd.items()
        }
        projected_daily_usd = projected_usd_available / days_until_target

        # Normalize daily payouts against projected daily emissions to get raw weights
        raw_debt_weights = DebtBasedScoring._compute_raw_weights_from_payouts(
            miner_daily_target_payouts_usd=miner_daily_target_payouts_usd,
            projected_daily_emissions_usd=projected_daily_usd,
            verbose=verbose
        )

        # Enforce minimum dust weights per bucket for all eligible hotkeys
        miner_weights_with_minimums = DebtBasedScoring._apply_minimum_weights(
            eligible_hotkeys=hotkeys_to_compute_weights_for,
            raw_debt_weights=raw_debt_weights,
            ledger_dict=filtered_debt_ledgers,
            challengeperiod_client=self._challengeperiod_client,
            miner_account_client=self._miner_account_client,
            current_time_ms=current_time_ms,
            verbose=verbose
        )

        if verbose:
            logger.info(
                f"[PAYOUT_DEBUG] WEIGHT SUMMARY: {len(miner_weights_with_minimums)} miners, "
                f"total_remaining=${total_remaining_payout_usd:.2f}, "
                f"projected_daily_emissions=${projected_daily_usd:.2f}, "
                f"days_until_target={days_until_target}"
            )
            for hk, w in sorted(miner_weights_with_minimums.items(), key=lambda x: -x[1]):
                daily_target = miner_daily_target_payouts_usd.get(hk, 0.0)
                if daily_target > 0:
                    logger.info(
                        f"[PAYOUT_DEBUG] TOP WEIGHT [{hk}]: weight={w:.8f}, "
                        f"daily_target=${daily_target:.2f}"
                    )

        # Normalize weights; excess emissions assigned to burn address
        checkpoint_results = DebtBasedScoring._normalize_with_burn_address(
            weights=miner_weights_with_minimums,
            metagraph_client=self._metagraph_client,
            is_testnet=not self.is_mainnet,
            verbose=verbose
        )

        logger.info(f"Debt-based scoring results: [{checkpoint_results}]")

        checkpoint_netuid_weights = []
        for miner, score in checkpoint_results:
            if miner in hotkey_to_idx:
                checkpoint_netuid_weights.append((
                    hotkey_to_idx[miner],
                    score
                ))
            else:
                logger.error(f"Miner {miner} not found in the metagraph.")

        return checkpoint_netuid_weights, checkpoint_results

    def _send_weight_request(self, transformed_list: List[Tuple[int, float]]):
        """
        Send weight setting request to SubtensorOpsManager via RPC.

        Args:
            transformed_list: List of (uid, weight) tuples
        """
        try:
            uids = [x[0] for x in transformed_list]
            weights = [x[1] for x in transformed_list]

            # Send request via RPC (synchronous - get success/failure feedback)
            result = self._subtensor_ops_client.set_weights_rpc(
                uids=uids,
                weights=weights,
                version_key=self.subnet_version
            )

            if result.get('success'):
                logger.info(f"Weight request succeeded: {len(uids)} UIDs via RPC")
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"Weight request failed: {error}")

                # NOTE: Don't send Slack alert here - SubtensorOpsManager handles alerting
                # with proper benign error filtering (e.g., "too soon to commit weights").

        except Exception as e:
            logger.error(f"Error sending weight request via RPC: {e}")
            logger.error(traceback.format_exc())

            if self.slack_notifier:
                compact_trace = ErrorUtils.get_compact_stacktrace(e)
                self.slack_notifier.send_message(
                    f"Weight request RPC error!\n"
                    f"Error: {str(e)}\n"
                    f"Trace: {compact_trace}",
                    level="error"
                )

    # ==================== Getter Methods ====================

    def get_checkpoint_results(self) -> List[Tuple[str, float]]:
        """Get latest checkpoint results (thread-safe)."""
        with self._results_lock:
            return list(self.checkpoint_results)

    def get_transformed_list(self) -> List[Tuple[int, float]]:
        """Get latest transformed weight list (thread-safe)."""
        with self._results_lock:
            return list(self.transformed_list)
