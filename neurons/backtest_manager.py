"""
BacktestManager - Comprehensive backtesting framework for the Proprietary Trading Network

This module provides backtesting capabilities with multiple position data sources:

1. Test Positions: Hardcoded test data for development
2. Database Positions: Live positions from database via taoshi.ts.ptn (NEW FEATURE)
3. Disk Positions: Cached positions from local disk files (default)

Database Position Integration:
- Set use_database_positions=True to enable database position loading
- Requires taoshi.ts.ptn module and proper database configuration
- Automatically sets required environment variables
- Supports filtering by time range and miner hotkeys
- Converts database format to Position objects automatically

Usage Examples:
    # Use database positions for backtesting
    use_database_positions = True

    # Configure time range and hotkey
    start_time_ms = 1735689600000
    end_time_ms = 1736035200000
    test_single_hotkey = '5HDmzyhrEco9w6Jv8eE3hDMcXSE4AGg1MuezPR4u2covxKwZ'
"""
import copy
import logging
import os
import time

import bittensor as bt

# Set up logger for this module
logger = logging.getLogger(__name__)

# Set environment variables for database access
os.environ["TAOSHI_TS_DEPLOYMENT"] = "DEVELOPMENT"
os.environ["TAOSHI_TS_PLATFORM"] = "LOCAL"

from runnable.generate_request_minerstatistics import MinerStatisticsManager  # noqa: E402
from shared_objects.sn8_multiprocessing import get_multiprocessing_pool, get_spark_session  # noqa: E402
from shared_objects.mock_metagraph import MockMetagraph # noqa: E402
from vali_objects.utils.position_source import PositionSourceManager, PositionSource# noqa: E402
from time_util.time_util import TimeUtil  # noqa: E402
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager  # noqa: E402
from vali_objects.utils.elimination_manager import EliminationManager  # noqa: E402
from vali_objects.utils.live_price_fetcher import LivePriceFetcher  # noqa: E402
from vali_objects.utils.plagiarism_detector import PlagiarismDetector  # noqa: E402
from vali_objects.utils.position_lock import PositionLocks  # noqa: E402
from vali_objects.utils.position_manager import PositionManager  # noqa: E402
from vali_objects.utils.price_slippage_model import PriceSlippageModel  # noqa: E402
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter  # noqa: E402
from vali_objects.utils.validator_contract_manager import ValidatorContractManager  # noqa: E402
from vali_objects.utils.vali_utils import ValiUtils  # noqa: E402
from vali_objects.vali_config import ValiConfig  # noqa: E402
from vali_objects.vali_dataclasses.perf_ledger import ParallelizationMode, PerfLedgerManager, \
    TP_ID_PORTFOLIO  # noqa: E402

def initialize_components(hotkeys, parallel_mode, build_portfolio_ledgers_only):
    """
    Initialize common components for backtesting.

    Args:
        hotkeys: List of miner hotkeys or single hotkey
        parallel_mode: Parallelization mode for performance ledger
        build_portfolio_ledgers_only: Whether to build only portfolio ledgers

    Returns:
        Tuple of (mmg, elimination_manager, position_manager, perf_ledger_manager)
    """

    # Handle single hotkey or list
    if isinstance(hotkeys, str):
        hotkeys = [hotkeys]

    mmg = MockMetagraph(hotkeys=hotkeys)
    elimination_manager = EliminationManager(mmg, None, None)
    position_manager = PositionManager(metagraph=mmg, running_unit_tests=False,
                                     elimination_manager=elimination_manager)
    perf_ledger_manager = PerfLedgerManager(mmg, position_manager=position_manager,
                                          running_unit_tests=False,
                                          enable_rss=False,
                                          parallel_mode=parallel_mode,
                                          build_portfolio_ledgers_only=build_portfolio_ledgers_only)

    return mmg, elimination_manager, position_manager, perf_ledger_manager

def save_positions_to_manager(position_manager, hk_to_positions):
    """
    Save positions to the position manager.

    Args:
        position_manager: The position manager instance
        hk_to_positions: Dictionary mapping hotkeys to Position objects
    """
    position_count = 0
    for hk, positions in hk_to_positions.items():
        for p in positions:
            position_manager.save_miner_position(p)
            position_count += 1

    bt.logging.info(f"Saved {position_count} positions for {len(hk_to_positions)} miners to position manager")

class BacktestManager:

    def __init__(self, positions_at_t_f, start_time_ms, secrets, scoring_func,
                 use_slippage=None,
                 fetch_slippage_data=False, recalculate_slippage=False, rebuild_all_positions=False,
                 parallel_mode: ParallelizationMode=ParallelizationMode.PYSPARK, build_portfolio_ledgers_only=False,
                 pool_size=0, target_ledger_window_ms=ValiConfig.TARGET_LEDGER_WINDOW_MS):
        if not secrets:
            raise Exception(
                "unable to get secrets data from "
                "validation/miner_secrets.json. Please ensure it exists"
            )
        self.secrets = secrets
        self.scoring_func = scoring_func
        self.start_time_ms = start_time_ms
        self.parallel_mode = parallel_mode

        # Stop Spark session if we created it
        spark, should_close = get_spark_session(self.parallel_mode)
        pool = get_multiprocessing_pool(self.parallel_mode, pool_size)
        self.spark = spark
        self.pool = pool
        self.should_close = should_close
        self.target_ledger_window_ms = target_ledger_window_ms

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        # IMPORTANT: Only update this variable in-place. Otherwise, the reference will be lost in the helper classes.
        self.metagraph = MockMetagraph(hotkeys=list(positions_at_t_f.keys()))
        shutdown_dict = {}

        self.live_price_fetcher = LivePriceFetcher(secrets=self.secrets, disable_ws=True, is_backtesting=True)

        self.contract_manager = ValidatorContractManager(is_backtesting=True)

        self.elimination_manager = EliminationManager(self.metagraph, None,  # Set after self.pm creation
                                                      None, shutdown_dict=shutdown_dict, is_backtesting=True,
                                                      contract_manager=self.contract_manager)

        self.perf_ledger_manager = PerfLedgerManager(self.metagraph,
                                                     shutdown_dict=shutdown_dict,
                                                     live_price_fetcher=None, # Don't want SSL objects to be pickled
                                                     is_backtesting=True,
                                                     position_manager=None,
                                                     enable_rss=False,
                                                     parallel_mode=parallel_mode,
                                                     secrets=self.secrets,
                                                     use_slippage=use_slippage,
                                                     build_portfolio_ledgers_only=build_portfolio_ledgers_only,
                                                     target_ledger_window_ms=target_ledger_window_ms)


        self.position_manager = PositionManager(metagraph=self.metagraph,
                                                perf_ledger_manager=self.perf_ledger_manager,
                                                elimination_manager=self.elimination_manager,
                                                is_backtesting=True,
                                                challengeperiod_manager=None)


        self.challengeperiod_manager = ChallengePeriodManager(self.metagraph,
                                                              perf_ledger_manager=self.perf_ledger_manager,
                                                              position_manager=self.position_manager,
                                                              is_backtesting=True,
                                                              contract_manager=self.contract_manager)

        # Attach the position manager to the other objects that need it
        for idx, obj in enumerate([self.perf_ledger_manager, self.position_manager, self.elimination_manager]):
            obj.position_manager = self.position_manager

        self.position_manager.challengeperiod_manager = self.challengeperiod_manager

        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager
        self.position_manager.perf_ledger_manager = self.perf_ledger_manager

        self.weight_setter = SubtensorWeightSetter(self.metagraph, position_manager=self.position_manager, is_backtesting=True, contract_manager=self.contract_manager)
        self.position_locks = PositionLocks(hotkey_to_positions=positions_at_t_f, is_backtesting=True)
        self.plagiarism_detector = PlagiarismDetector(self.metagraph)
        self.miner_statistics_manager = MinerStatisticsManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector,
            contract_manager=self.contract_manager,
        )
        self.psm = PriceSlippageModel(self.live_price_fetcher, is_backtesting=True, fetch_slippage_data=fetch_slippage_data,
                                      recalculate_slippage=recalculate_slippage)


        #Until slippage is added to the db, this will always have to be done since positions are sometimes rebuilt and would require slippage attributes on orders and initial_entry_price calculation
        self.psm.update_historical_slippage(positions_at_t_f)

        self.init_order_queue_and_current_positions(self.start_time_ms, positions_at_t_f, rebuild_all_positions=rebuild_all_positions)


    def update_current_hk_to_positions(self, cutoff_ms):
        #cutoff_ms_formatted = TimeUtil.millis_to_formatted_date_str(cutoff_ms)
        #current_oq = [TimeUtil.millis_to_formatted_date_str(o[0].processed_ms) for o in self.order_queue]
        #print(f'current_oq {current_oq}')
        while self.order_queue and self.order_queue[-1][0].processed_ms <= cutoff_ms:
            time_formatted = TimeUtil.millis_to_formatted_date_str(self.order_queue[-1][0].processed_ms)
            order, position = self.order_queue.pop()
            existing_positions = [p for p in self.position_manager.get_positions_for_one_hotkey(position.miner_hotkey)
                                  if p.position_uuid == position.position_uuid]
            assert len(existing_positions) <= 1, f"Found multiple positions with the same UUID: {existing_positions}"
            existing_position = existing_positions[0] if existing_positions else None
            if existing_position:
                logger.debug(f'OQU: Added order to existing position ({position.position_uuid}) with tp {position.trade_pair.trade_pair_id} at {time_formatted}')
                assert all(o.order_uuid != order.order_uuid for o in existing_position.orders), \
                    f"Order {order.order_uuid} already exists in position {existing_position.position_uuid}"
                existing_position.orders.append(order)
                existing_position.rebuild_position_with_updated_orders(self.live_price_fetcher)
                self.position_manager.save_miner_position(existing_position)
            else:  # first order. position must be inserted into list
                logger.debug(f'OQU: Created new position ({position.position_uuid}) with tp {position.trade_pair.trade_pair_id} at {time_formatted} for hk {position.miner_hotkey}')
                position.orders = [order]
                position.rebuild_position_with_updated_orders(self.live_price_fetcher)
                self.position_manager.save_miner_position(position)

    def init_order_queue_and_current_positions(self, cutoff_ms, positions_at_t_f, rebuild_all_positions=False):
        self.order_queue = []  # (order, position)
        for hk, positions in positions_at_t_f.items():
            for position in positions:
                if position.orders[-1].processed_ms <= cutoff_ms:
                    if rebuild_all_positions:
                        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
                    self.position_manager.save_miner_position(position)
                    continue
                orders_to_keep = []
                for order in position.orders:
                    if order.processed_ms <= cutoff_ms:
                        orders_to_keep.append(order)
                    else:
                        self.order_queue.append((order, position))
                if orders_to_keep:
                    if len(orders_to_keep) != len(position.orders):
                        position.orders = orders_to_keep
                        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
                    self.position_manager.save_miner_position(position)

        self.order_queue.sort(key=lambda x: x[0].processed_ms, reverse=True)
        current_hk_to_positions = self.position_manager.get_positions_for_all_miners()
        logger.debug(f'Order queue size: {len(self.order_queue)},'
              f' Current positions n hotkeys: {len(current_hk_to_positions)},'
              f' Current positions n total: {sum(len(v) for v in current_hk_to_positions.values())}')

    def update(self, current_time_ms:int, run_challenge=True, run_elimination=True):
        self.update_current_hk_to_positions(current_time_ms)

        if self.parallel_mode == ParallelizationMode.SERIAL:
            self.perf_ledger_manager.update(t_ms=current_time_ms)
        else:
            existing_perf_ledgers = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
            # Get positions and existing ledgers
            hotkey_to_positions, _ = self.perf_ledger_manager.get_positions_perf_ledger()

            # Run the parallel update
            updated_perf_ledgers = self.perf_ledger_manager.update_perf_ledgers_parallel(self.spark, self.pool,
                 hotkey_to_positions, existing_perf_ledgers, parallel_mode=self.parallel_mode, now_ms=current_time_ms, is_backtesting=True)

            #PerfLedgerManager.print_bundles(updated_perf_ledgers)
        if run_challenge:
            self.challengeperiod_manager.refresh(current_time=current_time_ms)
        else:
            self.challengeperiod_manager.add_all_miners_to_success(current_time_ms=current_time_ms, run_elimination=run_elimination)
        if run_elimination:
            self.elimination_manager.process_eliminations(self.position_locks)
        self.weight_setter.set_weights(current_time=current_time_ms)

    def validate_last_update_ms(self, prev_end_time_ms):
        perf_ledger_bundles = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
        for hk, bundles in perf_ledger_bundles.items():
            if prev_end_time_ms:
                for tp_id, b in bundles.items():
                    assert b.last_update_ms == prev_end_time_ms, (f"Ledger for {hk} in {tp_id} was not updated. "
                      f"last_update_ms={b.last_update_ms}, expected={prev_end_time_ms}, delta={prev_end_time_ms - b.last_update_ms}")

    def debug_print_ledgers(self, perf_ledger_bundles):
        for hk, v in perf_ledger_bundles.items():
            for tp_id, bundle in v.items():
                if tp_id != TP_ID_PORTFOLIO:
                    continue
                PerfLedgerManager.print_bundle(hk, v)



if __name__ == '__main__':
    bt.logging.enable_info()
    # ============= CONFIGURATION FLAGS =============
    use_test_positions = False         # Use hardcoded test positions
    use_database_positions = True     # NEW: Use positions from database via taoshi.ts.ptn
    run_challenge = False              # Run challenge period logic
    run_elimination = False            # Run elimination logic
    use_slippage = None              # Apply slippage modeling
    crypto_only = True              # Only include crypto trade pairs
    build_portfolio_ledgers_only = True  # Whether to build only the portfolio ledgers or per trade pair
    parallel_mode = ParallelizationMode.SERIAL  # 1 for pyspark, 2 for multiprocessing

    # NOTE: Only one of use_test_positions, use_database_positions, or default (disk) should be True
    # - use_test_positions=True: Uses hardcoded test data
    # - use_database_positions=True: Loads positions from database (requires taoshi.ts.ptn)
    # - Both False: Uses positions from disk (default behavior)

    # Validate configuration
    if use_test_positions and use_database_positions:
        raise ValueError("Cannot use both test positions and database positions. Choose one.")

    start_time_ms = 1740842786000
    end_time_ms = 1757517988000
    test_single_hotkey ='5D4gJ9QfbcMg338813wz3MKuRofTKfE6zR3iPaGHaWEnNKoo'

    # Determine position source
    if use_test_positions:
        position_source = PositionSource.TEST
    elif use_database_positions:
        position_source = PositionSource.DATABASE
    else:
        position_source = PositionSource.DISK

    # Create position source manager
    position_source_manager = PositionSourceManager(position_source)

    # Load positions based on source
    if position_source == PositionSource.DISK:
        # For disk-based positions, use existing logic
        # Initialize components with specified hotkey
        mmg, elimination_manager, position_manager, perf_ledger_manager = initialize_components(
            test_single_hotkey, parallel_mode, build_portfolio_ledgers_only)

        # Get positions from disk via perf ledger manager
        hk_to_positions, _ = perf_ledger_manager.get_positions_perf_ledger(testing_one_hotkey=test_single_hotkey)
    else:
        # For database/test positions, use position source manager
        hk_to_positions = position_source_manager.load_positions(
            end_time_ms=end_time_ms,
            hotkeys=[test_single_hotkey] if test_single_hotkey and position_source == PositionSource.DATABASE else None
        )

        # For test positions, update time range based on loaded data
        if position_source == PositionSource.TEST and hk_to_positions:
            # Calculate time range from test data
            all_order_times = []
            for positions in hk_to_positions.values():
                for pos in positions:
                    all_order_times.extend([order.processed_ms for order in pos.orders])
            if all_order_times:
                start_time_ms = min(all_order_times)
                end_time_ms = max(all_order_times) + 1

        # Initialize components with loaded hotkeys
        hotkeys_list = list(hk_to_positions.keys()) if hk_to_positions else [test_single_hotkey]
        mmg, elimination_manager, position_manager, perf_ledger_manager = initialize_components(
            hotkeys_list, parallel_mode, build_portfolio_ledgers_only)

        # Save loaded positions to position manager
        for hk, positions in hk_to_positions.items():
            if crypto_only:
                crypto_positions = [p for p in positions if p.trade_pair.is_crypto]
                hk_to_positions[hk] = crypto_positions
        save_positions_to_manager(position_manager, hk_to_positions)


    t0 = time.time()

    secrets = ValiUtils.get_secrets()  # {'polygon_apikey': '123', 'tiingo_apikey': '456'}
    btm = BacktestManager(hk_to_positions, start_time_ms, secrets, None,
                          use_slippage=use_slippage, fetch_slippage_data=False, recalculate_slippage=False,
                          parallel_mode=parallel_mode,
                          build_portfolio_ledgers_only=build_portfolio_ledgers_only)

    perf_ledger_bundles = {}
    interval_ms = 1000 * 60 * 60 * 24
    prev_end_time_ms = None
    for t_ms in range(start_time_ms, end_time_ms, interval_ms):
        btm.validate_last_update_ms(prev_end_time_ms)
        btm.update(t_ms, run_challenge=run_challenge, run_elimination=run_elimination)
        perf_ledger_bundles = btm.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
        #hk_to_perf_ledger_tps = {}
        #for k, v in perf_ledger_bundles.items():
        #    hk_to_perf_ledger_tps[k] = list(v.keys())
        #print('hk_to_perf_ledger_tps', hk_to_perf_ledger_tps)
        #print('formatted weights', btm.weight_setter.checkpoint_results)
        prev_end_time_ms = t_ms
    #btm.debug_print_ledgers(perf_ledger_bundles)
    btm.perf_ledger_manager.debug_pl_plot(test_single_hotkey)

    tf = time.time()
    bt.logging.success(f'Finished backtesting in {tf - t0} seconds')
