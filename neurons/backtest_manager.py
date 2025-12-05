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
import logging
import os
import time

import bittensor as bt

# Set up logger for this module
logger = logging.getLogger(__name__)

# Set environment variables for database access
os.environ["TAOSHI_TS_DEPLOYMENT"] = "DEVELOPMENT"
os.environ["TAOSHI_TS_PLATFORM"] = "LOCAL"

from shared_objects.sn8_multiprocessing import get_multiprocessing_pool, get_spark_session  # noqa: E402
from shared_objects.rpc.common_data_server import CommonDataServer  # noqa: E402
from shared_objects.rpc.metagraph_server import MetagraphServer  # noqa: E402
from shared_objects.rpc.metagraph_client import MetagraphClient
from shared_objects.rpc.port_manager import PortManager  # noqa: E402
from shared_objects.rpc.rpc_client_base import RPCClientBase  # noqa: E402
from shared_objects.rpc.rpc_server_base import RPCServerBase  # noqa: E402
from vali_objects.position_management.position_utils.position_source import PositionSourceManager, PositionSource  # noqa: E402
from time_util.time_util import TimeUtil  # noqa: E402
from vali_objects.utils.asset_selection.asset_selection_server import AssetSelectionServer  # noqa: E402
from vali_objects.challenge_period import ChallengePeriodServer  # noqa: E402
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient  # noqa: E402
from vali_objects.contract.contract_server import ContractServer  # noqa: E402
from vali_objects.utils.elimination.elimination_server import EliminationServer  # noqa: E402
from vali_objects.utils.elimination.elimination_client import EliminationClient  # noqa: E402
from vali_objects.utils.limit_order.limit_order_server import LimitOrderServer  # noqa: E402
from vali_objects.price_fetcher import LivePriceFetcherServer, LivePriceFetcherClient  # noqa: E402
from vali_objects.plagiarism.plagiarism_server import PlagiarismServer  # noqa: E402
from vali_objects.plagiarism.plagiarism_client import PlagiarismClient
from shared_objects.locks.position_lock import PositionLocks  # noqa: E402
from shared_objects.locks.position_lock_server import PositionLockServer  # noqa: E402
from vali_objects.position_management.position_manager_server import PositionManagerServer  # noqa: E402
from vali_objects.position_management.position_manager_client import PositionManagerClient  # noqa: E402
from vali_objects.utils.price_slippage_model import PriceSlippageModel  # noqa: E402
from vali_objects.utils.vali_utils import ValiUtils  # noqa: E402
from vali_objects.vali_config import ValiConfig  # noqa: E402
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import ParallelizationMode, TP_ID_PORTFOLIO  # noqa: E402
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_server import PerfLedgerServer  # noqa: E402
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient


def initialize_components(hotkeys, parallel_mode, build_portfolio_ledgers_only, running_unit_tests=False, skip_port_kill=False):
    """
    Initialize common components for backtesting using client/server architecture.

    Args:
        hotkeys: List of miner hotkeys or single hotkey
        parallel_mode: Parallelization mode for performance ledger
        build_portfolio_ledgers_only: Whether to build only portfolio ledgers
        running_unit_tests: Whether running in unit test mode
        skip_port_kill: Skip killing RPC ports (useful when caller already did it)

    Returns:
        Tuple of (metagraph_client, elimination_client, position_client, perf_ledger_client, server_handles)
    """
    # Handle single hotkey or list
    if isinstance(hotkeys, str):
        hotkeys = [hotkeys]

    # Kill any existing RPC ports (unless caller already did it)
    if not skip_port_kill:
        PortManager.force_kill_all_rpc_ports()

    metagraph_handle = MetagraphServer(start_server=True, running_unit_tests=running_unit_tests)
    common_data_server = CommonDataServer(start_server=True)

    # Start infrastructure servers
    secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)

    # Start LivePriceFetcherServer FIRST to give it maximum time to initialize
    live_price_server = LivePriceFetcherServer(
        secrets=secrets, disable_ws=True, start_server=True, running_unit_tests=running_unit_tests, is_backtesting=True
    )

    # Start other infrastructure servers
    asset_selection_server = AssetSelectionServer(start_server=True, running_unit_tests=running_unit_tests)

    # Start metagraph server and client
    metagraph_client = MetagraphClient()
    metagraph_client.set_hotkeys(hotkeys)

    # Start other servers
    position_lock_server = PositionLockServer(start_server=True, running_unit_tests=running_unit_tests)
    contract_handle = ContractServer(start_server=True, running_unit_tests=running_unit_tests, is_backtesting=True)
    perf_ledger_handle = PerfLedgerServer(
        start_server=True,
        running_unit_tests=running_unit_tests,
        is_backtesting=True,
        parallel_mode=parallel_mode,
        build_portfolio_ledgers_only=build_portfolio_ledgers_only
    )
    perf_ledger_client = PerfLedgerClient()

    challenge_period_handle = ChallengePeriodServer.spawn_process(
        running_unit_tests=running_unit_tests,
        start_daemon=False,
        is_backtesting=True
    )
    challenge_period_client = ChallengePeriodClient()

    elimination_handle = EliminationServer.spawn_process(
        running_unit_tests=running_unit_tests,
        is_backtesting=True
    )
    elimination_client = EliminationClient()

    limit_order_server = LimitOrderServer(running_unit_tests=running_unit_tests)

    # Start position server after challengeperiod server (dependency)
    position_server_handle = PositionManagerServer.spawn_process(
        running_unit_tests=running_unit_tests,
        is_backtesting=True
    )
    position_client = PositionManagerClient()

    plagiarism_handle = PlagiarismServer.spawn_process(running_unit_tests=running_unit_tests)
    plagiarism_client = PlagiarismClient()

    # Store server handles for cleanup
    server_handles = {
        'live_price_server': live_price_server,
        'common_data_server': common_data_server,
        'asset_selection_server': asset_selection_server,
        'metagraph_handle': metagraph_handle,
        'position_lock_server': position_lock_server,
        'contract_handle': contract_handle,
        'perf_ledger_handle': perf_ledger_handle,
        'challenge_period_handle': challenge_period_handle,
        'elimination_handle': elimination_handle,
        'limit_order_server': limit_order_server,
        'position_server_handle': position_server_handle,
        'plagiarism_handle': plagiarism_handle
    }

    return (metagraph_client, elimination_client, position_client, perf_ledger_client,
            challenge_period_client, plagiarism_client, server_handles)

def save_positions_to_manager(position_client, hk_to_positions):
    """
    Save positions to the position manager via client.

    Args:
        position_client: The position manager client instance
        hk_to_positions: Dictionary mapping hotkeys to Position objects
    """
    position_count = 0
    for hk, positions in hk_to_positions.items():
        for p in positions:
            position_client.save_miner_position(p)
            position_count += 1

    bt.logging.info(f"Saved {position_count} positions for {len(hk_to_positions)} miners to position manager")

class BacktestManager:

    def __init__(self, positions_at_t_f, start_time_ms, secrets, scoring_func,
                 use_slippage=None,
                 fetch_slippage_data=False, recalculate_slippage=False, rebuild_all_positions=False,
                 parallel_mode: ParallelizationMode=ParallelizationMode.PYSPARK, build_portfolio_ledgers_only=False,
                 pool_size=0, target_ledger_window_ms=ValiConfig.TARGET_LEDGER_WINDOW_MS,
                 running_unit_tests=False, skip_port_kill=False):
        if not secrets:
            raise Exception(
                "unable to get secrets data from "
                "validation/miner_secrets.json. Please ensure it exists"
            )
        self.secrets = secrets
        self.scoring_func = scoring_func
        self.start_time_ms = start_time_ms
        self.parallel_mode = parallel_mode
        self.running_unit_tests = running_unit_tests

        # Stop Spark session if we created it
        spark, should_close = get_spark_session(self.parallel_mode)
        pool = get_multiprocessing_pool(self.parallel_mode, pool_size)
        self.spark = spark
        self.pool = pool
        self.should_close = should_close
        self.target_ledger_window_ms = target_ledger_window_ms

        # Get hotkeys and initialize server/client architecture
        hotkeys = list(positions_at_t_f.keys())

        # Initialize all servers and clients
        (self.metagraph_client, self.elimination_client, self.position_client,
         self.perf_ledger_client, self.challenge_period_client, self.plagiarism_client,
         self.server_handles) = initialize_components(
            hotkeys, parallel_mode, build_portfolio_ledgers_only,
            running_unit_tests=running_unit_tests, skip_port_kill=skip_port_kill
        )

        # Create LivePriceFetcher client for local use
        self.live_price_client = LivePriceFetcherClient()

        # Initialize position locks (still needed for legacy compatibility)
        self.position_locks = PositionLocks(hotkey_to_positions=positions_at_t_f, is_backtesting=True)

        # Create price slippage model with client
        self.psm = PriceSlippageModel(
            self.live_price_client,
            is_backtesting=True,
            fetch_slippage_data=fetch_slippage_data,
            recalculate_slippage=recalculate_slippage
        )

        # Until slippage is added to the db, this will always have to be done since positions are
        # sometimes rebuilt and would require slippage attributes on orders and initial_entry_price calculation
        self.psm.update_historical_slippage(positions_at_t_f)

        # Initialize order queue and current positions
        self.init_order_queue_and_current_positions(
            self.start_time_ms, positions_at_t_f, rebuild_all_positions=rebuild_all_positions
        )


    def update_current_hk_to_positions(self, cutoff_ms):
        #cutoff_ms_formatted = TimeUtil.millis_to_formatted_date_str(cutoff_ms)
        #current_oq = [TimeUtil.millis_to_formatted_date_str(o[0].processed_ms) for o in self.order_queue]
        #print(f'current_oq {current_oq}')
        while self.order_queue and self.order_queue[-1][0].processed_ms <= cutoff_ms:
            time_formatted = TimeUtil.millis_to_formatted_date_str(self.order_queue[-1][0].processed_ms)
            order, position = self.order_queue.pop()
            existing_positions = [p for p in self.position_client.get_positions_for_one_hotkey(position.miner_hotkey)
                                  if p.position_uuid == position.position_uuid]
            assert len(existing_positions) <= 1, f"Found multiple positions with the same UUID: {existing_positions}"
            existing_position = existing_positions[0] if existing_positions else None
            if existing_position:
                logger.debug(f'OQU: Added order to existing position ({position.position_uuid}) with tp {position.trade_pair.trade_pair_id} at {time_formatted}')
                assert all(o.order_uuid != order.order_uuid for o in existing_position.orders), \
                    f"Order {order.order_uuid} already exists in position {existing_position.position_uuid}"
                existing_position.orders.append(order)
                existing_position.rebuild_position_with_updated_orders(self.live_price_client)
                self.position_client.save_miner_position(existing_position)
            else:  # first order. position must be inserted into list
                logger.debug(f'OQU: Created new position ({position.position_uuid}) with tp {position.trade_pair.trade_pair_id} at {time_formatted} for hk {position.miner_hotkey}')
                position.orders = [order]
                position.rebuild_position_with_updated_orders(self.live_price_client)
                self.position_client.save_miner_position(position)

    def init_order_queue_and_current_positions(self, cutoff_ms, positions_at_t_f, rebuild_all_positions=False):
        self.order_queue = []  # (order, position)
        for hk, positions in positions_at_t_f.items():
            for position in positions:
                if position.orders[-1].processed_ms <= cutoff_ms:
                    if rebuild_all_positions:
                        position.rebuild_position_with_updated_orders(self.live_price_client)
                    self.position_client.save_miner_position(position)
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
                        position.rebuild_position_with_updated_orders(self.live_price_client)
                    self.position_client.save_miner_position(position)

        self.order_queue.sort(key=lambda x: x[0].processed_ms, reverse=True)
        current_hk_to_positions = self.position_client.get_positions_for_all_miners()
        logger.debug(f'Order queue size: {len(self.order_queue)},'
              f' Current positions n hotkeys: {len(current_hk_to_positions)},'
              f' Current positions n total: {sum(len(v) for v in current_hk_to_positions.values())}')

    def update(self, current_time_ms:int, run_challenge=True, run_elimination=True):
        self.update_current_hk_to_positions(current_time_ms)

        # Update performance ledgers via client
        self.perf_ledger_client.update(t_ms=current_time_ms)

        # Update challenge period via client
        if run_challenge:
            self.challenge_period_client.refresh(current_time=current_time_ms)
        else:
            self.challenge_period_client.add_all_miners_to_success(
                current_time_ms=current_time_ms, run_elimination=run_elimination
            )

        # Process eliminations via client
        if run_elimination:
            self.elimination_client.process_eliminations()

        # Note: Weight setter is not part of the client/server architecture yet
        # This would need to be refactored separately if needed

    def validate_last_update_ms(self, prev_end_time_ms):
        perf_ledger_bundles = self.perf_ledger_client.get_perf_ledgers(portfolio_only=False)
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
                self.perf_ledger_client.print_bundle(hk, v)

    def cleanup(self):
        """Cleanup method to shutdown all servers and disconnect clients."""
        RPCClientBase.disconnect_all()
        RPCServerBase.shutdown_all(force_kill_ports=True)



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
    # NOTE: BacktestManager will initialize servers, so we don't call initialize_components here
    if position_source == PositionSource.DISK:
        # For disk-based positions, use a placeholder - positions will be loaded after BacktestManager init
        hk_to_positions = {test_single_hotkey: []}
    else:
        # For database/test positions, use position source manager (doesn't need servers)
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

        # Filter to crypto only if needed
        if crypto_only:
            for hk in list(hk_to_positions.keys()):
                crypto_positions = [p for p in hk_to_positions[hk] if p.trade_pair.is_crypto]
                hk_to_positions[hk] = crypto_positions

    t0 = time.time()

    secrets = ValiUtils.get_secrets()  # {'polygon_apikey': '123', 'tiingo_apikey': '456'}
    btm = BacktestManager(hk_to_positions, start_time_ms, secrets, None,
                          use_slippage=use_slippage, fetch_slippage_data=False, recalculate_slippage=False,
                          parallel_mode=parallel_mode,
                          build_portfolio_ledgers_only=build_portfolio_ledgers_only)

    # For disk-based positions, load after BacktestManager has initialized servers
    if position_source == PositionSource.DISK:
        hk_to_positions, _ = btm.perf_ledger_client.get_positions_perf_ledger(testing_one_hotkey=test_single_hotkey)
        # Save loaded positions
        if crypto_only:
            for hk in list(hk_to_positions.keys()):
                crypto_positions = [p for p in hk_to_positions[hk] if p.trade_pair.is_crypto]
                hk_to_positions[hk] = crypto_positions
        save_positions_to_manager(btm.position_client, hk_to_positions)
        # Re-initialize order queue with loaded positions
        btm.init_order_queue_and_current_positions(start_time_ms, hk_to_positions, rebuild_all_positions=False)

    perf_ledger_bundles = {}
    interval_ms = 1000 * 60 * 60 * 24
    prev_end_time_ms = None

    try:
        for t_ms in range(start_time_ms, end_time_ms, interval_ms):
            btm.validate_last_update_ms(prev_end_time_ms)
            btm.update(t_ms, run_challenge=run_challenge, run_elimination=run_elimination)
            perf_ledger_bundles = btm.perf_ledger_client.get_perf_ledgers(portfolio_only=False)
            #hk_to_perf_ledger_tps = {}
            #for k, v in perf_ledger_bundles.items():
            #    hk_to_perf_ledger_tps[k] = list(v.keys())
            #print('hk_to_perf_ledger_tps', hk_to_perf_ledger_tps)
            prev_end_time_ms = t_ms
        #btm.debug_print_ledgers(perf_ledger_bundles)
        btm.perf_ledger_client.debug_pl_plot(test_single_hotkey)

        tf = time.time()
        bt.logging.success(f'Finished backtesting in {tf - t0} seconds')

    finally:
        # Cleanup servers and clients
        bt.logging.info("Cleaning up servers and clients...")
        btm.cleanup()
