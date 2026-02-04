import datetime
import json
import os
import time
import traceback
from collections import defaultdict, Counter
from copy import deepcopy
from typing import List

import bittensor as bt
from setproctitle import setproctitle

from data_generator.polygon_data_service import PolygonDataService
from shared_objects.cache_controller import CacheController
from shared_objects.rpc.common_data_client import CommonDataClient
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from shared_objects.sn8_multiprocessing import ParallelizationMode
from time_util.time_util import UnifiedMarketCalendar, TimeUtil, timeme
from vali_objects.enums.misc import ShortcutReason, TradePairReturnStatus
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import RPCConnectionMode, ValiConfig
from vali_objects.vali_dataclasses import position as position_file
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import FeeCache, PerfLedger, TP_ID_PORTFOLIO
from vali_objects.vali_dataclasses.position import Position


class PerfLedgerManager(CacheController):
    def __init__(self, connection_mode: "RPCConnectionMode" = RPCConnectionMode.RPC,
                 use_slippage=None, running_unit_tests=False,
                 enable_rss=True, is_backtesting=False, parallel_mode=ParallelizationMode.SERIAL, secrets=None,
                 build_portfolio_ledgers_only=False, target_ledger_window_ms=ValiConfig.TARGET_LEDGER_WINDOW_MS):
        super().__init__(running_unit_tests=running_unit_tests, is_backtesting=is_backtesting, connection_mode=connection_mode)



        self.connection_mode = connection_mode
        self.perf_ledger_hks_to_invalidate = {}


        super().__init__(running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        self.running_unit_tests = running_unit_tests
        self.enable_rss = enable_rss
        self.parallel_mode = parallel_mode
        self.use_slippage = use_slippage
        position_file.ALWAYS_USE_SLIPPAGE = use_slippage
        self.build_portfolio_ledgers_only = build_portfolio_ledgers_only


        self.pl_elimination_rows = []
        self.hotkey_to_perf_bundle = {}
        self.running_unit_tests = running_unit_tests

        self._position_manager_client = PositionManagerClient(
            connect_immediately=False
        )

        # Create own ContractClient (forward compatibility - no parameter passing)
        # Lazy import to avoid circular dependency:
        # elimination_server -> contract_server -> ledger_utils -> perf_ledger -> contract_server
        from vali_objects.contract.contract_client import ContractClient
        self._contract_client = ContractClient(
            port=ValiConfig.RPC_CONTRACTMANAGER_PORT,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        # Create own EliminationClient (forward compatibility - no parameter passing)
        self._elimination_client = EliminationClient(
            port=ValiConfig.RPC_ELIMINATION_PORT,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        self._common_data_client = CommonDataClient(
            connect_immediately=False,  # Lazy connect on first use
            connection_mode=connection_mode
        )

        self.cached_miner_account_sizes = {}  # Deepcopy of contract_manager.miner_account_sizes
        self.cache_last_refreshed_date = None  # 'YYYY-MM-DD' format, refresh daily
        self.pds = None  # Load it later once the process starts so ipc works.

        # Create own LivePriceFetcherClient (forward compatibility - no parameter passing)
        self._live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests)

        # Every update, pick a hotkey to rebuild in case polygon 1s candle data changed.
        self.trade_pair_to_price_info = {'second':{}, 'minute':{}}
        self.trade_pair_to_position_ret = {}

        self.random_security_screenings = set()
        self.market_calendar = UnifiedMarketCalendar()
        self.n_api_calls = 0
        self.POLYGON_MAX_CANDLE_LIMIT = 49999
        self.UPDATE_LOOKBACK_MS = 600000  # 10 minutes ago. Want to give Polygon time to create candles on the backend.
        self.UPDATE_LOOKBACK_S = self.UPDATE_LOOKBACK_MS // 1000
        self.now_ms = 0  # The largest timestamp we want to buffer candles for. time.time() - UPDATE_LOOKBACK_S
        #self.base_dd_stats = {'worst_dd':1.0, 'last_dd':0, 'mrpv':1.0, 'n_closed_pos':0, 'n_checks':0, 'current_portfolio_return': 1.0}
        #self.hk_to_dd_stats = defaultdict(lambda: deepcopy(self.base_dd_stats))
        self.candidate_pl_elimination_rows = []
        self.hk_to_last_order_processed_ms = {}
        self.mode_to_n_updates = {}
        self.update_to_n_open_positions = {}
        self.position_uuid_to_cache = defaultdict(FeeCache)
        self.target_ledger_window_ms = target_ledger_window_ms
        bt.logging.info(f"Running performance ledger manager with mode {self.parallel_mode.name}")
        if self.is_backtesting or self.parallel_mode != ParallelizationMode.SERIAL:
            bt.logging.debug("[PERF_LEDGER] Skipping disk load (backtesting or non-SERIAL mode)")
        else:
            bt.logging.info("[PERF_LEDGER] Loading initial performance ledgers from disk...")
            initial_perf_ledgers = self.get_perf_ledgers(from_disk=True, portfolio_only=False)
            bt.logging.success(f"[PERF_LEDGER] Loaded {len(initial_perf_ledgers)} performance ledger bundles from disk")
            for k, v in initial_perf_ledgers.items():
                self.hotkey_to_perf_bundle[k] = v
            # ipc list does not update the object without using __setitem__
            temp = self.get_perf_ledger_eliminations(first_fetch=True)
            bt.logging.info(f"[PERF_LEDGER] Loaded {len(temp)} pl elimination rows from disk")
            self.pl_elimination_rows.extend(temp)
            for i, x in enumerate(temp):
                self.pl_elimination_rows[i] = x

        if secrets:
            self.secrets = secrets
        else:
            self.secrets = ValiUtils.get_secrets(running_unit_tests=self.running_unit_tests)

    @property
    def contract_manager(self):
        """Backward compatibility property that maps to _contract_client."""
        return self._contract_client

    def clear_all_ledger_data(self):
        # Clear in-memory and on-disk ledgers. Only for unit tests.
        assert self.running_unit_tests, 'this is only valid for unit tests'
        self.hotkey_to_perf_bundle.clear()
        self.clear_perf_ledgers_from_disk()  # Also clears in-memory
        self.pl_elimination_rows.clear()
        self.clear_perf_ledger_eliminations_from_disk()
        self.perf_ledger_hks_to_invalidate.clear()  # Clear invalidation list for test isolation

    def re_init_perf_ledger_data(self):
        """
        Reinitialize perf ledger data by reloading from disk.
        This is useful after clear_all_ledger_data() + save_perf_ledgers() to ensure
        all internal state (caches, counters, etc.) is properly reset.
        Only for unit tests.
        """
        assert self.running_unit_tests, 'this is only valid for unit tests'

        # Reload ledgers from disk into memory cache
        ledgers_from_disk = self.get_perf_ledgers(portfolio_only=False, from_disk=True)
        self.hotkey_to_perf_bundle.clear()
        for hk, bundle in ledgers_from_disk.items():
            self.hotkey_to_perf_bundle[hk] = bundle

        bt.logging.info(f"Reinitialized {len(self.hotkey_to_perf_bundle)} perf ledgers from disk")

    def __getstate__(self):
        """
        Custom pickle method to exclude unpicklable attributes.

        When using multiprocessing, the PerfLedgerManager needs to be pickled,
        but clients contain RPC connections with threading locks that cannot be pickled.
        These clients are not needed during parallel processing (positions are passed directly),
        so we exclude them from pickling.
        """
        state = self.__dict__.copy()
        # Remove unpicklable attributes that aren't needed during parallel processing
        state['_metagraph_client'] = None
        state['_position_manager_client'] = None
        state['_contract_client'] = None
        state['_elimination_client'] = None
        state['_common_data_client'] = None
        state['_live_price_client'] = None
        state['pds'] = None
        return state

    def __setstate__(self, state):
        """Restore state from pickle, with excluded attributes set to None."""
        self.__dict__.update(state)

    # ==================== Client Properties (forward compatibility) ====================

    @property
    def metagraph(self):
        """Get metagraph client (forward compatibility - created internally)."""
        return self._metagraph_client

    @metagraph.setter
    def metagraph(self, value):
        """
        Setter to handle base class CacheController assignment.
        We ignore the value since we use our internal _metagraph_client instead.
        """
        # CacheController.__init__ sets self.metagraph = metagraph (usually None)
        # We ignore this since we use _metagraph_client created in __init__
        pass

    def _is_shutdown(self):
        """Check if shutdown has been signaled via ShutdownCoordinator."""
        return ShutdownCoordinator.is_shutdown()

    @staticmethod
    def print_bundles(ans: dict[str, dict[str, PerfLedger]]):
        for hk, bundle in ans.items():
            print(f'-----------({hk})-----------')
            PerfLedgerManager.print_bundle(hk, bundle)

    @staticmethod
    def print_bundle(hk:str, bundle: dict[str, PerfLedger]):
        bt.logging.success(f'Hotkey: {hk}. Max return: {bundle[TP_ID_PORTFOLIO].max_return}. Initialization time: {TimeUtil.millis_to_timestamp(bundle[TP_ID_PORTFOLIO].initialization_time_ms)}')
        for tp_id, pl in sorted(bundle.items(), key=lambda x: 1 if x[0] == TP_ID_PORTFOLIO else ord(x[0][0]) / 27):
            bt.logging.info(f'  --{tp_id}-- ')
            for idx, x in enumerate(pl.cps):
                last_update_formatted = TimeUtil.millis_to_timestamp(x.last_update_ms)
                if 1:#idx == 0 or idx == len(pl.cps) - 1:
                    bt.logging.info(f'    {idx} {last_update_formatted} {x}')
            bt.logging.info(tp_id, f'max_perf_ledger_return: {pl.max_return}')

    def _is_v1_perf_ledger(self, ledger_value):
        if self.build_portfolio_ledgers_only:
            return False
        ans = False

        # Handle both PerfLedger objects (from pickle) and dicts (from JSON)
        if isinstance(ledger_value, PerfLedger):
            # Direct PerfLedger object = V1 format
            ans = True
        elif isinstance(ledger_value, dict):
            # Dict could be V1 ledger dict or V1 with single portfolio key
            if 'initialization_time_ms' in ledger_value:
                ans = True
            # "Faked" v2 ledger (single portfolio key)
            elif TP_ID_PORTFOLIO in ledger_value and len(ledger_value) == 1:
                ans = True

        return ans


    def get_perf_ledgers(self, portfolio_only=True, from_disk=False) -> dict[str, dict[str, PerfLedger]] | dict[str, PerfLedger]:
        ret = {}
        if from_disk:
            compressed_json_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)

            # Try compressed JSON first (primary format)
            if os.path.exists(compressed_json_path):
                data = ValiBkpUtils.read_compressed_json(compressed_json_path)
            # Fall back to migration from .pkl or .json
            elif ValiBkpUtils.migrate_perf_ledgers_to_compressed(self.running_unit_tests):
                # Migration succeeded, now read the newly created .json.gz file
                data = ValiBkpUtils.read_compressed_json(compressed_json_path)
            else:
                # No file exists to migrate
                return ret

            for hk, possible_bundles in data.items():
                if self._is_v1_perf_ledger(possible_bundles):
                    if portfolio_only:
                        # V1 dict format - convert to PerfLedger
                        ret[hk] = PerfLedger.from_dict(possible_bundles)
                    else:
                        # Incompatible but we can fake it for now
                        if 'initialization_time_ms' in possible_bundles:
                            # V1 dict format - convert to PerfLedger
                            ledger = PerfLedger.from_dict(possible_bundles)
                            ret[hk] = {TP_ID_PORTFOLIO: ledger}
                        elif TP_ID_PORTFOLIO in possible_bundles:
                            # Faked V2 dict format with single portfolio key
                            ledger = PerfLedger.from_dict(possible_bundles[TP_ID_PORTFOLIO])
                            ret[hk] = {TP_ID_PORTFOLIO: ledger}

                else:
                    if portfolio_only:
                        ret[hk] = PerfLedger.from_dict(possible_bundles[TP_ID_PORTFOLIO])
                    else:
                        # Convert all dicts to PerfLedger objects
                        ret[hk] = {k: PerfLedger.from_dict(v) for k, v in possible_bundles.items()}
            return ret

        # Everything here is in v2 format
        if portfolio_only:
            dat = dict(self.hotkey_to_perf_bundle)
            return {hk: bundle[TP_ID_PORTFOLIO] for hk, bundle in dat.items()}
        else:
            return dict(self.hotkey_to_perf_bundle)

    def get_returns(self, hotkey: str) -> float | None:
        """
        Calculate returns for a specific hotkey's portfolio.

        Args:
            hotkey: Miner hotkey

        Returns:
            Returns as float (e.g., 0.08 for 8%), or None if no data exists
        """
        if hotkey not in self.hotkey_to_perf_bundle:
            return None

        bundle = self.hotkey_to_perf_bundle[hotkey]
        if TP_ID_PORTFOLIO not in bundle:
            return None

        portfolio_ledger = bundle[TP_ID_PORTFOLIO]
        if not portfolio_ledger.cps:
            return None

        # TODO: updated perf ledger logic and compute returns from flow adjusted equity curve of checkpoints
        # Returns = current portfolio value - initial value (1.0)
        returns = sum([cp.realized_pnl for cp in portfolio_ledger.cps]) + portfolio_ledger.cps[-1].unrealized_pnl   # portfolio_ledger.cps[-1].prev_portfolio_ret - 1.0
        return returns

    def filtered_ledger_for_scoring(
            self,
            portfolio_only: bool = False,
            hotkeys: List[str] = None
    ) -> dict[str, dict[str, PerfLedger]] | dict[str, PerfLedger]:
        """
        Filter the ledger for a set of hotkeys.
        """

        if hotkeys is None:
            hotkeys = self._metagraph_client.get_hotkeys()

        # Build filtered ledger for all miners with positions
        filtered_ledger = {}

        if portfolio_only:
            # When portfolio_only=True, get_perf_ledgers() returns dict[hotkey, PerfLedger]
            for hotkey, perf_ledger in self.get_perf_ledgers(portfolio_only=True).items():
                if hotkey not in hotkeys:
                    continue

                if hotkey in self.perf_ledger_hks_to_invalidate:
                    bt.logging.warning(f"Skipping hotkey {hotkey} in filtered_ledger_for_scoring due to invalidation.")
                    continue

                if perf_ledger is None or len(perf_ledger.cps) == 0:
                    continue

                filtered_ledger[hotkey] = perf_ledger
        else:
            # When portfolio_only=False, get_perf_ledgers() returns dict[hotkey, dict[asset_class, PerfLedger]]
            for hotkey, asset_ledgers in self.get_perf_ledgers(portfolio_only=False).items():
                if hotkey not in hotkeys:
                    continue

                if hotkey in self.perf_ledger_hks_to_invalidate:
                    bt.logging.warning(f"Skipping hotkey {hotkey} in filtered_ledger_for_scoring due to invalidation.")
                    continue

                if asset_ledgers is None:
                    continue

                # Ensure we have the portfolio ledger with checkpoints
                miner_overall_ledger = asset_ledgers.get(TP_ID_PORTFOLIO, PerfLedger())
                if len(miner_overall_ledger.cps) == 0:
                    continue

                filtered_ledger[hotkey] = asset_ledgers

        return filtered_ledger

    def clear_perf_ledgers_from_disk(self):
        assert self.running_unit_tests, 'this is only valid for unit tests'
        self.hotkey_to_perf_bundle = {}

        # Clear compressed JSON file (current format)
        json_gz_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)
        if os.path.exists(json_gz_path):
            ValiBkpUtils.write_compressed_json(json_gz_path, {})

        # Clear .pkl file if it exists (from bug)
        pkl_path = ValiBkpUtils.get_perf_ledgers_path_pkl(self.running_unit_tests)
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        # Clear legacy uncompressed JSON file if it exists
        legacy_json_path = ValiBkpUtils.get_perf_ledgers_path_legacy(self.running_unit_tests)
        if os.path.exists(legacy_json_path):
            os.remove(legacy_json_path)

        for k in list(self.hotkey_to_perf_bundle.keys()):
            del self.hotkey_to_perf_bundle[k]

    def clear_perf_ledger_eliminations_from_disk(self):
        assert self.running_unit_tests, 'this is only valid for unit tests'
        self.pl_elimination_rows = []
        file_path = ValiBkpUtils.get_perf_ledger_eliminations_dir(running_unit_tests=self.running_unit_tests)
        if os.path.exists(file_path):
            ValiBkpUtils.write_file(file_path, [])

    @staticmethod
    def clear_perf_ledgers_from_disk_autosync(hotkeys:list):
        compressed_json_path = ValiBkpUtils.get_perf_ledgers_path(running_unit_tests=False)

        filtered_data = {}

        # Try compressed JSON first (primary format)
        if os.path.exists(compressed_json_path):
            existing_data = ValiBkpUtils.read_compressed_json(compressed_json_path)
        # Fall back to migration from .pkl or .json
        elif ValiBkpUtils.migrate_perf_ledgers_to_compressed(running_unit_tests=False):
            # Migration succeeded, now read the newly created .json.gz file
            existing_data = ValiBkpUtils.read_compressed_json(compressed_json_path)
        else:
            # No file exists to migrate
            existing_data = {}

        for hk, bundles in existing_data.items():
            if hk in hotkeys:
                # Convert PerfLedger objects to dicts if needed (defensive check)
                if isinstance(bundles, dict):
                    filtered_data[hk] = {}
                    for trade_pair_id, ledger in bundles.items():
                        if isinstance(ledger, PerfLedger):
                            filtered_data[hk][trade_pair_id] = ledger.to_dict()
                        else:
                            filtered_data[hk][trade_pair_id] = ledger
                elif isinstance(bundles, PerfLedger):
                    # V1 format - single PerfLedger (portfolio only)
                    filtered_data[hk] = bundles.to_dict()
                else:
                    # Already dict
                    filtered_data[hk] = bundles

        # Always write to compressed JSON format
        ValiBkpUtils.write_compressed_json(compressed_json_path, filtered_data)


    def run_update_loop(self):
        setproctitle(f"vali_{self.__class__.__name__}")
        bt.logging.enable_info()
        while not self._is_shutdown():
            try:
                if self.refresh_allowed(ValiConfig.PERF_LEDGER_REFRESH_TIME_MS):
                    self.update()
                    self.set_last_update_time(skip_message=True)

            except Exception as e:
                # Handle exceptions or log errors
                bt.logging.error(f"Error during perf ledger update: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc())
                time.sleep(30)
            time.sleep(1)

    def get_historical_position(self, position: Position, timestamp_ms: int):
        hk = position.miner_hotkey  # noqa: F841

        new_orders = []
        position_at_start_timestamp = deepcopy(position)
        position_at_end_timestamp = deepcopy(position)
        for o in position.orders:
            if o.processed_ms <= timestamp_ms:
                new_orders.append(o)

        position_at_start_timestamp.orders = new_orders[:-1]
        position_at_start_timestamp.rebuild_position_with_updated_orders(self._live_price_client)
        position_at_end_timestamp.orders = new_orders
        position_at_end_timestamp.rebuild_position_with_updated_orders(self._live_price_client)
        # Handle position that was forced closed due to realtime data (liquidated)
        if len(new_orders) == len(position.orders) and position.return_at_close == 0:
            position_at_end_timestamp.return_at_close = 0
            position_at_end_timestamp.close_out_position(position.close_ms)

        return position_at_start_timestamp, position_at_end_timestamp

    def generate_order_timeline(self, positions: list[Position], now_ms: int, hk: str) -> tuple[list[tuple], int]:
        # order to understand timestamps needing checking, position to understand returns per timestamp (will be adjusted)
        # (order, position)
        time_sorted_orders = []
        last_event_time_ms = 0

        for p in positions:
            last_event_time_ms = max(p.orders[-1].processed_ms, last_event_time_ms)

            if p.is_closed_position and len(p.orders) < 2:
                bt.logging.warning(f"perf ledger generate_order_timeline. Skipping closed position for hk {hk} with < 2 orders: {p}")
                continue
            for o in p.orders:
                if o.processed_ms <= now_ms:
                    time_sorted_orders.append((o, p))
        # sort
        time_sorted_orders.sort(key=lambda x: x[0].processed_ms)
        return time_sorted_orders, last_event_time_ms


    def _can_shortcut(self, tp_to_historical_positions: dict[str: Position], end_time_ms: int,
                      tp_id_to_realtime_position_to_pop: dict[str, Position], start_time_ms: int, perf_ledger_bundle: dict[str, PerfLedger]) -> (
            ShortcutReason, dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float],
            TradePairReturnStatus):

        tp_to_return = {}
        tp_to_realized_pnl = {}
        tp_to_unrealized_pnl = {}
        tp_to_spread_fee = {}
        tp_to_carry_fee = {}
        for k in list(tp_to_historical_positions.keys()) + [TP_ID_PORTFOLIO]:
            tp_to_return[k] = 1.0
            tp_to_realized_pnl[k] = 0.0
            tp_to_unrealized_pnl[k] = 0.0
            tp_to_spread_fee[k] = 1.0
            tp_to_carry_fee[k] = 1.0

        n_open_positions = 0
        portfolio_pl = perf_ledger_bundle[TP_ID_PORTFOLIO]
        # Set now_ms to end_time_ms when backtesting for historical perf ledger generation
        if self.is_backtesting:
            ledger_cutoff_ms = end_time_ms
        else:
            ledger_cutoff_ms = TimeUtil.now_in_millis() - portfolio_pl.target_ledger_window_ms

        n_positions = 0
        n_closed_positions = 0
        n_positions_newly_opened = 0
        any_open : TradePairReturnStatus = TradePairReturnStatus.TP_MARKET_NOT_OPEN

        for tp_id, historical_positions in tp_to_historical_positions.items():
            for i, historical_position in enumerate(historical_positions):
                n_positions += 1
                if len(historical_position.orders) == 0:
                    n_positions_newly_opened += 1
                elif historical_position.is_open_position:
                    n_open_positions += 1
                else:
                    n_closed_positions += 1
                if tp_id in tp_id_to_realtime_position_to_pop and i == len(historical_positions) - 1:
                    historical_position = tp_id_to_realtime_position_to_pop[tp_id]

                for k in [TP_ID_PORTFOLIO, tp_id]:
                    csf, _ = self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position, end_time_ms)
                    tp_to_spread_fee[k] *= csf
                    ccf, _ = self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(end_time_ms, historical_position)
                    tp_to_carry_fee[k] *= ccf
                    tp_to_return[k] *= historical_position.return_at_close
                    tp_to_realized_pnl[k] += historical_position.realized_pnl
                    tp_to_unrealized_pnl[k] += historical_position.unrealized_pnl

        for tp_id in list(tp_to_historical_positions.keys()) + [TP_ID_PORTFOLIO]:
            pl = perf_ledger_bundle.get(tp_id)
            # Check if we need to update _prev (compare just the return value, not the tuple)
            current_tuple = self.trade_pair_to_position_ret.get(tp_id)
            if pl and current_tuple and current_tuple[0] != tp_to_return[tp_id]:
                self.trade_pair_to_position_ret[tp_id + '_prev'] = current_tuple
            # Count positions for this trade pair
            position_count = len(tp_to_historical_positions.get(tp_id, [])) if tp_id != TP_ID_PORTFOLIO else n_positions
            self.trade_pair_to_position_ret[tp_id] = (tp_to_return[tp_id], position_count)

        assert tp_to_carry_fee[TP_ID_PORTFOLIO] > 0, (tp_to_carry_fee[TP_ID_PORTFOLIO], tp_to_spread_fee[TP_ID_PORTFOLIO])

        reason = ''
        ans = ShortcutReason.NO_SHORTCUT
        # When building from orders, we will always have at least one open position. When opening a position after a
        # period of all closed positions, we can shortcut by identifying that the new position is the only open position
        # and all other positions are closed. The time before this period, we have only closed positions.
        # Alternatively, we can be attempting to build the ledger after all orders have been accounted for. In this
        # case, we simply need to check if all positions are closed.
        if n_open_positions == 0:
            #if n_positions_newly_opened not in (0, 1):
            #    for tp, historical_positions in tp_to_historical_positions.items():
            #        for i, historical_position in enumerate(historical_positions):
            #            if len(historical_position.orders) == 0:
            #                print(historical_position)

            #    raise Exception(f'n_positions_newly_opened should be 0 or 1 but got {n_positions_newly_opened}')

            reason += 'No open positions. '
            ans = ShortcutReason.NO_OPEN_POSITIONS
            any_open = TradePairReturnStatus.TP_NO_OPEN_POSITIONS

        # This window would be dropped anyway
        if (end_time_ms < ledger_cutoff_ms):
            reason += 'Ledger cutoff. '
            ans = ShortcutReason.OUTSIDE_WINDOW

        if 0 and ans != ShortcutReason.NO_SHORTCUT:
            bt.logging.info('---------------------------------------------------------------------')
            for tp_id, historical_positions in tp_to_historical_positions.items():
                positions = []
                for i, historical_position in enumerate(historical_positions):
                    if tp_id in tp_id_to_realtime_position_to_pop and i == len(
                            historical_positions) - 1:
                        historical_position = tp_id_to_realtime_position_to_pop[tp_id]
                        foo = True
                    else:
                        foo = False
                    positions.append((historical_position.position_uuid, [x.price for x in historical_position.orders],
                                      historical_position.return_at_close, foo, historical_position.is_open_position))
                bt.logging.info(f'{tp_id}: {positions}')

            final_cp = None
            if perf_ledger_bundle and TP_ID_PORTFOLIO in perf_ledger_bundle and perf_ledger_bundle[TP_ID_PORTFOLIO].cps:
                final_cp = perf_ledger_bundle[TP_ID_PORTFOLIO].cps[-1]
            n_orders_per_position_counter = Counter()
            for tp_id, historical_positions in tp_to_historical_positions.items():
                for historical_position in historical_positions:
                    n_orders_per_position_counter[len(historical_position.orders)] += 1
            bt.logging.info(f' Skipping ({reason}) with n_positions: {n_positions} n_open_positions: {n_open_positions} n_closed_positions: '
                  f'{n_closed_positions}, n_positions_newly_opened: {n_positions_newly_opened}, '
                  f'start_time_ms: {TimeUtil.millis_to_formatted_date_str(start_time_ms)} ({start_time_ms}) , '
                  f'end_time_ms: {TimeUtil.millis_to_formatted_date_str(end_time_ms)} ({end_time_ms}) , '
                  f'portfolio_value: {tp_to_return[TP_ID_PORTFOLIO]} '
                  f'ledger_cutoff_ms: {TimeUtil.millis_to_formatted_date_str(ledger_cutoff_ms)}, '
                  f'trade_pair_to_position_ret: {self.trade_pair_to_position_ret} '
                  f'n_orders_per_position_counter: {n_orders_per_position_counter} '
                  f'final portfolio cp {final_cp}')
            bt.logging.info('---------------------------------------------------------------------')

        return ans, tp_to_return, tp_to_realized_pnl, tp_to_unrealized_pnl, tp_to_spread_fee, tp_to_carry_fee, any_open


    def new_window_intersects_old_window(self, start_time_ms, end_time_ms, existing_lb_ms, existing_ub_ms):
        # Check if new window intersects with the old window
        # An intersection occurs if the start of the new window is before the end of the old window,
        # and the end of the new window is after the start of the old window
        return start_time_ms <= existing_ub_ms and end_time_ms >= existing_lb_ms

    def align_t_ms_to_mode(self, t_ms, mode):
        if mode == 'second':
            return t_ms - (t_ms % 1000)
        elif mode == 'minute':
            return t_ms - (t_ms % 60000)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def refresh_price_info(self, t_ms, end_time_ms, tp, mode):
        def populate_price_info(pi, price_info_raw):
            for a in price_info_raw:
                pi[a.timestamp] = a.close

        min_candles_per_request = 3600 if mode == 'second' else 1440
        existing_lb_ms = None
        existing_ub_ms = None
        existing_window_ms = None
        if tp.trade_pair_id in self.trade_pair_to_price_info[mode]:
            price_info = self.trade_pair_to_price_info[mode][tp.trade_pair_id]
            existing_ub_ms = price_info['ub_ms']
            existing_lb_ms = price_info['lb_ms']
            existing_window_ms = existing_ub_ms - existing_lb_ms
            if existing_lb_ms <= t_ms <= existing_ub_ms:  # No refresh needed
                return
        #else:
        #    print('11111', tp.trade_pair, trade_pair_to_price_info.keys())

        start_time_ms = t_ms
        requested_milliseconds = end_time_ms - start_time_ms
        n_candles_requested = requested_milliseconds // 1000 if mode == 'second' else requested_milliseconds // 60000
        if n_candles_requested > self.POLYGON_MAX_CANDLE_LIMIT:  # Polygon limit
            end_time_ms = start_time_ms + self.POLYGON_MAX_CANDLE_LIMIT * 1000 if mode == 'second' else start_time_ms + self.POLYGON_MAX_CANDLE_LIMIT * 60000
        elif n_candles_requested < min_candles_per_request:  # Get a batch of candles to minimize number of fetches
            offset = min_candles_per_request * 1000 if mode == 'second' else min_candles_per_request * 60000
            end_time_ms = start_time_ms + offset

        end_time_ms = min(int(self.now_ms), end_time_ms)  # Don't fetch candles beyond check time or will fill in null.

        #t0 = time.time()
        #print(f"Starting #{requested_seconds} candle fetch for {tp.trade_pair}")
        if self.pds is None:
            if self.running_unit_tests:
                # Use LivePriceFetcherClient in test mode to support RPC test data injection
                # (e.g., via set_test_candle_data() RPC method)
                price_info_raw = self._live_price_client.unified_candle_fetcher(
                    trade_pair=tp, start_date=start_time_ms, order_date=end_time_ms, timespan=mode)
                self.n_api_calls += 1
                #print(f'Fetched candles for tp {tp.trade_pair} for window {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to {TimeUtil.millis_to_formatted_date_str(end_time_ms)}')
                #print(f'Got {len(price_info)} candles after request of {requested_seconds} candles for tp {tp.trade_pair} in {time.time() - t0}s')
            else:
                # Production path - create real price fetcher
                self.pds = PolygonDataService(api_key=self.secrets["polygon_apikey"], disable_ws=True, is_backtesting=self.is_backtesting, running_unit_tests=self.running_unit_tests)
                price_info_raw = self.pds.unified_candle_fetcher(
                    trade_pair=tp, start_timestamp_ms=start_time_ms, end_timestamp_ms=end_time_ms, timespan=mode)
                self.tp_to_mfs.update(self.pds.tp_to_mfs)
                self.n_api_calls += 1
        else:
            # Use existing PDS instance
            price_info_raw = self.pds.unified_candle_fetcher(
                trade_pair=tp, start_timestamp_ms=start_time_ms, end_timestamp_ms=end_time_ms, timespan=mode)
            self.tp_to_mfs.update(self.pds.tp_to_mfs)
            self.n_api_calls += 1
        #print(f'Fetched candles for tp {tp.trade_pair} for window {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to {TimeUtil.millis_to_formatted_date_str(end_time_ms)}')
        #print(f'Got {len(price_info)} candles after request of {requested_seconds} candles for tp {tp.trade_pair} in {time.time() - t0}s')

        #assert lb_ms >= start_time_ms, (lb_ms, start_time_ms)
        #assert ub_ms <= end_time_ms, (ub_ms, end_time_ms)
        # Can we build on top of existing data or should we wipe?
        perform_wipe = True
        if tp.trade_pair_id in self.trade_pair_to_price_info[mode]:
            new_window_size_ms = end_time_ms - start_time_ms
            candidate_window_size = new_window_size_ms + existing_window_ms
            candidate_n_candles_in_memory = candidate_window_size // 1000 if mode == 'second' else candidate_window_size // 60000
            if candidate_n_candles_in_memory < self.POLYGON_MAX_CANDLE_LIMIT and \
                    self.new_window_intersects_old_window(start_time_ms, end_time_ms, existing_lb_ms, existing_ub_ms):
                perform_wipe = False


        if perform_wipe:
            price_info = {}
            populate_price_info(price_info, price_info_raw)
            self.trade_pair_to_price_info[mode][tp.trade_pair_id] = price_info
            self.trade_pair_to_price_info[mode][tp.trade_pair_id]['lb_ms'] = start_time_ms
            self.trade_pair_to_price_info[mode][tp.trade_pair_id]['ub_ms'] = end_time_ms
        else:
            self.trade_pair_to_price_info[mode][tp.trade_pair_id]['ub_ms'] = max(existing_ub_ms, end_time_ms)
            self.trade_pair_to_price_info[mode][tp.trade_pair_id]['lb_ms'] = min(existing_lb_ms, start_time_ms)
            populate_price_info(self.trade_pair_to_price_info[mode][tp.trade_pair_id], price_info_raw)

        #print(f'Fetched {requested_seconds} s of candles for tp {tp.trade_pair} in {time.time() - t0}s')
        #print('22222', tp.trade_pair, trade_pair_to_price_info.keys())

    def positions_to_portfolio_return(self, possible_tp_ids, tp_to_historical_positions_dense: dict[str: Position],
                                      t_ms, mode, end_time_ms, tp_to_initial_return, tp_to_initial_realized_pnl, tp_to_initial_unrealized_pnl,
                                      tp_to_initial_spread_fee, tp_to_initial_carry_fee, portfolio_pl,
                                      tp_id_to_realtime_position_to_pop: dict[str, Position] = None):
        # Answers "What is the portfolio return at this time t_ms?"
        tp_to_any_open : dict[str, TradePairReturnStatus] = {x: TradePairReturnStatus.TP_NO_OPEN_POSITIONS for x in possible_tp_ids}
        tp_to_return = tp_to_initial_return.copy()
        tp_to_realized_pnl = tp_to_initial_realized_pnl.copy()
        tp_to_unrealized_pnl = tp_to_initial_unrealized_pnl.copy()
        tp_to_spread_fee = tp_to_initial_spread_fee.copy()
        tp_to_carry_fee = tp_to_initial_carry_fee.copy()
        t_ms = self.align_t_ms_to_mode(t_ms, mode)
        for tp_id, historical_positions in tp_to_historical_positions_dense.items():
            assert len(historical_positions) < 2, ('maybe a recently opened position?', historical_positions)

            # Determine which IDs to update for this trade pair
            tp_ids_to_build = [TP_ID_PORTFOLIO] if self.build_portfolio_ledgers_only else [tp_id, TP_ID_PORTFOLIO]

            for historical_position in historical_positions:
                if self._is_shutdown():
                    return tp_to_return, tp_to_realized_pnl, tp_to_unrealized_pnl, tp_to_any_open, tp_to_spread_fee, tp_to_carry_fee

                # Calculate fees for this position
                position_spread_fee, psf_updated = self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position, t_ms)
                position_carry_fee, pcf_updated = self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(t_ms, historical_position)

                # Apply fees to the appropriate IDs
                for x in tp_ids_to_build:
                    tp_to_spread_fee[x] *= position_spread_fee
                    tp_to_carry_fee[x] *= position_carry_fee

                # Check if market is open
                if not self.market_calendar.is_market_open(historical_position.trade_pair, t_ms):
                    # Check if this position closes at or before this checkpoint time
                    closed_position = None
                    if (tp_id_to_realtime_position_to_pop and
                        tp_id in tp_id_to_realtime_position_to_pop):
                        candidate = tp_id_to_realtime_position_to_pop[tp_id]
                        if (candidate.position_uuid == historical_position.position_uuid and
                            candidate.is_closed_position and
                            candidate.close_ms is not None and
                            t_ms >= candidate.close_ms):
                            closed_position = candidate

                    for x in tp_ids_to_build:
                        tp_to_return[x] *= historical_position.return_at_close

                        # Always add realized PnL (cumulative from all orders)
                        tp_to_realized_pnl[x] += historical_position.realized_pnl

                        # Only add unrealized PnL if position is NOT closed at this checkpoint
                        if not closed_position:
                            # Position is still open - add unrealized PnL
                            tp_to_unrealized_pnl[x] += historical_position.unrealized_pnl
                        # else: Position is closed - unrealized PnL is 0, don't add

                        # Only update to MARKET_NOT_OPEN if we haven't seen any open positions yet
                        if tp_to_any_open[x] == TradePairReturnStatus.TP_NO_OPEN_POSITIONS:
                            tp_to_any_open[x] = TradePairReturnStatus.TP_MARKET_NOT_OPEN
                    continue

                # Market is open - fetch price info
                self.refresh_price_info(t_ms, end_time_ms, historical_position.trade_pair, mode)
                price_at_t_ms = self.trade_pair_to_price_info[mode][tp_id].get(t_ms)

                # Determine if price changed
                price_changed = False
                if price_at_t_ms is not None:
                    # Only check portfolio ledger - no fallback
                    prev_price = None
                    prev_t_ms = None
                    if tp_id in portfolio_pl.last_known_prices:
                        prev_price, prev_t_ms = portfolio_pl.last_known_prices[tp_id]

                    price_changed = price_at_t_ms != prev_price

                # Update position returns based on current price
                if historical_position.is_open_position and price_at_t_ms is not None:
                    # Always update returns for open positions when we have a price
                    # This ensures returns are always current and prevents stale values
                    historical_position.set_returns(price_at_t_ms, self._live_price_client, time_ms=t_ms, total_fees=position_spread_fee * position_carry_fee)
                else:
                    # Closed positions or no price available - just update fees
                    historical_position.set_returns_with_updated_fees(position_spread_fee * position_carry_fee, t_ms, self._live_price_client)

                # Track last known prices for portfolio ledger to maintain continuity
                if price_at_t_ms is not None:
                    # Store previous price before updating
                    if tp_id in portfolio_pl.last_known_prices:
                        prev_price, prev_ts = portfolio_pl.last_known_prices[tp_id]
                        # Store previous price and timestamp in the same dict with _prev suffix
                        portfolio_pl.last_known_prices[tp_id + '_prev'] = (prev_price, prev_ts)
                    portfolio_pl.last_known_prices[tp_id] = (price_at_t_ms, t_ms)

                # Update returns for all relevant IDs
                # Check if this position closes at or before this checkpoint time
                closed_position = None
                if (tp_id_to_realtime_position_to_pop and
                    tp_id in tp_id_to_realtime_position_to_pop):
                    candidate = tp_id_to_realtime_position_to_pop[tp_id]
                    if (candidate.position_uuid == historical_position.position_uuid and
                        candidate.is_closed_position and
                        candidate.close_ms is not None and
                        t_ms >= candidate.close_ms):
                        closed_position = candidate

                for x in tp_ids_to_build:
                    tp_to_return[x] *= historical_position.return_at_close

                    # Always add realized PnL (cumulative from all orders)
                    tp_to_realized_pnl[x] += historical_position.realized_pnl

                    # Only add unrealized PnL if position is NOT closed at this checkpoint
                    if not closed_position:
                        # Position is still open - add unrealized PnL
                        tp_to_unrealized_pnl[x] += historical_position.unrealized_pnl
                    # else: Position is closed - unrealized PnL is 0, don't add

                # Update status based on price change
                # Use the enum ordering to ensure we keep the highest priority status
                if price_changed:
                    for x in tp_ids_to_build:
                        if tp_to_any_open[x] < TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE:
                            tp_to_any_open[x] = TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE
                else:
                    # Market is open but no price change
                    for x in tp_ids_to_build:
                        if tp_to_any_open[x] < TradePairReturnStatus.TP_MARKET_OPEN_NO_PRICE_CHANGE:
                            tp_to_any_open[x] = TradePairReturnStatus.TP_MARKET_OPEN_NO_PRICE_CHANGE

        for tp_id in list(tp_to_historical_positions_dense.keys()) + [TP_ID_PORTFOLIO]:
            if tp_id in self.trade_pair_to_position_ret:
                self.trade_pair_to_position_ret[tp_id + '_prev'] = self.trade_pair_to_position_ret[tp_id]
            if tp_id in tp_to_return:
                # Count positions for this trade pair
                position_count = len(tp_to_historical_positions_dense.get(tp_id, [])) if tp_id != TP_ID_PORTFOLIO else sum(len(positions) for positions in tp_to_historical_positions_dense.values())
                self.trade_pair_to_position_ret[tp_id] = (tp_to_return[tp_id], position_count)
        return tp_to_return, tp_to_realized_pnl, tp_to_unrealized_pnl, tp_to_any_open, tp_to_spread_fee, tp_to_carry_fee


    def check_liquidated(self, miner_hotkey, portfolio_return, t_ms, tp_to_historical_positions, perf_ledger_bundle):
        if portfolio_return == 0:
            bt.logging.warning(f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_ms}. Eliminating miner.")
            portfolio_pl = perf_ledger_bundle[TP_ID_PORTFOLIO]
            elimination_row = self.generate_elimination_row(miner_hotkey, 0.0, "LIQUIDATED", t_ms=t_ms, price_info=portfolio_pl.last_known_prices, return_info={'dd_stats': {}, 'returns': self.trade_pair_to_position_ret})
            self.candidate_pl_elimination_rows.append(elimination_row)
            self.candidate_pl_elimination_rows[-1] = elimination_row  # Trigger the update on the multiprocessing Manager
            #self.hk_to_dd_stats[miner_hotkey]['eliminated'] = True
            for _, v in tp_to_historical_positions.items():
                for pos in v:
                    print(
                        f"    time {TimeUtil.millis_to_formatted_date_str(t_ms)} hk {miner_hotkey[-5:]} {pos.trade_pair.trade_pair_id} return {pos.current_return} return_at_close {pos.return_at_close} closed@{'NA' if pos.is_open_position else TimeUtil.millis_to_formatted_date_str(pos.orders[-1].processed_ms)}")
            return True
        return False


    def cleanup_closed_position_prices(self, portfolio_pl: PerfLedger, open_positions_tp_ids: set):
        """
        Remove price tracking for trade pairs that no longer have open positions.

        Args:
            portfolio_pl: The portfolio performance ledger containing last_known_prices
            open_positions_tp_ids: Set of trade pair IDs that currently have open positions
        """
        if not portfolio_pl.last_known_prices:
            return

        # Find and remove trade pairs that are no longer open
        # Skip _prev keys in the check since they're not in open_positions_tp_ids
        tp_ids_to_remove = [
            tp_id for tp_id in portfolio_pl.last_known_prices
            if not tp_id.endswith('_prev') and tp_id not in open_positions_tp_ids
        ]

        for tp_id in tp_ids_to_remove:
            del portfolio_pl.last_known_prices[tp_id]
            # Also clean up the prev price tracking
            prev_price_key = tp_id + '_prev'
            if prev_price_key in portfolio_pl.last_known_prices:
                del portfolio_pl.last_known_prices[prev_price_key]
            bt.logging.debug(f"Removed closed position {tp_id} from price tracking")

    def condense_positions(self, tp_ids_to_build, tp_to_historical_positions: dict[str: Position]) -> (float, float, float, dict[str: Position]):
        tp_to_initial_return = {x: 1.0 for x in tp_ids_to_build}
        tp_to_initial_realized_pnl = {x: 0.0 for x in tp_ids_to_build}
        tp_to_initial_spread_fee = {x: 1.0 for x in tp_ids_to_build}
        tp_to_initial_carry_fee = {x: 1.0 for x in tp_ids_to_build}
        tp_to_historical_positions_dense = {}
        open_positions_tp_ids = set()
        for tp_id, historical_positions in tp_to_historical_positions.items():
            dense_positions = []
            for historical_position in historical_positions:
                if historical_position.is_closed_position:
                    tp_ids_to_build = [TP_ID_PORTFOLIO] if self.build_portfolio_ledgers_only else [tp_id, TP_ID_PORTFOLIO]
                    for x in tp_ids_to_build:
                        tp_to_initial_return[x] *= historical_position.return_at_close
                        tp_to_initial_realized_pnl[x] += historical_position.realized_pnl
                        tp_to_initial_spread_fee[x] *= self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position, historical_position.orders[-1].processed_ms)[0]
                        tp_to_initial_carry_fee[x] *= self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(historical_position.orders[-1].processed_ms, historical_position)[0]
                elif len(historical_position.orders) == 0:
                    continue
                else:
                    dense_positions.append(historical_position)
                    assert historical_position.trade_pair.trade_pair_id not in open_positions_tp_ids
                    open_positions_tp_ids.add(historical_position.trade_pair.trade_pair_id)
            if dense_positions:
                tp_to_historical_positions_dense[tp_id] = dense_positions
        return tp_to_initial_return, tp_to_initial_realized_pnl, tp_to_initial_spread_fee, tp_to_initial_carry_fee, tp_to_historical_positions_dense, open_positions_tp_ids

    def get_default_update_mode(self, start_time_ms, end_time_ms, n_open_positions):
        # Minutely mode requires only one open position since intervals are represented with 2 prices.
        if False:#n_open_positions > 1:
            default_mode = 'second'
        # Default mode becomes minute if there are at least 30 minutes between start and end time
        elif (end_time_ms - start_time_ms) > 1.8e+6:
            default_mode = 'minute'
        else:
            default_mode = 'second'
        return default_mode

    def get_current_update_mode(self, default_mode, start_time_ms, end_time_ms, accumulated_time_ms):
        mode = default_mode
        if default_mode == 'minute':
            candidate_t_ms = int((start_time_ms + accumulated_time_ms) // 1000) * 1000
            ms_from_minute_boundary = candidate_t_ms % 60000
            if ms_from_minute_boundary != 0:
                mode = 'second'
            elif end_time_ms - candidate_t_ms <= 60000:  # one min or less from end. go fine grained
                mode = 'second'
        return mode

    def get_bypass_values_if_applicable(self, perf_ledger: PerfLedger, tp_id: str, any_open: TradePairReturnStatus,
                                        calculated_return: float,
                                        calculated_spread_fee: float, calculated_carry_fee: float,
                                        tp_id_to_realtime_position_to_pop: dict[str, Position]) -> tuple[float, float, float]:
        """
        Returns values to pass to update_pl. Uses previous checkpoint values if in bypass mode
        (all positions closed + no position just closed) to prevent floating point drift.

        Args:
            perf_ledger: The performance ledger being updated
            tp_id: Trade pair ID for debugging
            any_open: Status indicating if any positions are open
            calculated_return: Freshly calculated portfolio return
            calculated_spread_fee: Freshly calculated spread fee
            calculated_carry_fee: Freshly calculated carry fee
            tp_id_to_realtime_position_to_pop: Trade pair ID of the position that just closed (realtime_position_to_pop)

        Returns:
            Tuple of (return, spread_fee, carry_fee) to pass to update_pl
        """
        # Check if we should use bypass (all closed + no position just closed + same trade pair if applicable)
        position_just_closed = any(pos is not None and not pos.is_open_position for pos in tp_id_to_realtime_position_to_pop.values())
        prev_cp = perf_ledger.cps[-1]
        use_bypass = (any_open == TradePairReturnStatus.TP_NO_OPEN_POSITIONS and
                      not position_just_closed and
                      (not tp_id_to_realtime_position_to_pop or tp_id in tp_id_to_realtime_position_to_pop) and
                      len(perf_ledger.cps) > 0 and
                      calculated_spread_fee == prev_cp.prev_portfolio_spread_fee and
                      calculated_carry_fee == prev_cp.prev_portfolio_carry_fee
                      )

        if use_bypass:
            # Reuse previous checkpoint's exact values to avoid floating point drift
            return_val = prev_cp.prev_portfolio_ret
        else:
            return_val = calculated_return

        return (return_val, calculated_spread_fee, calculated_carry_fee)

    def debug_significant_portfolio_drop(self, mode, portfolio_return, perf_ledger_bundle, t_ms, miner_hotkey,
                                         tp_to_historical_positions, open_positions_tp_ids, start_time_ms, end_time_ms):
        portfolio_pl = perf_ledger_bundle[TP_ID_PORTFOLIO]
        ratio_drop = portfolio_return / portfolio_pl.cps[-1].prev_portfolio_ret
        pl_last_update_time = TimeUtil.millis_to_formatted_date_str(portfolio_pl.last_update_ms)
        if mode == 'second' and ratio_drop < 0.98 or mode == 'minute' and ratio_drop < .90:
            time_since_last_update = t_ms - perf_ledger_bundle[TP_ID_PORTFOLIO].cps[-1].last_update_ms
            time_formatted = TimeUtil.millis_to_formatted_date_str(t_ms)
            start_formatted = TimeUtil.millis_to_formatted_date_str(start_time_ms)
            end_formatted = TimeUtil.millis_to_formatted_date_str(end_time_ms)
            # Format trade_pair_to_position_ret for display
            formatted_returns = {}
            for k, v in self.trade_pair_to_position_ret.items():
                if isinstance(v, tuple):
                    formatted_returns[k] = f"{v[0]:.6f} (n={v[1]})"
                else:
                    formatted_returns[k] = v

            print(
                f'perf ledger (pl_last_update_time {pl_last_update_time}) for hk {miner_hotkey} significant return drop on {time_formatted} from '
                f'{portfolio_pl.cps[-1].prev_portfolio_ret} to {portfolio_return} over'
                f' {time_since_last_update} ms ({t_ms}) when building up to {start_formatted} and {end_formatted} with open_positions_tp_ids {open_positions_tp_ids}, ',
                f'trade_pair_to_position_ret {formatted_returns}, mode {mode} ')
            for tp_id, historical_positions in tp_to_historical_positions.items():
                positions = []
                for historical_position in historical_positions:
                    if historical_position.is_open_position and len(historical_position.orders):
                        tpo_ms = [TimeUtil.millis_to_formatted_date_str(x.processed_ms) for x in historical_position.orders]
                        positions.append({'position_uuid': historical_position.position_uuid,
                                         'net_leverage': historical_position.net_leverage,
                                         'price_per_order': [x.price for x in historical_position.orders],
                                         'return_at_close': historical_position.return_at_close,
                                         'time_per_order_ms': tpo_ms})
                if positions:
                    # Look up last known price for this tp_id
                    last_price_info = None
                    if tp_id != TP_ID_PORTFOLIO and tp_id in portfolio_pl.last_known_prices:
                        last_price_info = portfolio_pl.last_known_prices[tp_id]
                    # Get current price info
                    current_price = last_price_info[0] if last_price_info else 'N/A'
                    price_timestamp = last_price_info[1] if last_price_info else 'N/A'

                    # Get previous price and timestamp from last_known_prices
                    prev_price_info = portfolio_pl.last_known_prices.get(tp_id + '_prev', None)
                    if prev_price_info and isinstance(prev_price_info, tuple):
                        prev_price, prev_timestamp = prev_price_info
                    else:
                        prev_price = prev_price_info if prev_price_info else 'N/A'
                        prev_timestamp = 'N/A'

                    # Calculate time delta between price updates
                    price_delta_str = ''
                    if prev_timestamp != 'N/A' and price_timestamp != 'N/A':
                        price_delta_ms = price_timestamp - prev_timestamp
                        price_delta_str = f', price_delta={price_delta_ms}ms'

                    # Get current and previous position returns (now stored as tuples)
                    current_tuple = self.trade_pair_to_position_ret.get(tp_id, None)
                    prev_tuple = self.trade_pair_to_position_ret.get(tp_id + '_prev', None)

                    if current_tuple:
                        current_ret, current_pos_count = current_tuple
                    else:
                        current_ret, current_pos_count = 'N/A', 0

                    if prev_tuple:
                        prev_ret, prev_pos_count = prev_tuple
                    else:
                        prev_ret, prev_pos_count = 'N/A', 0

                    # Calculate time since last order for open positions
                    time_since_last_order_str = ''
                    if positions and historical_positions:
                        # Since there's max one open position per trade pair, find it
                        for hist_pos in historical_positions:
                            if hist_pos.is_open_position and hist_pos.orders:
                                last_order_ms = hist_pos.orders[-1].processed_ms
                                if price_timestamp != 'N/A':
                                    time_diff_ms = price_timestamp - last_order_ms
                                    time_since_last_order_str = f', time_since_last_order={time_diff_ms}ms'
                                break  # Found the single open position

                    last_cp = perf_ledger_bundle[tp_id].cps[-1] if tp_id in perf_ledger_bundle else None
                    print(f'    tp_id {tp_id} price ({prev_price} -> {current_price}) @ {price_timestamp}{price_delta_str}{time_since_last_order_str},'
                          f' position_ret ({prev_ret} -> {current_ret}), n_positions ({prev_pos_count} -> {current_pos_count}). last_cp {last_cp}')
                for p in positions:
                    print(f'        position {p} ')


    def inc_accumulated_time(self, mode, accumulated_time_ms):
        old_accumulated_time = accumulated_time_ms

        if mode == 'second':
            accumulated_time_ms += 1000
            self.mode_to_n_updates['second'] += 1
        elif mode == 'minute':
            accumulated_time_ms += 60000
            self.mode_to_n_updates['minute'] += 1
        else:
            raise Exception(f"Unknown mode: {mode}")

        # Assert we only increment by expected amount
        increment = accumulated_time_ms - old_accumulated_time
        expected_increment = 1000 if mode == 'second' else 60000
        assert increment == expected_increment, f"Invalid time increment: {increment} ms in {mode} mode (expected {expected_increment} ms)"

        return accumulated_time_ms


    def build_perf_ledger(self, perf_ledger_bundle: dict[str, dict[str, PerfLedger]], tp_to_historical_positions: dict[str, Position], start_time_ms, end_time_ms, miner_hotkey, tp_id_to_realtime_position_to_pop: dict[str, Position]) -> bool:
        # tp_id_to_realtime_position_to_pop is a dictionary mapping trade pair IDs to their realtime positions
        portfolio_pl = perf_ledger_bundle[TP_ID_PORTFOLIO]
        is_first_update = len(portfolio_pl.cps) == 0

        # Check if we need to build the ledger forward in time
        # If start_time > end_time, this batch has already been processed
        # BUT: We still need to initialize any new trade pair ledgers before returning
        skip_time_advancement = start_time_ms > end_time_ms


        # For non-first updates, validate that we're continuing from where we left off
        # We should always start from the ledger's last update time
        if not is_first_update:
            # start_time_ms should match the ledger's last_update_ms + 1ms (smallest update interval)
            # If it doesn't, there's likely a bug in the calling code
            expected_start = portfolio_pl.last_update_ms + 1
            gap = start_time_ms - expected_start

            # We should start from exactly where we left off (gap = 0)
            # A negative gap means we're re-processing old data (regeneration)
            # A positive gap means start_time is in the future - this is a bug
            if gap != 0:
                bt.logging.error(f"BUG DETECTED: Attempting to build ledger starting from future time")
                bt.logging.error(f"  Ledger ID: {portfolio_pl.tp_id}")
                bt.logging.error(f"  Ledger last_update_ms: {expected_start} ({TimeUtil.millis_to_formatted_date_str(expected_start)})")
                bt.logging.error(f"  Requested start_time_ms: {start_time_ms} ({TimeUtil.millis_to_formatted_date_str(start_time_ms)})")
                bt.logging.error(f"  Gap: {gap/1000/60:.2f} minutes into the future")
                bt.logging.error(f"  End time: {TimeUtil.millis_to_formatted_date_str(end_time_ms)}")
                raise AssertionError(
                    f"Cannot start building from future time. "
                    f"Ledger at {TimeUtil.millis_to_formatted_date_str(expected_start)}, "
                    f"but start_time is {TimeUtil.millis_to_formatted_date_str(start_time_ms)}"
                )

        if len(portfolio_pl.cps) == 0:
            portfolio_pl.init_with_first_order(portfolio_pl.initialization_time_ms, point_in_time_dd=1.0, current_portfolio_value=1.0,
                                              current_portfolio_fee_spread=1.0, current_portfolio_carry=1.0)

        # Init per-trade-pair perf ledgers
        tp_ids_to_build = [TP_ID_PORTFOLIO]
        for i, (tp_id, positions) in enumerate(tp_to_historical_positions.items()):
            if self.build_portfolio_ledgers_only:
                break
            if tp_id in perf_ledger_bundle:
                # Can only build perf ledger between orders or after all orders have passed.
                tp_ids_to_build.append(tp_id)
            else:
                assert len(positions) == 1
                assert tp_id_to_realtime_position_to_pop and tp_id in tp_id_to_realtime_position_to_pop, (tp_id_to_realtime_position_to_pop.keys(), positions)
                assert len(positions[0].orders) == 0, (tp_id, positions[0], list(perf_ledger_bundle.keys()))

                initialization_time_ms = tp_id_to_realtime_position_to_pop[tp_id].orders[0].processed_ms
                perf_ledger_bundle[tp_id] = PerfLedger(initialization_time_ms=initialization_time_ms, target_ledger_window_ms=self.target_ledger_window_ms)
                # Initialize with the actual initialization time, not the end time
                perf_ledger_bundle[tp_id].init_with_first_order(initialization_time_ms, point_in_time_dd=1.0, current_portfolio_value=1.0,
                                                   current_portfolio_fee_spread=1.0, current_portfolio_carry=1.0)

        # Validate starting point for ALL ledgers that will be built
        for tp_id in tp_ids_to_build:
            perf_ledger = perf_ledger_bundle[tp_id]
            is_ledger_first_update = len(perf_ledger.cps) == 0

            if not is_ledger_first_update:
                gap_from_last_update = start_time_ms - perf_ledger.last_update_ms
                if gap_from_last_update != 1:
                    bt.logging.error(f"Gap validation failed for {tp_id}:")
                    bt.logging.error(f"  perf_ledger.last_update_ms: {perf_ledger.last_update_ms}")
                    bt.logging.error(f"  start_time_ms: {start_time_ms}")
                    bt.logging.error(f"  gap: {gap_from_last_update}")
                    bt.logging.error(f"  Ledger has {len(perf_ledger.cps)} checkpoints")
                    if len(perf_ledger.cps) > 0:
                        bt.logging.error(f"  Last checkpoint time: {perf_ledger.cps[-1].last_update_ms}")
                assert gap_from_last_update == 1, (
                    f"Gap detected for {tp_id} ledger between last_update_ms and start_time_ms: "
                    f"{gap_from_last_update/1000/60:.2f} minutes. "
                    f"Last update: {TimeUtil.millis_to_formatted_date_str(perf_ledger.last_update_ms)}, "
                    f"Start time: {TimeUtil.millis_to_formatted_date_str(start_time_ms)}"
                )

        # If we skipped time advancement (batch already processed), return now
        # We've already initialized any new trade pairs above, so we're done
        if skip_time_advancement:
            bt.logging.debug(f"Skipping time advancement for miner {miner_hotkey} "
                           f"(batch already processed at {TimeUtil.millis_to_formatted_date_str(end_time_ms)})")
            return False

        if portfolio_pl.initialization_time_ms == end_time_ms:
            return False  # Can only build perf ledger between orders or after all orders have passed.

        # "Shortcut" All positions closed and one newly open position OR before the ledger lookback window.
        shortcut_reason, initial_tp_to_return, initial_tp_to_realized_pnl, initial_tp_to_unrealized_pnl, initial_tp_to_spread_fee, initial_tp_to_carry_fee, any_open = \
            self._can_shortcut(tp_to_historical_positions, end_time_ms, tp_id_to_realtime_position_to_pop, start_time_ms, perf_ledger_bundle)
        if shortcut_reason != ShortcutReason.NO_SHORTCUT:
            for tp_id in tp_ids_to_build:
                perf_ledger = perf_ledger_bundle[tp_id]

                # Don't update if end_time is before the ledger's current state
                if perf_ledger.last_update_ms > 0 and end_time_ms < perf_ledger.last_update_ms:
                    bt.logging.warning(f"Skipping shortcut update for {tp_id} - end_time_ms ({TimeUtil.millis_to_formatted_date_str(end_time_ms)}) "
                                   f"is before last_update_ms ({TimeUtil.millis_to_formatted_date_str(perf_ledger.last_update_ms)})")
                    continue

                tp_return, tp_spread_fee, tp_carry_fee = self.get_bypass_values_if_applicable(
                    perf_ledger, tp_id, any_open,
                    initial_tp_to_return[tp_id], initial_tp_to_spread_fee[tp_id], initial_tp_to_carry_fee[tp_id],
                    tp_id_to_realtime_position_to_pop
                )

                tp_to_historical_positions_compact = {}
                for tp, ret in initial_tp_to_return.items():
                    if tp != TP_ID_PORTFOLIO:
                        for candpos in tp_to_historical_positions[tp]:
                            if candpos.return_at_close < .5:
                                tp_to_historical_positions_compact[tp] = candpos

                dd = {'initial_tp_to_return': initial_tp_to_return, 'miner_hotkey': miner_hotkey,
                      'shortcut_reason': shortcut_reason,
                      'tp_id': tp_id, 'start_time_ms': TimeUtil.millis_to_formatted_date_str(start_time_ms),
                      'end_time_ms': TimeUtil.millis_to_formatted_date_str(end_time_ms),
                      'tp_to_historical_positions_compact': tp_to_historical_positions_compact,
                      'realtime_position_to_pop': tp_id_to_realtime_position_to_pop.keys()
                      }
                perf_ledger.update_pl(tp_return, end_time_ms, miner_hotkey, TradePairReturnStatus.TP_MARKET_NOT_OPEN,
                                      tp_spread_fee, tp_carry_fee, initial_tp_to_realized_pnl[tp_id], initial_tp_to_unrealized_pnl[tp_id],
                                      tp_debug=tp_id + '_shortcut', debug_dict=dd)


                perf_ledger.purge_old_cps()
            return False

        #print(f"Building perf ledger for {miner_hotkey} from {TimeUtil.millis_to_verbose_formatted_date_str(start_time_ms)} to {TimeUtil.millis_to_verbose_formatted_date_str(end_time_ms)} ({(end_time_ms - start_time_ms) // 1000} s) \
        #       mode_to_n_updates {self.mode_to_n_updates}. update_to_n_open_positions {self.update_to_n_open_positions}")
        tp_to_closed_pos_unrealized_pnl = {x: 0.0 for x in tp_ids_to_build} # closed positions always have 0 unrealized pnl
        tp_to_closed_pos_return, tp_to_closed_pos_realized_pnl, tp_to_closed_pos_spread_fee, tp_to_closed_pos_carry_fee, tp_to_historical_positions_dense, \
            open_positions_tp_ids = self.condense_positions(tp_ids_to_build, tp_to_historical_positions)

        # Clean up prices for closed positions
        self.cleanup_closed_position_prices(portfolio_pl, open_positions_tp_ids)

        # We avoided a shortcut. Any trade pairs from open positions (tp_to_historical_positions_dense) need to be in the ledgers bundle.

        n_open_positions = len(open_positions_tp_ids)
        assert n_open_positions, ('zero open positions implies a shortcut should have been taken')
        self.update_to_n_open_positions[n_open_positions] += 1
        default_mode = self.get_default_update_mode(start_time_ms, end_time_ms, n_open_positions)

        accumulated_time_ms = 0

        # Initialize tracking for time increments
        self._last_loop_t_ms = {}
        self._last_ledger_update_ms = {}
        for tp_id in tp_ids_to_build:
            self._last_ledger_update_ms[tp_id] = perf_ledger_bundle[tp_id].last_update_ms

        # closed positions have the same stats throughout the interval. lets do a single update now
        # so that filling the void uses the current state of those position(s)
        for tp_id in tp_ids_to_build:
            if tp_id in open_positions_tp_ids or tp_id == TP_ID_PORTFOLIO:
                continue
            perf_ledger = perf_ledger_bundle[tp_id]
            assert perf_ledger.last_update_ms < end_time_ms, (perf_ledger.last_update_ms, end_time_ms, tp_id, perf_ledger.last_update_ms - end_time_ms)

            current_return, current_spread_fee, current_carry_fee = self.get_bypass_values_if_applicable(
                perf_ledger, tp_id, TradePairReturnStatus.TP_NO_OPEN_POSITIONS,
                tp_to_closed_pos_return[tp_id], tp_to_closed_pos_spread_fee[tp_id], tp_to_closed_pos_carry_fee[tp_id],
                tp_id_to_realtime_position_to_pop
            )

            perf_ledger.update_pl(current_return, start_time_ms, miner_hotkey, TradePairReturnStatus.TP_NO_OPEN_POSITIONS,
                                  current_spread_fee, current_carry_fee, tp_to_closed_pos_realized_pnl[tp_id], tp_to_closed_pos_unrealized_pnl[tp_id])


        # Check if the while loop will execute at all
        if start_time_ms + accumulated_time_ms >= end_time_ms:
            # This should have been caught by the shortcut logic, but handle it defensively
            # Initialize variables needed after the loop with initial values
            tp_to_current_return = initial_tp_to_return.copy()
            tp_to_any_open = {tp_id: TradePairReturnStatus.TP_NO_OPEN_POSITIONS for tp_id in tp_ids_to_build}
            tp_to_current_spread_fee = initial_tp_to_spread_fee.copy()
            tp_to_current_carry_fee = initial_tp_to_carry_fee.copy()
            tp_to_realized_pnl = initial_tp_to_realized_pnl.copy()
            tp_to_unrealized_pnl = initial_tp_to_unrealized_pnl.copy()

            bt.logging.warning(f"build_perf_ledger: while loop will not execute for miner {miner_hotkey}. "
                             f"start_time: {TimeUtil.millis_to_formatted_date_str(start_time_ms)}, "
                             f"end_time: {TimeUtil.millis_to_formatted_date_str(end_time_ms)}")

        while start_time_ms + accumulated_time_ms < end_time_ms:
            # Need high resolution at the start and end of the time window
            mode = self.get_current_update_mode(default_mode, start_time_ms, end_time_ms, accumulated_time_ms)
            t_ms = start_time_ms + accumulated_time_ms

            # Verify proper time increments for all ledgers being built
            if accumulated_time_ms > 0:
                for tp_id in tp_ids_to_build:
                    if tp_id in self._last_loop_t_ms:
                        actual_increment = t_ms - self._last_loop_t_ms[tp_id]

                        # Valid increments are 1000ms (second mode) or 60000ms (minute mode)
                        # Mode can switch during processing, so we accept either increment
                        valid_increments = [1000, 60000]

                        assert actual_increment in valid_increments, (
                            f"Time increment violation for {tp_id}: {actual_increment}ms "
                            f"(expected 1000ms or 60000ms). "
                            f"Current: {TimeUtil.millis_to_formatted_date_str(t_ms)}, "
                            f"Previous: {TimeUtil.millis_to_formatted_date_str(self._last_loop_t_ms[tp_id])}. "
                            f"Please alert a team member ASAP!"
                        )

            if t_ms < portfolio_pl.last_update_ms:
                time_diff_ms = portfolio_pl.last_update_ms - t_ms
                time_diff_days = time_diff_ms / (1000 * 60 * 60 * 24)

                bt.logging.error("CRITICAL TIMESTAMP BUG DETECTED:")
                bt.logging.error(f"  Current processing time: {t_ms} ({TimeUtil.millis_to_formatted_date_str(t_ms)})")
                bt.logging.error(f"  Last checkpoint time:    {portfolio_pl.last_update_ms} ({TimeUtil.millis_to_formatted_date_str(portfolio_pl.last_update_ms)})")
                bt.logging.error(f"  Time difference:         {time_diff_ms} ms ({time_diff_days:.1f} days)")
                bt.logging.error(f"  Mode: {mode}")
                bt.logging.error(f"  Parallel mode: {self.parallel_mode}")
                bt.logging.error(f"  Checkpoint details: accum_ms={portfolio_pl.cps[-1].accum_ms if portfolio_pl.cps else 'No checkpoints'}")

                if time_diff_days > 1:
                    bt.logging.error(f"  EXTREME TIMESTAMP ERROR: Checkpoint is {time_diff_days:.1f} days in the future!")
                    bt.logging.error("  This indicates a critical bug in void filling or boundary logic.")
                    bt.logging.error(f"  Portfolio PL object: {portfolio_pl}")

                raise Exception(f'CRITICAL TIMESTAMP BUG DETECTED: t_ms {t_ms} is before last_update_ms {portfolio_pl.last_update_ms}. '
                                f'Check logs for more details.')

            assert t_ms > portfolio_pl.last_update_ms, (f"t_ms: {t_ms}, "
                                                         f"last_update_ms: {TimeUtil.millis_to_formatted_date_str(portfolio_pl.last_update_ms)},"
                                                         f"mode: {mode},"
                                                         f" delta_ms: {(t_ms - portfolio_pl.last_update_ms)} ms. perf ledger {portfolio_pl}")

            tp_to_current_return, tp_to_realized_pnl, tp_to_unrealized_pnl, tp_to_any_open, tp_to_current_spread_fee, tp_to_current_carry_fee, = \
                self.positions_to_portfolio_return(tp_ids_to_build, tp_to_historical_positions_dense, t_ms, mode,
                   end_time_ms, tp_to_closed_pos_return, tp_to_closed_pos_realized_pnl, tp_to_closed_pos_unrealized_pnl, tp_to_closed_pos_spread_fee, tp_to_closed_pos_carry_fee, portfolio_pl,
                   tp_id_to_realtime_position_to_pop)
            portfolio_return = tp_to_current_return[TP_ID_PORTFOLIO]

            if portfolio_return == 0 and self.check_liquidated(miner_hotkey, portfolio_return, t_ms, tp_to_historical_positions, perf_ledger_bundle):
                return True

            self.debug_significant_portfolio_drop(mode, portfolio_return, perf_ledger_bundle, t_ms, miner_hotkey, tp_to_historical_positions, open_positions_tp_ids, start_time_ms, end_time_ms)

            tp_ids_to_update = [TP_ID_PORTFOLIO] if self.build_portfolio_ledgers_only else list(open_positions_tp_ids) + [TP_ID_PORTFOLIO]
            for tp_id in tp_ids_to_update:
                perf_ledger = perf_ledger_bundle[tp_id]

                current_return, current_spread_fee, current_carry_fee = self.get_bypass_values_if_applicable(
                    perf_ledger, tp_id, tp_to_any_open[tp_id],
                    tp_to_current_return[tp_id], tp_to_current_spread_fee[tp_id], tp_to_current_carry_fee[tp_id],
                    tp_id_to_realtime_position_to_pop
                )

                perf_ledger.update_pl(current_return, t_ms, miner_hotkey, tp_to_any_open[tp_id],
                                      current_spread_fee, current_carry_fee,
                                      tp_to_realized_pnl[tp_id], tp_to_unrealized_pnl[tp_id],
                                      tp_debug=tp_id)


                # Verify the ledger was updated to current t_ms
                assert perf_ledger.last_update_ms == t_ms, (
                    f"Ledger {tp_id} last_update_ms doesn't match current t_ms after update. "
                    f"Ledger: {TimeUtil.millis_to_formatted_date_str(perf_ledger.last_update_ms)}, "
                    f"t_ms: {TimeUtil.millis_to_formatted_date_str(t_ms)}"
                )

                # Verify continuous updates (no gaps)
                if tp_id in self._last_ledger_update_ms:
                    gap = perf_ledger.last_update_ms - self._last_ledger_update_ms[tp_id]

                    # Valid gaps are 1000ms (second mode) or 60000ms (minute mode)
                    # Mode can switch during processing, so we accept either gap
                    valid_gaps = [1, 1000, 60000]

                    assert gap in valid_gaps, (
                        f"Ledger {tp_id} jumped {gap}ms (expected 1000ms or 60000ms). "
                        f"Previous: {TimeUtil.millis_to_formatted_date_str(self._last_ledger_update_ms[tp_id])}, "
                        f"Current: {TimeUtil.millis_to_formatted_date_str(perf_ledger.last_update_ms)}. "
                        f"Please alert a team member ASAP!"
                    )

                self._last_ledger_update_ms[tp_id] = perf_ledger.last_update_ms
                self._last_loop_t_ms[tp_id] = t_ms

            # Verify all ledgers are synchronized
            portfolio_time = perf_ledger_bundle[TP_ID_PORTFOLIO].last_update_ms
            for tp_id in tp_ids_to_update:
                if tp_id == TP_ID_PORTFOLIO:
                    continue

                ledger_time = perf_ledger_bundle[tp_id].last_update_ms
                assert ledger_time == portfolio_time, (
                    f"Ledger {tp_id} out of sync with portfolio ledger. "
                    f"Portfolio: {TimeUtil.millis_to_formatted_date_str(portfolio_time)}, "
                    f"{tp_id}: {TimeUtil.millis_to_formatted_date_str(ledger_time)}"
                )

            accumulated_time_ms = self.inc_accumulated_time(mode, accumulated_time_ms)

        # Get last sliver of time for open positions and fill the void for closed positions.
        # This also ensures return aligns with the price baked into the Order object.
        # Note - nothing changes on closed positions over time, not even fees.
        for tp_id in tp_ids_to_build:
            perf_ledger = perf_ledger_bundle[tp_id]
            assert perf_ledger.last_update_ms <= end_time_ms, (perf_ledger.last_update_ms, end_time_ms)

            # Check if boundary correction is needed for this specific trade pair
            current_tp_position = tp_id_to_realtime_position_to_pop.get(tp_id) if tp_id != TP_ID_PORTFOLIO else None
            boundary_correction_enabled = (tp_id in tp_to_historical_positions_dense and
                                          current_tp_position and
                                          tp_id in tp_ids_to_build)

            # For portfolio, check if any position needs correction
            if tp_id == TP_ID_PORTFOLIO:
                # Apply boundary correction if any trade pair has a realtime_position_to_pop
                for check_tp_id, check_position in tp_id_to_realtime_position_to_pop.items():
                    if check_tp_id in tp_to_historical_positions_dense and check_tp_id in tp_ids_to_build:
                        boundary_correction_enabled = True
                        current_tp_position = check_position  # Use for correction calculation
                        break

            # Calculate normal boundary correction values first
            if boundary_correction_enabled and current_tp_position:
                correction_tp_id = current_tp_position.trade_pair.trade_pair_id
                calculated_return = (tp_to_current_return[tp_id] /
                                   tp_to_historical_positions_dense[correction_tp_id][0].return_at_close *
                                   current_tp_position.return_at_close)
            else:
                calculated_return = tp_to_current_return[tp_id]

            current_return, current_spread_fee, current_carry_fee = self.get_bypass_values_if_applicable(
                perf_ledger, tp_id, tp_to_any_open[tp_id],
                calculated_return, tp_to_current_spread_fee[tp_id], tp_to_current_carry_fee[tp_id],
                tp_id_to_realtime_position_to_pop
            )

            # Fix: Convert unrealized PnL to realized PnL for positions that closed during this checkpoint
            # When a position closes during a checkpoint period, tp_to_historical_positions_dense contains
            # the OPEN position (before FLAT order), while tp_id_to_realtime_position_to_pop contains the
            # CLOSED position (after FLAT order). The tick-by-tick loop accumulates unrealized PnL from
            # the open position, but we need to convert it to realized PnL from the closed position.
            if tp_id_to_realtime_position_to_pop:
                # For portfolio, check all positions; for trade pairs, check only the specific tp_id
                positions_to_check = []
                if tp_id == TP_ID_PORTFOLIO:
                    positions_to_check = list(tp_id_to_realtime_position_to_pop.items())
                elif tp_id in tp_id_to_realtime_position_to_pop:
                    positions_to_check = [(tp_id, tp_id_to_realtime_position_to_pop[tp_id])]

                for check_tp_id, closed_position in positions_to_check:
                    # Only process if position is actually closed
                    if not closed_position.is_closed_position:
                        continue

                    # Find the corresponding open position in historical positions
                    if check_tp_id in tp_to_historical_positions_dense:
                        for hist_pos in tp_to_historical_positions_dense[check_tp_id]:
                            if hist_pos.position_uuid == closed_position.position_uuid:
                                # Convert unrealized PnL to realized PnL
                                # Subtract the open position's unrealized PnL (accumulated during tick-by-tick)
                                tp_to_unrealized_pnl[tp_id] -= hist_pos.unrealized_pnl
                                # Add the closed position's realized PnL
                                tp_to_realized_pnl[tp_id] += closed_position.realized_pnl
                                bt.logging.debug(
                                    f"Converted PnL for position {closed_position.position_uuid[:8]} closing during checkpoint: "
                                    f"removed unrealized ${hist_pos.unrealized_pnl:.2f}, added realized ${closed_position.realized_pnl:.2f}"
                                )
                                break

            perf_ledger.update_pl(current_return, end_time_ms, miner_hotkey, tp_to_any_open[tp_id],
                                  current_spread_fee, current_carry_fee,
                                  tp_to_realized_pnl[tp_id], tp_to_unrealized_pnl[tp_id])

            perf_ledger.purge_old_cps()

        # Final validation: ensure all ledgers reached end_time_ms
        for tp_id in tp_ids_to_build:
            perf_ledger = perf_ledger_bundle[tp_id]
            assert perf_ledger.last_update_ms == end_time_ms, (
                f"Ledger {tp_id} not updated to end_time_ms after build_perf_ledger. "
                f"Last update: {TimeUtil.millis_to_formatted_date_str(perf_ledger.last_update_ms)}, "
                f"Expected: {TimeUtil.millis_to_formatted_date_str(end_time_ms)}"
            )


        #n_minutes_between_intervals = (end_time_ms - start_time_ms) // 60000
        #print(f'Updated between {TimeUtil.millis_to_formatted_date_str(start_time_ms)} and {TimeUtil.millis_to_formatted_date_str(end_time_ms)} ({n_minutes_between_intervals} min). mode_to_ticks {mode_to_ticks}. Default mode {default_mode}')
        return False

    def mutate_position_returns_for_continuity(self, tp_to_historical_positions, perf_ledger_bundle_candidate, t_ms, debug_str=''):
        if not perf_ledger_bundle_candidate:
            return {}
        if TP_ID_PORTFOLIO not in perf_ledger_bundle_candidate:
            return {}

        portfolio_ledger = perf_ledger_bundle_candidate[TP_ID_PORTFOLIO]

        # Collect continuity application data for aggregate logging
        continuity_applications = {}

        for tp_id, positions_list in tp_to_historical_positions.items():
            if tp_id in portfolio_ledger.last_known_prices:
                last_price, last_price_ms = portfolio_ledger.last_known_prices[tp_id]
                for position in positions_list:
                    if position.is_open_position:

                        if not position.orders:
                            # Position just opened with no orders yet. We are building ledgers right up to the point before this order.
                            continue

                        if last_price_ms <= position.orders[-1].processed_ms:
                            bt.logging.warning(f'Unexpected price continuity rejection for {tp_id} at {t_ms} with last known price {last_price} at {last_price_ms}. Position last order at {position.orders[-1].processed_ms}')
                            continue


                        # Record the price transition and return change for logging
                        last_order_price = position.orders[-1].price
                        old_return = position.return_at_close

                        # Calculate the return at the last known price point
                        position_spread_fee, _ = self.position_uuid_to_cache[position.position_uuid].get_spread_fee(position, t_ms)
                        position_carry_fee, _ = self.position_uuid_to_cache[position.position_uuid].get_carry_fee(t_ms, position)
                        position.set_returns(last_price, self._live_price_client, time_ms=t_ms, total_fees=position_spread_fee * position_carry_fee)

                        # Store info for aggregate logging with both price and return changes
                        new_return = position.return_at_close
                        continuity_applications[tp_id] = {
                            'price_change': f"{last_order_price:.6g} -> {last_price:.6g}",
                            'return_change': f"{old_return:.6g} -> {new_return:.6g}",
                            'leverage': position.net_leverage,
                            'position_uuid': position.position_uuid
                        }

        return continuity_applications

    def _log_continuity_summary(self, hotkey: str, continuity_changes: dict, tp_to_historical_positions: dict):
        """Log an aggregate summary of price continuity applications for a miner."""
        # Count open positions and unique trade pairs
        n_open_positions = sum(1 for tp_positions in tp_to_historical_positions.values()
                              for pos in tp_positions if pos.is_open_position)
        n_trade_pairs_traded = len(tp_to_historical_positions)

        # Format the changes - each entry has both price and return changes
        changes_parts = []
        for tp_id, changes in continuity_changes.items():
            price_change = changes['price_change']
            return_change = changes['return_change']
            leverage = changes['leverage']
            position_uuid = changes['position_uuid']
            changes_parts.append(f"{tp_id}: price({price_change}), return({return_change}), lev={leverage:.2f}, position_uuid={position_uuid}")

        changes_str = ", ".join(changes_parts)

        bt.logging.info(
            f"perf ledger price continuity applied for miner {hotkey[:8]}... | "
            f"Open positions: {n_open_positions} | "
            f"Trade pairs traded: {n_trade_pairs_traded} | "
            f"Updates: {{{changes_str}}}"
        )

    def update_one_perf_ledger_bundle(self, hotkey_i: int, n_hotkeys: int, hotkey: str, positions: List[Position],
                                      now_ms: int,
                                      existing_perf_ledger_bundles: dict[str, dict[str, PerfLedger]]) -> None | dict[str, PerfLedger]:
        # live_price_fetcher is now created in __init__ - no conditional needed
        eliminated = False
        self.n_api_calls = 0
        self.mode_to_n_updates = {'second': 0, 'minute': 0}
        self.tp_to_mfs = {}
        self.update_to_n_open_positions = defaultdict(int)

        t0 = time.time()
        perf_ledger_bundle_candidate = existing_perf_ledger_bundles.get(hotkey)
        if perf_ledger_bundle_candidate and TP_ID_PORTFOLIO in perf_ledger_bundle_candidate and now_ms < perf_ledger_bundle_candidate[TP_ID_PORTFOLIO].last_update_ms:
            now_formatted = TimeUtil.millis_to_formatted_date_str(now_ms)
            last_update_formatted = TimeUtil.millis_to_formatted_date_str(perf_ledger_bundle_candidate[TP_ID_PORTFOLIO].last_update_ms)
            raise Exception(f'Trying to update in the past for {hotkey}. now {now_formatted} < last update {last_update_formatted}')

        continuity_established = False  # Track if we've already established price continuity

        if perf_ledger_bundle_candidate and self._is_v1_perf_ledger(perf_ledger_bundle_candidate):
            bt.logging.warning(f"hotkey {hotkey} has legacy perf ledger. Wiping.")
            perf_ledger_bundle_candidate = None

        if perf_ledger_bundle_candidate is None:
            first_order_time_ms = min(p.orders[0].processed_ms for p in positions)
            perf_ledger_bundle_candidate = {TP_ID_PORTFOLIO: PerfLedger(initialization_time_ms=first_order_time_ms, target_ledger_window_ms=self.target_ledger_window_ms)}
            verbose = True
            bt.logging.info(f"Creating new perf ledger for {hotkey} with init time: {TimeUtil.millis_to_formatted_date_str(first_order_time_ms)}")
        else:
            perf_ledger_bundle_candidate = deepcopy(perf_ledger_bundle_candidate)
            verbose = False

        for tp_id, perf_ledger in perf_ledger_bundle_candidate.items():
            perf_ledger.init_max_portfolio_value()

        self.trade_pair_to_position_ret = {}
        #if hotkey in self.hk_to_dd_stats:
        #    del self.hk_to_dd_stats[hotkey]

        tp_to_historical_positions = defaultdict(list)
        sorted_timeline, last_event_time_ms = self.generate_order_timeline(positions, now_ms, hotkey)  # Enforces our "now_ms" constraint
        # There hasn't been a new order since the last update time. Just need to update for open positions
        building_from_new_orders = True
        if last_event_time_ms < perf_ledger_bundle_candidate[TP_ID_PORTFOLIO].last_update_ms:
            building_from_new_orders = False
            # Preserve returns from realtime positions
            sorted_timeline = []
            tp_to_historical_positions = {}
            for p in positions:
                symbol = p.trade_pair.trade_pair_id
                if symbol in tp_to_historical_positions:
                    tp_to_historical_positions[symbol].append(p)
                else:
                    tp_to_historical_positions[symbol] = [p]

        # Building for scratch or there have been order(s) since the last update time
        event_idx = 0
        tp_id_to_realtime_position_to_pop = {}
        while event_idx < len(sorted_timeline):
            for tp_id, realtime_position_to_pop in tp_id_to_realtime_position_to_pop.items():
                symbol = realtime_position_to_pop.trade_pair.trade_pair_id
                tp_to_historical_positions[symbol][-1] = realtime_position_to_pop
                if realtime_position_to_pop.return_at_close == 0:  # liquidated
                    self.check_liquidated(hotkey, 0.0, realtime_position_to_pop.close_ms, tp_to_historical_positions, perf_ledger_bundle_candidate)
                    eliminated = True
                    break

            # Collect all orders within the same second (ms // 1000)
            batch_order_timestamp = sorted_timeline[event_idx][0].processed_ms
            batch_events = []

            while event_idx < len(sorted_timeline) and sorted_timeline[event_idx][0].processed_ms == batch_order_timestamp:
                batch_events.append(sorted_timeline[event_idx])
                event_idx += 1

            # Process all orders in this second and collect realtime_position_to_pop per trade pair
            tp_id_to_realtime_position_to_pop = {}
            for (order, position) in batch_events:
                symbol = position.trade_pair.trade_pair_id
                pos, batch_realtime_position_to_pop = self.get_historical_position(position, order.processed_ms)

                # Track realtime_position_to_pop per trade pair
                if batch_realtime_position_to_pop:
                    tp_id = batch_realtime_position_to_pop.trade_pair.trade_pair_id
                    if tp_id in tp_id_to_realtime_position_to_pop:
                        pos1_dup_order = tp_id_to_realtime_position_to_pop[tp_id].orders[-1]
                        pos1_no_orders = tp_id_to_realtime_position_to_pop[tp_id]
                        pos1_no_orders.orders = []

                        pos2_dup_order = batch_realtime_position_to_pop.orders[-1]
                        pos2_no_orders = batch_realtime_position_to_pop
                        pos2_no_orders.orders = []
                        raise ValueError(f"Multiple realtime_position_to_pop for hotkey {hotkey} for same trade pair "
                                         f"{tp_id} in same millisecond {order.processed_ms}."
                                         f" pos1 {pos1_no_orders}, pos2 {pos2_no_orders}, "
                                         f"order1 {pos1_dup_order}, order2 {pos2_dup_order}"
                    )
                    tp_id_to_realtime_position_to_pop[tp_id] = batch_realtime_position_to_pop

                if (symbol in tp_to_historical_positions and
                        pos.position_uuid == tp_to_historical_positions[symbol][-1].position_uuid):
                    tp_to_historical_positions[symbol][-1] = pos
                else:
                    tp_to_historical_positions[symbol].append(pos)

                # Sanity check for each position
                n_open_positions = sum(1 for p in tp_to_historical_positions[symbol] if p.is_open_position)
                n_closed_positions = sum(1 for p in tp_to_historical_positions[symbol] if p.is_closed_position)

                # Diagnostic logging for assertion violations
                if n_open_positions > 1 or (n_open_positions == 1 and not tp_to_historical_positions[symbol][-1].is_open_position):
                    open_positions = [p for p in tp_to_historical_positions[symbol] if p.is_open_position]
                    last_position = tp_to_historical_positions[symbol][-1]

                    bt.logging.error(f"ASSERTION VIOLATION DIAGNOSTICS for hotkey {hotkey} trade_pair {symbol}:")
                    bt.logging.error(f"  n_open_positions: {n_open_positions}, n_closed_positions: {n_closed_positions}")
                    bt.logging.error(f"  last_position.is_open_position: {last_position.is_open_position}")
                    bt.logging.error(f"  last_position.is_closed_position: {last_position.is_closed_position}")
                    bt.logging.error(f"  last_position.close_ms: {last_position.close_ms}")
                    bt.logging.error(f"  last_position.position_type: {last_position.position_type}")
                    bt.logging.error(f"  last_position.n_orders: {len(last_position.orders)}")

                    # Check for inconsistent state: close_ms set but is_open_position=True
                    for i, p in enumerate(open_positions):
                        has_flat_order = any(o.order_type == OrderType.FLAT for o in p.orders)
                        bt.logging.error(f"  open_position[{i}]:")
                        bt.logging.error(f"    position_uuid: {p.position_uuid}")
                        bt.logging.error(f"    close_ms: {p.close_ms} (INCONSISTENT: should be None for open positions)")
                        bt.logging.error(f"    is_closed_position: {p.is_closed_position}")
                        bt.logging.error(f"    position_type: {p.position_type}")
                        bt.logging.error(f"    n_orders: {len(p.orders)}")
                        bt.logging.error(f"    has_flat_order: {has_flat_order}")
                        bt.logging.error(f"    order_types: {[o.order_type.value for o in p.orders]}")
                        bt.logging.error(f"    order_sources: {[o.src for o in p.orders]}")
                        if not has_flat_order:
                            bt.logging.error(f"    THEORY CONFIRMED: Open position WITHOUT FLAT order but likely manually closed!")

                assert n_open_positions == 0 or n_open_positions == 1, (n_open_positions, n_closed_positions, [p for p in tp_to_historical_positions[symbol] if p.is_open_position])
                if n_open_positions == 1:
                    assert tp_to_historical_positions[symbol][-1].is_open_position, (n_open_positions, n_closed_positions, [p for p in tp_to_historical_positions[symbol] if p.is_open_position])

            # Perf ledger is already built, we just need to run the above loop to build tp_to_historical_positions
            if not building_from_new_orders:
                continue

            # Building from a checkpoint ledger. Skip until we get to the new order(s).
            portfolio_ledger = perf_ledger_bundle_candidate[TP_ID_PORTFOLIO]
            portfolio_last_update_ms = portfolio_ledger.last_update_ms

            if portfolio_last_update_ms == 0:
                # If no checkpoints exist, use initialization time
                portfolio_last_update_ms = portfolio_ledger.initialization_time_ms

            # Skip batches that are strictly before the last update
            # (batches at the same timestamp will be handled by build_perf_ledger)
            if batch_order_timestamp < portfolio_last_update_ms:
                continue

            # Apply price continuity before building ledger (only if not already done)
            if not continuity_established:
                continuity_changes = self.mutate_position_returns_for_continuity(tp_to_historical_positions,
                 perf_ledger_bundle_candidate, portfolio_last_update_ms, debug_str=f'pre-batch {batch_order_timestamp}. '
                   f'start_time {TimeUtil.millis_to_formatted_date_str(portfolio_last_update_ms)} end_time {TimeUtil.millis_to_formatted_date_str(batch_order_timestamp)}')
                continuity_established = True

                # Log aggregate continuity info if changes were made
                #if continuity_changes:
                #    self._log_continuity_summary(hotkey, continuity_changes, tp_to_historical_positions)

            # Need to catch up from perf_ledger.last_update_ms to max timestamp in batch
            # Pass the dictionary of positions (empty dict if none, single entry if one, multiple if many)
            eliminated = self.build_perf_ledger(perf_ledger_bundle_candidate, tp_to_historical_positions,
                                               portfolio_last_update_ms + 1, batch_order_timestamp,
                                               hotkey, tp_id_to_realtime_position_to_pop)

            if eliminated:
                break

        if eliminated and self.parallel_mode != ParallelizationMode.SERIAL:
            return perf_ledger_bundle_candidate

        # We have processed all orders. Need to catch up to now_ms
        for tp_id, realtime_position_to_pop in tp_id_to_realtime_position_to_pop.items():
            symbol = realtime_position_to_pop.trade_pair.trade_pair_id
            tp_to_historical_positions[symbol][-1] = realtime_position_to_pop

        portfolio_perf_ledger = perf_ledger_bundle_candidate[TP_ID_PORTFOLIO]
        if now_ms > portfolio_perf_ledger.last_update_ms:
            # Always start from the current ledger state
            # The ledger may have been updated during order processing above
            current_last_update = portfolio_perf_ledger.last_update_ms
            if current_last_update == 0:
                # If no checkpoints exist, use initialization time
                current_last_update = portfolio_perf_ledger.initialization_time_ms

            # Apply price continuity before final build_perf_ledger call
            if not continuity_established:
                continuity_changes = self.mutate_position_returns_for_continuity(tp_to_historical_positions, perf_ledger_bundle_candidate, current_last_update, debug_str='final')
                continuity_established = True

                # Log aggregate continuity info if changes were made
                #if continuity_changes:
                #    self._log_continuity_summary(hotkey, continuity_changes, tp_to_historical_positions)

            self.build_perf_ledger(perf_ledger_bundle_candidate, tp_to_historical_positions,
                                   current_last_update + 1, now_ms, hotkey, {})

        self.hk_to_last_order_processed_ms[hotkey] = last_event_time_ms

        lag = (TimeUtil.now_in_millis() - portfolio_perf_ledger.last_update_ms) // 1000
        total_product = portfolio_perf_ledger.get_total_product()
        last_portfolio_value = portfolio_perf_ledger.prev_portfolio_ret
        pl_update_start_time_ms = perf_ledger_bundle_candidate[TP_ID_PORTFOLIO].last_update_ms
        if pl_update_start_time_ms == 0:
            pl_update_start_time_ms = perf_ledger_bundle_candidate[TP_ID_PORTFOLIO].initialization_time_ms
        if verbose:
            bt.logging.success(
                f"Done updating perf ledger for {hotkey} {hotkey_i + 1}/{n_hotkeys} in {time.time() - t0} "
                f"Update start time {TimeUtil.millis_to_formatted_date_str(pl_update_start_time_ms)}. End time {TimeUtil.millis_to_formatted_date_str(now_ms)}. "
                f"(s). Lag: {lag} (s). Total product: {total_product}. Last portfolio value: {last_portfolio_value}."
                f" n_api_calls: {self.n_api_calls} dd stats {None}. "
                f" last cp {portfolio_perf_ledger.cps[-1] if portfolio_perf_ledger.cps else None}. perf_ledger_mpv {portfolio_perf_ledger.max_return} "
                f"perf_ledger_initialization_time {TimeUtil.millis_to_formatted_date_str(portfolio_perf_ledger.initialization_time_ms)}. "
                f"mode_to_n_updates {self.mode_to_n_updates}. update_to_n_open_positions {self.update_to_n_open_positions}, self.tp_to_mfs {self.tp_to_mfs}")

        # If running in parallel mode, return the result instead of updating in place
        if self.parallel_mode != ParallelizationMode.SERIAL:
            return perf_ledger_bundle_candidate
        else:
            # Write candidate at the very end in case an exception leads to a partial update
            existing_perf_ledger_bundles[hotkey] = perf_ledger_bundle_candidate

    @timeme
    def write_perf_ledger_eliminations_to_disk(self, eliminations):
        output_location = ValiBkpUtils.get_perf_ledger_eliminations_dir(running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(output_location, eliminations)

    def get_perf_ledger_eliminations(self, first_fetch=False):
        if first_fetch:
            location = ValiBkpUtils.get_perf_ledger_eliminations_dir(running_unit_tests=self.running_unit_tests)
            cached_eliminations = ValiUtils.get_vali_json_file(location)
            return cached_eliminations
        else:
            return self.pl_elimination_rows

    def update_all_perf_ledgers(self, hotkey_to_positions: dict[str, List[Position]],
                                existing_perf_ledgers: dict[str, dict[str, PerfLedger]],
                                now_ms: int) -> None | dict[str, dict[str, PerfLedger]]:
        t_init = time.time()
        self.now_ms = now_ms
        self.candidate_pl_elimination_rows = []

        n_hotkeys = len(hotkey_to_positions)
        for hotkey_i, (hotkey, positions) in enumerate(hotkey_to_positions.items()):
            try:
                self.update_one_perf_ledger_bundle(hotkey_i, n_hotkeys, hotkey, positions, now_ms, existing_perf_ledgers)
            except Exception as e:
                bt.logging.error(f"Error updating perf ledger for {hotkey}: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc())
                continue

        n_perf_ledgers = len(existing_perf_ledgers) if existing_perf_ledgers else 0
        n_hotkeys_with_positions = len(hotkey_to_positions) if hotkey_to_positions else 0
        bt.logging.success(f"Done updating perf ledger for all hotkeys in {time.time() - t_init} s. n_perf_ledgers {n_perf_ledgers}. n_hotkeys_with_positions {n_hotkeys_with_positions}")
        if not self.is_backtesting:
            self.write_perf_ledger_eliminations_to_disk(self.candidate_pl_elimination_rows)
        # clear and populate proxy list in a multiprocessing-friendly way
        del self.pl_elimination_rows[:]
        self.pl_elimination_rows.extend(self.candidate_pl_elimination_rows)
        for i, x in enumerate(self.candidate_pl_elimination_rows):
            self.pl_elimination_rows[i] = x

        if self._is_shutdown():
            return

        self.save_perf_ledgers(existing_perf_ledgers)
        return existing_perf_ledgers


    def get_positions_perf_ledger(self, testing_one_hotkey=None):
        #testing_one_hotkey = '5GzYKUYSD5d7TJfK4jsawtmS2bZDgFuUYw8kdLdnEDxSykTU'
        hotkeys_with_no_positions = set()
        if testing_one_hotkey:
            hotkey_to_positions = self._position_manager_client.get_positions_for_hotkeys(
                [testing_one_hotkey], sort_positions=True
            )
        else:
            # live_price_fetcher is now created in __init__ - no conditional needed
            hotkey_to_positions = self._position_manager_client.get_positions_for_all_miners(sort_positions=True, filter_eliminations=True)
            n_positions_total = 0
            n_hotkeys_total = len(hotkey_to_positions)
            # Keep only hotkeys with positions
            for k, positions in hotkey_to_positions.items():
                # Rebuild closed positions to ensure returns are accurate WRT latest fee structure and retro prices.
                for p in positions:
                    if p.is_closed_position:
                        p.rebuild_position_with_updated_orders(self._live_price_client)
                n_positions = len(positions)
                n_positions_total += n_positions
                if n_positions == 0:
                    hotkeys_with_no_positions.add(k)
            for k in hotkeys_with_no_positions:
                del hotkey_to_positions[k]
            bt.logging.info('PERF LEDGERS TOTAL N POSITIONS IN MEMORY: ' + str(n_positions_total), 'TOTAL N HOTKEYS IN MEMORY: ' + str(n_hotkeys_total))

        return hotkey_to_positions, hotkeys_with_no_positions

    def generate_perf_ledgers_for_analysis(self, hotkey_to_positions: dict[str, List[Position]], t_ms: int = None) -> dict[str, dict[str, PerfLedger]]:
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis()  # Time to build the perf ledgers up to. Goes back 30 days from this time.
        existing_perf_ledgers = {}
        return self.update_all_perf_ledgers(hotkey_to_positions, existing_perf_ledgers, t_ms)

    @timeme
    def update(self, testing_one_hotkey=None, regenerate_all_ledgers=False, t_ms=None):
        # Use PerfLedgerManager's own metagraph client (forward compatibility)
        assert self.metagraph, "Metagraph must be loaded before updating perf ledgers"
        perf_ledger_bundles = self.get_perf_ledgers(portfolio_only=False)
        if self.is_backtesting:
            if not t_ms:
                raise Exception("t_ms must be provided in backtesting mode")
            bt.logging.info(f'Updating perf ledgers for backtesting at time {TimeUtil.millis_to_formatted_date_str(t_ms)}')
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis() - self.UPDATE_LOOKBACK_MS

        hotkey_to_positions, hotkeys_with_no_positions = self.get_positions_perf_ledger(testing_one_hotkey=testing_one_hotkey)

        def sort_key(x):
            # Highest priority. Want to rebuild this hotkey first in case it has an incorrect dd from a Polygon bug
            #if x == "5Et6DsfKyfe2PBziKo48XNsTCWst92q8xWLdcFy6hig427qH":
            #    return float('inf')
            # Otherwise, sort by the last trade time
            return hotkey_to_positions[x][-1].orders[-1].processed_ms

        # Sort the keys with the custom sort key
        hotkeys_ordered_by_last_trade = sorted(hotkey_to_positions.keys(), key=sort_key, reverse=True)

        # Remove keys from perf ledgers if they aren't inx the metagraph anymore
        metagraph_hotkeys = set(self._metagraph_client.get_hotkeys())
        hotkeys_to_delete = set([x for x in hotkeys_with_no_positions if x in perf_ledger_bundles])
        rss_modified = False
        # Recently re-registered
        hotkeys_rrr = []
        deltas = []
        n_valid_times = 0
        total_n_times = 0
        for hotkey in hotkey_to_positions:
            corresponding_ledger_bundle = perf_ledger_bundles.get(hotkey)
            if corresponding_ledger_bundle is None:
                continue
            portfolio_ledger = corresponding_ledger_bundle[TP_ID_PORTFOLIO]
            first_order_time_ms = min(p.orders[0].processed_ms for p in hotkey_to_positions[hotkey])
            total_n_times += 1
            if portfolio_ledger.initialization_time_ms != first_order_time_ms:
                hotkeys_rrr.append(hotkey)
                deltas.append(portfolio_ledger.initialization_time_ms - first_order_time_ms)
            else:
                n_valid_times += 1

        if hotkeys_rrr:
            bt.logging.warning(f'Removing recently re-registered hotkeys from perf ledgers. n_valid_times {n_valid_times} total_n_times {total_n_times}. pct valid {n_valid_times / total_n_times * 100:.2f}%')
            for x in list(zip(hotkeys_rrr, deltas)):
                bt.logging.warning(x)
            hotkeys_to_delete.update(hotkeys_rrr)

        # Determine which hotkeys to remove from the perf ledger
        hotkeys_to_iterate = [x for x in hotkeys_ordered_by_last_trade if x in perf_ledger_bundles]
        for k in perf_ledger_bundles.keys():  # Some hotkeys may not be in the positions (old, bugged, etc.)
            if k not in hotkeys_to_iterate:
                hotkeys_to_iterate.append(k)

        for hotkey in hotkeys_to_iterate:
            if hotkey not in metagraph_hotkeys:
                hotkeys_to_delete.add(hotkey)
            elif not len(hotkey_to_positions.get(hotkey, [])):
                hotkeys_to_delete.add(hotkey)
            elif self.enable_rss and not rss_modified and hotkey not in self.random_security_screenings:
                rss_modified = True
                self.random_security_screenings.add(hotkey)
                hotkeys_to_delete.add(hotkey)

        # Start over again
        if not rss_modified:
            self.random_security_screenings = set()

        # Regenerate checkpoints if a hotkey was modified during position sync
        self.hks_attempting_invalidations = list(self.perf_ledger_hks_to_invalidate.keys())
        if self.hks_attempting_invalidations:
            for hk, t in self.perf_ledger_hks_to_invalidate.items():
                hotkeys_to_delete.add(hk)
                bt.logging.info(f"perf ledger marked for full rebuild for hk {hk} due to position sync at time {t}")

        for k in hotkeys_to_delete:
            if k in perf_ledger_bundles:
                del perf_ledger_bundles[k]

        self.hk_to_last_order_processed_ms = {k: v for k, v in self.hk_to_last_order_processed_ms.items() if k in perf_ledger_bundles}

        #hk_to_last_update_date = {k: TimeUtil.millis_to_formatted_date_str(v.last_update_ms)
        #                            if v.last_update_ms else 'N/A' for k, v in perf_ledgers.items()}

        bt.logging.info(f"perf ledger PLM hotkeys to delete: {hotkeys_to_delete}. rss: {self.random_security_screenings}")

        if regenerate_all_ledgers or testing_one_hotkey:
            bt.logging.info("Regenerating all perf ledgers")
            for k in list(perf_ledger_bundles.keys()):
                del perf_ledger_bundles[k]
        try:
            self.restore_out_of_sync_ledgers(perf_ledger_bundles, hotkey_to_positions)
            if regenerate_all_ledgers or testing_one_hotkey:
                bt.logging.info(f"  After restore_out_of_sync_ledgers: {len(perf_ledger_bundles)} ledgers")
        except Exception as e:
            bt.logging.warning(f"Couldn't restore out of sync ledgers: {e}. Continuing...")
            bt.logging.warning(traceback.format_exc())

        # Time in the past to start updating the perf ledgers
        self.update_all_perf_ledgers(hotkey_to_positions, perf_ledger_bundles, t_ms)

        # Clear invalidations after successful update. Prevent race condition by only clearing if we attempted invalidation for specific hk
        if self.hks_attempting_invalidations:
            for x in self.hks_attempting_invalidations:
                if x in self.perf_ledger_hks_to_invalidate:
                    del self.perf_ledger_hks_to_invalidate[x]

        if testing_one_hotkey and not self.running_unit_tests:
            self.debug_pl_plot(testing_one_hotkey)

    def save_perf_ledgers_to_disk(self, perf_ledgers: dict[str, dict[str, PerfLedger]] | dict[str, dict[str, dict]], raw_json=False):
        file_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)

        # Convert PerfLedger objects to dictionaries for JSON serialization
        serializable_ledgers = {}
        for hotkey, bundle in perf_ledgers.items():
            if isinstance(bundle, dict):
                serializable_ledgers[hotkey] = {}
                for trade_pair_id, ledger in bundle.items():
                    if isinstance(ledger, PerfLedger):
                        serializable_ledgers[hotkey][trade_pair_id] = ledger.to_dict()
                    else:
                        # Already a dict
                        serializable_ledgers[hotkey][trade_pair_id] = ledger
            elif isinstance(bundle, PerfLedger):
                # V1 format - single PerfLedger (portfolio only)
                serializable_ledgers[hotkey] = bundle.to_dict()
            else:
                # Already serialized
                serializable_ledgers[hotkey] = bundle

        ValiBkpUtils.write_compressed_json(file_path, serializable_ledgers)

    def debug_pl_plot(self, testing_one_hotkey):
        all_bundles = self.get_perf_ledgers(portfolio_only=False)
        bundle = all_bundles[testing_one_hotkey]
        portfolio_ledger = bundle[TP_ID_PORTFOLIO]
        # print all attributes except cps: Note ledger is an object
        print(f'Portfolio ledger attributes: initialization_time_ms {portfolio_ledger.initialization_time_ms},'
              f' max_return {portfolio_ledger.max_return}')
        from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
        daily_returns = LedgerUtils.daily_return_ratio_by_date(portfolio_ledger, return_type='simple')
        datetime_to_daily_return = {datetime.datetime.combine(k, datetime.time.min).timestamp(): v for k, v in
                                    daily_returns.items()}
        returns = []
        returns_muled = []
        times = []
        n_contributing_tps = []
        mdds = []
        for i, x in enumerate(portfolio_ledger.cps):
            returns.append(x.prev_portfolio_ret)
            foo = 1.0
            n_contributing = 0
            mdds.append(x.mdd)
            for tp_id, ledger in bundle.items():
                if tp_id == TP_ID_PORTFOLIO:
                    continue
                rele_cp = None
                for y in ledger.cps:
                    if y.last_update_ms == x.last_update_ms:
                        rele_cp = y
                        break
                if rele_cp:
                    n_contributing += 1
                    foo *= rele_cp.prev_portfolio_ret
            returns_muled.append(foo)
            n_contributing_tps.append(n_contributing)
            times.append(TimeUtil.millis_to_timestamp(x.last_update_ms))

            last_update_formated = TimeUtil.millis_to_timestamp(x.last_update_ms)
            # assert the checkpoint ends on a 12 hour boundary
            if i != len(portfolio_ledger.cps) - 1:
                assert x.last_update_ms % portfolio_ledger.target_cp_duration_ms == 0, x.last_update_ms
            print(x, last_update_formated)
        # Plot time vs return using matplotlib as well as time vs dd. use a legend.
        import matplotlib.pyplot as plt

        returns_debug = []
        times_debug = []

        for t in times:
            ts = datetime.datetime.combine(t.date(), datetime.time.min).timestamp()
            if ts in datetime_to_daily_return:
                returns_debug.append(datetime_to_daily_return[ts])
                times_debug.append(t)

        # Make the plot bigger
        plt.figure(figsize=(10, 5))
        plt.plot(times, returns, color='red', label='Return')
        plt.plot(times, returns_muled, color='blue', label='Return_Mulled')
        plt.plot(times, mdds, color='green', label='MDD')
        # plt.plot(times_debug, returns_debug, color='orange', label='Daily Return Debug')
        # Labels
        plt.xlabel('Time')
        plt.title(f'Return vs Time for HK {testing_one_hotkey}')
        plt.legend(['Return', 'Return_Mulled', 'MDD', 'Daily Return Debug'])
        plt.show()

        for tp_id, pl in bundle.items():
            first_cp_time = TimeUtil.millis_to_formatted_date_str(pl.cps[0].last_update_ms) if pl.cps else 'N/A'
            last_cp_time = TimeUtil.millis_to_formatted_date_str(pl.cps[-1].last_update_ms) if pl.cps else 'N/A'
            print(
                f"perf ledger for {tp_id} ({first_cp_time} -> {last_cp_time})\n  first cp {pl.cps[0]}\n  last cp {pl.cps[-1]}")
            print('    total gain product', pl.get_product_of_gains(), ' total loss product', pl.get_product_of_loss(),
                  'total product', pl.get_total_product())

        print('validating returns:')
        for z in zip(returns, returns_muled, n_contributing_tps):
            print(z, z[0] - z[1])

    @timeme
    def save_perf_ledgers(self, perf_ledgers_copy: dict[str, dict[str, PerfLedger]] | dict[str, dict[str, dict]], raw_json=False):
        # We may have items in perf_ledger_hks_to_invalidate added after the iteration began.
        # Let's nuke them to allow freed hotkeys to escape elimination.
        for hk, t in self.perf_ledger_hks_to_invalidate.items():
            if hk not in self.hks_attempting_invalidations:
                bt.logging.warning(f"perf ledger invalidated for hk {hk} during update dat {self.perf_ledger_hks_to_invalidate[hk]}. Removing from perf ledgers.")
                del perf_ledgers_copy[hk]

        if not self.is_backtesting:
            self.save_perf_ledgers_to_disk(perf_ledgers_copy, raw_json=raw_json)

        for k in list(self.hotkey_to_perf_bundle.keys()):
            if k not in perf_ledgers_copy:
                del self.hotkey_to_perf_bundle[k]

        for k, v in perf_ledgers_copy.items():
            self.hotkey_to_perf_bundle[k] = v

    def restore_out_of_sync_ledgers(self, existing_bundles, hotkey_to_positions):
        # TODO: Write tests
        """
        Restore ledgers subject to race condition. Perf ledger fully update loop can take 30 min.
        An order can come in during update.

        We can only build perf ledgers between orders or after all orders
        """
        for hk, bundle in existing_bundles.items():
            last_acked_order_time_ms = self.hk_to_last_order_processed_ms.get(hk)
            if not last_acked_order_time_ms:
                continue
            ledger_last_update_time = bundle[TP_ID_PORTFOLIO].last_update_ms
            positions = hotkey_to_positions.get(hk)
            if positions is None:
                continue
            smallest_conflict_time_ms = float('inf')
            for p in positions:
                for o in p.orders:
                    # An order came in while the perf ledger was being updated. Trim the checkpoints to avoid a race condition.
                    if last_acked_order_time_ms < o.processed_ms < ledger_last_update_time:
                        smallest_conflict_time_ms = min(smallest_conflict_time_ms, o.processed_ms)
            if smallest_conflict_time_ms != float('inf'):
                order_time_str = TimeUtil.millis_to_formatted_date_str(smallest_conflict_time_ms)
                last_acked_time_str = TimeUtil.millis_to_formatted_date_str(last_acked_order_time_ms)
                ledger_last_update_time_str = TimeUtil.millis_to_formatted_date_str(ledger_last_update_time)
                bt.logging.info(f"Recovering checkpoints for {hk}. Order came in at {order_time_str} after last acked time {last_acked_time_str} but before perf ledger update time {ledger_last_update_time_str}")
                for tp_id, pl in bundle.items():
                    pl.trim_checkpoints(smallest_conflict_time_ms)
                    if len(pl.cps) == 0:
                        pl.max_return = 1.0

    def update_one_perf_ledger_parallel(self, data_tuple):
        t0 = time.time()
        hotkey_i, n_hotkeys, hotkey, positions, existing_bundle, now_ms, is_backtesting = data_tuple
        # Create a temporary manager for processing
        # This is to avoid sharing state between executors
        worker_plm = PerfLedgerManager(
            parallel_mode=self.parallel_mode,
            enable_rss=False,  # full rebuilds not necessary as we are building from scratch already
            secrets=self.secrets,
            build_portfolio_ledgers_only=self.build_portfolio_ledgers_only,
            target_ledger_window_ms=self.target_ledger_window_ms,
            is_backtesting=is_backtesting,
            use_slippage=self.use_slippage,
            running_unit_tests=self.running_unit_tests,
        )
        worker_plm.now_ms = now_ms

        new_bundle = worker_plm.update_one_perf_ledger_bundle(
            hotkey_i, n_hotkeys, hotkey, positions, now_ms, {hotkey:existing_bundle}
        )
        last_update_time_ms = existing_bundle[TP_ID_PORTFOLIO].last_update_ms if existing_bundle else new_bundle[TP_ID_PORTFOLIO].initialization_time_ms
        portfolio_pl = new_bundle[TP_ID_PORTFOLIO]
        pl_start_time = TimeUtil.millis_to_formatted_date_str(last_update_time_ms)
        pl_end_time = TimeUtil.millis_to_formatted_date_str(portfolio_pl.last_update_ms)

        bt.logging.success(f'Completed update_one_perf_ledger_parallel for {hotkey} in {time.time() - t0} s over '
              f'{pl_start_time} to {pl_end_time}.')
        return hotkey, new_bundle

    def update_perf_ledgers_parallel(self, spark, pool, hotkey_to_positions: dict[str, List[Position]],
                                     existing_perf_ledgers: dict[str, dict[str, PerfLedger]],
                                     parallel_mode: ParallelizationMode = ParallelizationMode.PYSPARK,
                                     now_ms: int = None, top_n_miners: int=None,
                                     is_backtesting: bool = False) -> dict[str, dict[str, PerfLedger]]:
        """
        Update all perf ledgers in parallel using PySpark.

        Args:
            spark: PySpark SparkSession
            pool: Multiprocessing pool
            hotkey_to_positions: Dictionary mapping hotkeys to their positions
            existing_perf_ledgers: Dictionary of existing performance ledger bundles
            now_ms: Current time in milliseconds
            top_n_miners: Number of miners to process (local testing)

        Returns:
            Updated performance ledger bundles
        """
        t_init = time.time()

        if now_ms is None:
            now_ms = TimeUtil.now_in_millis()
        else:
            # CRITICAL BUG FIX: Validate now_ms to prevent future timestamp issues
            current_time_ms = TimeUtil.now_in_millis()
            if now_ms > current_time_ms + 86400000:  # More than 1 day in the future
                bt.logging.error(
                    f"CRITICAL TIMESTAMP ERROR: now_ms ({now_ms}) is {(now_ms - current_time_ms) / 86400000:.1f} days "
                    f"in the future compared to current time ({current_time_ms}). "
                    f"This will cause assertion failures. Using current time instead.")
                now_ms = current_time_ms
        self.now_ms = now_ms

        # Create a list of hotkeys with their positions for RDD
        hotkey_data = []
        for i, (hotkey, positions) in enumerate(hotkey_to_positions.items()):
            hotkey_data.append((i, len(hotkey_to_positions), hotkey, positions, existing_perf_ledgers.get(hotkey), now_ms, is_backtesting))
            if top_n_miners and i == top_n_miners - 1:
                break

        if parallel_mode == ParallelizationMode.PYSPARK:
            bt.logging.info(
                f"Updating perf ledgers in parallel with {self.parallel_mode.name}. RDD size: {len(hotkey_data)}")
            # Create RDD from hotkey data
            hotkey_rdd = spark.sparkContext.parallelize(hotkey_data)
            # Process all hotkeys in parallel
            updated_perf_ledgers = hotkey_rdd.map(self.update_one_perf_ledger_parallel).collectAsMap()
        elif parallel_mode == ParallelizationMode.MULTIPROCESSING:
            # Use multiprocessing for parallel processing
            updated_perf_ledgers = dict(pool.map(self.update_one_perf_ledger_parallel, hotkey_data))
        else:
            raise ValueError(f"Invalid parallel mode: {parallel_mode}")

        n_perf_ledgers = len(updated_perf_ledgers)
        n_hotkeys_with_positions = len(hotkey_to_positions)
        bt.logging.success(f"Done updating perf ledgers with {self.parallel_mode.name} in {time.time() - t_init}s. "
                           f"n_perf_ledgers: {n_perf_ledgers}, n_hotkeys_with_positions: {n_hotkeys_with_positions}")

        self.save_perf_ledgers(updated_perf_ledgers)
        return updated_perf_ledgers
