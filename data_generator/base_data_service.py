import asyncio
import json
import threading
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List

import bittensor as bt
from polygon import WebSocketClient
from setproctitle import setproctitle
from tiingo import TiingoWebsocketClient

from shared_objects.error_utils import ErrorUtils
from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_objects.vali_config import TradePair, TradePairCategory, ValiConfig
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker
from vali_objects.vali_dataclasses.price_source import PriceSource

POLYGON_PROVIDER_NAME = "Polygon"
TIINGO_PROVIDER_NAME = "Tiingo"

def exception_handler_decorator():
    """
    Decorator to handle exceptions, log them, and return a default value.

    Uses a global logger (bt.logging.error) for logging.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_name = func.__name__
                thread_id = threading.get_native_id()
                bt.logging.error(f"Failed to get {func_name} with error: {e}, type: {type(e).__name__} in thread {thread_id}")
                return {}

        return wrapper

    return decorator

class BaseDataService():
    def __init__(self, provider_name, running_unit_tests=False):
        self.DEBUG_LOG_INTERVAL_S = 180
        self.MAX_TIME_NO_EVENTS_S = 120
        self.enabled_websocket_categories = {TradePairCategory.CRYPTO,
                                            TradePairCategory.FOREX}  # Exclude EQUITIES for now

        self.provider_name = provider_name
        self.running_unit_tests = running_unit_tests
        self.tpc_to_n_events = {x: 0 for x in self.enabled_websocket_categories}
        self.n_equity_events_skipped_afterhours = 0
        self.trade_pair_to_price_history = defaultdict(list)
        self.closed_market_prices = {tp: None for tp in TradePair}
        self.closed_market_prices_timestamp_ms = {tp: 0 for tp in TradePair}
        self.latest_websocket_events = {}
        self.n_flushes = 0
        self.websocket_manager_thread = None
        self.trade_pair_to_recent_events_realtime = defaultdict(RecentEventTracker)
        self.trade_pair_to_recent_events = defaultdict(RecentEventTracker)

        self.trade_pair_category_to_longest_allowed_lag_s = {tpc: 30 for tpc in TradePairCategory}
        self.timespan_to_ms = {'second': 1000, 'minute': 1000 * 60, 'hour': 1000 * 60 * 60, 'day': 1000 * 60 * 60 * 24}

        self.trade_pair_lookup = {pair.trade_pair: pair for pair in TradePair}
        self.trade_pair_to_longest_seen_lag_s = {}
        self.market_calendar = UnifiedMarketCalendar()
        
        # Unified websocket management
        self.websocket_tasks = {}
        self.task_locks = {}  # Will be initialized in async context
        self.restart_backoff = {}
        self.last_restart_time = {}
        self.tpc_to_last_event_time = {t: 0 for t in self.enabled_websocket_categories}

        # Reference ValiConfig constant for backward compatibility
        self.UNSUPPORTED_TRADE_PAIRS = ValiConfig.UNSUPPORTED_TRADE_PAIRS

        for trade_pair in TradePair:
            assert trade_pair.trade_pair_category in self.trade_pair_category_to_longest_allowed_lag_s, \
                f"Trade pair {trade_pair} has no allowed lag time"

        # Initialize websocket objects (tasks are managed separately)
        self.WEBSOCKET_OBJECTS = {}
        for tpc in self.enabled_websocket_categories:
            self.WEBSOCKET_OBJECTS[tpc] = None

        # Test-only override for market open status
        self._test_market_open_override = None  # None = use real calendar, True/False = override all markets



    def get_close_rest(
            self,
            trade_pair: TradePair,
            timestamp_ms: int
    ) -> PriceSource | None:
        pass

    def is_market_open(self, trade_pair: TradePair, time_ms=None) -> bool:
        # Check test override first
        if self._test_market_open_override is not None:
            return self._test_market_open_override

        if time_ms is None:
            time_ms = TimeUtil.now_in_millis()
        return self.market_calendar.is_market_open(trade_pair, time_ms)

    @ErrorUtils.require_test_mode
    def set_test_market_open(self, is_open: bool) -> None:
        """
        Test-only method to override market open status.
        Only works when running_unit_tests=True for safety.
        """
        self._test_market_open_override = is_open

    @ErrorUtils.require_test_mode
    def clear_test_market_open(self) -> None:
        """
        Clear market open override and use real calendar.
        Only works when running_unit_tests=True for safety.
        """
        self._test_market_open_override = None

    def get_close(self, trade_pair: TradePair) -> PriceSource | None:
        event = self.get_websocket_event(trade_pair)
        if not event:
            bt.logging.info(
                f"Fetching REST close for trade pair {trade_pair.trade_pair} using {self.provider_name} REST: {trade_pair.trade_pair}")
            event = self.get_close_rest(trade_pair)
            bt.logging.info(f"Received {self.provider_name} REST data for {trade_pair.trade_pair}: {event}")
        else:
            bt.logging.info(f"Using {self.provider_name} websocket data for {trade_pair.trade_pair}")
        if not event:
            bt.logging.warning(
                f"Failed to get close for {trade_pair.trade_pair} using {self.provider_name} websocket or REST.")
        return event

    def get_first_trade_pair_in_category(self, tpc: TradePairCategory) -> TradePair:
        # Use generator expression for efficiency
        return next((x for x in TradePair if x.trade_pair_category == tpc), None)

    def check_flush(self):
        t0 = time.time() if self.n_flushes % 500 == 0 else 0
        # Get a list of keys to avoid dictionary changed size during iteration
        # By creating a static list of keys first, we avoid the dictionary size change issue
        keys = list(self.trade_pair_to_recent_events_realtime.keys())
        c = {}
        for k in keys:
            c[k] = deepcopy(self.trade_pair_to_recent_events_realtime[k])
        # Flush the recent events to shared memory in one go
        self.trade_pair_to_recent_events.update(c)
        self.n_flushes += 1
        if t0:
            t1 = time.time()
            bt.logging.info(
                f"Flushed recent {self.provider_name} events to shared memory in {t1 - t0:.2f} seconds, n_flushes {self.n_flushes}")

    def stop_threads(self):
        """
        Stop the threads that are running the websocket clients.
        """
        if self.websocket_manager_thread:
            bt.logging.info(f"Stopping {self.provider_name} websocket manager thread")
            self.websocket_manager_thread.join(timeout=1)
            bt.logging.info(f"Stopped {self.provider_name} websocket manager thread")

        # Websockets are now managed as asyncio tasks in the manager thread

    def websocket_manager(self):
        """
        This runs in a separate thread. It manages websockets using asyncio tasks
        instead of threads, which simplifies the event loop management.
        """
        setproctitle(f"vali_ws_{self.provider_name}")
        bt.logging.enable_info()

        # Create a single event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Store references to websocket tasks for monitoring
        self.websocket_tasks = {}
        
        # Store loop reference for restart operations
        self._websocket_loop = loop

        # Define the websocket coroutine for each category
        async def run_websocket(category):
            while True:
                try:
                    # Create a new client
                    self._create_websocket_client(category)
                    self._subscribe_websockets(category)
                    client = self.WEBSOCKET_OBJECTS.get(category)

                    if client:
                        bt.logging.info(f"Connecting {self.provider_name} websocket for {category}")
                        # Use await instead of run_until_complete since we're in a coroutine
                        await client.connect(self.handle_msg)
                        bt.logging.warning(f"{self.provider_name}[{category}] connection closed, restarting")
                    else:
                        bt.logging.warning(f"{self.provider_name}[{category}] client not created, retrying")
                        await asyncio.sleep(5)
                        continue

                except asyncio.CancelledError:
                    bt.logging.info(f"{self.provider_name}[{category}] websocket task cancelled")
                    break  # Exit the loop if task is cancelled
                except Exception as e:
                    bt.logging.error(f"{self.provider_name}[{category}] websocket error: {e}")
                    bt.logging.error(traceback.format_exc())

                # Clean up before reconnecting
                try:
                    await self._cleanup_websocket(category)
                    # Wait before reconnecting
                    await asyncio.sleep(2)
                except Exception as e:
                    bt.logging.error(f"Error during websocket cleanup for {category}: {e}")
                    await asyncio.sleep(5)  # Back off on errors
        
        # Store run_websocket as instance method for access in restart
        self._run_websocket = run_websocket

        async def health_check():
            """Unified health check that handles both task death and stale connections"""
            # Initialize per-TPC locks in async context
            self.task_locks = {tpc: asyncio.Lock() for tpc in TradePairCategory}
            self.restart_backoff = {tpc: 1.0 for tpc in TradePairCategory}
            self.last_restart_time = {tpc: 0 for tpc in TradePairCategory}
            
            last_debug = time.time()
            
            while True:
                try:
                    now = time.time()
                    
                    # Check health of each websocket
                    for tpc in self.enabled_websocket_categories:
                        await self._check_websocket_health(tpc, loop)

                    if now - last_debug > self.DEBUG_LOG_INTERVAL_S:
                        try:
                            self.debug_log()
                        except Exception as e:
                            bt.logging.error(f"debug_log() failed: {e}")
                        last_debug = now

                except Exception as e:
                    bt.logging.error(f"Error in health check: {e}")
                    bt.logging.error(traceback.format_exc())

                await asyncio.sleep(5)  # Check every 5 seconds

        # Create and store tasks for each websocket category
        tasks = []
        for tpc in self.enabled_websocket_categories:
            task = loop.create_task(run_websocket(tpc))
            self.websocket_tasks[tpc] = task
            tasks.append(task)

        # Add the health check task
        health_task = loop.create_task(health_check())
        tasks.append(health_task)

        # Run the event loop with all tasks
        try:
            loop.run_until_complete(asyncio.gather(*tasks))
        except Exception as e:
            bt.logging.error(f"Main event loop error: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            try:
                # Clean up any pending tasks
                for task in tasks:
                    task.cancel()

                # Close the event loop
                loop.close()
            except Exception as e:
                bt.logging.error(f"Error during shutdown: {e}")

    async def _check_websocket_health(self, tpc: TradePairCategory, loop):
        """Check and maintain health of a single websocket"""
        async with self.task_locks[tpc]:
            task = self.websocket_tasks.get(tpc)
            last_event_time = self.tpc_to_last_event_time.get(tpc, 0)
            now = time.time()
            
            # Market check first
            # Get a representative trade pair for the category
            trade_pair = self.get_first_trade_pair_in_category(tpc)
            if trade_pair and not self.is_market_open(trade_pair):
                if task and not task.done():
                    bt.logging.info(f"{self.provider_name}[{tpc}] market closed, stopping")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    self.websocket_tasks[tpc] = None
                return
            
            # Determine if restart needed
            needs_restart = False
            reason = None
            
            if task is None:
                needs_restart = True
                reason = "No task"
            elif task.done():
                needs_restart = True
                reason = "Task completed"
                try:
                    await task  # Get any exception
                except Exception as e:
                    reason = f"Task failed: {e}"
            elif last_event_time > 0 and now - last_event_time > self.MAX_TIME_NO_EVENTS_S:
                needs_restart = True
                reason = f"No events for {now - last_event_time:.1f}s"
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            if needs_restart:
                await self._restart_websocket(tpc, loop, reason)

    async def _restart_websocket(self, tpc: TradePairCategory, loop, reason: str):
        """Restart websocket with exponential backoff"""
        bt.logging.warning(f"{self.provider_name}[{tpc}] restarting: {reason}")
        
        # Apply backoff
        now = time.time()
        time_since_last = now - self.last_restart_time.get(tpc, 0)
        
        if time_since_last < 60:  # Recent restart
            backoff = self.restart_backoff[tpc]
            bt.logging.info(f"Applying {backoff:.1f}s backoff for {tpc}")
            await asyncio.sleep(backoff)
            self.restart_backoff[tpc] = min(backoff * 1.5, 30.0)
        else:
            self.restart_backoff[tpc] = 1.0
        
        self.last_restart_time[tpc] = now
        
        # Clean up old websocket
        await self._cleanup_websocket(tpc)
        
        # Start new task
        try:
            new_task = loop.create_task(self._run_websocket(tpc))
            self.websocket_tasks[tpc] = new_task
        except Exception as e:
            bt.logging.error(f"Failed to create task for {tpc}: {e}")
            self.websocket_tasks[tpc] = None

    async def _cleanup_websocket(self, tpc: TradePairCategory):
        """Clean up websocket resources"""
        client = self.WEBSOCKET_OBJECTS.get(tpc)
        if client:
            try:
                if hasattr(client, 'unsubscribe_all'):
                    client.unsubscribe_all()
                if hasattr(client, '_should_close'):
                    client._should_close = True
                if hasattr(client, 'close'):
                    await client.close()
                bt.logging.info(f"Cleaned up {self.provider_name}[{tpc}] websocket")
            except Exception as e:
                bt.logging.error(f"Cleanup error for {tpc}: {e}")
            finally:
                self.WEBSOCKET_OBJECTS[tpc] = None


    def _create_websocket_client(self, tpc):
        raise NotImplementedError

    def _subscribe_websockets(self, tpc):
        raise NotImplementedError

    async def handle_msg(self, msg):
        raise NotImplementedError

    def instantiate_not_pickleable_objects(self):
        raise NotImplementedError

    def get_closes_websocket(self, trade_pairs: List[TradePair], time_ms) -> dict[str: PriceSource]:
        events = {}
        for trade_pair in trade_pairs:
            symbol = trade_pair.trade_pair
            if symbol not in self.trade_pair_to_recent_events:
                continue

            # Get the closest aligned event
            symbol = trade_pair.trade_pair
            latest_event = self.trade_pair_to_recent_events[symbol].get_closest_event(time_ms)
            events[trade_pair] = latest_event

        return events

    def get_closes_rest(self, trade_pairs: List[TradePair], time_ms) -> dict[str: float]:
        pass

    def get_websocket_lag_for_trade_pair_s(self, tp: str, now_ms: int) -> float | None:
        cur_event = self.latest_websocket_events.get(tp)
        if cur_event:
            return (now_ms - cur_event.end_ms) / 1000.0
        return None

    def spill_price_history(self):
        # Write the price history to disk in a format that will let us plot it
        filename = f"price_history_{self.provider_name}.json"
        with open(filename, 'w') as f:
            json.dump(self.trade_pair_to_price_history, f)

    def debug_log(self):
        now_ms = TimeUtil.now_in_millis()
        trade_pairs_to_track = list(self.latest_websocket_events.keys())
        for tp in trade_pairs_to_track:
            lag_s = self.get_websocket_lag_for_trade_pair_s(tp, now_ms)
            if tp not in self.trade_pair_to_longest_seen_lag_s:
                self.trade_pair_to_longest_seen_lag_s[tp] = lag_s
            else:
                if lag_s > self.trade_pair_to_longest_seen_lag_s[tp]:
                    self.trade_pair_to_longest_seen_lag_s[tp] = lag_s
        # log how long it has been since the last ping
        formatted_lags = {tp: f"{lag:.2f}" for tp, lag in self.trade_pair_to_longest_seen_lag_s.items()}
        formatted_lags = {k: v for k, v in formatted_lags.items() if float(v) > 10}
        bt.logging.info(f"{self.provider_name} Worst lags seen: {formatted_lags}")
        # Log the last time since websocket ping
        now_ms = TimeUtil.now_in_millis()
        formatted_lags = {tp: f"{(now_ms - price_source.end_ms) / 1000.0:.2f}" for tp, price_source in
                          self.latest_websocket_events.items()}
        bt.logging.info(f"{self.provider_name} Current websocket lags (s): {formatted_lags}")
        # Log the prices
        formatted_prices = {}
        for tp, price_source in self.latest_websocket_events.items():
            if TradePair.get_latest_tade_pair_from_trade_pair_str(tp).is_forex:
                formatted_prices[tp] = f"({price_source.bid:.5f}/{price_source.ask:.5f})"
            else:
                formatted_prices[tp] = f"{price_source.close:.2f}"

        bt.logging.info(f"{self.provider_name} Latest websocket prices: {formatted_prices}")
        bt.logging.info(f'{self.provider_name} websocket n_events_global: {self.tpc_to_n_events}. n_equity_events_skipped_afterhours: {self.n_equity_events_skipped_afterhours}')

    def get_price_before_market_close(self, trade_pair: TradePair) -> float | None:
        pass

    def get_websocket_event(self, trade_pair: TradePair) -> PriceSource | None:
        symbol = trade_pair.trade_pair
        cur_event = self.latest_websocket_events.get(symbol)
        if not cur_event:
            return None

        timestamp_ms = cur_event.end_ms
        max_allowed_lag_s = self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]
        lag_s = time.time() - timestamp_ms / 1000.0
        is_stale = lag_s > max_allowed_lag_s
        if is_stale:
            bt.logging.info(f"Found stale TD websocket data for {trade_pair.trade_pair}. Lag_s: {lag_s} "
                            f"seconds. Max allowed lag for category: {max_allowed_lag_s} seconds. Ignoring this data.")
        return cur_event
