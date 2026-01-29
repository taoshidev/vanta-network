import asyncio
import threading
import time
import traceback
import bittensor as bt
import databento as db
from setproctitle import setproctitle

from data_generator.base_data_service import BaseDataService
from vali_objects.vali_config import TradePair, TradePairCategory
from vali_objects.vali_dataclasses.price_source import PriceSource

DATABENTO_PROVIDER_NAME = "Databento"


class DatabentoDataService(BaseDataService):
    """Equities-only live WebSocket feed from Databento using bbo-1s schema."""

    DATASET = "EQUS.MINI"
    SCHEMA = "bbo-1s"

    def __init__(self, api_key: str, disable_ws=False, running_unit_tests=False):
        super().__init__(
            DATABENTO_PROVIDER_NAME,
            running_unit_tests,
            enabled_websocket_categories={TradePairCategory.EQUITIES}
        )
        self._api_key = api_key
        # Map instrument_id -> symbol (populated from SymbolMappingMsg)
        self._instrument_map = {}

        # Start websocket manager thread
        if disable_ws:
            self.websocket_manager_thread = None
        else:
            self.websocket_manager_thread = threading.Thread(target=self.websocket_manager, daemon=True)
            self.websocket_manager_thread.start()

    def _get_equity_symbols(self) -> list[str]:
        """Get all equity symbols from TradePair config."""
        symbols = []
        for tp in TradePair:
            if tp.is_equities and tp not in self.UNSUPPORTED_TRADE_PAIRS:
                symbols.append(tp.trade_pair)
        return symbols

    def _create_websocket_client(self, tpc: TradePairCategory):
        """Create Databento Live client for equities."""
        if tpc != TradePairCategory.EQUITIES:
            return

        client = db.Live(key=self._api_key)
        self.WEBSOCKET_OBJECTS[tpc] = client
        bt.logging.info(f"Created {self.provider_name} Live client for {tpc}")

    def _subscribe_websockets(self, tpc: TradePairCategory):
        """Subscribe to all equity symbols with bbo-1s schema."""
        if tpc != TradePairCategory.EQUITIES:
            return

        symbols = self._get_equity_symbols()
        if not symbols:
            bt.logging.warning("No equity symbols to subscribe to")
            return

        client = self.WEBSOCKET_OBJECTS.get(tpc)
        if client is None:
            bt.logging.error(f"No client available for {tpc}")
            return

        try:
            client.subscribe(
                dataset=self.DATASET,
                schema=self.SCHEMA,
                symbols=symbols,
            )
            bt.logging.info(
                f"{self.provider_name} subscribed to {len(symbols)} symbols: {symbols}"
            )
        except db.BentoError as e:
            bt.logging.error(f"{self.provider_name} subscription failed: {e}")
            self.WEBSOCKET_OBJECTS[tpc] = None
            raise

    async def handle_msg(self, msg):
        """Convert Databento BBO message to PriceSource and update state."""

        # Capture symbol mappings
        if isinstance(msg, db.SymbolMappingMsg):
            self._instrument_map[msg.instrument_id] = msg.stype_in_symbol
            return

        # Skip non-BBO messages
        if not isinstance(msg, db.BBOMsg):
            return

        # Resolve instrument_id to symbol
        instrument_id = msg.instrument_id
        symbol = self._instrument_map.get(instrument_id)
        if symbol is None:
            return

        tp = self.trade_pair_lookup.get(symbol)
        if tp is None:
            return

        # Convert nanoseconds to milliseconds
        timestamp_ms = msg.ts_event // 1_000_000

        # Get bid/ask from first level (prices are in fixed-point, divide by 1e9)
        bid = msg.levels[0].bid_px / 1e9
        ask = msg.levels[0].ask_px / 1e9
        mid = (bid + ask) / 2

        ps = PriceSource(
            source=f"{DATABENTO_PROVIDER_NAME}_ws",
            timespan_ms=0,  # Point-in-time for websocket
            open=mid,
            close=mid,
            vwap=mid,
            high=mid,
            low=mid,
            start_ms=timestamp_ms,
            websocket=True,
            lag_ms=0,
            bid=bid,
            ask=ask,
        )

        # bt.logging.info(f"DATABENTO WEBSOCKET MESSAGE: {tp.trade_pair} | bid: {bid}, ask: {ask}")

        # Update state
        self.latest_websocket_events[symbol] = ps
        self.trade_pair_to_recent_events[symbol].add_event(ps)
        self.tpc_to_n_events[TradePairCategory.EQUITIES] += 1
        self.tpc_to_last_event_time[TradePairCategory.EQUITIES] = time.time()

        # Reset closed market price
        self.closed_market_prices[tp] = None

    async def _cleanup_websocket(self, tpc: TradePairCategory):
        """Clean up Databento websocket resources."""
        client = self.WEBSOCKET_OBJECTS.get(tpc)
        if client:
            try:
                client.stop()
                bt.logging.info(f"Cleaned up {self.provider_name}[{tpc}] websocket")
            except Exception as e:
                bt.logging.error(f"Cleanup error for {tpc}: {e}")
            finally:
                self.WEBSOCKET_OBJECTS[tpc] = None

    def websocket_manager(self):
        """
        Override base class to use Databento's async iteration pattern instead of connect().
        """
        setproctitle(f"vali_ws_{self.provider_name}")
        bt.logging.enable_info()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.websocket_tasks = {}
        self._websocket_loop = loop

        async def run_websocket(category):
            while True:
                try:
                    self._create_websocket_client(category)
                    self._subscribe_websockets(category)
                    client = self.WEBSOCKET_OBJECTS.get(category)

                    if client is None:
                        bt.logging.warning(f"{self.provider_name}[{category}] client not created, retrying")
                        await asyncio.sleep(5)
                        continue

                    bt.logging.info(f"Starting {self.provider_name} async iteration for {category}")

                    # Databento uses async iteration instead of connect()
                    async for msg in client:
                        await self.handle_msg(msg)

                    bt.logging.warning(f"{self.provider_name}[{category}] iteration ended, restarting")

                except asyncio.CancelledError:
                    bt.logging.info(f"{self.provider_name}[{category}] websocket task cancelled")
                    break
                except Exception as e:
                    bt.logging.error(f"{self.provider_name}[{category}] websocket error: {e}")
                    bt.logging.error(traceback.format_exc())

                # Clean up before reconnecting
                try:
                    await self._cleanup_websocket(category)
                    await asyncio.sleep(2)
                except Exception as e:
                    bt.logging.error(f"Error during websocket cleanup for {category}: {e}")
                    await asyncio.sleep(5)

        self._run_websocket = run_websocket

        async def health_check():
            self.task_locks = {tpc: asyncio.Lock() for tpc in self.enabled_websocket_categories}
            self.restart_backoff = {tpc: 1.0 for tpc in self.enabled_websocket_categories}
            self.last_restart_time = {tpc: 0 for tpc in self.enabled_websocket_categories}

            last_debug = time.time()

            while True:
                try:
                    now = time.time()

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

                await asyncio.sleep(5)

        # Create tasks for each websocket category
        tasks = []
        for tpc in self.enabled_websocket_categories:
            task = loop.create_task(run_websocket(tpc))
            self.websocket_tasks[tpc] = task
            tasks.append(task)

        # Add health check task
        health_task = loop.create_task(health_check())
        tasks.append(health_task)

        try:
            loop.run_until_complete(asyncio.gather(*tasks))
        except Exception as e:
            bt.logging.error(f"Main event loop error: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            try:
                for task in tasks:
                    task.cancel()
                loop.close()
            except Exception as e:
                bt.logging.error(f"Error during shutdown: {e}")

    def instantiate_not_pickleable_objects(self):
        """Initialize non-pickleable clients after unpickling."""
        # Live client will be created in _create_websocket_client
        pass


if __name__ == "__main__":
    import asyncio
    from vali_objects.utils.vali_utils import ValiUtils

    secrets = ValiUtils.get_secrets()
    api_key = secrets.get("databento_apikey")

    if not api_key:
        print("Error: databento_apikey not found in secrets")
        exit(1)

    print(f"Creating DatabentoDataService...")
    # Use disable_ws=True to prevent background thread from starting
    service = DatabentoDataService(api_key=api_key, disable_ws=True)

    symbols = service._get_equity_symbols()
    print(f"Equity symbols ({len(symbols)}): {symbols}")

    # Create client and subscribe directly (not through service methods to avoid state issues)
    print(f"\nConnecting to {service.DATASET} with schema {service.SCHEMA}...")
    client = db.Live(key=api_key)
    client.subscribe(
        dataset=service.DATASET,
        schema=service.SCHEMA,
        symbols=symbols,
    )
    print(f"Subscribed to {len(symbols)} symbols")

    # Run the websocket
    async def run():
        msg_count = 0
        symbol_map = {}  # instrument_id -> symbol

        async for msg in client:
            # Capture symbol mappings
            if isinstance(msg, db.SymbolMappingMsg):
                # Print first one to see attributes
                if not symbol_map:
                    print(f"SymbolMappingMsg attrs: {[a for a in dir(msg) if not a.startswith('_')]}")
                symbol_map[msg.instrument_id] = msg.stype_in_symbol
                print(f"Mapped {msg.instrument_id} -> {msg.stype_in_symbol}")
                continue

            if not isinstance(msg, db.BBOMsg):
                continue

            msg_count += 1
            instrument_id = msg.instrument_id
            symbol = symbol_map.get(instrument_id, f"unknown:{instrument_id}")
            bid = msg.levels[0].bid_px / 1e9
            ask = msg.levels[0].ask_px / 1e9
            price = msg.price / 1e9
            print(f"[{msg_count}] {symbol}: price={price:.2f} bid={bid:.2f} ask={ask:.2f}")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Close the client connection to free up the connection slot
        if client:
            try:
                client.stop()
                print("Client connection closed")
            except Exception as e:
                print(f"Error closing client: {e}")
