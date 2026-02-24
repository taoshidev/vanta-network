"""
HyperliquidFillService — Orchestrator that manages WebSocket connections to
Hyperliquid for all registered HL miners and injects their fills as Vanta orders.

Lifecycle:
  1. start() spawns a daemon thread running an asyncio event loop.
  2. Every 60s, _reconcile_connections() reads the current set of HL miners
     from the AssetSelectionClient and adds/removes WebSocket tasks accordingly.
  3. Each WS task calls subscribe_to_user_fills() which reconnects automatically.
  4. Incoming fills are translated via HyperliquidFill and passed to
     MarketOrderManager._process_market_order() with price_sources pre-populated
     so that Polygon/Tiingo price fetching is skipped.
"""
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import bittensor as bt

from vali_objects.utils.hyperliquid.hl_fill_translator import HyperliquidFill
from vali_objects.utils.hyperliquid.hl_websocket_client import subscribe_to_user_fills
from vali_objects.vali_config import ValiConfig
from time_util.time_util import TimeUtil

logger = logging.getLogger(__name__)


class HyperliquidFillService:
    """Manages HL WebSocket subscriptions and injects fills into the order pipeline."""

    def __init__(self, market_order_manager, asset_selection_client):
        self._mom = market_order_manager
        self._asc = asset_selection_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: Optional[asyncio.Event] = None

        # hotkey -> asyncio.Task for active WS subscriptions
        self._ws_tasks: Dict[str, asyncio.Task] = {}

        # Deduplication set: stores fill dedup_ids we've already processed
        self._seen_fills: set = set()
        self._seen_fills_lock = threading.Lock()

        # Thread pool for calling sync methods from async context
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hl_fill")

    def start(self):
        """Spawn a daemon thread running the async event loop."""
        thread = threading.Thread(target=self._run_loop, daemon=True, name="HLFillService")
        thread.start()
        bt.logging.info("[HL_FILL_SVC] Started HyperliquidFillService daemon thread")

    def _run_loop(self):
        """Entry point for the daemon thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._shutdown_event = asyncio.Event()
        try:
            self._loop.run_until_complete(self._main())
        except Exception:
            logger.exception("[HL_FILL_SVC] Event loop crashed")
        finally:
            self._loop.close()

    async def _main(self):
        """Main async loop: reconcile connections periodically."""
        while not self._shutdown_event.is_set():
            try:
                await self._reconcile_connections()
            except Exception:
                logger.exception("[HL_FILL_SVC] Error in reconcile loop")
            # Wait 60s or until shutdown
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=60)
                break  # shutdown signalled
            except asyncio.TimeoutError:
                pass  # normal timeout, loop again

    async def _reconcile_connections(self):
        """
        Read the current HL miners from AssetSelectionClient and ensure we have
        exactly one WS task per miner, capped at HL_MAX_UNIQUE_USERS.
        """
        try:
            hl_miners = self._asc.get_all_hl_miners()
        except Exception:
            logger.exception("[HL_FILL_SVC] Failed to get HL miners")
            return

        if not hl_miners:
            return

        desired_hotkeys = set(list(hl_miners.keys())[:ValiConfig.HL_MAX_UNIQUE_USERS])
        current_hotkeys = set(self._ws_tasks.keys())

        # Remove tasks for miners no longer registered
        for hotkey in current_hotkeys - desired_hotkeys:
            task = self._ws_tasks.pop(hotkey)
            task.cancel()
            bt.logging.info(f"[HL_FILL_SVC] Cancelled WS task for {hotkey[:10]}...")

        # Add tasks for new miners
        for hotkey in desired_hotkeys - current_hotkeys:
            wallet_addr = hl_miners[hotkey]
            task = asyncio.ensure_future(
                subscribe_to_user_fills(
                    hl_wallet_address=wallet_addr,
                    on_fill=lambda fill, hk=hotkey: self._on_fill(hk, fill),
                    shutdown_event=self._shutdown_event,
                )
            )
            self._ws_tasks[hotkey] = task
            bt.logging.info(f"[HL_FILL_SVC] Started WS task for {hotkey[:10]}... -> {wallet_addr[:10]}...")

        # Clean up completed/failed tasks (they will be re-created next reconcile)
        for hotkey in list(self._ws_tasks.keys()):
            task = self._ws_tasks[hotkey]
            if task.done():
                exc = task.exception() if not task.cancelled() else None
                if exc:
                    logger.warning(f"[HL_FILL_SVC] WS task for {hotkey[:10]}... failed: {exc}")
                del self._ws_tasks[hotkey]

    def _on_fill(self, miner_hotkey: str, raw_fill: dict):
        """
        Callback invoked for each HL fill. Runs in the async event loop thread,
        so we dispatch the heavy sync work to the thread pool.
        """
        try:
            fill = HyperliquidFill.from_ws_data(raw_fill)
        except (KeyError, ValueError) as e:
            logger.warning(f"[HL_FILL_SVC] Failed to parse fill for {miner_hotkey[:10]}...: {e}")
            return

        # Deduplication
        dedup_id = fill.dedup_id
        with self._seen_fills_lock:
            if dedup_id in self._seen_fills:
                bt.logging.debug(f"[HL_FILL_SVC] Duplicate fill {dedup_id}, skipping")
                return
            self._seen_fills.add(dedup_id)

        # Validate fill has a supported trade pair
        if fill.trade_pair is None:
            bt.logging.warning(f"[HL_FILL_SVC] Unsupported coin {fill.coin} from {miner_hotkey[:10]}...")
            return

        if fill.order_type is None:
            bt.logging.warning(f"[HL_FILL_SVC] Unknown direction {fill.dir} from {miner_hotkey[:10]}...")
            return

        # Dispatch to thread pool (MarketOrderManager is sync)
        self._executor.submit(self._process_fill, miner_hotkey, fill)

    def _process_fill(self, miner_hotkey: str, fill: HyperliquidFill):
        """Process a single HL fill by injecting it into the Vanta order pipeline."""
        try:
            trade_pair = fill.trade_pair
            price_source = fill.to_price_source()
            signal = fill.to_signal_dict()
            now_ms = TimeUtil.now_in_millis()

            bt.logging.info(
                f"[HL_FILL_SVC] Processing fill: {miner_hotkey[:10]}... "
                f"{fill.dir} {fill.sz} {fill.coin} @ {fill.price}"
            )

            err_msg, position, order = self._mom._process_market_order(
                miner_order_uuid=fill.dedup_id,
                miner_repo_version="hl_fill",
                trade_pair=trade_pair,
                now_ms=now_ms,
                signal=signal,
                miner_hotkey=miner_hotkey,
                price_sources=[price_source],
                enforce_market_cooldown=False,
            )

            if err_msg:
                bt.logging.warning(
                    f"[HL_FILL_SVC] Order rejected for {miner_hotkey[:10]}...: {err_msg}"
                )
            else:
                bt.logging.success(
                    f"[HL_FILL_SVC] Fill processed for {miner_hotkey[:10]}... "
                    f"{fill.dir} {fill.sz} {fill.coin} @ {fill.price}"
                )

        except Exception:
            logger.exception(
                f"[HL_FILL_SVC] Error processing fill for {miner_hotkey[:10]}..."
            )

    def stop(self):
        """Signal shutdown and clean up."""
        if self._shutdown_event:
            self._shutdown_event.set()
        self._executor.shutdown(wait=False)
        bt.logging.info("[HL_FILL_SVC] Shutdown signalled")
