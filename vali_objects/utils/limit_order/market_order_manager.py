"""

Modularize the logic that was originally in validator.py. No IPC communication here.
"""
import time
import uuid
import threading

from vanta_api.websocket_notifier import WebSocketNotifierClient
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
import bittensor as bt

from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.vali_config import ValiConfig, TradePair, RPCConnectionMode
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_source_enum import OrderSource


class MarketOrderManager():
    def __init__(self, serve:bool, slack_notifier=None, running_unit_tests=False, connection_mode=RPCConnectionMode.RPC):
        self.serve = serve
        self.running_unit_tests = running_unit_tests

        # Use LOCAL mode for WebSocketNotifier in tests (server not started in test mode)
        ws_connection_mode = RPCConnectionMode.LOCAL if running_unit_tests else connection_mode
        self.websocket_notifier = WebSocketNotifierClient(connection_mode=ws_connection_mode, connect_immediately=False)
        # Create own ContractClient (forward compatibility - no parameter passing)
        from vali_objects.contract.contract_client import ContractClient
        self._contract_client = ContractClient(running_unit_tests=running_unit_tests, connection_mode=connection_mode)

        # Create own MinerAccountClient (source of truth for account sizes)
        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode)

        # Create own LivePriceFetcherClient (forward compatibility - no parameter passing)
        from vali_objects.price_fetcher import LivePriceFetcherClient
        self._live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests, connection_mode=connection_mode)

        # Create own PositionManagerClient (forward compatibility - no parameter passing)
        from vali_objects.position_management.position_manager_client import PositionManagerClient
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        # Create own PositionLockClient (forward compatibility - no parameter passing)
        from shared_objects.locks.position_lock_client import PositionLockClient
        self._position_lock_client = PositionLockClient(running_unit_tests=running_unit_tests)

        # PriceSlippageModel creates its own LivePriceFetcherClient internally
        self.price_slippage_model = PriceSlippageModel(running_unit_tests=running_unit_tests)

        # Cache to track last order time for each (miner_hotkey, trade_pair) combination
        self.last_order_time_cache = {}  # Key: (miner_hotkey, trade_pair_id), Value: last_order_time_ms

        # Start slippage feature refresher thread (disabled in tests)
        # This thread refreshes slippage features daily and pre-populates tomorrow's features
        if not running_unit_tests:
            self.slippage_refresher = PriceSlippageModel.FeatureRefresher(
                price_slippage_model=self.price_slippage_model,
                slack_notifier=slack_notifier
            )
            self.slippage_refresher_thread = threading.Thread(
                target=self.slippage_refresher.run_update_loop,
                daemon=True,
                name="SlippageRefresher"
            )
            self.slippage_refresher_thread.start()
            bt.logging.info("Slippage feature refresher thread started")
        else:
            self.slippage_refresher = None
            self.slippage_refresher_thread = None

    @property
    def live_price_fetcher(self):
        """Get live price fetcher client."""
        return self._live_price_client

    @property
    def position_manager(self):
        """Get position manager client."""
        return self._position_client

    @property
    def contract_manager(self):
        """Get contract client (forward compatibility - created internally)."""
        return self._contract_client

    def clear_order_cooldown_cache(self):
        """Clear the order cooldown cache. Used for testing."""
        if not self.running_unit_tests:
            raise Exception('clear_order_cooldown_cache can only be called in unit test mode')
        self.last_order_time_cache.clear()
        bt.logging.debug("Cleared market order cooldown cache")

    def _get_or_create_open_position_from_new_order(self, trade_pair: TradePair, order_type: OrderType, order_time_ms: int,
                                        miner_hotkey: str, miner_order_uuid: str, now_ms:int, price_sources, miner_repo_version, account_size):

        # Check if there's an existing open position for this specific trade pair (server-side filtered)
        existing_open_pos = self._position_client.get_open_position_for_trade_pair(
            miner_hotkey,
            trade_pair.trade_pair_id
        )
        if existing_open_pos:
            # If the position has too many orders, we need to close it out to make room.
            if len(existing_open_pos.orders) >= ValiConfig.MAX_ORDERS_PER_POSITION and order_type != OrderType.FLAT:
                bt.logging.info(
                    f"Miner [{miner_hotkey}] hit {ValiConfig.MAX_ORDERS_PER_POSITION} order limit. "
                    f"Automatically closing position for {trade_pair.trade_pair_id} "
                    f"with {len(existing_open_pos.orders)} orders to make room for new position."
                )
                force_close_order_time = now_ms - 1 # 2 orders for the same trade pair cannot have the same timestamp
                force_close_order_uuid = existing_open_pos.position_uuid[::-1] # uuid will stay the same across validators
                self._add_order_to_existing_position(existing_open_pos, trade_pair, OrderType.FLAT,
                                                     0.0, 0.0, 0.0, force_close_order_time, miner_hotkey,
                                                     price_sources, force_close_order_uuid, miner_repo_version,
                                                     OrderSource.MAX_ORDERS_PER_POSITION_CLOSE,
                                                     existing_open_pos.account_size)
                time.sleep(0.1)  # Put 100ms between two consecutive websocket writes for the same trade pair and hotkey. We need the new order to be seen after the FLAT.
            else:
                # If the position is closed, raise an exception. This can happen if the miner is eliminated in the main
                # loop thread.
                if existing_open_pos.is_closed_position:
                    raise SignalException(
                        f"miner [{miner_hotkey}] sent signal for "
                        f"closed position [{trade_pair}]")
                bt.logging.debug("adding to existing position")
                # Return existing open position (nominal path)
                return existing_open_pos


        # if the order is FLAT ignore (noop)
        if order_type == OrderType.FLAT:
            open_position = None
        else:
            if trade_pair.is_equities and order_type == OrderType.SHORT:
                raise SignalException(
                    f"Cannot open a SHORT position for equities. "
                    f"Miner [{miner_hotkey}] attempted to open a new position with SHORT order for [{trade_pair.trade_pair_id}]. "
                    f"Equities can only be opened with LONG orders."
                )

            # if a position doesn't exist, then make a new one
            open_position = Position(
                miner_hotkey=miner_hotkey,
                position_uuid=miner_order_uuid if miner_order_uuid else str(uuid.uuid4()),
                open_ms=order_time_ms,
                trade_pair=trade_pair,
                position_type=order_type,
                account_size=account_size
            )
        return open_position

    def _add_order_to_existing_position(self, existing_position: Position, trade_pair: TradePair, signal_order_type: OrderType,
                                        quantity: float, leverage: float, value: float, order_time_ms: int, miner_hotkey: str,
                                        price_sources, miner_order_uuid: str, miner_repo_version: str, src:OrderSource,
                                        account_size=None, usd_base_price=None, execution_type=ExecutionType.MARKET,
                                        fill_price=None, limit_price=None, stop_loss=None, take_profit=None, bracket_orders=None) -> Order:
        # Must be locked by caller
        step_start = TimeUtil.now_in_millis()

        best_price_source = price_sources[0]
        # Use fill_price if provided (for limit/bracket orders), otherwise use market price
        price = fill_price if fill_price else best_price_source.parse_appropriate_price(order_time_ms, trade_pair.is_forex, signal_order_type, existing_position)

        if existing_position.account_size <= 0:
            bt.logging.warning(
                f"Invalid account_size {existing_position.account_size} for position {existing_position.position_uuid}. "
                f"Using MIN_CAPITAL as fallback."
            )
            existing_position.account_size = ValiConfig.MIN_CAPITAL

        # Calculate USD conversions
        step_start = TimeUtil.now_in_millis()
        if usd_base_price is None:
            usd_base_price = self.live_price_fetcher.get_usd_base_conversion(trade_pair, order_time_ms, price, signal_order_type, existing_position)
            usd_conversion_ms = TimeUtil.now_in_millis() - step_start
            bt.logging.info(f"[ADD_ORDER_DETAIL] USD conversion calculation took {usd_conversion_ms}ms")

        step_start = TimeUtil.now_in_millis()
        net_portfolio_leverage = self.position_manager.calculate_net_portfolio_leverage(miner_hotkey)
        leverage_calc_ms = TimeUtil.now_in_millis() - step_start
        bt.logging.info(f"[ADD_ORDER_DETAIL] Net portfolio leverage calc took {leverage_calc_ms}ms")

        # Create order (margin_loan will be set after validation)
        order = Order(
            trade_pair=trade_pair,
            order_type=signal_order_type,
            quantity=quantity,
            value=value,
            leverage=leverage,
            price=price,
            processed_ms=order_time_ms,
            order_uuid=miner_order_uuid,
            price_sources=price_sources,
            bid=best_price_source.bid,
            ask=best_price_source.ask,
            src=src,
            limit_price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            execution_type=execution_type,
            usd_base_rate=usd_base_price,
            bracket_orders=bracket_orders,
        )
        bt.logging.info(f"[ORDER DETAIL] Using price source {price_sources}")

        # Override bid/ask with Databento quote for equities slippage calculation
        # if trade_pair.is_equities:
        #     db_bid, db_ask, _ = self._live_price_client.get_quote(trade_pair, order_time_ms)
        #     bt.logging.info(f"RECEIVED BID {db_bid} ASK {db_ask}")
        #     if db_bid and db_ask and db_bid > 0 and db_ask > 0:
        #         order.bid = db_bid
        #         order.ask = db_ask

        order_creation_ms = TimeUtil.now_in_millis() - step_start
        order.quote_usd_rate = self.live_price_fetcher.get_quote_usd_conversion(order, existing_position)
        bt.logging.info(f"[ADD_ORDER_DETAIL] Order object creation took {order_creation_ms}ms")

        # Refresh features - this may make expensive API calls on new day
        step_start = TimeUtil.now_in_millis()
        features_available = self.price_slippage_model.refresh_features_daily(
            time_ms=order_time_ms,
            allow_blocking=False  # Don't block order filling!
        )
        refresh_features_ms = TimeUtil.now_in_millis() - step_start

        if not features_available:
            bt.logging.error(
                f"[ADD_ORDER_DETAIL] ⚠️  Features not available for slippage calculation! "
                f"This will affect slippage accuracy."
            )

        if refresh_features_ms > 100:
            bt.logging.warning(
                f"[ADD_ORDER_DETAIL] ⚠️  refresh_features_daily took {refresh_features_ms}ms "
                f"(BLOCKING ORDER FILL)"
            )
        else:
            bt.logging.info(f"[ADD_ORDER_DETAIL] refresh_features_daily took {refresh_features_ms}ms")

        step_start = TimeUtil.now_in_millis()
        order.slippage = PriceSlippageModel.calculate_slippage(order.bid, order.ask, order, existing_position.account_size)
        slippage_calc_ms = TimeUtil.now_in_millis() - step_start
        bt.logging.info(f"[ADD_ORDER_DETAIL] Slippage calculation took {slippage_calc_ms}ms")

        # Validate order before processing cash balance (raises ValueError if invalid)
        existing_position.validate_order_size(order, net_portfolio_leverage)

        # Process cash balance after validation passes
        if order.order_type == existing_position.position_type:
            # Buy: pay value plus slippage cost (raises SignalException if invalid)
            order.margin_loan = self._miner_account_client.process_order_buy(miner_hotkey, abs(value) * (1 + order.slippage))
        else:
            # Sell: free capital_used and compound realized PNL to equity
            processed_qty = existing_position.net_quantity if order.order_type == OrderType.FLAT else quantity
            entry_value = abs(processed_qty) * trade_pair.lot_size * existing_position.average_entry_price * order.quote_usd_rate

            if existing_position.position_type == OrderType.SHORT:
                exit_price = order.price * (1 + order.slippage)
                order_realized_pnl = (existing_position.average_entry_price - exit_price) * abs(processed_qty) * trade_pair.lot_size * order.quote_usd_rate
            else:
                exit_price = order.price * (1 - order.slippage)
                order_realized_pnl = (exit_price - existing_position.average_entry_price) * abs(processed_qty) * trade_pair.lot_size * order.quote_usd_rate

            bt.logging.info(f"[ORDER] entry_value=${entry_value:.2f} realized_pnl=${order_realized_pnl:.2f}")

            loan_repaid = self._miner_account_client.process_order_sell(miner_hotkey, entry_value, order_realized_pnl, existing_position.margin_loan)
            # Store loan repayment as negative margin_loan so position.margin_loan sums correctly
            order.margin_loan = -loan_repaid

        step_start = TimeUtil.now_in_millis()
        existing_position.add_order(order, self.live_price_fetcher, net_portfolio_leverage, skip_validation=True)
        add_order_ms = TimeUtil.now_in_millis() - step_start
        bt.logging.info(f"[ADD_ORDER_DETAIL] Position.add_order() took {add_order_ms}ms")

        step_start = TimeUtil.now_in_millis()
        self.position_manager.save_miner_position(existing_position)
        save_position_ms = TimeUtil.now_in_millis() - step_start
        bt.logging.info(f"[ADD_ORDER_DETAIL] Save position to disk took {save_position_ms}ms")

        # Update cooldown cache after successful order processing
        self.last_order_time_cache[(miner_hotkey, trade_pair.trade_pair_id)] = order_time_ms
        # NOTE: UUID tracking happens in validator process, not here

        if self.serve:
            # Broadcast position update via RPC to WebSocket clients
            # Skip websocket messages for development hotkey
            step_start = TimeUtil.now_in_millis()
            success = self.websocket_notifier.broadcast_position_update(
                existing_position, miner_repo_version=miner_repo_version
            )
            websocket_ms = TimeUtil.now_in_millis() - step_start
            bt.logging.info(f"[ADD_ORDER_DETAIL] Websocket RPC broadcast took {websocket_ms}ms (success={success})")

        return order


    def enforce_order_cooldown(self, trade_pair_id, now_ms, miner_hotkey) -> str:
        """
        Enforce cooldown between orders for the same trade pair using an efficient cache.
        This method must be called within the position lock to prevent race conditions.
        """
        cache_key = (miner_hotkey, trade_pair_id)
        current_order_time_ms = now_ms

        # Get the last order time from cache
        cached_last_order_time = self.last_order_time_cache.get(cache_key, 0)
        msg = None
        if cached_last_order_time > 0:
            time_since_last_order_ms = current_order_time_ms - cached_last_order_time

            if time_since_last_order_ms < ValiConfig.ORDER_COOLDOWN_MS:
                previous_order_time = TimeUtil.millis_to_formatted_date_str(cached_last_order_time)
                current_time = TimeUtil.millis_to_formatted_date_str(current_order_time_ms)
                time_to_wait_in_s = (ValiConfig.ORDER_COOLDOWN_MS - time_since_last_order_ms) / 1000
                msg = (
                    f"Order for trade pair [{trade_pair_id}] was placed too soon after the previous order. "
                    f"Last order was placed at [{previous_order_time}] and current order was placed at [{current_time}]. "
                    f"Please wait {time_to_wait_in_s:.1f} seconds before placing another order."
                )

        return msg

    @staticmethod
    def parse_order_size(signal, usd_base_conversion, trade_pair, portfolio_value):
        """
        parses an order signal and calculates leverage, value, and quantity
        """
        leverage = signal.get("leverage")
        value = signal.get("value")
        quantity = signal.get("quantity")

        fields_set = [x is not None for x in (leverage, value, quantity)]
        if sum(fields_set) != 1:
            raise ValueError("Exactly one of 'leverage', 'value', or 'quantity' must be set")

        if quantity is not None:
            value = quantity * trade_pair.lot_size / usd_base_conversion
            leverage = value / portfolio_value
        elif leverage is not None:
            value = leverage * portfolio_value
            quantity = (value * usd_base_conversion) / trade_pair.lot_size
        elif value is not None:
            leverage = value / portfolio_value
            quantity = (value * usd_base_conversion) / trade_pair.lot_size

        return quantity, leverage, value

    def process_flat_all_order(self, order_uuid, miner_repo_version, miner_hotkey, now_ms):
        bt.logging.info(f"Processing FLAT_ALL order for miner [{miner_hotkey}]")

        # Get all open positions for this miner
        open_positions = self._position_client.get_positions_for_hotkeys([miner_hotkey], only_open_positions=True).get(miner_hotkey)

        if not open_positions:
            bt.logging.info(f"No open positions found for miner [{miner_hotkey}]")
            return {
                "positions_closed": 0,
                "positions_failed": 0,
                "failed_trade_pairs": []
            }

        total_positions = len(open_positions)
        cnt_positions_closed = 0
        cnt_positions_failed = 0
        failed_trade_pairs = []

        bt.logging.info(f"Found {total_positions} open positions for miner [{miner_hotkey}], closing all...")

        for i, position in enumerate(open_positions):
            trade_pair = position.trade_pair

            try:
                # Get price sources for this trade pair
                price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, now_ms)
                if not price_sources:
                    bt.logging.error(
                        f"[FLAT_ALL] No price sources available for position {i+1}/{total_positions} "
                        f"[{trade_pair.trade_pair_id}] for miner [{miner_hotkey}]"
                    )
                    cnt_positions_failed += 1
                    failed_trade_pairs.append(trade_pair.trade_pair_id)
                    continue

                # Acquire position lock before closing
                with self._position_lock_client.get_lock(miner_hotkey, trade_pair.trade_pair_id):
                    position_close_uuid = position.position_uuid[::-1]
                    position_close_time = now_ms - (total_positions - i - 1)

                    self._add_order_to_existing_position(
                        position, trade_pair, OrderType.FLAT,
                        0.0, 0.0, 0.0, position_close_time, miner_hotkey,
                        price_sources, position_close_uuid, miner_repo_version,
                        OrderSource.FLAT_ALL_CLOSE,
                        position.account_size
                    )

                    cnt_positions_closed += 1
                    bt.logging.info(
                        f"[FLAT_ALL] Closed position {i+1}/{total_positions} "
                        f"[{trade_pair.trade_pair_id}] for miner [{miner_hotkey}]"
                    )

                if i < total_positions - 1:
                    time.sleep(0.05)

            except Exception as e:
                bt.logging.error(
                    f"[FLAT_ALL] Failed to close position {i+1}/{total_positions} "
                    f"[{trade_pair.trade_pair_id}] for miner [{miner_hotkey}]: {str(e)}"
                )
                cnt_positions_failed += 1
                failed_trade_pairs.append(trade_pair.trade_pair_id)
                continue

        bt.logging.info(
            f"[FLAT_ALL] Completed for miner [{miner_hotkey}]: "
            f"{cnt_positions_closed}/{total_positions} positions closed, "
            f"{cnt_positions_failed} failed"
        )

        return {
            "positions_closed": cnt_positions_closed,
            "positions_failed": cnt_positions_failed,
            "failed_trade_pairs": failed_trade_pairs
        }

    def process_market_order(self, synapse, miner_order_uuid, miner_repo_version, trade_pair, now_ms, signal, miner_hotkey, price_sources=None):

        err_message, existing_position, created_order = self._process_market_order(miner_order_uuid, miner_repo_version, trade_pair,
                                                                    now_ms, signal, miner_hotkey, price_sources)
        if err_message:
            synapse.successfully_processed = False
            synapse.error_message = err_message
        if existing_position:
            synapse.order_json = existing_position.orders[-1].__str__()

        return created_order

    def _process_market_order(self, miner_order_uuid, miner_repo_version, trade_pair, now_ms, signal, miner_hotkey, price_sources, enforce_market_cooldown=True):
        # TIMING: Price fetching
        if price_sources is None:
            price_fetch_start = TimeUtil.now_in_millis()
            price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, now_ms)
            price_fetch_ms = TimeUtil.now_in_millis() - price_fetch_start
            bt.logging.info(f"[TIMING] Price fetching took {price_fetch_ms}ms")

        if not price_sources:
            raise SignalException(
                f"Ignoring order for [{miner_hotkey}] due to no live prices being found for trade_pair [{trade_pair}]. Please try again.")

        # TIMING: Extract signal data
        extract_start = TimeUtil.now_in_millis()
        signal_order_type = OrderType.from_string(signal["order_type"])
        execution_type = ExecutionType.from_string(signal.get("execution_type"))
        bracket_orders = signal.get("bracket_orders")
        extract_ms = TimeUtil.now_in_millis() - extract_start
        bt.logging.info(f"[TIMING] Extract signal data took {extract_ms}ms")

        # Multiple threads can run receive_signal at once. Don't allow two threads to trample each other.
        debug_lock_key = f"{miner_hotkey[:8]}.../{trade_pair.trade_pair_id}"

        # TIMING: Time from start to lock request
        time_to_lock_request = TimeUtil.now_in_millis() - now_ms
        bt.logging.info(f"[TIMING] Time from receive_signal start to lock request: {time_to_lock_request}ms")

        lock_request_time = TimeUtil.now_in_millis()
        bt.logging.info(f"[LOCK] Requesting position lock for {debug_lock_key}")
        err_msg = None
        existing_position = None
        with (self._position_lock_client.get_lock(miner_hotkey, trade_pair.trade_pair_id)):
            lock_acquired_time = TimeUtil.now_in_millis()
            lock_wait_ms = lock_acquired_time - lock_request_time
            bt.logging.info(f"[LOCK] Acquired lock for {debug_lock_key} after {lock_wait_ms}ms wait")

            # TIMING: Cooldown check
            if enforce_market_cooldown:
                cooldown_start = TimeUtil.now_in_millis()
                err_msg = self.enforce_order_cooldown(trade_pair.trade_pair_id, now_ms, miner_hotkey)
                cooldown_ms = TimeUtil.now_in_millis() - cooldown_start
                bt.logging.info(f"[LOCK_WORK] Cooldown check took {cooldown_ms}ms")

            if err_msg:
                bt.logging.error(err_msg)
                return err_msg, existing_position, None

            # TIMING: Get account size
            account_size_start = TimeUtil.now_in_millis()
            account_size = self._miner_account_client.get_miner_account_size(miner_hotkey, use_account_floor=True)
            account_size_ms = TimeUtil.now_in_millis() - account_size_start
            bt.logging.info(f"[LOCK_WORK] Get account size took {account_size_ms}ms")

            # TIMING: Get or create position
            get_position_start = TimeUtil.now_in_millis()
            existing_position = self._get_or_create_open_position_from_new_order(trade_pair, signal_order_type,
                                                                                 now_ms, miner_hotkey, miner_order_uuid,
                                                                                 now_ms, price_sources,
                                                                                 miner_repo_version, account_size)
            get_position_ms = TimeUtil.now_in_millis() - get_position_start
            bt.logging.info(f"[LOCK_WORK] Get/create position took {get_position_ms}ms")

            # TIMING: Add order to position
            created_order = None
            if existing_position:
                add_order_start = TimeUtil.now_in_millis()
                limit_price = signal.get("limit_price")
                stop_loss = signal.get("stop_loss")
                take_profit = signal.get("take_profit")
                fill_price = signal.get("price")

                if execution_type == ExecutionType.LIMIT:
                    new_src = OrderSource.LIMIT_FILLED
                elif execution_type == ExecutionType.BRACKET:
                    new_src = OrderSource.BRACKET_FILLED
                else:
                    new_src = OrderSource.ORGANIC

                # Calculate price and USD conversions
                # Use fill_price if provided, otherwise use market price
                best_price_source = price_sources[0]
                price = fill_price if fill_price else best_price_source.parse_appropriate_price(now_ms, trade_pair.is_forex, signal_order_type, existing_position)
                usd_base_price = self.live_price_fetcher.get_usd_base_conversion(trade_pair, now_ms, price, signal_order_type, existing_position)

                if signal_order_type != OrderType.FLAT:
                    # Parse order size (supports leverage, value, or quantity)
                    quantity, leverage, value = self.parse_order_size(signal, usd_base_price, trade_pair, existing_position.account_size)
                else:
                    quantity, leverage, value = 0.0, 0.0, 0.0

                created_order = self._add_order_to_existing_position(existing_position, trade_pair, signal_order_type,
                                                     quantity, leverage, value, now_ms, miner_hotkey,
                                                     price_sources, miner_order_uuid, miner_repo_version,
                                                     new_src, account_size, usd_base_price, execution_type,
                                                     fill_price, limit_price, stop_loss, take_profit, bracket_orders)
                add_order_ms = TimeUtil.now_in_millis() - add_order_start
                bt.logging.info(f"[LOCK_WORK] Add order to position took {add_order_ms}ms")
            else:
                # Happens if a FLAT is sent when no position exists
                pass

        lock_released_time = TimeUtil.now_in_millis()
        lock_hold_ms = lock_released_time - lock_acquired_time
        bt.logging.info(
            f"[LOCK] Released lock for {debug_lock_key} after holding for {lock_hold_ms}ms (wait={lock_wait_ms}ms, total={lock_released_time - lock_request_time}ms)")

        # TIMING: Time from lock release to try block end
        time_after_lock = TimeUtil.now_in_millis() - lock_released_time
        bt.logging.info(f"[TIMING] Time from lock release to try block end: {time_after_lock}ms")
        return err_msg, existing_position, created_order

