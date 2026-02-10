#!/usr/bin/env python3
"""
PTN WebSocket Client with Slack Integration
Runs the websocket client and posts messages to a Slack channel
"""

import asyncio
import json
import os
import sys
import signal
import random
import websockets
import traceback
import logging
import requests
from typing import Callable, List, Optional, Dict, Any
from datetime import datetime

from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from time_util.time_util import TimeUtil


# Configure logging
def setup_logger(name):
    """Set up and return a logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid duplicates
    if not logger.handlers:
        # Create console handler and set level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Add formatter to console handler
        console_handler.setFormatter(formatter)
        # Add console handler to logger
        logger.addHandler(console_handler)

    return logger


# Module-level logger
logger = setup_logger('ptn_websocket')


def send_to_slack(message: str, webhook_url: str) -> bool:
    """Send a message to Slack via webhook."""
    if not webhook_url:
        logger.error("No Slack webhook URL provided")
        return False
    try:
        payload = {"text": message}
        response = requests.post(webhook_url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to send message to Slack: {e}")
        return False


class VantaWebSocketMessage:
    """Wrapper for websocket messages with helpful accessors."""

    # Class logger
    logger = setup_logger('VantaWebSocketMessage')

    def __init__(self, raw_message: Dict[str, Any], clock_offset_ms):
        # logger.debug('raw_message %s', raw_message)
        if isinstance(raw_message, dict):
            raw_data = raw_message['data']
        elif isinstance(raw_message, str):
            raw_data = json.loads(raw_message)['data']
        else:
            raise Exception(f'Invalid message format: {raw_message}')

        data = raw_data['position']
        self.repo_version = raw_data.get('repo_version', None)
        # tp_id = data['trade_pair_id']
        # data['trade_pair'] = TradePair.get_latest_trade_pair_from_trade_pair_id(tp_id)
        self.position = Position(**data)
        self.new_order = self.position.orders[-1]

        self.sequence = raw_message.get("sequence")
        self.timestamp = raw_message.get("timestamp")
        self.clock_offset_ms = clock_offset_ms
        if self.timelag_from_order > 10000:
            self.logger.warning("Timelag from new order is unusually high: %sms. Consider ignoring this order.",
                                self.timelag_from_order)

    @property
    def timelag_from_order(self) -> int:
        """Return the timelag from the order in milliseconds."""
        now_ms = TimeUtil.now_in_millis()
        return abs(now_ms - self.new_order.processed_ms) - self.clock_offset_ms

    @property
    def timelag_from_queue(self) -> int:
        """Return the timelag from the queue in milliseconds."""
        now_ms = TimeUtil.now_in_millis()
        return abs(now_ms - self.timestamp) - self.clock_offset_ms

    def format_slack_message(self, server_host: str, server_alias: Optional[str] = None) -> str:
        """Format the message for Slack notification."""
        # Get server display name
        server_display = f"{server_alias} ({server_host})" if server_alias else server_host

        # Get position data
        position_dict = self.position.compact_dict_no_orders()
        order_dict = self.new_order.to_python_dict()

        # Format price sources
        price_sources_str = ""
        for ps in order_dict.get("price_sources", []):
            price_sources_str += f"\n  - {ps.source}: {ps.close} (lag: {ps.lag_ms}ms, span: {ps.timespan_ms}ms)"
            if ps.bid or ps.ask:
                price_sources_str += f" bid: {ps.bid}, ask: {ps.ask}"

        slack_message = (
            f"📊 *PTN Message* from `{server_display}`\n"
            f"```\n"
            f"Seq: {self.sequence}, Repo Version: {self.repo_version}\n"
            f"Position {position_dict.get('trade_pair', ['Unknown'])[0]} | {position_dict.get('position_type')} | {position_dict.get('net_leverage')}x:\n"
            f"  Miner: {position_dict.get('miner_hotkey')}, UUID: {position_dict.get('position_uuid')}\n"
            f"  Return: {position_dict.get('return_at_close')}, PnL: {position_dict.get('realized_pnl')}\n"
            f"Order ({order_dict.get('order_type')}) | {order_dict.get('leverage')}x"
            f"{price_sources_str}\n"
            f"Timelag: Queue={self.timelag_from_queue}ms, Order={self.timelag_from_order}ms\n"
            f"```"
        )
        return slack_message

    def __str__(self) -> str:

        # Format the position summary in a more readable way
        position_summary = json.dumps(self.position.compact_dict_no_orders(), indent=2, cls=CustomEncoder)

        # Format the new order in a more readable way
        new_order = json.dumps(self.new_order.to_python_dict(), indent=2, cls=CustomEncoder)

        return (f"VantaWebSocketMessage(seq={self.sequence}, repo_version={self.repo_version})\n"
                f"Position Summary:\n{position_summary}\n"
                f"New Order:\n{new_order}\n"
                f"Approx Timelag (ms): from_queue={self.timelag_from_queue}, from_order={self.timelag_from_order}")

    def __repr__(self) -> str:
        return self.__str__()


class VantaWebSocketClient:
    """Client for connecting to WebSocket server with authentication and subscription capabilities."""

    # Class logger
    logger = setup_logger('VantaWebSocketClient')

    def __init__(self,
                 api_key: Optional[str] = None,
                 host: str = "localhost",
                 port: int = 8765,
                 secure: bool = False,
                 slack_webhook_url: Optional[str] = None,
                 server_alias: Optional[str] = None):
        """Initialize the WebSocket client.

        Args:
            api_key: API key for authentication
            host: WebSocket server hostname
            port: WebSocket server port
            secure: Whether to use secure WebSocket (wss://) or not
            slack_webhook_url: Slack webhook URL for notifications
            server_alias: Human-friendly name for the server
        """
        protocol = "wss" if secure else "ws"
        self.uri = f"{protocol}://{host}:{port}"
        self.api_key = api_key
        self.host = host  # Store host for logging
        self.slack_webhook_url = slack_webhook_url
        self.server_alias = server_alias

        # Connection state
        self.connected = False
        self.websocket = None

        # Sequence tracking
        self.last_sequence = -1
        self._sequence_file = f"client_sequence_{api_key}.txt"
        self._load_sequence()

        # Subscription management
        self.subscribed = False

        # Message processing
        self.message_handlers = []
        self.message_buffer = {}

        # Statistics
        self.messages_received = 0
        self.messages_processed = 0

        # Retry logic
        self._initial_backoff = 1
        self._max_backoff = 60
        self._jitter = 0.1
        self.force_quit = False
        self.clock_offset_estimate_ms = 0
        self.n_pongs = 0

        self.logger.info("WebSocketClient initialized with URI: %s", self.uri)

    def stop(self):
        """Signal the client to shut down."""
        self.force_quit = True
        if self.websocket:
            try:
                # schedule a clean close; close() returns a coroutine
                asyncio.get_event_loop().create_task(self.websocket.close())
            except Exception:
                # if the loop is gone or websocket is already closing, ignore
                pass

    # Add to WebSocketClient class
    async def _periodic_ping(self):
        """Send periodic pings to measure latency."""
        while self.connected and not self.force_quit:
            if await self.send_ping():
                # logger.debug("Ping sent")
                pass
            await asyncio.sleep(30)  # Send ping every 30 seconds

    # Add to the WebSocketClient class
    async def send_ping(self):
        """Send a ping message to the server and track when it was sent."""
        if not self.connected or not self.websocket:
            return False

        ping_timestamp = TimeUtil.now_in_millis()
        ping_message = {
            "type": "ping",
            "timestamp": ping_timestamp
        }

        try:
            await self.websocket.send(json.dumps(ping_message))
            return True
        except Exception as e:
            self.logger.error("Error sending ping: %s", e)
            return False

    def _load_sequence(self) -> None:
        """Load the last processed sequence number from disk."""
        try:
            if os.path.exists(self._sequence_file):
                with open(self._sequence_file, 'r') as f:
                    self.last_sequence = int(f.read().strip())
        except Exception as e:
            self.logger.error("Error loading sequence number: %s", e)
            self.last_sequence = -1

    def _save_sequence(self) -> None:
        """Save the current sequence number to disk."""
        try:
            with open(self._sequence_file, 'w') as f:
                f.write(str(self.last_sequence))
        except Exception as e:
            self.logger.error("Error saving sequence number: %s", e)

    def subscribe(self) -> None:
        """Subscribe to all market data."""
        self.logger.info("Subscribing to all market data")
        self.subscribed = True  # Mark as subscribed regardless of connection state

        # If already connected, send subscription message
        if self.connected and self.websocket:
            asyncio.create_task(self._send_subscription())

    def unsubscribe(self) -> None:
        """Unsubscribe from all market data."""
        self.logger.info("Unsubscribing from all market data")

        # If connected, send unsubscription message
        if self.connected and self.websocket:
            asyncio.create_task(self._send_unsubscription())

    async def _send_subscription(self) -> None:
        """Send subscription message to the server."""
        if not self.connected or not self.websocket:
            return

        sub_message = {
            "type": "subscribe",
            "sender_timestamp": TimeUtil.now_in_millis(),
            "all": True
        }
        await self.websocket.send(json.dumps(sub_message))
        self.subscribed = True

    async def _send_unsubscription(self) -> None:
        """Send unsubscription message to the server."""
        if not self.connected or not self.websocket:
            return

        unsub_message = {
            "type": "unsubscribe",
            "all": True
        }
        await self.websocket.send(json.dumps(unsub_message))
        self.subscribed = False

    async def _authenticate(self) -> None:
        """Send authentication message with API key and last sequence number."""
        if not self.api_key:
            raise ValueError("API key is required for authentication")

        auth_message = {
            "api_key": self.api_key,
            "last_sequence": self.last_sequence
        }

        await self.websocket.send(json.dumps(auth_message))

        # Wait for auth response
        response = await self.websocket.recv()
        response_data = json.loads(response)

        if response_data.get("status") != "success":
            error_msg = response_data.get("message", "Authentication failed")
            raise Exception(f"Authentication error: {error_msg}")

        # Update sequence if needed
        server_sequence = response_data.get("current_sequence", 0)
        if server_sequence > self.last_sequence:
            self.last_sequence = server_sequence
            self._save_sequence()

        self.logger.info("Authentication successful, server sequence: %s", server_sequence)

        # Send connection notification to Slack
        if self.slack_webhook_url:
            server_display = f"{self.server_alias} ({self.host})" if self.server_alias else self.host

            connection_msg = f"✅ Connected to PTN WebSocket Server\n" \
                             f"Server: {server_display}:{self.uri.split(':')[-1]}\n" \
                             f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n" \
                             f"Starting sequence: {server_sequence}"
            send_to_slack(connection_msg, self.slack_webhook_url)

    async def _process_messages(self) -> None:
        """Process incoming messages and dispatch to handlers."""
        while True:
            try:
                # Add a timeout to detect server disconnection faster
                message = await asyncio.wait_for(self.websocket.recv(), timeout=60)
                data = json.loads(message)

                # Handle batch messages
                if data.get("type") == "batch" and "messages" in data:
                    messages = []
                    for item in data["messages"]:
                        if "sequence" in item:
                            self.last_sequence = max(self.last_sequence, item["sequence"])
                            messages.append(VantaWebSocketMessage(item, self.clock_offset_estimate_ms))

                    # Call handlers with the batch of messages
                    if messages and self.message_handlers:
                        for handler in self.message_handlers:
                            await self._call_handler(handler, messages)

                # Handle single messages
                elif "sequence" in data:
                    self.last_sequence = max(self.last_sequence, data["sequence"])
                    message_obj = VantaWebSocketMessage(data, self.clock_offset_estimate_ms)

                    # Call handlers with a list containing one message
                    if self.message_handlers:
                        for handler in self.message_handlers:
                            await self._call_handler(handler, [message_obj])

                # Handle subscription status messages
                elif data.get("type") == "subscription_status":
                    now_ms = TimeUtil.now_in_millis()
                    roundtrip_lag = now_ms - data.get("sender_timestamp", 0)
                    # Clocks on the sender and receiver are offset. Lets quantify this.
                    one_way_travel_time = roundtrip_lag // 2
                    expected_received_timestamp = data.get("sender_timestamp", 0) + one_way_travel_time
                    clock_offset = data.get("received_timestamp", 0) - expected_received_timestamp
                    self.clock_offset_estimate_ms = abs(clock_offset)
                    formatted_lags = f"roundtrip: {roundtrip_lag}ms. detected clock offset {clock_offset}ms"
                    self.logger.info("Subscription status: %s for action: %s with time lags: %s",
                                     data.get('status'), data.get('action'), formatted_lags)


                # Update the part in _process_messages that handles pong messages
                elif data.get("type") == "pong":
                    self.n_pongs += 1
                    send_timestamp = data.get("client_timestamp", 0)
                    server_timestamp = data.get("server_timestamp", 0)
                    received_timestamp = TimeUtil.now_in_millis()
                    roundtrip_ms = received_timestamp - send_timestamp
                    expected_server_timestamp = send_timestamp + roundtrip_ms // 2
                    clock_offset_estimate_ms = server_timestamp - expected_server_timestamp
                    if self.n_pongs % 5 == 0:
                        self.logger.info("Ping roundtrip: %sms, clock offset estimate: %sms",
                                         roundtrip_ms, clock_offset_estimate_ms)
                    self.clock_offset_estimate_ms = abs(clock_offset_estimate_ms)

            except asyncio.TimeoutError:
                # No message received within timeout, check if connection is still alive
                try:
                    # Send a ping frame to check connection
                    pong_waiter = await self.websocket.ping()
                    await asyncio.wait_for(pong_waiter, timeout=5)
                    self.logger.info("Connection still alive (ping-pong successful)")
                except Exception:
                    self.logger.warning("Connection appears to be dead (ping failed)")
                    # Let connection be re-established by breaking out of the loop
                    break
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("Connection closed by server")
                if self.slack_webhook_url:
                    server_display = self.server_alias or self.host
                    send_to_slack(f"⚠️ Connection closed by server {server_display}", self.slack_webhook_url)
                break
            except json.JSONDecodeError as e:
                self.logger.error("Received invalid JSON: %s", e)
                # Continue processing other messages
            except Exception as e:
                self.logger.error("Error processing message: %s", e)
                traceback.print_exc()
                # Continue processing other messages

    async def _call_handler(self, handler: Callable, messages: List[PTNWebSocketMessage]) -> None:
        """Call a message handler with proper error handling.

        Args:
            handler: Handler function/coroutine to call
            messages: List of PTNWebSocketMessage objects to pass to the handler
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(messages)
            else:
                # Call synchronous handler
                handler(messages)

            self.messages_processed += len(messages)
        except Exception as e:
            self.logger.error("Error in message handler: %s", e)
            traceback.print_exc()

    async def _connect(self) -> None:
        """Connect to the WebSocket server with retry logic."""
        backoff = self._initial_backoff

        while not self.force_quit:
            try:
                # First try to establish a connection
                try:
                    websocket = await websockets.connect(
                        self.uri,
                        compression="deflate",
                        ping_interval=30,
                        close_timeout=5,  # Give more time for graceful close
                        max_size=None  # No limit on message size
                    )
                    self.websocket = websocket
                    self.connected = True
                    self.logger.info("Connected to %s", self.uri)
                except Exception as conn_err:
                    self.logger.error("Connection error: %s", conn_err)
                    # Apply exponential backoff with jitter
                    jitter_amount = random.uniform(-self._jitter, self._jitter)
                    adjusted_backoff = backoff * (1 + jitter_amount)

                    self.logger.info("Reconnecting in %.2f seconds...", adjusted_backoff)
                    await asyncio.sleep(adjusted_backoff)

                    # Increase backoff for next attempt
                    backoff = min(backoff * 2, self._max_backoff)
                    continue  # Skip to next iteration to retry connection

                # Now that we have a connection, try to authenticate
                try:
                    # Authenticate
                    await self._authenticate()

                    # Reset backoff on successful connection and authentication
                    backoff = self._initial_backoff

                    # Always resubscribe after authentication, regardless of previous state
                    self.logger.info("Sending subscription after authentication")
                    await self._send_subscription()
                    self.subscribed = True

                    # Start periodic ping for latency measurement
                    ping_task = asyncio.create_task(self._periodic_ping())

                    try:
                        # Process messages until connection closes
                        await self._process_messages()
                    finally:
                        # Ensure ping task is cancelled when message processing ends
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass

                except (websockets.exceptions.ConnectionClosed, EOFError) as e:
                    self.logger.warning("Connection closed: %s", e)
                    self.connected = False
                    self.websocket = None

                    # We'll automatically resubscribe on reconnection, so no need to reset subscribed flag
                    self.logger.info("Connection lost. Will resubscribe on reconnection.")

                    # Use a shorter backoff for connection loss after successful connection
                    backoff = self._initial_backoff
                    await asyncio.sleep(1)  # Short delay before reconnect attempt
                    continue
                except Exception as e:
                    if "Authentication error" in str(e):
                        self.logger.error("%s. Invalid API key — stopping retries.", e)
                        if self.slack_webhook_url:
                            server_display = self.server_alias or self.host
                            send_to_slack(f"❌ Authentication failed for server {server_display}: Invalid API key",
                                          self.slack_webhook_url)
                        self.stop()
                        return

                    self.logger.error("Error during communication: %s", e)
                    if self.websocket:
                        await self.websocket.close()
                    self.connected = False
                    self.websocket = None
                    await asyncio.sleep(1)  # Short delay before reconnect attempt
                    continue

            except Exception as e:
                self.logger.error("Unexpected error in connection loop: %s", e)
                traceback.print_exc()
                self.connected = False
                self.websocket = None

                # Apply exponential backoff with jitter
                jitter_amount = random.uniform(-self._jitter, self._jitter)
                adjusted_backoff = backoff * (1 + jitter_amount)

                self.logger.info("Reconnecting in %.2f seconds...", adjusted_backoff)
                await asyncio.sleep(adjusted_backoff)

                # Increase backoff for next attempt
                backoff = min(backoff * 2, self._max_backoff)

    def run(self, handler: Optional[Callable[[List[VantaWebSocketMessage]], None]] = None) -> None:
        """Run the client with the given message handler.

        Args:
            handler: Function to call for each batch of messages
        """
        if handler:
            self.message_handlers.append(handler)

        try:
            # Set up proper signal handling
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Add signal handlers for graceful shutdown
            for signal_name in ('SIGINT', 'SIGTERM'):
                try:
                    loop.add_signal_handler(
                        getattr(signal, signal_name),
                        lambda: asyncio.create_task(self._shutdown(loop))
                    )
                except Exception as e:
                    self.logger.error("Error setting up signal handler for %s: %s. Ignoring.", signal_name, e)

            # Run the connection loop
            loop.run_until_complete(self._connect())
            loop.close()
        except KeyboardInterrupt:
            self.logger.info("Client shutting down due to keyboard interrupt...")
            self._save_sequence()
        except Exception as e:
            self.logger.error("Client error: %s", e)
            traceback.print_exc()
            self._save_sequence()

    async def _shutdown(self, loop):
        """Handle graceful shutdown."""
        self.logger.info("Shutting down client...")

        # Save sequence number
        self._save_sequence()

        # Close the websocket if it's open
        if self.websocket and self.connected:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.error("Error closing websocket: %s", e)

        # Send shutdown notification to Slack
        if self.slack_webhook_url:
            server_display = self.server_alias or self.host
            send_to_slack(f"🛑 PTN WebSocket Client shutting down for server {server_display}", self.slack_webhook_url)

        # Stop the event loop
        loop.stop()


# For standalone testing
if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) < 3:
        print("Usage: python websocket_client_slack.py <api_key> <host> [slack_webhook_url] [server_alias] [port]")
        sys.exit(1)

    api_key = sys.argv[1]
    host = sys.argv[2]
    slack_webhook_url = sys.argv[3] if len(sys.argv) > 3 else None
    server_alias = sys.argv[4] if len(sys.argv) > 4 else None
    port = int(sys.argv[5]) if len(sys.argv) > 5 else 8765

    # Main logger
    main_logger = setup_logger('ptn_websocket_main')


    # Define a message handler that logs to console and optionally to Slack
    def handle_messages(messages: List[VantaWebSocketMessage]) -> None:
        for msg in messages:
            # Log to console
            main_logger.info(
                "\nReceived message %s\n--------------------------------------------------------------------", msg)

            # Send to Slack if webhook URL is provided
            if slack_webhook_url:
                slack_message = msg.format_slack_message(host, server_alias)
                send_to_slack(slack_message, slack_webhook_url)


    # Create client
    main_logger.info("Connecting to ws://%s:%s with API key: %s", host, port, api_key)
    client = VantaWebSocketClient(api_key=api_key, host=host, port=port,
                                slack_webhook_url=slack_webhook_url, server_alias=server_alias)

    # Run client
    client.run(handle_messages)