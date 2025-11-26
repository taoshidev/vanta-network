import asyncio
from setproctitle import setproctitle
import json
import websockets
import traceback
import argparse
import os
import logging
from multiprocessing import Manager
from collections import defaultdict, deque
from multiprocessing import current_process
from typing import Dict, Any, Optional, Set, Deque
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.vali_bkp_utils import CustomEncoder, ValiBkpUtils

# Assuming APIKeyMixin is in api.api_key_refresh
from vanta_api.api_key_refresh import APIKeyMixin
from vali_objects.vali_config import TradePair

# Maximum number of websocket connections allowed per API key
MAX_N_WS_PER_API_KEY = 5


class WebSocketServer(APIKeyMixin):
    """Handles WebSocket connections with authentication and message broadcasting."""

    def __init__(self,
                 api_keys_file: str,
                 shared_queue: Optional[Any] = None,
                 host: str = "localhost",
                 port: int = 8765,
                 reconnect_interval: int = 3,
                 max_reconnect_attempts: int = 10,
                 refresh_interval: int = 15,
                 send_test_positions: bool = False,
                 test_position_interval: int = 5):
        """Initialize the WebSocket server.

        Args:
            api_keys_file: Path to the API keys file
            shared_queue: Queue for receiving messages from other processes
            host: Hostname to bind the WebSocket server to
            port: Port to bind the WebSocket server to
            reconnect_interval: Seconds between reconnection attempts
            max_reconnect_attempts: Maximum number of reconnection attempts (0=infinite)
            refresh_interval: How often to check for API key changes (seconds)
            send_test_positions: Whether to periodically send test orders (for testing only)
            test_positions_interval: How often to send test orders (seconds)
        """
        # Initialize API key handling
        APIKeyMixin.__init__(self, api_keys_file, refresh_interval)

        # WebSocket server configuration
        self.host = host
        self.port = port
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.server = None
        self.shutdown_event = None

        # Client tracking
        self.connected_clients: Dict[str, websockets.WebSocketServerProtocol] = {}

        # Track API key and tier for each client
        self.client_auth: Dict[str, Dict[str, Any]] = {}

        # API key tracking - maintain a FIFO queue for each API key
        self.api_key_clients: Dict[str, Deque[str]] = defaultdict(deque)

        # Message queueing and processing
        self.shared_queue = shared_queue
        self.queue_check_interval = 0.1  # seconds
        self.message_queue = None  # Will be initialized as asyncio.Queue in start()

        # Sequence number management
        self.sequence_number = 0
        self.sequence_file = ValiBkpUtils.get_sequence_number_file_path()
        self._load_sequence_number()

        # Tasks
        self.save_task = None
        self.queue_processor_task = None
        self.shared_queue_task = None
        self.test_positions_task = None
        self.loop = None

        # Subscriptions: set of subscribed client IDs
        self.subscribed_clients: Set[str] = set()

        # Test order configuration
        self.send_test_positions = send_test_positions
        self.test_positions_interval = test_position_interval

        # Start API key refresh thread
        self.start_refresh_thread()

        bt.logging.info(f"WebSocketServer: Initialized with {len(self.accessible_api_keys)} API keys")
        if self.send_test_positions:
            bt.logging.info(f"WebSocketServer: Test orders will be sent every {self.test_positions_interval} seconds")

    def _load_sequence_number(self) -> None:
        """Load the last sequence number from disk."""
        try:
            if os.path.exists(self.sequence_file):
                with open(self.sequence_file, 'r') as f:
                    self.sequence_number = int(f.read().strip())
                bt.logging.info(f"WebSocketServer: Loaded sequence number: {self.sequence_number}")
            else:
                self.sequence_number = 0
                bt.logging.info(f"WebSocketServer: Starting with new sequence number")
        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error loading sequence number: {e}")
            self.sequence_number = 0

    def _save_sequence_number(self) -> None:
        """Save the current sequence number to disk."""
        try:
            with open(self.sequence_file, 'w') as f:
                f.write(str(self.sequence_number))
        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error saving sequence number: {e}")

    async def _periodic_save_sequence(self) -> None:
        """Periodically save the sequence number to disk."""
        while True:
            try:
                await asyncio.sleep(10)  # Save every few seconds
                self._save_sequence_number()
            except asyncio.CancelledError:
                # Final save on shutdown
                self._save_sequence_number()
                break
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error in periodic save: {e}")

    async def _check_shared_queue(self) -> None:
        """Check the shared queue for messages from other processes."""
        if self.shared_queue is None:
            bt.logging.info(f"WebSocketServer: No shared queue available")
            return

        # Create an event loop to handle queue notifications
        loop = asyncio.get_running_loop()

        while True:
            try:
                # Check if shutdown is requested
                if self.shutdown_event and self.shutdown_event.is_set():
                    break

                # Use a future to get the next item from the queue without polling
                future = loop.create_future()

                # Schedule a callback to run in a thread pool to get an item without blocking
                def get_from_queue():
                    try:
                        # This is a blocking call but runs in a thread pool
                        item = self.shared_queue.get()
                        # Set the result on the future when an item is available
                        loop.call_soon_threadsafe(lambda: future.set_result(item))
                    except Exception as ex:
                        # Set exception on the future if there's an error
                        loop.call_soon_threadsafe(lambda exc=ex: future.set_exception(exc))

                # Run the get operation in a thread pool
                await loop.run_in_executor(None, get_from_queue)

                # Wait for the result from the queue
                message_data = await future

                # Forward the message immediately to our asyncio queue
                await self.message_queue.put(message_data)

            except asyncio.CancelledError:
                bt.logging.info(f"WebSocketServer: Shared queue monitor cancelled")
                break
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error in shared queue check: {e}")
                bt.logging.error(traceback.format_exc())
                # Very short recovery time
                await asyncio.sleep(1)

    async def _generate_test_ws_positions(self) -> None:
        """Periodically generate test positions and put them in the queue."""
        if not self.send_test_positions or self.shared_queue is None:
            return

        while True:
            try:
                # Create a test order
                current_time = TimeUtil.now_in_millis()
                # Queue it for processing
                await self.message_queue.put(self._create_test_position(current_time))

                bt.logging.info(f"WebSocketServer: Generated test position")

                # Wait before generating the next order
                await asyncio.sleep(self.test_positions_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error generating test order: {e}")
                await asyncio.sleep(self.test_positions_interval)

    def _create_test_position(self, timestamp: int) -> Dict[str, Any]:
        """Create a test order with randomized parameters."""
        p = Position(**
        {'miner_hotkey': '5EWKUhycaBQHiHnfE3i2suZ1BvxAAE3HcsFsp8TaR6mu3JrJ',
         'position_uuid': 'bbc676e5-9ab3-4c11-8be4-5f2022ae9208', 'open_ms': timestamp,
         'trade_pair': TradePair.BTCUSD,
         'orders': [{'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.SHORT, 'leverage': -0.5, 'price': 82872.75,
                     'bid': 82880.06, 'ask': 82880.07, 'slippage': 1e-05, 'processed_ms': timestamp,
                     'order_uuid': 'bbc676e5-9ab3-4c11-8be4-5f2022ae9208', 'price_sources': [
                         {'ask': 82880.07, 'bid': 82880.06, 'close': 82872.75, 'high': 82872.75, 'lag_ms': 1348,
                          'low': 82872.75,
                          'open': 82872.75, 'source': 'Tiingo_gdax_rest', 'start_ms': 1742357760395, 'timespan_ms': 0,
                          'vwap': 82872.75, 'websocket': True},
                         {'ask': 0.0, 'bid': 0.0, 'close': 82869.24, 'high': 82880.07, 'lag_ms': 2048, 'low': 82869.24,
                          'open': 82880.07, 'source': 'Polygon_rest', 'start_ms': 1742357756000, 'timespan_ms': 1000,
                          'vwap': 82875.8692, 'websocket': False},
                         {'ask': 0.0, 'bid': 0.0, 'close': 82047.16, 'high': 82047.16, 'lag_ms': 26043047, 'low': 82047.16,
                          'open': 82047.16, 'source': 'Polygon_ws', 'start_ms': 1742331716000, 'timespan_ms': 0,
                          'vwap': 82047.16,
                          'websocket': True}], 'src': 0},
                    {'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.FLAT, 'leverage': 0.5, 'price': 83202.0,
                     'bid': 83202.0, 'ask': 83202.0, 'slippage': 1e-05, 'processed_ms': timestamp + 15000,
                     'order_uuid': 'd6e1e768-9024-4cc5-84d7-d66691d82061', 'price_sources': [], 'src': 0}],
         'current_return': 0.9980034808990859, 'close_ms': 1742362860000, 'net_leverage': 0.0,
         'return_at_close': 0.9980034808990859, 'average_entry_price': 82871.92127250001,
         'position_type': OrderType.FLAT,
         'is_closed_position': True})
        return p.to_websocket_dict()

    async def _process_message_queue(self) -> None:
        """Process messages from the queue and broadcast them."""
        max_batch_size = 50  # Maximum number of messages to batch
        max_wait_time = 0.05  # Maximum time to wait for more messages (50ms)

        # Flag to enable/disable message batching
        enable_batching = False  # Set to False to disable batching and minimize latency

        while True:
            try:
                # Get the first message (this will block until a message is available)
                first_message = await self.message_queue.get()
                messages_to_broadcast = [first_message]
                self.message_queue.task_done()

                # Try to batch additional messages if batching is enabled
                if enable_batching:
                    # Use a timeout for collecting additional messages
                    batch_end_time = TimeUtil.now_in_millis() + max_wait_time

                    # Collect as many messages as available, up to max_batch_size
                    while len(messages_to_broadcast) < max_batch_size:
                        # Calculate remaining wait time
                        remaining_time = batch_end_time - TimeUtil.now_in_millis()
                        if remaining_time <= 0:
                            # We've waited long enough, stop collecting
                            break

                        try:
                            # Wait for a message with timeout
                            next_message = await asyncio.wait_for(
                                self.message_queue.get(),
                                timeout=remaining_time
                            )
                            messages_to_broadcast.append(next_message)
                            self.message_queue.task_done()
                        except asyncio.TimeoutError:
                            # No more messages available within our wait time
                            break

                # Broadcast the messages
                if len(messages_to_broadcast) > 1:
                    #print(f"[{current_process().name}] Broadcasting batch of {len(messages_to_broadcast)} messages")
                    await self.broadcast_message_batch(messages_to_broadcast)
                else:
                    #print(f"[{current_process().name}] Broadcasting single message")
                    await self.broadcast_message(messages_to_broadcast[0])

            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error processing message from queue: {e}")
                bt.logging.error(traceback.format_exc())

    async def broadcast_message_batch(self, messages) -> int:
        """Broadcast a batch of messages to subscribed clients.

        Args:
            messages: List of message data to broadcast

        Returns:
            The latest sequence number
        """
        # Create a list to hold all the sequenced messages
        sequenced_messages = []

        # Assign sequence numbers to all messages in the batch
        for message_data in messages:
            self.sequence_number += 1

            sequenced_messages.append({
                "sequence": self.sequence_number,
                "timestamp": TimeUtil.now_in_millis(),
                "data": message_data
            })

        # Create a batch message
        batch_message = {
            "type": "batch",
            "messages": sequenced_messages,
            "count": len(sequenced_messages)
        }

        serialized_message = json.dumps(batch_message, cls=CustomEncoder)

        # Send to all subscribed clients (iterate over copy to avoid race condition)
        disconnected_clients = []
        for client_id in list(self.subscribed_clients):
            if client_id in self.connected_clients:
                websocket = self.connected_clients[client_id]
                try:
                    await websocket.send(serialized_message)
                except websockets.exceptions.ConnectionClosed:
                    # Client disconnected, remove from subscribers
                    disconnected_clients.append(client_id)
                    bt.logging.info(f"WebSocketServer: Client {client_id} disconnected while sending batch")
                except Exception as e:
                    bt.logging.error(f"WebSocketServer: Error sending batch to client {client_id}: {e}")
                    disconnected_clients.append(client_id)

        # Remove disconnected clients
        for client_id in disconnected_clients:
            self._remove_client(client_id)

        return self.sequence_number

    async def broadcast_message(self, message_data) -> int:
        """Broadcast a single message to subscribed clients.

        Args:
            message_data: Message data to broadcast

        Returns:
            The new sequence number
        """
        # Increment sequence number
        self.sequence_number += 1

        # Add sequence number to message
        message_with_seq = {
            "sequence": self.sequence_number,
            "timestamp": TimeUtil.now_in_millis(),
            "data": message_data
        }

        serialized_message = json.dumps(message_with_seq, cls=CustomEncoder)

        # Send to all subscribed clients (iterate over copy to avoid race condition)
        disconnected_clients = []
        for client_id in list(self.subscribed_clients):
            if client_id in self.connected_clients:
                websocket = self.connected_clients[client_id]
                try:
                    await websocket.send(serialized_message)
                except websockets.exceptions.ConnectionClosed:
                    # Client disconnected, mark for removal
                    disconnected_clients.append(client_id)
                    bt.logging.info(f"WebSocketServer: Client {client_id} disconnected while sending")
                except Exception as e:
                    bt.logging.error(f"WebSocketServer: Error sending to client {client_id}: {e}")
                    disconnected_clients.append(client_id)

        # Remove disconnected clients
        for client_id in disconnected_clients:
            self._remove_client(client_id)

        return self.sequence_number

    def _remove_client(self, client_id: str) -> None:
        """Remove a client from all subscriptions and connected clients.

        Args:
            client_id: Client ID to remove
        """
        # Remove from connected clients
        if client_id in self.connected_clients:
            # Get API key associated with this client
            api_key = None
            for key, clients in self.api_key_clients.items():
                if client_id in clients:
                    api_key = key
                    break

            # Remove from API key tracking
            if api_key and client_id in self.api_key_clients[api_key]:
                self.api_key_clients[api_key].remove(client_id)
                api_key_alias = self.api_key_to_alias.get(api_key, "Unknown")
                bt.logging.info(f"WebSocketServer: Removed client {client_id} from API key {api_key_alias}")

            # Remove from connected clients
            self.connected_clients.pop(client_id, None)

            # Remove from client auth tracking
            self.client_auth.pop(client_id, None)

            # Remove from subscribed clients
            if client_id in self.subscribed_clients:
                self.subscribed_clients.remove(client_id)

            bt.logging.info(f"WebSocketServer: Client {client_id} removed from all registries")

    def send_message(self, message_data: Dict[str, Any]) -> bool:
        """Queue a message for broadcasting.

        Args:
            message_data: The message data to broadcast

        Returns:
            True if message was queued successfully, False otherwise
        """
        try:
            if self.loop is None:
                bt.logging.error(f"WebSocketServer: Cannot send message: server not started")
                return False

            # Use run_coroutine_threadsafe to safely run in the event loop
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(message_data),
                self.loop
            )
            return True
        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error queueing message: {e}")
            bt.logging.error(traceback.format_exc())
            return False

    async def handle_client(self, websocket) -> None:
        """Handle client connection with authentication and subscriptions.

        Args:
            websocket: WebSocket connection
        """
        client_id = str(id(websocket))
        bt.logging.info(f"WebSocketServer: New client connected (ID: {client_id})")

        try:
            # Wait for authentication message
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)

            if 'api_key' not in auth_data:
                error_msg = "Authentication required. Please provide an API key."
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": error_msg
                }))
                return

            api_key = auth_data['api_key']

            # Get last received sequence number from client if provided.
            # TODO: implement per-sequence recovery. until then, clients can make REST request to fillin detected gaps.
            # last_sequence = auth_data.get('last_sequence', -1)

            # Validate API key
            try:
                is_valid = self.is_valid_api_key(api_key)
            except Exception as e:
                error_msg = f"Error validating API key: {e}"
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": error_msg
                }))
                return

            if not is_valid:
                error_msg = "Invalid API key. Authentication failed."
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": error_msg
                }))
                return

            # Get the API key's tier
            api_key_tier = self.get_api_key_tier(api_key)

            # Websockets are only available for tier 100 clients
            if api_key_tier < 100:
                error_msg = "WebSocket connections require tier 100 access. Please upgrade your API key."
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": error_msg,
                    "code": "INSUFFICIENT_TIER"
                }))
                return

            # Check if we've reached the maximum number of clients for this API key
            if len(self.api_key_clients.get(api_key, [])) >= MAX_N_WS_PER_API_KEY:
                # Get the oldest client ID for this API key
                oldest_client_id = self.api_key_clients[api_key][0]

                # Remove the oldest client to make room for the new one
                if oldest_client_id in self.connected_clients:
                    try:
                        # Send disconnection message to the client
                        await self.connected_clients[oldest_client_id].send(json.dumps({
                            "status": "error",
                            "message": "Disconnected due to too many connections for this API key",
                            "code": "MAX_CONNECTIONS_EXCEEDED"
                        }))
                        # Close the connection
                        await self.connected_clients[oldest_client_id].close()
                    except Exception as e:
                        bt.logging.error(f"WebSocketServer: Error closing oldest connection {oldest_client_id}: {e}")

                # Remove the client from our records
                self._remove_client(oldest_client_id)
                api_key_alias = self.api_key_to_alias.get(api_key, "Unknown")

                bt.logging.info(f"WebSocketServer: Dropped oldest client {oldest_client_id} for API key "
                      f"{api_key_alias} to make room for new client {client_id}")

            # Send authentication success with tier information
            await websocket.send(json.dumps({
                "status": "success",
                "message": "Authentication successful.",
                "current_sequence": self.sequence_number,
                "tier": api_key_tier
            }))

            bt.logging.info(f"WebSocketServer: Client {client_id} authenticated successfully with tier {api_key_tier}")

            # Add to connected clients
            self.connected_clients[client_id] = websocket

            # Store the client's auth information
            self.client_auth[client_id] = {
                "api_key": api_key,
                "tier": api_key_tier
            }

            # Add to API key tracking (FIFO queue)
            self.api_key_clients[api_key].append(client_id)
            api_key_alias = self.api_key_to_alias.get(api_key, "Unknown")
            bt.logging.info(
                f"WebSocketServer: Client {client_id} added to API key {api_key_alias} (active: {len(self.api_key_clients[api_key])}/{MAX_N_WS_PER_API_KEY})")

            # Process client messages (subscriptions, etc.)
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    message_type = data.get("type", "")

                    # Update the ping handling in the server's handle_client method
                    if message_type == "ping":
                        # Get the client's timestamp
                        client_timestamp = data.get("timestamp", 0)
                        server_timestamp = TimeUtil.now_in_millis()

                        # Respond to ping with both timestamps
                        await websocket.send(json.dumps({
                            "type": "pong",
                            "client_timestamp": client_timestamp,
                            "server_timestamp": server_timestamp,
                            "timestamp": server_timestamp
                        }))

                    elif message_type == "subscribe":
                        # Handle subscription to all data
                        if data.get("all", False):
                            self.subscribed_clients.add(client_id)
                            bt.logging.info(f"WebSocketServer: Client {client_id} subscribed to all data")

                            # Confirm subscription
                            await websocket.send(json.dumps({
                                "type": "subscription_status",
                                "status": "success",
                                "all": True,
                                "action": "subscribe",
                                "sender_timestamp": data.get("sender_timestamp", 0),
                                "received_timestamp": TimeUtil.now_in_millis()
                            }))

                    elif message_type == "unsubscribe":
                        # Handle unsubscription
                        if data.get("all", False) and client_id in self.subscribed_clients:
                            self.subscribed_clients.remove(client_id)
                            bt.logging.info(f"WebSocketServer: Client {client_id} unsubscribed from all data")

                            # Confirm unsubscription
                            await websocket.send(json.dumps({
                                "type": "subscription_status",
                                "status": "success",
                                "all": True,
                                "action": "unsubscribe"
                            }))

                except websockets.exceptions.ConnectionClosed:
                    break
                except json.JSONDecodeError:
                    bt.logging.warning(f"WebSocketServer: Received invalid JSON from client {client_id}")
                except Exception as e:
                    bt.logging.error(f"WebSocketServer: Error processing message from client {client_id}: {e}")
                    bt.logging.error(traceback.format_exc())

        except websockets.exceptions.ConnectionClosed:
            bt.logging.info(f"WebSocketServer: Client {client_id} disconnected")
        except json.JSONDecodeError:
            bt.logging.warning(f"WebSocketServer: Received invalid JSON data from client {client_id}")
        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error handling client {client_id}: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            # Remove client from all subscriptions and connected clients
            self._remove_client(client_id)

    async def start(self) -> None:
        """Start the WebSocket server with retry logic."""
        attempts = 0

        # Store the event loop reference for thread-safe operations
        self.loop = asyncio.get_running_loop()

        # Initialize the message queue
        self.message_queue = asyncio.Queue()

        # Suppress EOFError logs from websockets library (caused by health monitor TCP checks)
        # These errors occur when the health monitor creates a raw TCP connection to check if the port is open,
        # but doesn't complete the WebSocket handshake. This is expected behavior and not an actual error.

        # Add a filter to suppress the specific EOF errors during handshake
        class HandshakeEOFFilter(logging.Filter):
            def filter(self, record):
                # Get the full message including exception info
                msg = record.getMessage()
                if record.exc_info:
                    import io
                    sio = io.StringIO()
                    traceback.print_exception(*record.exc_info, file=sio)
                    msg += '\n' + sio.getvalue()

                # Suppress "connection closed while reading HTTP request line" errors
                if 'connection closed while reading HTTP request line' in msg:
                    return False
                # Suppress "opening handshake failed" errors
                if 'opening handshake failed' in msg:
                    return False
                # Suppress "stream ends after 0 bytes" errors (health check connections)
                if 'stream ends after 0 bytes, before end of line' in msg:
                    return False
                # Suppress EOFError from health checks
                if 'EOFError' in msg and 'handshake' in msg.lower():
                    return False
                return True

        # Apply filter to websockets loggers
        for logger_name in ['websockets.server', 'websockets', 'websockets.protocol']:
            logger = logging.getLogger(logger_name)
            logger.addFilter(HandshakeEOFFilter())

        # Start sequence number periodic save task
        self.save_task = asyncio.create_task(self._periodic_save_sequence())

        # Start message queue processor
        self.queue_processor_task = asyncio.create_task(self._process_message_queue())

        # Start shared queue checker if available
        if self.shared_queue is not None:
            bt.logging.info(f"WebSocketServer: Starting shared queue monitor")
            self.shared_queue_task = asyncio.create_task(self._check_shared_queue())
        else:
            bt.logging.info(f"WebSocketServer: No shared queue provided, skipping monitor")

        # Start test order generator if enabled
        if self.send_test_positions and self.shared_queue is not None:
            bt.logging.info(f"WebSocketServer: Starting test order generator")
            self.test_positions_task = asyncio.create_task(self._generate_test_ws_positions())

        while attempts < self.max_reconnect_attempts or self.max_reconnect_attempts <= 0:
            try:
                # Create the server with appropriate handler
                bt.logging.info(f"WebSocketServer: Attempting to bind WebSocket server to {self.host}:{self.port} (attempt {attempts + 1})...")

                try:
                    self.server = await websockets.serve(
                        self.handle_client,
                        self.host,
                        self.port,
                        reuse_address=True,  # Allow reuse of the address
                        reuse_port=True  # Allow reuse of the port (on platforms that support it)

                    )
                    bt.logging.info(f"WebSocketServer: websockets.serve() completed successfully")
                except Exception as serve_error:
                    bt.logging.error(f"WebSocketServer: ERROR: websockets.serve() raised exception: {type(serve_error).__name__}: {serve_error}")
                    raise

                bt.logging.info(f"WebSocketServer: WebSocket server started at ws://{self.host}:{self.port}")
                bt.logging.info(f"WebSocketServer: Current sequence number: {self.sequence_number}")
                if self.shared_queue is not None:
                    bt.logging.info(f"WebSocketServer: Ready to receive messages from other processes via shared queue")

                # Keep the server running indefinitely
                bt.logging.info(f"WebSocketServer: Entering main event loop (await asyncio.Future())...")
                await asyncio.Future()

            except OSError as e:
                attempts += 1
                bt.logging.error(
                    f"WebSocketServer: Failed to start WebSocket server (attempt {attempts}/{self.max_reconnect_attempts}): OSError {e.errno}: {e}")
                bt.logging.error(f"WebSocketServer: Error details - errno: {e.errno}, strerror: {e.strerror}")

                if attempts < self.max_reconnect_attempts or self.max_reconnect_attempts <= 0:
                    bt.logging.info(f"WebSocketServer: Retrying in {self.reconnect_interval} seconds...")
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    bt.logging.error(f"WebSocketServer: Maximum retry attempts reached. Giving up.")
                    raise
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                if self.save_task:
                    self.save_task.cancel()
                    try:
                        await self.save_task
                    except asyncio.CancelledError:
                        pass

                if self.queue_processor_task:
                    self.queue_processor_task.cancel()
                    try:
                        await self.queue_processor_task
                    except asyncio.CancelledError:
                        pass

                if self.shared_queue_task:
                    self.shared_queue_task.cancel()
                    try:
                        await self.shared_queue_task
                    except asyncio.CancelledError:
                        pass

                if self.test_positions_task:
                    self.test_positions_task.cancel()
                    try:
                        await self.test_positions_task
                    except asyncio.CancelledError:
                        pass

                # Final save of sequence number
                self._save_sequence_number()
                raise
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Unexpected error starting WebSocket server: {type(e).__name__}: {e}")
                bt.logging.error(f"WebSocketServer: Full traceback:")
                bt.logging.error(f"WebSocketServer: {traceback.format_exc()}")
                # Save sequence number before crashing
                self._save_sequence_number()
                raise

    async def shutdown(self) -> None:
        """Gracefully shut down the WebSocket server."""
        bt.logging.info(f"WebSocketServer: Shutting down WebSocket server...")

        # Signal the shutdown to all tasks
        if self.shutdown_event:
            self.shutdown_event.set()

        # Close the server
        if self.server:
            self.server.close()
            try:
                await self.server.wait_closed()
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error while waiting for server to close: {e}")

        # Close all client connections
        for client_id, websocket in list(self.connected_clients.items()):
            try:
                await websocket.close()
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error closing client {client_id}: {e}")

        # Wait a bit for connections to close
        await asyncio.sleep(0.5)

        # Cancel all tasks with proper handling
        tasks_to_cancel = []

        if self.save_task and not self.save_task.done():
            self.save_task.cancel()
            tasks_to_cancel.append(self.save_task)

        if self.queue_processor_task and not self.queue_processor_task.done():
            self.queue_processor_task.cancel()
            tasks_to_cancel.append(self.queue_processor_task)

        if self.shared_queue_task and not self.shared_queue_task.done():
            self.shared_queue_task.cancel()
            tasks_to_cancel.append(self.shared_queue_task)

        if self.test_positions_task and not self.test_positions_task.done():
            self.test_positions_task.cancel()
            tasks_to_cancel.append(self.test_positions_task)

        # Wait for all tasks to complete cancellation with exception handling
        if tasks_to_cancel:
            bt.logging.info(f"WebSocketServer: Waiting for {len(tasks_to_cancel)} tasks to cancel...")
            for task in tasks_to_cancel:
                try:
                    # Use wait_for with a timeout to avoid hanging
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    # This is expected
                    pass
                except Exception as e:
                    bt.logging.error(f"WebSocketServer: Error cancelling task: {e}")

        # Final save of sequence number
        bt.logging.info(f"WebSocketServer: Saving final sequence number: {self.sequence_number}")
        self._save_sequence_number()
        bt.logging.info(f"WebSocketServer: WebSocket server shutdown complete")

    def run(self):
        """Start the server in the current process."""
        bt.logging.info(f"WebSocketServer: Starting WebSocket server...")
        setproctitle(f"vali_{self.__class__.__name__}")
        try:
            # Create a new event loop for this thread
            bt.logging.info(f"WebSocketServer: Creating new event loop...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create shutdown event
            self.shutdown_event = asyncio.Event()

            # Create main task
            bt.logging.info(f"WebSocketServer: Creating main task for start()...")
            main_task = loop.create_task(self.start())

            # Run the loop until keyboard interrupt
            bt.logging.info(f"WebSocketServer: Running event loop with run_until_complete()...")
            try:
                loop.run_until_complete(main_task)
                bt.logging.info(f"WebSocketServer: Event loop completed (this shouldn't happen unless shutting down)")
            except KeyboardInterrupt:
                bt.logging.info(f"WebSocketServer: Keyboard interrupt detected! Shutting down...")
                # Set shutdown event - this will signal all tasks to stop
                self.shutdown_event.set()

                # Create and run shutdown task
                shutdown_task = loop.create_task(self.shutdown())
                loop.run_until_complete(shutdown_task)

                # Cancel any remaining tasks
                for task in asyncio.all_tasks(loop):
                    if not task.done():
                        task.cancel()

                # Run until all tasks complete cancellation
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                bt.logging.info(f"WebSocketServer: All tasks have been stopped.")
            finally:
                # Close the loop
                loop.close()

        except Exception as e:
            bt.logging.error(f"WebSocketServer: FATAL ERROR in WebSocket server run(): {type(e).__name__}: {e}")
            bt.logging.error(f"WebSocketServer: Full traceback:")
            bt.logging.error(f"{traceback.format_exc()}")
            # Make sure sequence number is saved
            self._save_sequence_number()
            bt.logging.error(f"WebSocketServer: WebSocket server process exiting due to error")
            raise


# For standalone testing
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='WebSocket Server for PTN Data API')
    parser.add_argument('--api-keys-file', type=str, help='Path to the API keys file', default="api_keys.json")
    parser.add_argument('--host', type=str, help='Hostname to bind the server to', default="localhost")
    parser.add_argument('--port', type=int, help='Port to bind the server to', default=8765)
    parser.add_argument('--test-positions', action='store_true', help='Enable periodic test positions', default=True)
    parser.add_argument('--test-position-interval', type=int, help='Interval in seconds between test positions', default=5)
    parser.set_defaults(test_positions=True)

    args = parser.parse_args()

    # Create a test API keys file if it doesn't exist
    if not os.path.exists(args.api_keys_file):
        with open(args.api_keys_file, "w") as f:
            json.dump({"test_user": "test_key", "client": "abc"}, f)
        bt.logging.info(f"WebSocketServer: Created test API keys file at {args.api_keys_file}")

    # Create a manager instance for testing
    mp_manager = Manager()
    test_queue = mp_manager.Queue()

    bt.logging.info(f"WebSocketServer: Starting WebSocket server on {args.host}:{args.port}")
    bt.logging.info(f"WebSocketServer: Test positions: {'Enabled' if args.test_positions else 'Disabled'}")
    if args.test_positions:
        bt.logging.info(f"WebSocketServer: Test position interval: {args.test_position_interval} seconds")

    # Create and run the server
    server = WebSocketServer(
        api_keys_file=args.api_keys_file,
        shared_queue=test_queue,
        host=args.host,
        port=args.port,
        send_test_positions=args.test_positions,
        test_position_interval=args.test_position_interval
    )

    # Run the server
    server.run()