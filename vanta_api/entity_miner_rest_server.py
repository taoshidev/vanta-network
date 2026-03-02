# developer: jbonilla
# Copyright (c) 2025 Taoshi Inc
"""
Entity Miner REST Server - Gateway for Hyperliquid entity miners.

Connects to the validator WebSocket as an entity-authenticated client,
receives dashboard data and rejection notifications for all subaccounts,
and exposes REST/SSE endpoints for Chrome extensions and other consumers.

Endpoints:
    GET  /api/hl/<hl_address>/dashboard  - Cached dashboard data
    GET  /api/hl/<hl_address>/events     - Ring buffer of order events
    GET  /api/hl/<hl_address>/stream     - SSE real-time stream
    POST /api/create-subaccount          - Create standard subaccount
    POST /api/create-hl-subaccount       - Create HL-linked subaccount
    GET  /api/health                     - Health check
"""
import asyncio
import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Set

import bittensor as bt
from flask import jsonify, request, Response

from miner_config import MinerConfig
from vali_objects.utils.vali_utils import ValiUtils
from vanta_api.base_rest_server import BaseRestServer

try:
    import websockets
except ImportError:
    websockets = None

try:
    from bittensor_wallet import Wallet
except ImportError:
    Wallet = None


# ==================== Data Classes ====================

@dataclass
class OrderEvent:
    """Represents a single order event (accepted or rejected)."""
    timestamp_ms: int
    hl_address: str
    trade_pair: str
    order_type: str
    status: str  # "accepted" | "rejected"
    error_message: str = ""
    fill_hash: str = ""
    synthetic_hotkey: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class OrderEventStore:
    """Bounded ring buffer of OrderEvents per HL address."""

    MAX_EVENTS_PER_ADDRESS = 100

    def __init__(self):
        self._events: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def add(self, event: OrderEvent) -> None:
        with self._lock:
            addr = event.hl_address
            if addr not in self._events:
                self._events[addr] = deque(maxlen=self.MAX_EVENTS_PER_ADDRESS)
            self._events[addr].append(event)

    def get_events(self, hl_address: str, since_ms: int = 0) -> list:
        """Get events for an address, optionally filtered by timestamp."""
        with self._lock:
            events = self._events.get(hl_address, deque())
            if since_ms > 0:
                return [e.to_dict() for e in events if e.timestamp_ms > since_ms]
            return [e.to_dict() for e in events]


# ==================== Entity Miner REST Server ====================

class EntityMinerRestServer(BaseRestServer):
    """
    Gateway server for entity miners.

    Connects to the validator WebSocket using entity hotkey auth,
    receives dashboard + rejection data, and exposes REST/SSE endpoints.
    """

    def __init__(self, api_keys_file, flask_host="0.0.0.0", flask_port=8089,
                 slack_notifier=None, **kwargs):
        self.slack_notifier = slack_notifier

        # Internal state (initialized before super().__init__ calls _initialize_clients)
        self._event_store = OrderEventStore()
        self._dashboard_cache: Dict[str, dict] = {}  # hl_address -> latest dashboard
        self._hl_to_synthetic: Dict[str, str] = {}  # hl_address -> synthetic_hotkey
        self._synthetic_to_hl: Dict[str, str] = {}  # synthetic_hotkey -> hl_address

        # SSE subscriber tracking: hl_address -> set of Queue objects
        self._sse_subscribers: Dict[str, Set[queue.Queue]] = {}
        self._sse_lock = threading.Lock()

        # WebSocket connection state
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_connected = False
        self._ws_stop_event = threading.Event()

        # Wallet/secrets (loaded in _initialize_clients)
        self._hotkey = None
        self._coldkey = None
        self._validator_url = None
        self._validator_ws_host = None
        self._validator_ws_port = None

        super().__init__(
            api_keys_file=api_keys_file,
            service_name="EntityMinerRestServer",
            refresh_interval=15,
            metrics_interval_minutes=5,
            flask_host=flask_host,
            flask_port=flask_port,
            **kwargs
        )

    def _initialize_clients(self, **kwargs):
        """Load wallet secrets and start WebSocket listener."""
        try:
            secrets = ValiUtils.get_secrets(secrets_path=MinerConfig.get_secrets_file_path())
            wallet_name = secrets.get('wallet_name')
            wallet_hotkey = secrets.get('wallet_hotkey')
            wallet_password = ValiUtils.get_secret('wallet_password', secrets_path=MinerConfig.get_secrets_file_path())
            self._validator_url = secrets.get('validator_url')
            self._validator_ws_host = secrets.get('validator_ws_host')
            self._validator_ws_port = int(secrets.get('validator_ws_port', 8765))

            if Wallet and wallet_name and wallet_hotkey:
                wallet = Wallet(name=wallet_name, hotkey=wallet_hotkey)
                self._coldkey = wallet.get_coldkey(password=wallet_password)
                self._hotkey = wallet.hotkey
                del wallet_password
                print(f"[ENTITY-GW-INIT] Wallet loaded: {self._hotkey.ss58_address}")
            else:
                print(f"[ENTITY-GW-INIT] WARNING: Could not load wallet")
        except Exception as e:
            bt.logging.error(f"[ENTITY-GW-INIT] Failed to load wallet secrets: {e}")

        # Load HL address mappings from validator
        self._load_hl_mappings()

        # Start WebSocket listener thread
        self._start_ws_listener()

    def _register_routes(self):
        """Register entity miner gateway endpoints."""
        self.app.route("/api/hl/<hl_address>/dashboard", methods=["GET"])(self.dashboard_endpoint)
        self.app.route("/api/hl/<hl_address>/events", methods=["GET"])(self.events_endpoint)
        self.app.route("/api/hl/<hl_address>/stream", methods=["GET"])(self.stream_endpoint)
        self.app.route("/api/create-subaccount", methods=["POST"])(self.create_subaccount_endpoint)
        self.app.route("/api/create-hl-subaccount", methods=["POST"])(self.create_hl_subaccount_endpoint)
        self.app.route("/api/health", methods=["GET"])(self.health_endpoint)
        print(f"[ENTITY-GW-INIT] 6 endpoints registered")

    # ==================== HL Address Mapping ====================

    def _load_hl_mappings(self):
        """Load HL address -> synthetic hotkey mappings from validator."""
        if not self._validator_url or not self._hotkey:
            return

        try:
            import requests
            resp = requests.get(
                f"{self._validator_url}/entity/data",
                params={"entity_hotkey": self._hotkey.ss58_address},
                timeout=15
            )
            if resp.status_code != 200:
                bt.logging.warning(f"[ENTITY-GW] Failed to load entity data: {resp.status_code}")
                return

            data = resp.json()
            entity_data = data.get("entity_data", data)
            subaccounts = entity_data.get("subaccounts", {})

            for sub_id, sub_info in subaccounts.items():
                hl_addr = sub_info.get("hl_address")
                synthetic = sub_info.get("synthetic_hotkey", f"{self._hotkey.ss58_address}_{sub_id}")
                if hl_addr:
                    self._hl_to_synthetic[hl_addr] = synthetic
                    self._synthetic_to_hl[synthetic] = hl_addr

            bt.logging.info(f"[ENTITY-GW] Loaded {len(self._hl_to_synthetic)} HL address mappings")
        except Exception as e:
            bt.logging.error(f"[ENTITY-GW] Error loading HL mappings: {e}")

    # ==================== WebSocket Listener ====================

    def _start_ws_listener(self):
        """Start the WebSocket listener thread."""
        if not self._hotkey or not self._validator_ws_host:
            bt.logging.warning("[ENTITY-GW] Cannot start WS listener: missing hotkey or ws host config")
            return

        self._ws_stop_event.clear()
        self._ws_thread = threading.Thread(
            target=self._run_ws_listener,
            daemon=True,
            name="entity-gw-ws"
        )
        self._ws_thread.start()
        bt.logging.info("[ENTITY-GW] WebSocket listener thread started")

    def _run_ws_listener(self):
        """Entry point for the WS listener thread — runs its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_listen_loop())
        except Exception as e:
            bt.logging.error(f"[ENTITY-GW] WS listener loop crashed: {e}")
        finally:
            loop.close()

    async def _ws_listen_loop(self):
        """Connect to validator WS, authenticate with entity hotkey, receive messages."""
        if websockets is None:
            bt.logging.error("[ENTITY-GW] websockets library not installed")
            return

        backoff_s = 1.0
        ws_url = f"ws://{self._validator_ws_host}:{self._validator_ws_port}"

        while not self._ws_stop_event.is_set():
            try:
                async with websockets.connect(ws_url, ping_interval=30) as ws:
                    # Authenticate with entity hotkey + signature
                    now_ms = int(time.time() * 1000)
                    nonce = str(uuid.uuid4())
                    message_dict = {
                        "entity_hotkey": self._hotkey.ss58_address,
                        "nonce": nonce,
                        "timestamp": now_ms
                    }
                    message_bytes = json.dumps(message_dict, sort_keys=True).encode('utf-8')
                    signature = self._hotkey.sign(message_bytes).hex()

                    auth_msg = {
                        "entity_hotkey": self._hotkey.ss58_address,
                        "timestamp": now_ms,
                        "nonce": nonce,
                        "signature": signature
                    }
                    await ws.send(json.dumps(auth_msg))

                    # Wait for auth response
                    auth_resp = json.loads(await ws.recv())
                    if auth_resp.get("status") != "success":
                        bt.logging.error(f"[ENTITY-GW] WS auth failed: {auth_resp.get('message')}")
                        await asyncio.sleep(backoff_s)
                        backoff_s = min(backoff_s * 2, 60.0)
                        continue

                    self._ws_connected = True
                    backoff_s = 1.0
                    bt.logging.info(
                        f"[ENTITY-GW] WS connected and authenticated "
                        f"(subscribed to {auth_resp.get('subscribed_subaccounts', 0)} subaccounts)")

                    # Message receive loop
                    while not self._ws_stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue

                        try:
                            msg = json.loads(raw)
                            self._handle_ws_message(msg)
                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                bt.logging.warning(f"[ENTITY-GW] WS connection error: {e}")
            finally:
                self._ws_connected = False

            if self._ws_stop_event.is_set():
                break

            bt.logging.info(f"[ENTITY-GW] WS reconnecting in {backoff_s:.1f}s...")
            await asyncio.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, 60.0)

    def _handle_ws_message(self, msg: dict):
        """Process incoming WS messages from the validator."""
        msg_type = msg.get("type")
        synthetic_hotkey = msg.get("synthetic_hotkey")

        if not synthetic_hotkey:
            return

        # Handle subscription_status before HL address resolution
        # (new subaccounts won't be in the mapping yet)
        if msg_type == "subscription_status":
            action = msg.get("action")
            if action == "new_subaccount_subscribed":
                # New subaccount was auto-subscribed — update mappings
                self._load_hl_mappings()
            return

        # Resolve HL address
        hl_address = self._synthetic_to_hl.get(synthetic_hotkey)
        if not hl_address:
            return

        if msg_type == "subaccount_dashboard":
            # Update dashboard cache
            data = msg.get("data", {})
            self._dashboard_cache[hl_address] = {
                "timestamp_ms": msg.get("timestamp", int(time.time() * 1000)),
                "synthetic_hotkey": synthetic_hotkey,
                "hl_address": hl_address,
                **data
            }
            # Push to SSE
            self._push_sse(hl_address, {"type": "dashboard", "data": self._dashboard_cache[hl_address]})

        elif msg_type == "error":
            data = msg.get("data", {})
            error_msg = data.get("error_msg", "Unknown error")

            # Store event
            event = OrderEvent(
                timestamp_ms=msg.get("timestamp", int(time.time() * 1000)),
                hl_address=hl_address,
                trade_pair=data.get("trade_pair", ""),
                order_type=data.get("order_type", ""),
                status="rejected",
                error_message=error_msg,
                synthetic_hotkey=synthetic_hotkey
            )
            self._event_store.add(event)

            # Push to SSE
            self._push_sse(hl_address, {"type": "event", "data": event.to_dict()})

    # ==================== SSE ====================

    def _push_sse(self, hl_address: str, data: dict):
        """Push data to all SSE subscribers for a given HL address."""
        with self._sse_lock:
            subscribers = self._sse_subscribers.get(hl_address, set()).copy()

        dead = []
        for q in subscribers:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)

        # Remove dead subscribers
        if dead:
            with self._sse_lock:
                for q in dead:
                    self._sse_subscribers.get(hl_address, set()).discard(q)

    def _subscribe_sse(self, hl_address: str) -> queue.Queue:
        """Register an SSE subscriber for an HL address."""
        q = queue.Queue(maxsize=50)
        with self._sse_lock:
            if hl_address not in self._sse_subscribers:
                self._sse_subscribers[hl_address] = set()
            self._sse_subscribers[hl_address].add(q)
        return q

    def _unsubscribe_sse(self, hl_address: str, q: queue.Queue):
        """Unregister an SSE subscriber."""
        with self._sse_lock:
            if hl_address in self._sse_subscribers:
                self._sse_subscribers[hl_address].discard(q)
                if not self._sse_subscribers[hl_address]:
                    del self._sse_subscribers[hl_address]

    # ==================== Endpoints ====================

    def dashboard_endpoint(self, hl_address):
        """GET /api/hl/<hl_address>/dashboard - Return cached dashboard data."""
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        dashboard = self._dashboard_cache.get(hl_address)
        if not dashboard:
            return jsonify({'status': 'no_data', 'hl_address': hl_address}), 404

        return jsonify(dashboard), 200

    def events_endpoint(self, hl_address):
        """GET /api/hl/<hl_address>/events?since=<ms> - Return order events."""
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        since_ms = request.args.get('since', 0, type=int)
        events = self._event_store.get_events(hl_address, since_ms=since_ms)
        return jsonify({'hl_address': hl_address, 'events': events, 'count': len(events)}), 200

    def stream_endpoint(self, hl_address):
        """GET /api/hl/<hl_address>/stream - SSE real-time stream."""
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        subscriber_queue = self._subscribe_sse(hl_address)

        def event_stream():
            try:
                while True:
                    try:
                        data = subscriber_queue.get(timeout=30)
                        yield f"data: {json.dumps(data)}\n\n"
                    except queue.Empty:
                        # Send keepalive heartbeat
                        yield f": heartbeat\n\n"
            except GeneratorExit:
                pass
            finally:
                self._unsubscribe_sse(hl_address, subscriber_queue)

        return Response(
            event_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )

    def create_subaccount_endpoint(self):
        """POST /api/create-subaccount - Create a standard subaccount via validator."""
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        return self._proxy_subaccount_creation("/entity/create-subaccount")

    def create_hl_subaccount_endpoint(self):
        """POST /api/create-hl-subaccount - Create an HL-linked subaccount via validator."""
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        return self._proxy_subaccount_creation("/entity/create-hl-subaccount")

    def _proxy_subaccount_creation(self, endpoint: str):
        """
        Proxy subaccount creation to the validator.

        Signs the request with the entity coldkey and forwards to the validator REST API.
        """
        import requests as http_requests
        start_time = time.time()

        try:
            request_data = request.get_json()
            if not request_data:
                return jsonify({'status': 'error', 'message': 'Missing JSON body'}), 400
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Invalid request: {e}'}), 400

        if not self._coldkey or not self._hotkey or not self._validator_url:
            return jsonify({'status': 'error', 'message': 'Wallet not configured'}), 500

        # Build and sign payload
        try:
            asset_class = request_data.get("asset_class", "crypto")
            account_size = float(request_data.get("account_size", 0))
            admin = request_data.get("admin", False)
            hl_address = request_data.get("hl_address")
            payout_address = request_data.get("payout_address")

            message_dict = {
                "account_size": account_size,
                "admin": admin,
                "entity_coldkey": self._coldkey.ss58_address,
                "entity_hotkey": self._hotkey.ss58_address
            }
            if hl_address:
                message_dict["hl_address"] = hl_address
                if payout_address:
                    message_dict["payout_address"] = payout_address
            else:
                message_dict["asset_class"] = asset_class

            message = json.dumps(message_dict, sort_keys=True).encode('utf-8')
            signature = self._coldkey.sign(message).hex()

            payload = {
                "entity_hotkey": self._hotkey.ss58_address,
                "entity_coldkey": self._coldkey.ss58_address,
                "account_size": account_size,
                "admin": admin,
                "signature": signature,
                "version": "2.0.0"
            }
            if hl_address:
                payload["hl_address"] = hl_address
                if payout_address:
                    payload["payout_address"] = payout_address
            else:
                payload["asset_class"] = asset_class

        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Signing error: {e}'}), 500

        # Forward to validator
        try:
            resp = http_requests.post(
                f"{self._validator_url}{endpoint}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )

            try:
                response_data = resp.json()
            except json.JSONDecodeError:
                return jsonify({'status': 'error', 'message': 'Invalid response from validator'}), 500

            if resp.status_code == 200:
                # Update HL mappings if this was an HL subaccount
                subaccount = response_data.get('subaccount', {})
                new_hl = subaccount.get('hl_address') or hl_address
                synthetic = subaccount.get('synthetic_hotkey')
                if new_hl and synthetic:
                    self._hl_to_synthetic[new_hl] = synthetic
                    self._synthetic_to_hl[synthetic] = new_hl

                if self.slack_notifier:
                    elapsed_s = time.time() - start_time
                    self.slack_notifier.send_message(
                        f"Subaccount created: {subaccount.get('synthetic_hotkey')} "
                        f"({subaccount.get('asset_class')}, ${subaccount.get('account_size', 0):,.2f}) "
                        f"in {elapsed_s:.2f}s",
                        level="success",
                        bypass_cooldown=True
                    )
                return jsonify(response_data), 200
            else:
                return jsonify(response_data), resp.status_code

        except http_requests.exceptions.Timeout:
            return jsonify({'status': 'error', 'message': 'Validator request timed out'}), 504
        except http_requests.exceptions.ConnectionError:
            return jsonify({'status': 'error', 'message': 'Cannot connect to validator'}), 503
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Validator error: {e}'}), 500

    def health_endpoint(self):
        """GET /api/health - Health check."""
        return jsonify({
            'status': 'healthy',
            'service': 'EntityMinerRestServer',
            'ws_connected': self._ws_connected,
            'hl_addresses_tracked': len(self._hl_to_synthetic),
            'dashboard_cache_size': len(self._dashboard_cache),
            'sse_subscribers': sum(len(s) for s in self._sse_subscribers.values()),
            'timestamp': time.time()
        }), 200

    def shutdown(self):
        """Gracefully shutdown the gateway."""
        self._ws_stop_event.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)
        self.stop_flask_server()
        bt.logging.info("[ENTITY-GW] Shutdown complete")
