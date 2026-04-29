# developer: jbonilla
# Copyright (c) 2025 Taoshi Inc
"""
Entity Miner REST Server - Gateway for Hyperliquid entity miners.

Connects to the validator WebSocket as an entity-authenticated client,
receives dashboard data and rejection notifications for all subaccounts,
and exposes REST/SSE endpoints for Chrome extensions and other consumers.

Inherits from MinerRestServer:
    POST /api/submit-order               - Synchronous order submission
    GET  /api/order-status/<order_uuid>   - Query order status

Entity-specific endpoints:
    GET  /api/hl/<hl_address>/dashboard  - Cached dashboard data
    GET  /api/hl/<hl_address>/events     - Ring buffer of order events
    GET  /api/hl/<hl_address>/stream     - SSE real-time stream
    POST /api/create-subaccount          - Create standard subaccount
    POST /api/create-hl-subaccount       - Create HL-linked subaccount
    GET  /api/health                     - Health check (extended with WS status)
"""
import asyncio
import json
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set

import bittensor as bt
from flask import jsonify, request, Response

from miner_config import MinerConfig
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from vanta_api.miner_rest_server import MinerRestServer

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

    @staticmethod
    def _normalize_hl_address(hl_address: str) -> str:
        if not isinstance(hl_address, str):
            return ""
        return hl_address.lower()

    def add(self, event: OrderEvent) -> None:
        with self._lock:
            addr = self._normalize_hl_address(event.hl_address)
            if not addr:
                return
            if addr not in self._events:
                self._events[addr] = deque(maxlen=self.MAX_EVENTS_PER_ADDRESS)
            self._events[addr].append(event)

    def get_events(self, hl_address: str, since_ms: int = 0) -> list:
        """Get events for an address, optionally filtered by timestamp."""
        with self._lock:
            normalized = self._normalize_hl_address(hl_address)
            events = self._events.get(normalized, deque())
            if since_ms > 0:
                return [e.to_dict() for e in events if e.timestamp_ms > since_ms]
            return [e.to_dict() for e in events]


# ==================== Entity Miner REST Server ====================

class EntityMinerRestServer(MinerRestServer):
    """
    Gateway server for entity miners.

    Extends MinerRestServer with entity-specific functionality:
    - WebSocket connection to validator for dashboard/rejection data
    - SSE real-time streaming to Chrome extensions and consumers
    - HL address mapping and subaccount management

    Inherits all MinerRestServer endpoints (submit-order, order-status)
    and adds entity-specific endpoints (dashboard, events, stream,
    create-subaccount, create-hl-subaccount).
    """

    DASHBOARD_CACHE_TTL_MS = 10_000
    MAPPING_REFRESH_TTL_MS = 5_000

    def __init__(self, api_keys_file, flask_host="0.0.0.0", flask_port=8088,
                 slack_notifier=None, prop_net_order_placer=None, **kwargs):
        # Internal state (initialized before super().__init__ calls _initialize_clients)
        self._event_store = OrderEventStore()
        self._dashboard_cache: Dict[str, dict] = {}  # hl_address -> latest dashboard
        self._hl_to_synthetic: Dict[str, str] = {}  # hl_address -> synthetic_hotkey
        self._synthetic_to_hl: Dict[str, str] = {}  # synthetic_hotkey -> hl_address
        self._dashboard_cache_updated_ms: Dict[str, int] = {}  # hl_address -> cache write time
        self._mapping_last_refresh_ms: Dict[str, int] = {}  # hl_address -> last validator mapping refresh time

        # SSE subscriber tracking: hl_address -> set of Queue objects
        self._sse_subscribers: Dict[str, Set[queue.Queue]] = {}
        self._sse_lock = threading.Lock()

        # WebSocket connection state
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_connected = False
        self._ws_stop_event = threading.Event()

        # Payment daemon state
        self._payment_thread: Optional[threading.Thread] = None
        self._payment_stop_event = threading.Event()
        self._payment_service = None
        self._payment_ledger = None

        # Max HL traders limit (loaded in _initialize_clients)
        self._max_hl_traders: Optional[int] = None

        # Wallet/secrets (loaded in _initialize_clients)
        self._hotkey = None
        self._coldkey = None
        self._validator_url = None
        self._validator_ws_url = None
        self._api_key = None
        self._mappings_file: Optional[str] = None

        super().__init__(
            prop_net_order_placer=prop_net_order_placer,
            api_keys_file=api_keys_file,
            service_name="EntityMinerRestServer",
            refresh_interval=15,
            metrics_interval_minutes=5,
            flask_host=flask_host,
            flask_port=flask_port,
            slack_notifier=slack_notifier,
            **kwargs
        )

    def _initialize_clients(self, **kwargs):
        """Load wallet secrets and start WebSocket listener."""
        super()._initialize_clients(**kwargs)
        try:
            secrets = ValiUtils.get_secrets(secrets_path=MinerConfig.get_secrets_file_path())
            wallet_name = secrets.get('wallet_name')
            wallet_hotkey = secrets.get('wallet_hotkey')
            wallet_password = ValiUtils.get_secret('wallet_password', secrets_path=MinerConfig.get_secrets_file_path())
            self._validator_url = secrets.get('validator_url')
            self._validator_ws_url = secrets.get('validator_ws_url')
            self._api_key = secrets.get('validator_api_key')

            if Wallet and wallet_name and wallet_hotkey:
                wallet = Wallet(name=wallet_name, hotkey=wallet_hotkey)
                self._coldkey = wallet.get_coldkey(password=wallet_password)
                self._hotkey = wallet.hotkey
                del wallet_password
                print(f"[ENTITY-GW-INIT] Wallet loaded: {self._hotkey.ss58_address}")
            else:
                print(f"[ENTITY-GW-INIT] WARNING: Could not load wallet")

            # Derive mappings file path alongside the secrets file
            secrets_dir = os.path.dirname(MinerConfig.get_secrets_file_path())
            self._mappings_file = os.path.join(secrets_dir, "entity_hl_mappings.json")

            # Load max HL traders limit (env var takes precedence)
            max_hl_env = os.environ.get("MAX_HL_TRADERS")
            max_hl_secret = secrets.get("max_hl_traders")
            raw_max_hl = max_hl_env if max_hl_env is not None else max_hl_secret
            if raw_max_hl is not None:
                try:
                    self._max_hl_traders = int(raw_max_hl)
                    print(f"[ENTITY-GW-INIT] Max HL traders limit set to {self._max_hl_traders}")
                except (ValueError, TypeError):
                    bt.logging.warning(f"[ENTITY-GW-INIT] Invalid max_hl_traders value: {raw_max_hl}, ignoring")
        except Exception as e:
            bt.logging.error(f"[ENTITY-GW-INIT] Failed to load wallet secrets: {e}")

        # Load HL address mappings from local persistence file
        self._load_hl_mappings()

        # Send endpoint URL to validator if configured
        self._send_endpoint_url(secrets)

        # Start WebSocket listener thread
        self._start_ws_listener()

        # Initialize USDC payment daemon if configured
        self._initialize_payment_daemon(secrets)

    # ==================== Payment Daemon ====================

    def _initialize_payment_daemon(self, secrets: dict):
        """Initialize USDC payment service and start daemon thread if configured."""
        try:
            # Check if auto payouts are enabled
            enable_auto_payouts = (
                os.environ.get("ENABLE_AUTO_PAYOUTS", "").lower() == "true"
                or secrets.get("enable_auto_payouts", False)
            )
            if not enable_auto_payouts:
                bt.logging.info("[ENTITY-GW] Auto payouts not enabled (set enable_auto_payouts in secrets)")
                return

            # Load required secrets
            usdc_private_key = os.environ.get("USDC_PRIVATE_KEY") or secrets.get("usdc_private_key")
            if not usdc_private_key:
                bt.logging.warning("[ENTITY-GW] USDC_PRIVATE_KEY not configured, payment daemon disabled")
                return

            usdc_rpc_url = (
                os.environ.get("USDC_RPC_URL")
                or secrets.get("usdc_rpc_url")
                or MinerConfig.BASE_DEFAULT_RPC
            )
            validator_payout_api_key = (
                os.environ.get("VALIDATOR_PAYOUT_API_KEY")
                or secrets.get("validator_payout_api_key")
            )
            if not validator_payout_api_key:
                bt.logging.warning("[ENTITY-GW] VALIDATOR_PAYOUT_API_KEY not configured, payment daemon disabled")
                return

            if not self._validator_url:
                bt.logging.warning("[ENTITY-GW] validator_url not configured, payment daemon disabled")
                return

            from entity_management.payment_ledger import PaymentLedger
            from entity_management.usdc_payment_service import USDCPaymentService

            self._payment_ledger = PaymentLedger(MinerConfig.get_payment_ledger_file_path())
            self._payment_service = USDCPaymentService(
                private_key=usdc_private_key,
                rpc_url=usdc_rpc_url,
                validator_url=self._validator_url,
                validator_api_key=validator_payout_api_key,
                payment_ledger=self._payment_ledger,
                slack_notifier=self.slack_notifier
            )

            # Check for pending payments from previous run
            self._payment_service.check_pending_payments()

            # Read schedule config
            self._payout_schedule_day = secrets.get("payout_schedule_day", "sunday").lower()
            self._payout_schedule_hour = int(secrets.get("payout_schedule_hour", 1))

            # Start daemon thread
            self._payment_stop_event.clear()
            self._payment_thread = threading.Thread(
                target=self._payment_daemon_loop,
                daemon=True,
                name="entity-gw-payment"
            )
            self._payment_thread.start()

            bt.logging.info(
                f"[ENTITY-GW] Payment daemon started: "
                f"wallet={self._payment_service.get_wallet_address()}, "
                f"schedule={self._payout_schedule_day} {self._payout_schedule_hour:02d}:00 UTC"
            )
        except Exception as e:
            bt.logging.error(f"[ENTITY-GW] Failed to initialize payment daemon: {e}")

    def _payment_daemon_loop(self):
        """Main loop for the payment daemon thread."""
        DAY_NAMES = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        target_day = DAY_NAMES.get(self._payout_schedule_day, 6)

        while not self._payment_stop_event.is_set():
            try:
                now = datetime.now(timezone.utc)

                # Calculate next run time
                days_ahead = target_day - now.weekday()
                if days_ahead < 0 or (days_ahead == 0 and now.hour >= self._payout_schedule_hour):
                    days_ahead += 7

                next_run = now.replace(
                    hour=self._payout_schedule_hour, minute=0, second=0, microsecond=0
                ) + timedelta(days=days_ahead)

                sleep_seconds = (next_run - now).total_seconds()
                bt.logging.info(
                    f"[USDC_PAYMENT_DAEMON] Next payout run at {next_run.isoformat()} "
                    f"(in {sleep_seconds / 3600:.1f} hours)"
                )

                # Sleep until next run (check stop event every 60s)
                while sleep_seconds > 0 and not self._payment_stop_event.is_set():
                    wait_time = min(sleep_seconds, 60.0)
                    self._payment_stop_event.wait(timeout=wait_time)
                    sleep_seconds -= wait_time

                if self._payment_stop_event.is_set():
                    break

                # Calculate previous week's period (Sunday 00:00 UTC -> Sunday 00:00 UTC)
                period_end = now.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) - timedelta(days=now.weekday()) + timedelta(days=(target_day - now.weekday()) % 7)
                # Adjust: period_end is the start of today's target day
                # period_start is 7 days before that
                if period_end > now:
                    period_end -= timedelta(days=7)
                period_start = period_end - timedelta(days=7)

                start_ms = int(period_start.timestamp() * 1000)
                end_ms = int(period_end.timestamp() * 1000)

                bt.logging.info(
                    f"[USDC_PAYMENT_DAEMON] Running payout for period "
                    f"{period_start.isoformat()} to {period_end.isoformat()}"
                )

                entity_hotkey = self._hotkey.ss58_address if self._hotkey else None
                if not entity_hotkey:
                    bt.logging.error("[USDC_PAYMENT_DAEMON] No hotkey available, skipping payout run")
                    continue

                result = self._payment_service.process_payouts(
                    entity_hotkey=entity_hotkey,
                    start_time_ms=start_ms,
                    end_time_ms=end_ms
                )

                bt.logging.info(
                    f"[USDC_PAYMENT_DAEMON] Payout run {result.run_id} complete: "
                    f"{result.successful_count} successful, {result.failed_count} failed, "
                    f"${result.total_usd_paid:.2f} paid"
                )

            except Exception as e:
                bt.logging.error(f"[USDC_PAYMENT_DAEMON] Error in payment daemon loop: {e}")
                # Sleep 5 minutes before retrying on error
                self._payment_stop_event.wait(timeout=300)

    @staticmethod
    def _normalize_hl_address(hl_address: str) -> str:
        if not isinstance(hl_address, str):
            return ""
        return hl_address.lower()

    def _register_routes(self):
        """Register miner endpoints (inherited) plus entity-specific endpoints."""
        # Register all MinerRestServer routes (submit-order, order-status, health).
        # health_endpoint is overridden in this class, so the parent's registration
        # will use our version via Python MRO.
        super()._register_routes()

        # Register entity-specific endpoints
        self.app.route("/api/hl/<hl_address>/dashboard", methods=["GET"])(self.dashboard_endpoint)
        self.app.route("/api/hl/<hl_address>/events", methods=["GET"])(self.events_endpoint)
        self.app.route("/api/hl/<hl_address>/stream", methods=["GET"])(self.stream_endpoint)
        self.app.route("/api/create-subaccount", methods=["POST"])(self.create_subaccount_endpoint)
        self.app.route("/api/create-hl-subaccount", methods=["POST"])(self.create_subaccount_endpoint)
        print(f"[ENTITY-GW-INIT] 8 endpoints registered (3 inherited + 5 entity-specific)")

    # ==================== HL Address Mapping ====================

    def _load_hl_mappings(self):
        """Load HL address -> synthetic hotkey mappings from local persistence file."""
        if not self._mappings_file:
            return

        try:
            if not os.path.exists(self._mappings_file):
                bt.logging.info(f"[ENTITY-GW] No mappings file found at {self._mappings_file}, starting fresh")
                return

            with open(self._mappings_file, "r") as f:
                data = json.load(f)

            raw_hl_to_synthetic = data.get("hl_to_synthetic", {})
            raw_synthetic_to_hl = data.get("synthetic_to_hl", {})

            # Normalize stored addresses so lookup is case-insensitive.
            normalized_hl_to_synthetic = {}
            normalized_synthetic_to_hl = {}

            for hl_addr, synthetic in raw_hl_to_synthetic.items():
                normalized_hl = self._normalize_hl_address(hl_addr)
                if normalized_hl and synthetic:
                    normalized_hl_to_synthetic[normalized_hl] = synthetic
                    normalized_synthetic_to_hl[synthetic] = normalized_hl

            for synthetic, hl_addr in raw_synthetic_to_hl.items():
                normalized_hl = self._normalize_hl_address(hl_addr)
                if normalized_hl and synthetic:
                    normalized_synthetic_to_hl[synthetic] = normalized_hl
                    normalized_hl_to_synthetic[normalized_hl] = synthetic

            self._hl_to_synthetic = normalized_hl_to_synthetic
            self._synthetic_to_hl = normalized_synthetic_to_hl
            bt.logging.info(f"[ENTITY-GW] Loaded {len(self._hl_to_synthetic)} HL address mappings from disk")
        except Exception as e:
            bt.logging.error(f"[ENTITY-GW] Error loading HL mappings: {e}")

    def _save_hl_mappings(self):
        """Persist HL address -> synthetic hotkey mappings to disk."""
        if not self._mappings_file:
            return

        try:
            with open(self._mappings_file, "w") as f:
                json.dump({
                    "hl_to_synthetic": self._hl_to_synthetic,
                    "synthetic_to_hl": self._synthetic_to_hl,
                }, f, indent=2)
        except Exception as e:
            bt.logging.error(f"[ENTITY-GW] Error saving HL mappings: {e}")

    def _apply_hl_mappings(self, hl_mappings: dict):
        """Apply HL address -> synthetic hotkey mappings received from validator (e.g. WS auth response)."""
        mapping_changed = False
        for hl_addr, synthetic in hl_mappings.items():
            normalized_hl = self._normalize_hl_address(hl_addr)
            if not normalized_hl or not synthetic:
                continue
            mapping_changed = self._set_hl_mapping(normalized_hl, synthetic, source="ws_auth") or mapping_changed
        if mapping_changed:
            self._save_hl_mappings()
        bt.logging.info(f"[ENTITY-GW] Applied {len(hl_mappings)} HL mappings from validator")

    def _set_hl_mapping(self, hl_address: str, synthetic_hotkey: str, source: str = "unknown") -> bool:
        """Update HL<->synthetic mapping and evict stale dashboard on reassignment."""
        if not hl_address or not synthetic_hotkey:
            return False

        old_synthetic = self._hl_to_synthetic.get(hl_address)
        old_hl_for_synthetic = self._synthetic_to_hl.get(synthetic_hotkey)
        mapping_changed = (
            old_synthetic is not None and old_synthetic != synthetic_hotkey
        ) or (
            old_hl_for_synthetic is not None and old_hl_for_synthetic != hl_address
        )

        if old_synthetic and old_synthetic != synthetic_hotkey:
            self._synthetic_to_hl.pop(old_synthetic, None)
            self._evict_dashboard_cache(hl_address)
            bt.logging.info(
                f"[ENTITY-GW] HL mapping reassigned ({source}): {hl_address} {old_synthetic} -> {synthetic_hotkey}"
            )

        if old_hl_for_synthetic and old_hl_for_synthetic != hl_address:
            self._hl_to_synthetic.pop(old_hl_for_synthetic, None)
            self._evict_dashboard_cache(old_hl_for_synthetic)
            bt.logging.info(
                f"[ENTITY-GW] Synthetic reassigned ({source}): {synthetic_hotkey} {old_hl_for_synthetic} -> {hl_address}"
            )

        self._hl_to_synthetic[hl_address] = synthetic_hotkey
        self._synthetic_to_hl[synthetic_hotkey] = hl_address
        return mapping_changed

    def _evict_dashboard_cache(self, hl_address: str):
        """Remove any cached dashboard payload for an HL address."""
        self._dashboard_cache.pop(hl_address, None)
        self._dashboard_cache_updated_ms.pop(hl_address, None)

    def _fetch_validator_hl_trader(self, hl_address: str) -> Optional[dict]:
        """Fetch canonical HL trader snapshot from validator."""
        if not self._validator_url:
            return None

        try:
            import requests as http_requests
            response = http_requests.get(f"{self._validator_url}/hl-traders/{hl_address}", timeout=8)
            if response.status_code != 200:
                return None
            return response.json()
        except Exception as e:
            bt.logging.warning(f"[ENTITY-GW] Failed to fetch validator HL trader for {hl_address}: {e}")
            return None

    @staticmethod
    def _extract_synthetic_from_validator_payload(payload: dict) -> Optional[str]:
        """Extract synthetic hotkey from validator /hl-traders payload."""
        if not isinstance(payload, dict):
            return None

        dashboard = payload.get("dashboard", {})
        if isinstance(dashboard, dict):
            if dashboard.get("synthetic_hotkey"):
                return dashboard.get("synthetic_hotkey")
            subaccount_info = dashboard.get("subaccount_info", {})
            if isinstance(subaccount_info, dict) and subaccount_info.get("synthetic_hotkey"):
                return subaccount_info.get("synthetic_hotkey")

        return payload.get("synthetic_hotkey")

    def _refresh_dashboard_from_validator(self, hl_address: str) -> Optional[dict]:
        """
        Refresh dashboard + mapping from validator canonical source.
        Returns a normalized dashboard payload or None if refresh failed.
        """
        payload = self._fetch_validator_hl_trader(hl_address)
        now_ms = int(time.time() * 1000)
        self._mapping_last_refresh_ms[hl_address] = now_ms
        if not payload:
            return None

        active_synthetic = self._extract_synthetic_from_validator_payload(payload)
        if active_synthetic:
            mapping_changed = self._set_hl_mapping(hl_address, active_synthetic, source="validator_poll")
            if mapping_changed:
                self._save_hl_mappings()

        dashboard = payload.get("dashboard", {})
        normalized_payload = None
        if isinstance(dashboard, dict):
            subaccount_info = dashboard.get("subaccount_info", {})
            if isinstance(subaccount_info, dict) and subaccount_info:
                normalized_payload = {
                    "timestamp_ms": payload.get("timestamp", now_ms),
                    "synthetic_hotkey": active_synthetic or subaccount_info.get("synthetic_hotkey", ""),
                    "hl_address": hl_address,
                    **subaccount_info,
                }
            elif dashboard:
                normalized_payload = {
                    "timestamp_ms": payload.get("timestamp", now_ms),
                    "synthetic_hotkey": active_synthetic or dashboard.get("synthetic_hotkey", ""),
                    "hl_address": hl_address,
                    **dashboard,
                }

        if normalized_payload:
            self._dashboard_cache[hl_address] = normalized_payload
            self._dashboard_cache_updated_ms[hl_address] = now_ms

        return normalized_payload

    def _resolve_active_synthetic_hotkey(self, hl_address: str) -> Optional[str]:
        """Resolve active synthetic hotkey, refreshing from validator periodically."""
        now_ms = int(time.time() * 1000)
        cached_synthetic = self._hl_to_synthetic.get(hl_address)
        last_refresh_ms = self._mapping_last_refresh_ms.get(hl_address, 0)

        if cached_synthetic and (now_ms - last_refresh_ms) < self.MAPPING_REFRESH_TTL_MS:
            return cached_synthetic

        refreshed_dashboard = self._refresh_dashboard_from_validator(hl_address)
        if refreshed_dashboard:
            return refreshed_dashboard.get("synthetic_hotkey") or self._hl_to_synthetic.get(hl_address)
        return cached_synthetic

    # ==================== Endpoint URL Registration ====================

    def _send_endpoint_url(self, secrets: dict):
        """
        Send the entity miner's public endpoint URL to the validator at startup.

        Reads from ENTITY_MINER_ENDPOINT_URL env var, falling back to
        entity_endpoint_url in miner_secrets.json.

        Args:
            secrets: The loaded miner secrets dict
        """
        endpoint_url = os.environ.get("ENTITY_MINER_ENDPOINT_URL") or secrets.get("entity_endpoint_url")

        if not endpoint_url:
            bt.logging.info("[ENTITY-GW] No endpoint URL configured (set ENTITY_MINER_ENDPOINT_URL or entity_endpoint_url in secrets)")
            return

        if not self._coldkey or not self._hotkey or not self._validator_url:
            bt.logging.warning("[ENTITY-GW] Cannot send endpoint URL: wallet or validator_url not configured")
            return

        try:
            import requests as http_requests

            entity_hotkey = self._hotkey.ss58_address
            entity_coldkey = self._coldkey.ss58_address

            # Sign the message (sorted keys)
            message_dict = {
                "endpoint_url": endpoint_url,
                "entity_coldkey": entity_coldkey,
                "entity_hotkey": entity_hotkey
            }
            message = json.dumps(message_dict, sort_keys=True).encode('utf-8')
            signature = self._coldkey.sign(message).hex()

            payload = {
                "entity_hotkey": entity_hotkey,
                "entity_coldkey": entity_coldkey,
                "endpoint_url": endpoint_url,
                "signature": signature
            }

            resp = http_requests.post(
                f"{self._validator_url}/entity/set-endpoint",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if resp.status_code == 200:
                bt.logging.info(f"[ENTITY-GW] Endpoint URL registered: {endpoint_url}")
            else:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('error', resp.text)
                except json.JSONDecodeError:
                    error_msg = resp.text
                bt.logging.warning(f"[ENTITY-GW] Failed to register endpoint URL ({resp.status_code}): {error_msg}")

        except Exception as e:
            bt.logging.error(f"[ENTITY-GW] Error sending endpoint URL: {e}")

    # ==================== WebSocket Listener ====================

    def _start_ws_listener(self):
        """Start the WebSocket listener thread."""
        if not self._hotkey or not self._validator_ws_url:
            bt.logging.warning("[ENTITY-GW] Cannot start WS listener: missing hotkey or ws url config")
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
        ws_url = self._validator_ws_url

        while not self._ws_stop_event.is_set():
            try:
                extra_headers = {}
                if self._api_key:
                    extra_headers["Authorization"] = f"Bearer {self._api_key}"

                async with websockets.connect(ws_url, additional_headers=extra_headers, ping_interval=30) as ws:
                    # Wait for auth response (server sends it immediately after header validation)
                    auth_resp = json.loads(await ws.recv())
                    if auth_resp.get("status") != "success":
                        bt.logging.error(f"[ENTITY-GW] WS auth failed: {auth_resp.get('message')}")
                        await asyncio.sleep(backoff_s)
                        backoff_s = min(backoff_s * 2, 60.0)
                        continue

                    self._ws_connected = True
                    backoff_s = 1.0

                    # Populate HL mappings from auth response (validator sends these on every connect)
                    hl_mappings = auth_resp.get("hl_mappings", {})
                    if hl_mappings:
                        loop_ref = asyncio.get_running_loop()
                        await loop_ref.run_in_executor(None, self._apply_hl_mappings, hl_mappings)

                    bt.logging.info(
                        f"[ENTITY-GW] WS connected and authenticated "
                        f"(subscribed to {auth_resp.get('subscribed_subaccounts', 0)} subaccounts, "
                        f"hl_mappings={len(hl_mappings)})")

                    # Message receive loop
                    loop = asyncio.get_running_loop()
                    while not self._ws_stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue

                        try:
                            msg = json.loads(raw)
                            await loop.run_in_executor(None, self._handle_ws_message, msg)
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
            bt.logging.debug(
                f"[ENTITY-GW] Dropping WS message for {synthetic_hotkey}: "
                f"no HL mapping (known mappings={len(self._synthetic_to_hl)})"
            )
            return

        if msg_type == "subaccount_dashboard":
            data = msg.get("data", {})

            # Accepted fill event payloads are sent through the dashboard channel.
            order_event_data = data.get("order_event") if isinstance(data, dict) else None
            if isinstance(order_event_data, dict) and order_event_data.get("status") == "accepted":
                event = OrderEvent(
                    timestamp_ms=msg.get("timestamp", int(time.time() * 1000)),
                    hl_address=hl_address,
                    trade_pair=order_event_data.get("trade_pair", ""),
                    order_type=order_event_data.get("order_type", ""),
                    status="accepted",
                    fill_hash=order_event_data.get("fill_hash", ""),
                    synthetic_hotkey=synthetic_hotkey
                )
                self._event_store.add(event)
                self._push_sse(hl_address, {"type": "event", "data": event.to_dict()})
                return

            # Update dashboard cache
            self._dashboard_cache[hl_address] = {
                "timestamp_ms": msg.get("timestamp", int(time.time() * 1000)),
                "synthetic_hotkey": synthetic_hotkey,
                "hl_address": hl_address,
                **data
            }
            self._dashboard_cache_updated_ms[hl_address] = int(time.time() * 1000)
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
        """GET /api/hl/<hl_address>/dashboard - Return dashboard data with mapping/TTL reconciliation."""
        normalized_hl = self._normalize_hl_address(hl_address)
        now_ms = int(time.time() * 1000)
        active_synthetic = self._resolve_active_synthetic_hotkey(normalized_hl)
        dashboard = self._dashboard_cache.get(normalized_hl)
        cache_updated_ms = self._dashboard_cache_updated_ms.get(normalized_hl, 0)
        cache_is_fresh = dashboard is not None and (now_ms - cache_updated_ms) <= self.DASHBOARD_CACHE_TTL_MS
        cache_matches_mapping = (
            dashboard is not None and (
                not active_synthetic or dashboard.get("synthetic_hotkey") == active_synthetic
            )
        )

        # Force refresh when entry is stale or points to the wrong synthetic hotkey.
        if not cache_is_fresh or not cache_matches_mapping:
            refreshed = self._refresh_dashboard_from_validator(normalized_hl)
            if refreshed:
                return jsonify(refreshed), 200

            # Stale-if-error fallback only if mapping still matches or mapping is unknown.
            if dashboard and cache_matches_mapping:
                return jsonify(dashboard), 200

            return jsonify({
                'status': 'no_data',
                'hl_address': normalized_hl,
                'message': 'Dashboard cache invalidated and refresh failed'
            }), 404

        if not dashboard:
            refreshed = self._refresh_dashboard_from_validator(normalized_hl)
            if refreshed:
                return jsonify(refreshed), 200
            return jsonify({'status': 'no_data', 'hl_address': normalized_hl}), 404

        return jsonify(dashboard), 200

    def events_endpoint(self, hl_address):
        """GET /api/hl/<hl_address>/events?since=<ms> - Return order events (no API key required)."""
        since_ms = request.args.get('since', 0, type=int)
        normalized_hl = self._normalize_hl_address(hl_address)
        active_synthetic = self._resolve_active_synthetic_hotkey(normalized_hl)
        events = self._event_store.get_events(normalized_hl, since_ms=since_ms)
        if active_synthetic:
            events = [event for event in events if event.get("synthetic_hotkey") == active_synthetic]
        return jsonify({'hl_address': normalized_hl, 'events': events, 'count': len(events)}), 200

    def stream_endpoint(self, hl_address):
        """GET /api/hl/<hl_address>/stream - SSE real-time stream (no API key required)."""
        normalized_hl = self._normalize_hl_address(hl_address)
        subscriber_queue = self._subscribe_sse(normalized_hl)

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
                self._unsubscribe_sse(normalized_hl, subscriber_queue)

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
        """
        POST /api/create-subaccount (and /api/create-hl-subaccount) - Create a subaccount via validator.

        When hl_address is provided, creates an HL-linked subaccount whose trades are
        automatically forwarded from the HyperliquidTracker as Vanta signals.

        Request body (JSON) — standard:
        {
            "asset_class": "crypto" | "forex" | "equities",  // Required
            "account_size": float,                           // Required, must be > 0
            "admin": bool                                    // Optional, default false
        }

        Request body (JSON) — HL-linked:
        {
            "hl_address": "0x...",     // Required, 0x + 40 hex chars
            "account_size": float,     // Required, must be > 0
            "payout_address": "0x...", // Optional, EVM address for USDC payouts
            "admin": bool              // Optional, default false
        }
        """
        import requests as http_requests
        start_time = time.time()

        # 1. Validate API key
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # 2. Parse and validate request body
        try:
            request_data = request.get_json()
            if not request_data:
                return jsonify({'status': 'error', 'message': 'Invalid request: missing JSON body'}), 400

            if "account_size" not in request_data:
                return jsonify({'status': 'error', 'message': 'Missing required field: account_size'}), 400

            is_hl = 'hl_address' in request_data

            if is_hl:
                hl_address = request_data["hl_address"]
                payout_address = request_data.get("payout_address")
                asset_class = "crypto"
            else:
                hl_address = None
                payout_address = None
                if "asset_class" not in request_data:
                    return jsonify({'status': 'error', 'message': 'Missing required field: asset_class'}), 400
                asset_class = request_data["asset_class"]

            admin = request_data.get("admin")

            try:
                account_size = float(request_data["account_size"])
            except (ValueError, TypeError):
                return jsonify({'status': 'error', 'message': 'account_size must be a number'}), 400

            if admin is not None and not isinstance(admin, bool):
                return jsonify({'status': 'error', 'message': 'admin must be a boolean'}), 400

            if asset_class not in ["crypto", "forex", "equities"]:
                return jsonify({
                    'status': 'error',
                    'message': f"Invalid asset_class: {asset_class}. Must be 'crypto', 'forex', or 'equities'"
                }), 400

            if account_size <= 0:
                return jsonify({'status': 'error', 'message': 'account_size must be positive'}), 400

            if is_hl:
                if not isinstance(hl_address, str) or not re.match(ValiConfig.HL_ADDRESS_REGEX, hl_address):
                    return jsonify({
                        'status': 'error',
                        'message': 'hl_address must be a valid Hyperliquid address (0x followed by 40 hex characters)'
                    }), 400

                if payout_address is not None:
                    if not isinstance(payout_address, str) or not re.match(ValiConfig.HL_ADDRESS_REGEX, payout_address):
                        return jsonify({
                            'status': 'error',
                            'message': 'payout_address must be a valid EVM address (0x followed by 40 hex characters)'
                        }), 400

                # Check max HL traders limit
                if self._max_hl_traders is not None:
                    current_count = len(self._hl_to_synthetic)
                    if current_count >= self._max_hl_traders:
                        bt.logging.warning(
                            f"[ENTITY-GW] HL trader limit reached: {current_count}/{self._max_hl_traders}"
                        )
                        return jsonify({
                            'status': 'error',
                            'message': f'Maximum number of Hyperliquid traders ({self._max_hl_traders}) reached. '
                                       f'Cannot register more HL subaccounts.'
                        }), 403

        except Exception as e:
            bt.logging.error(f"Error parsing request body: {e}")
            return jsonify({'status': 'error', 'message': f'Invalid request: {str(e)}'}), 400

        # 3. Check wallet is configured
        if not self._coldkey or not self._hotkey or not self._validator_url:
            return jsonify({'status': 'error', 'message': 'Wallet not configured'}), 500

        # 4. Sign message
        try:
            message_dict = {
                "account_size": account_size,
                "asset_class": asset_class,
                "entity_coldkey": self._coldkey.ss58_address,
                "entity_hotkey": self._hotkey.ss58_address,
            }
            if admin is not None:
                message_dict["admin"] = admin
            if is_hl:
                message_dict["hl_address"] = hl_address
                if payout_address is not None:
                    message_dict["payout_address"] = payout_address
            message = json.dumps(message_dict, sort_keys=True).encode('utf-8')
            signature = self._coldkey.sign(message).hex()
        except Exception as e:
            bt.logging.error(f"Error signing message: {e}")
            return jsonify({'status': 'error', 'message': f'Wallet error: {str(e)}'}), 500

        # 5. Send request to validator
        try:
            payload = {
                "entity_hotkey": self._hotkey.ss58_address,
                "entity_coldkey": self._coldkey.ss58_address,
                "account_size": account_size,
                "asset_class": asset_class,
                "signature": signature,
                "version": "2.0.0"
            }
            if admin is not None:
                payload["admin"] = admin
            if is_hl:
                payload["hl_address"] = hl_address
                if payout_address is not None:
                    payload["payout_address"] = payout_address

            resp = http_requests.post(
                f"{self._validator_url}/entity/create-subaccount",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            elapsed_s = time.time() - start_time

            try:
                response_data = resp.json()
            except json.JSONDecodeError:
                return jsonify({'status': 'error', 'message': 'Invalid JSON response from validator'}), 500

            if resp.status_code == 200:
                subaccount = response_data.get('subaccount', {})

                if is_hl:
                    # Update local HL address mapping and persist to disk
                    synthetic = subaccount.get('synthetic_hotkey')
                    normalized_hl = self._normalize_hl_address(hl_address)
                    if synthetic and normalized_hl:
                        self._set_hl_mapping(normalized_hl, synthetic, source="create_subaccount")
                        self._save_hl_mappings()

                if self.slack_notifier:
                    from datetime import datetime, timezone
                    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                    if is_hl:
                        payout_line = f"Payout Address: {payout_address}\n" if payout_address else ""
                        msg = (
                            f"HL Subaccount created successfully!\n"
                            f"ID: {subaccount.get('subaccount_id')}\n"
                            f"UUID: {subaccount.get('subaccount_uuid')}\n"
                            f"Synthetic Hotkey: {subaccount.get('synthetic_hotkey')}\n"
                            f"HL Address: {hl_address}\n"
                            f"{payout_line}"
                            f"Asset Class: {subaccount.get('asset_class')}\n"
                            f"Account Size: ${subaccount.get('account_size', 0):,.2f}\n"
                            f"Message: {response_data.get('message', '')}\n"
                            f"Created: {timestamp}\n"
                            f"Time: {elapsed_s:.2f}s"
                        )
                    else:
                        msg = (
                            f"Subaccount created successfully!\n"
                            f"ID: {subaccount.get('subaccount_id')}\n"
                            f"UUID: {subaccount.get('subaccount_uuid')}\n"
                            f"Synthetic Hotkey: {subaccount.get('synthetic_hotkey')}\n"
                            f"Asset Class: {subaccount.get('asset_class')}\n"
                            f"Account Size: ${subaccount.get('account_size'):,.2f}\n"
                            f"Message: {response_data.get('message', '')}\n"
                            f"Created: {timestamp}\n"
                            f"Time: {elapsed_s:.2f}s"
                        )
                    self.slack_notifier.send_message(msg, level="success", bypass_cooldown=True)

                return jsonify(response_data), 200
            else:
                error_message = response_data.get('error', response_data.get('message', 'Unknown error from validator'))
                if self.slack_notifier:
                    hl_address_line = f"HL Address: {hl_address}\n" if is_hl else ""
                    self.slack_notifier.send_message(
                        f"Subaccount creation failed\n"
                        f"{hl_address_line}"
                        f"Asset Class: {asset_class}\n"
                        f"Account Size: ${account_size:,.2f}\n"
                        f"Error: {error_message}",
                        level="error"
                    )
                return jsonify({'status': 'error', 'message': error_message}), resp.status_code

        except http_requests.exceptions.Timeout:
            if self.slack_notifier:
                hl_address_line = f"HL Address: {hl_address}\n" if is_hl else ""
                self.slack_notifier.send_message(
                    f"Subaccount creation failed\n"
                    f"{hl_address_line}"
                    f"Asset Class: {asset_class}\n"
                    f"Account Size: ${account_size:,.2f}\n"
                    f"Error: Request to validator timed out",
                    level="error"
                )
            return jsonify({'status': 'error', 'message': 'Request to validator timed out'}), 504

        except http_requests.exceptions.ConnectionError:
            if self.slack_notifier:
                hl_address_line = f"HL Address: {hl_address}\n" if is_hl else ""
                self.slack_notifier.send_message(
                    f"Subaccount creation failed\n"
                    f"{hl_address_line}"
                    f"Asset Class: {asset_class}\n"
                    f"Account Size: ${account_size:,.2f}\n"
                    f"Error: Could not connect to validator",
                    level="error"
                )
            return jsonify({'status': 'error', 'message': 'Could not connect to validator'}), 503

        except Exception as e:
            bt.logging.error(f"Error communicating with validator: {e}")
            if self.slack_notifier:
                hl_address_line = f"HL Address: {hl_address}\n" if is_hl else ""
                self.slack_notifier.send_message(
                    f"Subaccount creation failed\n"
                    f"{hl_address_line}"
                    f"Asset Class: {asset_class}\n"
                    f"Account Size: ${account_size:,.2f}\n"
                    f"Error: {str(e)}",
                    level="error"
                )
            return jsonify({'status': 'error', 'message': f'Validator communication error: {str(e)}'}), 500

    def health_endpoint(self):
        """GET /api/health - Health check."""
        health = {
            'status': 'healthy',
            'service': 'EntityMinerRestServer',
            'ws_connected': self._ws_connected,
            'hl_addresses_tracked': len(self._hl_to_synthetic),
            'max_hl_traders': self._max_hl_traders,
            'dashboard_cache_size': len(self._dashboard_cache),
            'sse_subscribers': sum(len(s) for s in self._sse_subscribers.values()),
            'payment_daemon_active': self._payment_thread is not None and self._payment_thread.is_alive(),
            'timestamp': time.time()
        }
        return jsonify(health), 200

    def shutdown(self):
        """Gracefully shutdown the gateway."""
        self._ws_stop_event.set()
        self._payment_stop_event.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)
        if self._payment_thread and self._payment_thread.is_alive():
            self._payment_thread.join(timeout=5.0)
        self.stop_flask_server()
        bt.logging.info("[ENTITY-GW] Shutdown complete")
