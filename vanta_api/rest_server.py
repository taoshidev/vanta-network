import statistics
from string import hexdigits

import bittensor as bt
import threading
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, Optional

from flask import Flask, jsonify, request, Response, g
import os
import time
import json
import gzip
import traceback
from setproctitle import setproctitle
from waitress import serve
# from flask_compress import Compress  # Removed: causes double-compression of pre-compressed data
from bittensor_wallet import Keypair

from time_util.time_util import TimeUtil
from vali_objects.utils.limit_order.market_order_manager import MarketOrderManager
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.limit_order.order_processor import OrderProcessor
from multiprocessing import current_process
from vanta_api.api_key_refresh import APIKeyMixin
from vanta_api.nonce_manager import NonceManager
from shared_objects.rpc.rpc_server_base import RPCServerBase


class APIMetricsTracker:
    """
    Tracks API usage metrics and logs them periodically.
    Uses a rolling time window approach to track:
    1. API key usage counts
    2. Endpoint performance metrics (request count, avg processing time)
    3. Failed request tracking
    """

    def __init__(self, log_interval_minutes: int = 5, api_key_mapping: Dict = None):
        """
        Initialize the metrics tracker with the given log interval.

        Args:
            log_interval_minutes: How often to log metrics (in minutes)
        """
        self.log_interval_minutes = log_interval_minutes
        self.log_interval_seconds = log_interval_minutes * 60

        # Maps API key name to deque of timestamps
        self.api_key_hits: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10000))

        # Maps endpoint to deque of (timestamp, latency) tuples
        self.endpoint_hits: Dict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=10000))

        # Track per-endpoint API key usage: maps (endpoint, user_id) to deque of timestamps
        self.endpoint_api_key_hits: Dict[Tuple[str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=10000))

        # Track failed requests: maps (user_id, endpoint, status_code) to deque of timestamps
        self.failed_requests: Dict[Tuple[str, str, int], Deque[float]] = defaultdict(lambda: deque(maxlen=1000))

        # Lock for thread safety
        self.metrics_lock = threading.Lock()

        # Reference to API key to user ID mapping
        self.api_key_to_alias = api_key_mapping or {}  # Use provided mapping or empty dict

        # Start logging thread
        self.start_logging_thread()

    def _get_user_id_from_api_key(self, api_key: str) -> str:
        """
        Get user ID from API key, handling unknown keys properly.

        Args:
            api_key: The API key to look up

        Returns:
            The user ID or a unique unknown_key identifier
        """
        # Get user_id from api_key if available
        user_id = self.api_key_to_alias.get(api_key, "unknown_key")
        return user_id

    def track_request(self, api_key: str, endpoint: str, duration: float, status_code: int = 200):
        """
        Track a request with its associated API key, endpoint, and duration.

        Args:
            api_key: The API key used for the request
            endpoint: The endpoint that was accessed
            duration: Request processing time in seconds
            status_code: HTTP status code of the response
        """
        # Get user_id from api_key
        user_id = self._get_user_id_from_api_key(api_key)

        now = time.time()

        with self.metrics_lock:
            self.api_key_hits[user_id].append(now)
            self.endpoint_hits[endpoint].append((now, duration))

            # Track per-endpoint API key usage
            self.endpoint_api_key_hits[(endpoint, user_id)].append(now)

            # Track failed requests
            if status_code >= 400:
                self.failed_requests[(user_id, endpoint, status_code)].append(now)

    def log_metrics(self):
        """Log the current metrics based on the rolling time window."""
        current_time = time.time()
        cutoff_time = current_time - self.log_interval_seconds

        # Process metrics with lock to ensure thread safety
        api_counts = {}
        endpoint_stats = {}
        failed_stats = {}

        with self.metrics_lock:
            # Process API key hits
            empty_keys = []
            for key, timestamps in self.api_key_hits.items():
                # Remove outdated entries
                while timestamps and timestamps[0] < cutoff_time:
                    timestamps.popleft()

                # Count remaining hits in the window
                count = len(timestamps)
                if count > 0:
                    api_counts[key] = count
                else:
                    # Mark empty keys for removal
                    empty_keys.append(key)

            # Remove empty keys
            for key in empty_keys:
                del self.api_key_hits[key]

            # Process endpoint hits
            empty_endpoints = []
            for endpoint, entries in self.endpoint_hits.items():
                # Remove outdated entries
                while entries and entries[0][0] < cutoff_time:
                    entries.popleft()

                # Calculate stats for remaining hits
                count = len(entries)
                if count > 0:
                    # Extract just the durations
                    durations = [duration for _, duration in entries]
                    # Calculate multiple statistics
                    stats = {
                        "count": count,
                        "mean": statistics.mean(durations),
                        "median": statistics.median(durations),
                        "min": min(durations),
                        "max": max(durations)
                    }
                    # Remove None values for percentiles that couldn't be calculated
                    stats = {k: v for k, v in stats.items() if v is not None}
                    endpoint_stats[endpoint] = stats
                else:
                    # Mark empty endpoints for removal
                    empty_endpoints.append(endpoint)

            # Remove empty endpoints
            for endpoint in empty_endpoints:
                del self.endpoint_hits[endpoint]

            # Process failed requests
            empty_failed = []
            for key, timestamps in self.failed_requests.items():
                # Remove outdated entries
                while timestamps and timestamps[0] < cutoff_time:
                    timestamps.popleft()

                count = len(timestamps)
                if count > 0:
                    failed_stats[key] = count
                else:
                    empty_failed.append(key)

            # Remove empty failed request entries
            for key in empty_failed:
                del self.failed_requests[key]

            # Process per-endpoint API key usage
            endpoint_api_key_counts = {}
            empty_endpoint_keys = []
            for (endpoint, user_id), timestamps in self.endpoint_api_key_hits.items():
                # Remove outdated entries
                while timestamps and timestamps[0] < cutoff_time:
                    timestamps.popleft()

                count = len(timestamps)
                if count > 0:
                    if endpoint not in endpoint_api_key_counts:
                        endpoint_api_key_counts[endpoint] = {}
                    endpoint_api_key_counts[endpoint][user_id] = count
                else:
                    empty_endpoint_keys.append((endpoint, user_id))

            # Remove empty endpoint-key combinations
            for key in empty_endpoint_keys:
                del self.endpoint_api_key_hits[key]

        # Skip logging if there's no activity
        if not api_counts and not endpoint_stats and not failed_stats:
            bt.logging.info(f"No API activity in the last {self.log_interval_minutes} minutes")
            return

        # Format and log the metrics report
        log_lines = [f"\n===== API Metrics (Last {self.log_interval_minutes} minutes) ====="]

        # Log API key usage
        log_lines.append("\nAPI Key Usage:")
        if api_counts:
            for key, count in sorted(api_counts.items(), key=lambda x: x[1], reverse=True):
                log_lines.append(f"  {key}: {count} requests")
        else:
            log_lines.append("  No API requests in this period")

        # Log endpoint metrics
        log_lines.append("\nEndpoint Performance:")
        if endpoint_stats:
            for endpoint, stats in sorted(endpoint_stats.items(),
                                          key=lambda x: x[1]["count"], reverse=True):
                # Create API key breakdown for this endpoint
                api_key_breakdown = {}
                if endpoint in endpoint_api_key_counts:
                    for user_id, count in endpoint_api_key_counts[endpoint].items():
                        api_key_breakdown[user_id] = count

                # Format API key breakdown
                if api_key_breakdown:
                    breakdown_str = str(api_key_breakdown)
                    log_lines.append(f"  {endpoint}: {stats['count']} requests {breakdown_str}")
                else:
                    log_lines.append(f"  {endpoint}: {stats['count']} requests")

                log_lines.append(f"    mean: {stats['mean'] * 1000:.2f}ms")
                log_lines.append(f"    median: {stats['median'] * 1000:.2f}ms")
                log_lines.append(f"    min/max: {stats['min'] * 1000:.2f}ms / {stats['max'] * 1000:.2f}ms")
        else:
            log_lines.append("  No endpoint activity in this period")

        # Log failed requests
        if failed_stats:
            log_lines.append("\nFailed Requests:")
            for (user_id, endpoint, status_code), count in sorted(failed_stats.items(),
                                                                  key=lambda x: x[1], reverse=True):
                display_user_id = user_id

                # Add status code description for common failure codes
                status_desc = {
                    400: "Bad Request",
                    401: "Unauthorized",
                    403: "Forbidden/Insufficient Tier",
                    404: "Not Found",
                    413: "Request Too Large",
                    500: "Internal Server Error",
                    503: "Service Unavailable"
                }.get(status_code, "Unknown Error")

                log_lines.append(f"  {display_user_id} -> {endpoint} [{status_code} {status_desc}]: {count} failures")

        # Log the complete report
        final_str = "\n".join(log_lines)
        bt.logging.info(final_str)

    def periodic_logging(self):
        """Periodically log metrics based on the configured interval."""
        while True:
            # Sleep for the log interval
            time.sleep(self.log_interval_seconds)

            # Log metrics with exception handling
            try:
                self.log_metrics()
            except Exception as e:
                print(f"Error in metrics logging: {e}")
                traceback.print_exc()

    def start_logging_thread(self):
        """Start the periodic logging thread."""
        logging_thread = threading.Thread(target=self.periodic_logging, daemon=True)
        logging_thread.start()
        bt.logging.info(f"API metrics logging started (interval: {self.log_interval_minutes} minutes)")


class VantaRestServer(RPCServerBase, APIKeyMixin):
    """Handles REST API requests with Flask and Waitress.

    Inherits from:
    - APIKeyMixin: Provides API key authentication and refresh
    - RPCServerBase: Provides RPC server lifecycle management for health checks/control

    The server runs TWO servers:
    - Flask HTTP server on port 48888 (REST API)
    - RPC server on port 50022 (health checks, control, monitoring)
    """

    service_name = ValiConfig.RPC_REST_SERVER_SERVICE_NAME
    service_port = ValiConfig.RPC_REST_SERVER_PORT

    def __init__(self, api_keys_file, shared_queue=None, refresh_interval=15,
                 metrics_interval_minutes=5, running_unit_tests=False,
                 connection_mode:RPCConnectionMode = RPCConnectionMode.RPC,
                 start_server=True, flask_host=None, flask_port=None, **kwargs):
        """Initialize the REST server with API key handling and routing.

        Note: Creates own clients internally
        - PositionManagerClient
        - AssetSelectionClient
        - LimitOrderClient
        - ContractClient
        - CoreOutputsClient
        - StatisticsOutputsClient
        - DebtLedgerClient
        - PerfLedgerClient

        The server runs on configurable endpoints (defaults from ValiConfig):
        - Flask HTTP: flask_host:flask_port (default: ValiConfig.REST_API_HOST:REST_API_PORT)
        - RPC health: ValiConfig.RPC_REST_SERVER_PORT (50022)

        Args:
            api_keys_file: Path to the JSON file containing API keys
            shared_queue: Optional shared queue for communication with WebSocket server
            refresh_interval: How often to check for API key changes (seconds)
            metrics_interval_minutes: How often to log API metrics (minutes)
            running_unit_tests: Whether running in unit test mode
        """
        self.running_unit_tests = running_unit_tests

        print(f"[REST-INIT] Step 1/9: Initializing API key handling...")
        # Initialize API key handling
        APIKeyMixin.__init__(self, api_keys_file, refresh_interval)
        print(f"[REST-INIT] Step 1/9: API key handling initialized ✓")

        print(f"[REST-INIT] Step 2/9: Creating PositionManagerClient...")
        # Create own PositionManagerClient (forward compatibility - no parameter passing)
        from vali_objects.position_management.position_manager_client import PositionManagerClient
        self._position_client = PositionManagerClient(connection_mode=connection_mode)
        self._debt_ledger_client = DebtLedgerClient(connection_mode=connection_mode)
        self._perf_ledger_client = PerfLedgerClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2/9: PositionManagerClient created ✓")

        print(f"[REST-INIT] Step 2b/9: Creating AssetSelectionClient...")
        # Create own AssetSelectionClient (forward compatibility - no parameter passing)
        from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
        self._asset_selection_client = AssetSelectionClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2b/9: AssetSelectionClient created ✓")

        print(f"[REST-INIT] Step 2c/9: Creating LimitOrderClient...")
        # Create own LimitOrderClient (forward compatibility - no parameter passing)
        from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
        self._limit_order_client = LimitOrderClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2c/9: LimitOrderClient created ✓")

        print(f"[REST-INIT] Step 2d/9: Creating ContractClient...")
        # Create own ContractClient (forward compatibility - no parameter passing)
        from vali_objects.contract.contract_client import ContractClient
        self._contract_client = ContractClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2d/9: ContractClient created ✓")

        print(f"[REST-INIT] Step 2e/9: Creating CoreOutputsClient...")
        # Create own CoreOutputsClient (forward compatibility - no parameter passing)
        from vali_objects.data_export.core_outputs_client import CoreOutputsClient
        self._core_outputs_client = CoreOutputsClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2e/9: CoreOutputsClient created ✓")

        print(f"[REST-INIT] Step 2f/9: Creating StatisticsOutputsClient...")
        # Create own StatisticsOutputsClient (forward compatibility - no parameter passing)
        from vali_objects.statistics.miner_statistics_client import MinerStatisticsClient
        self._statistics_outputs_client = MinerStatisticsClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2f/9: StatisticsOutputsClient created ✓")

        print(f"[REST-INIT] Step 3/9: Setting REST server configuration...")
        # IMPORTANT: Store Flask HTTP server config separately from RPC port
        # Flask serves REST API on configurable host/port (self.flask_host/flask_port)
        # RPC server runs on port 50022 (self.service_port) for health checks
        # Use provided host/port or fall back to ValiConfig defaults
        self.flask_host = flask_host if flask_host is not None else ValiConfig.REST_API_HOST
        self.flask_port = flask_port if flask_port is not None else ValiConfig.REST_API_PORT

        # REST server configuration
        self.shared_queue = shared_queue
        self.nonce_manager = NonceManager()
        self.market_order_manager = MarketOrderManager(serve=False)
        self.data_path = ValiConfig.BASE_DIR
        print(f"[REST-INIT] Step 3/9: Configuration set ✓")

        print(f"[REST-INIT] Step 4/9: Creating Flask app...")
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 256 * 1024  # 256 KB upper bound
        print(f"[REST-INIT] Step 4/9: Flask app created ✓")

        print(f"[REST-INIT] Step 5/9: Loading contract owner...")
        self._contract_client.load_contract_owner()
        print(f"[REST-INIT] Step 5/9: Contract owner loaded ✓")

        # Flask-Compress removed to prevent double-compression of pre-compressed endpoints
        # Our critical endpoints (validator-checkpoint, minerstatistics) serve pre-compressed data

        print(f"[REST-INIT] Step 6/9: Setting up metrics tracking...")
        # Initialize metrics tracking
        self._setup_metrics(metrics_interval_minutes)
        print(f"[REST-INIT] Step 6/9: Metrics tracking initialized ✓")

        print(f"[REST-INIT] Step 7/9: Registering routes...")
        # Register routes
        self._register_routes()
        print(f"[REST-INIT] Step 7/9: Routes registered ✓")

        print(f"[REST-INIT] Step 8/9: Registering error handlers...")
        # Register error handlers
        self._register_error_handlers()
        print(f"[REST-INIT] Step 8/9: Error handlers registered ✓")

        print(f"[REST-INIT] Step 9/9: Starting API key refresh thread...")
        # Start API key refresh thread
        self.start_refresh_thread()
        print(f"[REST-INIT] Step 9/9: API key refresh thread started ✓")

        print(f"[REST-INIT] Step 10/10: Initializing RPC server for health checks...")
        # Initialize RPCServerBase (provides RPC server for health checks on port 50022)
        RPCServerBase.__init__(
            self,
            service_name=self.service_name,
            port=self.service_port,
            connection_mode=connection_mode,
            start_server=start_server,
            start_daemon=False,  # Flask runs in main thread, no daemon needed
            **kwargs
        )
        print(f"[REST-INIT] Step 10/10: RPC server initialized on port {self.service_port} ✓")

        print(f"[{current_process().name}] RestServer initialized with {len(self.accessible_api_keys)} API keys")
        print(f"[{current_process().name}] Flask HTTP server will run on {self.flask_host}:{self.flask_port}")
        print(f"[{current_process().name}] RPC health server running on port {self.service_port}")

        # Flask server state (thread-based, similar to RPC server)
        self._flask_thread: Optional[threading.Thread] = None
        self._flask_ready = threading.Event()

        # Start Flask server if requested (same pattern as RPC server in RPCServerBase)
        if start_server and connection_mode == RPCConnectionMode.RPC:
            self.start_flask_server()

    # ============================================================================
    # FLASK SERVER LIFECYCLE (follows RPCServerBase pattern)
    # ============================================================================

    def start_flask_server(self):
        """
        Start the Flask HTTP server in a background thread.

        Follows the same pattern as RPCServerBase.start_rpc_server():
        - Runs in background thread
        - Sets ready event when listening
        - Waits for readiness before returning
        """
        if self._flask_thread is not None and self._flask_thread.is_alive():
            bt.logging.warning(f"{self.service_name} Flask server already started")
            return

        start_time = time.time()

        # Start Flask server in background thread
        self._flask_thread = threading.Thread(
            target=self.run,  # run() method contains the waitress serve() call
            daemon=True,
            name=f"{self.service_name}_Flask"
        )
        self._flask_thread.start()

        # Wait for server to be ready (Flask signals this in run())
        if not self._flask_ready.wait(timeout=5.0):
            bt.logging.warning(f"{self.service_name} Flask server may not be fully ready")

        elapsed_ms = (time.time() - start_time) * 1000
        bt.logging.success(
            f"{self.service_name} Flask HTTP server started on {self.flask_host}:{self.flask_port} ({elapsed_ms:.0f}ms)"
        )

    def stop_flask_server(self):
        """Stop the Flask HTTP server."""
        if self._flask_thread is None:
            return

        bt.logging.info(f"{self.service_name} stopping Flask server...")

        # Flask/Waitress doesn't have a clean shutdown mechanism from outside
        # The thread will be terminated when the process exits (daemon=True)
        self._flask_thread = None
        self._flask_ready.clear()

        bt.logging.info(f"{self.service_name} Flask server stopped")

    def shutdown(self):
        """Override shutdown to stop Flask server in addition to RPC server."""
        bt.logging.info(f"{self.service_name} shutting down...")
        self.stop_flask_server()
        # Call parent shutdown to stop RPC server and daemon
        super().shutdown()
        bt.logging.info(f"{self.service_name} shutdown complete")

    # ============================================================================
    # POSITION MANAGER ACCESS (forward compatibility - creates own client)
    # ============================================================================

    @property
    def position_manager(self):
        """Get position manager client."""
        return self._position_client

    @property
    def contract_manager(self):
        """Get contract client (forward compatibility - created internally)."""
        return self._contract_client

    # ============================================================================
    # RPCServerBase REQUIRED METHODS
    # ============================================================================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work.

        Note: PTNRestServer doesn't need a daemon loop - all work is done
        in Flask request handlers. This is a no-op.
        """
        pass

    def _jsonify_with_custom_encoder(self, data, status_code=200):
        """
        Create a JSON response using CustomEncoder to handle BaseModel objects.

        Args:
            data: The data to jsonify
            status_code: HTTP status code (default 200)

        Returns:
            Flask Response object with proper JSON serialization
        """
        json_str = json.dumps(data, cls=CustomEncoder)
        response = Response(json_str, content_type='application/json')
        response.status_code = status_code
        return response

    def _setup_metrics(self, metrics_interval_minutes):
        """Set up API metrics tracking."""
        # Initialize the metrics tracker as instance variable
        self.metrics = APIMetricsTracker(metrics_interval_minutes, self.api_key_to_alias)

        # Set up Flask request hooks for automatic metrics tracking
        @self.app.before_request
        def start_timer():
            # Store start time in Flask's g object for this request
            g.start_time = time.time()

        @self.app.after_request
        def record_metrics(response):
            # Calculate request duration
            end_time = time.time()
            duration = end_time - getattr(g, 'start_time', end_time)

            # Get API key - handle errors gracefully
            try:
                api_key = self._get_api_key_safe()
            except Exception:
                api_key = None

            # Get endpoint (use rule if available, otherwise path)
            url = request.url_rule.rule if request.url_rule else request.path

            # Track the request using the instance metrics tracker
            self.metrics.track_request(api_key, url, duration, response.status_code)

            return response

    def _register_error_handlers(self):
        """Register custom error handlers for common exceptions."""

        @self.app.errorhandler(400)
        def handle_bad_request(e):
            # Log the error with user context
            api_key = self._get_api_key_safe()
            user_id = self.metrics._get_user_id_from_api_key(api_key)

            bt.logging.warning(
                f"Bad Request: user={user_id} endpoint={request.path} method={request.method} "
                f"error={str(e).split(':')[0] if ':' in str(e) else str(e)[:50]}"
            )

            return jsonify({'error': 'Bad request'}), 400

        @self.app.errorhandler(401)
        def handle_unauthorized(e):
            return jsonify({'error': 'Unauthorized access'}), 401

        @self.app.errorhandler(403)
        def handle_forbidden(e):
            return jsonify({'error': 'Forbidden'}), 403

        @self.app.errorhandler(404)
        def handle_not_found(e):
            return jsonify({'error': 'Resource not found'}), 404

        @self.app.errorhandler(500)
        def handle_internal_error(e):
            # Log the error with user context
            api_key = self._get_api_key_safe()
            user_id = self.metrics._get_user_id_from_api_key(api_key)

            bt.logging.error(
                f"Internal Error: user={user_id} endpoint={request.path} method={request.method} "
                f"error={str(e)[:100]}"
            )

            return jsonify({'error': 'Internal server error'}), 500

        @self.app.errorhandler(Exception)
        def handle_exception(e):
            # Log unexpected errors
            api_key = self._get_api_key_safe()
            user_id = self.metrics._get_user_id_from_api_key(api_key)

            bt.logging.error(
                f"Unhandled Exception: user={user_id} endpoint={request.path} method={request.method} "
                f"error_type={type(e).__name__} error={str(e)[:100]}"
            )

            # Only log full traceback for truly unexpected errors
            if not isinstance(e, (json.JSONDecodeError, KeyError, ValueError)):
                bt.logging.debug(f"Full traceback:\n{traceback.format_exc()}")

            return jsonify({'error': 'An error occurred processing your request'}), 500

    def _register_routes(self):
        """Register all API routes."""

        @self.app.route("/miner-positions", methods=["GET"])
        def get_miner_positions():
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Get the 'tier' query parameter from the request
            requested_tier = str(request.args.get('tier', 100))
            is_gz_data = True

            # Validate the 'tier' parameter
            if requested_tier not in ['0', '30', '50', '100']:
                return jsonify({'error': 'Invalid tier value. Allowed values are 0, 30, 50, or 100'}), 400

            # Check if API key has sufficient tier access
            if not self.can_access_tier(api_key, int(requested_tier)):
                return jsonify({'error': f'Your API key does not have access to tier {requested_tier} data'}), 403

            f = ValiBkpUtils.get_miner_positions_output_path(suffix_dir=requested_tier)

            # Attempt to retrieve the file
            data = self._get_file(f, binary=is_gz_data)

            if data is None:
                return jsonify({'error': 'Data not found'}), 404
            return Response(data, content_type='application/json', headers={
                'Content-Encoding': 'gzip'
            })

        @self.app.route("/miner-positions/<minerid>", methods=["GET"])
        def get_miner_positions_unique(minerid):
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Use the API key's tier for access
            api_key_tier = self.get_api_key_tier(api_key)
            if self.can_access_tier(api_key, 100) and self.position_manager:
                existing_positions: list[Position] = self.position_manager.get_positions_for_one_hotkey(minerid,
                                                                                                        sort_positions=True)
                if not existing_positions:
                    return jsonify({'error': f'Miner ID {minerid} not found'}), 404
                filtered_data = self._position_client.positions_to_dashboard_dict(existing_positions,
                                                                                  TimeUtil.now_in_millis())
            else:
                requested_tier = str(api_key_tier)
                f = ValiBkpUtils.get_miner_positions_output_path(suffix_dir=requested_tier)
                data = self._get_file(f)

                if data is None:
                    return jsonify({'error': 'Data not found'}), 404
                # Filter the data for the specified miner ID
                filtered_data = data.get(minerid, None)

            if not filtered_data:
                return jsonify({'error': f'Miner ID {minerid} not found'}), 404

            return jsonify(filtered_data)

        @self.app.route("/miner-hotkeys", methods=["GET"])
        def get_miner_hotkeys():
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            if self.position_manager:
                # Use the position manager to get miner hotkeys
                miner_hotkeys = list(self.position_manager.get_miner_hotkeys_with_at_least_one_position())
            else:
                f = ValiBkpUtils.get_miner_positions_output_path()
                data = self._get_file(f)

                if data is None:
                    return jsonify({'error': 'Data not found'}), 404

                miner_hotkeys = list(data.keys())

            if len(miner_hotkeys) == 0:
                return jsonify({'error': 'No miner hotkeys found'}), 404
            else:
                return jsonify(miner_hotkeys)

        @self.app.route("/emissions-ledger/<minerid>", methods=["GET"])
        def get_emissions_ledger(minerid):
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Use RPC getter to access emissions ledger via debt ledger manager
            data = self._debt_ledger_client.get_emissions_ledger(minerid)

            if data is None:
                return jsonify({'error': 'Emissions ledger data not found'}), 404
            else:
                return self._jsonify_with_custom_encoder(data)

        @self.app.route("/debt-ledger/<minerid>", methods=["GET"])
        def get_miner_debt_ledger(minerid):
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            data = self._debt_ledger_client.get_ledger(minerid)

            if data is None:
                return jsonify({'error': 'Debt ledger data not found'}), 404
            else:
                return self._jsonify_with_custom_encoder(data)

        @self.app.route("/perf-ledger/<minerid>", methods=["GET"])
        def get_perf_ledger(minerid):
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Check if perf ledger client is available
            if not self._perf_ledger_client:
                return jsonify({'error': 'Perf ledger data not available'}), 503

            try:
                # Use dedicated RPC method to get only this miner's ledger (efficient - no bulk transfer)
                data = self._perf_ledger_client.get_perf_ledger_for_hotkey(minerid)

                if data is None:
                    return jsonify({'error': f'Perf ledger data not found for miner {minerid}'}), 404

                return self._jsonify_with_custom_encoder(data)

            except Exception as e:
                bt.logging.error(f"Error retrieving perf ledger for {minerid}: {e}")
                return jsonify({'error': 'Internal server error retrieving perf ledger data'}), 500

        @self.app.route("/debt-ledger", methods=["GET"])
        def get_debt_ledger():
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Check if debt ledger manager is available
            if not self._debt_ledger_client:
                return jsonify({'error': 'Debt ledger data not available'}), 503

            try:
                # Get compressed summaries directly from RPC (faster than disk I/O)
                # RPC call retrieves pre-compressed gzip bytes from memory
                compressed_data = self._debt_ledger_client.get_compressed_summaries_rpc()

                if compressed_data is None or len(compressed_data) == 0:
                    return jsonify({'error': 'Debt ledger data not found'}), 404

                # Return pre-compressed data with gzip header (browser decompresses automatically)
                return Response(compressed_data, content_type='application/json', headers={
                    'Content-Encoding': 'gzip'
                })

            except Exception as e:
                bt.logging.error(f"Error retrieving debt ledger summaries via RPC: {e}")
                return jsonify({'error': 'Internal server error retrieving debt ledger data'}), 500

        @self.app.route("/penalty-ledger/<minerid>", methods=["GET"])
        def get_penalty_ledger(minerid):
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Use RPC getter to access penalty ledger via debt ledger manager
            data = self._debt_ledger_client.get_penalty_ledger(minerid)

            if data is None:
                return jsonify({'error': 'Penalty ledger data not found'}), 404
            else:
                return self._jsonify_with_custom_encoder(data)

        @self.app.route("/validator-checkpoint", methods=["GET"])
        def get_validator_checkpoint():
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Validator checkpoint data is only available for tier 100
            if not self.can_access_tier(api_key, 100):
                return jsonify({'error': 'Validator checkpoint data requires tier 100 access'}), 403

            # Try to get compressed data from memory cache first via CoreOutputsClient
            compressed_data = None
            if self._core_outputs_client:
                try:
                    compressed_data = self._core_outputs_client.get_compressed_checkpoint_from_memory()
                except Exception as e:
                    bt.logging.debug(f"Error accessing compressed checkpoint cache: {e}")

            if compressed_data is not None:
                # Return pre-compressed data with appropriate headers
                return Response(compressed_data, content_type='application/json', headers={
                    'Content-Encoding': 'gzip'
                })

            # Fallback to file read if memory unavailable
            # Checkpoint is always stored as compressed file
            f_gz = ValiBkpUtils.get_vcp_output_path()

            if os.path.exists(f_gz):
                # Read pre-compressed file directly
                try:
                    with open(f_gz, 'rb') as fh:
                        compressed_data = fh.read()
                    return Response(compressed_data, content_type='application/json', headers={
                        'Content-Encoding': 'gzip'
                    })
                except Exception as e:
                    bt.logging.error(f"Failed to read compressed checkpoint: {e}")
                    return jsonify({'error': 'Failed to read checkpoint data'}), 500
            else:
                return jsonify({'error': 'Checkpoint data not found'}), 404

        @self.app.route("/statistics", methods=["GET"])
        def get_validator_checkpoint_statistics():
            api_key = self._get_api_key_safe()
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Grab the optional "checkpoints" query param; default it to "true"
            show_checkpoints = request.args.get("checkpoints", "true").lower()
            include_checkpoints = show_checkpoints == "true"

            # PRIMARY: Try to use pre-compressed payload from memory cache (fastest)
            if self._statistics_outputs_client:
                compressed_data = self._statistics_outputs_client.get_compressed_statistics(include_checkpoints)
                if compressed_data:
                    # Return pre-compressed JSON directly
                    return Response(compressed_data, content_type='application/json', headers={
                        'Content-Encoding': 'gzip'
                    })

            # FALLBACK 1: If no modification needed, serve compressed file directly
            if show_checkpoints == "true":
                f_gz = ValiBkpUtils.get_miner_stats_dir() + ".gz"
                if os.path.exists(f_gz):
                    compressed_data = self._get_file(f_gz, binary=True)
                    return Response(compressed_data, content_type='application/json', headers={
                        'Content-Encoding': 'gzip'
                    })

            # FALLBACK 2: Decompress and modify if needed (checkpoints=false or no .gz file)
            f = ValiBkpUtils.get_miner_stats_dir()
            data = self._get_file(f)
            if not data:
                return jsonify({'error': 'Statistics data not found'}), 404

            # If checkpoints=false, remove the "checkpoints" key from each element in data
            if show_checkpoints == "false":
                for element in data.get("data", []):
                    element.pop("checkpoints", None)

            return self._jsonify_with_custom_encoder(data)

        @self.app.route("/statistics/<minerid>/", methods=["GET"])
        def get_validator_checkpoint_statistics_unique(minerid):
            api_key = self._get_api_key_safe()
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Get statistics data from disk
            f = ValiBkpUtils.get_miner_stats_dir()
            data = self._get_file(f)
            if not data:
                return jsonify({'error': 'Statistics data not found'}), 404

            data_summary = data.get("data", [])
            if not data_summary:
                return jsonify({'error': 'No data found'}), 404

            # Grab the optional "checkpoints" query param; default it to "true"
            show_checkpoints = request.args.get("checkpoints", "true").lower()

            for element in data_summary:
                if element.get("hotkey", None) == minerid:
                    # If the user set checkpoints=false, remove them from this element
                    if show_checkpoints == "false":
                        element.pop("checkpoints", None)
                    return jsonify(element)

            return jsonify({'error': f'Miner ID {minerid} not found'}), 404

        @self.app.route("/eliminations", methods=["GET"])
        def get_eliminations():
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = ValiBkpUtils.get_eliminations_dir()
            data = self._get_file(f)

            if data is None:
                return jsonify({'error': 'Eliminations data not found'}), 404
            else:
                return self._jsonify_with_custom_encoder(data)

        @self.app.route("/limit-orders/<minerid>", methods=["GET"])
        def get_limit_orders_unique(minerid):
            api_key = self._get_api_key_safe()

            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            api_key_tier = self.get_api_key_tier(api_key)
            if self.can_access_tier(api_key, 100) and self._limit_order_client:
                orders_data = self._limit_order_client.to_dashboard_dict(minerid)
                if not orders_data:
                    return jsonify({'error': f'No limit orders found for miner {minerid}'}), 404
            else:
                try:
                    orders_data = ValiBkpUtils.get_limit_orders(minerid, unfilled_only=True, running_unit_tests=False)
                    if not orders_data:
                        return jsonify({'error': f'No limit orders found for miner {minerid}'}), 404
                except Exception as e:
                    bt.logging.error(f"Error retrieving limit orders for {minerid}: {e}")
                    return jsonify({'error': 'Error retrieving limit orders'}), 500

            return jsonify(orders_data)

        @self.app.route("/collateral/deposit", methods=["POST"])
        def deposit_collateral():
            """Process collateral deposit with encoded extrinsic."""
            MAX_EXTRINSIC_HEX = 200_000 # ~100 KB decoded;

            # Check if contract manager is available
            if not self.contract_manager:
                return jsonify({'error': 'Collateral operations not available'}), 503

            try:
                # Parse JSON request
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400

                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Invalid JSON body'}), 400

                # Check vanta-cli version FIRST - reject outdated versions
                vanta_cli_version = (
                    data.get('version')
                    or data.get('ptncli_version')
                    or '0.0.0'
                )
                vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
                if vanta_cli_error:
                    return jsonify({'error': vanta_cli_error}), 400
                    
                # Validate required fields
                required_fields = ['extrinsic']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400

                # Validate extrinsic
                extrinsic = data.get('extrinsic')
                if not isinstance(extrinsic, str):
                    return jsonify({'error': 'extrinsic must be a hex string'}), 400
                if len(extrinsic) > MAX_EXTRINSIC_HEX:
                    return jsonify({'error': 'extrinsic too large'}), 413
                if len(extrinsic) % 2 != 0 or not all(c in hexdigits for c in extrinsic):
                    return jsonify({'error': 'extrinsic must be even-length hex'}), 400
                        
                # Process the deposit using raw data
                result = self.contract_manager.process_deposit_request(
                    extrinsic_hex=extrinsic
                )

                # Return response
                return jsonify(result)
                
            except Exception as e:
                bt.logging.error(f"Error processing collateral deposit: {e}")
                return jsonify({'error': 'Internal server error processing deposit'}), 500

        @self.app.route("/collateral/query-withdraw", methods=["POST"])
        def query_withdraw_collateral():
            """Query collateral withdrawal request for potential slashed amount"""
            # Check if contract manager is available
            if not self.contract_manager:
                return jsonify({'error': 'Collateral operations not available'}), 503

            try:
                # Parse JSON request
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400

                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Invalid JSON body'}), 400

                # Check vanta-cli version FIRST - reject outdated versions
                vanta_cli_version = (
                    data.get('version')
                    or data.get('ptncli_version')
                    or '0.0.0'
                )
                vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
                if vanta_cli_error:
                    return jsonify({'error': vanta_cli_error}), 400

                # Validate required fields for withdrawal query
                required_fields = ['amount', 'miner_hotkey']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400

                # Validate amount is a positive number
                try:
                    amount = float(data['amount'])
                    if amount <= 0:
                        return jsonify({'error': 'Amount must be a positive number'}), 400
                except (ValueError, TypeError):
                    return jsonify({'error': 'Amount must be a valid number'}), 400

                # Validate miner_hotkey is a valid SS58 address
                miner_hotkey = data['miner_hotkey']
                try:
                    # Attempt to create a Keypair to validate SS58 format
                    Keypair(ss58_address=miner_hotkey)
                except Exception:
                    return jsonify({'error': 'Invalid SS58 address format for miner_hotkey'}), 400

                # Process the withdrawal query
                result = self.contract_manager.query_withdrawal_request(
                    amount=amount,
                    miner_hotkey=miner_hotkey
                )

                # Return response
                return jsonify(result)

            except Exception as e:
                bt.logging.error(f"Error processing collateral withdrawal query: {e}")
                return jsonify({'error': 'Internal server error processing withdrawal query'}), 500

        @self.app.route("/collateral/withdraw", methods=["POST"])
        def withdraw_collateral():
            """Process collateral withdrawal request."""
            # Check if contract manager is available
            if not self.contract_manager:
                return jsonify({'error': 'Collateral operations not available'}), 503

            try:
                # Parse JSON request
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400

                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Invalid JSON body'}), 400

                # Check vanta-cli version FIRST - reject outdated versions
                vanta_cli_version = (
                    data.get('version')
                    or data.get('ptncli_version')
                    or '0.0.0'
                )
                vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
                if vanta_cli_error:
                    return jsonify({'error': vanta_cli_error}), 400

                # Validate required fields for signed withdrawal
                required_fields = ['amount', 'miner_coldkey', 'miner_hotkey', 'nonce', 'timestamp', 'signature']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400

                # Verify the withdrawal signature
                keypair = Keypair(ss58_address=data['miner_coldkey'])
                message = json.dumps({
                    "amount": data['amount'],
                    "miner_coldkey": data['miner_coldkey'],
                    "miner_hotkey": data['miner_hotkey'],
                    "nonce": data['nonce'],
                    "timestamp": data['timestamp']
                }, sort_keys=True).encode('utf-8')
                is_valid = keypair.verify(message, bytes.fromhex(data['signature']))
                if not is_valid:
                    return jsonify({'error': 'Invalid signature. Withdrawal request unauthorized'}), 401

                # Verify coldkey-hotkey ownership using subtensor
                owns_hotkey = self._verify_coldkey_owns_hotkey(data['miner_coldkey'], data['miner_hotkey'])
                if not owns_hotkey:
                    return jsonify({'error': 'Coldkey does not own the specified hotkey'}), 403

                # Verify nonce
                nonce_key = f"{data['miner_coldkey']}::{data['miner_hotkey']}"
                is_valid, error_msg = self.nonce_manager.is_valid_request(
                    address=nonce_key,
                    nonce=str(data['nonce']),
                    timestamp=int(data['timestamp'])
                )
                if not is_valid:
                    return jsonify({'error': f'{error_msg}'}), 401

                # Validate amount is a positive number
                try:
                    amount = float(data['amount'])
                    if amount <= 0:
                        return jsonify({'error': 'Amount must be a positive number'}), 400
                except (ValueError, TypeError):
                    return jsonify({'error': 'Amount must be a valid number'}), 400

                # Process the withdrawal using verified data
                result = self.contract_manager.process_withdrawal_request(
                    amount=data['amount'],
                    miner_coldkey=data['miner_coldkey'],
                    miner_hotkey=data['miner_hotkey']
                )

                # Return response
                return jsonify(result)
                
            except Exception as e:
                bt.logging.error(f"Error processing collateral withdrawal: {e}")
                return jsonify({'error': 'Internal server error processing withdrawal'}), 500

        @self.app.route("/collateral/", methods=["GET"])
        def get_all_collateral_data():
            """Get collateral data for all miners.
            
            Example curl requests:
            
            # Get all collateral data for all miners
            curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/collateral/
            
            # Get collateral data for a specific miner
            curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/collateral/?hotkey=5GhDr...
            
            # Get only the most recent collateral record for each miner
            curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/collateral/?most_recent=true
            
            # Combine filters: specific miner's most recent record
            curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/collateral/?hotkey=5GhDr...&most_recent=true
            
            Response format:
            {
                "status": "success",
                "data": {
                    "hotkey1": [{
                        "account_size": 1000.0,
                        "account_size_theta": 10.0,
                        "update_time_ms": 1234567890000,
                        "valid_date_timestamp": 1234567890000
                    }],
                    ...
                },
                "miner_count": 5,
                "total_records": 25,
                "timestamp": 1234567890000
            }
            """
            
            # Check API key authentication
            api_key = self._get_api_key_safe()
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401
            
            # Check if contract manager is available
            if not self.contract_manager:
                return jsonify({'error': 'Collateral operations not available'}), 503

            try:
                # Get query parameters for filtering
                hotkey_filter = request.args.get('hotkey')
                most_recent_only = request.args.get('most_recent', 'false').lower() == 'true'
                
                # Get all collateral data using the proper serialization method
                # Pass most_recent_only directly to avoid double iteration
                data = self.contract_manager.miner_account_sizes_dict(most_recent_only=most_recent_only)
                
                # Apply hotkey filter if requested
                if hotkey_filter and hotkey_filter in data:
                    data = {hotkey_filter: data[hotkey_filter]}
                
                # Return consistent response format
                return jsonify({
                    'status': 'success',
                    'data': data,
                    'miner_count': len(data),
                    'total_records': sum(len(records) for records in data.values()),
                    'timestamp': TimeUtil.now_in_millis()
                })

            except Exception as e:
                bt.logging.error(f"Error getting all collateral data: {e}")
                return jsonify({'error': 'Internal server error retrieving data'}), 500

        @self.app.route("/collateral/balance/<miner_address>", methods=["GET"])
        def get_collateral_balance(miner_address):
            """Get a miner's collateral balance."""
            # Check if contract manager is available
            if not self.contract_manager:
                return jsonify({'error': 'Collateral operations not available'}), 503
                
            try:
                # Get the balance
                balance = self.contract_manager.get_miner_collateral_balance(miner_address)
                
                if balance is None:
                    return jsonify({'error': 'Failed to retrieve collateral balance'}), 500
                    
                return jsonify({
                    'miner_address': miner_address,
                    'balance_theta': balance
                })
                
            except Exception as e:
                bt.logging.error(f"Error getting collateral balance for {miner_address}: {e}")
                return jsonify({'error': 'Internal server error retrieving balance'}), 500

        @self.app.route("/asset-selection", methods=["POST"])
        def asset_selection():
            """Process asset selection request."""
            try:
                # Parse JSON request
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400

                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Invalid JSON body'}), 400

                # Check vanta-cli version FIRST - reject outdated versions
                vanta_cli_version = (
                    data.get('version')
                    or data.get('ptncli_version')
                    or '0.0.0'
                )
                vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
                if vanta_cli_error:
                    return jsonify({'error': vanta_cli_error}), 400

                # Validate required fields for signed withdrawal
                required_fields = ['asset_selection', 'miner_coldkey', 'miner_hotkey', 'signature']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400

                # Verify the withdrawal signature
                keypair = Keypair(ss58_address=data['miner_coldkey'])
                message = json.dumps({
                    "asset_selection": data['asset_selection'],
                    "miner_coldkey": data['miner_coldkey'],
                    "miner_hotkey": data['miner_hotkey']
                }, sort_keys=True).encode('utf-8')
                is_valid = keypair.verify(message, bytes.fromhex(data['signature']))
                if not is_valid:
                    return jsonify({'error': 'Invalid signature. Asset selection request unauthorized'}), 401

                # Verify coldkey-hotkey ownership using subtensor
                owns_hotkey = self._verify_coldkey_owns_hotkey(data['miner_coldkey'], data['miner_hotkey'])
                if not owns_hotkey:
                    return jsonify({'error': 'Coldkey does not own the specified hotkey'}), 403

                # Process the asset selection using verified data
                result = self._asset_selection_client.process_asset_selection_request(
                    asset_selection=data['asset_selection'],
                    miner=data['miner_hotkey']
                )

                # Return response
                return jsonify(result)

            except Exception as e:
                bt.logging.error(f"Error processing asset selection: {e}")
                return jsonify({'error': 'Internal server error processing asset selection'}), 500

        @self.app.route("/miner-selections", methods=["GET"])
        def get_miner_selections():
            """Get all miner asset selection data."""
            try:
                # Check API key authentication
                api_key = self._get_api_key_safe()

                # Check if the API key is valid
                if not self.is_valid_api_key(api_key):
                    return jsonify({'error': 'Unauthorized access'}), 401

                # Check if asset selection client is available
                if not self._asset_selection_client:
                    return jsonify({'error': 'Asset selection data not available'}), 503

                # Get all miner selection data using the getter method
                selections_data = self._asset_selection_client.get_all_miner_selections()

                return jsonify({
                    'miner_selections': selections_data,
                    'total_miners': len(selections_data),
                    'timestamp': TimeUtil.now_in_millis()
                })

            except Exception as e:
                bt.logging.error(f"Error retrieving miner selections: {e}")
                return jsonify({'error': 'Internal server error retrieving miner selections'}), 500

        @self.app.route("/development/order", methods=["POST"])
        def process_development_order():
            """
            Process development orders for testing market, limit, and cancel operations.
            Uses fixed hotkey 'DEVELOPMENT' for all operations.
            Requires tier 200 access.

            Example requests:

            # Market order
            curl -X POST http://localhost:48888/development/order \\
              -H "Authorization: Bearer YOUR_API_KEY" \\
              -H "Content-Type: application/json" \\
              -d '{"execution_type": "MARKET", "trade_pair_id": "BTCUSD", "order_type": "LONG", "leverage": 1.0}'

            # Limit order
            curl -X POST http://localhost:48888/development/order \\
              -H "Authorization: Bearer YOUR_API_KEY" \\
              -H "Content-Type": application/json" \\
              -d '{"execution_type": "LIMIT", "trade_pair_id": "BTCUSD", "order_type": "LONG", "leverage": 1.0, "limit_price": 50000.0}'

            # Bracket order (requires existing position)
            curl -X POST http://localhost:48888/development/order \\
              -H "Authorization: Bearer YOUR_API_KEY" \\
              -H "Content-Type: application/json" \\
              -d '{"execution_type": "BRACKET", "trade_pair_id": "BTCUSD", "stop_loss": 48000.0, "take_profit": 52000.0}'

            # Cancel specific limit order
            curl -X POST http://localhost:48888/development/order \\
              -H "Authorization: Bearer YOUR_API_KEY" \\
              -H "Content-Type: application/json" \\
              -d '{"execution_type": "LIMIT_CANCEL", "trade_pair_id": "BTCUSD", "order_uuid": "specific-uuid"}'

            # Cancel all limit orders for trade pair
            curl -X POST http://localhost:48888/development/order \\
              -H "Authorization: Bearer YOUR_API_KEY" \\
              -H "Content-Type: application/json" \\
              -d '{"execution_type": "LIMIT_CANCEL", "trade_pair_id": "BTCUSD"}'
            """
            DEVELOPMENT_HOTKEY = ValiConfig.DEVELOPMENT_HOTKEY

            # Check API key authentication
            api_key = self._get_api_key_safe()
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Check if API key has tier 200 access
            if not self.can_access_tier(api_key, 200):
                return jsonify({'error': 'Development order endpoint requires tier 200 access'}), 403

            try:
                # Parse and validate request
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400

                # Log raw request data for debugging JSON parse errors
                raw_data = request.get_data(as_text=True)
                bt.logging.debug(f"[DEV_ORDER] Raw request body (first 300 chars): {raw_data[:300]}")
                bt.logging.debug(f"[DEV_ORDER] Request body length: {len(raw_data)} chars")

                try:
                    data = request.get_json()
                except json.JSONDecodeError as e:
                    bt.logging.error(
                        f"[DEV_ORDER] JSON parse error at position {e.pos}: {e.msg}\n"
                        f"  Raw body: {raw_data}\n"
                        f"  Error context (char {max(0, e.pos-20)} to {min(len(raw_data), e.pos+20)}): "
                        f"{raw_data[max(0, e.pos-20):min(len(raw_data), e.pos+20)]}"
                    )
                    return jsonify({
                        'error': f'Invalid JSON at position {e.pos}: {e.msg}',
                        'position': e.pos
                    }), 400

                if not data:
                    return jsonify({'error': 'Invalid JSON body'}), 400

                # Create signal dict from request data
                signal = {
                    'trade_pair': {'trade_pair_id': data.get('trade_pair_id')},
                    'order_type': data.get('order_type', '').upper(),
                    'leverage': data.get('leverage'),
                    'value': data.get('value'),
                    'quantity': data.get('quantity'),
                    'execution_type': data.get('execution_type', 'MARKET').upper()
                }

                # Add limit_price for limit orders
                if 'limit_price' in data:
                    signal['limit_price'] = data['limit_price']

                if 'stop_loss' in data:
                    signal['stop_loss'] = data['stop_loss']

                if 'take_profit' in data:
                    signal['take_profit'] = data['take_profit']

                now_ms = TimeUtil.now_in_millis()
                miner_repo_version = "development"

                # Use unified OrderProcessor dispatcher (replaces lines 1466-1553)
                result = OrderProcessor.process_order(
                    signal=signal,
                    miner_order_uuid=data.get('order_uuid'),
                    now_ms=now_ms,
                    miner_hotkey=DEVELOPMENT_HOTKEY,
                    miner_repo_version=miner_repo_version,
                    limit_order_client=self._limit_order_client,
                    market_order_manager=self.market_order_manager
                )

                # Consistent response format across all order types
                return jsonify({
                    'status': 'success',
                    'execution_type': result.execution_type.value,
                    'order_uuid': data.get('order_uuid'),
                    'order': result.get_response_json()
                })

            except SignalException as e:
                bt.logging.error(f"SignalException in development order: {e}")
                return jsonify({'error': f'Signal error: {str(e)}'}), 400

            except Exception as e:
                bt.logging.error(f"Error processing development order: {e}")
                bt.logging.error(traceback.format_exc())
                return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    def _verify_coldkey_owns_hotkey(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        """
        Verify that a coldkey owns the specified hotkey using subtensor.

        Args:
            coldkey_ss58: The coldkey SS58 address
            hotkey_ss58: The hotkey SS58 address to verify ownership of

        Returns:
            bool: True if coldkey owns the hotkey, False otherwise
        """
        try:
            return self.contract_manager.verify_coldkey_owns_hotkey(coldkey_ss58, hotkey_ss58)
        except Exception as e:
            bt.logging.error(f"Error verifying coldkey-hotkey ownership: {e}")
            return False

    def _get_api_key_safe(self) -> Optional[str]:
        """
        Safely get the API key from the Authorization header.
        Reject keys in query params or request bodies to prevent accidental leakage.
        Returns None if there's any error accessing the API key.
        """
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return None  # No key presented

            # Handle both "Bearer <token>" and plain token formats
            parts = auth_header.split(' ', 1)
            if len(parts) != 2 or parts[0].lower() != 'bearer':
                return None  # Malformed or non‐bearer auth

            return parts[1]
        except Exception as e:
            bt.logging.debug(f"Error extracting API key: {e}")
            return None

    def _get_api_key(self):
        """
        Get the API key from the query parameters or request headers.
        This is the original method kept for backward compatibility.
        """
        api_key = self._get_api_key_safe()
        if api_key is None:
            # Log when no API key is found
            bt.logging.debug(f"No API key found in request to {request.path}")
        return api_key

    def _get_file(self, f, attempts=3, binary=False):
        """Read file with multiple attempts and return its contents."""
        file_path = os.path.abspath(os.path.join(self.data_path, f))
        if not os.path.exists(file_path):
            return None

        for attempt_number in range(attempts):
            try:
                if binary:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                else:
                    if file_path.endswith('.gz'):
                        with gzip.open(file_path, 'rt', encoding='utf-8') as fh:
                            data = json.load(fh)
                    else:
                        with open(file_path, "r") as file:
                            data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                if attempt_number == attempts - 1:
                    bt.logging.error(f"Failed to decode JSON after {attempts} attempts: {file_path}")
                    raise
                else:
                    bt.logging.debug(
                        f"Attempt {attempt_number + 1} failed with JSONDecodeError {e}, retrying..."
                    )
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                bt.logging.error(f"Unexpected error reading file {file_path}: {type(e).__name__}: {str(e)}")
                raise

    @staticmethod
    def check_vanta_cli_version(version: str) -> Optional[str]:
        """
        Check if vanta-cli version meets minimum requirements.
        This is now an enforced requirement - requests will be rejected if version is outdated.

        Args:
            version: vanta-cli version string (e.g., "1.0.5")

        Returns:
            Error message string if version is outdated or invalid, None if OK
        """
        try:
            # Parse version strings into tuples for comparison
            current = tuple(int(x) for x in version.split('.')[:3])
            minimum = tuple(int(x) for x in ValiConfig.VANTA_CLI_MINIMUM_VERSION.split('.')[:3])

            if current < minimum:
                return (f"Your vanta-cli version {version} is outdated and no longer supported. "
                        f"Please upgrade to vanta-cli >= {ValiConfig.VANTA_CLI_MINIMUM_VERSION}: "
                        f"pip install --upgrade git+https://github.com/taoshidev/vanta-cli.git")
        except (ValueError, AttributeError, IndexError):
            # Invalid version format - treat as error for security
            return (f"Invalid vanta-cli version format: {version}. "
                    f"Please reinstall vanta-cli: pip install --upgrade git+https://github.com/taoshidev/vanta-cli.git")
        return None

    def run(self):
        """
        Start the Flask REST server using Waitress.

        Called in background thread by start_flask_server().
        Signals _flask_ready event once Waitress is listening.
        """
        print(f"[{current_process().name}] Starting Flask REST server at http://{self.flask_host}:{self.flask_port}")
        setproctitle(f"vali_{self.__class__.__name__}")

        # Signal that Flask is about to start (Waitress will bind to port immediately)
        # Note: Waitress doesn't provide a callback for when it's ready, so we signal before serve()
        # The actual readiness check happens via the timeout in start_flask_server()
        self._flask_ready.set()

        # Start serving (blocks until shutdown)
        serve(
            self.app,
            host=self.flask_host,
            port=self.flask_port,
            connection_limit=1000,
            threads=10,  # Increased from 6 to handle queue depth
            channel_timeout=60,  # Reduced from 120 to close stuck connections faster
            cleanup_interval=10,  # Reduced from 30 for more aggressive cleanup
            backlog=2048,  # Increased to handle bursts better
            send_bytes=65536,  # Increase send buffer from default 1 byte
            outbuf_overflow=1048576,  # 1MB output buffer overflow size
            asyncore_use_poll=True  # Use poll() instead of select() for better performance
        )


# This allows the module to be run directly for testing
if __name__ == "__main__":
    import argparse

    bt.logging.enable_info()

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run the REST API server with API key authentication")
    parser.add_argument("--api-keys", type=str, default="api_keys.json", help="Path to API keys JSON file")

    args = parser.parse_args()

    # Create test API keys file if it doesn't exist
    if not os.path.exists(args.api_keys):
        with open(args.api_keys, "w") as f:
            json.dump({"test_user": "test_key", "client": "abc"}, f)
        print(f"Created test API keys file at {args.api_keys}")

    print(f"REST server will run on {ValiConfig.REST_API_HOST}:{ValiConfig.REST_API_PORT} (hardcoded in ValiConfig)")

    # Create and run the server (host/port read from ValiConfig)
    server = VantaRestServer(
        api_keys_file=args.api_keys,
        metrics_interval_minutes=1
    )
    server.run()
