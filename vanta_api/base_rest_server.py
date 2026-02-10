# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
Base REST Server - Shared Flask functionality for all REST servers.

This module provides a base class that consolidates common REST server patterns:
- Flask app initialization and lifecycle
- Waitress WSGI server management
- API key authentication (via APIKeyMixin)
- Metrics tracking with APIMetricsTracker
- Standard error handlers

Does NOT include RPC - subclasses add RPC if needed (e.g., VantaRestServer).
"""

import statistics
import time
import json
import traceback
import threading
import bittensor as bt
from abc import ABC, abstractmethod
from typing import Optional, Dict, Deque, Tuple
from collections import defaultdict, deque
from flask import Flask, jsonify, request, Response, g
from waitress import serve
from setproctitle import setproctitle
from multiprocessing import current_process

from vanta_api.api_key_refresh import APIKeyMixin


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


class BaseRestServer(APIKeyMixin, ABC):
    """
    Base class for REST API servers providing:
    - Flask app initialization and lifecycle
    - Waitress WSGI server management
    - API key authentication (via APIKeyMixin)
    - Metrics tracking with APIMetricsTracker
    - Standard error handlers

    Does NOT include RPC - subclasses add RPC if needed (e.g., VantaRestServer).

    Subclasses must implement:
    - _initialize_clients(**kwargs): Create RPC clients, direct references, etc.
    - _register_routes(): Register Flask route handlers
    """

    def __init__(self, api_keys_file, service_name,
                 refresh_interval=15, metrics_interval_minutes=5,
                 flask_host=None, flask_port=None, **kwargs):
        """
        Initialize base REST server.

        Args:
            api_keys_file: Path to API keys JSON file
            service_name: Name of the service for logging
            refresh_interval: How often to check for API key changes (seconds)
            metrics_interval_minutes: How often to log API metrics (minutes)
            flask_host: Host address for Flask server (default: "0.0.0.0")
            flask_port: Port for Flask server (default: 8088)
            **kwargs: Additional arguments passed to _initialize_clients()
        """
        # Store service name for logging
        self.service_name = service_name

        print(f"[REST-INIT] Step 1/9: Initializing API key handling...")
        # Initialize API key handling
        APIKeyMixin.__init__(self, api_keys_file, refresh_interval)
        print(f"[REST-INIT] Step 1/9: API key handling initialized ✓")

        print(f"[REST-INIT] Step 3/9: Setting REST server configuration...")
        # Flask configuration
        self.flask_host = flask_host or "0.0.0.0"
        self.flask_port = flask_port or 8088
        print(f"[REST-INIT] Step 3/9: Configuration set ✓")

        print(f"[REST-INIT] Step 4/9: Creating Flask app...")
        # Create Flask app
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 256 * 1024  # 256 KB
        print(f"[REST-INIT] Step 4/9: Flask app created ✓")

        print(f"[REST-INIT] Step 5/9: Contract owner loaded ✓")

        from flask_compress import Compress
        Compress(self.app)

        print(f"[REST-INIT] Step 6/9: Setting up metrics tracking...")
        # Setup metrics tracking
        self._setup_metrics(metrics_interval_minutes)
        print(f"[REST-INIT] Step 6/9: Metrics tracking initialized ✓")

        # Let subclass initialize clients/references (Step 2 happens in subclass)
        self._initialize_clients(**kwargs)

        print(f"[REST-INIT] Step 7/9: Registering routes...")
        # Let subclass register routes
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

        # Flask server state
        self._flask_thread: Optional[threading.Thread] = None
        self._flask_ready = threading.Event()

        # Start Flask server in background thread
        self.start_flask_server()

    @abstractmethod
    def _initialize_clients(self, **kwargs):
        """
        Initialize clients, RPC connections, or direct references.

        Subclasses must implement this to set up their specific dependencies.
        """
        pass

    @abstractmethod
    def _register_routes(self):
        """
        Register Flask route handlers.

        Subclasses must implement this to register their specific endpoints.
        """
        pass

    # ============================================================================
    # FLASK SERVER LIFECYCLE
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

    def shutdown(self):
        """Gracefully shutdown Flask server."""
        bt.logging.info(f"{self.service_name} shutting down...")
        self.stop_flask_server()
        bt.logging.info(f"{self.service_name} shutdown complete")

    # ============================================================================
    # METRICS TRACKING
    # ============================================================================

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

    # ============================================================================
    # ERROR HANDLERS
    # ============================================================================

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

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

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

            # Support both "Bearer <key>" and raw key formats
            if auth_header.startswith('Bearer '):
                return auth_header[7:]  # Remove "Bearer " prefix
            else:
                return auth_header  # Raw key
        except Exception:
            return None
