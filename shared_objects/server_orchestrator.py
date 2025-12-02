"""
Server Orchestrator - Centralized server lifecycle management.

This module provides a singleton that manages all RPC servers, ensuring:
- Servers start once and are shared across tests, backtesting, and validator
- Fast test execution (no per-test-class server startup)
- Graceful cleanup on process exit, interruption (Ctrl+C), or kill signals
- Thread-safe initialization

Usage in tests:

    from shared_objects.server_orchestrator import ServerOrchestrator

    class TestMyFeature(TestBase):
        @classmethod
        def setUpClass(cls):
            # Get shared servers (starts them if not already running)
            orchestrator = ServerOrchestrator.get_instance()

            # Get clients (servers guaranteed ready)
            cls.position_client = orchestrator.get_client('position_manager')
            cls.perf_ledger_client = orchestrator.get_client('perf_ledger')

        def setUp(self):
            # Clear data for test isolation (fast - no server restart)
            self.position_client.clear_all_miner_positions_and_disk()
            self.perf_ledger_client.clear_all_ledger_data()

Usage in validator.py:

    from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode

    # Start all servers once at validator startup
    orchestrator = ServerOrchestrator.get_instance()
    orchestrator.start_all_servers(
        mode=ServerMode.PRODUCTION,
        secrets=self.secrets,
        config=self.config
    )

    # Get clients
    self.position_client = orchestrator.get_client('position_manager')

Usage in miner.py:

    from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
    from vali_objects.utils.vali_utils import ValiUtils

    # Start only required servers for miners (common_data, metagraph)
    orchestrator = ServerOrchestrator.get_instance()
    secrets = ValiUtils.get_secrets(running_unit_tests=False)
    orchestrator.start_all_servers(
        mode=ServerMode.MINER,
        secrets=secrets
    )

    # Get client (servers guaranteed ready, no connection errors)
    self.metagraph_client = orchestrator.get_client('metagraph')

Usage in backtesting:

    from shared_objects.server_orchestrator import ServerOrchestrator

    # Reuse same infrastructure
    orchestrator = ServerOrchestrator.get_instance()
    orchestrator.start_all_servers(
        secrets=secrets,
        config=config
    )

Cleanup and Interruption Handling:

    The orchestrator automatically registers cleanup handlers to ensure servers
    are properly shut down in all scenarios:

    - Normal exit: atexit handler calls shutdown_all_servers()
    - Ctrl+C (SIGINT): Signal handler catches interrupt and shuts down gracefully
    - Kill signal (SIGTERM): Signal handler catches and shuts down gracefully
    - Destructor: __del__ method ensures cleanup even if handlers fail

    This prevents:
    - Orphaned server processes
    - Port conflicts on subsequent test runs
    - Resource leaks
    - Stale RPC connections

    No manual cleanup needed - the orchestrator handles it automatically!
"""

import threading
import signal
import atexit
import sys
import bittensor as bt
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from shared_objects.port_manager import PortManager
from shared_objects.rpc_client_base import RPCClientBase
from shared_objects.rpc_server_base import RPCServerBase


class ServerMode(Enum):
    """Server execution mode."""
    TESTING = "testing"           # Unit tests - minimal servers, no WebSockets
    BACKTESTING = "backtesting"   # Backtesting - full servers, no WebSockets
    PRODUCTION = "production"     # Live validator - full servers with WebSockets
    MINER = "miner"              # Live miner - minimal servers needed for signal submission


@dataclass
class ServerConfig:
    """Configuration for a single server."""
    server_class: type              # Server class (e.g., PositionManagerServer)
    client_class: type              # Client class (e.g., PositionManagerClient)
    required_in_testing: bool       # Whether needed in TESTING mode
    required_in_miner: bool = False # Whether needed in MINER mode (signal submission)
    spawn_kwargs: Dict[str, Any] = None  # Additional kwargs for spawn_process()

    def __post_init__(self):
        if self.spawn_kwargs is None:
            self.spawn_kwargs = {}


class ServerOrchestrator:
    """
    Singleton that manages all RPC server lifecycle.

    Ensures servers are started once and shared across:
    - Multiple test classes
    - Backtesting runs
    - Validator execution

    Thread-safe initialization with lazy server startup.
    """

    _instance: Optional['ServerOrchestrator'] = None
    _lock = threading.Lock()

    # Server registry - defines all available servers
    # Format: server_name -> ServerConfig
    SERVERS = {
        'common_data': ServerConfig(
            server_class=None,  # Imported lazily to avoid circular imports
            client_class=None,
            required_in_testing=True,
            required_in_miner=True,  # Miners need shared state
            spawn_kwargs={}
        ),
        'metagraph': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            required_in_miner=True,  # Miners need metagraph data
            spawn_kwargs={'start_server': True}  # Miners need RPC server for MetagraphUpdater
        ),
        'position_lock': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'contract': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'perf_ledger': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'challenge_period': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}
        ),
        'elimination': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'position_manager': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'plagiarism': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'plagiarism_detector': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'limit_order': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'asset_selection': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'live_price_fetcher': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            required_in_miner=False,  # Miners generate own signals, don't need price data
            spawn_kwargs={'disable_ws': True}  # No WebSockets in testing
        ),
        'debt_ledger': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # No daemon in testing
        ),
        'core_outputs': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # No daemon in testing
        ),
        'miner_statistics': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # No daemon in testing
        ),
        'mdd_checker': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # No daemon in testing
        ),
        'websocket_notifier': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=False,  # Optional - only in PRODUCTION mode
            required_in_miner=False,
            spawn_kwargs={}
        ),
    }

    @classmethod
    def get_instance(cls) -> 'ServerOrchestrator':
        """
        Get singleton instance (thread-safe).

        Returns:
            ServerOrchestrator instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton (for testing only).

        Shuts down all servers and clears the singleton.
        Use in test teardown to ensure clean state.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown_all_servers()
                cls._instance = None

    def __init__(self):
        """Initialize orchestrator (private - use get_instance())."""
        if ServerOrchestrator._instance is not None:
            raise RuntimeError("Use ServerOrchestrator.get_instance() instead of direct instantiation")

        self._servers: Dict[str, Any] = {}  # server_name -> handle
        self._clients: Dict[str, Any] = {}  # server_name -> client instance
        self._mode: Optional[ServerMode] = None
        self._started = False
        self._init_lock = threading.Lock()

        # Lazy-load server/client classes to avoid circular imports
        self._classes_loaded = False

        # Register cleanup handlers for graceful shutdown on interruption
        self._register_cleanup_handlers()

    def _load_classes(self):
        """Lazy-load server and client classes (avoids circular imports)."""
        if self._classes_loaded:
            return

        # Import all server/client classes
        from shared_objects.common_data_server import CommonDataServer, CommonDataClient
        from shared_objects.metagraph_server import MetagraphServer, MetagraphClient
        from vali_objects.utils.position_lock_server import PositionLockServer, PositionLockClient
        from vali_objects.utils.contract_server import ContractServer, ContractClient
        from vali_objects.vali_dataclasses.perf_ledger_server import PerfLedgerServer, PerfLedgerClient
        from vali_objects.utils.challengeperiod_server import ChallengePeriodServer
        from vali_objects.utils.challengeperiod_client import ChallengePeriodClient
        from vali_objects.utils.elimination_server import EliminationServer
        from vali_objects.utils.elimination_client import EliminationClient
        from vali_objects.utils.position_manager_server import PositionManagerServer
        from vali_objects.utils.position_manager_client import PositionManagerClient
        from vali_objects.utils.plagiarism_server import PlagiarismServer, PlagiarismClient
        from vali_objects.utils.plagiarism_detector_server import PlagiarismDetectorServer, PlagiarismDetectorClient
        from vali_objects.utils.limit_order_server import LimitOrderServer, LimitOrderClient
        from vali_objects.utils.asset_selection_server import AssetSelectionServer
        from vali_objects.utils.asset_selection_client import AssetSelectionClient
        from vali_objects.utils.live_price_server import LivePriceFetcherServer, LivePriceFetcherClient
        from vali_objects.vali_dataclasses.debt_ledger_server import DebtLedgerServer, DebtLedgerClient
        from runnable.core_outputs_server import CoreOutputsServer, CoreOutputsClient
        from runnable.miner_statistics_server import MinerStatisticsServer, MinerStatisticsClient
        from vali_objects.utils.mdd_checker_server import MDDCheckerServer
        from vali_objects.utils.mdd_checker_client import MDDCheckerClient
        from vanta_api.websocket_server import WebSocketServer
        from vanta_api.websocket_notifier import WebSocketNotifierClient

        # Update registry with classes
        self.SERVERS['common_data'].server_class = CommonDataServer
        self.SERVERS['common_data'].client_class = CommonDataClient

        self.SERVERS['metagraph'].server_class = MetagraphServer
        self.SERVERS['metagraph'].client_class = MetagraphClient

        self.SERVERS['position_lock'].server_class = PositionLockServer
        self.SERVERS['position_lock'].client_class = PositionLockClient

        self.SERVERS['contract'].server_class = ContractServer
        self.SERVERS['contract'].client_class = ContractClient

        self.SERVERS['perf_ledger'].server_class = PerfLedgerServer
        self.SERVERS['perf_ledger'].client_class = PerfLedgerClient

        self.SERVERS['challenge_period'].server_class = ChallengePeriodServer
        self.SERVERS['challenge_period'].client_class = ChallengePeriodClient

        self.SERVERS['elimination'].server_class = EliminationServer
        self.SERVERS['elimination'].client_class = EliminationClient

        self.SERVERS['position_manager'].server_class = PositionManagerServer
        self.SERVERS['position_manager'].client_class = PositionManagerClient

        self.SERVERS['plagiarism'].server_class = PlagiarismServer
        self.SERVERS['plagiarism'].client_class = PlagiarismClient

        self.SERVERS['plagiarism_detector'].server_class = PlagiarismDetectorServer
        self.SERVERS['plagiarism_detector'].client_class = PlagiarismDetectorClient

        self.SERVERS['limit_order'].server_class = LimitOrderServer
        self.SERVERS['limit_order'].client_class = LimitOrderClient

        self.SERVERS['asset_selection'].server_class = AssetSelectionServer
        self.SERVERS['asset_selection'].client_class = AssetSelectionClient

        self.SERVERS['live_price_fetcher'].server_class = LivePriceFetcherServer
        self.SERVERS['live_price_fetcher'].client_class = LivePriceFetcherClient

        self.SERVERS['debt_ledger'].server_class = DebtLedgerServer
        self.SERVERS['debt_ledger'].client_class = DebtLedgerClient

        self.SERVERS['core_outputs'].server_class = CoreOutputsServer
        self.SERVERS['core_outputs'].client_class = CoreOutputsClient

        self.SERVERS['miner_statistics'].server_class = MinerStatisticsServer
        self.SERVERS['miner_statistics'].client_class = MinerStatisticsClient

        self.SERVERS['mdd_checker'].server_class = MDDCheckerServer
        self.SERVERS['mdd_checker'].client_class = MDDCheckerClient

        self.SERVERS['websocket_notifier'].server_class = WebSocketServer
        self.SERVERS['websocket_notifier'].client_class = WebSocketNotifierClient

        self._classes_loaded = True

    def _register_cleanup_handlers(self):
        """
        Register signal handlers and atexit hook for graceful cleanup.

        This ensures servers are properly shut down even if:
        - User hits Ctrl+C (SIGINT)
        - Process is killed (SIGTERM)
        - Python exits normally (atexit)
        """
        # Register atexit handler for normal exit
        atexit.register(self._cleanup_on_exit)

        # Register signal handlers for interruptions
        # Use a flag to prevent recursive signal handling
        self._shutting_down = False

        def signal_handler(signum, frame):
            """Handle SIGINT and SIGTERM gracefully."""
            if self._shutting_down:
                return
            self._shutting_down = True

            signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            bt.logging.info(f"\n{signal_name} received, shutting down servers gracefully...")

            try:
                self.shutdown_all_servers()
            except Exception as e:
                bt.logging.error(f"Error during signal cleanup: {e}")
            finally:
                # Re-raise to allow default signal handling
                sys.exit(0)

        # Register handlers for SIGINT (Ctrl+C) and SIGTERM (kill)
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (ValueError, OSError):
            # Signal registration may fail in some contexts (e.g., threads)
            bt.logging.debug("Could not register signal handlers (not in main thread?)")

    def _cleanup_on_exit(self):
        """Cleanup handler called by atexit on normal exit."""
        if not self._shutting_down and self._started:
            bt.logging.debug("atexit: Cleaning up servers...")
            try:
                self.shutdown_all_servers()
            except Exception as e:
                bt.logging.error(f"Error during atexit cleanup: {e}")

    def start_all_servers(
        self,
        mode: ServerMode = ServerMode.TESTING,
        secrets: Optional[Dict] = None,
        config: Optional[Any] = None,
        **kwargs
    ) -> None:
        """
        Start all required servers for the specified mode.

        This is idempotent - calling multiple times is safe.
        If servers are already running, does nothing.

        Args:
            mode: ServerMode enum (TESTING, BACKTESTING, PRODUCTION)
            secrets: API secrets dictionary (required for live_price_fetcher)
            config: Optional configuration object
            **kwargs: Additional server-specific kwargs

        Example:
            # In tests
            orchestrator.start_all_servers(mode=ServerMode.TESTING)

            # In validator
            orchestrator.start_all_servers(
                mode=ServerMode.PRODUCTION,
                secrets=self.secrets,
                config=self.config
            )
        """
        with self._init_lock:
            if self._started and self._mode == mode:
                bt.logging.debug(f"Servers already started in {mode.value} mode")
                return

            if self._started and self._mode != mode:
                bt.logging.warning(
                    f"Servers already started in {self._mode.value} mode, "
                    f"but {mode.value} mode requested. Shutting down and restarting..."
                )
                self.shutdown_all_servers()

            bt.logging.info(f"Starting servers in {mode.value} mode...")
            self._mode = mode
            self._load_classes()

            # Kill any stale servers from previous runs
            PortManager.force_kill_all_rpc_ports()

            # Determine which servers to start based on mode
            servers_to_start = []
            for server_name, config in self.SERVERS.items():
                if mode == ServerMode.TESTING and not config.required_in_testing:
                    continue
                if mode == ServerMode.MINER and not config.required_in_miner:
                    continue
                servers_to_start.append(server_name)

            # Start servers in dependency order
            start_order = self._get_start_order(servers_to_start)

            for server_name in start_order:
                self._start_server(server_name, secrets=secrets, mode=mode, **kwargs)

            self._started = True
            bt.logging.success(f"All servers started in {mode.value} mode")

    def _get_start_order(self, server_names: list) -> list:
        """
        Get server start order respecting dependencies.

        Dependency graph:
        - common_data: no dependencies (start first)
        - metagraph: no dependencies
        - position_lock: no dependencies
        - contract: no dependencies
        - perf_ledger: no dependencies
        - live_price_fetcher: no dependencies
        - debt_ledger: depends on perf_ledger
        - asset_selection: depends on common_data
        - challenge_period: depends on common_data, asset_selection
        - elimination: depends on perf_ledger, challenge_period
        - position_manager: depends on challenge_period, elimination
        - websocket_notifier: depends on position_manager (broadcasts position updates)
        - plagiarism: depends on position_manager
        - plagiarism_detector: depends on plagiarism, position_manager
        - limit_order: depends on position_manager
        - mdd_checker: depends on position_manager, elimination
        - core_outputs: depends on all above (aggregates checkpoint data)
        - miner_statistics: depends on all above (generates miner statistics)

        Returns:
            List of server names in start order
        """
        # Define dependency order (servers with no deps first)
        order = [
            'common_data',
            'metagraph',
            'position_lock',
            'contract',
            'perf_ledger',
            'live_price_fetcher',
            'debt_ledger',
            'asset_selection',
            'challenge_period',
            'elimination',
            'position_manager',
            'websocket_notifier',
            'plagiarism',
            'plagiarism_detector',
            'limit_order',
            'mdd_checker',
            'core_outputs',
            'miner_statistics'
        ]

        # Filter to only requested servers, preserving order
        return [s for s in order if s in server_names]

    def _start_server(
        self,
        server_name: str,
        secrets: Optional[Dict] = None,
        mode: ServerMode = ServerMode.TESTING,
        **kwargs
    ) -> None:
        """Start a single server."""
        if server_name in self._servers:
            bt.logging.debug(f"{server_name} server already started")
            return

        config = self.SERVERS[server_name]
        server_class = config.server_class

        if server_class is None:
            raise RuntimeError(f"Server class not loaded for {server_name}")

        # Prepare spawn kwargs
        spawn_kwargs = {
            'running_unit_tests': mode == ServerMode.TESTING,
            'is_backtesting': mode == ServerMode.BACKTESTING,
            **config.spawn_kwargs,
            **kwargs
        }

        # Add secrets for servers that need them
        if server_name == 'live_price_fetcher':
            if secrets is None:
                raise ValueError("secrets required for live_price_fetcher server")
            spawn_kwargs['secrets'] = secrets

        # Add api_keys_file for WebSocketNotifierServer
        if server_name == 'websocket_notifier':
            from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
            spawn_kwargs['api_keys_file'] = ValiBkpUtils.get_api_keys_file_path()

        # Handle WebSocket configuration based on mode
        if mode in (ServerMode.TESTING, ServerMode.BACKTESTING, ServerMode.MINER):
            # Testing/backtesting/miner: no WebSockets
            # Miners generate their own signals, validators need WebSockets for live validation
            if server_name == 'live_price_fetcher':
                spawn_kwargs['disable_ws'] = True

        bt.logging.info(f"Starting {server_name} server...")

        # Spawn server process (blocks until ready)
        handle = server_class.spawn_process(**spawn_kwargs)
        self._servers[server_name] = handle

        bt.logging.success(f"{server_name} server started")

    def get_client(self, server_name: str) -> Any:
        """
        Get client for a server (creates on first call, caches afterward).

        Args:
            server_name: Name of server (e.g., 'position_manager')

        Returns:
            Client instance

        Raises:
            RuntimeError: If servers not started or server not found

        Example:
            client = orchestrator.get_client('position_manager')
            positions = client.get_all_miner_positions('hotkey')
        """
        if not self._started:
            raise RuntimeError(
                "Servers not started. Call start_all_servers() first."
            )

        if server_name not in self.SERVERS:
            raise ValueError(f"Unknown server: {server_name}")

        # Return cached client if exists
        if server_name in self._clients:
            return self._clients[server_name]

        # Create new client
        config = self.SERVERS[server_name]
        client_class = config.client_class

        if client_class is None:
            raise RuntimeError(f"Client class not loaded for {server_name}")

        bt.logging.debug(f"Creating client for {server_name}")

        # Create client (will connect to running server)
        client = client_class(running_unit_tests=(self._mode == ServerMode.TESTING))
        self._clients[server_name] = client

        return client

    def clear_all_test_data(self) -> None:
        """
        Clear all test data from all servers for test isolation.

        This is a convenience method for tests to reset state between test methods.
        Calls clear methods on all relevant clients.

        Example usage in test setUp():
            def setUp(self):
                self.orchestrator.clear_all_test_data()
                self._create_test_data()
        """
        if not self._started:
            bt.logging.warning("Servers not started, cannot clear test data")
            return

        bt.logging.debug("Clearing all test data...")

        # Clear position manager data (positions and disk)
        if 'position_manager' in self._clients:
            self._clients['position_manager'].clear_all_miner_positions_and_disk()

        # Clear perf ledger data
        if 'perf_ledger' in self._clients:
            self._clients['perf_ledger'].clear_all_ledger_data()

        # Clear elimination data
        if 'elimination' in self._clients:
            self._clients['elimination'].clear_eliminations()

        # Clear challenge period data
        if 'challenge_period' in self._clients:
            self._clients['challenge_period']._clear_challengeperiod_in_memory_and_disk()
            self._clients['challenge_period'].clear_elimination_reasons()

        # Clear plagiarism data
        if 'plagiarism' in self._clients:
            self._clients['plagiarism'].clear_plagiarism_data()

        # Clear limit order data
        if 'limit_order' in self._clients:
            self._clients['limit_order'].clear_limit_orders()

        bt.logging.debug("All test data cleared")

    def is_running(self) -> bool:
        """Check if servers are running."""
        return self._started

    def get_mode(self) -> Optional[ServerMode]:
        """Get current server mode."""
        return self._mode

    def shutdown_all_servers(self) -> None:
        """
        Shutdown all servers and disconnect all clients.

        This is called automatically at process exit.
        Can also be called manually for cleanup.
        """
        if not self._started:
            bt.logging.debug("No servers to shutdown")
            return

        # Prevent recursive shutdowns
        if hasattr(self, '_shutting_down') and self._shutting_down:
            return
        if hasattr(self, '_shutting_down'):
            self._shutting_down = True

        bt.logging.info("Shutting down all servers...")

        # Disconnect all clients first
        RPCClientBase.disconnect_all()
        self._clients.clear()

        # Shutdown all servers
        RPCServerBase.shutdown_all(force_kill_ports=True)
        self._servers.clear()

        self._started = False
        self._mode = None

        bt.logging.success("All servers shutdown complete")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown_all_servers()
        except Exception:
            pass


# Convenience function for common usage pattern
def get_orchestrator() -> ServerOrchestrator:
    """
    Get the singleton ServerOrchestrator instance.

    Convenience alias for ServerOrchestrator.get_instance().

    Returns:
        ServerOrchestrator instance
    """
    return ServerOrchestrator.get_instance()
