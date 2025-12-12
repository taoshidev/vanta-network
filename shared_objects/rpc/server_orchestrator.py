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

    from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode, NeuronContext

    # Start all servers once at validator startup (recommended pattern with context)
    orchestrator = ServerOrchestrator.get_instance()
    context = NeuronContext(
        slack_notifier=self.slack_notifier,
        config=self.config,
        wallet=self.wallet,
        secrets=self.secrets,
        is_mainnet=self.is_mainnet
    )
    orchestrator.start_all_servers(mode=ServerMode.VALIDATOR, context=context)

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

    from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode

    # Reuse same infrastructure
    orchestrator = ServerOrchestrator.get_instance()
    orchestrator.start_all_servers(
        mode=ServerMode.BACKTESTING,
        secrets=secrets
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
import time
import inspect
import bittensor as bt
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from shared_objects.rpc.port_manager import PortManager
from shared_objects.rpc.rpc_client_base import RPCClientBase
from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.vali_config import RPCConnectionMode


class ServerMode(Enum):
    """Server execution mode."""
    TESTING = "testing"           # Unit tests - minimal servers, no WebSockets
    BACKTESTING = "backtesting"   # Backtesting - full servers, no WebSockets
    PRODUCTION = "production"     # Live validator - full servers with WebSockets
    VALIDATOR = "validator"       # Live validator - full servers with all features
    MINER = "miner"              # Live miner - minimal servers needed for signal submission


@dataclass
class NeuronContext:
    """Context object for neuron-specific server configuration (validators and miners)."""
    slack_notifier: Any = None
    config: Any = None
    wallet: Any = None
    secrets: Dict | None = None
    is_mainnet: bool = False
    is_miner: bool = False  # True for miners, False for validators

    @property
    def validator_hotkey(self) -> str:
        """Extract hotkey from wallet."""
        return self.wallet.hotkey.ss58_address if self.wallet else None


@dataclass
class ServerConfig:
    """
    Configuration for a single server.

    Note: server_class and client_class are Optional to support lazy loading
    (avoiding circular imports). They are populated in _load_classes() before use.
    """
    server_class: Optional[type]              # Server class (e.g., PositionManagerServer) - loaded lazily
    client_class: Optional[type]              # Client class (e.g., PositionManagerClient) - loaded lazily
    required_in_testing: bool                 # Whether needed in TESTING mode
    required_in_miner: bool = False           # Whether needed in MINER mode (signal submission)
    required_in_validator: bool = True        # Whether needed in VALIDATOR mode (default: all servers)
    spawn_kwargs: Optional[Dict[str, Any]] = None  # Additional kwargs for spawn_process()

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
    # IMPORTANT: Servers are ordered by dependency (servers with no deps first)
    # This order is used directly for startup sequence - do not reorder without updating dependencies!
    SERVERS = {
        # Tier 1: No dependencies (foundation servers)
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
            spawn_kwargs={'start_server': True}  # Miners need RPC server for SubtensorOpsManager
        ),
        'subtensor_ops': ServerConfig(
            server_class=None,  # SubtensorOpsServer
            client_class=None,  # SubtensorOpsClient
            required_in_testing=True,
            required_in_miner=True,
            required_in_validator=True,
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator
        ),
        'position_lock': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'perf_ledger': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator
        ),
        'live_price_fetcher': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            required_in_miner=False,  # Miners generate own signals, don't need price data
            spawn_kwargs={'disable_ws': True}  # No WebSockets in testing
        ),

        # Tier 2: Depends on tier 1
        'challenge_period': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator
        ),
        'elimination': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator
        ),

        # Tier 3: Depends on tier 2
        'position_manager': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator
        ),

        # Tier 4: Depends on tier 3
        'contract': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'debt_ledger': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # No daemon in testing
        ),
        'plagiarism': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator (not currently used)
        ),
        'plagiarism_detector': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator (overrides default=True)
        ),
        'limit_order': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator
        ),
        'mdd_checker': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={'start_daemon': False}  # No daemon in testing
        ),

        # Tier 5: Aggregation and statistics (depends on all above)
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
        'asset_selection': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            spawn_kwargs={}
        ),
        'entity': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            required_in_miner=False,  # Miners don't need entity management
            required_in_validator=True,  # Validators need entity management for subaccount tracking
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator
        ),
        'weight_calculator': ServerConfig(
            server_class=None,
            client_class=None,
            required_in_testing=True,
            required_in_miner=False,
            required_in_validator=True,  # Auto-started with other servers
            spawn_kwargs={'start_daemon': False}  # Daemon started later via orchestrator.start_server_daemons()
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
        self._servers_lock = threading.Lock()  # Protect dict writes during parallel startup

        # Lazy-load server/client classes to avoid circular imports
        self._classes_loaded = False

        # Register cleanup handlers for graceful shutdown on interruption
        self._register_cleanup_handlers()

    def _load_classes(self):
        """Lazy-load server and client classes (avoids circular imports)."""
        if self._classes_loaded:
            return

        # Import all server/client classes
        from shared_objects.rpc.common_data_server import CommonDataServer
        from shared_objects.rpc.common_data_client import CommonDataClient
        from shared_objects.rpc.metagraph_server import MetagraphServer
        from shared_objects.rpc.metagraph_client import MetagraphClient
        from shared_objects.subtensor_ops.subtensor_ops_server import SubtensorOpsServer
        from shared_objects.subtensor_ops.subtensor_ops_client import SubtensorOpsClient
        from shared_objects.locks.position_lock_server import PositionLockServer
        from shared_objects.locks.position_lock_client import PositionLockClient
        from vali_objects.contract.contract_server import ContractServer
        from vali_objects.contract.contract_client import ContractClient
        from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_server import PerfLedgerServer
        from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
        from vali_objects.challenge_period.challengeperiod_server import ChallengePeriodServer
        from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
        from vali_objects.utils.elimination.elimination_server import EliminationServer
        from vali_objects.utils.elimination.elimination_client import EliminationClient
        from vali_objects.position_management.position_manager_server import PositionManagerServer
        from vali_objects.position_management.position_manager_client import PositionManagerClient
        from vali_objects.plagiarism.plagiarism_server import PlagiarismServer
        from vali_objects.plagiarism.plagiarism_client import PlagiarismClient
        from vali_objects.plagiarism.plagiarism_detector_server import PlagiarismDetectorServer
        from vali_objects.plagiarism.plagiarism_detector_client import PlagiarismDetectorClient
        from vali_objects.utils.limit_order.limit_order_server import LimitOrderServer
        from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
        from vali_objects.utils.asset_selection.asset_selection_server import AssetSelectionServer
        from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
        from vali_objects.price_fetcher.live_price_server import LivePriceFetcherServer
        from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_server import DebtLedgerServer
        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
        from vali_objects.data_export.core_outputs_server import CoreOutputsServer
        from vali_objects.data_export.core_outputs_client import CoreOutputsClient
        from vali_objects.statistics.miner_statistics_server import MinerStatisticsServer
        from vali_objects.statistics.miner_statistics_client import MinerStatisticsClient
        from vali_objects.utils.mdd_checker.mdd_checker_server import MDDCheckerServer
        from vali_objects.utils.mdd_checker.mdd_checker_client import MDDCheckerClient
        from vali_objects.scoring.weight_calculator_server import WeightCalculatorServer
        from vali_objects.scoring.weight_calculator_client import WeightCalculatorClient
        from entitiy_management.entity_server import EntityServer
        from entitiy_management.entity_client import EntityClient

        # Update registry with classes
        self.SERVERS['common_data'].server_class = CommonDataServer
        self.SERVERS['common_data'].client_class = CommonDataClient

        self.SERVERS['metagraph'].server_class = MetagraphServer
        self.SERVERS['metagraph'].client_class = MetagraphClient

        self.SERVERS['subtensor_ops'].server_class = SubtensorOpsServer
        self.SERVERS['subtensor_ops'].client_class = SubtensorOpsClient

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

        self.SERVERS['weight_calculator'].server_class = WeightCalculatorServer
        self.SERVERS['weight_calculator'].client_class = WeightCalculatorClient

        self.SERVERS['entity'].server_class = EntityServer
        self.SERVERS['entity'].client_class = EntityClient

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
            try:
                self.shutdown_all_servers()
            except Exception:
                # Silently ignore errors during atexit cleanup
                pass

    def start_all_servers(
        self,
        mode: ServerMode = ServerMode.TESTING,
        secrets: Optional[Dict] = None,
        context: Optional[NeuronContext] = None,
        **kwargs
    ) -> None:
        """
        Start all required servers for the specified mode.

        This is idempotent - calling multiple times is safe.
        If servers are already running, does nothing.

        Args:
            mode: ServerMode enum (TESTING, BACKTESTING, PRODUCTION, VALIDATOR)
            secrets: API secrets dictionary (required for live_price_fetcher in legacy mode)
            context: NeuronContext for VALIDATOR/MINER mode (contains config, wallet, slack_notifier, secrets, etc.)
            **kwargs: Additional server-specific kwargs

        Example:
            # In tests
            orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)

            # In miner
            orchestrator.start_all_servers(mode=ServerMode.MINER, secrets=secrets)

            # In validator (recommended pattern with context)
            context = NeuronContext(
                slack_notifier=self.slack_notifier,
                config=self.config,
                wallet=self.wallet,
                secrets=self.secrets,
                is_mainnet=self.is_mainnet
            )
            orchestrator.start_all_servers(mode=ServerMode.VALIDATOR, context=context)
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

            # Store context for use in _start_server
            self._context = context

            # Kill any stale servers from previous runs
            PortManager.force_kill_all_rpc_ports()

            # Determine which servers to start based on mode
            servers_to_start = []
            for server_name, server_config in self.SERVERS.items():
                if mode == ServerMode.TESTING and not server_config.required_in_testing:
                    continue
                if mode == ServerMode.MINER and not server_config.required_in_miner:
                    continue
                if mode == ServerMode.VALIDATOR and not server_config.required_in_validator:
                    continue
                servers_to_start.append(server_name)

            # Start servers in parallel using ThreadPoolExecutor
            start_order = self._get_start_order(servers_to_start)
            assert len(start_order) == len(set(start_order)), "Duplicate servers in start order"

            total_servers = len(start_order)
            bt.logging.info(f"Starting {total_servers} servers in parallel...")

            # Track progress and errors
            completed_count = 0
            errors = []

            # Start all servers in parallel
            with ThreadPoolExecutor(max_workers=total_servers) as executor:
                # Submit all server startup tasks
                future_to_server = {
                    executor.submit(self._start_server, server_name, secrets, mode, **kwargs): server_name
                    for server_name in start_order
                }

                # Wait for all to complete and track progress
                for future in as_completed(future_to_server):
                    server_name = future_to_server[future]
                    try:
                        future.result()  # Raises exception if startup failed
                        completed_count += 1
                        bt.logging.debug(f"[{completed_count}/{total_servers}] {server_name} started successfully")
                    except Exception as e:
                        errors.append((server_name, e))
                        bt.logging.error(f"Failed to start {server_name}: {e}")

            # Check for errors
            if errors:
                error_summary = "\n".join([f"  - {name}: {str(e)}" for name, e in errors])
                raise RuntimeError(
                    f"Failed to start {len(errors)}/{total_servers} servers:\n{error_summary}"
                )

            self._started = True
            bt.logging.success(f"All {total_servers} servers started in {mode.value} mode (parallel startup)")

    def _get_start_order(self, server_names: list) -> list:
        """
        Get server start order respecting dependencies.

        The dependency order is defined by the order of servers in the SERVERS dict.
        See SERVERS definition for detailed dependency documentation.

        Args:
            server_names: List of server names to start

        Returns:
            List of server names in start order (filtered from SERVERS dict order)
        """
        # Filter to only requested servers, preserving SERVERS dict order
        return [s for s in self.SERVERS.keys() if s in server_names]

    def _start_server(
        self,
        server_name: str,
        secrets: Optional[Dict] = None,
        mode: ServerMode = ServerMode.TESTING,
        **kwargs
    ) -> None:
        """Start a single server with context-aware configuration."""
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

        # Inject context-specific parameters if context is available
        context = getattr(self, '_context', None)
        if context:
            # Add slack_notifier to ALL servers in validator mode
            if context.slack_notifier and 'slack_notifier' not in spawn_kwargs:
                spawn_kwargs['slack_notifier'] = context.slack_notifier

            # Server-specific context injection
            if server_name == 'live_price_fetcher':
                if context.secrets:
                    spawn_kwargs['secrets'] = context.secrets
                if mode == ServerMode.VALIDATOR:
                    spawn_kwargs['disable_ws'] = False  # Validator needs WebSockets
                    spawn_kwargs['start_daemon'] = True

            elif server_name == 'weight_calculator':
                if context.config:
                    spawn_kwargs['config'] = context.config
                if context.validator_hotkey:
                    spawn_kwargs['hotkey'] = context.validator_hotkey
                spawn_kwargs['is_mainnet'] = context.is_mainnet
                # Daemon started later via orchestrator.start_server_daemons()

            elif server_name == 'debt_ledger':
                if context.config and hasattr(context.config, 'slack_error_webhook_url'):
                    spawn_kwargs['slack_webhook_url'] = context.config.slack_error_webhook_url
                if context.validator_hotkey:
                    spawn_kwargs['validator_hotkey'] = context.validator_hotkey
                if context.config and hasattr(context.config, 'netuid'):
                    spawn_kwargs['netuid'] = context.config.netuid

            elif server_name in ('contract', 'asset_selection', 'entity'):
                if context.config:
                    spawn_kwargs['config'] = context.config

            elif server_name == 'elimination':
                if context.config and hasattr(context.config, 'serve'):
                    spawn_kwargs['serve'] = context.config.serve

            elif server_name == 'common_data':
                spawn_kwargs['start_daemon'] = False  # No daemon for common_data

            elif server_name == 'metagraph':
                spawn_kwargs['start_daemon'] = False  # No daemon for metagraph

        # Servers now create their own mock configs/wallets when running_unit_tests=True
        # No need for orchestrator to create them here

        # Legacy support: Add secrets for servers that need them (if not already added via context)
        if server_name == 'live_price_fetcher' and 'secrets' not in spawn_kwargs:
            if secrets is None:
                raise ValueError("secrets required for live_price_fetcher server")
            spawn_kwargs['secrets'] = secrets

        # Add api_keys_file for WebSocketNotifierServer
        if server_name == 'websocket_notifier':
            from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
            spawn_kwargs['api_keys_file'] = ValiBkpUtils.get_api_keys_file_path()

        # Handle WebSocket configuration based on mode (if not already set by context)
        if mode in (ServerMode.TESTING, ServerMode.BACKTESTING, ServerMode.MINER):
            # Testing/backtesting/miner: no WebSockets
            # Miners generate their own signals, validators need WebSockets for live validation
            if server_name == 'live_price_fetcher' and 'disable_ws' not in spawn_kwargs:
                spawn_kwargs['disable_ws'] = True

        # MINER MODE: All servers run in-process (LOCAL mode) to avoid port conflicts
        # This allows multiple miners to run on the same machine without port conflicts
        if mode == ServerMode.MINER:
            # Force connection_mode to LOCAL for all miner servers
            spawn_kwargs['connection_mode'] = RPCConnectionMode.LOCAL
            spawn_kwargs['start_server'] = False  # Don't start RPC server in LOCAL mode

            # Inject context parameters for all miner servers
            if context:
                if context.config and 'config' not in spawn_kwargs:
                    spawn_kwargs['config'] = context.config
                if context.wallet and 'wallet' not in spawn_kwargs:
                    spawn_kwargs['wallet'] = context.wallet
                if context.slack_notifier and 'slack_notifier' not in spawn_kwargs:
                    spawn_kwargs['slack_notifier'] = context.slack_notifier

            # Special handling for subtensor_ops: set is_miner flag
            if server_name == 'subtensor_ops':
                spawn_kwargs['is_miner'] = True

            bt.logging.info(f"Starting {server_name} (LOCAL mode - in-process for miner)...")
            instance = server_class(**spawn_kwargs)

            # THREAD-SAFE: Store server instance with lock
            with self._servers_lock:
                self._servers[server_name] = instance

            bt.logging.success(f"{server_name} started (LOCAL mode)")
            return

        # VALIDATOR/TESTING/BACKTESTING: Normal RPC mode (separate processes)
        # SubtensorOpsServer also runs in LOCAL mode for validators (in main validator process)
        if server_name == 'subtensor_ops':
            # Inject wallet and config from context
            if context:
                if context.config:
                    spawn_kwargs['config'] = context.config
                if context.wallet:
                    spawn_kwargs['wallet'] = context.wallet

            # SubtensorOpsServer now creates its own mock config/wallet when running_unit_tests=True
            # No need for orchestrator to create them here

            # Determine if miner or validator (will be False here since miners handled above)
            spawn_kwargs['is_miner'] = False

            # SubtensorOpsServer runs in LOCAL mode and has specific parameter requirements
            # Dynamically filter spawn_kwargs to only include parameters accepted by SubtensorOpsServer
            # This uses introspection to automatically adapt to signature changes
            sig = inspect.signature(server_class.__init__)
            accepted_params = set(sig.parameters.keys()) - {'self'}  # Exclude 'self'

            # Filter spawn_kwargs to only accepted parameters
            filtered_kwargs = {k: v for k, v in spawn_kwargs.items() if k in accepted_params}

            # Debug: Log filtered out parameters if any were removed
            removed_params = set(spawn_kwargs.keys()) - set(filtered_kwargs.keys())
            if removed_params:
                bt.logging.trace(
                    f"[subtensor_ops] Filtered out unsupported parameters: {removed_params}"
                )

            bt.logging.info(f"Starting {server_name} (LOCAL mode - in main validator process)...")
            instance = server_class(**filtered_kwargs)

            # THREAD-SAFE: Store server instance with lock
            with self._servers_lock:
                self._servers[server_name] = instance

            bt.logging.success(f"{server_name} started (LOCAL mode)")
            return

        # All other servers: spawn as separate process with RPC
        handle = server_class.spawn_process(**spawn_kwargs)

        # THREAD-SAFE: Store server handle with lock
        with self._servers_lock:
            self._servers[server_name] = handle

        bt.logging.success(f"{server_name} server started (RPC mode)")

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
        # Special handling for clients with local cache support - enable for fast lookups without RPC
        if server_name == 'asset_selection':
            # Enable local cache with 5-second refresh period for fast lookups without RPC
            # This prevents "Selected asset class: [unknown]" errors in validator.py:543
            client = client_class(
                running_unit_tests=(self._mode == ServerMode.TESTING),
                local_cache_refresh_period_ms=5000
            )
        elif server_name == 'elimination':
            # Enable local cache with 5-second refresh period for fast lookups without RPC
            # Used by validator.py:473 (get_elimination_local_cache) and validator.py:487 (get_departed_hotkey_info_local_cache)
            # Saves 66.81ms per order for elimination check, 11.26ms per order for re-registration check
            client = client_class(
                running_unit_tests=(self._mode == ServerMode.TESTING),
                local_cache_refresh_period_ms=5000
            )
        else:
            client = client_class(running_unit_tests=(self._mode == ServerMode.TESTING))

        # MINER MODE: Connect client directly to in-process server (LOCAL mode bypass RPC)
        if self._mode == ServerMode.MINER:
            server_instance = self._servers[server_name]
            client.set_direct_server(server_instance)
            bt.logging.debug(f"Client for {server_name} using LOCAL mode (direct server connection)")

        self._clients[server_name] = client

        return client

    def get_server(self, server_name: str) -> Any:
        """
        Get server instance directly (for LOCAL mode servers).

        Use this for servers that run in-process like subtensor_ops.
        For RPC servers, use get_client() instead.

        Args:
            server_name: Name of server (e.g., 'subtensor_ops')

        Returns:
            Server instance

        Raises:
            RuntimeError: If servers not started or server not found

        Example:
            subtensor_ops_server = orchestrator.get_server('subtensor_ops')
            subtensor = subtensor_ops_server.get_subtensor()
        """
        if not self._started:
            raise RuntimeError(
                "Servers not started. Call start_all_servers() first."
            )

        if server_name not in self.SERVERS:
            raise ValueError(f"Unknown server: {server_name}")

        if server_name not in self._servers:
            raise RuntimeError(f"Server {server_name} not started")

        return self._servers[server_name]

    def clear_all_test_data(self) -> None:
        """
        Clear all test data from all servers for test isolation.

        This is a convenience method for tests to reset state between test methods.
        Calls clear methods on all relevant clients, creating clients if they don't exist yet.

        Handles server failures gracefully - if a server has crashed, logs warning and continues.

        Note: If your test starts any daemons, you should explicitly stop them in tearDown()
        using orchestrator.stop_all_daemons() or by calling stop_daemon() on specific clients.

        Example usage in test setUp():
            def setUp(self):
                self.orchestrator.clear_all_test_data()
                self._create_test_data()
        """
        if not self._started:
            bt.logging.warning("Servers not started, cannot clear test data")
            return

        bt.logging.debug("Clearing all test data...")

        # Helper to get client (creates if doesn't exist)
        def get_client_safe(server_name: str):
            """Get client, creating it if it doesn't exist yet."""
            if server_name in self._servers:  # Only if server is running
                return self.get_client(server_name)
            return None

        # Helper to safely call clear method (handles server crashes)
        def safe_clear(server_name: str, clear_func, error_msg: str = ""):
            """
            Safely call a clear function, catching RPC errors.

            If server has crashed (BrokenPipeError, ConnectionError, etc.), log warning and continue.
            This prevents one crashed server from blocking cleanup of other servers.
            """
            try:
                clear_func()
            except (BrokenPipeError, ConnectionRefusedError, ConnectionError, EOFError) as e:
                bt.logging.warning(
                    f"Failed to clear {server_name} (server may have crashed): {type(e).__name__}: {e}. "
                    f"Continuing with other servers..."
                )
            except Exception as e:
                bt.logging.error(
                    f"Unexpected error clearing {server_name}: {type(e).__name__}: {e}. "
                    f"Continuing with other servers..."
                )

        # Clear metagraph data (must be first to avoid cascading issues)
        metagraph_client = get_client_safe('metagraph')
        if metagraph_client:
            safe_clear('metagraph', lambda: metagraph_client.set_hotkeys([]))

        # Clear common_data state (includes all test-sensitive state)
        common_data_client = get_client_safe('common_data')
        if common_data_client:
            # Use comprehensive clear_test_state() to reset shutdown_dict, sync_in_progress, sync_epoch
            safe_clear('common_data', lambda: common_data_client.clear_test_state())

        # Clear position manager data (positions and disk)
        position_client = get_client_safe('position_manager')
        if position_client:
            safe_clear('position_manager', lambda: position_client.clear_all_miner_positions_and_disk())

        # Clear perf ledger data
        perf_ledger_client = get_client_safe('perf_ledger')
        if perf_ledger_client:
            safe_clear('perf_ledger', perf_ledger_client.clear_all_ledger_data)

        # Clear elimination data (includes all test-sensitive state)
        elimination_client = get_client_safe('elimination')
        if elimination_client:
            # Use comprehensive clear_test_state() instead of clear_eliminations() alone
            # This resets ALL test-sensitive flags (eliminations, departed_hotkeys, first_refresh_ran, etc.)
            safe_clear('elimination', lambda: elimination_client.clear_test_state())

        # Clear challenge period data (includes all test-sensitive state)
        challenge_period_client = get_client_safe('challenge_period')
        if challenge_period_client:
            # Use comprehensive clear_test_state() instead of individual clear methods
            # This resets ALL test-sensitive flags (active_miners, elimination_reasons, refreshed_challengeperiod_start_time, etc.)
            safe_clear('challenge_period', lambda: challenge_period_client.clear_test_state())

        # Clear plagiarism data
        plagiarism_client = get_client_safe('plagiarism')
        if plagiarism_client:
            safe_clear('plagiarism', lambda: plagiarism_client.clear_plagiarism_data())

        # Clear plagiarism events (class-level cache - not RPC, always safe)
        try:
            from vali_objects.plagiarism import PlagiarismEvents
            PlagiarismEvents.clear_plagiarism_events()
        except Exception as e:
            bt.logging.warning(f"Failed to clear plagiarism events: {e}")

        # Clear limit order data
        limit_order_client = get_client_safe('limit_order')
        if limit_order_client:
            safe_clear('limit_order', lambda: limit_order_client.clear_limit_orders())

        # Clear asset selection data
        asset_selection_client = get_client_safe('asset_selection')
        if asset_selection_client:
            safe_clear('asset_selection', lambda: asset_selection_client.clear_asset_selections_for_test())

        # Clear live price fetcher test data (test candles, test price sources, and market open override)
        live_price_client = get_client_safe('live_price_fetcher')
        if live_price_client:
            def clear_live_price():
                live_price_client.clear_test_candle_data()
                live_price_client.clear_test_price_sources()
                live_price_client.clear_test_market_open()
            safe_clear('live_price_fetcher', clear_live_price)

        # Clear contract data (collateral balances and account sizes)
        contract_client = get_client_safe('contract')
        if contract_client:
            def clear_contract():
                contract_client.clear_test_collateral_balances()
                contract_client.sync_miner_account_sizes_data({})  # Empty dict = clear all
                contract_client.re_init_account_sizes()  # Reload from disk
            safe_clear('contract', clear_contract)

        # Clear entity data (entities and subaccounts)
        self.get_client('entity').clear_all_entities()

        bt.logging.debug("All test data cleared")

    def is_running(self) -> bool:
        """Check if servers are running."""
        return self._started

    def get_mode(self) -> Optional[ServerMode]:
        """Get current server mode."""
        return self._mode

    def start_individual_server(self, server_name: str, **kwargs) -> None:
        """
        Start a single server that was not started during initial startup.

        This is useful for servers that have required_in_validator=False but need
        to be started manually after certain dependencies are available.

        Args:
            server_name: Name of server to start
            **kwargs: Additional kwargs to pass to spawn_process

        Example:
            # Start weight_calculator after SubtensorOpsManager is running
            orchestrator.start_individual_server('weight_calculator')
        """
        if server_name in self._servers:
            bt.logging.debug(f"{server_name} server already started")
            return

        if server_name not in self.SERVERS:
            raise ValueError(f"Unknown server: {server_name}")

        bt.logging.info(f"Starting individual server: {server_name}")
        self._start_server(server_name, secrets=None, mode=self._mode, **kwargs)

    def start_server_daemons(self, server_names: list = None, warmup_caches: bool = True) -> None:
        """
        Start daemons for servers that defer daemon initialization.

        Optionally warms up client local caches after daemons start,
        ensuring caches are ready before validator accepts orders.

        This is useful for servers that spawn with start_daemon=False
        and need their daemons started after all servers are initialized.

        Args:
            server_names: List of server names to start daemons for.
                         If None, automatically starts daemons for all servers
                         with 'start_daemon': False in their spawn_kwargs.
            warmup_caches: If True, warm up caches after daemon startup (default: True).
                          This eliminates race conditions where orders arrive before
                          caches are populated.

        Example:
            # Explicit list (recommended for clarity)
            orchestrator.start_server_daemons([
                'position_manager',
                'elimination',
                'challenge_period',
                'perf_ledger',
                'debt_ledger'
            ])  # Cache warmup happens automatically

            # Automatic detection (starts all deferred daemons)
            orchestrator.start_server_daemons()
        """
        if not self._started:
            bt.logging.warning("Servers not started, cannot start daemons")
            return

        # If no list provided, auto-detect servers with deferred daemon startup
        if server_names is None:
            server_names = []
            for name, config in self.SERVERS.items():
                # Check if server has deferred daemon startup
                if (name in self._servers and  # Server is running
                    config.spawn_kwargs.get('start_daemon') is False):  # Daemon deferred
                    server_names.append(name)
            bt.logging.info(f"Auto-detected {len(server_names)} servers with deferred daemons: {server_names}")

        for server_name in server_names:
            # Skip servers that don't have a client class (e.g., weight_calculator)
            config = self.SERVERS.get(server_name)
            if not config or config.client_class is None:
                bt.logging.debug(f"Skipping {server_name} daemon start (no client class - server manages its own daemon)")
                continue

            client = self.get_client(server_name)
            if hasattr(client, 'start_daemon'):
                bt.logging.info(f"Starting daemon for {server_name}...")
                client.start_daemon()
                bt.logging.success(f"{server_name} daemon started")
            else:
                bt.logging.warning(f"{server_name} client has no start_daemon method")

        # Warm up caches now that daemons are running
        # This ensures caches are populated before validator starts accepting orders
        if warmup_caches:
            self.warmup_client_caches()

    def stop_all_daemons(self) -> None:
        """
        Stop all daemons for test isolation.

        Tests that start daemons should explicitly call this in tearDown() to prevent
        cross-test contamination. Not called automatically by clear_all_test_data().

        Handles failures gracefully - if a daemon can't be stopped, logs warning and continues.

        Dynamically determines which servers have daemons by checking for stop_daemon method.

        Example usage:
            def tearDown(self):
                self.orchestrator.stop_all_daemons()
        """
        if not self._started:
            return

        bt.logging.debug("Stopping all daemons...")

        # Dynamically find all servers with daemon support by checking clients
        # This eliminates the need to maintain a hardcoded list
        stopped_count = 0
        for server_name, client in list(self._clients.items()):
            # Check if this client supports daemon operations
            if not hasattr(client, 'stop_daemon'):
                continue

            try:
                client.stop_daemon()
                bt.logging.debug(f"Stopped daemon for {server_name}")
                stopped_count += 1
            except (BrokenPipeError, ConnectionRefusedError, ConnectionError, EOFError) as e:
                bt.logging.debug(
                    f"Failed to stop {server_name} daemon (server may have crashed): {type(e).__name__}. "
                    f"Continuing..."
                )
            except Exception as e:
                bt.logging.warning(
                    f"Error stopping {server_name} daemon: {type(e).__name__}: {e}. "
                    f"Continuing..."
                )

        bt.logging.debug(f"All daemons stopped ({stopped_count} servers with daemon support)")

    def call_pre_run_setup(self, perform_order_corrections: bool = True) -> None:
        """
        Call pre_run_setup on PositionManagerClient.

        Handles order corrections, perf ledger wiping, etc.

        Args:
            perform_order_corrections: Whether to perform order corrections

        Example:
            orchestrator.call_pre_run_setup(perform_order_corrections=True)
        """
        if not self._started:
            raise Exception("Servers not started, cannot run pre_run_setup")

        bt.logging.info("Running pre_run_setup on PositionManagerClient...")
        self._clients['position_manager'].pre_run_setup(
            perform_order_corrections=perform_order_corrections
        )
        bt.logging.success("pre_run_setup completed")

    def warmup_client_caches(self, timeout_per_client_s: float = 10.0) -> None:
        """
        Warm up local caches for clients that use background cache refresh.

        This ensures all caches have valid data before the validator starts
        accepting orders, preventing race conditions where orders arrive before
        caches are populated (e.g., "Selected asset class: [unknown]" errors).

        Automatically detects which clients need warmup by checking for:
        1. _local_cache_refresh_period_ms attribute (not None)
        2. populate_cache() method existence

        Should be called after start_server_daemons() and before ValidatorBase.__init__()
        since cache population may depend on daemon state being initialized.

        Args:
            timeout_per_client_s: Max time to wait for each client's cache (default: 10s)

        Raises:
            RuntimeError: If any client's cache fails to populate within timeout

        Example:
            orchestrator.start_server_daemons([...])
            orchestrator.warmup_client_caches()  # Blocks until caches populated
            # Now safe to start axon (ValidatorBase.__init__)
        """
        if not self._started:
            bt.logging.warning("Servers not started, cannot warm up caches")
            return

        # Dynamically discover which clients have local cache enabled
        # by checking if they were configured with local_cache_refresh_period_ms
        cacheable_clients = []
        for server_name in self.SERVERS.keys():
            # Skip if server not running
            if server_name not in self._servers:
                continue

            # Get client if already created (don't force creation)
            if server_name not in self._clients:
                continue

            client = self._clients[server_name]

            # Check if client has local cache enabled
            if (hasattr(client, '_local_cache_refresh_period_ms') and
                client._local_cache_refresh_period_ms is not None and
                hasattr(client, 'populate_cache')):
                cacheable_clients.append(server_name)

        if not cacheable_clients:
            bt.logging.debug("No clients with local cache found - skipping warmup")
            return

        bt.logging.info(f"Warming up caches for {len(cacheable_clients)} clients: {cacheable_clients}")

        for server_name in cacheable_clients:
            client = self._clients[server_name]

            # Force immediate synchronous population
            start_time = time.time()
            try:
                bt.logging.debug(f"Populating {server_name} cache...")
                cache_data = client.populate_cache()
                elapsed_ms = (time.time() - start_time) * 1000

                # Update the client's cache atomically (same as daemon does)
                with client._local_cache_lock:
                    client._local_cache = cache_data

                cache_size = len(cache_data) if isinstance(cache_data, dict) else 'N/A'
                bt.logging.success(
                    f"{server_name} cache warmed up in {elapsed_ms:.0f}ms ({cache_size} entries)"
                )

            except Exception as e:
                bt.logging.error(f"Failed to warm up {server_name} cache: {e}")
                raise RuntimeError(
                    f"Cache warmup failed for {server_name}. "
                    f"Cannot safely start validator without populated caches."
                ) from e

        bt.logging.success(f"All {len(cacheable_clients)} client caches warmed up and ready")


    def start_neuron_servers(
        self,
        context: NeuronContext,
        metagraph_timeout: float = 60.0
    ) -> None:
        """
        Start all servers for a neuron (miner or validator) with proper initialization sequence.

        Automatically detects neuron type from context.is_miner and starts appropriate servers.

        CRITICAL: Handles metagraph synchronization - subtensor_ops daemon starts
        and blocks until metagraph populates before other daemons can start.

        Args:
            context: Neuron context (slack_notifier, config, wallet, secrets, etc.)
                    Must have is_miner=True for miners, is_miner=False for validators
            metagraph_timeout: Max time to wait for metagraph (default: 60s)

        Example (Validator):
            context = NeuronContext(
                slack_notifier=self.slack_notifier,
                config=self.config,
                wallet=self.wallet,
                secrets=self.secrets,
                is_mainnet=self.is_mainnet,
                is_miner=False
            )
            orchestrator.start_neuron_servers(context)

        Example (Miner):
            context = NeuronContext(
                slack_notifier=self.slack_notifier,
                config=self.config,
                wallet=self.wallet,
                secrets=None,
                is_mainnet=is_mainnet,
                is_miner=True
            )
            orchestrator.start_neuron_servers(context)
        """
        # Determine mode from context
        mode = ServerMode.MINER if context.is_miner else ServerMode.VALIDATOR
        neuron_type = "miner" if context.is_miner else "validator"

        # Start all servers with context injection (daemons deferred)
        self.start_all_servers(mode=mode, context=context)

        # Critical synchronization: Wait for metagraph population
        bt.logging.info(f"[INIT] Starting SubtensorOps daemon for {neuron_type}, waiting for metagraph...")

        subtensor_ops = self.get_server('subtensor_ops')

        # Start subtensor_ops daemon
        subtensor_ops.start_daemon()

        # Block until metagraph populated (critical for dependent operations)
        subtensor_ops.wait_for_initial_update(max_wait_time=metagraph_timeout)

        bt.logging.success(f"[INIT] Metagraph populated, ready for other daemons")
        bt.logging.success(f"All {neuron_type} servers started and initialized")

    def start_validator_servers(
        self,
        context: NeuronContext,
        metagraph_timeout: float = 60.0
    ) -> None:
        """
        DEPRECATED: Use start_neuron_servers() instead.

        Backwards compatibility wrapper for validators.
        """
        # Ensure is_miner is False for validators
        context.is_miner = False
        self.start_neuron_servers(context=context, metagraph_timeout=metagraph_timeout)

    def start_miner_servers(
        self,
        context: NeuronContext,
        metagraph_timeout: float = 60.0
    ) -> None:
        """
        DEPRECATED: Use start_neuron_servers() instead.

        Backwards compatibility wrapper for miners.
        """
        # Ensure is_miner is True for miners
        context.is_miner = True
        self.start_neuron_servers(context=context, metagraph_timeout=metagraph_timeout)

    def shutdown_all_servers(self) -> None:
        """
        Shutdown all servers and disconnect all clients.

        This is called automatically at process exit.
        Can also be called manually for cleanup.
        """
        if not self._started:
            try:
                bt.logging.debug("No servers to shutdown")
            except (ValueError, OSError):
                pass  # Logging stream already closed (pytest teardown)
            return

        # Prevent recursive shutdowns
        if hasattr(self, '_shutting_down') and self._shutting_down:
            return
        if hasattr(self, '_shutting_down'):
            self._shutting_down = True

        try:
            bt.logging.info("Shutting down all servers...")
        except (ValueError, OSError):
            pass  # Logging stream already closed (pytest teardown)

        # Disconnect all clients first
        RPCClientBase.disconnect_all()
        self._clients.clear()

        # Shutdown all servers
        RPCServerBase.shutdown_all(force_kill_ports=True)
        self._servers.clear()

        self._started = False
        self._mode = None

        try:
            bt.logging.success("All servers shutdown complete")
        except (ValueError, OSError):
            pass  # Logging stream already closed (pytest teardown)

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
