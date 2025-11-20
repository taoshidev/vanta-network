# developer: jbonilla
# Copyright © 2024 Taoshi Inc
from enum import Enum
from typing import Dict, Set, List, Optional, Tuple
from multiprocessing import Process
import threading

from shared_objects.rpc_service_base import RPCServiceBase
from shared_objects.cache_controller import CacheController
from vali_objects.vali_config import ValiConfig

import bittensor as bt


class EliminationReason(Enum):
    ZOMBIE = "ZOMBIE"
    PLAGIARISM = "PLAGIARISM"
    MAX_TOTAL_DRAWDOWN = "MAX_TOTAL_DRAWDOWN"
    FAILED_CHALLENGE_PERIOD_TIME = "FAILED_CHALLENGE_PERIOD_TIME"
    FAILED_CHALLENGE_PERIOD_DRAWDOWN = "FAILED_CHALLENGE_PERIOD_DRAWDOWN"
    LIQUIDATED = "LIQUIDATED"


# Constants for departed hotkeys tracking
DEPARTED_HOTKEYS_KEY = "departed_hotkeys"


class EliminationManager(RPCServiceBase, CacheController):
    """"
    RPC Client for EliminationManager - manages elimination state via RPC.

    This client connects to EliminationManagerServer running in a separate process.
    Much faster than IPC managerized dicts (50-200x improvement on batch operations).

    We basically want to zero out the weights of the eliminated miners
    for long enough that BT deregisters them. However, there is no guarantee that they get deregistered and
    we may need to handle the case where we allow the miner to participate again. In this case, the elimination
    would already be cleared and their weight would be calculated as normal.
    """

    def __init__(self, metagraph, position_manager, challengeperiod_rpc_address=None,
                 running_unit_tests=False, shutdown_dict=None, is_backtesting=False,
                 websocket_notifier=None, contract_manager=None, position_locks=None,
                 sync_in_progress=None, slack_notifier=None, sync_epoch=None, limit_order_manager=None,
                 start_server=True):
        """
        Initialize EliminationManager.

        Args:
            metagraph: Metagraph instance
            position_manager: PositionManager instance
            challengeperiod_rpc_address: Tuple of (host, port) for ChallengePeriodManager RPC server.
                                        If None, CP integration will be disabled.
                                        Example: ("localhost", 50003)
            running_unit_tests: Whether running in test mode
            shutdown_dict: Shared shutdown flag
            is_backtesting: Whether backtesting
            websocket_notifier: WebSocketNotifier RPC client for broadcasting position updates
            contract_manager: Contract manager instance
            position_locks: Position locks manager
            sync_in_progress: Sync flag
            slack_notifier: Slack notifier
            sync_epoch: Sync epoch counter
            limit_order_manager: Limit order manager
            start_server: Whether to start RPC server immediately (vs deferred)
        """

        # Initialize RPCServiceBase
        RPCServiceBase.__init__(
            self,
            service_name=ValiConfig.RPC_ELIMINATION_SERVICE_NAME,
            port=ValiConfig.RPC_ELIMINATION_PORT,
            running_unit_tests=running_unit_tests,
            enable_health_check=True,
            health_check_interval_s=60,
            max_consecutive_failures=3,
            enable_auto_restart=True,
            slack_notifier=slack_notifier
        )

        # Initialize CacheController
        CacheController.__init__(self, metagraph=metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)

        # Store dependencies needed for server creation (using private attributes for properties)
        self._position_manager = position_manager
        self._contract_manager = contract_manager
        self.shutdown_dict = shutdown_dict
        self.is_backtesting = is_backtesting
        self.websocket_notifier = websocket_notifier
        self.position_locks = position_locks
        self.sync_in_progress = sync_in_progress
        self.sync_epoch = sync_epoch
        self.limit_order_manager = limit_order_manager

        # Store peer RPC address (NOT the object itself)
        # Server will create its own RPC client to communicate with ChallengePeriodManager
        self.challengeperiod_rpc_address = challengeperiod_rpc_address

        # Local cache for fast lookups (refreshed by background daemon thread)
        self._eliminations_cache = {}  # {hotkey: elimination_dict}
        self._departed_hotkeys_cache = {}  # {hotkey: departure_info_dict}
        self._cache_lock = threading.Lock()  # Thread-safe access

        # Start the RPC service (unless deferred via start_server=False)
        if start_server:
            self._initialize_service()

            # Start cache refresh daemon (only if not running unit tests)
            if not running_unit_tests:
                self._start_cache_refresh_daemon()
        else:
            bt.logging.info("EliminationManager: Deferring server start until initialize_server() is called")

    def initialize_server(self):
        """
        Initialize the RPC server after dependencies have been set.

        This method should be called after position_manager and challengeperiod_manager
        have been set via property setters (when start_server=False was passed to __init__).

        Example:
            # Create with deferred server start
            elim_mgr = EliminationManager(..., start_server=False)

            # Set dependencies
            elim_mgr.position_manager = position_manager
            elim_mgr.challengeperiod_manager = challengeperiod_manager

            # Now start the server with all dependencies ready
            elim_mgr.initialize_server()

            # Start daemon after server is ready
            elim_mgr.start_daemon()
        """
        if hasattr(self, '_server_proxy') and self._server_proxy is not None:
            bt.logging.warning("EliminationManager: Server already initialized, ignoring")
            return

        bt.logging.info("EliminationManager: Initializing RPC server with all dependencies")
        self._initialize_service()

        # Start cache refresh daemon (only if not running unit tests)
        if not self.running_unit_tests:
            self._start_cache_refresh_daemon()

        bt.logging.success("EliminationManager: RPC server initialized successfully")

    def start_daemon(self):
        """
        Start the daemon thread for continuous elimination processing.

        This method should be called AFTER all dependencies (position_manager,
        challengeperiod_manager) have been set via property setters.

        The daemon will process eliminations every 5 minutes, including:
        - MDD eliminations
        - Challenge period eliminations
        - Zombie hotkey cleanup
        - Performance ledger eliminations
        """
        if self.running_unit_tests:
            bt.logging.info("EliminationManager: Test mode - daemon not started")
            return

        if not self._server_proxy:
            bt.logging.warning("EliminationManager: Server not ready, cannot start daemon")
            return

        # Trigger daemon start via RPC
        self._server_proxy.start_daemon_rpc()
        bt.logging.success("EliminationManager: Daemon start requested via RPC")

    # ==================== Dependency Properties (auto-sync to server in test mode) ====================

    @property
    def position_manager(self):
        """Position manager dependency"""
        return self._position_manager

    @position_manager.setter
    def position_manager(self, value):
        """Set position manager and sync to server in test mode only"""
        self._position_manager = value
        # In test mode, also update server instance directly (no RPC needed in test mode)
        if self.running_unit_tests and hasattr(self, '_server_proxy') and self._server_proxy:
            self._server_proxy.position_manager = value
        # In production, RPC client objects cannot be sent via RPC (unpicklable).
        # Dependencies must be passed at server creation time via multiprocessing.Process args.

    @property
    def challengeperiod_manager(self):
        """
        ChallengePeriod manager dependency (test mode only).

        In production, the server uses challengeperiod_rpc_address to create its own RPC client.
        In test mode, we allow setting a direct reference for backward compatibility.
        """
        return getattr(self, '_challengeperiod_manager', None)

    @challengeperiod_manager.setter
    def challengeperiod_manager(self, value):
        """
        Set challengeperiod manager and sync to server in test mode only.

        This property exists for test mode backward compatibility where tests
        set circular references directly. In production, this is ignored.
        """
        self._challengeperiod_manager = value
        # In test mode, also update server instance directly (no RPC needed in test mode)
        if self.running_unit_tests and hasattr(self, '_server_proxy') and self._server_proxy:
            self._server_proxy.challengeperiod_manager = value
        # In production, RPC client objects cannot be sent via RPC (unpicklable).
        # Dependencies must be passed at server creation time via multiprocessing.Process args.

    @property
    def contract_manager(self):
        """Contract manager dependency"""
        return self._contract_manager

    @contract_manager.setter
    def contract_manager(self, value):
        """Set contract manager and auto-sync to server in test mode"""
        self._contract_manager = value
        # In test mode, also update server instance
        if self.running_unit_tests and hasattr(self, '_server_proxy') and self._server_proxy:
            self._server_proxy.contract_manager = value


    def _create_direct_server(self):
        """Create direct in-memory instance for tests"""
        from vali_objects.utils.elimination_manager_server import EliminationManagerServer

        return EliminationManagerServer(
            metagraph=self.metagraph,
            position_manager=self.position_manager,
            challengeperiod_rpc_address=self.challengeperiod_rpc_address,  # Pass address, not object
            running_unit_tests=self.running_unit_tests,
            shutdown_dict=self.shutdown_dict,
            is_backtesting=self.is_backtesting,
            websocket_notifier=self.websocket_notifier,
            contract_manager=self.contract_manager,
            position_locks=self.position_locks,
            sync_in_progress=self.sync_in_progress,
            slack_notifier=self.slack_notifier,
            sync_epoch=self.sync_epoch,
            limit_order_manager=self.limit_order_manager
        )

    def _start_server_process(self, address, authkey, server_ready):
        """Start RPC server in separate process"""
        from vali_objects.utils.elimination_manager_server import start_elimination_manager_server

        # Store authkey for client-only reconnection (when pickled to other processes)
        self._authkey = ValiConfig.get_rpc_authkey(self.service_name, self.port)

        process = Process(
            target=start_elimination_manager_server,
            args=(
                self.metagraph,
                self.position_manager,
                self.challengeperiod_rpc_address,  # Pass address, not object
                self.running_unit_tests,
                self.shutdown_dict,
                self.is_backtesting,
                self.websocket_notifier,
                self.contract_manager,
                self.position_locks,
                self.sync_in_progress,
                self.slack_notifier,
                self.sync_epoch,
                self.limit_order_manager,
                address,
                self._authkey,
                server_ready
            ),
            daemon=True
        )
        process.start()
        return process

    def __getstate__(self):
        """
        Prepare object for pickling (when passed to child processes).

        When elimination_manager is passed to other components (ChallengePeriodManager,
        LimitOrderManager) that run in separate processes, this method ensures the
        object can be pickled properly.

        The unpickled object will be a client-only instance that connects to the
        existing RPC server.

        Note: challengeperiod_rpc_address is a simple tuple and pickles normally.
        """
        import os

        msg = (
            f"[ELIMINATION_PICKLE] __getstate__ called in PID {os.getpid()}, "
            f"running_unit_tests={self.running_unit_tests}, port={self.port}"
        )
        bt.logging.info(msg)
        print(msg, flush=True)  # Also print to ensure visibility

        state = self.__dict__.copy()

        # Mark as client-only so unpickled instance connects to existing server
        state['_is_client_only'] = True

        # Don't pickle process/proxy objects (they're not picklable anyway)
        state['_server_process'] = None
        state['_client_manager'] = None
        state['_server_proxy'] = None

        # Don't pickle locks (they're not picklable)
        state['_cache_lock'] = None

        # Don't pickle cache data (will be refreshed after reconnection)
        state['_eliminations_cache'] = {}
        state['_departed_hotkeys_cache'] = {}

        # challengeperiod_rpc_address is a simple tuple - pickles normally

        bt.logging.debug(
            f"[ELIMINATION_PICKLE] __getstate__ complete, removed unpicklable objects"
        )

        return state

    def __setstate__(self, state):
        """
        Restore object after unpickling (in child process).

        Automatically reconnects to existing RPC server running in the main validator process.
        """
        import threading
        import os

        msg = (
            f"[ELIMINATION_UNPICKLE] __setstate__ called in PID {os.getpid()}, "
            f"running_unit_tests={state.get('running_unit_tests', 'UNKNOWN')}"
        )
        bt.logging.info(msg)
        print(msg, flush=True)  # Also print to ensure visibility

        self.__dict__.update(state)

        # Recreate cache lock (threading locks can't be pickled)
        self._cache_lock = threading.Lock()
        bt.logging.debug("[ELIMINATION_UNPICKLE] Recreated cache lock")

        # In test mode, recreate direct server instance
        if self.running_unit_tests:
            bt.logging.info("[ELIMINATION_UNPICKLE] Test mode - recreating direct server instance")
            self._server_proxy = self._create_direct_server()
            bt.logging.success(
                f"[ELIMINATION_UNPICKLE] Direct server created, type: {type(self._server_proxy)}"
            )
            return

        # Reconnect to existing RPC server
        msg_reconnect = f"[ELIMINATION_UNPICKLE] Production mode - reconnecting to RPC server on port {self.port}"
        bt.logging.info(msg_reconnect)
        print(msg_reconnect, flush=True)

        self._connect_client_only()

        # Verify connection succeeded
        if self._server_proxy is None:
            msg_error = "[ELIMINATION_UNPICKLE] CRITICAL: _server_proxy is still None after _connect_client_only()!"
            bt.logging.error(msg_error)
            print(msg_error, flush=True)
        else:
            msg_success = f"[ELIMINATION_UNPICKLE] _server_proxy successfully set: {type(self._server_proxy)}"
            bt.logging.success(msg_success)
            print(msg_success, flush=True)

    def _connect_client_only(self):
        """
        Connect to existing RPC server (client-only mode for child processes).

        This is called when elimination_manager is unpickled in a child process
        (ChallengePeriodManagerServer, LimitOrderManager, etc.).
        """
        import traceback

        bt.logging.info(
            f"[ELIMINATION_UNPICKLE] Starting client-only reconnection on port {self.port}"
        )

        # Use stable authkey (must match what server used)
        if not hasattr(self, '_authkey'):
            self._authkey = ValiConfig.get_rpc_authkey(self.service_name, self.port)
            bt.logging.debug(f"[ELIMINATION_UNPICKLE] Generated authkey from port {self.port}")

        # Connect to existing server (inherited from RPCServiceBase)
        try:
            bt.logging.info(f"[ELIMINATION_UNPICKLE] Attempting to connect to {self._address}")
            self._connect_client()
            bt.logging.success(
                f"[ELIMINATION_UNPICKLE] ✓ Client-only mode connected to existing RPC server at {self._address}"
            )
            bt.logging.info(f"[ELIMINATION_UNPICKLE] _server_proxy type: {type(self._server_proxy)}")
        except Exception as e:
            error_trace = traceback.format_exc()
            bt.logging.error(
                f"[ELIMINATION_UNPICKLE] ✗ Failed to reconnect in client-only mode: {e}\n"
                f"Address: {self._address}\n"
                f"Port: {self.port}\n"
                f"Traceback:\n{error_trace}"
            )
            # Don't raise - allow child process to continue, but log the error
            # The _server_proxy will be None and methods will fail with clearer errors
            bt.logging.error(
                "[ELIMINATION_UNPICKLE] Child process will not be able to use elimination_manager methods. "
                "This likely means the RPC server is not running or not accessible from this process."
            )

    # ==================== Cache Refresh Daemon ====================

    def _cache_refresh_loop(self):
        """
        Background daemon that periodically refreshes local cache from RPC server.

        Fetches elimination data from server and updates local cache.
        This allows validator to do pure local lookups with zero RPC calls.
        """
        import bittensor as bt
        from setproctitle import setproctitle
        import time
        from vali_objects.vali_config import ValiConfig

        setproctitle("vali_EliminationClientCacheRefresher")
        bt.logging.info(f"Elimination manager cache refresh daemon started ({ValiConfig.ELIMINATION_CACHE_REFRESH_INTERVAL_S}-second interval)")

        while not self.shutdown_dict:
            try:
                time.sleep(ValiConfig.ELIMINATION_CACHE_REFRESH_INTERVAL_S)

                # Fetch elimination data from RPC server (server has its own daemon refreshing from disk)
                eliminations, departed = self._server_proxy.get_cached_elimination_data_rpc()

                # Update local cache atomically
                with self._cache_lock:
                    self._eliminations_cache = eliminations
                    self._departed_hotkeys_cache = departed

                bt.logging.debug(
                    f"[ELIMINATION_CACHE_REFRESH] Synced: {len(eliminations)} eliminated, "
                    f"{len(departed)} departed hotkeys"
                )

            except Exception as e:
                bt.logging.error(f"Error in elimination cache refresh daemon: {e}")
                time.sleep(ValiConfig.ELIMINATION_CACHE_REFRESH_INTERVAL_S)

        bt.logging.info("Elimination manager cache refresh daemon shutting down")

    def _start_cache_refresh_daemon(self):
        """Start the background cache refresh thread."""
        import bittensor as bt
        import threading

        # Initial cache population (blocking, before daemon starts)
        try:
            eliminations, departed = self._server_proxy.get_cached_elimination_data_rpc()

            with self._cache_lock:
                self._eliminations_cache = eliminations
                self._departed_hotkeys_cache = departed

            bt.logging.info(
                f"Initial elimination cache populated: {len(eliminations)} eliminated, "
                f"{len(departed)} departed hotkeys"
            )
        except Exception as e:
            bt.logging.error(f"Error populating initial elimination cache: {e}")

        # Start daemon thread
        refresh_thread = threading.Thread(target=self._cache_refresh_loop, daemon=True)
        refresh_thread.start()
        bt.logging.info("Started elimination manager cache refresh daemon")

    # ==================== Client Methods (proxy to RPC) ====================

    def get_eliminations_lock(self):
        """
        Get the shared eliminations lock for cross-process synchronization.

        NOTE: This method should NOT be called on the RPC client. The lock is local
        to the server process. If you need synchronized access, make RPC calls which
        are automatically synchronized server-side.

        Raises:
            NotImplementedError: Always, because lock is server-side only
        """

        raise NotImplementedError(
            "get_eliminations_lock() is not available on RPC client. "
            "Locking happens automatically on server side for all RPC calls. "
            "If you need synchronized access, make RPC method calls instead."
        )

    def is_hotkey_eliminated(self, hotkey: str) -> bool:
        """
        Fast-path check if a hotkey is eliminated (O(1)).
        Use this in performance-critical paths like should_fail_early().

        Returns:
            bool: True if hotkey is eliminated, False otherwise
        """
        return self._server_proxy.is_hotkey_eliminated_rpc(hotkey)

    def hotkey_in_eliminations(self, hotkey: str) -> Optional[dict]:
        """
        Get full elimination details for a hotkey (O(1)).
        Returns the complete elimination dict with all metadata.

        Returns:
            dict or None: Elimination details if found, None otherwise
        """
        return self._server_proxy.get_elimination_rpc(hotkey)

    def get_elimination(self, hotkey: str) -> Optional[dict]:
        """
        Get elimination details for a hotkey.

        Args:
            hotkey: The hotkey to look up

        Returns:
            Elimination dict if found, None otherwise

        Example:
            elim = manager.get_elimination("miner_hotkey")
            if elim:
                print(f"Eliminated for: {elim['reason']}")
        """
        return self._server_proxy.get_elimination_rpc(hotkey)

    def get_eliminated_hotkeys(self) -> Set[str]:
        """Get all eliminated hotkeys as a set"""
        return self._server_proxy.get_eliminated_hotkeys_rpc()

    def get_eliminations_from_memory(self) -> List[dict]:
        """Get all eliminations as a list"""
        return self._server_proxy.get_eliminations_from_memory_rpc()

    def get_eliminations_from_disk(self) -> list:
        """Load eliminations from disk"""
        return self._server_proxy.get_eliminations_from_disk_rpc()

    def append_elimination_row(self, hotkey: str, current_dd: float, reason: str,
                                t_ms: int = None, price_info: dict = None, return_info: dict = None) -> None:
        """
        Add elimination row (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            hotkey: The hotkey to eliminate
            current_dd: Current drawdown
            reason: Elimination reason
            t_ms: Optional timestamp in milliseconds
            price_info: Optional price information
            return_info: Optional return information
        """
        self._server_proxy.append_elimination_row_rpc(hotkey, current_dd, reason,
                                                       t_ms=t_ms, price_info=price_info,
                                                       return_info=return_info)

    def add_elimination(self, hotkey: str, elimination_data: dict) -> bool:
        """
        Add or update an elimination record.

        Args:
            hotkey: The hotkey to eliminate
            elimination_data: Elimination dict with required fields

        Returns:
            True if added (new), False if already exists (updated)

        Raises:
            ValueError: If elimination_data is invalid

        Example:
            manager.add_elimination("miner_hotkey", {
                'hotkey': "miner_hotkey",
                'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
                'dd': 0.12,
                'elimination_initiated_time_ms': TimeUtil.now_in_millis()
            })
        """
        return self._server_proxy.add_elimination_rpc(hotkey, elimination_data)

    def remove_elimination(self, hotkey: str) -> bool:
        """
        Remove a single elimination.

        Args:
            hotkey: The hotkey to remove

        Returns:
            True if removed, False if not found

        Example:
            if manager.remove_elimination("miner_hotkey"):
                print("Elimination removed")
        """
        return self._server_proxy.remove_elimination_rpc(hotkey)

    def sync_eliminations(self, dat: list) -> list:
        """
        Sync eliminations from external source (batch update).

        Args:
            dat: List of elimination dicts to sync

        Returns:
            List of removed hotkeys
        """
        removed = self._server_proxy.sync_eliminations_rpc(dat)
        bt.logging.info(f'sync_eliminations: removed {len(removed)} hotkeys')
        return removed

    def clear_eliminations(self) -> None:
        """Clear all eliminations"""
        self._server_proxy.clear_eliminations_rpc()

    def is_hotkey_re_registered(self, hotkey: str) -> bool:
        """
        Check if a hotkey is re-registered (was previously de-registered and has re-registered).

        Args:
            hotkey: The hotkey to check

        Returns:
            True if the hotkey is in the metagraph AND in the departed_hotkeys dict, False otherwise
        """
        return self._server_proxy.is_hotkey_re_registered_rpc(hotkey)

    def get_departed_hotkeys(self) -> Dict[str, dict]:
        """Get all departed hotkeys"""
        return self._server_proxy.get_departed_hotkeys_rpc()

    def get_departed_hotkey_info(self, hotkey: str) -> Optional[dict]:
        """
        Get departed info for a single hotkey (O(1) RPC call).

        Args:
            hotkey: The hotkey to look up

        Returns:
            dict with 'detected_ms' if hotkey is departed, None otherwise
        """
        return self._server_proxy.get_departed_hotkey_info_rpc(hotkey)

    def get_cached_elimination_data(self) -> Tuple[dict, dict]:
        """
        Get cached elimination data from local cache (no RPC call!).

        The local cache is automatically refreshed by a background daemon thread.
        This method performs a fast local read with zero network overhead.

        Performance impact:
        - hotkey_in_eliminations: 66.81ms RPC → <0.01ms dict lookup
        - get_departed_hotkey_info: 11.26ms RPC → <0.01ms dict lookup

        Returns:
            Tuple of (eliminations_cache: dict, departed_hotkeys_cache: dict)
            - eliminations_cache: Dict mapping hotkey to elimination dict
            - departed_hotkeys_cache: Dict mapping hotkey to departure info dict

        Example:
            eliminations, departed = manager.get_cached_elimination_data()
            if miner_hotkey in eliminations:
                print(f"Miner is eliminated: {eliminations[miner_hotkey]}")
            if miner_hotkey in departed:
                print(f"Miner departed at {departed[miner_hotkey]['detected_ms']}")
        """
        with self._cache_lock:
            return (dict(self._eliminations_cache), dict(self._departed_hotkeys_cache))

    def get_elimination_cached(self, hotkey: str) -> Optional[dict]:
        """
        Fast local check if hotkey is eliminated AND get elimination info in one call (no RPC call!).

        Uses local cache refreshed by background daemon thread.
        This is more efficient than separate calls to check elimination status and get info.

        Args:
            hotkey: The hotkey to check

        Returns:
            Elimination dict if hotkey is eliminated, None if not eliminated

        Example:
            elim_info = manager.get_elimination_cached(hotkey)
            if elim_info:
                print(f"Eliminated for {elim_info['reason']}")
        """
        with self._cache_lock:
            return self._eliminations_cache.get(hotkey)

    def get_departed_hotkey_info_cached(self, hotkey: str) -> Optional[dict]:
        """
        Fast local check if hotkey is departed (no RPC call!).

        Uses local cache refreshed by background daemon thread.

        Args:
            hotkey: The hotkey to check

        Returns:
            Departure info dict if hotkey is departed, None otherwise
        """
        with self._cache_lock:
            return self._departed_hotkeys_cache.get(hotkey)

    def delete_eliminations(self, deleted_hotkeys):
        """
        Delete multiple eliminations.

        Note: This is not exposed as RPC. Use remove_elimination() for single deletions
        or sync_eliminations() for batch updates.
        """
        for hotkey in deleted_hotkeys:
            self.remove_elimination(hotkey)

    def process_eliminations(self, position_locks=None, iteration_epoch=None):
        """
        Trigger elimination processing.
        Uses RPC in both test and production modes.

        Args:
            position_locks: Position locks manager (optional, uses default if None)
            iteration_epoch: Epoch captured at start of iteration (optional)
        """
        self._server_proxy.process_eliminations_rpc(position_locks=position_locks, iteration_epoch=iteration_epoch)

    def handle_perf_ledger_eliminations(self, position_locks=None, iteration_epoch=None):
        """
        Process performance ledger eliminations (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            position_locks: Position locks manager (optional, uses default if None)
            iteration_epoch: Epoch captured at start of iteration (optional)
        """
        self._server_proxy.handle_perf_ledger_eliminations_rpc(position_locks=position_locks, iteration_epoch=iteration_epoch)

    def handle_first_refresh(self, position_locks, iteration_epoch=None):
        """
        Handle first refresh on startup (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            position_locks: Position locks manager
            iteration_epoch: Epoch captured at start of iteration (optional)
        """
        self._server_proxy.handle_first_refresh_rpc(position_locks, iteration_epoch)

    @property
    def first_refresh_ran(self) -> bool:
        """
        Get the first_refresh_ran flag.
        Indicates whether the first refresh has been executed after validator startup.

        Returns:
            bool: True if first refresh has run, False otherwise
        """
        return self._server_proxy.get_first_refresh_ran_rpc()

    @first_refresh_ran.setter
    def first_refresh_ran(self, value: bool):
        """
        Set the first_refresh_ran flag.

        Args:
            value: Boolean value to set
        """
        self._server_proxy.set_first_refresh_ran_rpc(value)

    def is_zombie_hotkey(self, hotkey: str, all_hotkeys_set: set) -> bool:
        """
        Check if a hotkey is a zombie (not in metagraph).
        Uses RPC in both test and production modes.

        Args:
            hotkey: The hotkey to check
            all_hotkeys_set: Set of all current hotkeys in metagraph

        Returns:
            bool: True if hotkey is a zombie, False otherwise
        """
        return self._server_proxy.is_zombie_hotkey_rpc(hotkey, all_hotkeys_set)

    def handle_mdd_eliminations(self, position_locks=None, iteration_epoch=None):
        """
        Check for maximum drawdown eliminations (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            position_locks: Position locks manager (optional, uses default if None)
            iteration_epoch: Epoch captured at start of iteration (optional)
        """
        self._server_proxy.handle_mdd_eliminations_rpc(position_locks=position_locks, iteration_epoch=iteration_epoch)

    def save_eliminations(self):
        """
        Save eliminations to disk.
        Uses RPC in both test and production modes.
        """
        self._server_proxy.save_eliminations_rpc()

    def write_eliminations_to_disk(self, eliminations: list):
        """
        Write eliminations to disk.
        Uses RPC in both test and production modes.

        Args:
            eliminations: List of elimination dicts to write
        """
        self._server_proxy.write_eliminations_to_disk_rpc(eliminations)

    @property
    def eliminations(self) -> Dict[str, dict]:
        """
        Get eliminations dict (readonly copy).
        For test mode compatibility.

        Returns:
            dict: Copy of eliminations dict mapping hotkey to elimination data
        """
        return self._server_proxy.get_eliminations_dict_rpc()
