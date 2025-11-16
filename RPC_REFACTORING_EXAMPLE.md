# RPC Service Base Class - Refactoring Example

This document shows how to refactor existing RPC services to use the new `RPCServiceBase` class.

## Benefits

1. **~80-100 lines of boilerplate eliminated per service**
2. **Consistent patterns** across all services:
   - Secure authkey generation (`secrets.token_bytes(32)`)
   - Event-based readiness signaling (10s timeout)
   - Stale server cleanup (port conflict resolution)
   - Retry logic for client connections
   - Clean shutdown with graceful termination
   - Optional health checking with auto-restart

3. **Future services trivial to add** - just implement 2 methods
4. **Built-in health monitoring** - optional health checks with:
   - Rate limiting (configurable interval)
   - Consecutive failure tracking
   - Auto-restart on failure threshold
   - Slack notifications for failures and recovery
   - Recovery detection and logging

## Before & After Comparison

### PositionManager (BEFORE) - ~100 lines of setup code

```python
class PositionManager(CacheController):
    def __init__(self, ...):
        # ... dependencies ...

        self._server_process = None
        self._rpc_client = None
        self._rpc_server_proxy = None

        if running_unit_tests:
            self._start_direct_server()
        else:
            self._start_rpc_server()

    def _start_direct_server(self):
        from vali_objects.utils.position_manager_server import PositionManagerServer
        load_from_disk = True if self.split_positions_on_disk_load else None
        self._rpc_server_proxy = PositionManagerServer(
            running_unit_tests=self.running_unit_tests,
            is_backtesting=self.is_backtesting,
            load_from_disk=load_from_disk,
            split_positions_on_disk_load=self.split_positions_on_disk_load
        )
        bt.logging.success("PositionManager using direct in-memory server (unit test mode)")

    def _start_rpc_server(self):
        from vali_objects.utils.position_manager_server import (
            start_position_manager_server,
            cleanup_stale_position_manager_server
        )
        from multiprocessing import Event
        import secrets

        address = ('localhost', 50002)
        authkey = secrets.token_bytes(32)

        # Cleanup stale servers
        cleanup_stale_position_manager_server(address[1])

        server_ready = Event()

        # Start server process
        self._server_process = Process(
            target=start_position_manager_server,
            args=(address, authkey, self.running_unit_tests, self.is_backtesting,
                  self.split_positions_on_disk_load, self.start_compaction_daemon, server_ready),
            daemon=True
        )
        self._server_process.start()
        bt.logging.info(f"Started PositionManager RPC server process (PID: {self._server_process.pid})")

        # Wait for readiness
        if not server_ready.wait(timeout=10.0):
            bt.logging.error("PositionManager RPC server failed to start within 10 seconds")
            self._server_process.terminate()
            raise TimeoutError("PositionManager RPC server startup timeout")

        bt.logging.debug("PositionManager RPC server is ready, connecting client...")

        # Connect as client
        class PositionManagerClient(BaseManager):
            pass

        PositionManagerClient.register('PositionManagerServer')
        manager = PositionManagerClient(address=address, authkey=authkey)
        manager.connect()

        self._rpc_server_proxy = manager.PositionManagerServer()
        self._rpc_client = manager
        bt.logging.success(f"PositionManager RPC client connected to server at {address}")

    def shutdown(self):
        if self.running_unit_tests:
            self._rpc_server_proxy = None
            return

        if hasattr(self, '_rpc_client') and self._rpc_client:
            try:
                bt.logging.debug("Shutting down RPC client connection")
                self._rpc_client.shutdown()
            except Exception as e:
                bt.logging.trace(f"Error shutting down RPC client: {e}")
            finally:
                self._rpc_client = None
                self._rpc_server_proxy = None

        if hasattr(self, '_server_process') and self._server_process and self._server_process.is_alive():
            bt.logging.debug(f"Terminating PositionManager RPC server process (PID: {self._server_process.pid})")
            self._server_process.terminate()
            self._server_process.join(timeout=2)
            if self._server_process.is_alive():
                bt.logging.warning(f"Force killing PositionManager RPC server process")
                self._server_process.kill()
                self._server_process.join()
            time.sleep(1.5)
```

### PositionManager (AFTER) - ~20 lines with base class

```python
from shared_objects.rpc_service_base import RPCServiceBase

class PositionManager(RPCServiceBase, CacheController):
    def __init__(self, metagraph=None, running_unit_tests=False, ...):
        # Initialize RPC base
        RPCServiceBase.__init__(
            self,
            service_name="PositionManagerServer",
            port=50002,
            running_unit_tests=running_unit_tests
        )

        # Initialize cache controller
        CacheController.__init__(self, metagraph=metagraph, running_unit_tests=running_unit_tests, ...)

        # Store dependencies for server creation
        self.is_backtesting = is_backtesting
        self.split_positions_on_disk_load = split_positions_on_disk_load
        self.start_compaction_daemon = closed_position_daemon

        # Start the RPC service
        self._initialize_service()

    def _create_direct_server(self):
        """Create direct in-memory instance for tests"""
        from vali_objects.utils.position_manager_server import PositionManagerServer

        load_from_disk = True if self.split_positions_on_disk_load else None
        return PositionManagerServer(
            running_unit_tests=self.running_unit_tests,
            is_backtesting=self.is_backtesting,
            load_from_disk=load_from_disk,
            split_positions_on_disk_load=self.split_positions_on_disk_load
        )

    def _start_server_process(self, address, authkey, server_ready):
        """Start RPC server in separate process"""
        from multiprocessing import Process
        from vali_objects.utils.position_manager_server import start_position_manager_server

        process = Process(
            target=start_position_manager_server,
            args=(
                address, authkey,
                self.running_unit_tests,
                self.is_backtesting,
                self.split_positions_on_disk_load,
                self.start_compaction_daemon,
                server_ready
            ),
            daemon=True
        )
        process.start()
        return process

    # All the RPC methods stay the same - just call self._server_proxy
    def get_positions_for_one_hotkey(self, hotkey, only_open_positions=False):
        positions = self._server_proxy.get_positions_for_one_hotkey_rpc(hotkey, only_open_positions)
        # ... rest of logic ...
```

## What Gets Eliminated

Per service, you eliminate:

1. ✅ **Process startup boilerplate** (~20 lines)
2. ✅ **Client connection with retries** (~15 lines)
3. ✅ **Readiness event handling** (~10 lines)
4. ✅ **Authkey generation** (~2 lines)
5. ✅ **Stale server cleanup** (~50 lines)
6. ✅ **Shutdown logic** (~20 lines)
7. ✅ **Direct vs RPC mode switching** (~10 lines)

**Total: ~127 lines → ~20 lines = 107 lines saved per service**

## Migration Plan

### Phase 1: Refactor PositionManager (Proof of Concept)
- ✅ Create `RPCServiceBase` class
- Refactor `PositionManager` to use it
- Run full test suite to verify
- Benefits: Immediate ~100 line reduction

### Phase 2: Refactor LimitOrderManager
- Apply same pattern
- Verify tests pass
- Benefits: Another ~80 line reduction

### Phase 3: Refactor LivePriceFetcher
- ✅ Health check logic extracted to base class
- Apply refactoring pattern to use base class
- Benefits: Another ~80 line reduction + health check available for all services

### Phase 4: Future Services
- Any new RPC service just extends `RPCServiceBase`
- Implement 2 methods: `_create_direct_server()` and `_start_server_process()`
- Benefits: New services take ~20 lines instead of ~120

## Key Design Decisions

### 1. Abstract Methods
Subclasses must implement:
- `_create_direct_server()` - Create in-memory instance for tests
- `_start_server_process()` - Start RPC server in separate process

### 2. Helper Method: `_serve_rpc()`
Base class provides `_serve_rpc()` helper that subclasses can call to reduce boilerplate further:

```python
def _start_server_process(self, address, authkey, server_ready):
    def server_main():
        from setproctitle import setproctitle
        setproctitle(f"vali_{self.service_name}")

        server = MyService(...)
        # Use base class helper to handle BaseManager setup
        self._serve_rpc(server, address, authkey, server_ready)

    process = Process(target=server_main, daemon=True)
    process.start()
    return process
```

### 3. Service Name as Identifier
The `service_name` parameter is used for:
- Logging messages
- BaseManager registration
- Process name detection during cleanup
- Process title (setproctitle)

### 4. Secure by Default
- Always uses `secrets.token_bytes(32)` for authkey (cryptographically secure)
- Never hardcodes authkeys like `b'limit_order_manager'`

### 5. Robust Cleanup
- POSIX-only (graceful on other systems)
- Checks process name before killing
- SIGTERM → SIGKILL escalation
- Port release delay (1.5s)

## Testing Strategy

1. **Unit Tests**: Verify direct mode still works
2. **Integration Tests**: Verify RPC mode still works
3. **Port Conflict Tests**: Verify stale server cleanup
4. **Shutdown Tests**: Verify clean termination

## Example: New Service in 20 Lines

Adding a new RPC service becomes trivial:

```python
from shared_objects.rpc_service_base import RPCServiceBase

class MyNewService(RPCServiceBase):
    def __init__(self, dependency1, dependency2, running_unit_tests=False):
        super().__init__(
            service_name="MyNewService",
            port=50003,  # Next available port
            running_unit_tests=running_unit_tests
        )
        self.dependency1 = dependency1
        self.dependency2 = dependency2
        self._initialize_service()

    def _create_direct_server(self):
        return MyNewServiceServer(self.dependency1, self.dependency2)

    def _start_server_process(self, address, authkey, server_ready):
        def server_main():
            from setproctitle import setproctitle
            setproctitle("vali_MyNewService")
            server = MyNewServiceServer(self.dependency1, self.dependency2)
            self._serve_rpc(server, address, authkey, server_ready)

        from multiprocessing import Process
        process = Process(target=server_main, daemon=True)
        process.start()
        return process

    # Client API
    def do_something(self, arg):
        return self._server_proxy.do_something_rpc(arg)
```

**That's it!** All the process management, connection handling, cleanup, etc. is handled by the base class.

## Health Check Implementation

The base class now includes modular health check functionality extracted from LivePriceFetcher with improvements.

### Server-Side Requirements

Your RPC server must implement a `health_check_rpc()` method:

```python
class MyServiceServer:
    def health_check_rpc(self) -> dict:
        """Health check endpoint for RPC monitoring"""
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            # Optional: Add service-specific health info
            "active_connections": len(self.connections),
            "queue_size": self.queue.qsize()
        }
```

### Client-Side Usage

#### Option 1: Manual Health Checks

Enable health checks and call periodically:

```python
class MyServiceClient(RPCServiceBase):
    def __init__(self, slack_notifier=None):
        super().__init__(
            service_name="MyService",
            port=50003,
            enable_health_check=True,
            health_check_interval_s=60,  # Check every 60 seconds
            max_consecutive_failures=3,  # Restart after 3 failures
            enable_auto_restart=True,
            slack_notifier=slack_notifier
        )
        self._initialize_service()

# In your main loop:
while True:
    current_time = TimeUtil.now_in_millis()
    service.health_check(current_time)
    time.sleep(1)
```

#### Option 2: Dedicated Health Check Daemon

Create a separate process for monitoring (recommended for production):

```python
from multiprocessing import Process
from setproctitle import setproctitle

class HealthCheckDaemon:
    def __init__(self, service_client, slack_notifier=None):
        self.service = service_client
        self.slack_notifier = slack_notifier

    def run(self):
        setproctitle("vali_MyService_HealthChecker")
        while True:
            try:
                current_time = TimeUtil.now_in_millis()
                is_healthy = self.service.health_check(current_time)

                if not is_healthy:
                    bt.logging.warning(f"Health check failed")

                time.sleep(60)  # Check every minute

            except Exception as e:
                bt.logging.error(f"Health check daemon error: {e}")
                if self.slack_notifier:
                    self.slack_notifier.send_message(
                        f"❌ Health check daemon error: {e}",
                        level="error"
                    )
                time.sleep(60)

# Start the daemon
service = MyServiceClient(slack_notifier=slack_notifier)
daemon = HealthCheckDaemon(service, slack_notifier)
health_process = Process(target=daemon.run, daemon=True)
health_process.start()
```

### Health Check Features

1. **Rate Limiting**: Only performs actual RPC check if `health_check_interval_s` has passed
2. **Consecutive Failure Tracking**: Counts failures before triggering restart
3. **Auto-Restart**: Gracefully restarts server on failure threshold
4. **Recovery Detection**: Logs and notifies when server recovers
5. **Slack Integration**: Optional notifications for failures and recovery
6. **Graceful Degradation**: Can disable auto-restart for manual intervention

### Improvements Over Original Implementation

The base class implementation improves upon LivePriceFetcher's health check:

1. **Configurable Parameters**: All timeouts and thresholds are configurable
2. **Better Error Handling**: Distinguishes between RPC errors and status errors
3. **Recovery Notifications**: Alerts when service recovers after failures
4. **Traceback Logging**: Full error traces for debugging
5. **Graceful Shutdown**: Proper cleanup before restart
6. **Port Release Delay**: Ensures port is available before restart (2s delay)
7. **Disable Auto-Restart**: Option to alert but not restart for manual intervention

### Example with All Features

```python
from shared_objects.rpc_service_base import RPCServiceBase

class LivePriceFetcher(RPCServiceBase):
    def __init__(self, slack_notifier=None, running_unit_tests=False):
        super().__init__(
            service_name="LivePriceFetcher",
            port=50001,
            running_unit_tests=running_unit_tests,
            # Health check configuration
            enable_health_check=True,
            health_check_interval_s=60,      # Check every 60 seconds
            max_consecutive_failures=3,       # Restart after 3 failures
            enable_auto_restart=True,         # Auto-restart on failure
            slack_notifier=slack_notifier     # Slack notifications
        )
        self._initialize_service()

    def _create_direct_server(self):
        from data_generator.live_price_fetcher_server import LivePriceFetcherServer
        return LivePriceFetcherServer(running_unit_tests=True)

    def _start_server_process(self, address, authkey, server_ready):
        from multiprocessing import Process
        from setproctitle import setproctitle
        from data_generator.live_price_fetcher_server import LivePriceFetcherServer

        def server_main():
            setproctitle("vali_LivePriceFetcher")
            server = LivePriceFetcherServer()
            self._serve_rpc(server, address, authkey, server_ready)

        process = Process(target=server_main, daemon=True)
        process.start()
        return process

    # Client API
    def get_latest_price(self, symbol):
        return self._server_proxy.get_latest_price_rpc(symbol)
```
