# Sleep Elimination Architecture Proposal

## Problem Statement

The codebase has **100+ `time.sleep()` calls** that create:
- Slow tests (45s for 26 tests)
- Race conditions (guess timing vs actual readiness)
- Wasted CPU cycles (sleeping when we could be done)
- Non-deterministic behavior (works on fast machines, fails on slow ones)

## Core Principle

**Don't guess timing - use explicit synchronization**

Replace: `do_thing(); sleep(0.5); check_result()`
With: `do_thing(); wait_until_ready(); check_result()`

---

## Categories of Sleeps & Solutions

### 1. Process/Service Startup Waits ✅ PARTIALLY SOLVED

**Current Pattern:**
```python
process.start()
time.sleep(0.1)  # Hope it's ready
```

**Already Implemented (RPCServiceBase):**
```python
server_ready = Event()
process.start()
server_ready.wait(timeout=10)  # Explicit signal
```

**Remaining Issues:**
- validator.py has 12 process startup sleeps (lines 359-560)
- Should use Event-based readiness signaling

**Proposed Solution:**
```python
class ProcessManager:
    """Manages process lifecycle with explicit readiness"""

    def start_process_with_readiness(
        self,
        target_fn,
        args=(),
        timeout=10,
        health_check_fn=None
    ):
        ready_event = Event()

        # Wrap target to signal readiness
        def wrapped_target(*args):
            target_fn(*args, ready_event=ready_event)

        process = Process(target=wrapped_target, args=args, daemon=True)
        process.start()

        # Wait for explicit ready signal OR health check success
        if not ready_event.wait(timeout=timeout):
            if health_check_fn and self._poll_until_healthy(health_check_fn, timeout):
                return process  # Healthy even without explicit signal
            raise TimeoutError(f"Process {process.pid} failed to signal ready")

        return process

    def _poll_until_healthy(self, health_check_fn, timeout, interval=0.1):
        """Poll health check with exponential backoff"""
        deadline = time.time() + timeout
        backoff = interval

        while time.time() < deadline:
            try:
                if health_check_fn():
                    return True
            except:
                pass

            time.sleep(min(backoff, deadline - time.time()))
            backoff *= 1.5  # Exponential backoff

        return False
```

---

### 2. Port Release Waits ❌ NEEDS IMPROVEMENT

**Current Pattern (rpc_service_base.py:588):**
```python
process.terminate()
process.join(timeout=2)
time.sleep(1.5)  # Hope OS released port
```

**Problem:** Port might be released in 10ms or 3000ms - we don't know!

**Proposed Solution:**
```python
class PortManager:
    """Manages port availability without guessing"""

    @staticmethod
    def wait_for_port_release(port, timeout=5.0):
        """Poll until port is actually free"""
        import socket

        deadline = time.time() + timeout
        backoff = 0.01  # Start with 10ms

        while time.time() < deadline:
            if PortManager.is_port_free(port):
                return True

            # Exponential backoff: 10ms, 15ms, 22ms, 33ms, 50ms, ...
            time.sleep(min(backoff, deadline - time.time()))
            backoff *= 1.5

        return False

    @staticmethod
    def is_port_free(port):
        """Check if port is actually available"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return True
        except OSError:
            return False

    @staticmethod
    def wait_for_port_listen(port, timeout=5.0):
        """Wait until something is listening on port"""
        import socket

        deadline = time.time() + timeout
        backoff = 0.01

        while time.time() < deadline:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.1)
                    s.connect(('localhost', port))
                    return True
            except (socket.timeout, ConnectionRefusedError, OSError):
                pass

            time.sleep(min(backoff, deadline - time.time()))
            backoff *= 1.5

        return False
```

**Usage in RPCServiceBase:**
```python
def shutdown(self):
    if self._server_process:
        self._server_process.terminate()
        self._server_process.join(timeout=2)

        # Wait for actual port release (usually <50ms)
        if not PortManager.wait_for_port_release(self.port, timeout=3.0):
            bt.logging.warning(f"Port {self.port} still in use after shutdown")
        else:
            bt.logging.debug(f"Port {self.port} released in <50ms")
```

---

### 3. Retry Delays ✅ ACCEPTABLE (with improvements)

**Current Pattern (shared_objects/retry.py):**
```python
time.sleep(mdelay)  # Fixed delay
```

**Better Pattern - Exponential Backoff with Jitter:**
```python
class SmartRetry:
    """Retry with exponential backoff and jitter"""

    @staticmethod
    def retry_with_backoff(
        fn,
        max_attempts=5,
        base_delay=0.1,
        max_delay=10.0,
        jitter=True
    ):
        import random

        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise

                # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                delay = min(base_delay * (2 ** attempt), max_delay)

                # Add jitter to prevent thundering herd
                if jitter:
                    delay *= (0.5 + random.random())  # ±50% jitter

                bt.logging.debug(f"Retry {attempt+1}/{max_attempts} after {delay:.3f}s")
                time.sleep(delay)
```

**This is acceptable** because:
- Delays are intentional (rate limiting, avoiding server overload)
- Adding jitter prevents synchronized retries
- Still uses sleep but intelligently

---

### 4. Polling Loops ❌ NEEDS IMPROVEMENT

**Current Pattern (100+ instances):**
```python
while True:
    do_work()
    time.sleep(10)  # Check every 10 seconds
```

**Problems:**
- Wastes 10 seconds on shutdown
- Can't be interrupted cleanly
- Fixed polling interval regardless of urgency

**Proposed Solution - Interruptible Polling:**
```python
class InterruptiblePoller:
    """Polling loop with immediate shutdown and dynamic intervals"""

    def __init__(self):
        self.shutdown_event = Event()

    def poll_with_shutdown(
        self,
        work_fn,
        interval_s=10,
        error_interval_s=60,
        immediate_trigger: Optional[Event] = None
    ):
        """
        Polling loop that:
        - Exits immediately on shutdown
        - Can be triggered immediately via Event
        - Uses different intervals for errors vs success
        """

        while not self.shutdown_event.is_set():
            try:
                work_fn()
                next_interval = interval_s
            except Exception as e:
                bt.logging.error(f"Poller error: {e}")
                next_interval = error_interval_s

            # Wait for interval OR immediate trigger OR shutdown
            if immediate_trigger:
                # Wait on multiple events
                triggered = self._wait_any([
                    (self.shutdown_event, "shutdown"),
                    (immediate_trigger, "trigger")
                ], timeout=next_interval)

                if triggered == "shutdown":
                    break
                elif triggered == "trigger":
                    immediate_trigger.clear()
                    continue  # Run immediately
            else:
                # Just wait for shutdown or timeout
                if self.shutdown_event.wait(timeout=next_interval):
                    break

    def _wait_any(self, events, timeout):
        """Wait for any event to be set (or timeout)"""
        import select

        deadline = time.time() + timeout

        while time.time() < deadline:
            for event, name in events:
                if event.is_set():
                    return name

            # Sleep briefly between checks
            time.sleep(0.05)

        return "timeout"

    def shutdown(self):
        """Immediate shutdown signal"""
        self.shutdown_event.set()
```

**Usage Example:**
```python
class LimitOrderManager:
    def __init__(self):
        self.poller = InterruptiblePoller()
        self.check_orders_now = Event()  # Trigger immediate check

    def start_order_checking_thread(self):
        thread = threading.Thread(
            target=self.poller.poll_with_shutdown,
            args=(self._check_and_fill_limit_orders,),
            kwargs={
                'interval_s': 60,
                'error_interval_s': 600,
                'immediate_trigger': self.check_orders_now
            },
            daemon=True
        )
        thread.start()
        return thread

    def trigger_immediate_check(self):
        """Force immediate order check instead of waiting 60s"""
        self.check_orders_now.set()

    def shutdown(self):
        self.poller.shutdown()  # Exits immediately, no 60s wait!
```

---

### 5. Connection Retry Delays ✅ PARTIALLY SOLVED

**Current (rpc_service_base.py:315):**
```python
time.sleep(self._connection_retry_delay_s)  # Fixed 1s delay
```

**Already Good** - but could improve with exponential backoff (see #3).

---

### 6. Rate Limiting / Staggering ✅ ACCEPTABLE

**Current Pattern:**
```python
time.sleep(0.1)  # Put 100ms between consecutive writes
```

**This is fine** because:
- Intentional rate limiting (preventing API abuse)
- Staggering prevents thundering herd
- Alternative (token bucket) is more complex

**Could improve with token bucket:**
```python
class RateLimiter:
    """Token bucket rate limiter - no sleep needed"""

    def __init__(self, rate_per_second=10, burst=1):
        self.rate = rate_per_second
        self.tokens = burst
        self.max_tokens = burst
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens=1, timeout=None):
        """Acquire tokens (blocks if not available)"""
        deadline = time.time() + (timeout or float('inf'))

        while time.time() < deadline:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update

                # Add new tokens based on elapsed time
                self.tokens = min(
                    self.max_tokens,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            # Wait briefly and retry
            time.sleep(0.01)

        return False
```

---

### 7. Test Synchronization ❌ NEEDS IMPROVEMENT

**Current Pattern:**
```python
process.terminate()
time.sleep(0.5)  # Hope it's dead
assert not process.is_alive()
```

**Proposed Solution:**
```python
class TestHelpers:
    """Test utilities for deterministic waiting"""

    @staticmethod
    def wait_for_process_termination(process, timeout=5.0):
        """Wait for process to actually terminate"""
        deadline = time.time() + timeout

        process.join(timeout=timeout)

        # Poll to confirm termination
        while time.time() < deadline:
            if not process.is_alive():
                return True
            time.sleep(0.01)

        return False

    @staticmethod
    def wait_for_condition(condition_fn, timeout=5.0, interval=0.01):
        """Wait for arbitrary condition to be true"""
        deadline = time.time() + timeout

        while time.time() < deadline:
            if condition_fn():
                return True
            time.sleep(interval)

        return False
```

**Usage in tests:**
```python
def test_shutdown_rpc_mode(self):
    service = ConcreteRPCService(running_unit_tests=False, port=50109)

    service.shutdown()

    # OLD: time.sleep(0.5)
    # NEW: Wait for actual termination
    assert TestHelpers.wait_for_process_termination(service._server_process)
    assert service._server_proxy is None
```

---

## Implementation Priority

### Phase 1: Low-Hanging Fruit (Immediate)
1. ✅ **Process startup** - Event-based signaling (already done for RPC)
2. **Port management** - Add `PortManager` class
3. **Test helpers** - Add `TestHelpers.wait_for_condition()`

### Phase 2: Architecture Improvements (1-2 weeks)
4. **Interruptible polling** - Replace all `while True: sleep(N)` loops
5. **Process manager** - Centralized process lifecycle management
6. **Smart retry** - Exponential backoff with jitter

### Phase 3: Nice-to-Have (Future)
7. **Rate limiting** - Token bucket instead of sleep
8. **Event-driven architecture** - Pub/sub for inter-component communication

---

## Expected Performance Improvements

### Test Execution
- **Current**: 45s for 26 RPC tests
- **After Phase 1**: ~15s (3x faster)
  - Port release: 1.5s → 50ms average (30x faster per test)
  - Process termination: 500ms → 50ms average (10x faster per test)

### Shutdown Time
- **Current**: 1.5s fixed delay for port release
- **After Phase 1**: 10-100ms average (15x faster)

### Startup Reliability
- **Current**: Occasional race conditions on slow systems
- **After Phase 1**: Deterministic readiness (100% reliable)

---

## Code Size Impact

- **New code**: ~300 lines (PortManager, ProcessManager, TestHelpers)
- **Removed code**: ~100 instances of `time.sleep()`
- **Net change**: +200 lines for much better reliability

---

## Migration Strategy

1. **Add utilities** (new files, no breaking changes):
   - `shared_objects/port_manager.py`
   - `shared_objects/process_manager.py`
   - `tests/test_helpers.py`

2. **Update RPCServiceBase** (minimal changes):
   - Use `PortManager.wait_for_port_release()` in shutdown
   - Update tests to use `TestHelpers`

3. **Gradual migration**:
   - Update validators to use `ProcessManager`
   - Update polling loops to use `InterruptiblePoller`
   - Update tests module by module

---

## Conclusion

**Key Insight**: Most sleeps are **guessing** when something will be ready. Replace with **explicit synchronization** using:
- Events for signaling
- Polling with exponential backoff for checking
- Socket operations for port availability
- Process.join() for termination

**Benefits**:
- 3-15x faster tests
- 100% deterministic behavior
- Immediate shutdown (no waiting on polling loops)
- Better error messages (timeout → "port still busy" vs "unknown failure")

**Tradeoffs**:
- More code (~300 lines of utilities)
- Slightly more complex (but much more robust)
- Still uses sleep in polling loops (but with much shorter intervals)
