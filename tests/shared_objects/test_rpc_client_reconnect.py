"""
Unit coverage for RPCClientBase's self-heal-on-server-bounce path.

The existing client tests all run in LOCAL mode (set_direct_server), which short-circuits
_invoke_rpc before any transport logic — so none of them exercise the reconnect cycle, the
retry=False fail-fast opt-out, or _reset_connection's "keep the instance registered + cache
daemon alive" contract. These tests drive that path directly with a scripted fake proxy and a
mocked connect(), no real RPC server required.
"""
import threading

import pytest

import shared_objects.rpc.rpc_client_base as rcb
from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode


class _HarnessClient(RPCClientBase):
    """Minimal RPC-mode client; never connects on init (we inject/mocking transport ourselves)."""

    def __init__(self):
        super().__init__(
            service_name="ReconnectTestSvc",
            port=59997,
            connection_mode=RPCConnectionMode.RPC,
            connect_immediately=False,
        )


class _ScriptedProxy:
    """
    Stand-in for a multiprocessing BaseManager proxy. Every method access returns a callable that
    runs `behavior(method_name, args, kwargs)` — which either returns a value or raises, letting a
    test script transient failures / recovery.
    """

    def __init__(self, behavior):
        self.__dict__["_behavior"] = behavior

    def __getattr__(self, name):
        behavior = self.__dict__["_behavior"]

        def _method(*args, **kwargs):
            return behavior(name, args, kwargs)

        return _method


@pytest.fixture
def client(monkeypatch):
    # Never actually sleep during the self-heal cycle.
    monkeypatch.setattr(rcb.time, "sleep", lambda *_a, **_k: None)
    c = _HarnessClient()
    yield c
    # Cleanup: drop any sentinel cache-thread so disconnect() doesn't try to .join() a non-Thread.
    c._cache_refresh_thread = None
    if c in RPCClientBase._active_instances:
        c.disconnect()


def _seed_live_but_poisoned(client, behavior, generation=1):
    """Put the client in the 'connected, but the server just bounced under us' state."""
    client._proxy = _ScriptedProxy(behavior)
    client._connected = True
    client._connection_generation = generation


def test_self_heal_recovers_after_transient_readiness_race(client):
    """A just-restarted server accepts the connect before it can serve → the CYCLE must recover."""
    state = {"ready": False, "connects": 0}

    def behavior(name, args, kwargs):
        if not state["ready"]:
            raise BrokenPipeError(32, "Broken pipe")
        return "OK"

    _seed_live_but_poisoned(client, behavior)

    def fake_connect(max_retries=None, retry_delay=None):
        state["connects"] += 1
        # Model the readiness window: only serves from the 2nd reconnect onward.
        if state["connects"] >= 2:
            state["ready"] = True
        client._proxy = _ScriptedProxy(behavior)
        client._connected = True
        client._connection_generation += 1
        return True

    monkeypatch_connect(client, fake_connect)

    result = client._invoke_rpc("echo_rpc", ("hi",))

    assert result == "OK"
    assert state["connects"] >= 2  # a single reconnect+retry would NOT have recovered
    assert client._connected is True


def test_self_heal_via_server_wrapper(client):
    """The typed-wrapper path (self._server.foo_rpc()) must self-heal, not just call()."""
    state = {"ready": False, "connects": 0}

    def behavior(name, args, kwargs):
        if not state["ready"]:
            raise EOFError()
        return {"method": name, "args": args}

    _seed_live_but_poisoned(client, behavior)

    def fake_connect(max_retries=None, retry_delay=None):
        state["connects"] += 1
        state["ready"] = True
        client._proxy = _ScriptedProxy(behavior)
        client._connected = True
        client._connection_generation += 1
        return True

    monkeypatch_connect(client, fake_connect)

    # Exactly how a typed client method delegates.
    result = client._server.echo_rpc("x")

    assert result == {"method": "echo_rpc", "args": ("x",)}
    assert state["connects"] == 1


def test_retry_false_fails_fast_and_resets(client):
    """Fail-fast opt-out (health_check et al.): raise at once, but still drop the poisoned proxy."""
    state = {"connects": 0}

    def behavior(name, args, kwargs):
        raise BrokenPipeError(32, "Broken pipe")

    _seed_live_but_poisoned(client, behavior)
    monkeypatch_connect(client, lambda **_k: state.__setitem__("connects", state["connects"] + 1))

    with pytest.raises(BrokenPipeError):
        client._invoke_rpc("health_check_rpc", retry=False)

    assert state["connects"] == 0        # no reconnect cycle
    assert client._proxy is None         # poisoned connection dropped...
    assert client._connected is False    # ...so the NEXT call reconnects lazily


def test_business_error_not_retried_and_connection_kept(client):
    """A non-transient (business) error must not reconnect, retry, or drop the connection."""
    state = {"connects": 0, "calls": 0}

    def behavior(name, args, kwargs):
        state["calls"] += 1
        raise ValueError("business rejection")

    _seed_live_but_poisoned(client, behavior)
    monkeypatch_connect(client, lambda **_k: state.__setitem__("connects", state["connects"] + 1))

    with pytest.raises(ValueError):
        client._invoke_rpc("do_thing_rpc")

    assert state["calls"] == 1          # called exactly once
    assert state["connects"] == 0       # never reconnected
    assert client._connected is True    # connection NOT dropped for a business error


def test_self_heal_exhausts_then_raises_and_resets(client):
    """Server never comes back: exhaust the bounded cycles, reset, and surface the transient error."""
    state = {"connects": 0}

    def behavior(name, args, kwargs):
        raise BrokenPipeError(32, "Broken pipe")

    _seed_live_but_poisoned(client, behavior)

    def fake_connect(max_retries=None, retry_delay=None):
        state["connects"] += 1
        client._proxy = _ScriptedProxy(behavior)
        client._connected = True
        client._connection_generation += 1
        return True

    monkeypatch_connect(client, fake_connect)

    with pytest.raises(BrokenPipeError):
        client._invoke_rpc("echo_rpc")

    assert state["connects"] == client._RECONNECT_MAX_RETRIES  # one connect per cycle
    assert client._proxy is None        # final reset so a later call reconnects cleanly
    assert client._connected is False


def test_reset_connection_preserves_registration_and_cache_daemon(client):
    """_reset_connection drops only the transport; disconnect() is what unregisters."""
    assert client in RPCClientBase._active_instances

    sentinel_thread = object()
    client._cache_refresh_thread = sentinel_thread
    client._connected = True
    client._proxy = object()

    client._reset_connection()

    # Transport dropped...
    assert client._proxy is None
    assert client._connected is False
    # ...but the instance stays tracked and the cache-refresh daemon is left untouched.
    assert client in RPCClientBase._active_instances
    assert client._cache_refresh_thread is sentinel_thread

    # disconnect(), by contrast, DOES unregister (fixture resets the sentinel first).
    client._cache_refresh_thread = None
    client.disconnect()
    assert client not in RPCClientBase._active_instances


def test_server_raised_oserror_with_healthy_transport_not_retried(client):
    """A server-side OSError-family BUSINESS exception (e.g. position-lock TimeoutError) arrives
    type-identical to a dead socket — but the transport probe (health_check_rpc on the same
    connection) succeeds, so the client must re-raise WITHOUT reconnecting or re-executing."""
    state = {"connects": 0, "target_calls": 0}

    def behavior(name, args, kwargs):
        if name == "health_check_rpc":
            return {"status": "healthy"}  # transport is alive
        state["target_calls"] += 1
        raise TimeoutError("Failed to acquire lock after 10.0s")  # OSError subclass, server-raised

    _seed_live_but_poisoned(client, behavior)
    monkeypatch_connect(client, lambda **_k: state.__setitem__("connects", state["connects"] + 1))

    with pytest.raises(TimeoutError):
        client._invoke_rpc("execute_order_rpc")

    assert state["target_calls"] == 1   # NOT re-executed 5 more times
    assert state["connects"] == 0       # healthy connection never torn down
    assert client._connected is True


def test_circuit_breaker_fails_fast_after_exhaustion_then_closes_on_success(client):
    """After self-heal exhaustion the breaker opens: the next call gets ONE fast attempt (no
    multi-cycle self-heal). A later success closes the breaker."""
    state = {"connects": 0, "calls": 0, "ready": False}

    def behavior(name, args, kwargs):
        state["calls"] += 1
        if not state["ready"]:
            raise BrokenPipeError(32, "Broken pipe")
        return "OK"

    _seed_live_but_poisoned(client, behavior)

    def fake_connect(max_retries=None, retry_delay=None):
        state["connects"] += 1
        client._proxy = _ScriptedProxy(behavior)
        client._connected = True
        client._connection_generation += 1
        return True

    monkeypatch_connect(client, fake_connect)

    # Exhaust the self-heal — opens the breaker.
    with pytest.raises(BrokenPipeError):
        client._invoke_rpc("echo_rpc")
    assert client._backoff_until > 0
    connects_after_exhaustion = state["connects"]

    # Breaker open: the next call must fail fast — one reconnect at most, no settle cycles.
    with pytest.raises(BrokenPipeError):
        client._invoke_rpc("echo_rpc")
    assert state["connects"] - connects_after_exhaustion <= 1

    # Server recovers; a successful call closes the breaker.
    state["ready"] = True
    client._proxy = _ScriptedProxy(behavior)
    client._connected = True
    assert client._invoke_rpc("echo_rpc") == "OK"
    assert client._backoff_until == 0.0


def monkeypatch_connect(client, fn):
    """Replace the instance's connect with a plain function (no `self`), matching how
    _invoke_rpc calls self.connect(max_retries=..., retry_delay=...)."""
    client.connect = fn
