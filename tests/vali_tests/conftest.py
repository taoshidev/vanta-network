"""
Pytest configuration for vali_tests.

This module provides session-scoped fixtures for managing ServerOrchestrator
lifecycle across all tests, ensuring:
- Fast test startup (servers pre-started before any test runs)
- Clean shutdown to prevent CI hangs
"""
import os
import sys
import time
import pytest
import bittensor as bt
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from vali_objects.utils.vali_utils import ValiUtils


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """
    Force a clean, immediate process exit once results are reported.

    The validator suite leaves lingering non-daemon threads (e.g. HealthMonitor
    threads) and/or un-reaped RPC server child processes alive after the session
    fixture teardown runs. The Python interpreter then blocks at shutdown trying
    to join them, so pytest prints its summary but the process never returns and
    CI hangs until the job wall-clock timeout.

    This hook is registered trylast, so it runs AFTER the terminal reporter has
    printed (and flushed) the summary line. At that point pytest's result is
    final, so we exit hard with the real status code -- CI still sees the correct
    pass/fail, but we skip the interpreter-shutdown join that would otherwise hang.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exitstatus))


@pytest.fixture(scope="session", autouse=True)
def orchestrator_lifecycle():
    """
    Session-scoped fixture that manages ServerOrchestrator lifecycle.

    SETUP (before any tests):
    - Pre-starts all RPC servers ONCE at session start
    - Eliminates "slow first test" problem in CI
    - All test classes get already-running servers

    TEARDOWN (after all tests):
    - Shuts down all RPC servers
    - Closes all client connections
    - Prevents hanging processes in CI

    This dramatically speeds up CI by:
    1. Starting servers once before test collection completes
    2. Avoiding per-test-class server startup costs
    """
    # SETUP: Pre-start all servers before any test runs
    start_time = time.time()
    print("\n[conftest] Pre-starting all RPC servers for test session...")

    try:
        orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )
        elapsed = time.time() - start_time
        print(f"[conftest] All servers started in {elapsed:.1f}s")
        bt.logging.info(f"Session setup: All servers started in {elapsed:.1f}s")
    except Exception as e:
        print(f"[conftest] ERROR starting servers: {e}")
        raise

    yield orchestrator

    # TEARDOWN: Shut down all servers after ALL tests complete
    try:
        orchestrator = ServerOrchestrator.get_instance()
        orchestrator.shutdown_all_servers()
        bt.logging.info("Session cleanup: All servers shut down successfully")
    except Exception as e:
        # Use print as fallback since logging stream may be closed
        print(f"Session cleanup: Error during shutdown: {e}")
