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


# Final exit status, recorded at session finish and used by the forced exit below.
_session_exit_code = {"value": 0}


def pytest_sessionfinish(session, exitstatus):
    """Record the final status code for the forced exit in pytest_unconfigure."""
    _session_exit_code["value"] = int(exitstatus)


def pytest_unconfigure(config):
    """
    Force a clean, immediate process exit AFTER results are fully reported.

    Why this is needed: the validator suite leaves RPC server child processes and
    non-daemon threads (e.g. HealthMonitor) alive. At interpreter shutdown Python
    blocks joining the threads, and -- more importantly in CI -- the live child
    processes keep the step's stdout pipe open, so the runner never receives EOF
    and the job hangs to its wall-clock timeout even after pytest is done.

    Why pytest_unconfigure (not pytest_sessionfinish): the terminal reporter prints
    its "N passed / N failed" summary in the post-yield half of its sessionfinish
    hookwrapper, which runs after all plain sessionfinish hooks. Exiting from
    sessionfinish therefore swallows the summary. pytest_unconfigure runs later,
    after the summary is written, so results are preserved.

    We kill the multiprocessing children (releasing the stdout pipe) and os._exit
    with the real status code (skipping the thread-join), so CI still reports the
    correct pass/fail but the job actually terminates.
    """
    import multiprocessing
    sys.stdout.flush()
    sys.stderr.flush()
    for child in multiprocessing.active_children():
        try:
            child.kill()
        except Exception:
            pass
    os._exit(_session_exit_code["value"])


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
