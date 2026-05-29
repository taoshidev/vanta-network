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
import signal
import threading
import faulthandler
import pytest
import bittensor as bt
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from vali_objects.utils.vali_utils import ValiUtils


# ---------------------------------------------------------------------------
# Hang watchdog
#
# pytest-timeout's `signal` method cannot interrupt this suite's hangs: they
# occur in C-level blocking calls (RPC socket recv, multiprocessing joins, lock
# acquires) that never return to the interpreter, so SIGALRM is never handled
# and the per-test timeout silently does nothing -- the job then runs to the CI
# wall clock with no diagnostics.
#
# This watchdog is a daemon THREAD, so it does not depend on the main thread
# being interruptible. If any single test (including its setup/teardown) exceeds
# WATCHDOG_PER_TEST_S, it dumps the stacks of all threads (so we can see what is
# stuck) and then SIGKILLs the whole process group. Killing the group takes out
# the RPC server child processes too, which closes the CI stdout pipe so the
# runner receives EOF and the job actually ends.
#
# The cap is set just above pytest-timeout's signal timeout (pytest.ini) so the
# (cheaper, non-fatal) signal timeout gets first chance to fail an *interruptible*
# hang and let the suite continue; the watchdog is the hard backstop for the
# uninterruptible ones. No single test should legitimately take this long.
# ---------------------------------------------------------------------------
WATCHDOG_PER_TEST_S = 120

_current_test = {"id": None, "start": None}
_watchdog_started = False


def pytest_runtest_logstart(nodeid, location):
    _current_test["id"] = nodeid
    _current_test["start"] = time.time()


def pytest_runtest_logfinish(nodeid, location):
    _current_test["start"] = None


def _watchdog_loop():
    while True:
        time.sleep(5)
        start = _current_test["start"]
        if start is None:
            continue
        elapsed = time.time() - start
        if elapsed > WATCHDOG_PER_TEST_S:
            sys.stderr.write(
                f"\n[watchdog] Test '{_current_test['id']}' exceeded "
                f"{WATCHDOG_PER_TEST_S}s ({elapsed:.0f}s elapsed). Dumping all "
                f"thread stacks and killing the process group.\n"
            )
            sys.stderr.flush()
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            sys.stderr.flush()
            sys.stdout.flush()
            try:
                # SIGKILL the whole group: pytest + RPC server children + pool
                # workers. Closes the CI stdout pipe so the runner stops waiting.
                os.killpg(os.getpgid(0), signal.SIGKILL)
            except Exception:
                os._exit(1)


def _start_watchdog():
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True
    threading.Thread(target=_watchdog_loop, name="CIHangWatchdog", daemon=True).start()


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
    # Start the hang watchdog before anything else can block.
    _start_watchdog()

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
