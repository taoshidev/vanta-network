"""
Pytest configuration for vali_tests.

This module provides session-scoped fixtures for managing ServerOrchestrator
lifecycle across all tests, ensuring clean shutdown to prevent CI hangs.
"""
import pytest
import bittensor as bt
from shared_objects.rpc.server_orchestrator import ServerOrchestrator


@pytest.fixture(scope="session", autouse=True)
def orchestrator_cleanup():
    """
    Session-scoped fixture that ensures ServerOrchestrator shuts down cleanly.

    This runs automatically after ALL tests in the session complete, ensuring:
    - All RPC servers are stopped
    - All client connections are closed
    - No hanging processes in CI environments

    The fixture uses yield to run cleanup code after all tests finish.
    """
    # Setup: Nothing needed here (tests create orchestrator as needed)
    yield

    # Teardown: Shut down all servers after ALL tests complete
    try:
        orchestrator = ServerOrchestrator.get_instance()
        orchestrator.shutdown_all_servers()
        bt.logging.info("Session cleanup: All servers shut down successfully")
    except Exception as e:
        # Use print as fallback since logging stream may be closed
        print(f"Session cleanup: Error during shutdown: {e}")
