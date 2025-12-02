# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import os
import unittest

from shared_objects.port_manager import PortManager
from shared_objects.rpc_client_base import RPCClientBase
from shared_objects.rpc_server_base import RPCServerBase
from shared_objects.common_data_server import CommonDataClient


class TestBase(unittest.TestCase):

    def setUp(self) -> None:
        if "vm" in os.environ:
            del os.environ["vm"]
        # Aggressive cleanup before test starts - kill any stale processes on RPC ports
        PortManager.force_kill_all_rpc_ports()

    def tearDown(self) -> None:
        """Clean up all RPC clients and servers after each test to prevent port conflicts."""
        # Signal shutdown via CommonDataServer first (while servers still running)
        # Use max_retries=1 to avoid slow retries if server is already dead
        try:
            common_data_client = CommonDataClient(connect_immediately=False, max_retries=1)
            if common_data_client.connect():
                common_data_client.set_shutdown(True)
                common_data_client.disconnect()
        except Exception:
            pass  # Server may not be running, that's ok

        # Disconnect all clients (fast - no retries)
        RPCClientBase.disconnect_all()
        # Shutdown servers with force kill (no sleeps needed)
        RPCServerBase.shutdown_all(force_kill_ports=True)
