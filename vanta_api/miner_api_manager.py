# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
Miner API Manager - Simple orchestrator for miner REST server.

This module manages the lifecycle of the miner REST server:
- Creates MinerRestServer in-process with direct PropNetOrderPlacer reference
- Keeps main thread alive for server to handle requests
- Handles graceful shutdown

Much simpler than validator's APIManager:
- No spawn_process() (in-process instead of separate process)
- No health monitoring (no RPC)
- Single REST server (no WebSocket)
"""
import json
import os
import time
import bittensor as bt

from miner_config import MinerConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vanta_api.miner_rest_server import MinerRestServer


class MinerAPIManager:
    """Manages miner REST API server lifecycle."""

    def __init__(self, prop_net_order_placer, miner_hotkey=None,
                 refresh_interval=15, api_host="0.0.0.0", api_rest_port=8088,
                 slack_notifier=None):
        """
        Initialize miner API manager.

        Args:
            prop_net_order_placer: Direct reference to miner's PropNetOrderPlacer
            miner_hotkey: Miner hotkey for identification in logs
            refresh_interval: How often to check for API key changes (seconds)
            api_host: REST server host address
            api_rest_port: REST server port
            slack_notifier: Optional SlackNotifier for notifications
        """
        self.prop_net_order_placer = prop_net_order_placer
        self.miner_hotkey = miner_hotkey
        self.api_host = api_host
        self.api_rest_port = api_rest_port
        self.refresh_interval = refresh_interval
        self.slack_notifier = slack_notifier

        # Get default API keys file path
        self.api_keys_file = ValiBkpUtils.get_api_keys_file_path()

        # Verify API keys file exists
        if not os.path.exists(self.api_keys_file):
            print(f"WARNING: API keys file '{self.api_keys_file}' not found!")
        else:
            print(f"API keys file found at: {self.api_keys_file}")
            # Check if it's a valid JSON file
            try:
                with open(self.api_keys_file, "r") as f:
                    keys = json.load(f)
                print(f"API keys file contains {len(keys)} keys")
            except Exception as e:
                print(f"ERROR reading API keys file: {e}")

        # REST server instance (created in run())
        self.rest_server = None

    def run(self):
        """
        Main entry point - creates REST server and keeps alive.

        REST server runs in-process (Flask in background thread).
        This method blocks until KeyboardInterrupt.
        """
        bt.logging.info("Starting Miner REST API server...")

        # Create REST server in-process with direct PropNetOrderPlacer reference
        try:
            self.rest_server = MinerRestServer(
                prop_net_order_placer=self.prop_net_order_placer,
                api_keys_file=self.api_keys_file,
                refresh_interval=self.refresh_interval,
                flask_host=self.api_host,
                flask_port=self.api_rest_port,
                slack_notifier=self.slack_notifier
            )
            # BaseRestServer.__init__ already started Flask in background thread

            bt.logging.success(
                f"Miner REST API server started at http://{self.api_host}:{self.api_rest_port}"
            )
            bt.logging.info(f"Endpoints available:")
            bt.logging.info(f"  POST   /api/submit-order      - Synchronous order submission")
            bt.logging.info(f"  POST   /api/create-subaccount - Entity subaccount creation")
            bt.logging.info(f"  GET    /api/order-status/<uuid> - Query order status")
            bt.logging.info(f"  GET    /api/health           - Health check")

        except Exception as e:
            bt.logging.error(f"Failed to start Miner REST API server: {e}")
            raise

        # Keep main thread alive
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            bt.logging.info("Shutting down Miner API server...")
            self.shutdown()

    def shutdown(self):
        """Gracefully shutdown REST server."""
        if self.rest_server:
            bt.logging.info("Stopping REST server...")
            self.rest_server.shutdown()
            bt.logging.info("REST server stopped")
