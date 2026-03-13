# developer: jbonilla
# Copyright (c) 2025 Taoshi Inc
"""
Entity Miner API Manager - Lifecycle manager for the Entity Miner Gateway.

Same pattern as MinerAPIManager: creates EntityMinerRestServer, runs forever,
handles shutdown.
"""
import json
import os
import time
import bittensor as bt

from miner_config import MinerConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vanta_api.entity_miner_rest_server import EntityMinerRestServer


class EntityMinerAPIManager:
    """Manages entity miner gateway server lifecycle."""

    def __init__(self, api_host="0.0.0.0", api_port=8089, slack_notifier=None,
                 prop_net_order_placer=None):
        """
        Initialize entity miner API manager.

        Args:
            api_host: REST server host address
            api_port: REST server port
            slack_notifier: Optional SlackNotifier for notifications
            prop_net_order_placer: Optional PropNetOrderPlacer for order submission endpoints
        """
        self.api_host = api_host
        self.api_port = api_port
        self.slack_notifier = slack_notifier
        self.prop_net_order_placer = prop_net_order_placer

        # Get API keys file path (reuse miner's API keys)
        self.api_keys_file = ValiBkpUtils.get_api_keys_file_path()

        if not os.path.exists(self.api_keys_file):
            print(f"WARNING: API keys file '{self.api_keys_file}' not found!")
        else:
            try:
                with open(self.api_keys_file, "r") as f:
                    keys = json.load(f)
                print(f"Entity gateway API keys file contains {len(keys)} keys")
            except Exception as e:
                print(f"ERROR reading API keys file: {e}")

        # Server instance (created in run())
        self.rest_server = None

    def run(self):
        """
        Main entry point - creates REST server and keeps alive.
        This method blocks until KeyboardInterrupt.
        """
        bt.logging.info("Starting Entity Miner Gateway server...")

        try:
            self.rest_server = EntityMinerRestServer(
                api_keys_file=self.api_keys_file,
                flask_host=self.api_host,
                flask_port=self.api_port,
                slack_notifier=self.slack_notifier,
                prop_net_order_placer=self.prop_net_order_placer
            )

            bt.logging.success(
                f"Entity Miner Gateway started at http://{self.api_host}:{self.api_port}"
            )
            bt.logging.info(f"Endpoints available:")
            bt.logging.info(f"  POST   /api/submit-order         - Synchronous order submission (inherited)")
            bt.logging.info(f"  GET    /api/order-status/<uuid>   - Query order status (inherited)")
            bt.logging.info(f"  GET    /api/hl/<addr>/dashboard   - Cached dashboard")
            bt.logging.info(f"  GET    /api/hl/<addr>/events      - Order events")
            bt.logging.info(f"  GET    /api/hl/<addr>/stream      - SSE stream")
            bt.logging.info(f"  POST   /api/create-subaccount     - Create subaccount")
            bt.logging.info(f"  POST   /api/create-hl-subaccount  - Create HL subaccount")
            bt.logging.info(f"  GET    /api/health                - Health check")

        except Exception as e:
            bt.logging.error(f"Failed to start Entity Miner Gateway: {e}")
            raise

        # Keep alive
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            bt.logging.info("Shutting down Entity Miner Gateway...")
            self.shutdown()

    def shutdown(self):
        """Gracefully shutdown the gateway server."""
        if self.rest_server:
            bt.logging.info("Stopping Entity Miner Gateway...")
            self.rest_server.shutdown()
            bt.logging.info("Entity Miner Gateway stopped")
