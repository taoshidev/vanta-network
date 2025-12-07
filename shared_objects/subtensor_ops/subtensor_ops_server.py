# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc

"""
RPC Server wrapper for SubtensorOpsManager.

This wrapper integrates SubtensorOpsManager into the ServerOrchestrator lifecycle
while maintaining its unique execution model (runs in main validator process).
"""

import time
import bittensor as bt

from shared_objects.rpc.rpc_server_base import RPCServerBase
from shared_objects.subtensor_ops.subtensor_ops import SubtensorOpsManager
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class SubtensorOpsServer(RPCServerBase):
    """
    RPC Server wrapper for SubtensorOpsManager.

    Runs in LOCAL mode (no RPCServerBase RPC) as a managed component within
    the main validator process. Integrates with ServerOrchestrator lifecycle.

    SubtensorOpsManager maintains its own RPC server (port 50001) for
    WeightCalculatorClient and other remote callers.

    NOTE: Instantiated directly by orchestrator (not spawned as process).
    """

    service_name = "subtensor_ops"
    service_port = 0  # No RPCServerBase port (LOCAL mode)

    def __init__(
        self,
        config,
        wallet,
        is_miner=False,
        slack_notifier=None,
        running_unit_tests=False,
        start_server=False,
        start_daemon=False,
        **kwargs
    ):
        # Daemon interval based on node type
        daemon_interval_s = (
            ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_MINER_MS / 1000.0
            if is_miner else
            ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_VALIDATOR_MS / 1000.0
        )

        super().__init__(
            service_name=self.service_name,
            port=0,
            slack_notifier=slack_notifier,
            connection_mode=RPCConnectionMode.LOCAL, # Must always be in the same process as the axon
            start_server=False,
            start_daemon=start_daemon,
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=300.0,
            **kwargs
        )

        self.config = config
        self.wallet = wallet
        self.is_miner = is_miner

        # Create SubtensorOpsManager (manages own RPC server on port 50001)
        self.manager = SubtensorOpsManager(
            config=config,
            hotkey=wallet.hotkey.ss58_address,
            is_miner=is_miner,
            slack_notifier=slack_notifier,
            running_unit_tests=running_unit_tests
        )

    def run_daemon_iteration(self) -> None:
        """Single metagraph update iteration."""
        if self._is_shutdown():
            return
        self.manager.update_metagraph()

    def get_daemon_name(self) -> str:
        mode = "miner" if self.is_miner else "validator"
        return f"vali_SubtensorOpsDaemon_{mode}"

    # Public API
    def get_subtensor(self):
        return self.manager.get_subtensor()

    def get_metagraph_client(self):
        return self.manager.get_metagraph()

    def is_hotkey_registered_cached(self, hotkey: str) -> bool:
        return self.manager.is_hotkey_registered_cached(hotkey)

    def wait_for_initial_update(self, max_wait_time=60):
        """
        Block until metagraph populates (critical for dependent servers).
        Must be called after daemon starts.
        """
        bt.logging.info("Waiting for initial metagraph population...")
        start_time = time.time()

        while (not self.manager._metagraph_client.get_hotkeys() and
               (time.time() - start_time) < max_wait_time):
            if self._is_shutdown():
                raise RuntimeError("Shutdown during metagraph wait")
            time.sleep(1)

        if not self.manager._metagraph_client.get_hotkeys():
            error_msg = f"Failed to populate metagraph within {max_wait_time}s"
            bt.logging.error(error_msg)
            if self.slack_notifier:
                self.slack_notifier.send_message(f"❌ {error_msg}", level="error")
            raise RuntimeError(error_msg)

        bt.logging.success(
            f"Metagraph populated with {len(self.manager._metagraph_client.get_hotkeys())} hotkeys"
        )
