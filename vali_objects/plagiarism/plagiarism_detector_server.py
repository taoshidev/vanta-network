# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
PlagiarismDetectorServer - RPC server for plagiarism detection.

This server runs in its own process and exposes plagiarism detection via RPC.
Clients connect using PlagiarismDetectorClient.
The server creates its own MetagraphClient internally (forward compatibility pattern).

Usage:
    # Validator spawns the server at startup
    from vali_objects.plagiarism.plagiarism_detector_server import PlagiarismDetectorServer

    server = PlagiarismDetectorServer(
        start_server=True,
        start_daemon=True
    )

    # Other processes connect via PlagiarismDetectorClient
    from vali_objects.plagiarism.plagiarism_detector_server import PlagiarismDetectorClient
    client = PlagiarismDetectorClient()  # Uses ValiConfig.RPC_PLAGIARISM_DETECTOR_PORT
"""
import os
import shutil
import time
from vali_objects.position_management.position_manager_client import PositionManagerClient

from typing import Dict, List

import bittensor as bt
from setproctitle import setproctitle

from shared_objects.cache_controller import CacheController
from shared_objects.rpc.rpc_server_base import RPCServerBase
from time_util.time_util import TimeUtil
from vali_objects.plagiarism.plagiarism_definitions import (
    FollowPercentage,
    LagDetection,
    CopySimilarity,
    TwoCopySimilarity,
    ThreeCopySimilarity
)
from vali_objects.plagiarism.plagiarism_pipeline import PlagiarismPipeline
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig


# ==================== Server Implementation ====================

class PlagiarismDetectorServer(RPCServerBase, CacheController):
    """
    RPC server for plagiarism detection.

    Inherits from:
    - RPCServerBase: Provides RPC server lifecycle, daemon management, watchdog
    - CacheController: Provides cache file management utilities

    All public methods ending in _rpc are exposed via RPC to PlagiarismDetectorClient.
    Internal state (plagiarism_data, plagiarism_raster, etc.) is kept local to this process.

    Architecture:
    - Runs in its own process (or thread in test mode)
    - Ports are obtained from ValiConfig
    """
    service_name = ValiConfig.RPC_PLAGIARISM_DETECTOR_SERVICE_NAME
    service_port = ValiConfig.RPC_PLAGIARISM_DETECTOR_PORT

    def __init__(
        self,
        running_unit_tests: bool = False,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = True
    ):
        """
        Initialize PlagiarismDetectorServer.

        Args:
            running_unit_tests: Whether running in test mode
            slack_notifier: Slack notifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon for periodic detection
        """
        # Initialize CacheController first (for cache file setup)
        CacheController.__init__(self, running_unit_tests=running_unit_tests)

        # Initialize RPCServerBase (handles RPC server and daemon lifecycle)
        # daemon_interval_s: 1 day (plagiarism detection is infrequent)
        # hang_timeout_s: Dynamically set to 2x interval to prevent false alarms during normal sleep
        daemon_interval_s = ValiConfig.PLAGIARISM_REFRESH_TIME_MS / 1000.0  # 1 day (86400s)
        hang_timeout_s = daemon_interval_s * 2.0  # 2 days (2x interval)

        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_PLAGIARISM_DETECTOR_SERVICE_NAME,
            port=ValiConfig.RPC_PLAGIARISM_DETECTOR_PORT,
            slack_notifier=None,
            start_server=start_server,
            start_daemon=False,  # Defer daemon start until all init complete
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s
        )

        # Local state (no IPC)
        self.plagiarism_data = {}
        self.plagiarism_raster = {}
        self.plagiarism_positions = {}
        self.plagiarism_classes = [
            FollowPercentage,
            LagDetection,
            CopySimilarity,
            TwoCopySimilarity,
            ThreeCopySimilarity
        ]

        # Create own PositionManagerClient (forward compatibility - no parameter passing)
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            connect_immediately=not running_unit_tests
        )

        self.plagiarism_pipeline = PlagiarismPipeline(self.plagiarism_classes)

        # Ensure plagiarism directories exist
        plagiarism_dir = ValiBkpUtils.get_plagiarism_dir(running_unit_tests=self.running_unit_tests)
        if not os.path.exists(plagiarism_dir):
            ValiBkpUtils.make_dir(ValiBkpUtils.get_plagiarism_dir(running_unit_tests=self.running_unit_tests))
            ValiBkpUtils.make_dir(ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests))

        bt.logging.success(f"PlagiarismDetectorServer initialized on port {ValiConfig.RPC_PLAGIARISM_DETECTOR_PORT}")

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.

        Runs plagiarism detection if refresh is allowed.
        """
        if self.refresh_allowed(ValiConfig.PLAGIARISM_REFRESH_TIME_MS):
            self.detect(hotkeys=self.metagraph.get_hotkeys())
            self.set_last_update_time(skip_message=False)

    @property
    def metagraph(self):
        """Get metagraph client (forward compatibility - created internally)."""
        return self._metagraph_client

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "num_plagiarism_data": len(self.plagiarism_data)
        }

    def get_plagiarism_scores_from_disk_rpc(self) -> Dict[str, float]:
        """
        Get plagiarism scores from disk.

        Returns:
            Dict mapping hotkeys to their plagiarism scores
        """
        return self.get_plagiarism_scores_from_disk()

    def get_plagiarism_data_from_disk_rpc(self) -> Dict[str, dict]:
        """
        Get detailed plagiarism data from disk.

        Returns:
            Dict mapping hotkeys to their full plagiarism data
        """
        return self.get_plagiarism_data_from_disk()

    def get_miner_plagiarism_data_from_disk_rpc(self, hotkey: str) -> dict:
        """
        Get plagiarism data for a specific miner from disk.

        Args:
            hotkey: Miner hotkey to look up

        Returns:
            Dict of plagiarism data for the miner, or empty dict if not found
        """
        return self.get_miner_plagiarism_data_from_disk(hotkey)

    def detect_rpc(self, hotkeys: List[str] = None, hotkey_positions: dict = None) -> None:
        """
        Run plagiarism detection via RPC.

        Args:
            hotkeys: List of hotkeys to analyze (optional)
            hotkey_positions: Pre-fetched positions (optional, for testing)
        """
        self.detect(hotkeys=hotkeys, hotkey_positions=hotkey_positions)

    def clear_plagiarism_from_disk_rpc(self, target_hotkey: str = None) -> None:
        """
        Clear plagiarism data from disk.

        Args:
            target_hotkey: Specific hotkey to clear, or None to clear all
        """
        self.clear_plagiarism_from_disk(target_hotkey=target_hotkey)

    # ==================== Internal Methods (business logic) ====================

    def detect(self, hotkeys: List[str] = None, hotkey_positions: dict = None) -> None:
        """
        Kick off the plagiarism detection process.

        Args:
            hotkeys: List of hotkeys to analyze
            hotkey_positions: Pre-fetched positions (optional)
        """
        if self.running_unit_tests:
            current_time = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        else:
            current_time = TimeUtil.now_in_millis()

        if hotkeys is None:
            hotkeys = self.metagraph.get_hotkeys()
            assert hotkeys, f"No hotkeys found in metagraph {self.metagraph}"

        if hotkey_positions is None:
            hotkey_positions = self._position_client.get_positions_for_hotkeys(
                hotkeys,
                filter_eliminations=True  # Automatically fetch and filter eliminations internally
            )

        bt.logging.info("Starting Plagiarism Detection")

        plagiarism_data, raster_positions, positions = self.plagiarism_pipeline.run_reporting(
            positions=hotkey_positions, current_time=current_time
        )

        self.write_plagiarism_scores_to_disk(plagiarism_data)
        self.write_plagiarism_raster_to_disk(raster_positions)
        self.write_plagiarism_positions_to_disk(positions)

        bt.logging.info("Plagiarism Detection Complete")

    def clear_plagiarism_from_disk(self, target_hotkey: str = None) -> None:
        """
        Clear all files and directories in the plagiarism scores directory.

        Args:
            target_hotkey: Specific hotkey to clear, or None to clear all
        """
        dir = ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests)
        for file in os.listdir(dir):
            if target_hotkey and file != target_hotkey:
                continue
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def write_plagiarism_scores_to_disk(self, plagiarism_data: list) -> None:
        """Write plagiarism scores to disk."""
        for plagiarist in plagiarism_data:
            self.write_plagiarism_score_to_disk(plagiarist["plagiarist"], plagiarist)

    def write_plagiarism_score_to_disk(self, hotkey: str, plagiarism_data: dict) -> None:
        """Write single plagiarism score to disk."""
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_plagiarism_score_file_location(
                hotkey=hotkey, running_unit_tests=self.running_unit_tests
            ),
            plagiarism_data
        )

    def write_plagiarism_raster_to_disk(self, raster_positions: dict) -> None:
        """Write raster positions to disk."""
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_plagiarism_raster_file_location(running_unit_tests=self.running_unit_tests),
            raster_positions
        )

    def write_plagiarism_positions_to_disk(self, plagiarism_positions: dict) -> None:
        """Write plagiarism positions to disk."""
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_plagiarism_positions_file_location(running_unit_tests=self.running_unit_tests),
            plagiarism_positions
        )

    def get_plagiarism_scores_from_disk(self) -> Dict[str, float]:
        """
        Get plagiarism scores from disk.

        Returns:
            Dict mapping hotkeys to their plagiarism scores
        """
        plagiarist_dir = ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(plagiarist_dir)

        # Retrieve hotkeys from plagiarism file names
        all_hotkeys = ValiBkpUtils.get_hotkeys_from_file_name(all_files)

        plagiarism_data = {
            hotkey: self.get_miner_plagiarism_data_from_disk(hotkey)
            for hotkey in all_hotkeys
        }
        plagiarism_scores = {}
        for hotkey in plagiarism_data:
            plagiarism_scores[hotkey] = plagiarism_data[hotkey].get("overall_score", 0)

        bt.logging.trace(f"Loaded [{len(plagiarism_scores)}] plagiarism scores from disk. Dir: {plagiarist_dir}")
        return plagiarism_scores

    def get_plagiarism_data_from_disk(self) -> Dict[str, dict]:
        """
        Get detailed plagiarism data from disk.

        Returns:
            Dict mapping hotkeys to their full plagiarism data
        """
        plagiarist_dir = ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(plagiarist_dir)

        # Retrieve hotkeys from plagiarism file names
        all_hotkeys = ValiBkpUtils.get_hotkeys_from_file_name(all_files)

        plagiarism_data = {
            hotkey: self.get_miner_plagiarism_data_from_disk(hotkey)
            for hotkey in all_hotkeys
        }

        bt.logging.trace(f"Loaded [{len(plagiarism_data)}] plagiarism scores from disk. Dir: {plagiarist_dir}")
        return plagiarism_data

    def get_miner_plagiarism_data_from_disk(self, hotkey: str) -> dict:
        """
        Get plagiarism data for a specific miner from disk.

        Args:
            hotkey: Miner hotkey to look up

        Returns:
            Dict of plagiarism data for the miner, or empty dict if not found
        """
        plagiarist_dir = ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests)
        file_path = os.path.join(plagiarist_dir, f"{hotkey}.json")

        if os.path.exists(file_path):
            data = ValiUtils.get_vali_json_file(file_path)
            return data
        else:
            return {}

    def _update_plagiarism_scores_in_memory(self) -> None:
        """Update plagiarism scores in memory from disk."""
        raster_positions_location = ValiBkpUtils.get_plagiarism_raster_file_location(
            running_unit_tests=self.running_unit_tests
        )
        self.plagiarism_raster = ValiUtils.get_vali_json_file(raster_positions_location)

        positions_location = ValiBkpUtils.get_plagiarism_positions_file_location(
            running_unit_tests=self.running_unit_tests
        )
        self.plagiarism_positions = ValiUtils.get_vali_json_file(positions_location)

        self.plagiarism_data = self.get_plagiarism_data_from_disk()


# ==================== Server Entry Point ====================

def start_plagiarism_detector_server(
    running_unit_tests: bool = False,
    server_ready=None
):
    """
    Entry point for server process.

    The server creates its own MetagraphClient internally (forward compatibility pattern).

    Args:
        running_unit_tests: Whether running in test mode
        server_ready: Event to signal when server is ready
    """
    from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator

    setproctitle("vali_PlagiarismDetectorServerProcess")

    # Create server with auto-start of RPC server and daemon
    # Server creates its own MetagraphClient internally
    server_instance = PlagiarismDetectorServer(
        running_unit_tests=running_unit_tests,
        start_server=True,
        start_daemon=True
    )

    if server_ready:
        server_ready.set()

    # Block until shutdown (RPCServerBase runs server in background thread)
    while not ShutdownCoordinator.is_shutdown():
        time.sleep(1)

    # Graceful shutdown
    server_instance.shutdown()
    bt.logging.info("PlagiarismDetectorServer process exiting")
