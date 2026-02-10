import hashlib
import statistics
from string import hexdigits

import bittensor as bt
import threading
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, Optional

from flask import Flask, jsonify, request, Response, g
import os
import time
import json
import gzip
import traceback
from setproctitle import setproctitle
from waitress import serve
from bittensor_wallet import Keypair

from time_util.time_util import TimeUtil
from vali_objects.utils.limit_order.market_order_manager import MarketOrderManager
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.limit_order.order_processor import OrderProcessor
from multiprocessing import current_process
from vanta_api.api_key_refresh import APIKeyMixin
from vanta_api.nonce_manager import NonceManager
from vanta_api.base_rest_server import BaseRestServer
from shared_objects.rpc.rpc_server_base import RPCServerBase
from entity_management.entity_client import EntityClient


class ValidatorRestServer(BaseRestServer, RPCServerBase):
    """Handles REST API requests with Flask and Waitress.

    Multiple inheritance:
    - BaseRestServer: Provides Flask app, metrics tracking, error handlers
    - RPCServerBase: Provides RPC server lifecycle management for health checks/control

    The server runs TWO servers:
    - Flask HTTP server on port 48888 (REST API) - from BaseRestServer
    - RPC server on port 50022 (health checks, control, monitoring) - from RPCServerBase
    """

    service_name = ValiConfig.RPC_REST_SERVER_SERVICE_NAME
    service_port = ValiConfig.RPC_REST_SERVER_PORT

    def __init__(self, api_keys_file, shared_queue=None, refresh_interval=15,
                 metrics_interval_minutes=5, running_unit_tests=False,
                 connection_mode:RPCConnectionMode = RPCConnectionMode.RPC,
                 start_server=True, flask_host=None, flask_port=None, **kwargs):
        """Initialize the REST server with API key handling and routing.

        Uses multiple inheritance pattern:
        - BaseRestServer handles Flask app, metrics, error handlers, API keys
        - RPCServerBase handles RPC health monitoring server

        Note: Creates own clients internally via _initialize_clients():
        - PositionManagerClient
        - AssetSelectionClient
        - LimitOrderClient
        - ContractClient
        - CoreOutputsClient
        - StatisticsOutputsClient
        - DebtLedgerClient
        - PerfLedgerClient
        - EntityClient

        The server runs on configurable endpoints (defaults from ValiConfig):
        - Flask HTTP: flask_host:flask_port (default: ValiConfig.REST_API_HOST:REST_API_PORT)
        - RPC health: ValiConfig.RPC_REST_SERVER_PORT (50022)

        Args:
            api_keys_file: Path to the JSON file containing API keys
            shared_queue: Optional shared queue for communication with WebSocket server
            refresh_interval: How often to check for API key changes (seconds)
            metrics_interval_minutes: How often to log API metrics (minutes)
            running_unit_tests: Whether running in unit test mode
            connection_mode: RPC or LOCAL mode
            start_server: Whether to start servers immediately
            flask_host: Host for Flask HTTP server
            flask_port: Port for Flask HTTP server
        """
        # Store validator-specific config before initializing base classes
        self.running_unit_tests = running_unit_tests
        self.shared_queue = shared_queue
        self.nonce_manager = NonceManager()
        self.market_order_manager = MarketOrderManager(serve=False)
        self.data_path = ValiConfig.BASE_DIR

        # Store connection_mode for use in _initialize_clients
        self._connection_mode = connection_mode

        print(f"[REST-INIT] Initializing VantaRestServer with multiple inheritance...")

        # Initialize BaseRestServer first (Flask, metrics, error handlers)
        # This will call _initialize_clients() and _register_routes()
        print(f"[REST-INIT] Step 1/2: Initializing BaseRestServer (Flask)...")
        BaseRestServer.__init__(
            self,
            api_keys_file=api_keys_file,
            service_name=self.service_name,
            refresh_interval=refresh_interval,
            metrics_interval_minutes=metrics_interval_minutes,
            flask_host=flask_host if flask_host is not None else ValiConfig.REST_API_HOST,
            flask_port=flask_port if flask_port is not None else ValiConfig.REST_API_PORT,
            # Pass connection_mode and running_unit_tests to _initialize_clients via kwargs
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests
        )
        print(f"[REST-INIT] Step 1/2: BaseRestServer initialized ✓")

        # Initialize RPCServerBase (health monitoring)
        print(f"[REST-INIT] Step 2/2: Initializing RPCServerBase (health monitoring)...")
        RPCServerBase.__init__(
            self,
            service_name=self.service_name,
            port=self.service_port,
            connection_mode=connection_mode,
            start_server=start_server,
            start_daemon=False,  # Flask runs in background thread, no daemon needed
            **kwargs
        )
        print(f"[REST-INIT] Step 2/2: RPCServerBase initialized on port {self.service_port} ✓")

        print(f"[{current_process().name}] VantaRestServer initialized with {len(self.accessible_api_keys)} API keys")
        print(f"[{current_process().name}] Flask HTTP server running on {self.flask_host}:{self.flask_port}")
        print(f"[{current_process().name}] RPC health server running on port {self.service_port}")

    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (from BaseRestServer)
    # ============================================================================

    def _initialize_clients(self, connection_mode=RPCConnectionMode.RPC, running_unit_tests=False, **kwargs):
        """
        Initialize 9 RPC clients for validator data access.

        Called by BaseRestServer.__init__() after Flask app is created but before routes are registered.

        Args:
            connection_mode: RPC or LOCAL mode for client connections
            running_unit_tests: Whether running in unit test mode
        """
        print(f"[REST-INIT] Step 2/9: Creating PositionManagerClient...")
        # Create own PositionManagerClient (forward compatibility - no parameter passing)
        from vali_objects.position_management.position_manager_client import PositionManagerClient
        self._position_client = PositionManagerClient(connection_mode=connection_mode)
        self._debt_ledger_client = DebtLedgerClient(connection_mode=connection_mode)
        self._perf_ledger_client = PerfLedgerClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2/9: PositionManagerClient created ✓")

        print(f"[REST-INIT] Step 2b/9: Creating AssetSelectionClient...")
        # Create own AssetSelectionClient (forward compatibility - no parameter passing)
        from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
        self._asset_selection_client = AssetSelectionClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2b/9: AssetSelectionClient created ✓")

        print(f"[REST-INIT] Step 2c/9: Creating LimitOrderClient...")
        # Create own LimitOrderClient (forward compatibility - no parameter passing)
        from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
        self._limit_order_client = LimitOrderClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2c/9: LimitOrderClient created ✓")

        print(f"[REST-INIT] Step 2d/9: Creating ContractClient...")
        # Create own ContractClient (forward compatibility - no parameter passing)
        from vali_objects.contract.contract_client import ContractClient
        self._contract_client = ContractClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2d/9: ContractClient created ✓")

        print(f"[REST-INIT] Step 2d2/9: Creating MinerAccountClient...")
        # Create own MinerAccountClient (for account sizes data)
        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2d2/9: MinerAccountClient created ✓")

        print(f"[REST-INIT] Step 2e/9: Creating CoreOutputsClient...")
        # Create own CoreOutputsClient (forward compatibility - no parameter passing)
        from vali_objects.data_export.core_outputs_client import CoreOutputsClient
        self._core_outputs_client = CoreOutputsClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2e/9: CoreOutputsClient created ✓")

        print(f"[REST-INIT] Step 2f/9: Creating StatisticsOutputsClient...")
        # Create own StatisticsOutputsClient (forward compatibility - no parameter passing)
        from vali_objects.statistics.miner_statistics_client import MinerStatisticsClient
        self._statistics_outputs_client = MinerStatisticsClient(connection_mode=connection_mode)
        print(f"[REST-INIT] Step 2f/9: StatisticsOutputsClient created ✓")

        print(f"[REST-INIT] Step 2g/9: Creating EntityClient...")
        # Create own EntityClient (forward compatibility - no parameter passing)
        self._entity_client = EntityClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        print(f"[REST-INIT] Step 2g/9: EntityClient created ✓")

    # ============================================================================
    # LIFECYCLE MANAGEMENT (multiple inheritance coordination)
    # ============================================================================

    def shutdown(self):
        """
        Override shutdown to coordinate both parent classes.

        Shuts down both Flask server (BaseRestServer) and RPC server (RPCServerBase).
        """
        bt.logging.info(f"{self.service_name} shutting down...")
        # Stop Flask server (from BaseRestServer)
        BaseRestServer.shutdown(self)
        # Stop RPC server and daemon (from RPCServerBase)
        RPCServerBase.shutdown(self)
        bt.logging.info(f"{self.service_name} shutdown complete")

    # ============================================================================
    # POSITION MANAGER ACCESS (forward compatibility - creates own client)
    # ============================================================================

    @property
    def position_manager(self):
        """Get position manager client."""
        return self._position_client

    @property
    def contract_manager(self):
        """Get contract client (forward compatibility - created internally)."""
        return self._contract_client

    # ============================================================================
    # RPCServerBase REQUIRED METHODS
    # ============================================================================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work.

        Note: PTNRestServer doesn't need a daemon loop - all work is done
        in Flask request handlers. This is a no-op.
        """
        pass

    def _jsonify_with_custom_encoder(self, data, status_code=200):
        """
        Create a JSON response using CustomEncoder to handle BaseModel objects.

        Args:
            data: The data to jsonify
            status_code: HTTP status code (default 200)

        Returns:
            Flask Response object with proper JSON serialization
        """
        json_str = json.dumps(data, cls=CustomEncoder)
        response = Response(json_str, content_type='application/json')
        response.status_code = status_code
        return response

    def _register_routes(self):
        """Register all API routes."""
        print(f"[REST-INIT] Registering validator endpoints...")

        # Miner position endpoints
        self.app.route("/miner-positions", methods=["GET"])(self.get_miner_positions)
        self.app.route("/miner-positions/<minerid>", methods=["GET"])(self.get_miner_positions_single)
        self.app.route("/miner-hotkeys", methods=["GET"])(self.get_miner_hotkeys)

        # Ledger endpoints
        self.app.route("/emissions-ledger/<minerid>", methods=["GET"])(self.get_emissions_ledger)
        self.app.route("/debt-ledger/<minerid>", methods=["GET"])(self.get_miner_debt_ledger)
        self.app.route("/perf-ledger/<minerid>", methods=["GET"])(self.get_perf_ledger)
        self.app.route("/debt-ledger", methods=["GET"])(self.get_debt_ledger)
        self.app.route("/penalty-ledger/<minerid>", methods=["GET"])(self.get_penalty_ledger)

        # Statistics endpoints
        self.app.route("/validator-checkpoint", methods=["GET"])(self.get_validator_checkpoint)
        self.app.route("/statistics", methods=["GET"])(self.get_validator_checkpoint_statistics)
        self.app.route("/statistics/<minerid>/", methods=["GET"])(self.get_validator_checkpoint_statistics_unique)
        self.app.route("/eliminations", methods=["GET"])(self.get_eliminations)

        # Trading endpoints
        self.app.route("/limit-orders/<minerid>", methods=["GET"])(self.get_limit_orders_unique)
        self.app.route("/orders/<minerid>", methods=["GET"])(self.get_orders_for_miner)
        self.app.route("/asset-selection", methods=["POST"])(self.asset_selection)
        self.app.route("/miner-selections", methods=["GET"])(self.get_miner_selections)
        self.app.route("/development/order", methods=["POST"])(self.process_development_order)

        # Collateral endpoints
        self.app.route("/collateral/deposit", methods=["POST"])(self.deposit_collateral)
        self.app.route("/collateral/query-withdraw", methods=["POST"])(self.query_withdraw_collateral)
        self.app.route("/collateral/withdraw", methods=["POST"])(self.withdraw_collateral)
        self.app.route("/collateral/", methods=["GET"])(self.get_all_collateral_data)
        self.app.route("/collateral/balance/<miner_address>", methods=["GET"])(self.get_collateral_balance)

        # Entity management endpoints
        self.app.route("/entity/register", methods=["POST"])(self.register_entity)
        self.app.route("/entity/create-subaccount", methods=["POST"])(self.create_subaccount)
        self.app.route("/entity/<entity_hotkey>", methods=["GET"])(self.get_entity)
        self.app.route("/entities", methods=["GET"])(self.get_all_entities)
        self.app.route("/entity/subaccount/eliminate", methods=["POST"])(self.eliminate_subaccount)
        self.app.route("/entity/subaccount/<synthetic_hotkey>", methods=["GET"])(self.get_subaccount_dashboard)
        self.app.route("/entity/subaccount/payout", methods=["POST"])(self.calculate_subaccount_payout)

        print(f"[REST-INIT] 29 validator endpoints registered ✓")

    # ============================================================================
    # MINER POSITION ENDPOINTS
    # ============================================================================

    def get_miner_positions(self):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Get the 'tier' query parameter from the request
        requested_tier = str(request.args.get('tier', 100))
        is_gz_data = True

        # Validate the 'tier' parameter
        if requested_tier not in ['0', '30', '50', '100']:
            return jsonify({'error': 'Invalid tier value. Allowed values are 0, 30, 50, or 100'}), 400

        # Check if API key has sufficient tier access
        if not self.can_access_tier(api_key, int(requested_tier)):
            return jsonify({'error': f'Your API key does not have access to tier {requested_tier} data'}), 403

        f = ValiBkpUtils.get_miner_positions_output_path(suffix_dir=requested_tier)

        # Attempt to retrieve the file
        data = self._get_file(f, binary=is_gz_data)

        if data is None:
            return jsonify({'error': 'Data not found'}), 404
        return Response(data, content_type='application/json', headers={
            'Content-Encoding': 'gzip'
        })

    def get_miner_positions_single(self, minerid):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Use the API key's tier for access
        api_key_tier = self.get_api_key_tier(api_key)
        if self.can_access_tier(api_key, 100) and self.position_manager:
            existing_positions: list[Position] = self.position_manager.get_positions_for_one_hotkey(minerid,
                                                                                                    sort_positions=True)
            if not existing_positions:
                return jsonify({'error': f'Miner ID {minerid} not found', 'positions':[]}), 404
            filtered_data = self._position_client.positions_to_dashboard_dict(existing_positions,
                                                                              TimeUtil.now_in_millis())
        else:
            requested_tier = str(api_key_tier)
            f = ValiBkpUtils.get_miner_positions_output_path(suffix_dir=requested_tier)
            data = self._get_file(f)

            if data is None:
                return jsonify({'error': 'Data not found'}), 404
            # Filter the data for the specified miner ID
            filtered_data = data.get(minerid, None)

        if not filtered_data:
            return jsonify({'error': f'Miner ID {minerid} not found', 'positions':[]}), 404

        return jsonify(filtered_data)

    def get_miner_hotkeys(self):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        if self.position_manager:
            # Use the position manager to get miner hotkeys
            miner_hotkeys = list(self.position_manager.get_miner_hotkeys_with_at_least_one_position())
        else:
            f = ValiBkpUtils.get_miner_positions_output_path()
            data = self._get_file(f)

            if data is None:
                return jsonify({'error': 'Data not found'}), 404

            miner_hotkeys = list(data.keys())

        if len(miner_hotkeys) == 0:
            return jsonify({'error': 'No miner hotkeys found'}), 404
        else:
            return jsonify(miner_hotkeys)

    # ============================================================================
    # LEDGER ENDPOINTS
    # ============================================================================

    def get_emissions_ledger(self, minerid):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Use RPC getter to access emissions ledger via debt ledger manager
        data = self._debt_ledger_client.get_emissions_ledger(minerid)

        if data is None:
            return jsonify({'error': 'Emissions ledger data not found'}), 404
        else:
            return self._jsonify_with_custom_encoder(data)

    def get_miner_debt_ledger(self, minerid):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        data = self._debt_ledger_client.get_ledger(minerid)

        if data is None:
            return jsonify({'error': 'Debt ledger data not found'}), 404
        else:
            return self._jsonify_with_custom_encoder(data)

    def get_perf_ledger(self, minerid):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check if perf ledger client is available
        if not self._perf_ledger_client:
            return jsonify({'error': 'Perf ledger data not available'}), 503

        try:
            # Use dedicated RPC method to get only this miner's ledger (efficient - no bulk transfer)
            data = self._perf_ledger_client.get_perf_ledger_for_hotkey(minerid)

            if data is None:
                return jsonify({'error': f'Perf ledger data not found for miner {minerid}'}), 404

            return self._jsonify_with_custom_encoder(data)

        except Exception as e:
            bt.logging.error(f"Error retrieving perf ledger for {minerid}: {e}")
            return jsonify({'error': 'Internal server error retrieving perf ledger data'}), 500

    def get_debt_ledger(self):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check if debt ledger manager is available
        if not self._debt_ledger_client:
            return jsonify({'error': 'Debt ledger data not available'}), 503

        try:
            # Get compressed summaries directly from RPC (faster than disk I/O)
            # RPC call retrieves pre-compressed gzip bytes from memory
            compressed_data = self._debt_ledger_client.get_compressed_summaries_rpc()

            if compressed_data is None or len(compressed_data) == 0:
                return jsonify({'error': 'Debt ledger data not found'}), 404

            # Return pre-compressed data with gzip header (browser decompresses automatically)
            return Response(compressed_data, content_type='application/json', headers={
                'Content-Encoding': 'gzip'
            })

        except Exception as e:
            bt.logging.error(f"Error retrieving debt ledger summaries via RPC: {e}")
            return jsonify({'error': 'Internal server error retrieving debt ledger data'}), 500

    def get_penalty_ledger(self, minerid):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Use RPC getter to access penalty ledger via debt ledger manager
        data = self._debt_ledger_client.get_penalty_ledger(minerid)

        if data is None:
            return jsonify({'error': 'Penalty ledger data not found'}), 404
        else:
            return self._jsonify_with_custom_encoder(data)

    # ============================================================================
    # STATISTICS ENDPOINTS
    # ============================================================================

    def get_validator_checkpoint(self):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Validator checkpoint data is only available for tier 100
        if not self.can_access_tier(api_key, 100):
            return jsonify({'error': 'Validator checkpoint data requires tier 100 access'}), 403

        # Try to get compressed data from memory cache first via CoreOutputsClient
        compressed_data = None
        if self._core_outputs_client:
            try:
                compressed_data = self._core_outputs_client.get_compressed_checkpoint_from_memory()
            except Exception as e:
                bt.logging.debug(f"Error accessing compressed checkpoint cache: {e}")

        if compressed_data is not None:
            # Return pre-compressed data with appropriate headers
            return Response(compressed_data, content_type='application/json', headers={
                'Content-Encoding': 'gzip'
            })

        # Fallback to file read if memory unavailable
        # Checkpoint is always stored as compressed file
        f_gz = ValiBkpUtils.get_vcp_output_path()

        if os.path.exists(f_gz):
            # Read pre-compressed file directly
            try:
                with open(f_gz, 'rb') as fh:
                    compressed_data = fh.read()
                return Response(compressed_data, content_type='application/json', headers={
                    'Content-Encoding': 'gzip'
                })
            except Exception as e:
                bt.logging.error(f"Failed to read compressed checkpoint: {e}")
                return jsonify({'error': 'Failed to read checkpoint data'}), 500
        else:
            return jsonify({'error': 'Checkpoint data not found'}), 404

    def get_validator_checkpoint_statistics(self):
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Grab the optional "checkpoints" query param; default it to "true"
        show_checkpoints = request.args.get("checkpoints", "true").lower()
        include_checkpoints = show_checkpoints == "true"

        # PRIMARY: Try to use pre-compressed payload from memory cache (fastest)
        if self._statistics_outputs_client:
            compressed_data = self._statistics_outputs_client.get_compressed_statistics(include_checkpoints)
            if compressed_data:
                # Return pre-compressed JSON directly
                return Response(compressed_data, content_type='application/json', headers={
                    'Content-Encoding': 'gzip'
                })

        # FALLBACK 1: If no modification needed, serve compressed file directly
        if show_checkpoints == "true":
            f_gz = ValiBkpUtils.get_miner_stats_dir() + ".gz"
            if os.path.exists(f_gz):
                compressed_data = self._get_file(f_gz, binary=True)
                return Response(compressed_data, content_type='application/json', headers={
                    'Content-Encoding': 'gzip'
                })

        # FALLBACK 2: Decompress and modify if needed (checkpoints=false or no .gz file)
        f = ValiBkpUtils.get_miner_stats_dir()
        data = self._get_file(f)
        if not data:
            return jsonify({'error': 'Statistics data not found'}), 404

        # If checkpoints=false, remove the "checkpoints" key from each element in data
        if show_checkpoints == "false":
            for element in data.get("data", []):
                element.pop("checkpoints", None)

        return self._jsonify_with_custom_encoder(data)

    def get_validator_checkpoint_statistics_unique(self, minerid):
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Get statistics data from disk
        f = ValiBkpUtils.get_miner_stats_dir()
        data = self._get_file(f)
        if not data:
            return jsonify({'error': 'Statistics data not found'}), 404

        data_summary = data.get("data", [])
        if not data_summary:
            return jsonify({'error': 'No data found'}), 404

        # Grab the optional "checkpoints" query param; default it to "true"
        show_checkpoints = request.args.get("checkpoints", "true").lower()

        for element in data_summary:
            if element.get("hotkey", None) == minerid:
                # If the user set checkpoints=false, remove them from this element
                if show_checkpoints == "false":
                    element.pop("checkpoints", None)
                return jsonify(element)

        return jsonify({'error': f'Miner ID {minerid} not found'}), 404

    def get_eliminations(self):
        api_key = self._get_api_key_safe()

        # Check if the API key is valid
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        f = ValiBkpUtils.get_eliminations_dir()
        data = self._get_file(f)

        if data is None:
            return jsonify({'error': 'Eliminations data not found'}), 404
        else:
            return self._jsonify_with_custom_encoder(data)

    # ============================================================================
    # TRADING ENDPOINTS
    # ============================================================================

    def get_limit_orders_unique(self, minerid):
        api_key = self._get_api_key_safe()

        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Parse status filter from query param
        status_param = request.args.get('status')
        status_filter = None
        if status_param:
            status_filter = [s.strip().lower() for s in status_param.split(',')]
            valid = {'unfilled', 'filled', 'cancelled'}
            invalid = set(status_filter) - valid
            if invalid:
                return jsonify({'error': f'Invalid status values: {invalid}. Valid values are: unfilled, filled, cancelled'}), 400

        api_key_tier = self.get_api_key_tier(api_key)
        if self.can_access_tier(api_key, 100) and self._limit_order_client:
            orders_data = self._limit_order_client.to_dashboard_dict(minerid, status_filter)
            if not orders_data:
                return jsonify({'error': f'No limit orders found for miner {minerid}'}), 404
        else:
            try:
                orders_data = ValiBkpUtils.get_limit_orders(minerid, unfilled_only=True, running_unit_tests=False)
                if not orders_data:
                    return jsonify({'error': f'No limit orders found for miner {minerid}'}), 404
            except Exception as e:
                bt.logging.error(f"Error retrieving limit orders for {minerid}: {e}")
                return jsonify({'error': 'Error retrieving limit orders'}), 500

        return jsonify(orders_data)

    def get_orders_for_miner(self, minerid):
        """
        Get all orders for a miner, grouped by status.

        Query params:
            status: Comma-separated list (unfilled, filled, cancelled)

        Returns:
            {"unfilled": [...], "filled": [...], "cancelled": [...]}
        """
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        if not self.can_access_tier(api_key, 100):
            return jsonify({'error': 'Your API key does not have access to tier 100 data'}), 403

        # Parse status filter
        status_param = request.args.get('status')
        status_filter = None
        if status_param:
            status_filter = [s.strip().lower() for s in status_param.split(',')]
            valid = {'unfilled', 'filled', 'cancelled'}
            invalid = set(status_filter) - valid
            if invalid:
                return jsonify({'error': f'Invalid status values: {invalid}. Valid values are: unfilled, filled, cancelled'}), 400
        else:
            status_filter = ['unfilled', 'filled', 'cancelled']

        result = {s: [] for s in status_filter}

        try:
            # Get unfilled/cancelled from LimitOrderClient (same as /limit-orders)
            if ('unfilled' in status_filter or 'cancelled' in status_filter) and self._limit_order_client:
                limit_statuses = [s for s in status_filter if s in ('unfilled', 'cancelled')]
                limit_data = self._limit_order_client.to_dashboard_dict(minerid, limit_statuses)
                if limit_data:
                    for status in limit_statuses:
                        if status in limit_data:
                            result[status] = limit_data[status]

            # Get filled orders from positions
            if 'filled' in status_filter and self.position_manager:
                positions = self.position_manager.get_positions_for_one_hotkey(minerid, sort_positions=True)
                if positions:
                    for position in positions:
                        for order in position.orders:
                            result['filled'].append(order.to_python_dict())

            # Sort statuses by processed_ms
            for status in result:
                result[status].sort(key=lambda o: o.get('processed_ms', 0))

            if not any(result.values()):
                return jsonify({'error': f'No orders found for miner {minerid}'}), 404

            return jsonify(result)

        except Exception as e:
            bt.logging.error(f"Error retrieving orders for {minerid}: {e}")
            return jsonify({'error': 'Error retrieving orders'}), 500

    # ============================================================================
    # COLLATERAL ENDPOINTS
    # ============================================================================

    def deposit_collateral(self):
        """Process collateral deposit with encoded extrinsic."""
        MAX_EXTRINSIC_HEX = 200_000 # ~100 KB decoded;

        # Check if contract manager is available
        if not self.contract_manager:
            return jsonify({'error': 'Collateral operations not available'}), 503

        try:
            # Parse JSON request
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            # Check vanta-cli version FIRST - reject outdated versions
            vanta_cli_version = (
                data.get('version')
                or data.get('ptncli_version')
                or '0.0.0'
            )
            vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
            if vanta_cli_error:
                return jsonify({'error': vanta_cli_error}), 400

            # Validate required fields
            required_fields = ['extrinsic']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            # Validate extrinsic
            extrinsic = data.get('extrinsic')
            if not isinstance(extrinsic, str):
                return jsonify({'error': 'extrinsic must be a hex string'}), 400
            if len(extrinsic) > MAX_EXTRINSIC_HEX:
                return jsonify({'error': 'extrinsic too large'}), 413
            if len(extrinsic) % 2 != 0 or not all(c in hexdigits for c in extrinsic):
                return jsonify({'error': 'extrinsic must be even-length hex'}), 400

            # Process the deposit using raw data
            result = self.contract_manager.process_deposit_request(
                extrinsic_hex=extrinsic
            )

            # Return response
            return jsonify(result)

        except Exception as e:
            bt.logging.error(f"Error processing collateral deposit: {e}")
            return jsonify({'error': 'Internal server error processing deposit'}), 500

    def query_withdraw_collateral(self):
        """Query collateral withdrawal request for potential slashed amount"""
        # Check if contract manager is available
        if not self.contract_manager:
            return jsonify({'error': 'Collateral operations not available'}), 503

        try:
            # Parse JSON request
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            # Check vanta-cli version FIRST - reject outdated versions
            vanta_cli_version = (
                data.get('version')
                or data.get('ptncli_version')
                or '0.0.0'
            )
            vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
            if vanta_cli_error:
                return jsonify({'error': vanta_cli_error}), 400

            # Validate required fields for withdrawal query
            required_fields = ['amount', 'miner_hotkey']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            # Validate amount is a positive number
            try:
                amount = float(data['amount'])
                if amount <= 0:
                    return jsonify({'error': 'Amount must be a positive number'}), 400
            except (ValueError, TypeError):
                return jsonify({'error': 'Amount must be a valid number'}), 400

            # Validate miner_hotkey is a valid SS58 address
            miner_hotkey = data['miner_hotkey']
            try:
                # Attempt to create a Keypair to validate SS58 format
                Keypair(ss58_address=miner_hotkey)
            except Exception:
                return jsonify({'error': 'Invalid SS58 address format for miner_hotkey'}), 400

            # Process the withdrawal query
            result = self.contract_manager.query_withdrawal_request(
                amount=amount,
                miner_hotkey=miner_hotkey
            )

            # Return response
            return jsonify(result)

        except Exception as e:
            bt.logging.error(f"Error processing collateral withdrawal query: {e}")
            return jsonify({'error': 'Internal server error processing withdrawal query'}), 500

    def withdraw_collateral(self):
        """Process collateral withdrawal request."""
        # Check if contract manager is available
        if not self.contract_manager:
            return jsonify({'error': 'Collateral operations not available'}), 503

        try:
            # Parse JSON request
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            # Check vanta-cli version FIRST - reject outdated versions
            vanta_cli_version = (
                data.get('version')
                or data.get('ptncli_version')
                or '0.0.0'
            )
            vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
            if vanta_cli_error:
                return jsonify({'error': vanta_cli_error}), 400

            # Validate required fields for signed withdrawal
            required_fields = ['amount', 'miner_coldkey', 'miner_hotkey', 'nonce', 'timestamp', 'signature']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            # Verify the withdrawal signature
            keypair = Keypair(ss58_address=data['miner_coldkey'])
            message = json.dumps({
                "amount": data['amount'],
                "miner_coldkey": data['miner_coldkey'],
                "miner_hotkey": data['miner_hotkey'],
                "nonce": data['nonce'],
                "timestamp": data['timestamp']
            }, sort_keys=True).encode('utf-8')
            is_valid = keypair.verify(message, bytes.fromhex(data['signature']))
            if not is_valid:
                return jsonify({'error': 'Invalid signature. Withdrawal request unauthorized'}), 401

            # Verify coldkey-hotkey ownership using subtensor
            owns_hotkey = self._verify_coldkey_owns_hotkey(data['miner_coldkey'], data['miner_hotkey'])
            if not owns_hotkey:
                return jsonify({'error': 'Coldkey does not own the specified hotkey'}), 403

            # Verify nonce
            nonce_key = f"{data['miner_coldkey']}::{data['miner_hotkey']}"
            is_valid, error_msg = self.nonce_manager.is_valid_request(
                address=nonce_key,
                nonce=str(data['nonce']),
                timestamp=int(data['timestamp'])
            )
            if not is_valid:
                return jsonify({'error': f'{error_msg}'}), 401

            # Validate amount is a positive number
            try:
                amount = float(data['amount'])
                if amount <= 0:
                    return jsonify({'error': 'Amount must be a positive number'}), 400
            except (ValueError, TypeError):
                return jsonify({'error': 'Amount must be a valid number'}), 400

            # Process the withdrawal using verified data
            result = self.contract_manager.process_withdrawal_request(
                amount=data['amount'],
                miner_coldkey=data['miner_coldkey'],
                miner_hotkey=data['miner_hotkey']
            )

            # Return response
            return jsonify(result)

        except Exception as e:
            bt.logging.error(f"Error processing collateral withdrawal: {e}")
            return jsonify({'error': 'Internal server error processing withdrawal'}), 500

    def get_all_collateral_data(self):
        """Get collateral data for all miners.

        Example curl requests:

        # Get all collateral data for all miners
        curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/collateral/

        # Get collateral data for a specific miner
        curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/collateral/?hotkey=5GhDr...

        # Get only the most recent collateral record for each miner
        curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/collateral/?most_recent=true

        # Combine filters: specific miner's most recent record
        curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/collateral/?hotkey=5GhDr...&most_recent=true

        Response format:
        {
            "status": "success",
            "data": {
                "hotkey1": [{
                    "account_size": 1000.0,
                    "account_size_theta": 10.0,
                    "update_time_ms": 1234567890000,
                    "valid_date_timestamp": 1234567890000
                }],
                ...
            },
            "miner_count": 5,
            "total_records": 25,
            "timestamp": 1234567890000
        }
        """

        # Check API key authentication
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check if contract manager is available
        if not self.contract_manager:
            return jsonify({'error': 'Collateral operations not available'}), 503

        try:
            # Get query parameters for filtering
            hotkey_filter = request.args.get('hotkey')
            most_recent_only = request.args.get('most_recent', 'false').lower() == 'true'

            # Get all collateral data using the proper serialization method
            # Pass most_recent_only directly to avoid double iteration
            data = self._miner_account_client.accounts_dict(most_recent_only=most_recent_only)

            # Apply hotkey filter if requested
            if hotkey_filter and hotkey_filter in data:
                data = {hotkey_filter: data[hotkey_filter]}

            # Return consistent response format
            return jsonify({
                'status': 'success',
                'data': data,
                'miner_count': len(data),
                'total_records': sum(len(records) for records in data.values()),
                'timestamp': TimeUtil.now_in_millis()
            })

        except Exception as e:
            bt.logging.error(f"Error getting all collateral data: {e}")
            return jsonify({'error': 'Internal server error retrieving data'}), 500

    def get_collateral_balance(self, miner_address):
        """Get a miner's collateral balance."""
        # Check if contract manager is available
        if not self.contract_manager:
            return jsonify({'error': 'Collateral operations not available'}), 503

        try:
            # Get the balance
            balance = self.contract_manager.get_miner_collateral_balance(miner_address)

            if balance is None:
                return jsonify({'error': 'Failed to retrieve collateral balance'}), 500

            return jsonify({
                'miner_address': miner_address,
                'balance_theta': balance
            })

        except Exception as e:
            bt.logging.error(f"Error getting collateral balance for {miner_address}: {e}")
            return jsonify({'error': 'Internal server error retrieving balance'}), 500

    def asset_selection(self):
        """Process asset selection request."""
        try:
            # Parse JSON request
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            # Check vanta-cli version FIRST - reject outdated versions
            vanta_cli_version = (
                data.get('version')
                or data.get('ptncli_version')
                or '0.0.0'
            )
            vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
            if vanta_cli_error:
                return jsonify({'error': vanta_cli_error}), 400

            # Validate required fields for signed withdrawal
            required_fields = ['asset_selection', 'miner_coldkey', 'miner_hotkey', 'signature']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            # Verify the withdrawal signature
            keypair = Keypair(ss58_address=data['miner_coldkey'])
            message = json.dumps({
                "asset_selection": data['asset_selection'],
                "miner_coldkey": data['miner_coldkey'],
                "miner_hotkey": data['miner_hotkey']
            }, sort_keys=True).encode('utf-8')
            is_valid = keypair.verify(message, bytes.fromhex(data['signature']))
            if not is_valid:
                return jsonify({'error': 'Invalid signature. Asset selection request unauthorized'}), 401

            # Verify coldkey-hotkey ownership using subtensor
            owns_hotkey = self._verify_coldkey_owns_hotkey(data['miner_coldkey'], data['miner_hotkey'])
            if not owns_hotkey:
                return jsonify({'error': 'Coldkey does not own the specified hotkey'}), 403

            # Process the asset selection using verified data
            result = self._asset_selection_client.process_asset_selection_request(
                asset_selection=data['asset_selection'],
                miner=data['miner_hotkey']
            )

            # Return response
            return jsonify(result)

        except Exception as e:
            bt.logging.error(f"Error processing asset selection: {e}")
            return jsonify({'error': 'Internal server error processing asset selection'}), 500

    def get_miner_selections(self):
        """Get all miner asset selection data."""
        try:
            # Check API key authentication
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Check if asset selection client is available
            if not self._asset_selection_client:
                return jsonify({'error': 'Asset selection data not available'}), 503

            # Get all miner selection data using the getter method
            selections_data = self._asset_selection_client.get_all_miner_selections()

            return jsonify({
                'miner_selections': selections_data,
                'total_miners': len(selections_data),
                'timestamp': TimeUtil.now_in_millis()
            })

        except Exception as e:
            bt.logging.error(f"Error retrieving miner selections: {e}")
            return jsonify({'error': 'Internal server error retrieving miner selections'}), 500

    def process_development_order(self):
        """
        Process development orders for testing market, limit, and cancel operations.
        Uses fixed hotkey 'DEVELOPMENT' for all operations.
        Requires tier 200 access.

        Example requests:

        # Market order
        curl -X POST http://localhost:48888/development/order \\
          -H "Authorization: Bearer YOUR_API_KEY" \\
          -H "Content-Type: application/json" \\
          -d '{"execution_type": "MARKET", "trade_pair_id": "BTCUSD", "order_type": "LONG", "leverage": 1.0}'

        # Limit order
        curl -X POST http://localhost:48888/development/order \\
          -H "Authorization: Bearer YOUR_API_KEY" \\
          -H "Content-Type": application/json" \\
          -d '{"execution_type": "LIMIT", "trade_pair_id": "BTCUSD", "order_type": "LONG", "leverage": 1.0, "limit_price": 50000.0}'

        # Bracket order (requires existing position)
        curl -X POST http://localhost:48888/development/order \\
          -H "Authorization: Bearer YOUR_API_KEY" \\
          -H "Content-Type: application/json" \\
          -d '{"execution_type": "BRACKET", "trade_pair_id": "BTCUSD", "stop_loss": 48000.0, "take_profit": 52000.0}'

        # Cancel specific limit order
        curl -X POST http://localhost:48888/development/order \\
          -H "Authorization: Bearer YOUR_API_KEY" \\
          -H "Content-Type: application/json" \\
          -d '{"execution_type": "LIMIT_CANCEL", "trade_pair_id": "BTCUSD", "order_uuid": "specific-uuid"}'

        # Cancel all limit orders for trade pair
        curl -X POST http://localhost:48888/development/order \\
          -H "Authorization: Bearer YOUR_API_KEY" \\
          -H "Content-Type: application/json" \\
          -d '{"execution_type": "LIMIT_CANCEL", "trade_pair_id": "BTCUSD"}'
        """
        DEVELOPMENT_HOTKEY = ValiConfig.DEVELOPMENT_HOTKEY

        # Check API key authentication
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check if API key has tier 200 access
        if not self.can_access_tier(api_key, 200):
            return jsonify({'error': 'Development order endpoint requires tier 200 access'}), 403

        try:
            # Parse and validate request
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            # Log raw request data for debugging JSON parse errors
            raw_data = request.get_data(as_text=True)
            bt.logging.debug(f"[DEV_ORDER] Raw request body (first 300 chars): {raw_data[:300]}")
            bt.logging.debug(f"[DEV_ORDER] Request body length: {len(raw_data)} chars")

            try:
                data = request.get_json()
            except json.JSONDecodeError as e:
                bt.logging.error(
                    f"[DEV_ORDER] JSON parse error at position {e.pos}: {e.msg}\n"
                    f"  Raw body: {raw_data}\n"
                    f"  Error context (char {max(0, e.pos-20)} to {min(len(raw_data), e.pos+20)}): "
                    f"{raw_data[max(0, e.pos-20):min(len(raw_data), e.pos+20)]}"
                )
                return jsonify({
                    'error': f'Invalid JSON at position {e.pos}: {e.msg}',
                    'position': e.pos
                }), 400

            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            # Create signal dict from request data
            signal = {
                'trade_pair': {'trade_pair_id': data.get('trade_pair_id')},
                'order_type': data.get('order_type', '').upper(),
                'leverage': data.get('leverage'),
                'value': data.get('value'),
                'quantity': data.get('quantity'),
                'execution_type': data.get('execution_type', 'MARKET').upper(),
                'limit_price': data.get('limit_price'),
                'stop_loss': data.get('stop_loss'),
                'take_profit': data.get('take_profit'),
                'bracket_orders': data.get('bracket_orders'),
            }

            now_ms = TimeUtil.now_in_millis()
            miner_repo_version = "development"

            # Use unified OrderProcessor dispatcher (replaces lines 1466-1553)
            result = OrderProcessor.process_order(
                signal=signal,
                miner_order_uuid=data.get('order_uuid'),
                now_ms=now_ms,
                miner_hotkey=DEVELOPMENT_HOTKEY,
                miner_repo_version=miner_repo_version,
                limit_order_client=self._limit_order_client,
                market_order_manager=self.market_order_manager
            )

            # Consistent response format across all order types
            return jsonify({
                'status': 'success',
                'execution_type': result.execution_type.value,
                'order_uuid': data.get('order_uuid'),
                'order': result.get_response_json()
            })

        except SignalException as e:
            bt.logging.error(f"SignalException in development order: {e}")
            return jsonify({'error': f'Signal error: {str(e)}'}), 400

        except Exception as e:
            bt.logging.error(f"Error processing development order: {e}")
            bt.logging.error(traceback.format_exc())
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    # ============================================================================
    # ENTITY MANAGEMENT ENDPOINTS
    # ============================================================================

    def register_entity(self):
        """
        Register a new entity.

        Example:
        curl -X POST http://localhost:48888/entity/register \\
          -H "Content-Type: application/json" \\
          -d '{
            "entity_hotkey": "5GhDr...",
            "entity_coldkey": "5FxY...",
            "max_subaccounts": 500,
            "signature": "0x..."
          }'
        """
        # Check if entity client is available
        if not self._entity_client:
            return jsonify({'error': 'Entity management not available'}), 503

        try:
            # Parse and validate request
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            # Check vanta-cli version FIRST - reject outdated versions
            vanta_cli_version = (
                data.get('version')
                or data.get('ptncli_version')
                or '0.0.0'
            )
            vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
            if vanta_cli_error:
                return jsonify({'error': vanta_cli_error}), 400

            # Validate required fields
            required_fields = ['entity_coldkey', 'entity_hotkey', 'signature']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            entity_coldkey = data['entity_coldkey']
            entity_hotkey = data['entity_hotkey']

            # Verify signature
            keypair = Keypair(ss58_address=entity_coldkey)
            message = json.dumps({
                "entity_coldkey": entity_coldkey,
                "entity_hotkey": entity_hotkey
            }, sort_keys=True).encode('utf-8')

            is_valid = keypair.verify(message, bytes.fromhex(data['signature']))
            if not is_valid:
                return jsonify({'error': 'Invalid signature. Entity registration unauthorized'}), 401

            # Verify coldkey-hotkey ownership using subtensor
            owns_hotkey = self._verify_coldkey_owns_hotkey(entity_coldkey, entity_hotkey)
            if not owns_hotkey:
                return jsonify({'error': 'Coldkey does not own the specified hotkey'}), 403

            # Register entity via RPC
            success, message = self._entity_client.register_entity(
                entity_hotkey=entity_hotkey
            )

            if success:
                return jsonify({
                    'status': 'success',
                    'message': message,
                    'entity_hotkey': entity_hotkey
                }), 200
            else:
                return jsonify({'error': message}), 400

        except Exception as e:
            bt.logging.error(f"Error registering entity: {e}")
            return jsonify({'error': 'Internal server error registering entity'}), 500

    def create_subaccount(self):
        """
        Create a new subaccount for an entity.

        Requires account_size (USD) and asset_class in request payload.
        These values are immutable once the subaccount is created.

        Example:
        curl -X POST http://localhost:48888/entity/create-subaccount \\
          -H "Content-Type: application/json" \\
          -d '{
            "entity_hotkey": "5GhDr...",
            "entity_coldkey": "5FxY...",
            "account_size": 25000,
            "asset_class": "crypto",
            "signature": "0x..."
          }'
        """
        import time
        t_start = time.time()
        timings = {}

        # Check if entity client is available
        if not self._entity_client:
            return jsonify({'error': 'Entity management not available'}), 503

        try:
            # Parse and validate request
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            # Check vanta-cli version FIRST - reject outdated versions
            vanta_cli_version = (
                data.get('version')
                or data.get('ptncli_version')
                or '0.0.0'
            )
            vanta_cli_error = self.check_vanta_cli_version(vanta_cli_version)
            if vanta_cli_error:
                return jsonify({'error': vanta_cli_error}), 400

            # Validate required fields
            required_fields = ['entity_coldkey', 'entity_hotkey', 'account_size', 'asset_class', 'signature']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

            entity_coldkey = data['entity_coldkey']
            entity_hotkey = data['entity_hotkey']
            account_size = data['account_size']
            asset_class = data['asset_class']

            # Validate account_size is a positive number
            try:
                account_size = float(account_size)
                if account_size <= 0:
                    return jsonify({'error': 'account_size must be a positive number'}), 400
            except (TypeError, ValueError):
                return jsonify({'error': 'account_size must be a valid number'}), 400

            # Validate asset_class is a non-empty string
            if not isinstance(asset_class, str) or not asset_class.strip():
                return jsonify({'error': 'asset_class must be a non-empty string'}), 400

            asset_class = asset_class.strip()

            # Verify signature
            t0 = time.time()
            keypair = Keypair(ss58_address=entity_coldkey)
            message = json.dumps({
                "entity_coldkey": entity_coldkey,
                "entity_hotkey": entity_hotkey,
                "account_size": account_size,
                "asset_class": asset_class
            }, sort_keys=True).encode('utf-8')

            is_valid = keypair.verify(message, bytes.fromhex(data['signature']))
            timings['verify_signature'] = int((time.time() - t0) * 1000)
            if not is_valid:
                return jsonify({'error': 'Invalid signature. Subaccount creation unauthorized'}), 401

            # Verify coldkey-hotkey ownership using subtensor
            t0 = time.time()
            owns_hotkey = self._verify_coldkey_owns_hotkey(entity_coldkey, entity_hotkey)
            timings['verify_coldkey_ownership'] = int((time.time() - t0) * 1000)
            if not owns_hotkey:
                return jsonify({'error': 'Coldkey does not own the specified hotkey'}), 403

            # Create subaccount via RPC
            t0 = time.time()
            success, subaccount_info, message = self._entity_client.create_subaccount(
                entity_hotkey, account_size, asset_class
            )
            timings['create_subaccount_rpc'] = int((time.time() - t0) * 1000)

            if success:
                # Broadcast subaccount registration to other validators
                try:
                    t0 = time.time()
                    self._entity_client.broadcast_subaccount_registration(
                        entity_hotkey=entity_hotkey,
                        subaccount_id=subaccount_info['subaccount_id'],
                        subaccount_uuid=subaccount_info['subaccount_uuid'],
                        synthetic_hotkey=subaccount_info['synthetic_hotkey'],
                        account_size=subaccount_info['account_size'],
                        asset_class=subaccount_info['asset_class']
                    )
                    timings['broadcast_rpc'] = int((time.time() - t0) * 1000)
                    bt.logging.info(f"[REST_API] Broadcasted subaccount registration for {subaccount_info['synthetic_hotkey']}")
                except Exception as e:
                    # Don't fail the request if broadcast fails - it's a background operation
                    bt.logging.warning(f"[REST_API] Failed to broadcast subaccount registration: {e}")

                total_ms = int((time.time() - t_start) * 1000)
                bt.logging.info(f"[REST_API] create_subaccount completed ({total_ms} ms) | timings: {timings}")

                return jsonify({
                    'status': 'success',
                    'message': message,
                    'subaccount': subaccount_info
                }), 200
            else:
                return jsonify({'error': message}), 400

        except Exception as e:
            bt.logging.error(f"Error creating subaccount: {e}")
            return jsonify({'error': 'Internal server error creating subaccount'}), 500

    def get_entity(self, entity_hotkey):
        """
        Get entity data for a specific entity.

        Example:
        curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/entity/5GhDr...
        """
        # Check API key authentication
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check if API key has tier 200 access
        if not self.can_access_tier(api_key, 200):
            return jsonify({'error': 'Your API key does not have access to tier 200 data'}), 403

        # Check if entity client is available
        if not self._entity_client:
            return jsonify({'error': 'Entity management not available'}), 503

        try:
            # Get entity data via RPC
            entity_data = self._entity_client.get_entity_data(entity_hotkey)

            if entity_data:
                return jsonify({
                    'status': 'success',
                    'entity': entity_data
                }), 200
            else:
                return jsonify({'error': f'Entity {entity_hotkey} not found'}), 404

        except Exception as e:
            bt.logging.error(f"Error retrieving entity {entity_hotkey}: {e}")
            return jsonify({'error': 'Internal server error retrieving entity'}), 500

    def get_all_entities(self):
        """
        Get all registered entities.

        Example:
        curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:48888/entities
        """
        # Check API key authentication
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check if API key has tier 200 access
        if not self.can_access_tier(api_key, 200):
            return jsonify({'error': 'Your API key does not have access to tier 200 data'}), 403

        # Check if entity client is available
        if not self._entity_client:
            return jsonify({'error': 'Entity management not available'}), 503

        try:
            # Get all entities via RPC
            entities = self._entity_client.get_all_entities()

            return jsonify({
                'status': 'success',
                'entities': entities,
                'entity_count': len(entities),
                'timestamp': TimeUtil.now_in_millis()
            }), 200

        except Exception as e:
            bt.logging.error(f"Error retrieving all entities: {e}")
            return jsonify({'error': 'Internal server error retrieving entities'}), 500

    def eliminate_subaccount(self):
        """
        Eliminate a subaccount.

        Example:
        curl -X POST http://localhost:48888/entity/subaccount/eliminate \\
          -H "Authorization: Bearer YOUR_API_KEY" \\
          -H "Content-Type: application/json" \\
          -d '{"entity_hotkey": "5GhDr...", "subaccount_id": 0, "reason": "manual_elimination"}'
        """
        # Check API key authentication
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check if API key has tier 200 access
        if not self.can_access_tier(api_key, 200):
            return jsonify({'error': 'Your API key does not have access to tier 200 data'}), 403

        # Check if entity client is available
        if not self._entity_client:
            return jsonify({'error': 'Entity management not available'}), 503

        try:
            # Parse and validate request
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            # Validate required fields
            required_fields = ['entity_hotkey', 'subaccount_id']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            entity_hotkey = data['entity_hotkey']
            subaccount_id = data['subaccount_id']
            reason = data.get('reason', 'manual_elimination')

            # Validate subaccount_id is an integer
            try:
                subaccount_id = int(subaccount_id)
            except (ValueError, TypeError):
                return jsonify({'error': 'subaccount_id must be an integer'}), 400

            # Eliminate subaccount via RPC
            success, message = self._entity_client.eliminate_subaccount(
                entity_hotkey=entity_hotkey,
                subaccount_id=subaccount_id,
                reason=reason
            )

            if success:
                return jsonify({
                    'status': 'success',
                    'message': message
                }), 200
            else:
                return jsonify({'error': message}), 400

        except Exception as e:
            bt.logging.error(f"Error eliminating subaccount: {e}")
            return jsonify({'error': 'Internal server error eliminating subaccount'}), 500

    def get_subaccount_dashboard(self, synthetic_hotkey):
        """
        Get comprehensive dashboard data for a subaccount.

        This endpoint aggregates data from multiple RPC services:
        - Subaccount info (status, timestamps)
        - Challenge period status (bucket, start time)
        - Debt ledger data (DebtLedger instance)
        - Position data (positions, leverage)
        - Statistics (cached miner statistics with metrics, scores, rankings)
        - Elimination status (if eliminated)

        Example:
        curl -H "Authorization: Bearer YOUR_API_KEY" \
             http://localhost:48888/entity/subaccount/entity_alpha_0
        """
        # Check API key authentication
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check if API key has tier 200 access
        if not self.can_access_tier(api_key, 200):
            return jsonify({'error': 'Your API key does not have access to tier 200 data'}), 403

        # Check if entity client is available
        if not self._entity_client:
            return jsonify({'error': 'Entity management not available'}), 503

        try:
            # Get dashboard data via RPC
            dashboard_data = self._entity_client.get_subaccount_dashboard_data(synthetic_hotkey)

            if dashboard_data:
                # Serialize the dashboard payload (excluding the timestamp wrapper which changes every call)
                dashboard_json = json.dumps(dashboard_data, cls=CustomEncoder, sort_keys=True)

                # Compute ETag from dashboard content
                etag = '"' + hashlib.md5(dashboard_json.encode()).hexdigest() + '"'

                # Check If-None-Match
                if_none_match = request.headers.get('If-None-Match')
                if if_none_match == etag:
                    return Response(status=304, headers={'ETag': etag})

                # Build full response with ETag
                response_data = json.dumps({
                    'status': 'success',
                    'dashboard': dashboard_data,
                    'timestamp': TimeUtil.now_in_millis()
                }, cls=CustomEncoder)

                response = Response(response_data, content_type='application/json')
                response.headers['ETag'] = etag
                return response, 200
            else:
                return jsonify({'error': f'Subaccount {synthetic_hotkey} not found'}), 404

        except Exception as e:
            bt.logging.error(f"Error retrieving dashboard for {synthetic_hotkey}: {e}")
            return jsonify({'error': 'Internal server error retrieving dashboard'}), 500

    def calculate_subaccount_payout(self):
        """
        Calculate payout for a subaccount based on debt ledger checkpoints.

        Request body:
        {
            "subaccount_uuid": "uuid-string",
            "start_time_ms": 1234567890000,
            "end_time_ms": 1234567890000
        }

        Response:
        {
            "status": "success",
            "payout_data": {
                "hotkey": "entity_hotkey_0",
                "total_checkpoints": 10,
                "checkpoints": [...],
                "payout": 1234.56
            },
            "timestamp": 1234567890000
        }

        Requires tier 200 access.
        """
        # Check API key authentication
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check tier 200 access
        if not self.can_access_tier(api_key, 200):
            return jsonify({'error': 'Your API key does not have access to tier 200 data'}), 403

        # Check if entity client is available
        if not self._entity_client:
            return jsonify({'error': 'Entity management not available'}), 503

        try:
            # Parse JSON request
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            # Validate required fields
            required_fields = ['subaccount_uuid', 'start_time_ms', 'end_time_ms']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            subaccount_uuid = data['subaccount_uuid']

            # Validate timestamps
            try:
                start_time_ms = int(data['start_time_ms'])
                end_time_ms = int(data['end_time_ms'])

                if start_time_ms < 0 or end_time_ms < 0:
                    return jsonify({'error': 'Timestamps must be non-negative'}), 400

                if start_time_ms > end_time_ms:
                    return jsonify({'error': 'start_time_ms must be <= end_time_ms'}), 400
            except (ValueError, TypeError):
                return jsonify({'error': 'Timestamps must be valid integers'}), 400

            # Calculate payout via EntityClient
            payout_data = self._entity_client.calculate_subaccount_payout(
                subaccount_uuid,
                start_time_ms,
                end_time_ms
            )

            if payout_data:
                return jsonify({
                    'status': 'success',
                    'data': payout_data,
                    'timestamp': TimeUtil.now_in_millis()
                }), 200
            else:
                return jsonify({
                    'error': f'Subaccount {subaccount_uuid} not found or has no debt ledger data'
                }), 404

        except Exception as e:
            error_msg = str(e)
            bt.logging.error(f"Error calculating subaccount payout: {error_msg}")
            bt.logging.error(traceback.format_exc())

            return jsonify({
                'error': 'Internal server error calculating payout',
                'detail': error_msg if self.running_unit_tests else None
            }), 500

    def _verify_coldkey_owns_hotkey(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        """
        Verify that a coldkey owns the specified hotkey using subtensor.

        Args:
            coldkey_ss58: The coldkey SS58 address
            hotkey_ss58: The hotkey SS58 address to verify ownership of

        Returns:
            bool: True if coldkey owns the hotkey, False otherwise
        """
        try:
            return self.contract_manager.verify_coldkey_owns_hotkey(coldkey_ss58, hotkey_ss58)
        except Exception as e:
            bt.logging.error(f"Error verifying coldkey-hotkey ownership: {e}")
            return False


    def _get_api_key(self):
        """
        Get the API key from the query parameters or request headers.
        This is the original method kept for backward compatibility.
        """
        api_key = self._get_api_key_safe()
        if api_key is None:
            # Log when no API key is found
            bt.logging.debug(f"No API key found in request to {request.path}")
        return api_key

    def _get_file(self, f, attempts=3, binary=False):
        """Read file with multiple attempts and return its contents."""
        file_path = os.path.abspath(os.path.join(self.data_path, f))
        if not os.path.exists(file_path):
            return None

        for attempt_number in range(attempts):
            try:
                if binary:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                else:
                    if file_path.endswith('.gz'):
                        with gzip.open(file_path, 'rt', encoding='utf-8') as fh:
                            data = json.load(fh)
                    else:
                        with open(file_path, "r") as file:
                            data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                if attempt_number == attempts - 1:
                    bt.logging.error(f"Failed to decode JSON after {attempts} attempts: {file_path}")
                    raise
                else:
                    bt.logging.debug(
                        f"Attempt {attempt_number + 1} failed with JSONDecodeError {e}, retrying..."
                    )
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                bt.logging.error(f"Unexpected error reading file {file_path}: {type(e).__name__}: {str(e)}")
                raise

    @staticmethod
    def check_vanta_cli_version(version: str) -> Optional[str]:
        """
        Check if vanta-cli version meets minimum requirements.
        This is now an enforced requirement - requests will be rejected if version is outdated.

        Args:
            version: vanta-cli version string (e.g., "1.0.5")

        Returns:
            Error message string if version is outdated or invalid, None if OK
        """
        try:
            # Parse version strings into tuples for comparison
            current = tuple(int(x) for x in version.split('.')[:3])
            minimum = tuple(int(x) for x in ValiConfig.VANTA_CLI_MINIMUM_VERSION.split('.')[:3])

            if current < minimum:
                return (f"Your vanta-cli version {version} is outdated and no longer supported. "
                        f"Please upgrade to vanta-cli >= {ValiConfig.VANTA_CLI_MINIMUM_VERSION}: "
                        f"pip install --upgrade git+https://github.com/taoshidev/vanta-cli.git")
        except (ValueError, AttributeError, IndexError):
            # Invalid version format - treat as error for security
            return (f"Invalid vanta-cli version format: {version}. "
                    f"Please reinstall vanta-cli: pip install --upgrade git+https://github.com/taoshidev/vanta-cli.git")
        return None



# This allows the module to be run directly for testing
if __name__ == "__main__":
    import argparse

    bt.logging.enable_info()

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run the REST API server with API key authentication")
    parser.add_argument("--api-keys", type=str, default="api_keys.json", help="Path to API keys JSON file")

    args = parser.parse_args()

    # Create test API keys file if it doesn't exist
    if not os.path.exists(args.api_keys):
        with open(args.api_keys, "w") as f:
            json.dump({"test_user": "test_key", "client": "abc"}, f)
        print(f"Created test API keys file at {args.api_keys}")

    print(f"REST server will run on {ValiConfig.REST_API_HOST}:{ValiConfig.REST_API_PORT} (hardcoded in ValiConfig)")

    # Create and run the server (host/port read from ValiConfig)
    server = ValidatorRestServer(
        api_keys_file=args.api_keys,
        metrics_interval_minutes=1
    )
    server.run()
