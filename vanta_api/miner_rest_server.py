"""
Miner REST Server - REST API for miners to receive order submissions.

This module provides a REST API server for miners that:
- Accepts order submissions from external traders via HTTP POST
- Provides synchronous feedback on validator acceptance/rejection
- Follows miner's LOCAL mode pattern (in-process, no RPC)
- Direct method calls to PropNetOrderPlacer (no IPC, no separate process)

Key differences from VantaRestServer:
- Only inherits BaseRestServer (no RPC health monitoring)
- Runs in-process with miner (not spawned as separate process)
- Direct reference to PropNetOrderPlacer for synchronous processing
- Simpler architecture suitable for miner use case
"""

import os
import json
import time
import uuid
import requests
import bittensor as bt
from typing import Optional
from flask import jsonify, request
from bittensor_wallet import Wallet

from vanta_api.base_rest_server import BaseRestServer
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from miner_config import MinerConfig


class MinerRestServer(BaseRestServer):
    """
    Miner REST API server with synchronous order processing via direct calls.

    Follows miner's LOCAL mode pattern:
    - In-process (no spawn_process)
    - No RPC health monitoring
    - Direct method calls to PropNetOrderPlacer

    The server provides:
    - Synchronous order submission with validator feedback
    - Legacy file-based signal reception (backward compatible)
    - Entity miner subaccount creation
    - Order status queries
    - Health check endpoint
    """

    def __init__(self, prop_net_order_placer, api_keys_file,
                 refresh_interval=15, metrics_interval_minutes=5,
                 flask_host=None, flask_port=None, **kwargs):
        """
        Initialize miner REST server with direct PropNetOrderPlacer reference.

        Args:
            prop_net_order_placer: Direct reference to PropNetOrderPlacer instance
            api_keys_file: Path to miner API keys file
            refresh_interval: How often to check for API key changes (seconds)
            metrics_interval_minutes: How often to log API metrics (minutes)
            flask_host: Host address for Flask server (default: "0.0.0.0")
            flask_port: Port for Flask server (default: 8088)
        """
        # Store direct reference to order placer (no IPC, no RPC!)
        self.order_placer = prop_net_order_placer

        print(f"[MINER-REST-INIT] Initializing MinerRestServer...")

        # Call BaseRestServer.__init__ (Flask only, no RPC)
        super().__init__(
            api_keys_file=api_keys_file,
            service_name="MinerRestServer",
            refresh_interval=refresh_interval,
            metrics_interval_minutes=metrics_interval_minutes,
            flask_host=flask_host or "0.0.0.0",
            flask_port=flask_port or 8088,
            **kwargs
        )

        print(f"[MINER-REST-INIT] MinerRestServer initialized on {self.flask_host}:{self.flask_port}")

    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (from BaseRestServer)
    # ============================================================================

    def _initialize_clients(self, **kwargs):
        """
        No clients needed - we have direct reference to order placer.

        Called by BaseRestServer.__init__() but miner doesn't need RPC clients.
        """
        print(f"[MINER-REST-INIT] No RPC clients needed (direct PropNetOrderPlacer reference)")

    def _register_routes(self):
        """Register miner-specific endpoints."""
        print(f"[MINER-REST-INIT] Registering miner endpoints...")

        # Synchronous order submission (new primary endpoint)
        self.app.route("/api/submit-order", methods=["POST"])(self.submit_order_endpoint)

        # Entity miner subaccount creation
        self.app.route("/api/create-subaccount", methods=["POST"])(self.create_subaccount_endpoint)

        # Order status query
        self.app.route("/api/order-status/<order_uuid>", methods=["GET"])(self.order_status_endpoint)

        # Health check
        self.app.route("/api/health", methods=["GET"])(self.health_endpoint)

        print(f"[MINER-REST-INIT] 4 miner endpoints registered ✓")

    # ============================================================================
    # ENDPOINT HANDLERS
    # ============================================================================

    def submit_order_endpoint(self):
        """
        Synchronous order submission with direct call to PropNetOrderPlacer.

        This runs in a Flask worker thread. Multiple concurrent requests
        are handled by Flask's thread pool (default 10 threads).

        Request body (JSON):
        {
            "order_uuid": "optional-uuid",  // Auto-generated if not provided
            "trade_pair": "BTC/USD",
            "order_type": "LONG" | "SHORT" | "FLAT",
            "leverage": 0.1,  // Exactly one of leverage, value, or quantity required
            "value": 1000.0,  // Exactly one of leverage, value, or quantity required
            "quantity": 0.5,  // Exactly one of leverage, value, or quantity required
            "execution_type": "MARKET" | "LIMIT",
            "price": 50000.0,  // Required for LIMIT orders
            "subaccount_id": "optional-subaccount-id",
            "verbose": false  // If true, return all validators; if false, return only MOTHERSHIP (default: false)
        }

        Response (200 OK):
        {
            "success": true,  // MOTHERSHIP validator success if verbose=false
            "order_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "validators_processed": 5,
            "validators_succeeded": 5,
            "high_trust_total": 5,
            "high_trust_succeeded": 5,
            "created_orders": {...},  // Filtered to MOTHERSHIP only if verbose=false
            "error_messages": {...},  // Filtered to MOTHERSHIP only if verbose=false
            "processing_time": 23.456,
            "message": "Order successfully processed"
        }

        Response (400 Bad Request):
        {
            "success": false,
            "error": "Invalid request: missing required field 'trade_pair'"
        }

        Response (401 Unauthorized):
        {
            "error": "Unauthorized access"
        }
        """
        # 1. Validate API key
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # 2. Parse and validate request body
        try:
            signal_data = request.get_json()
            if not signal_data:
                return jsonify({'success': False, 'error': 'Invalid request: missing JSON body'}), 400

            # Get execution type first (needed to determine required fields)
            execution_type = signal_data.get('execution_type', 'MARKET')

            # Validate required fields based on execution type
            # LIMIT_CANCEL only needs order_uuid (trade_pair and order_type are not required)
            if execution_type == 'LIMIT_CANCEL':
                required_fields = []
            elif execution_type == 'BRACKET':
                required_fields = ['trade_pair']
            else:
                required_fields = ['trade_pair', 'order_type']

            missing_fields = [field for field in required_fields if field not in signal_data]
            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': f"Invalid request: missing required fields: {', '.join(missing_fields)}"
                }), 400

            # Validate exactly one of leverage/value/quantity is present
            # Exception: BRACKET, LIMIT_CANCEL, LIMIT_EDIT don't require size fields
            if execution_type not in ['BRACKET', 'LIMIT_CANCEL', 'LIMIT_EDIT']:
                position_size_fields = ['leverage', 'value', 'quantity']
                provided_fields = [field for field in position_size_fields if field in signal_data]

                if len(provided_fields) != 1:
                    return jsonify({
                        'success': False,
                        'error': f"Invalid request: must provide exactly one of: leverage, value, or quantity, got: {', '.join(provided_fields)}"
                    }), 400

            # Generate order_uuid if not provided
            order_uuid = signal_data.get('order_uuid', str(uuid.uuid4()))
            subaccount_id = signal_data.get('subaccount_id')

            # Extract verbose flag (default to false)
            verbose = signal_data.get('verbose', False)
            if not isinstance(verbose, bool):
                # Handle string values for flexibility
                verbose = str(verbose).lower() in ('true', '1', 'yes')

            bt.logging.debug(f"Processing order {order_uuid} with verbose={verbose}")

        except Exception as e:
            bt.logging.error(f"Error parsing request body: {e}")
            return jsonify({'success': False, 'error': f'Invalid request: {str(e)}'}), 400

        # 3. Call order_placer.process_a_signal_for_rest() directly (blocks 20-60s)
        try:
            bt.logging.info(f"Processing order {order_uuid} synchronously...")
            start_time = time.time()

            result = self.order_placer.process_a_signal_for_rest(
                order_uuid=order_uuid,
                signal_data=signal_data,
                subaccount_id=subaccount_id,
                verbose=verbose
            )

            elapsed = time.time() - start_time
            bt.logging.info(f"Order {order_uuid} processed in {elapsed:.2f}s: success={result.get('success')}")

            # 4. Return formatted response
            status_code = 200 if result.get('success') else 400
            return jsonify(result), status_code

        except Exception as e:
            bt.logging.error(f"Error processing order {order_uuid}: {e}")
            return jsonify({
                'success': False,
                'order_uuid': order_uuid,
                'error': f'Internal error processing order: {str(e)}'
            }), 500

    def create_subaccount_endpoint(self):
        """
        Entity miner subaccount creation.

        Request body (JSON):
        {
            "asset_class": "crypto" | "forex",  // Required
            "account_size": float                // Required, must be > 0
        }

        Response (200 OK):
        {
            "status": "success",
            "message": "...",
            "subaccount": {
                "subaccount_id": 0,
                "subaccount_uuid": "uuid-string",
                "synthetic_hotkey": "5xxx_0",
                "account_size": 10000.0,
                "asset_class": "crypto"
            }
        }
        """
        # 1. Validate API key
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # 2. Parse request body
        try:
            request_data = request.get_json()
            if not request_data:
                return jsonify({'status': 'error', 'message': 'Invalid request: missing JSON body'}), 400

            # Validate required fields
            if "asset_class" not in request_data:
                return jsonify({'status': 'error', 'message': 'Missing required field: asset_class'}), 400

            if "account_size" not in request_data:
                return jsonify({'status': 'error', 'message': 'Missing required field: account_size'}), 400

            asset_class = request_data["asset_class"]

            # Type conversion with error handling
            try:
                account_size = float(request_data["account_size"])
            except (ValueError, TypeError):
                return jsonify({'status': 'error', 'message': 'account_size must be a number'}), 400

            # Asset class validation
            if asset_class not in ["crypto", "forex"]:
                return jsonify({
                    'status': 'error',
                    'message': f"Invalid asset_class: {asset_class}. Must be 'crypto' or 'forex'"
                }), 400

            # Account size validation
            if account_size <= 0:
                return jsonify({'status': 'error', 'message': 'account_size must be positive'}), 400

        except Exception as e:
            bt.logging.error(f"Error parsing request body: {e}")
            return jsonify({'status': 'error', 'message': f'Invalid request: {str(e)}'}), 400

        # 3. Load wallet secrets
        try:
            secrets_file = MinerConfig.get_secrets_file_path()
            with open(secrets_file, 'r') as f:
                secrets = json.load(f)

            wallet_name = secrets.get('wallet_name')
            wallet_hotkey = secrets.get('wallet_hotkey')
            wallet_password = secrets.get('wallet_password')
            validator_url = secrets.get('validator_url')

            if not all([wallet_name, wallet_hotkey, wallet_password, validator_url]):
                return jsonify({
                    'status': 'error',
                    'message': 'Missing wallet configuration in secrets file'
                }), 500

        except Exception as e:
            bt.logging.error(f"Error loading wallet secrets: {e}")
            return jsonify({'status': 'error', 'message': 'Failed to load wallet configuration'}), 500

        # 4. Initialize wallet and sign message
        try:
            wallet = Wallet(name=wallet_name, hotkey=wallet_hotkey)
            coldkey = wallet.get_coldkey(password=wallet_password)
            hotkey = wallet.hotkey

            # Build message dict - CRITICAL: must use sort_keys=True for deterministic ordering
            message_dict = {
                "account_size": account_size,
                "asset_class": asset_class,
                "entity_coldkey": coldkey.ss58_address,
                "entity_hotkey": hotkey.ss58_address
            }
            message = json.dumps(message_dict, sort_keys=True).encode('utf-8')

            # Sign with coldkey
            signature = coldkey.sign(message).hex()

        except Exception as e:
            bt.logging.error(f"Error signing message: {e}")
            return jsonify({'status': 'error', 'message': f'Wallet error: {str(e)}'}), 500

        # 5. Send request to validator
        try:
            payload = {
                "entity_hotkey": hotkey.ss58_address,
                "entity_coldkey": coldkey.ss58_address,
                "account_size": account_size,
                "asset_class": asset_class,
                "signature": signature,
                "version": "2.0.0"
            }

            response = requests.post(
                f"{validator_url}/entity/create-subaccount",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            # Parse response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid JSON response from validator'
                }), 500

            # Return validator response
            if response.status_code == 200:
                return jsonify(response_data), 200
            else:
                error_message = response_data.get('message', 'Unknown error from validator')
                return jsonify({
                    'status': 'error',
                    'message': error_message
                }), response.status_code

        except requests.exceptions.Timeout:
            return jsonify({'status': 'error', 'message': 'Request to validator timed out'}), 504

        except requests.exceptions.ConnectionError:
            return jsonify({'status': 'error', 'message': 'Could not connect to validator'}), 503

        except Exception as e:
            bt.logging.error(f"Error communicating with validator: {e}")
            return jsonify({'status': 'error', 'message': f'Validator communication error: {str(e)}'}), 500

    def order_status_endpoint(self, order_uuid):
        """
        Query order status by UUID.

        Checks processed_signals/ and failed_signals/ directories for order details.

        Response (200 OK):
        {
            "order_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "status": "completed" | "failed" | "not_found",
            "details": {...}  // Signal data if found
        }
        """
        # 1. Validate API key
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # 2. Search for order in processed_signals/ and failed_signals/
        try:
            processed_dir = MinerConfig.get_miner_processed_signals_dir()
            failed_dir = MinerConfig.get_miner_failed_signals_dir()

            # Check processed_signals/
            processed_file = os.path.join(processed_dir, order_uuid)
            if os.path.exists(processed_file):
                signal_data = ValiBkpUtils.get_file(processed_file)
                return jsonify({
                    'order_uuid': order_uuid,
                    'status': 'completed',
                    'details': json.loads(signal_data)
                }), 200

            # Check failed_signals/
            failed_file = os.path.join(failed_dir, order_uuid)
            if os.path.exists(failed_file):
                signal_data = ValiBkpUtils.get_file(failed_file)
                return jsonify({
                    'order_uuid': order_uuid,
                    'status': 'failed',
                    'details': json.loads(signal_data)
                }), 200

            # Not found
            return jsonify({
                'order_uuid': order_uuid,
                'status': 'not_found',
                'message': 'Order not found in processed or failed signals'
            }), 404

        except Exception as e:
            bt.logging.error(f"Error querying order status: {e}")
            return jsonify({'error': f'Internal error: {str(e)}'}), 500

    def health_endpoint(self):
        """
        Server health check.

        Response (200 OK):
        {
            "status": "healthy",
            "service": "MinerRestServer",
            "timestamp": 1234567890.123
        }
        """
        return jsonify({
            'status': 'healthy',
            'service': 'MinerRestServer',
            'timestamp': time.time()
        }), 200
