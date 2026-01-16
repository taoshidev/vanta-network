# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
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
import bittensor as bt
from typing import Optional
from flask import jsonify, request

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

        # Legacy file-based signal reception (backward compatible)
        self.app.route("/api/receive-signal", methods=["POST"])(self.receive_signal_legacy)

        # Entity miner subaccount creation
        self.app.route("/api/create-subaccount", methods=["POST"])(self.create_subaccount_endpoint)

        # Order status query
        self.app.route("/api/order-status/<order_uuid>", methods=["GET"])(self.order_status_endpoint)

        # Health check
        self.app.route("/api/health", methods=["GET"])(self.health_endpoint)

        print(f"[MINER-REST-INIT] 5 miner endpoints registered ✓")

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
            "leverage": 0.1,
            "execution_type": "MARKET" | "LIMIT",
            "price": 50000.0,  // Required for LIMIT orders
            "subaccount_id": "optional-subaccount-id"
        }

        Response (200 OK):
        {
            "success": true,
            "order_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "validators_processed": 5,
            "validators_succeeded": 5,
            "high_trust_total": 5,
            "high_trust_succeeded": 5,
            "created_orders": {...},
            "error_messages": {},
            "processing_time": 23.456,
            "message": "Order successfully processed by 5/5 high-trust validators"
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

            # Validate required fields
            required_fields = ['trade_pair', 'order_type', 'leverage']
            missing_fields = [field for field in required_fields if field not in signal_data]
            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': f"Invalid request: missing required fields: {', '.join(missing_fields)}"
                }), 400

            # Generate order_uuid if not provided
            order_uuid = signal_data.get('order_uuid', str(uuid.uuid4()))
            subaccount_id = signal_data.get('subaccount_id')

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
                subaccount_id=subaccount_id
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

    def receive_signal_legacy(self):
        """
        Legacy file-based signal reception (backward compatible).

        This endpoint writes the signal to disk for the miner's main loop to pick up.
        No synchronous feedback - returns 200 immediately.

        Request body (JSON):
        {
            "trade_pair": "BTC/USD",
            "order_type": "LONG" | "SHORT" | "FLAT",
            "leverage": 0.1,
            ... (same as submit_order_endpoint)
        }

        Response (200 OK):
        {
            "message": "Signal received and queued for processing",
            "signal_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        }
        """
        # 1. Validate API key
        api_key = self._get_api_key_safe()
        if not self.is_valid_api_key(api_key):
            return jsonify({'error': 'Unauthorized access'}), 401

        # 2. Parse request body
        try:
            signal_data = request.get_json()
            if not signal_data:
                return jsonify({'error': 'Invalid request: missing JSON body'}), 400

            # Generate signal UUID
            signal_uuid = str(uuid.uuid4())
            signal_data['order_uuid'] = signal_uuid

        except Exception as e:
            bt.logging.error(f"Error parsing request body: {e}")
            return jsonify({'error': f'Invalid request: {str(e)}'}), 400

        # 3. Write to disk for miner main loop to pick up
        try:
            signals_dir = MinerConfig.get_miner_received_signals_dir()
            os.makedirs(signals_dir, exist_ok=True)

            signal_file = os.path.join(signals_dir, signal_uuid)
            ValiBkpUtils.write_file(signal_file, json.dumps(signal_data))

            bt.logging.info(f"Legacy signal {signal_uuid} written to {signal_file}")

            return jsonify({
                'message': 'Signal received and queued for processing',
                'signal_uuid': signal_uuid
            }), 200

        except Exception as e:
            bt.logging.error(f"Error writing signal to disk: {e}")
            return jsonify({'error': f'Internal error: {str(e)}'}), 500

    def create_subaccount_endpoint(self):
        """
        Entity miner subaccount creation.

        Request body (JSON):
        {
            "entity_id": "entity-uuid",
            "subaccount_name": "optional-name"
        }

        Response (200 OK):
        {
            "success": true,
            "subaccount_id": "subaccount-uuid",
            "entity_id": "entity-uuid",
            "message": "Subaccount created successfully"
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
                return jsonify({'error': 'Invalid request: missing JSON body'}), 400

            entity_id = request_data.get('entity_id')
            if not entity_id:
                return jsonify({'error': 'Invalid request: missing entity_id'}), 400

            subaccount_name = request_data.get('subaccount_name')

        except Exception as e:
            bt.logging.error(f"Error parsing request body: {e}")
            return jsonify({'error': f'Invalid request: {str(e)}'}), 400

        # 3. Create subaccount (placeholder - actual implementation TBD)
        try:
            # TODO: Implement actual subaccount creation logic
            subaccount_id = str(uuid.uuid4())

            bt.logging.info(f"Created subaccount {subaccount_id} for entity {entity_id}")

            return jsonify({
                'success': True,
                'subaccount_id': subaccount_id,
                'entity_id': entity_id,
                'message': 'Subaccount created successfully'
            }), 200

        except Exception as e:
            bt.logging.error(f"Error creating subaccount: {e}")
            return jsonify({'error': f'Internal error: {str(e)}'}), 500

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
