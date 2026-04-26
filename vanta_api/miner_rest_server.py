"""
Miner REST Server - REST API for miners to receive order submissions.

This module provides a REST API server for miners that:
- Accepts order submissions from external traders via HTTP POST
- Provides synchronous feedback on validator acceptance/rejection
- Provides order status queries
- Follows miner's LOCAL mode pattern (in-process, no RPC)
- Direct method calls to PropNetOrderPlacer (no IPC, no separate process)

Note: Subaccount creation endpoints are in EntityMinerRestServer, not here.

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
from flask import jsonify, request

from vanta_api.base_rest_server import BaseRestServer
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from miner_config import MinerConfig
from vali_objects.vali_dataclasses.order_signal import Signal
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from vali_objects.enums.execution_type_enum import ExecutionType


class MinerRestServer(BaseRestServer):
    """
    Miner REST API server with synchronous order processing via direct calls.

    Follows miner's LOCAL mode pattern:
    - In-process (no spawn_process)
    - No RPC health monitoring
    - Direct method calls to PropNetOrderPlacer

    The server provides:
    - Synchronous order submission with validator feedback
    - Order status queries
    - Health check endpoint
    """

    def __init__(self, prop_net_order_placer, api_keys_file,
                 refresh_interval=15, metrics_interval_minutes=5,
                 flask_host=None, flask_port=None, slack_notifier=None,
                 service_name="MinerRestServer", **kwargs):
        """
        Initialize miner REST server with direct PropNetOrderPlacer reference.

        Args:
            prop_net_order_placer: Direct reference to PropNetOrderPlacer instance
            api_keys_file: Path to miner API keys file
            refresh_interval: How often to check for API key changes (seconds)
            metrics_interval_minutes: How often to log API metrics (minutes)
            flask_host: Host address for Flask server (default: "0.0.0.0")
            flask_port: Port for Flask server (default: 8088)
            slack_notifier: Optional SlackNotifier for notifications
            service_name: Service name for logging (default: "MinerRestServer")
        """
        # Store direct reference to order placer (no IPC, no RPC!)
        self.order_placer = prop_net_order_placer
        self.slack_notifier = slack_notifier

        print(f"[MINER-REST-INIT] Initializing {service_name}...")

        # Call BaseRestServer.__init__ (Flask only, no RPC)
        super().__init__(
            api_keys_file=api_keys_file,
            service_name=service_name,
            refresh_interval=refresh_interval,
            metrics_interval_minutes=metrics_interval_minutes,
            flask_host=flask_host or "0.0.0.0",
            flask_port=flask_port or 8088,
            **kwargs
        )

        print(f"[MINER-REST-INIT] {service_name} initialized on {self.flask_host}:{self.flask_port}")

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

        # Order status query
        self.app.route("/api/order-status/<order_uuid>", methods=["GET"])(self.order_status_endpoint)

        # Health check
        self.app.route("/api/health", methods=["GET"])(self.health_endpoint)

        print(f"[MINER-REST-INIT] 3 miner endpoints registered")

    # ============================================================================
    # ENDPOINT HANDLERS
    # ============================================================================

    def submit_order_endpoint(self):
        """
        Synchronous order submission with direct call to PropNetOrderPlacer.

        This runs in a Flask worker thread. Multiple concurrent requests
        are handled by Flask's thread pool (default 10 threads). Each thread
        submits async work to the shared event loop in PropNetOrderPlacer.

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
            "subaccount_id": "optional-subaccount-id"
        }

        Response (200 OK):
        {
            "success": true,
            "order_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "order_json": "...",
            "error_message": "",
            "processing_time": 1.23,
            "message": "Order successfully processed by Taoshi validator"
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

            # Generate order_uuid if not provided
            order_uuid = signal_data.get('order_uuid', str(uuid.uuid4()))

            bt.logging.debug(f"Processing order {order_uuid}")

        except Exception as e:
            bt.logging.error(f"Error parsing request body: {e}")
            return jsonify({'success': False, 'error': f'Invalid request: {str(e)}'}), 400

        # 2.5. Validate signal data
        try:
            # let Signal class model validators handle validation
            signal = Signal(
                trade_pair=signal_data.get('trade_pair'),
                order_type=OrderType.from_string(signal_data['order_type'].upper()) if 'order_type' in signal_data else None,
                leverage=float(signal_data['leverage']) if 'leverage' in signal_data else None,
                value=float(signal_data['value']) if 'value' in signal_data else None,
                quantity=float(signal_data['quantity']) if 'quantity' in signal_data else None,
                execution_type=ExecutionType.from_string(signal_data.get('execution_type', 'MARKET').upper()),
                limit_price=float(signal_data['limit_price']) if 'limit_price' in signal_data else None,
                stop_loss=float(signal_data['stop_loss']) if 'stop_loss' in signal_data else None,
                take_profit=float(signal_data['take_profit']) if 'take_profit' in signal_data else None,
                stop_price=float(signal_data['stop_price']) if 'stop_price' in signal_data else None,
                stop_condition=StopCondition.from_string(signal_data['stop_condition'].upper()) if 'stop_condition' in signal_data else None,
                trailing_stop=signal_data.get('trailing_stop'),
                bracket_orders=signal_data.get('bracket_orders')
            )

            bt.logging.debug(f"Signal validation passed for order {order_uuid}: {signal}")

        except ValueError as e:
            bt.logging.warning(f"Signal validation failed for order {order_uuid}")
            return jsonify({
                'success': False,
                'error': f'Invalid signal data: {str(e)}'
            }), 400
        except Exception as e:
            bt.logging.error(f"Unexpected error during signal validation for order {order_uuid}")
            return jsonify({
                'success': False,
                'error': f'Signal validation error: {str(e)}'
            }), 400

        # 3. Call order_placer.process_a_signal_for_rest() directly
        try:
            bt.logging.info(f"Processing order: {signal}...")

            result = self.order_placer.process_a_signal_for_rest(
                order_uuid=order_uuid,
                signal=signal,
                subaccount_id=signal_data.get('subaccount_id', None)
            )

            bt.logging.info(f"Order {order_uuid} processed in {result.get('processing_time', 0):.2f}s: success={result.get('success')}")

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
