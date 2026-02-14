import warnings
import json
import os
import traceback
import uuid
from flask import Flask, request, jsonify

import waitress
import requests
import bittensor as bt
from bittensor_wallet import Wallet

from miner_config import MinerConfig
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order_signal import Signal

print("=" * 60)
print("DEPRECATED: run_receive_signals_server.py is deprecated.")
print("Use the REST server running natively from miner.py instead.")
print("New endpoint: POST /api/submit-order")
print("=" * 60)
warnings.warn(
    "run_receive_signals_server.py is deprecated. Use MinerRestServer instead.",
    DeprecationWarning,
    stacklevel=2
)

app = Flask(__name__)

secrets_json_path = ValiConfig.BASE_DIR + "/mining/miner_secrets.json"

# Load secrets file
if os.path.exists(secrets_json_path):
    with open(secrets_json_path, "r") as file:
        data = file.read()
    MINER_SECRETS = json.loads(data)

    # Extract all secrets
    API_KEY = MINER_SECRETS["api_key"]
    WALLET_NAME = MINER_SECRETS.get("wallet_name")
    WALLET_HOTKEY = MINER_SECRETS.get("wallet_hotkey")
    WALLET_PASSWORD = MINER_SECRETS.get("wallet_password")
    VALIDATOR_URL = "http://34.187.154.219:48888"   # MINER_SECRETS.get("validator_url")

    # Validate required fields for subaccount creation
    if not all([WALLET_NAME, WALLET_HOTKEY, WALLET_PASSWORD, VALIDATOR_URL]):
        missing = []
        if not WALLET_NAME: missing.append("wallet_name")
        if not WALLET_HOTKEY: missing.append("wallet_hotkey")
        if not WALLET_PASSWORD: missing.append("wallet_password")
        if not VALIDATOR_URL: missing.append("validator_url")
        print(f"WARNING: Missing wallet config in miner_secrets.json: {', '.join(missing)}")
        print("Subaccount creation will be unavailable")
else:
    raise Exception(f"{secrets_json_path} not found", 404)


def get_miner_wallet():
    """
    Initialize Bittensor wallet from secrets configuration.

    Returns:
        Wallet: Initialized Bittensor wallet with coldkey and hotkey

    Raises:
        Exception: If wallet initialization fails
    """
    try:
        # Create wallet instance
        wallet = Wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)

        # # Unlock coldkey with password
        # coldkey = wallet.get_coldkey(password=WALLET_PASSWORD)

        return wallet
    except Exception as e:
        raise Exception(f"Failed to initialize wallet: {e}")


# Endpoint to handle JSON POST requests
@app.route("/api/receive-signal", methods=["POST"])
def handle_data():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Check if 'Authorization' header is provided
    data = request.json

    if data is None:
        return jsonify({"error": "Invalid message"}), 400

    print("received data:", data)

    if "api_key" in data:
        token = data["api_key"]
    else:
        return jsonify({"error": "Missing or invalid Authorization header"}), 401

    # Validate the API key
    if token != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401

    # Check if request is JSON
    if not request.json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        # ensure to fits rules for a Signal
        if isinstance(data['trade_pair'], dict):
            signal_trade_pair_str = data["trade_pair"]["trade_pair_id"]
        elif isinstance(data['trade_pair'], str):
            signal_trade_pair_str = data["trade_pair"]
        else:
            raise Exception("trade_pair must be a string or a dict")

        trade_pair = TradePair.from_trade_pair_id(signal_trade_pair_str)
        if trade_pair is None:
            return jsonify({"error": "Invalid trade pair"}), 401

        signal = Signal(
            trade_pair=trade_pair,
            order_type=OrderType.from_string(data["order_type"].upper()),
            leverage=float(data["leverage"]) if "leverage" in data else None,
            value=float(data["value"]) if "value" in data else None,
            quantity=float(data["quantity"]) if "quantity" in data else None,
            execution_type = ExecutionType.from_string(data.get("execution_type", "MARKET").upper()),
            limit_price=float(data["limit_price"]) if "limit_price" in data else None,
            stop_loss=float(data["stop_loss"]) if "stop_loss" in data else None,
            take_profit=float(data["take_profit"]) if "take_profit" in data else None,
            bracket_orders=data.get("bracket_orders")
        )
        # make miner received signals dir if doesnt exist
        ValiBkpUtils.make_dir(MinerConfig.get_miner_received_signals_dir())
        # store miner signal
        signal_file_uuid = data["order_uuid"] if "order_uuid" in data else str(uuid.uuid4())
        signal_path = os.path.join(MinerConfig.get_miner_received_signals_dir(), signal_file_uuid)

        # Add subaccount_id to signal data if provided
        signal_dict = dict(signal)
        if "subaccount_id" in data and data["subaccount_id"] is not None:
            signal_dict["subaccount_id"] = data["subaccount_id"]

        ValiBkpUtils.write_file(signal_path, signal_dict)
    except IOError as e:
        print(traceback.format_exc())
        return jsonify({"error": f"Error writing signal to file: {e}"}), 500
    except ValueError as e:
        print(traceback.format_exc())
        return jsonify({"error": f"improperly formatted signal received. {e}"}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"error storing signal on miner. {e}"}), 400

    return (
        jsonify({"message": "Signal {} received successfully".format(str(signal))}),
        200,
    )

@app.route("/api/create-subaccount", methods=["POST"])
def create_subaccount_endpoint():
    """
    Create a new subaccount by sending a signed request to the validator.

    Request JSON:
        {
            "api_key": "string",       # Required: API key
            "asset_class": "string",   # Required: "crypto" or "forex"
            "account_size": float      # Required: USD value
        }

    Response JSON (Success):
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

    Response JSON (Error):
        {
            "status": "error",
            "message": "..."
        }
    """
    # Validate JSON request
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.json
    if data is None:
        return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400

    print("Received create-subaccount request:", {k: v for k, v in data.items() if k != 'api_key'})

    # Authenticate with API key
    if "api_key" not in data or data["api_key"] != API_KEY:
        return jsonify({"status": "error", "message": "Invalid API key"}), 401

    # Validate required fields
    if "asset_class" not in data:
        return jsonify({"status": "error", "message": "Missing required field: asset_class"}), 400

    if "account_size" not in data:
        return jsonify({"status": "error", "message": "Missing required field: account_size"}), 400

    # Extract parameters
    asset_class = data["asset_class"]
    try:
        account_size = float(data["account_size"])
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "account_size must be a number"}), 400

    # Validate asset_class
    if asset_class not in ["crypto", "forex"]:
        return jsonify({
            "status": "error",
            "message": f"Invalid asset_class: {asset_class}. Must be 'crypto' or 'forex'"
        }), 400

    # Validate account_size
    if account_size <= 0:
        return jsonify({"status": "error", "message": "account_size must be positive"}), 400

    # Check wallet configuration
    if not all([WALLET_NAME, WALLET_HOTKEY, WALLET_PASSWORD, VALIDATOR_URL]):
        return jsonify({
            "status": "error",
            "message": "Wallet not configured. Check miner_secrets.json"
        }), 500

    try:
        # Initialize wallet and get coldkey/hotkey
        print("Initializing miner wallet...")
        wallet = get_miner_wallet()
        coldkey = wallet.get_coldkey(password=WALLET_PASSWORD)
        hotkey = wallet.hotkey

        print(f"Wallet initialized - Hotkey: {hotkey.ss58_address}, Coldkey: {coldkey.ss58_address}")

        # Build message for signature (MUST use sort_keys=True!)
        message_dict = {
            "account_size": account_size,
            "asset_class": asset_class,
            "entity_coldkey": coldkey.ss58_address,
            "entity_hotkey": hotkey.ss58_address
        }
        message = json.dumps(message_dict, sort_keys=True).encode('utf-8')

        # Sign message with coldkey
        signature = coldkey.sign(message).hex()
        print(f"Message signed with coldkey")

        # Build request payload for validator
        payload = {
            "entity_hotkey": hotkey.ss58_address,
            "entity_coldkey": coldkey.ss58_address,
            "account_size": account_size,
            "asset_class": asset_class,
            "signature": signature,
            "version": "2.0.0"
        }

        # Send request to validator
        validator_endpoint = f"{VALIDATOR_URL}/entity/create-subaccount"
        print(f"Sending request to validator: {validator_endpoint}")

        response = requests.post(
            validator_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        # Parse response
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            return jsonify({
                "status": "error",
                "message": f"Invalid JSON response from validator: {response.text}"
            }), 500

        # Check response status
        if response.status_code == 200:
            print(f"Subaccount created successfully: {response_data.get('subaccount', {}).get('synthetic_hotkey')}")
            return jsonify(response_data), 200
        else:
            error_msg = response_data.get("error", "Unknown error from validator")
            print(f"Validator returned error ({response.status_code}): {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg
            }), response.status_code

    except requests.exceptions.Timeout:
        error_msg = "Request to validator timed out"
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 504

    except requests.exceptions.ConnectionError as e:
        error_msg = f"Could not connect to validator: {e}"
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 503

    except Exception as e:
        error_msg = f"Error creating subaccount: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": error_msg}), 500

if __name__ == "__main__":
    waitress.serve(app, host="0.0.0.0", port=8088, connection_limit=1000)
    print('Successfully started run_receive_signals_server.')
