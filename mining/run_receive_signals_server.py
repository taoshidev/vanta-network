import json
import os
import traceback
import uuid
from flask import Flask, request, jsonify

import waitress

from miner_config import MinerConfig
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order_signal import Signal

app = Flask(__name__)

secrets_json_path = ValiConfig.BASE_DIR + "/mining/miner_secrets.json"
# Define your API key
if os.path.exists(secrets_json_path):
    with open(secrets_json_path, "r") as file:
        data = file.read()
    API_KEY = json.loads(data)["api_key"]
else:
    raise Exception(f"{secrets_json_path} not found", 404)


# Endpoint to handle JSON POST requests
@app.route("/api/receive-signal", methods=["POST"])
def handle_data():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Check if 'Authorization' header is provided
    data = request.json

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

        signal = Signal(trade_pair=TradePair.from_trade_pair_id(signal_trade_pair_str),
                        leverage=float(data["leverage"]) if data.get("leverage") is not None else None,
                        value=float(data["value"]) if data.get("value") is not None else None,
                        quantity=float(data["quantity"]) if data.get("quantity") is not None else None,
                        order_type=OrderType.from_string(data["order_type"].upper()))
        # make miner received signals dir if doesnt exist
        ValiBkpUtils.make_dir(MinerConfig.get_miner_received_signals_dir())
        # store miner signal
        signal_file_uuid = str(uuid.uuid4())
        signal_path = os.path.join(MinerConfig.get_miner_received_signals_dir(), signal_file_uuid)
        ValiBkpUtils.write_file(signal_path, dict(signal))
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

if __name__ == "__main__":
    waitress.serve(app, host="0.0.0.0", port=8088, connection_limit=1000)
    print('Successfully started run_receive_signals_server.')
