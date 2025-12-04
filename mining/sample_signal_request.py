import sys

import requests
import json

from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair, TradePairCategory


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TradePair) or isinstance(obj, OrderType) or isinstance(obj, ExecutionType):
            return obj.__json__()  # Use the to_dict method to serialize TradePair

        if isinstance(obj, TradePairCategory):
            # Return the value of the Enum member, which is a string
            return obj.value

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    # Set the default URL endpoint
    default_base_url = 'http://127.0.0.1:8088'

    # Check if the URL argument is provided
    if len(sys.argv) == 2:
        # Extract the URL from the command line argument
        base_url = sys.argv[1]
    else:
        # Use the default URL if no argument is provided
        base_url = default_base_url

    print("base URL endpoint:", base_url)

    url = f'{base_url}/api/receive-signal'

    # Define the JSON data to be sent in the request
    data = {
        'execution_type': ExecutionType.MARKET,  # Execution types [MARKET, LIMIT, BRACKET, LIMIT_CANCEL]
        'trade_pair': TradePair.BTCUSD,
        'order_type': OrderType.LONG,

        # Order size
        'leverage': 0.1,    # leverage
        # 'value': 10_000,  # USD value
        # 'quantity': 0.1,  # base asset quantity (lots, shares, coins, etc.)

        # LIMIT/BRACKET Order fields
        # 'limit_price': 2000,      # Required for LIMIT orders; price at which order should fill
        # 'stop_loss': 5000,        # Optional for LIMIT orders; creates bracket order on fill
        # 'take_profit': 6000,      # Optional for LIMIT orders; creates bracket order on fill
        # 'order_uuid': "",         # Required for LIMIT_CANCEL; UUID of order to cancel

        'api_key': 'xxxx'
    }

    # Convert the Python dictionary to JSON format
    json_data = json.dumps(data, cls=CustomEncoder)
    print(json_data)
    # Set the headers to specify that the content is in JSON format
    headers = {
        'Content-Type': 'application/json',
    }

    # Make the POST request with JSON data
    response = requests.post(url, data=json_data, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("POST request was successful.")
        print("Response:")
        print(response.json())  # Print the response data
    else:
        print(response.__dict__)
        print("POST request failed with status code:", response.status_code)
