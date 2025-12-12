# Running Signals Server

This document outlines how to run the signals server to receive external signals to your miner to automatically send over the network.

## Requirements

To run the signals server, you need to first setup your API key.

Setup your `miner_secrets.json` file - inside the repository, you’ll go to the mining directory and add a file called `mining/miner_secrets.json`. Inside the file you should provide a unique API key value for your server to receive signals.

The file should look like so:

```
# replace xxxx with your API key
{
  "api_key": "xxxx"
}
```

Once you have your secrets file setup, you should keep it to reference from other systems
to send in signals.

## Running the receive signals server

First, you'll want to make sure you have your venv setup. You can do this by following the
Installation portion of the README.

Once you have your venv setup you can run the signals server. We've setup a convenient
script that will ensure your server is running at all times named `run_receive_signals_server.sh`.

You can run it with the following command inside the vanta-network directory:

`sh run_receive_signals_server.sh`

## Stopping the receive signals server

If you want to stop your server at any time
you can search for its PID and kill it with the following commands.

`pkill -f run_receive_signals_server.sh` </br>
`pkill -f run_receive_signals_server.py`

## API Endpoint Documentation

### Receive Signal

`POST /api/receive-signal`

This endpoint receives trading signals from external systems and stores them locally for the miner to process and send to validators.

**Required Headers**:
```
Content-Type: application/json
```

**Request Body Fields**:

#### Required Fields

- `api_key` (string): Your API key as configured in `mining/miner_secrets.json`. Used for authentication.
- `execution_type` (string): The execution type for the order. Must be one of:
  - `"MARKET"`: Execute immediately at current market price
  - `"LIMIT"`: Execute at a specific price when market reaches that level
  - `"BRACKET"`: Limit order with attached stop-loss and/or take-profit orders
  - `"LIMIT_CANCEL"`: Cancel an existing limit order
- `trade_pair` (string or object): The trading pair for the order. Can be either:
  - Trade pair ID string (e.g., `"BTCUSD"`, `"ETHUSD"`, `"EURUSD"`)
  - Trade pair object with `trade_pair_id` field
- `order_type` (string): The direction of the order. Must be one of:
  - `"LONG"`: Open or increase a long position
  - `"SHORT"`: Open or increase a short position
  - `"FLAT"`: Close the current position

#### Order Size (Exactly ONE Required)

You must provide **exactly one** of the following fields to specify the order size:

- `leverage` (float): The portfolio weight for the position (e.g., `0.1` for 10% weight)
- `value` (float): The USD value of the order (e.g., `10000` for $10,000)
- `quantity` (float): The quantity in base asset units (lots, shares, coins, etc.)

#### Optional Fields for LIMIT and BRACKET Orders

- `limit_price` (float): **Required for LIMIT/BRACKET orders**. The price at which the limit order should fill.
- `stop_loss` (float): Optional for LIMIT orders. Creates a stop-loss bracket order upon fill.
- `take_profit` (float): Optional for LIMIT orders. Creates a take-profit bracket order upon fill.

#### Optional Fields for LIMIT_CANCEL Orders

- `order_uuid` (string): **Required for LIMIT_CANCEL orders**. The UUID of the limit order to cancel.

#### Optional Fields for Entity Miners

- `subaccount_id` (integer): The subaccount ID for entity miners (e.g., `0`, `1`, `2`). Only applicable for registered entity miners with subaccounts. Regular miners should omit this field.

**Example Requests**:

#### Market Order (Standard Miner)
```json
{
  "api_key": "your_api_key_here",
  "execution_type": "MARKET",
  "trade_pair": "BTCUSD",
  "order_type": "LONG",
  "leverage": 0.1
}
```

#### Limit Order with Brackets
```json
{
  "api_key": "your_api_key_here",
  "execution_type": "BRACKET",
  "trade_pair": "ETHUSD",
  "order_type": "SHORT",
  "leverage": 0.2,
  "limit_price": 3500.00,
  "stop_loss": 3600.00,
  "take_profit": 3300.00
}
```

#### Market Order with USD Value
```json
{
  "api_key": "your_api_key_here",
  "execution_type": "MARKET",
  "trade_pair": "EURUSD",
  "order_type": "LONG",
  "value": 10000
}
```

#### Close Position (Flat Order)
```json
{
  "api_key": "your_api_key_here",
  "execution_type": "MARKET",
  "trade_pair": "BTCUSD",
  "order_type": "FLAT",
  "leverage": 0
}
```

#### Cancel Limit Order
```json
{
  "api_key": "your_api_key_here",
  "execution_type": "LIMIT_CANCEL",
  "trade_pair": "BTCUSD",
  "order_type": "FLAT",
  "order_uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Entity Miner Subaccount Order
```json
{
  "api_key": "your_api_key_here",
  "execution_type": "MARKET",
  "trade_pair": "BTCUSD",
  "order_type": "LONG",
  "leverage": 0.1,
  "subaccount_id": 0
}
```

**Response**:

Success (200):
```json
{
  "message": "Signal {'trade_pair': ..., 'order_type': 'LONG', ...} received successfully"
}
```

Error (400):
```json
{
  "error": "Error message describing the issue"
}
```

Error (401):
```json
{
  "error": "Invalid API key"
}
```

**Supported Trade Pairs**:

- **Crypto**: BTCUSD, ETHUSD, SOLUSD, XRPUSD, DOGEUSD, ADAUSD
- **Forex**: EURUSD, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD, and other major currency pairs

For the complete list of supported trade pairs and their current status, refer to `vali_objects/vali_config.py`.

**Notes**:

1. Orders are stored locally and processed by the miner in the order they are received
2. The miner will send these orders to validators via the Bittensor network
3. Only one order per trade pair is processed at a time; duplicate signals for the same trade pair will overwrite previous unprocessed signals
4. For entity miners, the `subaccount_id` is used to construct a synthetic hotkey for position tracking
5. Regular miners should omit the `subaccount_id` field entirely

## Testing sending a signal

You can test a sample signal to ensure your server is running properly by running the
`sample_signal_request.py` script inside the `mining` directory.

1. Be sure to activate your venv
2. go to `vanta-network/mining/`
3. run `python sample_signal_request.py`
