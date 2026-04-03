# Miner REST Server

This document outlines how to run the miner REST server to receive external signals and submit orders to the network.

> **Entity miners** (Hyperliquid-linked subaccounts) use an extended server with additional endpoints. See [entity_miner_rest_server.md](entity_miner_rest_server.md).

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

## Quick Start

The miner REST server starts automatically when you run your miner (no --serve flag needed). The server runs on port 8088 by default and provides the following endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/submit-order` | POST | Submit trading orders synchronously (returns validator feedback) |
| `/api/order-status/<uuid>` | GET | Query order processing status |
| `/api/health` | GET | Health check (no auth required) |

## API Endpoint Documentation

### Submit Order (Synchronous)

`POST /api/submit-order`

This endpoint receives trading signals from external systems and processes them **synchronously**, returning immediate feedback on validator acceptance/rejection. The miner will send the order to validators and wait for responses before returning (typically 20-60 seconds).

**Required Headers**:
```
'Content-Type': 'application/json',
'Authorization': 'xxxx'   # (string): Your API key as configured in `mining/miner_secrets.json`. Used for authentication.
```

**Request Body Fields**:

#### Required Fields

- `execution_type` (string): The execution type for the order. Must be one of:
  - `"MARKET"`: Execute immediately at current market price
  - `"LIMIT"`: Execute at a specific price when market reaches that level
  - `"BRACKET"`: Limit order with attached stop-loss and/or take-profit orders
  - `"LIMIT_CANCEL"`: Cancel an existing limit order
- `trade_pair` (string or object): The trading pair for the order. Can be either:
  - Trade pair ID string (e.g., `"BTCUSD"`, `"ETHUSD"`, `"EURUSD"`)
  - Trade pair object with `trade_pair_id` field (e.g., `{"trade_pair_id": "BTCUSD"}`)
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
- `verbose` (boolean): Optional. Defaults to `false`.
  - `false`: Return only Taoshi validator responses (concise, recommended for most users)
  - `true`: Return responses from all validators (detailed debugging information)

**Example Requests**:

#### Basic Market Order
```json
{
  "execution_type": "MARKET",
  "trade_pair": "BTCUSD",
  "order_type": "LONG",
  "leverage": 0.1
}
```

#### Market Order with Verbose Response
```json
{
  "execution_type": "BRACKET",
  "trade_pair": "ETHUSD",
  "order_type": "SHORT",
  "leverage": 0.2,
  "limit_price": 3500.00,
  "stop_loss": 3600.00,
  "take_profit": 3300.00
  "verbose": true
}
```

#### Limit Order
```json
{
  "trade_pair": "EURUSD",
  "order_type": "LONG",
  "value": 10000
  "execution_type": "LIMIT",
  "price": 50000.0
}
```

#### Order with USD Value
```json
{
  "execution_type": "MARKET",
  "trade_pair": "BTCUSD",
  "order_type": "LONG",
  "value": 10000
}
```

#### Cancel Limit Order
```json
{
  "execution_type": "LIMIT_CANCEL",
  "trade_pair": "BTCUSD",
  "order_type": "FLAT",
  "order_uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Entity Miner Subaccount Order
```json
{
  "execution_type": "MARKET",
  "trade_pair": "BTCUSD",
  "order_type": "LONG",
  "leverage": 0.1,
  "subaccount_id": 0
}
```

**Response**:

Success (200) - verbose=false (default):
```json
{
  "success": true,
  "order_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "validators_processed": 5,
  "validators_succeeded": 5,
  "high_trust_total": 5,
  "high_trust_succeeded": 5,
  "all_high_trust_succeeded": true,
  "created_orders": "{'trade_pair': 'BTCUSD', 'order_type': 'LONG', ...}",
  "error_messages": null,
  "processing_time": 23.456,
  "message": "Order successfully processed by Taoshi validator"
}
```

Success (200) - verbose=true:
```json
{
  "success": true,
  "order_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "validators_processed": 5,
  "validators_succeeded": 5,
  "high_trust_total": 5,
  "high_trust_succeeded": 5,
  "all_high_trust_succeeded": true,
  "created_orders": {
    "5FeNwZ...UZmo": "{'trade_pair': 'BTCUSD', 'order_type': 'LONG', ...}",
    "5GTNzN...bHLN": "{'trade_pair': 'BTCUSD', 'order_type': 'LONG', ...}",
    "...": "..."
  },
  "error_messages": {},
  "processing_time": 23.456,
  "message": "Order successfully processed by 5/5 high-trust validators"
}
```

Validation Error (400):
```json
{
  "success": false,
  "error": "Invalid request: must provide exactly one of: leverage, value, or quantity"
}
```

Processing Error (400):
```json
{
  "success": false,
  "order_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "validators_processed": 5,
  "validators_succeeded": 2,
  "high_trust_total": 5,
  "high_trust_succeeded": 2,
  "all_high_trust_succeeded": false,
  "created_orders": "{'trade_pair': 'BTCUSD', ...}",
  "error_messages": ["Validator error message"],
  "processing_time": 25.123,
  "message": "Order failed on Taoshi validator"
}
```

Authentication Error (401):
```json
{
  "error": "Unauthorized access"
}
```

Internal Error (500):
```json
{
  "success": false,
  "order_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "error": "Internal error processing order: ..."
}
```

**Response Fields**:

- `success`: Boolean indicating if the order was successfully processed by the Taoshi validator (verbose=false) or all high-trust validators (verbose=true)
- `order_uuid`: Unique identifier for this order
- `validators_processed`: Total number of validators that were contacted
- `validators_succeeded`: Number of validators that successfully accepted the order
- `high_trust_total`: Number of high-trust validators contacted
- `high_trust_succeeded`: Number of high-trust validators that accepted the order
- `all_high_trust_succeeded`: Boolean indicating if all high-trust validators succeeded
- `created_orders`: When verbose=false, contains the Taoshi validator's response; when verbose=true, contains a dictionary mapping all validator hotkeys to their responses
- `error_messages`: When verbose=false, contains Taoshi validator errors if any; when verbose=true, contains all validator errors
- `processing_time`: Total time in seconds for the request processing
- `message`: Human-readable description of the result

**Supported Trade Pairs**:

- **Crypto**: BTCUSD, ETHUSD, SOLUSD, XRPUSD, DOGEUSD, ADAUSD
- **Forex**: EURUSD, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD, and other major currency pairs

For the complete list of supported trade pairs and their current status, refer to `vali_objects/vali_config.py`.

**Notes**:

1. **Synchronous Processing**: This endpoint blocks for 20-60 seconds while the order is sent to validators and responses are collected. Use this for real-time trading systems that need immediate feedback.
2. **Verbose Flag**: Default (verbose=false) returns only Taoshi validator responses for concise feedback. Set verbose=true for debugging or to see all validator responses.
3. **Validator Trust**: The numeric metrics (validators_processed, validators_succeeded, etc.) always reflect actual processing regardless of the verbose flag.
4. **Entity Miners**: Use the `subaccount_id` field to route orders to specific subaccounts. The ID is used to construct a synthetic hotkey for position tracking.
5. **Regular Miners**: Omit the `subaccount_id` field entirely if you're not using entity miner subaccounts.

## Testing sending a signal

You can test a sample signal to ensure your server is running properly by running the
`sample_signal_request.py` script inside the `mining` directory.

1. Be sure to activate your venv
2. go to `vanta-network/mining/`
3. run `python sample_signal_request.py`
