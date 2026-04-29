# Vanta API and Websocket Data

## Overview

This repository provides a comprehensive API server and client for the [Vanta Network - Bittensor subnet 8](https://github.com/taoshidev/vanta-network/blob/main/docs/validator.md). It features both REST and WebSocket endpoints that enable real-time access to trading data, positions, statistics, and other critical information.

The REST API server is designed for validators to efficiently provide Vanta data to consumers, with support for different data freshness tiers, real-time updates, and secure authentication.

The Websocket server allows for real-time streaming of trading data, enabling clients to receive updates as they happen.

The Websocket client is built to interact with the Websocket server, allowing for real-time data streaming and processing. 

## Features

- **Dual API Support**:
  - REST API for standard HTTP requests
  - WebSocket API for real-time data streaming
- **Secure Authentication**:
  - Dynamic API key management with automatic refresh
  - Token-based authentication for all endpoints
- **Data Tiering**:
  - Support for different data freshness tiers (0%, 30%, 50%, 100%)
  - Customizable pricing models for data access
- **Performance Optimizations**:
  - GZIP compression for large payloads
  - Batch message processing for WebSocket communications
  - Efficient sequence tracking for message reliability
- **Fault Tolerance**:
  - Automatic reconnection with exponential backoff
  - Process isolation for stability

## Architecture

The system consists of three main components:

1. **API Manager**: Coordinates the REST and WebSocket services, handles process management, and maintains shared state.
2. **REST Server**: Provides HTTP endpoints for querying historical data and statistics.
3. **WebSocket Server**: Enables real-time data streaming with sequence tracking.

## Configuration

### API Keys Management

API keys are stored in a JSON file with the following format:

```json
    {
      "user_id": {
        "key": "the_api_key_string",
        "tier": <0,30,50,100>
      },
      ...
    }

    Where tier values represent access levels:
    - 0: Basic access (24-hour lagged data, no real-time access)
    - 30: Standard access (30% real-time data)
    - 50: Enhanced access (50% real-time data)
    - 100: Premium access (100% data freshness + WebSocket support)
```

By default, the system looks for this file at the relative path `vanta_api/api_keys.json`. The API keys are automatically refreshed from disk, allowing you to add or remove keys without restarting the server.


The [Request Network](https://request.taoshi.io/) is a Taoshi product which serves subnet data while handling security, rate limiting, data customization, and provides a polished customer-facing and validator setup UI. Running this repo's APIManager is a prerequisite to serving data on the Request Network

For end users who want to access Vanta data, you will need a Request Network API key. If you have any issues or questions, please reach out to the Taoshi team on Discord.

### Command Line Options

```
standalone usage: python api_manager.py [-h] [--serve] [--api-host API_HOST] [--api-rest-port API_REST_PORT]
                     [--api-ws-port API_WS_PORT] [--netuid NETUID] ...

optional arguments:
  -h, --help                       Show this help message and exit
  --serve                          Start the API server for REST and WebSocket endpoints
  --api-host API_HOST              Host address for the API server (default: 127.0.0.1)
  --api-rest-port API_REST_PORT    Port for the REST API server (default: 48888)
  --api-ws-port API_WS_PORT        Port for the WebSocket server (default: 8765)
  
 Provide the same args when running as a part of your SN8 validator.
```

### Making Server Accessible

By default, the server binds to `127.0.0.1` which only allows local requests. To allow access from any IP address:

```bash
python main.py --serve --api-host 0.0.0.0
```

## REST API Endpoints

### All Miners Positions 

`GET /miner-positions`

This endpoint retrieves the positions of all miners, optionally filtered by a specified data freshness tier.

**Tier Parameter**:
- `tier` (optional): Specifies the data freshness tier ['0', '30', '50', '100']
- 0: 100% of positions show data with a 24-hour delay.
- 30: 30% of positions show real-time data, 70% show data with a 24-hour delay. (superset of tier 0)
- 50: 50% of positions show real-time data, 50% show data with a 24-hour delay. (superset of tier 30)
- 100: 100% of positions show real-time data. Equivalent to not providing a tier. (superset of tier 50)

**Response**:
A JSON file containing all miner positions at the specified tier.

e.x:

```json
{
    "5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ": {
        "all_time_returns": 1.046956054826957,
        "n_positions": 15,
        "percentage_profitable": 0.8666666666666667,
        "positions": [
            {
                "average_entry_price": 0.59559,
                "close_ms": 1714156813363,
                "current_return": 1.0002165919508386,
                "initial_entry_price": 0.59559,
                "is_closed_position": true,
                "miner_hotkey": "5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ",
                "net_leverage": 0.0,
                "open_ms": 1714139478990,
                "orders": [
                    {
                        "leverage": -0.1,
                        "order_type": "SHORT",
                        "order_uuid": "18ca3cdf-b785-4f88-90a9-d2c06e8653b1",
                        "price": 0.59559,
                        "price_sources": [],
                        "processed_ms": 1714139478990
                    },
                    {
                        "leverage": 0.0,
                        "order_type": "FLAT",
                        "order_uuid": "c902c428-fcfb-43ca-ab79-117c957dbbfa",
                        "price": 0.5943,
                        "price_sources": [],
                        "processed_ms": 1714156813363
                    }
                ],
                "position_type": "FLAT",
                "position_uuid": "1f3f427f-6cbe-497c-af11-2fbef2ca3c10",
                "return_at_close": 1.0002095904346948,
                "trade_pair": [
                    "NZDUSD",
                    "NZD/USD",
                    7e-05,
                    0.001,
                    500
                ]
            },
...
```

Hotkeys are mapped to a data dict. The data dict contains positions which contain orders.

**Explanation of Schema:**
* miner_hotkey: A unique identifier for a miner. This is the same as the Bittensor metagraph hotkey value.
* all_time_returns: The miner's total return on investment across all positions over all time.
* n_positions: The number of positions held by the miner.
* percentage_profitable: The proportion of the miner's positions that have been profitable.
* positions: A list of individual trading positions held by the miner.
* average_entry_price: The average price at which the miner entered the position.
* current_return: The current return on the position with no fees.
* return_at_close: The return on the position at the time it was closed with all fees applied.
* initial_entry_price: The price at which the position was first opened.
* is_closed_position: Indicates if the position is closed.
* net_leverage: The leverage currently used in the position. 0 if the position is closed.
* open_ms: The timestamp (in milliseconds) when the position was opened.
* close_ms: The timestamp (in milliseconds) when the position was closed. 0 if not closed.
* orders: A list of orders executed within the position.
    - leverage: The leverage applied to the order.
    - order_type: The type of order (e.g., SHORT, LONG, FLAT).
    - order_uuid: A unique identifier for the order. Must be unique across all miners
    - price: The price at which the order was executed (filled).
    - price_sources: Used for debugging. Info about the price sources used to determine the price. At the time of this writing the sources are Polygon and Tiingo.
    - src: 0 if the order was placed by the miner, 1 if it was an automatically generated FLAT order due to elimination, 2 for automatically generated FLAT order to due trade pair deprecation. 
    - bid: The bid price at the time of order execution.
    - ask: The ask price at the time of order execution.
* processed_ms: The timestamp (in milliseconds) when the order was processed.
* position_type: The current status of the position (e.g., FLAT, SHORT, LONG).
* position_uuid: A unique identifier for the position.
* trade_pair: Information about the trade pair (e.g., currency pair BTCUSD).

### Single Miner Positions

`GET /miner-positions/<minerid>`

Returns position data for a specific miner identified by their hotkey.

### Miner Hotkeys

`GET /miner-hotkeys`

Returns all the hotkeys as seen in the metagraph from the validator's perspective.

### Allowed Trade Pairs

`GET /trade-pairs`

Returns the currently allowed trading pairs and their maximum leverage limits. This endpoint excludes blocked pairs (e.g., indices, commodities, deprecated forex pairs) and unsupported pairs (e.g., TAOUSD). Use it to discover which `trade_pair_id` values are valid when placing orders and what leverage constraints apply per pair.

**Authentication:** None required. Public endpoint.

**Response:**
```json
{
  "allowed_trade_pairs": [
    {
      "trade_pair_id": "BTCUSD",
      "trade_pair": "BTC/USD",
      "trade_pair_category": "CRYPTO",
      "max_leverage": 0.5
    },
    {
      "trade_pair_id": "ETHUSD",
      "trade_pair": "ETH/USD",
      "trade_pair_category": "CRYPTO",
      "max_leverage": 0.5
    },
    {
      "trade_pair_id": "EURUSD",
      "trade_pair": "EUR/USD",
      "trade_pair_category": "FOREX",
      "max_leverage": 5
    }
  ],
  "allowed_trade_pair_ids": ["BTCUSD", "ETHUSD", "EURUSD", "..."],
  "total_trade_pairs": 45,
  "timestamp": 1749234567890
}
```

**Response fields:**
- `allowed_trade_pairs`: Array of objects, each with:
  - `trade_pair_id`: Identifier to use in order requests (e.g., `"BTCUSD"`)
  - `trade_pair`: Display format (e.g., `"BTC/USD"`)
  - `trade_pair_category`: Asset class (`CRYPTO`, `FOREX`, `EQUITIES`, `INDICES`)
  - `max_leverage`: Maximum allowed leverage for this pair
- `allowed_trade_pair_ids`: Flat list of valid `trade_pair_id` values
- `total_trade_pairs`: Count of allowed pairs
- `timestamp`: Response timestamp in milliseconds

**Example:**
```bash
curl http://localhost:48888/trade-pairs
```

**Error responses:**
```json
// 500 Internal Server Error
{
  "error": "Internal server error retrieving allowed trade pairs"
}
```

### All Miners Statistics 

`GET /statistics`

Returns statistics relevant for scoring miners, consumed by the Taoshi dashboard.

**Parameters**:
- `checkpoints` (optional): Include checkpoint data (default: "true")

### Single Miner Statistics 

`GET /statistics/<minerid>/`

Returns statistics for a specific miner.

### Eliminations

`GET /eliminations`

Returns information about which miners have been eliminated and why. Note: deregistered miners are not shown in this list.
More information can be found here: https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/miner.md#miner

e.x:
```json
 "eliminations": [
    {
      "dd": 0.0,
      "elimination_initiated_time_ms": 1711184954891,
      "hotkey": "5Dk2u35LRYEi9SC5cWamtzRkdXJJDLES7gABuey6cJ6t1ajK",
      "reason": "LIQUIDATED"
    },
    {
      "elimination_initiated_time_ms": 1711204151504,
      "hotkey": "5G1iDH2gvdAyrpUD4QfZXATvGEtstRBiXWieRDeaDPRfPEcU",
      "reason": "PLAGIARISM"
    },
    {
```

### Validator Checkpoint 

`GET /validator-checkpoint`

Everything required for a validator to restore it's state when starting for the first time. This includes all miner positions as well as derived data such as perf ledgers, challenge period data, and eliminations.

Perf ledgers are portfolio-level only (no per-trade-pair ledgers). Each hotkey maps directly to a single `PerfLedger` object.

Perf Ledger schema
```json
"perf_ledgers": {
    "5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ": {
      "initialization_time_ms": 1714140000000,
      "max_return": 1.0423,
      "target_cp_duration_ms": 21600000,
      "target_ledger_window_ms": 15552000000,
      "last_known_prices": {
        "BTCUSD": [67000.0, 1714182650595]
      },
      "cps": [
        {
          "last_update_ms": 1714161050595,
          "prev_portfolio_ret": 0.9999907311080851,
          "prev_portfolio_realized_pnl": -12.34,
          "prev_portfolio_unrealized_pnl": 5.67,
          "accum_ms": 21600000,
          "open_ms": 21599768,
          "n_updates": 17213,
          "gain": 0.12586433994869853,
          "loss": -0.12587360888356938,
          "mdd": 0.9998,
          "mpv": 1.0001,
          "realized_pnl": -12.34,
          "unrealized_pnl": 5.67,
          "equity_ret": 0.9999,
          "cumulative_fees_usd": 8.21
        }
      ]
    }
}
```

**PerfCheckpoint fields:**
| Field | Type | Description |
|-------|------|-------------|
| `last_update_ms` | int | Timestamp of the last price update in this checkpoint |
| `prev_portfolio_ret` | float | Portfolio return multiplier at checkpoint end (1.0 = break-even) |
| `prev_portfolio_realized_pnl` | float | Cumulative realized PnL in USD up to this checkpoint |
| `prev_portfolio_unrealized_pnl` | float | Unrealized PnL snapshot in USD at checkpoint end |
| `accum_ms` | int | Total accumulated time covered by this checkpoint (ms) |
| `open_ms` | int | Time with at least one open position during checkpoint (ms) |
| `n_updates` | int | Number of price ticks processed in this checkpoint |
| `gain` | float | Sum of positive return contributions |
| `loss` | float | Sum of negative return contributions |
| `mdd` | float | Maximum drawdown ratio within this checkpoint (1.0 = no drawdown) |
| `mpv` | float | Maximum portfolio value achieved in this checkpoint |
| `realized_pnl` | float | Realized PnL in USD during this checkpoint period only (not cumulative) |
| `unrealized_pnl` | float | Unrealized PnL snapshot in USD at end of checkpoint |
| `equity_ret` | float | `(account_size + cumulative_realized_pnl - cumulative_fees_usd + unrealized_pnl) / account_size` |
| `cumulative_fees_usd` | float | Running total of all fees paid up to and including this checkpoint |

Perf ledgers are built based off realtime price data and are consumed in the scoring logic. More info in the Vanta repo.

### Perf Ledger (Single Miner)

`GET /perf-ledger/<minerid>`

Returns the portfolio-level performance ledger for a single miner. Requires a valid API key.

**Parameters:**
- `minerid`: The miner's hotkey (SS58 address)

**Response:**
```json
{
  "<hotkey>": {
    "initialization_time_ms": 1714140000000,
    "max_return": 1.0423,
    "target_cp_duration_ms": 21600000,
    "target_ledger_window_ms": 15552000000,
    "last_known_prices": {
      "BTCUSD": [67000.0, 1714182650595]
    },
    "cps": [
      {
        "last_update_ms": 1714161050595,
        "prev_portfolio_ret": 0.9999907311080851,
        "prev_portfolio_realized_pnl": -12.34,
        "prev_portfolio_unrealized_pnl": 5.67,
        "accum_ms": 21600000,
        "open_ms": 21599768,
        "n_updates": 17213,
        "gain": 0.12586433994869853,
        "loss": -0.12587360888356938,
        "mdd": 0.9998,
        "mpv": 1.0001,
        "realized_pnl": -12.34,
        "unrealized_pnl": 5.67,
        "equity_ret": 0.9999,
        "cumulative_fees_usd": 8.21
      }
    ]
  }
}
```

**Error responses:**
- `404` — miner not found or no ledger data available
- `503` — perf ledger service not available

> **Breaking change from V2 bundle format:** Previously the response was nested as `{ hotkey: { "portfolio": { "cps": [...] }, "BTCUSD": { "cps": [...] }, ... } }`. The ledger is now portfolio-only and returned flat as `{ hotkey: { "cps": [...], ... } }`.


## Collateral Management

The API includes comprehensive collateral management endpoints for miners to deposit, withdraw, and check their collateral balances. These endpoints interact with the collateral smart contract system.

### Deposit Collateral

`POST /collateral/deposit`

Process a collateral deposit with encoded extrinsic data.

**Request Body:**
```json
{
  "extrinsic": "0x1234567890abcdef..."
}
```

**Response:**
```json
{
  "successfully_processed": true,
  "error_message": ""
}
```

**Parameters:**
- `extrinsic_data` (string): Hex-encoded signed extrinsic for stake transfer

### Withdraw Collateral

`POST /collateral/withdraw`

Process a collateral withdrawal request.

**Request Body:**
```json
{
  "amount": 5.0,
  "miner_coldkey": "5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2",
  "miner_hotkey": "5FrLxJsyJ5x9n2rmxFwosFraxFCKcXZDngEP9H7qjkKgHLcK",
  "nonce": "0x1234567890abcdef...",
  "timestamp": 1751409821967,
  "signature": "0x1234567890abcdef..."
}
```

**Response:**
```json
{
  "successfully_processed": true,
  "error_message": "",
  "returned_amount": 5.0,
  "returned_to": "5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2"
}
```

**Parameters:**
- `amount` (float): Amount to withdraw in theta tokens
- `miner_coldkey` (string): Miner's coldkey SS58 address
- `miner_hotkey` (string): Miner's hotkey SS58 address
- `nonce` (string): Request nonce
- `timestamp` (int): Request timestamp
- `signature` (string): Request signature

### Get Collateral Balance

`GET /collateral/balance/<miner_address>`

Retrieve a miner's current collateral balance.

**Response:**
```json
{
  "miner_address": "5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2",
  "balance_theta": 15.5
}
```

## Asset Class Selection

The asset class selection endpoint allows miners to permanently select their asset class (forex, crypto). This selection cannot be undone.

### Asset Selection

`POST /asset-selection`

Process an asset class selection.

**Request Body:**
```json
{
  "asset_selection": "forex",
  "miner_coldkey": "5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2",
  "miner_hotkey": "5FrLxJsyJ5x9n2rmxFwosFraxFCKcXZDngEP9H7qjkKgHLcK",
  "signature": "0x1234567890abcdef..."
}
```

**Response:**
```json
{
  "successfully_processed": true,
  "success_message": "Miner 5FrLxJsyJ5x9n2rmxFwosFraxFCKcXZDngEP9H7qjkKgHLcK successfully selected asset class: forex",
  "error_message": ""
}
```

**Parameters:**
- `asset_selection` (string): Miner's asset class selection
- `miner_coldkey` (string): Miner's coldkey SS58 address
- `miner_hotkey` (string): Miner's hotkey SS58 address
- `signature` (string): Request signature

## Limit Orders

The limit orders endpoint allows querying a miner's limit and bracket orders with optional status filtering.

### Get Limit Orders

`GET /limit-orders/<minerid>`

Retrieve limit orders for a specific miner. Supports optional filtering by order status.

**Query Parameters:**
- `status` (optional): Comma-separated list of status values to filter by. Valid values: `unfilled`, `filled`, `cancelled`

**Response Format:**

Without `status` parameter - returns a flat list (backward compatible):
```json
[
  {
    "trade_pair": ["BTCUSD", "BTC/USD"],
    "order_type": "OrderType.LONG",
    "processed_ms": 1702345678901,
    "limit_price": 95000.0,
    "price": 0.0,
    "leverage": 0.1,
    "value": null,
    "quantity": null,
    "src": 5,
    "execution_type": "LIMIT",
    "order_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "stop_loss": 90000.0,
    "take_profit": 100000.0
  }
]
```

With `status` parameter - returns orders grouped by status:
```json
{
  "unfilled": [
    {
      "trade_pair": ["BTCUSD", "BTC/USD"],
      "order_type": "OrderType.LONG",
      "processed_ms": 1702345678901,
      "limit_price": 95000.0,
      "price": 0.0,
      "leverage": 0.1,
      "value": null,
      "quantity": null,
      "src": 5,
      "execution_type": "LIMIT",
      "order_uuid": "550e8400-e29b-41d4-a716-446655440000",
      "stop_loss": 90000.0,
      "take_profit": 100000.0
    }
  ],
  "filled": [],
  "cancelled": []
}
```

**Examples:**

```bash
# Get all unfilled limit orders (default behavior)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:48888/limit-orders/5GhDr...

# Get only unfilled orders (grouped response)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:48888/limit-orders/5GhDr...?status=unfilled"

# Get filled and cancelled orders
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:48888/limit-orders/5GhDr...?status=filled,cancelled"

# Get all order statuses
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:48888/limit-orders/5GhDr...?status=unfilled,filled,cancelled"
```

**Response Fields:**
- `trade_pair`: Array of [trade_pair_id, display_name]
- `order_type`: Order direction (LONG, SHORT)
- `processed_ms`: Timestamp when order was created/processed
- `limit_price`: Target price for limit orders (null for bracket orders)
- `price`: Fill price (0 if unfilled)
- `leverage`: Position leverage
- `value`: Order value in USD (if specified)
- `quantity`: Order quantity (if specified)
- `src`: Order source enum (5=LIMIT_UNFILLED, 6=LIMIT_FILLED, 7=LIMIT_CANCELLED, 8=BRACKET_UNFILLED, 9=BRACKET_FILLED, 10=BRACKET_CANCELLED)
- `execution_type`: LIMIT or BRACKET
- `order_uuid`: Unique order identifier
- `stop_loss`: Stop-loss price (for bracket orders or limit orders with SL/TP)
- `take_profit`: Take-profit price (for bracket orders or limit orders with SL/TP)

**Error Responses:**

```json
// Invalid status value
{
  "error": "Invalid status values: {'invalid'}. Valid values are: unfilled, filled, cancelled"
}

// No orders found
{
  "error": "No limit orders found for miner 5GhDr..."
}
```

## All Orders (Unified)

The orders endpoint provides a unified view of all orders for a miner, combining unfilled/cancelled limit orders with filled orders from positions.

**Access Requirements:**
- Requires **tier 100 API access**

### Get All Orders

`GET /orders/<minerid>`

Retrieve all orders for a specific miner, grouped by status. This endpoint combines:
- **Unfilled/Cancelled**: Limit and bracket orders from the LimitOrderManager (same source as `/limit-orders`)
- **Filled**: Orders extracted from the miner's positions

**Query Parameters:**
- `status` (optional): Comma-separated list of status values to filter by. Valid values: `unfilled`, `filled`, `cancelled`. Defaults to all three if not specified.

**Response Format:**

Orders are always grouped by status:
```json
{
  "unfilled": [
    {
      "trade_pair_id": "BTCUSD",
      "trade_pair": ["BTCUSD", "BTC/USD"],
      "order_type": "LONG",
      "processed_ms": 1702345678901,
      "limit_price": 95000.0,
      "price": 0.0,
      "leverage": 0.1,
      "value": null,
      "quantity": null,
      "src": 5,
      "execution_type": "LIMIT",
      "order_uuid": "550e8400-e29b-41d4-a716-446655440000",
      "stop_loss": 90000.0,
      "take_profit": 100000.0
    }
  ],
  "filled": [
    {
      "trade_pair_id": "ETHUSD",
      "trade_pair": ["ETHUSD", "ETH/USD"],
      "order_type": "SHORT",
      "processed_ms": 1702345600000,
      "price": 2500.50,
      "leverage": -0.2,
      "value": 5000.0,
      "quantity": 2.0,
      "src": 0,
      "execution_type": "MARKET",
      "order_uuid": "660e8400-e29b-41d4-a716-446655440001",
      "bid": 2500.40,
      "ask": 2500.60,
      "slippage": 0.0001,
      "quote_usd_rate": 1.0,
      "usd_base_rate": 0.0004,
      "stop_loss": null,
      "take_profit": null
    }
  ],
  "cancelled": []
}
```

**Examples:**

```bash
# Get all orders (unfilled, filled, and cancelled)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:48888/orders/5GhDr...

# Get only filled orders
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:48888/orders/5GhDr...?status=filled"

# Get unfilled and cancelled orders
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:48888/orders/5GhDr...?status=unfilled,cancelled"
```

**Response Fields:**

Common fields across all statuses:
- `trade_pair_id`: Trade pair identifier (e.g., "BTCUSD")
- `trade_pair`: Array of [trade_pair_id, display_name]
- `order_type`: Order direction (LONG, SHORT, FLAT)
- `processed_ms`: Timestamp when order was created/filled
- `leverage`: Position leverage
- `value`: Order value in USD (if specified)
- `quantity`: Order quantity (if specified)
- `src`: Order source enum
- `execution_type`: MARKET, LIMIT, or BRACKET
- `order_uuid`: Unique order identifier
- `stop_loss`: Stop-loss price (if set)
- `take_profit`: Take-profit price (if set)

Additional fields for filled orders:
- `price`: Fill price
- `bid`: Bid price at execution time
- `ask`: Ask price at execution time
- `slippage`: Slippage amount
- `quote_usd_rate`: Quote currency to USD conversion rate
- `usd_base_rate`: USD to base currency conversion rate

Additional fields for unfilled orders:
- `limit_price`: Target price for limit orders

**Error Responses:**

```json
// Insufficient tier access
{
  "error": "Your API key does not have access to tier 100 data"
}

// Invalid status value
{
  "error": "Invalid status values: {'invalid'}. Valid values are: unfilled, filled, cancelled"
}

// No orders found
{
  "error": "No orders found for miner 5GhDr..."
}
```

## Entity Management

The entity management endpoints enable entity miners to register, create subaccounts, and manage trading under a hierarchical account structure. Entity miners can operate multiple subaccounts (each with its own synthetic hotkey) for diversified trading strategies.

**Access Requirements:**
- All entity management endpoints require **tier 200 API access**
- Standard API keys (tier 0-100) will receive a 403 Forbidden response

### Key Concepts

**Entity Miner:** A parent account that can create and manage multiple subaccounts. Entity miners register with a unique hotkey (VANTA_ENTITY_HOTKEY).

**Subaccount:** A trading account under an entity with its own synthetic hotkey. Each subaccount can place orders independently and has separate performance tracking.

**Synthetic Hotkey:** A generated identifier for subaccounts following the format `{entity_hotkey}_{subaccount_id}` (e.g., `5GhDr3xy...abc_0`). Synthetic hotkeys are used for all trading operations.

### Register Entity

`POST /entity/register`

Register a new entity miner that can create and manage subaccounts.

**Request Body:**
```json
{
  "entity_hotkey": "5GhDr3xy...abc"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Entity 5GhDr3xy...abc registered successfully",
  "entity_hotkey": "5GhDr3xy...abc"
}
```

**Parameters:**
- `entity_hotkey` (string, required): The entity's hotkey SS58 address
- `collateral_amount` (float, optional): Collateral amount in alpha tokens (default: 0.0)

**Example:**
```bash
curl -X POST http://localhost:48888/entity/register \
  -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"entity_hotkey": "5GhDr3xy...abc"}'
```

### Request Entity API Key

`POST /request-api-key`

Returns (or creates) the validator API key for a registered entity. This key grants tier 200 access and is used to authenticate the entity miner gateway's WebSocket connection. Calling this endpoint multiple times for the same entity hotkey returns the same key (idempotent).

**Authentication:** Coldkey signature (no existing API key required).

**Request Body:**
```json
{
  "entity_coldkey": "<coldkey_ss58>",
  "entity_hotkey": "<hotkey_ss58>",
  "signature": "<coldkey_signature>"
}
```

The signature is produced by signing `{"entity_coldkey": "...", "entity_hotkey": "..."}` (JSON, sorted keys) with the coldkey — the same signing scheme as `/entity/register`.

**Response (200):**
```json
{
  "api_key": "<token>"
}
```

**Error Responses:**

| Code | Cause |
|------|-------|
| 400 | Missing or invalid field |
| 401 | Signature verification failed or coldkey does not own hotkey |
| 403 | Entity hotkey is not registered |

**Example:**
```bash
curl -X POST http://localhost:48888/request-api-key \
  -H "Content-Type: application/json" \
  -d '{
    "entity_coldkey": "5FxY...",
    "entity_hotkey": "5GhDr3xy...abc",
    "signature": "0x..."
  }'
```

Store the returned `api_key` as `validator_api_key` in the entity miner's `miner_secrets.json`. This key is typically obtained via `vanta entity apikey` rather than calling this endpoint directly.

### Create Subaccount

`POST /entity/create-subaccount`
`POST /entity/create-hl-subaccount` *(alias — both routes call the same implementation)*

Create a new trading subaccount under an entity. The subaccount receives a unique synthetic hotkey that can be used for order placement. This endpoint handles both standard subaccounts and Hyperliquid-linked subaccounts — the presence of the `hl_address` field in the request body selects the HL path.

**Authentication:** Coldkey signature (no API key required — the request is signed by the entity's coldkey).

**Request Body — Standard subaccount:**
```json
{
  "entity_hotkey": "5GhDr3xy...abc",
  "entity_coldkey": "5FxY...",
  "account_size": 50000,
  "asset_class": "crypto",
  "signature": "0x...",
  "version": "2.0.0"
}
```

**Request Body — Hyperliquid-linked subaccount:**
```json
{
  "entity_hotkey": "5GhDr3xy...abc",
  "entity_coldkey": "5FxY...",
  "account_size": 50000,
  "asset_class": "crypto",
  "hl_address": "0xabcd1234...ef56",
  "payout_address": "0xAbCd...1234",
  "signature": "0x...",
  "version": "2.0.0"
}
```

**Parameters:**
- `entity_hotkey` (string, required): The entity's hotkey SS58 address
- `entity_coldkey` (string, required): The entity's coldkey SS58 address
- `account_size` (float, required): Account size in USD. Must be positive.
- `asset_class` (string, required): `"crypto"`, `"forex"`, or `"equities"`. HL subaccounts always use `"crypto"`.
- `signature` (string, required): Coldkey signature over the sorted-JSON of `{account_size, asset_class, entity_coldkey, entity_hotkey}` (plus `hl_address` and optionally `payout_address` for HL subaccounts).
- `hl_address` (string, optional): Hyperliquid wallet address (`0x` + 40 hex chars). Presence selects the HL subaccount path.
- `payout_address` (string, optional, HL only): EVM address for USDC payouts (`0x` + 40 hex chars).
- `version` (string, optional): vanta-cli version string for compatibility checking.

**Response:**
```json
{
  "status": "success",
  "message": "Subaccount 0 created successfully",
  "subaccount": {
    "subaccount_id": 0,
    "subaccount_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "synthetic_hotkey": "5GhDr3xy...abc_0",
    "asset_class": "crypto",
    "account_size": 50000.0,
    "status": "active",
    "created_at_ms": 1702345678901,
    "eliminated_at_ms": null
  }
}
```

**Response Fields:**
- `subaccount_id`: Monotonically increasing ID (0, 1, 2, ...)
- `subaccount_uuid`: Unique identifier for the subaccount
- `synthetic_hotkey`: Generated hotkey for trading operations (`{entity_hotkey}_{id}`)
- `asset_class`: Asset class assigned to this subaccount
- `account_size`: USD account size
- `status`: Current status (`"active"`, `"eliminated"`, or `"unknown"`)
- `created_at_ms`: Timestamp when subaccount was created
- `eliminated_at_ms`: Timestamp when eliminated (null if active)

**Important Notes:**
- Subaccount IDs are monotonically increasing and never reused
- The synthetic hotkey must be used for all trading operations
- Entity hotkeys cannot place orders directly (only subaccounts can trade)
- New subaccounts are automatically broadcasted to all validators in the network
- The entity miner gateway (`EntityMinerRestServer`) handles signing and forwarding — end users typically call the miner-side endpoint rather than this one directly

### Set Entity Endpoint

`POST /entity/set-endpoint`

Register the entity miner's public REST gateway URL with the validator. The validator stores this URL and uses it to direct HL address lookups. Called automatically by the entity miner gateway at startup.

**Authentication:** Coldkey signature (no API key required).

**Request Body:**
```json
{
  "entity_hotkey": "5GhDr3xy...abc",
  "entity_coldkey": "5FxY...",
  "endpoint_url": "https://my-gateway.example.com",
  "signature": "0x..."
}
```

**Parameters:**
- `entity_hotkey` (string, required): The entity's hotkey SS58 address
- `entity_coldkey` (string, required): The entity's coldkey SS58 address
- `endpoint_url` (string, required): Publicly reachable URL of the entity miner gateway
- `signature` (string, required): Coldkey signature over sorted-JSON of `{endpoint_url, entity_coldkey, entity_hotkey}`

**Response:**
```json
{
  "status": "success",
  "message": "Endpoint URL set successfully",
  "entity_hotkey": "5GhDr3xy...abc",
  "endpoint_url": "https://my-gateway.example.com"
}
```

### Get Entity Endpoint

`GET /entity/endpoint`

Look up the registered gateway URL for an entity miner. Resolves by HL address or synthetic hotkey.

**Authentication:** None required. Public endpoint.

**Query Parameters (one required):**
- `hl_address` (string): Hyperliquid wallet address
- `subaccount` (string): Synthetic hotkey (e.g., `5GhDr3xy...abc_0`)

**Response:**
```json
{
  "status": "success",
  "entity_hotkey": "5GhDr3xy...abc",
  "endpoint_url": "https://my-gateway.example.com"
}
```

**Examples:**
```bash
curl "http://localhost:48888/entity/endpoint?hl_address=0xabcd1234..."
curl "http://localhost:48888/entity/endpoint?subaccount=5GhDr3xy...abc_0"
```

### Get Entity Data

`GET /entity/<entity_hotkey>`

Retrieve comprehensive data for a specific entity, including all subaccounts and their status.

**Response:**
```json
{
  "status": "success",
  "entity": {
    "entity_hotkey": "5GhDr3xy...abc",
    "subaccounts": {
      "0": {
        "subaccount_id": 0,
        "subaccount_uuid": "550e8400-e29b-41d4-a716-446655440000",
        "synthetic_hotkey": "5GhDr3xy...abc_0",
        "status": "active",
        "created_at_ms": 1702345678901,
        "eliminated_at_ms": null
      },
      "1": {
        "subaccount_id": 1,
        "subaccount_uuid": "550e8400-e29b-41d4-a716-446655440001",
        "synthetic_hotkey": "5GhDr3xy...abc_1",
        "status": "active",
        "created_at_ms": 1702345688902,
        "eliminated_at_ms": null
      }
    },
    "next_subaccount_id": 2,
    "collateral_amount": 5000.0,
    "max_subaccounts": 10,
    "registered_at_ms": 1702345670000
  }
}
```

**Example:**
```bash
curl -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
     http://localhost:48888/entity/5GhDr3xy...abc
```

### Get All Entities

`GET /entities`

Retrieve all registered entities in the system.

**Response:**
```json
{
  "status": "success",
  "entities": {
    "5GhDr3xy...abc": {
      "entity_hotkey": "5GhDr3xy...abc",
      "subaccounts": { /* ... */ },
      "next_subaccount_id": 2,
      "collateral_amount": 5000.0,
      "max_subaccounts": 10,
      "registered_at_ms": 1702345670000
    },
    "5FghJk...xyz": {
      /* ... another entity ... */
    }
  },
  "entity_count": 2,
  "timestamp": 1702345690000
}
```

**Example:**
```bash
curl -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
     http://localhost:48888/entities
```

### Get Subaccount Dashboard

`GET /entity/subaccount/<synthetic_hotkey>`

Retrieve comprehensive dashboard data for a specific subaccount by aggregating information from multiple systems.

**Aggregated Data Includes:**
- Subaccount info (status, timestamps, entity parent)
- Challenge period status (bucket, start time, progress)
- Drawdown stats (current equity, daily open equity, EOD high water mark — synthetic hotkeys only)
- Debt ledger data (performance metrics, returns)
- Position data (open positions, leverage, PnL)
- Statistics (cached metrics, scores, rankings)
- Elimination status (if eliminated)

**Response:**
```json
{
  "status": "success",
  "dashboard": {
    "subaccount_info": {
      "synthetic_hotkey": "5GhDr3xy...abc_0",
      "entity_hotkey": "5GhDr3xy...abc",
      "subaccount_id": 0,
      "status": "active",
      "created_at_ms": 1702345678901,
      "eliminated_at_ms": null
    },
    "challenge_period": {
      "bucket": "CHALLENGE",
      "start_time_ms": 1702345678901
    },
    // drawdown is only included for synthetic hotkeys and only after the first evaluation cycle
    "drawdown": {
      "current_equity": 1.032,
      "daily_open_equity": 1.045,
      "eod_hwm": 1.065,
      "last_eod_equity": 1.050,
      "intraday_drawdown_pct": 1.24,
      "eod_drawdown_pct": 1.41,
      "subaccount_challenge_intraday_drawdown_threshold": 0.05,
      "subaccount_challenge_eod_drawdown_threshold": 0.05
    },
    "ledger": {
      "hotkey": "5GhDr3xy...abc_0",
      "total_checkpoints": 2,
      "summary": {
        "cumulative_emissions_alpha": 10.5,
        "cumulative_emissions_tao": 0.05,
        "cumulative_emissions_usd": 25.0,
        "portfolio_return": 1.035,
        "weighted_score": 1.0143,
        "total_fees": -25.0
      },
      "checkpoints": [
        {
          "timestamp_ms": 1702345678901,
          "timestamp_utc": "2023-12-12T08:01:18.901000+00:00",
          "emissions": {
            "chunk_alpha": 10.5,
            "chunk_tao": 0.05,
            "chunk_usd": 25.0,
            "avg_alpha_to_tao_rate": 210.0,
            "avg_tao_to_usd_rate": 500.0,
            "tao_balance_snapshot": 0.05,
            "alpha_balance_snapshot": 10.5
          },
          "performance": {
            "portfolio_return": 1.035,
            "realized_pnl": 350.0,
            "unrealized_pnl": 50.0,
            "spread_fee_loss": -15.0,
            "carry_fee_loss": -10.0,
            "total_fees": -25.0,
            "max_drawdown": 0.985,
            "max_portfolio_value": 1.065,
            "open_ms": 21500000,
            "accum_ms": 21600000,
            "n_updates": 1250
          },
          "penalties": {
            "drawdown": 1.0,
            "risk_profile": 0.98,
            "min_collateral": 1.0,
            "risk_adjusted_performance": 1.0,
            "cumulative": 0.98,
            "challenge_period_status": "CHALLENGE"
          },
          "derived": {
            "return_after_fees": 1.015,
            "weighted_score": 0.9947
          }
        }
      ]
    },
    "positions": {
      "positions": [
        {
          "account_size": 100000.0,
          "average_entry_price": 1877.94680527407,
          "close_ms": 1743202810942,
          "cumulative_entry_value": 9550.0,
          "current_return": 1.0010596467171977,
          "is_closed_position": true,
          "miner_hotkey": "5GhDr3xy...abc_0",
          "net_leverage": 0.0,
          "net_quantity": 0.0,
          "net_value": 0.0,
          "open_ms": 1743152426534,
          "orders": [
            {
              "ask": 0.0,
              "bid": 0.0,
              "execution_type": "MARKET",
              "leverage": 0.025,
              "limit_price": null,
              "order_type": "LONG",
              "order_uuid": "4dcad73e-f120-4dda-aa7b-1471426cccc3",
              "price": 1905.82,
              "price_sources": [],
              "processed_ms": 1743152426534,
              "quantity": 1.3117713110367193,
              "quote_usd_rate": 1.0,
              "slippage": 0.0,
              "src": 0,
              "stop_loss": null,
              "take_profit": null,
              "usd_base_rate": 0.0005247085244146877,
              "value": 2500.0
            }
          ],
          "position_type": "LONG",
          "position_uuid": "4dcad73e-f120-4dda-aa7b-1471426cccc3",
          "realized_pnl": 105.96467171976529,
          "return_at_close": 1.000855671087971,
          "trade_pair": [
            "ETHUSD",
            "ETH/USD",
            0.003,
            0.01,
            0.5
          ],
          "unrealized_pnl": 0.0
        }
      ],
      "thirty_day_returns": 1.028,
      "all_time_returns": 1.035,
      "n_positions": 2,
      "percentage_profitable": 0.65,
      "total_leverage": 0.0
    },
    "statistics": {
      "hotkey": "5GhDr3xy...abc_0",
      "challengeperiod": {
        "bucket": "CHALLENGE",
        "start_time_ms": 1702345678901,
        "returns": 0.035,
        "drawdown": 0.015,
        "days_elapsed": 15,
        "pass_threshold_returns": 0.03,
        "pass_threshold_drawdown": 0.06
      },
      "scores": {
        "crypto": {
          "omega_score": {
            "value": 0.8542,
            "rank": 12,
            "percentile": 92.5,
            "overall_contribution": 0.0
          },
          "sharpe_score": {
            ...
          },
          "sortino_score": {
            ...
          },
          "calmar_score": {
            ...
          },
          "return_score": {
            ...
          }
        }
      },
      "augmented_scores": {
        "crypto": {
          "omega_score": {
            "value": 0.8765,
            "rank": 10,
            "percentile": 94.2,
            "overall_contribution": 0.0
          },
          "sharpe_score": {
            ...
          },
          "sortino_score": {
            ...
          },
          "calmar_score": {
            ...
          },
          "return_score": {
            ...
          }
        }
      },
      "daily_returns": [
        {
          "date": "2023-12-01",
          "value": 1.2
        }
      ],
      "volatility": {
        "annual": 0.185,
        "annual_downside": 0.092
      },
      "drawdowns": {
        "instantaneous_max_drawdown": 0.015,
        "daily_max_drawdown": 0.012
      },
      "plagiarism": 0.0,
      "engagement": {
        "n_checkpoints": 120,
        "n_positions": 45,
        "position_duration": 86400000,
        "checkpoint_durations": [21600000, 21600000, 21600000],
        "minimum_days_boolean": true
      },
      "risk_profile": {
        "risk_profile_score": 0.85,
        "risk_profile_penalty": 0.98
      },
      "asset_class_performance": {
        "crypto": {
          "score": 0.0425,
          "rank": 8,
          "percentile": 95.5
        }
      },
      "pnl_info": {
        "raw_pnl": {
          "value": 3500.0,
          "rank": 9,
          "percentile": 94.8
        }
      },
      "account_size_info": {
        "account_size_statistics": {
          "value": 50000.0,
          "rank": 15,
          "percentile": 88.5
        },
        "account_sizes": [
          {
            "account_size": 50000.0,
            "timestamp_ms": 1702345678901
          }
        ]
      },
      "penalties": {
        "drawdown_threshold": 1.0,
        "risk_profile": 1.0,
        "total": 1.0
      },
      "weight": {
        "value": 0.035,
        "rank": 11,
        "percentile": 93.2
      }
    },
    "elimination": null
  },
  "timestamp": 1702345690000
}
```

**Example:**
```bash
curl -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
     http://localhost:48888/entity/subaccount/5GhDr3xy...abc_0
```

**Response Field Descriptions:**

**Subaccount Info:**
- `synthetic_hotkey`: The subaccount's synthetic hotkey ({entity_hotkey}_{subaccount_id})
- `entity_hotkey`: The parent entity's hotkey
- `subaccount_id`: Monotonically increasing subaccount ID
- `status`: Current status ("active", "eliminated", or "unknown")
- `created_at_ms`: Timestamp when subaccount was created
- `eliminated_at_ms`: Timestamp when eliminated (null if active)

**Challenge Period:**
- `bucket`: Challenge period bucket ("CHALLENGE", "MAINCOMP", "PROBATION", etc.)
- `start_time_ms`: Timestamp when challenge period started

**Ledger (DebtLedger):**
- `hotkey`: The synthetic hotkey for the subaccount
- `total_checkpoints`: Total number of checkpoints in the ledger
- `summary`: Aggregated summary statistics
  - `cumulative_emissions_alpha`: Total alpha emissions across all checkpoints
  - `cumulative_emissions_tao`: Total TAO emissions across all checkpoints
  - `cumulative_emissions_usd`: Total USD value of emissions across all checkpoints
  - `portfolio_return`: Current portfolio return multiplier (1.0 = break-even, 1.035 = 3.5% gain)
  - `weighted_score`: Current weighted score (return × penalties)
  - `total_fees`: Total fees from latest checkpoint
- `checkpoints`: Array of performance snapshots over time
  - `timestamp_ms`: Checkpoint timestamp in milliseconds
  - `timestamp_utc`: Checkpoint timestamp in UTC ISO format
  - `emissions`: Chunk emissions data (not cumulative)
    - `chunk_alpha`: Alpha earned in this checkpoint period
    - `chunk_tao`: TAO earned in this checkpoint period
    - `chunk_usd`: USD value earned in this checkpoint period
    - `avg_alpha_to_tao_rate`: Average alpha-to-TAO conversion rate
    - `avg_tao_to_usd_rate`: Average TAO/USD price
    - `tao_balance_snapshot`: TAO balance at checkpoint end
    - `alpha_balance_snapshot`: Alpha balance at checkpoint end
  - `performance`: Performance metrics for this checkpoint
    - `portfolio_return`: Portfolio return multiplier (1.0 = break-even)
    - `realized_pnl`: Net realized profit/loss during this checkpoint
    - `unrealized_pnl`: Net unrealized profit/loss during this checkpoint
    - `spread_fee_loss`: Spread fee losses during this checkpoint
    - `carry_fee_loss`: Carry fee losses during this checkpoint
    - `total_fees`: Sum of all fees paid
    - `max_drawdown`: Worst loss from peak (0.985 = 1.5% drawdown)
    - `max_portfolio_value`: Best portfolio value achieved
    - `open_ms`: Time with open positions (milliseconds)
    - `accum_ms`: Duration of this checkpoint period (milliseconds)
    - `n_updates`: Number of performance updates in this period
  - `penalties`: Penalty multipliers applied
    - `drawdown`: Drawdown threshold penalty multiplier
    - `risk_profile`: Risk profile penalty multiplier
    - `min_collateral`: Minimum collateral penalty multiplier
    - `risk_adjusted_performance`: Risk-adjusted performance penalty multiplier
    - `cumulative`: Combined penalty multiplier (product of all penalties)
    - `challenge_period_status`: Challenge period status for this checkpoint
  - `derived`: Derived/computed fields
    - `return_after_fees`: Portfolio return after deducting all fees
    - `weighted_score`: Final score after applying all penalties

**Positions:**
- `positions`: Array of trading positions (open and recently closed)
  - `position_uuid`: Unique identifier for this position
  - `miner_hotkey`: Subaccount's synthetic hotkey
  - `position_type`: LONG, SHORT, or FLAT
  - `is_closed_position`: Whether position is closed (false = still open)
  - `trade_pair`: [symbol, display_name, spread_fee_rate, min_leverage, max_leverage]
  - `open_ms`: When position was opened (timestamp)
  - `close_ms`: When position was closed (0 if still open)
  - `net_leverage`: Current leverage (positive = LONG, negative = SHORT, 0 = FLAT)
  - `average_entry_price`: Average price across all entries
  - `current_return`: Current return multiplier (1.0235 = 2.35% gain)
  - `return_at_close`: Final return when position closes
  - `realized_pnl`: Net realized profit/loss for this position
  - `unrealized_pnl`: Net unrealized profit/loss for this position
  - `orders`: Array of orders within this position
    - `order_uuid`: Unique identifier for this order
    - `order_type`: LONG, SHORT, or FLAT
    - `leverage`: Leverage applied to this order
    - `price`: Execution price
    - `bid`: Bid price at execution time
    - `ask`: Ask price at execution time
    - `processed_ms`: When order was processed
    - `price_sources`: Price data sources used for execution
    - `src`: 0 = miner order, 1 = auto-flatten (elimination), 2 = auto-flatten (pair deprecation)
- `thirty_day_returns`: Return multiplier over the last 30 days
- `all_time_returns`: Return multiplier across all positions
- `n_positions`: Total number of positions (open + closed in last 30 days)
- `percentage_profitable`: Percentage of closed positions that were profitable (0-1)
- `total_leverage`: Sum of net leverage across all open positions

**Statistics:**
- `hotkey`: The synthetic hotkey
- `challengeperiod`: Detailed challenge period progress (if applicable)
  - `bucket`: Current bucket (CHALLENGE/MAINCOMP/PROBATION/PLAGIARISM/UNKNOWN)
  - `start_time_ms`: Timestamp when miner entered challenge period
  - `returns`: Current returns during challenge period (e.g., 0.035 = 3.5%)
  - `drawdown`: Current drawdown during challenge period (e.g., 0.015 = 1.5%)
  - `days_elapsed`: Days spent in challenge period
  - `pass_threshold_returns`: Required returns to pass (e.g., 0.03 = 3%)
  - `pass_threshold_drawdown`: Maximum allowed drawdown (e.g., 0.06 = 6%)
- `scores`: Individual metric scores with rankings
  - `omega_score`, `sharpe_score`, `sortino_score`, `calmar_score`, `return_score`: Performance metrics
    - `value`: Calculated metric value
    - `rank`: Rank among all miners (lower is better, 1 = best)
    - `percentile`: Percentile ranking (higher is better, 100 = best)
    - `overall_contribution`: Weight in combined score (0-1, sum = 1.0)
- `augmented_scores`: Augmented scores (implementation-specific)
- `daily_returns`: Array of daily return data
- `volatility`: Volatility metrics
- `drawdowns`: Drawdown metrics
- `plagiarism`: Plagiarism score (0.0 = no plagiarism detected)
- `engagement`: Engagement metrics
- `risk_profile`: Risk profile analysis
- `asset_class_performance`: Performance by asset class
- `pnl_info`: Detailed PnL information
- `account_size_info`: Account size information
- `penalties`: Penalty breakdown
  - `drawdown_threshold`: Drawdown threshold penalty
  - `risk_profile`: Risk profile penalty
  - `total`: Total penalty multiplier
- `weight`: Weight assignment for this subaccount
  - `value`: Actual weight value
  - `rank`: Rank among all miners
  - `percentile`: Percentile ranking

**Elimination:**
- `null` if subaccount is active
- Object with elimination details if eliminated:
  - `hotkey`: The eliminated hotkey
  - `reason`: Elimination reason (LIQUIDATED, PLAGIARISM, etc.)
  - `elimination_initiated_time_ms`: When elimination occurred
  - `dd`: Drawdown at elimination (if applicable)
  - `price_info`: Price information at elimination (optional)
  - `return_info`: Return information at elimination (optional)

**Use Cases:**
- Frontend dashboards for displaying subaccount performance
- Real-time monitoring of subaccount trading activity
- Challenge period progress tracking
- Position and risk management

### Get Subaccount Dashboard (version 2)

`GET /v2/entity/subaccount/<synthetic_hotkey>`

Retrieve comprehensive dashboard data for a specific subaccount by aggregating information from multiple systems.

**Aggregated Data Includes:**
- Subaccount info (status, timestamps, entity parent)
- Challenge period status (bucket, start time, progress)
- Drawdown stats (current equity, daily open equity, EOD high water mark — synthetic hotkeys only)
- Debt ledger data (performance metrics, returns)
- Position data (open positions, leverage, PnL)
- Statistics (daily returns)
- Elimination status (if eliminated)

**Parameters:**
- `positions_time_ms` (int): Only include positions after this timestamp (milliseconds, exclusive)
- `limit_orders_time_ms` (int): Only include limit orders after this timestamp (milliseconds, exclusive)
- `checkpoints_time_ms` (int): Only include ledger checkpoints after this timestamp (milliseconds, exclusive)
- `daily_returns_time_ms` (int): Only include daily returns after this timestamp (milliseconds, exclusive)

**Example:**
```bash
curl -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
     http://localhost:48888/entity/subaccount/5GhDr3xy...abc_123?positions_time_ms=1770027342742
```

**Response:**
```json
{
  "status": "success",
  "dashboard": {
    "subaccount_info": {
      "synthetic_hotkey": "5GhDr3xy...abc_123",
      "subaccount_uuid": "abc...789",
      "asset_class": "crypto",
      "account_size": 100000.0,
      "status": "active",
      "created_at_ms": 1770657674533,
      "eliminated_at_ms": null
    },
    "challenge_period": {
      "bucket": "CHALLENGE",
      "start_time_ms": 1702345678901
    },
    // drawdown is only included for synthetic hotkeys and only after the first evaluation cycle
    "drawdown": {
      "current_equity": 1.032,
      "daily_open_equity": 1.045,
      "eod_hwm": 1.065,
      "last_eod_equity": 1.050,
      "intraday_drawdown_pct": 1.24,
      "eod_drawdown_pct": 1.41,
      "subaccount_challenge_intraday_drawdown_threshold": 0.05,
      "subaccount_challenge_eod_drawdown_threshold": 0.05
    },
    // elimination is only included if the subaccount is eliminated
    "elimination": {
      "elimination_initiated_time_ms": 1771893304364,
      "reason": "FAILED_CHALLENGE_PERIOD_DRAWDOWN",
      "dd": 99.99929388128832
    },
    "account_size_data": {
      "account_size": 100000,
      "total_realized_pnl": -580.4117834499989,
      "capital_used": 0.0,
      "balance": 98339.3684339573,
      "buying_power": 122924.21054244661,
      "max_return": 1.0
    },
    "positions": {
      // positions is only included if there are open positions or closed positions newer
      // than the positions_time_ms query parameter
      "positions": { 
        "7413e788-e000-465a-a37c-da22d7b94acc": { // position_uuid
          "tp": "SOL/USD", // trade_pair
          "t": "FLAT", // position_type
          "o": 1770727691818, // open_ms
          "r": 1.0001031522298502, // current_return
          "nl": 0.1, // net_leverage (if not zero)
          "ap": 83.88350685527055, // average_entry_price
          "rp": 10.315222985026267, // realized_pnl
          "c": 1770727882140, // close_ms (if closed)
          "rc": 0.9999031315994042, // return_at_close (if closed)
          "fo": { // filled_orders (if filled orders)
            "7413e788-e000-465a-a37c-da22d7b94acc": { // order_uuid
              "t": "LONG", // order_type
              "l": 0.1, // leverage (if not null)
              "q": 119.26058437686345, // quantity (if not null)
              "pr": 83.85, // price (if not zero)
              "v": 10000.0, // value
              "e": "MARKET", // execution_type
              "p": 1770727691818, // processed_ms
              "lp": 83.85, // limit_price (if not null)
              "sl": 78.34, // stop_loss (if not null)
              "tk": 88.42, // take_profit (if not null)
              "tsl": {"pct": 0.02}, // trailing_stop (if not null) — {"pct": <trailing_percent>} or {"val": <trailing_value>}
              "sp": 80.00, // stop_price (if STOP_LIMIT)
              "cond": "GTE" // stop_condition (if STOP_LIMIT) — "GTE" or "LTE"
            }
          },
          "uo": { // unfilled_orders (if unfilled orders)
            "7413e788-e000-465a-a37c-da22d7b94acc": { // order_uuid
              "t": "LONG", // order_type
              "l": 0.1, // leverage (if not null)
              "q": 119.26058437686345, // quantity (if not null)
              "pr": 83.85, // price (if not zero)
              "v": 10000.0, // value
              "e": "MARKET", // execution_type
              "p": 1770727691818, // processed_ms
              "lp": 83.85, // limit_price (if not null)
              "sl": 78.34, // stop_loss (if not null)
              "tk": 88.42, // take_profit (if not null)
              "tsl": {"pct": 0.02}, // trailing_stop (if not null) — {"pct": <trailing_percent>} or {"val": <trailing_value>}
              "sp": 80.00, // stop_price (if STOP_LIMIT)
              "cond": "GTE" // stop_condition (if STOP_LIMIT) — "GTE" or "LTE"
            }
          },
          "fh": { // fee_history (if fee history)
            "1770727691818": { // time_ms
              "t": "transaction", // fee_type
              "a": 4.54 // amount
            }
          }
        }
      },
      // all_time_returns is only included if there are closed positions newer than the
      // positions_time_ms query parameter
      "all_time_returns": 1.035,
      "total_leverage": 0.0,
      "positions_time_ms": 1770727691818 // Use as a query parameter in next request
    },
    // limit_orders is only included if any limit orders newer than the limit_order_time_ms
    // query parameter
    "limit_orders": {
      "open_orders": {
        "7413e788-e000-465a-a37c-da22d7b94acc": { // order_uuid
          "tp": "SOL/USD", // trade_pair
          "t": "LONG", // order_type
          "l": 0.1, // leverage (if not null)
          "q": 119.26058437686345, // quantity (if not null)
          "pr": 83.85, // price (if not zero)
          "v": 10000.0, // value
          "e": "MARKET", // execution_type
          "p": 1770727691818, // processed_ms
          "lp": 83.85, // limit_price (if not null)
          "sl": 78.34, // stop_loss (if not null)
          "tk": 88.42, // take_profit (if not null)
          "tsl": {"pct": 0.02}, // trailing_stop (if not null) — {"pct": <trailing_percent>} or {"val": <trailing_value>}
          "sp": 80.00, // stop_price (if STOP_LIMIT)
          "cond": "GTE" // stop_condition (if STOP_LIMIT) — "GTE" or "LTE"
        }
      },
      "closed_orders": [
        "7413e788-e000-465a-a37c-da22d7b94acc" // order_uuid
      ],
      "limit_orders_time_ms": 1701388800000, // Use as a query parameter in next request
    },
    "ledger": {
      "checkpoints": {
        "1770768000000": { // timestamp_ms
          "r": -11.400847781869729, // realized_pnl (if not zero)
          "u": 9.2605843763286, // unrealized_pnl (if not zero)
          "m": 1.0001092732205306, // max_portfolio_value
          "s": "CHALLENGE" // challenge_period_status
        }
      },
      "portfolio_return": 2.2204460492503128e-16,
      "checkpoints_time_ms": 1770768000000 // Use as a query parameter in next request
    },
    // statistics is only included if any returns newer than the daily_returns_time_ms
    // query parameter
    "statistics": {
      "daily_returns": { 
        "2023-12-01" : 1.2 // date : value
      },
      "daily_returns_time_ms": 1701388800000 // Use as a query parameter in next request
    }
  },
  "timestamp": 1702345690000
}
```

**Response Field Descriptions:**

**Subaccount Info:**

This is the only guaranteed section of the response. All other sections may be missing if they do not contain new information or the manager that supplies the information is not available.
- `synthetic_hotkey`: The subaccount's synthetic hotkey ({entity_hotkey}_{subaccount_id})
- `subaccount_uuid`: Unique identifier for this subaccount
- `asset_class`: Asset class (crypto, forex, etc.)
- `account_size`: Current account size (in USD)
- `status`: Current status ("active", "eliminated", or "unknown")
- `created_at_ms`: Timestamp when subaccount was created
- `eliminated_at_ms`: Timestamp when eliminated (null if active)

**Challenge Period:**
- `bucket`: Challenge period bucket ("CHALLENGE", "MAINCOMP", "PROBATION", etc.)
- `start_time_ms`: Timestamp when challenge period started

**Drawdown (synthetic hotkeys only):**

Only present after the first challenge period evaluation cycle (~60s after startup). Values match exactly what the evaluation loop computed.
- `current_equity`: Portfolio equity at last evaluation: `(balance + unrealized_pnl) / account_size`
- `daily_open_equity`: Equity at today's midnight UTC checkpoint (Rule 1 baseline). Defaults to `1.0` if no midnight checkpoint exists yet.
- `eod_hwm`: Highest end-of-day equity across all midnight checkpoints ever (Rule 2 high water mark). Defaults to `1.0` if no midnight checkpoints exist.
- `last_eod_equity`: Most recent midnight checkpoint equity. Defaults to `1.0` if no midnight checkpoints exist yet.
- `intraday_drawdown_pct`: Percentage drop from `daily_open_equity` to `current_equity`. Positive = drawdown, negative = gain since open.
- `eod_drawdown_pct`: Percentage drop from `eod_hwm` to `last_eod_equity`. `0.0` if no midnight checkpoints exist.
- `subaccount_challenge_intraday_drawdown_threshold`: Elimination threshold for Rule 1 (e.g. `0.05` = 5%).
- `subaccount_challenge_eod_drawdown_threshold`: Elimination threshold for Rule 2 (e.g. `0.05` = 5%).

**Elimination:**

This section is only included if the subaccount is eliminated.
  - `elimination_initiated_time_ms`: When elimination occurred
  - `reason`: Elimination reason (LIQUIDATED, PLAGIARISM, etc.)
  - `dd`: Drawdown at elimination (if applicable)

**Positions:**
- `positions`: Dictionary of trading positions
  - `position_uuid`: Unique identifier for this position
  - `trade_pair`: display_name
  - `position_type`: LONG, SHORT, or FLAT
  - `open_ms`: When position was opened (timestamp)
  - `current_return`: Current return multiplier (1.0235 = 2.35% gain)
  - `net_leverage`: Current leverage (positive = LONG, negative = SHORT, 0 = FLAT)
  - `average_entry_price`: Average price across all entries
  - `close_ms`: When position was closed (only included if closed)
  - `return_at_close`: Final return when position closes (only included if closed)
  - `realized_pnl`: Net realized profit/loss for this position
  - `filled_orders`: Dictionary of filled orders within this position
    - `order_uuid`: Unique identifier for this order
    - `order_type`: LONG, SHORT, or FLAT
    - `leverage`: Leverage applied to this order
    - `quanity`: Quanity executed
    - `price`: Execution price
    - `value`: Value of order (quantity * price)
    - `execution_type`: MARKET, LIMIT, or BRACKET
    - `processed_ms`: When order was processed
    - `limit_price`: Limit price (if applicable)
    - `stop_loss`: Stop loss price (if applicable)
    - `take_profit`: Take profit price (if applicable)
    - `trailing_stop`: Trailing stop (if applicable) — `{"pct": <trailing_percent>}` or `{"val": <trailing_value>}`
    - `stop_price`: Trigger price for STOP_LIMIT orders (if applicable)
    - `stop_condition`: Trigger direction for STOP_LIMIT orders — `"GTE"` (trigger when price >= stop_price) or `"LTE"` (trigger when price <= stop_price)
  - `unfilled_orders`: Dictionary of unfilled orders within this position
  - `fee_history`: Dictionary of fee events within this position
    - `time_ms`: Timestamp of fee
    - `fee_type`: transaction, carry, or interest
    - `amount`: Amount of fee` 
- `all_time_returns`: Return multiplier across all positions
- `total_leverage`: Sum of net leverage across all open positions
- `positions_time_ms`: Timestamp of last position or order (used as a query parameter in the next request)

**Limit Orders:**
- `open_orders`: Dictionary of untriggered limit orders
  - `order_uuid`: Unique identifier for this order
  - `trade_pair`: display_name
  - `order_type`: LONG, SHORT, or FLAT
  - `leverage`: Leverage applied to this order
  - `quanity`: Quanity executed
  - `price`: Execution price
  - `value`: Value of order (quantity * price)
  - `execution_type`: MARKET, LIMIT, or BRACKET
  - `processed_ms`: When order was processed
  - `limit_price`: Limit price (if applicable)
  - `stop_loss`: Stop loss price (if applicable)
  - `take_profit`: Take profit price (if applicable)
  - `trailing_stop`: Trailing stop (if applicable) — `{"pct": <trailing_percent>}` or `{"val": <trailing_value>}`
  - `stop_price`: Trigger price for STOP_LIMIT orders (if applicable)
  - `stop_condition`: Trigger direction for STOP_LIMIT orders — `"GTE"` (trigger when price >= stop_price) or `"LTE"` (trigger when price <= stop_price)
- `closed_orders`: Array of closed limit orders
  - `order_uuid` Unique identifier for a closed order
- `limit_orders_time_ms`: Timestamp of last limit order (used as a query parameter in the next request)

**Ledger:**
- `checkpoints`: Array of performance snapshots over time
  - `timestamp_ms`: Checkpoint timestamp in milliseconds
  - `realized_pnl`: Net realized profit/loss during this checkpoint
  - `unrealized_pnl`: Net unrealized profit/loss during this checkpoint
  - `max_portfolio_value`: Best portfolio value achieved
  - `challenge_period_status`: Challenge period status for this checkpoint
- `portfolio_return`: Portfolio return multiplier (1.0 = break-even)
- `checkpoints_time_ms`: Timestamp of last checkpoint (used as a query parameter in the next request)

**Statistics:**
- `daily_returns`: Dictionary of daily return data
  - `date`
  - `value`
- `daily_returns_time_ms`: Timestamp of the last daily return (used as a query parameter in the next request)

**Use Cases:**
- Frontend dashboards for displaying subaccount performance
- Real-time monitoring of subaccount trading activity
- Challenge period progress tracking
- Position and risk management

### Eliminate Subaccount

`POST /entity/subaccount/eliminate`

Manually eliminate a subaccount. This permanently disables trading for the subaccount.

**Request Body:**
```json
{
  "entity_hotkey": "5GhDr3xy...abc",
  "subaccount_id": 0,
  "reason": "manual_elimination"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Subaccount 0 eliminated successfully"
}
```

**Parameters:**
- `entity_hotkey` (string, required): The entity's hotkey SS58 address
- `subaccount_id` (int, required): The subaccount ID to eliminate
- `reason` (string, optional): Reason for elimination (default: "manual_elimination")

**Example:**
```bash
curl -X POST http://localhost:48888/entity/subaccount/eliminate \
  -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_hotkey": "5GhDr3xy...abc",
    "subaccount_id": 0,
    "reason": "manual_elimination"
  }'
```

**Important Notes:**
- Eliminated subaccounts cannot be reactivated
- The subaccount ID will never be reused for this entity
- All open positions for the subaccount are automatically closed (FLAT order)
- Elimination is permanent and cannot be undone

### Calculate Subaccount Payout

`POST /entity/subaccount/payout`

Calculate payout for a subaccount based on debt ledger checkpoints within a specified time range. This endpoint aggregates performance data from the debt ledger to determine earnings over a period.

**Request Body:**
```json
{
  "subaccount_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "start_time_ms": 1772409600000,
  "end_time_ms": 1775088000000
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "hotkey": "5GhDr3xy...abc_0",
    "total_checkpoints": 10,
    "checkpoints": [...],
    "weekly_settlements": [
      {
        "start_ms": 1772409600000,
        "end_ms": 1773014400000,
        "eow_balance": 130.0,
        "eow_unrealized": -5.0,
        "payout": 125.0,
        "orders": [...]
      }
    ],
    "payout": 125.5
  },
  "timestamp": 1702857600000
}
```

**Parameters:**
- `subaccount_uuid` (string, required): The unique UUID of the subaccount
- `start_time_ms` (int, required): Start timestamp in milliseconds (inclusive). Must be Monday 00:00:00 UTC or `0`
- `end_time_ms` (int, required): End timestamp in milliseconds (inclusive). Must align to a 12-hour boundary

**Example:**
```bash
curl -X POST http://localhost:48888/entity/subaccount/payout \
  -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "subaccount_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "start_time_ms": 1702252800000,
    "end_time_ms": 1702857600000
  }'
```

**Response Fields:**
- `hotkey`: The synthetic hotkey of the subaccount
- `total_checkpoints`: Total number of debt ledger checkpoints across the subaccount's history
- `checkpoints`: Array of all debt ledger checkpoint data for the subaccount
- `weekly_settlements`: Array of per-week payout breakdowns. Each entry includes:
  - `start_ms`: Monday 00:00:00 UTC start of the week
  - `end_ms`: Monday 00:00:00 UTC end of the week (exclusive)
  - `eow_balance`: Cumulative realized PnL minus fees at end of week
  - `eow_unrealized`: Unrealized PnL at the nearest checkpoint before week end
  - `payout`: `max(0, eow_balance - prev_hwm + min(0, eow_unrealized))` for this week
  - `orders`: Orders with non-zero realized PnL that settled in this week
- `payout`: Sum of weekly payouts for weeks whose `start_ms >= start_time_ms`

**Important Notes:**
- Timestamps must be valid integers and non-negative
- `start_time_ms` must be Monday 00:00:00 UTC or `0`
- `end_time_ms` must align to a 12-hour boundary
- `start_time_ms` must be less than or equal to `end_time_ms`
- Returns 404 if the subaccount is not found, is not in a funded/alpha bucket, or has no orders
- Payout uses a weekly high-water mark (HWM) model: only weeks where balance exceeds the previous HWM contribute to payout

### Entity Trading Workflow

**1. Register as an entity miner:**
```bash
curl -X POST http://localhost:48888/entity/register \
  -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"entity_hotkey": "5GhDr..."}'
```

**2. Create subaccounts:**
```bash
curl -X POST http://localhost:48888/entity/create-subaccount \
  -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"entity_hotkey": "5GhDr..."}'
```

**3. Place orders using synthetic hotkeys:**
- Use the `synthetic_hotkey` (e.g., `5GhDr..._0`) returned from subaccount creation
- Submit orders via the standard Vanta order placement mechanism
- Each subaccount trades independently with its own positions and performance tracking

**4. Monitor performance:**
```bash
# Get dashboard data for a subaccount
curl -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
     http://localhost:48888/entity/subaccount/5GhDr..._0

# Get all entity data
curl -H "Authorization: Bearer YOUR_TIER_200_API_KEY" \
     http://localhost:48888/entity/5GhDr...
```

### Entity Management with Python

```python
import requests

API_KEY = "YOUR_TIER_200_API_KEY"
BASE_URL = "http://localhost:48888"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Register entity
response = requests.post(
    f"{BASE_URL}/entity/register",
    headers=HEADERS,
    json={
        "entity_hotkey": "5GhDr3xy...abc",
        "collateral_amount": 5000.0,
        "max_subaccounts": 10
    }
)
print(f"Entity registered: {response.json()}")

# Create subaccount
response = requests.post(
    f"{BASE_URL}/entity/create-subaccount",
    headers=HEADERS,
    json={"entity_hotkey": "5GhDr3xy...abc"}
)
subaccount = response.json()["subaccount"]
synthetic_hotkey = subaccount["synthetic_hotkey"]
print(f"Created subaccount with synthetic hotkey: {synthetic_hotkey}")

# Get entity data
response = requests.get(
    f"{BASE_URL}/entity/5GhDr3xy...abc",
    headers=HEADERS
)
entity_data = response.json()["entity"]
print(f"Entity has {len(entity_data['subaccounts'])} subaccounts")

# Get subaccount dashboard
response = requests.get(
    f"{BASE_URL}/entity/subaccount/{synthetic_hotkey}",
    headers=HEADERS
)
dashboard = response.json()["dashboard"]
print(f"Subaccount status: {dashboard['subaccount_info']['status']}")

# Eliminate subaccount (if needed)
response = requests.post(
    f"{BASE_URL}/entity/subaccount/eliminate",
    headers=HEADERS,
    json={
        "entity_hotkey": "5GhDr3xy...abc",
        "subaccount_id": 0,
        "reason": "poor_performance"
    }
)
print(f"Elimination result: {response.json()}")
```

### Error Responses

All entity endpoints may return the following error responses:

**401 Unauthorized:**
```json
{
  "error": "Unauthorized access"
}
```
Missing or invalid API key.

**403 Forbidden:**
```json
{
  "error": "Your API key does not have access to tier 200 data"
}
```
API key does not have tier 200 access required for entity management.

**404 Not Found:**
```json
{
  "error": "Entity 5GhDr... not found"
}
```
Requested entity or subaccount does not exist.

**400 Bad Request:**
```json
{
  "error": "Missing required field: entity_hotkey"
}
```
Invalid request format or missing required parameters.

**503 Service Unavailable:**
```json
{
  "error": "Entity management not available"
}
```
Entity management service is not running or unavailable.

## Hyperliquid Trader Endpoints

These endpoints expose data for Hyperliquid-linked subaccounts. **No authentication required** — they are public.

### Get HL Trader Dashboard

`GET /hl-traders/<hl_address>`

Resolve a Hyperliquid wallet address to its synthetic hotkey and return the full subaccount dashboard (same structure as `GET /entity/subaccount/<synthetic_hotkey>` v1 format).

**Authentication:** None required.

**Query Parameters (optional):**
- `positions_time_ms` (int): Only include positions newer than this timestamp (ms)
- `limit_orders_time_ms` (int): Only include limit orders newer than this timestamp (ms)

**Response:**
```json
{
  "status": "success",
  "dashboard": {
    "subaccount_info": {
      "synthetic_hotkey": "5GhDr3xy...abc_0",
      "entity_hotkey": "5GhDr3xy...abc",
      "subaccount_id": 0,
      "asset_class": "crypto",
      "hl_address": "0xabcd1234...",
      "payout_address": "0xAbCd...",
      "status": "active",
      "created_at_ms": 1702345678901,
      "eliminated_at_ms": null
    },
    "challenge_period": { "bucket": "CHALLENGE", "start_time_ms": 1702345678901 },
    "drawdown": { "..." : "..." },
    "elimination": null,
    "account_size_data": { "..." : "..." },
    "positions": { "..." : "..." },
    "limit_orders": { "..." : "..." }
  },
  "timestamp": 1702345690000
}
```

**Example:**
```bash
curl http://localhost:48888/hl-traders/0xabcd1234...
```

### Get HL Trader Limits

`GET /hl-traders/<hl_address>/limits`

Returns the trading limits for a Hyperliquid subaccount based on its account size and challenge period status.

**Authentication:** None required.

**Response:**
```json
{
  "status": "success",
  "hl_address": "0xabcd1234...",
  "account_size": 50000.0,
  "max_position_per_pair_usd": 25000.0,
  "max_portfolio_usd": 500000.0,
  "in_challenge_period": true,
  "timestamp": 1702345690000
}
```

**Response Fields:**
- `account_size`: Subaccount account size in USD
- `max_position_per_pair_usd`: Maximum USD exposure per trade pair (`account_size × max_leverage`; halved during challenge period)
- `max_portfolio_usd`: Maximum total portfolio value (`account_size × portfolio_cap`; halved during challenge period)
- `in_challenge_period`: Whether the subaccount is currently in the challenge period

**Example:**
```bash
curl http://localhost:48888/hl-traders/0xabcd1234.../limits
```

### Get HL Leaderboard

`GET /hl-leaderboard`

Returns aggregated leaderboard data for all Hyperliquid traders, including summary metrics and trader rankings.

**Authentication:** None required.

**Example:**
```bash
curl http://localhost:48888/hl-leaderboard
```

## Compression Support

The API server supports automatic gzip compression for REST responses, which can significantly reduce payload sizes and improve performance. Compression is particularly beneficial for large responses like miner positions and statistics.

### Compression Benefits

- **Reduced Bandwidth**: Responses can be compressed by 70-90%, especially for JSON data
- **Faster Transmission**: Smaller payloads lead to quicker response times
- **Lower Costs**: Reduced data transfer costs with cloud hosting providers

### Using Compression with curl

To request compressed responses with curl, use the `--compressed` flag:

```bash
# Get positions with compression
curl --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key"

# Get positions with a specific tier (compressed)
curl --compressed -X GET "http://<server_ip>:48888/miner-positions?tier=30" \
  -H "Authorization: Bearer your_api_key"

# Check the size of the response without displaying it
curl -s --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key" -o /dev/null -w 'Size: %{size_download} bytes\n'

# View detailed headers to confirm compression
curl -v --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key" > /dev/null
```

## WebSocket API

The WebSocket API enables real-time streaming of trading data and updates.

### Connection and Authentication

After connecting to the WebSocket server at `ws://<server_ip>:8765`, clients must authenticate:

```json
{
  "api_key": "your_api_key",
  "last_sequence": -1
}
```

The `last_sequence` parameter helps track message continuity and allows clients to resume from where they left off.

### Websocket Message Format

Messages from the websocket server are parsed into VantaWebsocketMessage ([vanta_api/websocket_client.py](vanta_api/websocket_client.py)) objects which mirror the position data from the REST endpoint.

This data is serialized into a Vanta Position object, and the server will send a message with the sequence number of the last message sent. Sequence number can be used to detect gaps in the message stream. The client can use this to make a REST call to fill in the gaps.

Lag info is also included in the message. lag from the queue is lag from when the order was made available to the websocket server. Lag from the order is lag from when the order was placed and is larger due to variable time in price filling and order processing in the Vanta repo.

Here a snippet of a terminal printing VantaWebsocketMessage objects:
```bash
Received message VantaWebSocketMessage(seq=237)
Position Summary:
{
  "miner_hotkey": "5CUUWxGzf4qU5DCgLcL65qAKsQF1ezUTvBzfD548zPEDzxmR",
  "position_uuid": "0c2aa350-2bea-4ccd-8cce-6da220d5bae6",
  "open_ms": 1745897174103,
  "trade_pair": [
    "USDJPY",
    "USD/JPY",
    7e-05,
    0.1,
    5
  ],
  "current_return": 1.0,
  "close_ms": null,
  "net_leverage": -0.3000000000000007,
  "return_at_close": 1.0,
  "average_entry_price": 142.37951594322527,
  "cumulative_entry_value": -42.71385478296768,
  "realized_pnl": 0.0,
  "position_type": "SHORT",
  "is_closed_position": false
}
New Order:
{
  "trade_pair_id": "USDJPY",
  "order_type": "SHORT",
  "leverage": -0.3000000000000007,
  "price": 142.386,
  "bid": 142.386,
  "ask": 142.399,
  "slippage": 4.553858367205138e-05,
  "processed_ms": 1745897174103,
  "price_sources": [
    {
      "source": "Polygon_rest",
      "timespan_ms": 1000,
      "open": 142.39249999999998,
      "close": 142.39249999999998,
      "vwap": null,
      "high": 142.39249999999998,
      "low": 142.39249999999998,
      "start_ms": 1745897173000,
      "websocket": false,
      "lag_ms": 104,
      "bid": 142.386,
      "ask": 142.399
    }
  ],
  "order_uuid": "0c2aa350-2bea-4ccd-8cce-6da220d5bae6",
  "src": 0
}
Approx Timelag (ms): from_queue=49, from_order=1111
```

### API Key Management

Each API key is limited to a maximum of 5 concurrent WebSocket connections. When this limit is reached, the oldest connection will be automatically disconnected to make room for new connections.

## Running the Server

### Prerequisites

- Python 3.8+
- Required packages: flask, waitress, flask_compress, websockets

### Launching the Server

```bash
python validator_api_manager.py --serve
```

## Client Usage Examples

### REST API with cURL

```bash
# Get positions with compression
curl --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key"

# Get positions with a specific tier
curl --compressed -X GET "http://<server_ip>:48888/miner-positions?tier=30" \
  -H "Authorization: Bearer your_api_key"

# Check the size of the response without displaying it
curl -s --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key" -o /dev/null -w 'Size: %{size_download} bytes\n'
```

### REST API with Python

```python
import requests
import json

# Basic request (compression is enabled by default in requests)
url = 'http://<server_ip>:48888/validator-checkpoint'
headers = {'Authorization': 'Bearer your_api_key'}

response = requests.get(url, headers=headers)
data = response.json()

# Check if compression was used
if 'Content-Encoding' in response.headers:
    print(f"Response was compressed using: {response.headers['Content-Encoding']}")
    # The requests library automatically decompresses the response
    print(f"Original compressed size (approximate): {len(response.content)} bytes")
    print(f"Decompressed size: {len(response.text)} bytes")

# Save to file
with open('validator_checkpoint.json', 'w') as f:
    json.dump(data, f)

# Using a session for multiple requests (more efficient)
with requests.Session() as session:
    session.headers.update({'Authorization': 'Bearer your_api_key'})
    
    # Get miner positions
    pos_response = session.get('http://<server_ip>:48888/miner-positions')
    positions = pos_response.json()
    
    # Get statistics
    stats_response = session.get('http://<server_ip>:48888/statistics')
    statistics = stats_response.json()
```

### WebSocket Client with Python

Our WebSocket client ([vanta_api/websocket_client.py](vanta_api/websocket_client.py)) provides a simple interface for receiving real-time data:

```python
from vanta_api.websocket_client import VantaWebSocketClient
import sys

...

if __name__ == "__main__":
    # Get API key from command line argument or use default
    api_key = sys.argv[1] if len(sys.argv) > 1 else "test_key"
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765


    # Define a simple message handler. print position and order details
    def handle_messages(messages):
        for msg in messages:
            print(f"\nReceived message {msg}")
    # Create client
    print(f"Connecting to ws://{host}:{port} with API key: {api_key}")
    client = VantaWebSocketClient(api_key=api_key, host=host, port=port)

    # Run client
    client.run(handle_messages)
```

### Async Message Handler

For more complex processing, you can use async handlers:

```python
from vanta_api.websocket_client import VantaWebSocketClient, VantaWebSocketMessage
import asyncio
from typing import List


async def process_message(msg: VantaWebSocketMessage):
    # Example: Check bid/ask spread
    if msg.new_order.bid and msg.new_order.ask:
        spread = msg.new_order.ask - msg.new_order.bid
        spread_bps = (spread / msg.new_order.price) * 10000  # Basis points

        # Only log significant spreads
        if spread_bps > 10:  # More than 10 bps
            return f"{msg.new_order.trade_pair}: Spread {spread_bps:.1f} bps"

    # Example: Track specific pairs
    if msg.new_order.trade_pair == "BTCUSD":
        return f"BTC: ${msg.new_order.price:.2f} ({msg.new_order.order_type})"

    return None


async def async_handler(messages: List[VantaWebSocketMessage]):
    # Process messages concurrently
    results = await asyncio.gather(*[process_message(msg) for msg in messages])

    # Print non-None results
    for result in filter(None, results):
        print(result)


# Run with async handler
client = VantaWebSocketClient(api_key="your_api_key_here")
client.subscribe()
client.run(async_handler)
```


## Best Practices for Trading Logic

1. **Use the Provided Client:**
   - Our WebSocketClient handles all the complexity of reliable message delivery
   - Focus on your trading strategy instead of communication infrastructure

2. **Process Messages Efficiently:**
    - Create separate handler functions for different order types
    - Use properties of the order object to make trading decisions
    - Track processed order UUIDs to avoid duplicate processing
    - Check the timestamp and time lag to ensure orders aren't stale before acting on them

3. **Optimize Network Usage:**
    - Always use compression for REST API requests to reduce bandwidth
    - For bulk data retrieval, use the REST API with compression
    - For real-time updates, use the WebSocket API
    - Monitor compression ratios to ensure they're working as expected

## Security Considerations

Store your API keys in a secure location with appropriate file permissions. The system will automatically reload the keys when the file changes.

For production, deploy the API server behind a reverse proxy that handles SSL/TLS termination:

```
Client <--HTTPS--> Nginx/Apache <--HTTP--> API Server
```


## Best Practices for Trading Logic

1. **Use the Provided Client:**
   - Our WebSocketClient handles all the complexity of reliable message delivery
   - Focus on your trading strategy instead of communication infrastructure

2. **Process Messages Efficiently:**
    - Create separate handler functions for different message types
    - Use a switch/case pattern based on message content
    - Track processed order uuids to avoid duplicate processing
    - Check the timestamp of orders to ensure they aren't stale before acting on them

3. **Optimize Network Usage:**
    - Always use compression for REST API requests to reduce bandwidth
    - For bulk data retrieval, use the REST API with compression
    - For real-time updates, use the WebSocket API

## Security Considerations

Store your API keys in a secure location with appropriate file permissions. The system will automatically reload the keys when the file changes.

For production, deploy the API server behind a reverse proxy that handles SSL/TLS termination:

```
Client <--HTTPS--> Nginx/Apache <--HTTP--> API Server
```

## Final Notes

The [Request Network](https://request.taoshi.io/) is a Taoshi product which serves subnet data while handling security, rate limiting, data customization, and provides a polished customer-facing and validator setup UI. Running this repo is a prerequisite to serving data on the Request Network.
