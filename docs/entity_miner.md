# Running an Entity Miner

An entity miner allows a single registered hotkey to manage multiple independent trading subaccounts. Each subaccount gets its own synthetic hotkey, independent position tracking, performance ledger, and rate limits. This enables running multiple trading strategies under one entity.

## Prerequisites

- Python 3.10+
- Registered miner hotkey on the Vanta Network (netuid 8 mainnet, 116 testnet)
- Bittensor wallet with coldkey and hotkey
- Access to a validator's REST API and WebSocket endpoints

## 1. Environment Setup

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python3 -m pip install -e .
```

## 2. Configure Miner Secrets

Create `mining/miner_secrets.json` with your wallet credentials and validator connection details:

```json
{
  "api_key": "your_api_key",
  "wallet_name": "your_wallet_name",
  "wallet_hotkey": "your_hotkey_name",
  "wallet_password": "your_wallet_password",
  "validator_url": "http://validator_ip:48888",
  "validator_ws_host": "validator_ip",
  "validator_ws_port": 8765
}
```

| Field | Description |
|-------|-------------|
| `api_key` | API key for authenticating requests to your miner's REST server |
| `wallet_name` | Bittensor wallet name |
| `wallet_hotkey` | Bittensor hotkey name |
| `wallet_password` | Wallet coldkey password (used for signing subaccount creation requests) |
| `validator_url` | Validator REST API URL (port 48888) |
| `validator_ws_host` | Validator WebSocket host (for real-time dashboard/rejection streams) |
| `validator_ws_port` | Validator WebSocket port (default: 8765) |

Register your entity miner's public endpoint URL with the validator, add:

```json
{
  "entity_endpoint_url": "https://your-domain.com:8089"
}
```

Or set the `ENTITY_MINER_ENDPOINT_URL` environment variable instead.

## 3. Start the Miner

Run the miner with the `--entity-miner` flag to enable the Entity Miner Gateway:

```bash
python neurons/miner.py \
  --netuid 8 \
  --wallet.name <wallet> \
  --wallet.hotkey <hotkey> \
  --entity-miner
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--netuid` | 1 | Subnet UID (8 for mainnet, 116 for testnet) |
| `--entity-miner` | disabled | Enable the Entity Miner Gateway |
| `--entity-api-port` | 8089 | Port for the Entity Miner Gateway REST server |
| `--api-host` | 0.0.0.0 | Host address for both miner and entity API servers |
| `--api-rest-port` | 8088 | Port for the standard Miner REST API |
| `--run-position-inspector` | disabled | Enable the position inspector thread |
| `--start-dashboard` | disabled | Start the miner dashboard frontend |

This starts two API servers:
- **Miner REST API** on port 8088 — for submitting orders (including subaccount orders)
- **Entity Miner Gateway** on port 8089 — for subaccount management and Hyperliquid monitoring

## 4. Create Subaccounts

Once the miner is running, create subaccounts via the Entity Miner Gateway.

### Standard Subaccount

```bash
curl -X POST http://localhost:8089/api/create-subaccount \
  -H "Content-Type: application/json" \
  -H "Authorization: your_api_key" \
  -d '{
    "asset_class": "crypto",
    "account_size": 10000.0
  }'
```

### Hyperliquid-Linked Subaccount

```bash
curl -X POST http://localhost:8089/api/create-hl-subaccount \
  -H "Content-Type: application/json" \
  -H "Authorization: your_api_key" \
  -d '{
    "hl_address": "0xYourHyperliquidAddress",
    "account_size": 10000.0,
    "payout_address": "0xOptionalPayoutAddress"
  }'
```

### Response

```json
{
  "status": "success",
  "message": "Subaccount created successfully",
  "subaccount": {
    "subaccount_id": 0,
    "subaccount_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "synthetic_hotkey": "5xxx..._0",
    "account_size": 10000.0,
    "asset_class": "crypto",
    "status": "active"
  }
}
```

### Create Subaccount Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_class` | string | Yes (standard) | `"crypto"` or `"forex"` |
| `account_size` | float | Yes | Account size in USD |
| `hl_address` | string | Yes (HL) | Hyperliquid wallet address |
| `payout_address` | string | No | Optional payout address for HL subaccounts |
| `admin` | bool | No | Admin subaccount flag |

## 5. Submit Orders to Subaccounts

Send orders to specific subaccounts by including `subaccount_id` in your order request to the Miner REST API (port 8088):

```bash
curl -X POST http://localhost:8088/api/submit-order \
  -H "Content-Type: application/json" \
  -H "Authorization: your_api_key" \
  -d '{
    "execution_type": "MARKET",
    "trade_pair": "BTCUSD",
    "order_type": "LONG",
    "leverage": 0.1,
    "subaccount_id": 0
  }'
```

The `subaccount_id` maps to the synthetic hotkey `{entity_hotkey}_{subaccount_id}`. Each subaccount has independent rate limits, so you can submit orders across subaccounts in parallel.

For full order submission documentation (execution types, order sizing, limit orders, etc.), see [miner_rest_server.md](miner_rest_server.md).

## 6. Monitoring

### Health Check

```bash
curl http://localhost:8089/api/health
```

```json
{
  "status": "healthy",
  "service": "EntityMinerRestServer",
  "ws_connected": true,
  "hl_addresses_tracked": 5,
  "dashboard_cache_size": 3,
  "sse_subscribers": 0,
  "timestamp": 1700000000.0
}
```

### Hyperliquid Dashboard (HL subaccounts only)

Get cached dashboard data for a Hyperliquid address:

```bash
curl http://localhost:8089/api/hl/<hl_address>/dashboard \
  -H "Authorization: your_api_key"
```

### Order Events (HL subaccounts only)

Get the ring buffer of recent order events (accepted/rejected):

```bash
curl "http://localhost:8089/api/hl/<hl_address>/events?since=1700000000000" \
  -H "Authorization: your_api_key"
```

### Real-Time SSE Stream (HL subaccounts only)

Subscribe to a server-sent events stream for real-time dashboard updates and rejection notifications:

```bash
curl -N http://localhost:8089/api/hl/<hl_address>/stream \
  -H "Authorization: your_api_key"
```

## Key Concepts

### Synthetic Hotkeys

Each subaccount is identified by a synthetic hotkey with the format `{entity_hotkey}_{subaccount_id}`. For example, if your entity hotkey is `5abc...xyz` and you create subaccount 0, the synthetic hotkey is `5abc...xyz_0`.

- Entity hotkeys **cannot** place orders directly — only subaccounts can
- Eliminated subaccount IDs are never reused; new subaccounts always get the next incremental ID

### Challenge Period

New subaccounts enter a 90-day challenge period upon creation:
- **Pass criteria**: Achieve 3% returns with less than 6% max drawdown (instantaneous pass)
- **Failure**: Subaccounts that don't meet the criteria within 90 days are eliminated
- Assessment runs automatically via the validator's EntityServer daemon every 5 minutes

### Limits

| Constraint | Value |
|------------|-------|
| Max active subaccounts per entity | 500 |
| Max account size | $100,000 USD |
| Challenge period duration | 90 days |
| Challenge period returns threshold | 3% |
| Challenge period drawdown threshold | 6% |

### Elimination

Subaccounts can be eliminated for:
- **Challenge period failure** — not meeting 3%/6% criteria within 90 days
- **Max drawdown** — exceeding 10% total drawdown
- **Plagiarism** — detected order similarity with other miners

Eliminated subaccounts are permanently retired. Create a new subaccount to replace an eliminated one.

### Scoring

All subaccount performance metrics (debt ledgers) are aggregated into a single entity-level debt ledger keyed by the entity hotkey. This aggregated ledger is used for weight calculation and scoring on the network. Eliminated subaccounts are excluded from the aggregation.

## Entity Miner Gateway Endpoints Summary

| Endpoint | Method | Port | Description |
|----------|--------|------|-------------|
| `/api/create-subaccount` | POST | 8089 | Create standard subaccount |
| `/api/create-hl-subaccount` | POST | 8089 | Create Hyperliquid-linked subaccount |
| `/api/hl/<addr>/dashboard` | GET | 8089 | Cached HL dashboard data |
| `/api/hl/<addr>/events` | GET | 8089 | Order event ring buffer |
| `/api/hl/<addr>/stream` | GET | 8089 | SSE real-time stream |
| `/api/health` | GET | 8089 | Health check |
| `/api/submit-order` | POST | 8088 | Submit orders (with `subaccount_id`) |

## Troubleshooting

### Entity Miner Gateway fails to start
- Verify `mining/miner_secrets.json` exists and contains valid wallet credentials
- Check that the `wallet_password` decrypts the coldkey successfully
- Ensure port 8089 is not already in use

### WebSocket not connecting
- Confirm `validator_ws_host` and `validator_ws_port` in secrets match the validator's WebSocket server
- Check network connectivity to the validator
- The gateway retries with exponential backoff (1s to 60s) on connection failures

### Subaccount creation fails
- Ensure your entity is registered with the validator
- Verify the validator REST API is reachable at the configured `validator_url`
- Check that you haven't exceeded the 500 active subaccount limit

### Orders rejected for subaccount
- Confirm the subaccount is active (not eliminated)
- Verify you're using `subaccount_id` (not the full synthetic hotkey) in the order request
- Check that the subaccount's asset class matches the trade pair
