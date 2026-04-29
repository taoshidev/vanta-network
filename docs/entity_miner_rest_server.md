# Entity Miner REST Server

The `EntityMinerRestServer` is an extended version of the [miner REST server](miner_rest_server.md) that adds Hyperliquid subaccount management, real-time SSE streaming, and optional USDC payout automation.

It is started automatically when you run the entity miner with the `--entity-miner` flag. All endpoints from the base miner server are inherited, plus the entity-specific endpoints documented here.

**Port:** 8088 (same as the standard miner REST server)

## Inherited Endpoints

The entity miner server inherits all endpoints from `MinerRestServer`. See [miner_rest_server.md](miner_rest_server.md) for full documentation.

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/submit-order` | POST | API key | Submit trading orders synchronously |
| `/api/order-status/<uuid>` | GET | API key | Query order processing status |
| `/api/health` | GET | None | Health check (extended with entity fields) |

The health endpoint returns additional fields when running as an entity miner:

```json
{
  "status": "healthy",
  "service": "EntityMinerRestServer",
  "ws_connected": true,
  "hl_addresses_tracked": 12,
  "max_hl_traders": 100,
  "dashboard_cache_size": 12,
  "sse_subscribers": 3,
  "payment_daemon_active": false,
  "timestamp": 1702345690.123
}
```

## Entity-Specific Endpoints

### Get HL Trader Dashboard

`GET /api/hl/<hl_address>/dashboard`

Returns the cached dashboard for a Hyperliquid address. The cache is populated in real-time from the validator WebSocket and refreshed from the validator REST API when stale (10 second TTL).

**Authentication:** None required.

**Response (200):**
```json
{
  "timestamp_ms": 1702345690000,
  "synthetic_hotkey": "5GhDr3xy...abc_0",
  "hl_address": "0xabcd1234...",
  "subaccount_id": 0,
  "asset_class": "crypto",
  "status": "active",
  "created_at_ms": 1702345678901,
  "eliminated_at_ms": null,
  "challenge_period": { "bucket": "CHALLENGE", "start_time_ms": 1702345678901 },
  "drawdown": {
    "current_equity": 1.032,
    "daily_open_equity": 1.045,
    "eod_hwm": 1.065,
    "intraday_drawdown_pct": 1.24,
    "eod_drawdown_pct": 1.41
  }
}
```

**Error Responses:**
```json
// 404 — not found or cache invalidated and refresh failed
{ "status": "no_data", "hl_address": "0xabcd1234...", "message": "..." }
```

**Example:**
```bash
curl http://localhost:8088/api/hl/0xabcd1234.../dashboard
```

### Get HL Trader Order Events

`GET /api/hl/<hl_address>/events`

Returns the ring-buffered order event history for a Hyperliquid address (max 100 events per address). Events include accepted fills and rejected orders received from the validator WebSocket.

**Authentication:** None required.

**Query Parameters:**
- `since` (int, optional): Only return events with `timestamp_ms > since`. Defaults to 0 (all events).

**Response (200):**
```json
{
  "hl_address": "0xabcd1234...",
  "events": [
    {
      "timestamp_ms": 1702345678901,
      "hl_address": "0xabcd1234...",
      "trade_pair": "BTCUSD",
      "order_type": "LONG",
      "status": "accepted",
      "error_message": "",
      "fill_hash": "0xdeadbeef...",
      "synthetic_hotkey": "5GhDr3xy...abc_0"
    },
    {
      "timestamp_ms": 1702345699000,
      "hl_address": "0xabcd1234...",
      "trade_pair": "ETHUSD",
      "order_type": "SHORT",
      "status": "rejected",
      "error_message": "Insufficient buying power",
      "fill_hash": "",
      "synthetic_hotkey": "5GhDr3xy...abc_0"
    }
  ],
  "count": 2
}
```

**Event Fields:**
- `timestamp_ms`: When the event was received
- `hl_address`: The Hyperliquid wallet address
- `trade_pair`: Trade pair identifier (e.g., `"BTCUSD"`)
- `order_type`: `"LONG"`, `"SHORT"`, or `"FLAT"`
- `status`: `"accepted"` or `"rejected"`
- `error_message`: Error reason for rejected orders (empty string for accepted)
- `fill_hash`: HL fill hash for accepted orders (empty string for rejected)
- `synthetic_hotkey`: The subaccount's synthetic hotkey

**Example:**
```bash
# All events
curl http://localhost:8088/api/hl/0xabcd1234.../events

# Only events since a specific timestamp
curl "http://localhost:8088/api/hl/0xabcd1234.../events?since=1702345678000"
```

### Stream HL Trader Events (SSE)

`GET /api/hl/<hl_address>/stream`

Server-Sent Events (SSE) stream for real-time order acceptance and rejection notifications. Connects are long-lived — the server pushes events as they arrive from the validator WebSocket.

**Authentication:** None required.

**Response:** `Content-Type: text/event-stream`

Each message is a JSON-encoded object on a `data:` line:

```
data: {"type": "event", "data": {"timestamp_ms": 1702345678901, "status": "accepted", ...}}

data: {"type": "dashboard", "data": {"timestamp_ms": 1702345678901, "synthetic_hotkey": "...", ...}}

: heartbeat
```

**Message Types:**
- `event`: An order acceptance or rejection event (same schema as `/events`)
- `dashboard`: A full dashboard update (same schema as `/dashboard`)
- `: heartbeat`: Keepalive comment sent every 30 seconds when no other messages arrive

**Example (curl):**
```bash
curl -N http://localhost:8088/api/hl/0xabcd1234.../stream
```

**Example (JavaScript EventSource):**
```javascript
const es = new EventSource('http://localhost:8088/api/hl/0xabcd1234.../stream');
es.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'event') console.log('Order event:', msg.data);
  if (msg.type === 'dashboard') console.log('Dashboard update:', msg.data);
};
```

### Create Standard Subaccount

`POST /api/create-subaccount`

Creates a new standard (non-HL) trading subaccount under this entity miner. The server signs the request with the entity coldkey and forwards it to the validator.

**Authentication:** API key required.

**Required Headers:**
```
Content-Type: application/json
Authorization: Bearer <api_key>
```

**Request Body:**
```json
{
  "asset_class": "crypto",
  "account_size": 50000.0,
}
```

**Parameters:**
- `asset_class` (string, required): `"crypto"`, `"forex"`, or `"equities"`
- `account_size` (float, required): Account size in USD. Must be positive.

**Success Response (200):**
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

**Error Responses:**

| Code | Cause |
|------|-------|
| 400 | Missing/invalid field (`asset_class`, `account_size`) |
| 401 | Invalid or missing API key |
| 403 | Max HL traders limit reached (HL path only) |
| 500 | Wallet not configured or signing error |
| 503 | Validator unreachable |
| 504 | Request to validator timed out |

**Example:**
```bash
curl -X POST http://localhost:8088/api/create-subaccount \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"asset_class": "crypto", "account_size": 50000}'
```

**Notes:**
- Requires `wallet_name`, `wallet_hotkey`, `wallet_password`, and `validator_url` in `miner_secrets.json`
- Once created, use the returned `subaccount_id` in `/api/submit-order` requests via the `subaccount_id` field

### Create Hyperliquid-Linked Subaccount

`POST /api/create-hl-subaccount`

Creates a new Hyperliquid-linked subaccount. Trades detected on Hyperliquid for the given `hl_address` will be automatically forwarded as Vanta signals. This route is an alias for `/api/create-subaccount` — the `hl_address` field selects the HL path.

**Authentication:** API key required.

**Required Headers:**
```
Content-Type: application/json
Authorization: Bearer <api_key>
```

**Request Body:**
```json
{
  "hl_address": "0xabcd1234...ef56",
  "account_size": 50000.0,
  "payout_address": "0xAbCd...1234",
}
```

**Parameters:**
- `hl_address` (string, required): Hyperliquid wallet address (`0x` + 40 hex characters). Case-insensitive; stored normalized to lowercase.
- `account_size` (float, required): Account size in USD. Must be positive.
- `payout_address` (string, optional): EVM address for USDC payouts (`0x` + 40 hex characters). If omitted, USDC payouts will not be sent for this subaccount.

The `asset_class` is always `"crypto"` for HL-linked subaccounts and does not need to be provided.

**Success Response (200):**
```json
{
  "status": "success",
  "message": "Subaccount 1 created successfully",
  "subaccount": {
    "subaccount_id": 1,
    "subaccount_uuid": "550e8400-e29b-41d4-a716-446655440001",
    "synthetic_hotkey": "5GhDr3xy...abc_1",
    "asset_class": "crypto",
    "account_size": 50000.0,
    "status": "active",
    "created_at_ms": 1702345678901,
    "eliminated_at_ms": null
  }
}
```

**Error Responses:** Same as `/api/create-subaccount`, plus:
- `400 Bad Request` — `hl_address` or `payout_address` is not a valid EVM address
- `403 Forbidden` — `max_hl_traders` limit reached

**Example:**
```bash
curl -X POST http://localhost:8088/api/create-hl-subaccount \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "hl_address": "0xabcd1234...ef56",
    "account_size": 50000,
    "payout_address": "0xAbCd...1234"
  }'
```

**Notes:**
- The HL address is stored in `entity_hl_mappings.json` alongside `miner_secrets.json` and persists across restarts
- After creation, the gateway subscribes to the validator WebSocket for dashboard updates for this address
- If `max_hl_traders` is configured, the gateway enforces it before forwarding the request to the validator

## WebSocket Connection

The entity miner gateway maintains a persistent WebSocket connection to the validator at startup. This connection:

- Authenticates via `Authorization: Bearer <validator_api_key>` header sent during the WebSocket upgrade handshake — no post-connect auth message is required
- Receives real-time dashboard updates for all registered subaccounts
- Receives order rejection notifications
- Populates the dashboard cache and SSE subscriber queues

The API key is obtained via `vanta entity apikey` and stored as `validator_api_key` in `miner_secrets.json`.

On successful authentication, the validator immediately sends a JSON message containing:
- `subscribed_subaccounts`: list of synthetic hotkeys the gateway has been auto-subscribed to
- `hl_mappings`: mapping of `synthetic_hotkey → hl_address` for all HL-linked subaccounts

Connection is automatic and includes exponential backoff reconnection (1s → 60s max). The validator sends the full HL address mapping on every successful connection, which the gateway uses to keep its local `hl_address ↔ synthetic_hotkey` mapping up to date.

## USDC Payment Daemon

When `enable_auto_payouts` is `true` (and required secrets are configured), the gateway runs a background daemon that automatically distributes USDC earnings to HL trader payout addresses on a configurable weekly schedule.

**Required configuration:**
- `usdc_private_key`: Private key of the wallet funding USDC payments
- `validator_payout_api_key`: API key for querying the validator's payout endpoint
- `validator_url`: Validator REST API URL

**Schedule:** Defaults to Sunday at 01:00 UTC. Configurable via `payout_schedule_day` and `payout_schedule_hour` in `miner_secrets.json`.

**Payment ledger:** All payment history is recorded in `entity_payment_ledger.json` (alongside `miner_secrets.json`). Pending payments from a previous run are automatically re-checked on startup.

Payment status is reported in the `/api/health` response via `payment_daemon_active`.
