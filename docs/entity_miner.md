# Entity Miner

Entity miners are a type of participant in the Vanta Network distinct from regular miners. Rather than operating a single trading account, an entity creates and manages multiple **subaccounts** — each acting as an independent trader competing in the network.

The **entity hotkey** identifies the operator on the validator. Under it, the entity creates **subaccounts** that submit orders and earn incentives. Each subaccount is identified by a **synthetic hotkey** in the format `{entity_hotkey}_{subaccount_id}` (e.g., `5GhDr..._0`, `5GhDr..._1`). These synthetic hotkeys participate in trading exactly like regular miners.

**Key rule: Entity hotkeys cannot submit orders directly. Only subaccounts can place trades.**

## Basic Rules

1. Entity hotkeys must be registered on the Bittensor network and have sufficient Theta collateral.
2. An entity pays a one-time registration fee of **5,000 Theta**, which is permanently slashed on registration.
3. Each subaccount requires collateral proportional to its account size (see [Collateral Requirements](#collateral-requirements)).
4. Each subaccount selects an asset class (`crypto` or `forex`) at creation — this **cannot be changed**.
5. New subaccounts enter a **challenge period** with stricter thresholds and reduced leverage (see [Challenge Period](#challenge-period--subaccount-lifecycle)).
6. Entity hotkeys **cannot place orders**. Orders must be submitted using the subaccount's synthetic hotkey.
7. Subaccounts follow the same trading rules as regular miners: uni-directional positions, leverage limits, market hours, rate limits, etc.
8. A maximum of **5 entities** can be registered on the network at any time.
9. Each entity supports multiple subaccounts.
10. **CRITICAL**: Never reuse synthetic hotkeys from eliminated subaccounts. Eliminated synthetic hotkeys are permanently blacklisted.

## Collateral Requirements

Collateral is denominated in **Theta**, deposited via the Vanta CLI.

| Action | Theta Required |
|---|---|
| Entity registration (one-time) | 5,000 Theta |
| Subaccount with $5,000 account size | 1 Theta |
| Subaccount with $10,000 account size | 2 Theta |
| Subaccount with $25,000 account size | 5 Theta |
| Subaccount with $50,000 account size | 10 Theta |
| Subaccount with $100,000 account size (max) | 20 Theta |

**Formula:** `required_theta = account_size_usd / 5,000`

The maximum account size per subaccount is **$100,000 USD**. Collateral for each subaccount is slashed asynchronously after creation. A subaccount starts in `pending` status and transitions to `active` once the slash succeeds. If the slash fails (e.g., insufficient balance), the subaccount is marked `failed`.

## Challenge Period & Subaccount Lifecycle

Every new subaccount enters a challenge period. The lifecycle is:

```
pending → active → [SUBACCOUNT_CHALLENGE] → [SUBACCOUNT_FUNDED] → [SUBACCOUNT_ALPHA]
                                ↓
                           eliminated
```

| Stage | Bucket | Description |
|---|---|---|
| SUBACCOUNT_CHALLENGE | 1× dust | Challenge phase — reduced leverage, no payout |
| SUBACCOUNT_FUNDED | earning | Passed challenge — full leverage, earns payouts |
| SUBACCOUNT_ALPHA | earning | 90+ days in FUNDED — continues earning payouts |
| eliminated | — | Permanently removed from competition |

### Challenge Period Requirements

**To pass the challenge period**, a subaccount must achieve:

| Asset Class | Minimum Return Required |
|---|-------------------------|
| Forex | ≥ 8%                    |
| Crypto | ≥ 10%                   |

Passing is evaluated continuously — a subaccount is promoted immediately upon hitting the threshold. Assessment runs automatically via the validator's EntityServer daemon every 5 minutes.

**Elimination during challenge:** A subaccount is eliminated if its intraday drawdown or drawdown from the end-of-day high-water mark reaches **5%**.

**Leverage reduction:** During the challenge period, a subaccount's maximum leverage and account multiplier are divided by **4** to limit risk exposure.

### After the Challenge Period

Once in SUBACCOUNT_FUNDED:
- Standard **8% max drawdown** elimination applies (same as regular miners).
- After **90 days** in SUBACCOUNT_FUNDED meeting the thresholds, the subaccount is promoted to SUBACCOUNT_ALPHA and is eligible for additional funding.

## Getting Started

### Prerequisites

- Python 3.10+
- [Bittensor](https://github.com/opentensor/bittensor#install)
- Vanta CLI installed:
  ```bash
  pip install git+https://github.com/taoshidev/vanta-cli.git
  ```

### 1. Install Vanta

Clone the repository:

```bash
git clone https://github.com/taoshidev/vanta-network.git
cd vanta-network
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
. venv/bin/activate
```

Install dependencies:

```bash
export PIP_NO_CACHE_DIR=1
pip install -r requirements.txt
python3 -m pip install -e .
```

### 2. Create Wallets

Create a coldkey and hotkey for your entity:

```bash
btcli wallet new_coldkey --wallet.name <wallet>
btcli wallet new_hotkey --wallet.name <wallet> --wallet.hotkey <entity>
```

Save your mnemonics.

### 3. Register on the Subnet

Register your entity hotkey on the subnet:

```bash
# Mainnet (netuid 8)
btcli subnet register --wallet.name <wallet> --wallet.hotkey <entity>

# Testnet (netuid 116)
btcli subnet register --wallet.name <wallet> --wallet.hotkey <entity> --subtensor.network test --netuid 116
```

| Environment | Netuid |
|---|---|
| Mainnet | 8 |
| Testnet | 116 |

### 4. Add Stake

Before depositing Theta collateral, add TAO stake for your hotkey:

```bash
# Mainnet
btcli stake add --wallet.name <wallet> --wallet.hotkey <entity>

# Testnet
btcli stake add --wallet.name <wallet> --wallet.hotkey <entity> --subtensor.network test
```

### 5. Deposit Collateral

Deposit Theta collateral via the Vanta CLI. You need at least **5,000 Theta** to register an entity, plus additional Theta for each subaccount you plan to create.

```bash
# Mainnet
vanta collateral deposit --wallet-name <wallet> --wallet-hotkey <entity> --amount <theta>

# Testnet
vanta collateral deposit --wallet-name <wallet> --wallet-hotkey <entity> --amount <theta> --network test
```

Check your balance:

```bash
vanta collateral list --wallet-name <wallet> --wallet-hotkey <entity>
```

Withdraw collateral:

```bash
vanta collateral withdraw --wallet-name <wallet> --wallet-hotkey <entity> --amount <theta>
```

### 6. Register the Entity

Register your entity hotkey on the validator. This costs **5,000 Theta** (permanently slashed):

```bash
# Mainnet
vanta entity register --wallet-name <wallet> --wallet-hotkey <entity>

# Testnet
vanta entity register --wallet-name <wallet> --wallet-hotkey <entity> --network test
```

On success, the entity hotkey is assigned to the `ENTITY` bucket and receives a baseline 4× dust weight in the incentive system.

### 7. Configure Miner Secrets

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
|---|---|
| `api_key` | API key for authenticating requests to your miner's REST server |
| `wallet_name` | Bittensor wallet name |
| `wallet_hotkey` | Bittensor hotkey name |
| `wallet_password` | Wallet coldkey password (used for signing subaccount creation requests) |
| `validator_url` | Validator REST API URL (port 48888) |
| `validator_ws_host` | Validator WebSocket host (for real-time dashboard/rejection streams) |
| `validator_ws_port` | Validator WebSocket port (default: 8765) |
| `max_hl_traders` | Maximum number of Hyperliquid traders that can be registered (optional, no limit if unset). Can also be set via `MAX_HL_TRADERS` env var (env var takes precedence). |

To register your entity miner's public endpoint URL with the validator, add:

```json
{
  "entity_endpoint_url": "https://your-domain.com:8088"
}
```

Or set the `ENTITY_MINER_ENDPOINT_URL` environment variable instead.

### 8. Run the Miner

Run the miner with the `--entity-miner` flag to enable the Entity Miner Gateway:

```bash
# Mainnet
python neurons/miner.py \
  --netuid 8 \
  --wallet.name <wallet> \
  --wallet.hotkey <entity> \
  --entity-miner

# Testnet
python neurons/miner.py \
  --netuid 116 \
  --subtensor.network test \
  --wallet.name <wallet> \
  --wallet.hotkey <entity> \
  --entity-miner
```

### Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--netuid` | 1 | Subnet UID (8 for mainnet, 116 for testnet) |
| `--entity-miner` | disabled | Enable the Entity Miner Gateway |
| `--entity-api-port` | 8088 | Port for the Entity Miner Gateway REST server |
| `--api-host` | 0.0.0.0 | Host address for the API server |
| `--api-rest-port` | 8088 | Port for the standard Miner REST API |
| `--run-position-inspector` | disabled | Enable the position inspector thread |
| `--start-dashboard` | disabled | Start the miner dashboard frontend |

This starts the miner REST API on port 8088, which handles both order submission and subaccount management.

### 9. Create Subaccounts

Create subaccounts under your entity via the Vanta CLI or directly via the Entity Miner Gateway.

#### Standard Subaccount

```bash
# Via Vanta CLI
vanta entity create-subaccount \
  --wallet-name <wallet> \
  --wallet-hotkey <entity> \
  --account-size <usd_amount> \
  --asset-class <crypto|forex>

# Via Entity Miner Gateway (requires miner running)
curl -X POST http://localhost:8088/api/create-subaccount \
  -H "Content-Type: application/json" \
  -H "Authorization: your_api_key" \
  -d '{"asset_class": "crypto", "account_size": 10000.0}'
```

#### Hyperliquid-Linked Subaccount

Hyperliquid-linked subaccounts automatically forward trades from a Hyperliquid address as Vanta signals. They always use `crypto` as their asset class.

```bash
# Via Vanta CLI
vanta entity create-hl-subaccount \
  --wallet-name <wallet> \
  --wallet-hotkey <entity> \
  --account-size <usd_amount> \
  --hl-address <0x...>

# Via Entity Miner Gateway (requires miner running)
curl -X POST http://localhost:8088/api/create-hl-subaccount \
  -H "Content-Type: application/json" \
  -H "Authorization: your_api_key" \
  -d '{
    "hl_address": "0xYourHyperliquidAddress",
    "account_size": 10000.0,
    "payout_address": "0xOptionalPayoutAddress"
  }'
```

#### Response

```json
{
  "status": "success",
  "message": "Subaccount created successfully",
  "subaccount": {
    "subaccount_id": 0,
    "subaccount_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "synthetic_hotkey": "5GhDr..._0",
    "account_size": 10000.0,
    "asset_class": "crypto",
    "status": "active"
  }
}
```

#### Subaccount Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `asset_class` | string | Yes (standard) | `"crypto"` or `"forex"` |
| `account_size` | float | Yes | Account size in USD |
| `hl_address` | string | Yes (HL) | Hyperliquid wallet address (0x + 40 hex chars) |
| `payout_address` | string | No | Optional EVM payout address for HL subaccounts |
| `admin` | bool | No | Admin subaccount flag (default: false) |

### 10. Submit Orders

Send orders to specific subaccounts by including `subaccount_id` in your order request:

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

The `subaccount_id` is the integer returned when the subaccount was created (e.g., `0`, `1`, `2`). It maps to the synthetic hotkey `{entity_hotkey}_{subaccount_id}`. Each subaccount has independent rate limits, so orders across subaccounts can be submitted in parallel.

**Do not use the entity hotkey directly** — it will be rejected. Only subaccount orders (identified by `subaccount_id`) are accepted.

For full order submission documentation (execution types, order sizing, limit orders, etc.), see [miner_rest_server.md](miner_rest_server.md).

## Monitoring

### Health Check

```bash
curl http://localhost:8088/api/health \
  -H "Authorization: your_api_key"
```

```json
{
  "status": "healthy",
  "service": "EntityMinerRestServer",
  "ws_connected": true,
  "hl_addresses_tracked": 5,
  "max_hl_traders": 50,
  "dashboard_cache_size": 3,
  "sse_subscribers": 0,
  "timestamp": 1700000000.0
}
```

### Hyperliquid Dashboard (HL subaccounts only)

Get cached dashboard data for a Hyperliquid address:

```bash
curl http://localhost:8088/api/hl/<hl_address>/dashboard \
  -H "Authorization: your_api_key"
```

### Order Events (HL subaccounts only)

Get the ring buffer of recent order events (accepted/rejected):

```bash
curl "http://localhost:8088/api/hl/<hl_address>/events?since=1700000000000" \
  -H "Authorization: your_api_key"
```

### Real-Time SSE Stream (HL subaccounts only)

Subscribe to a server-sent events stream for real-time dashboard updates and rejection notifications:

```bash
curl -N http://localhost:8088/api/hl/<hl_address>/stream \
  -H "Authorization: your_api_key"
```

## Payout Computation

Entity miner payouts use the same **debt-based scoring system** as regular miners, with a few differences:

- **SUBACCOUNT_CHALLENGE**: No payout — subaccounts in the challenge period do not earn incentives.
- **SUBACCOUNT_FUNDED** and **SUBACCOUNT_ALPHA**: Subaccounts earn payouts based on their PnL performance checkpoints, exactly like MAINCOMP miners.
- **Entity hotkey**: Receives a baseline **4× dust weight** (the minimum floor weight) regardless of subaccount performance.

The payout for a subaccount is calculated from its debt ledger checkpoints that fall within the SUBACCOUNT_FUNDED or SUBACCOUNT_ALPHA status window. Performance is weighted 100% on average daily PnL (same as regular miners). All subaccount debt ledgers are aggregated into a single entity-level ledger for weight calculation. Eliminated subaccounts are excluded from aggregation.

**Dust weight multipliers:**

| Bucket | Dust Multiplier |
|---|---|
| ENTITY (entity hotkey) | 4× dust |
| MAINCOMP (regular miner) | 3× dust |
| SUBACCOUNT_FUNDED / SUBACCOUNT_ALPHA | earning (proportional to debt) |
| SUBACCOUNT_CHALLENGE | 1× dust |
| UNKNOWN | 0× dust |

To query a subaccount's payout for a time period:

```bash
POST https://validator.<mainnet|testnet>.vantatrading.io/entity/subaccount/payout
Authorization: <api_key>
Content-Type: application/json

{
  "subaccount_uuid": "<uuid>",
  "start_time_ms": 1700000000000,
  "end_time_ms": 1700604800000
}
```

Response:

```json
{
  "status": "success",
  "payout_data": {
    "hotkey": "5GhDr..._0",
    "total_checkpoints": 14,
    "checkpoints": [...],
    "payout": 123.45
  },
  "timestamp": 1700604800000
}
```

## REST API Reference

### Validator REST Server

All validator endpoints require a valid API key (tier 200) in the `Authorization` header.

**Base URL:**
- Mainnet: `https://validator.mainnet.vantatrading.io`
- Testnet: `https://validator.testnet.vantatrading.io`
- Local: `http://<validator-ip>:48888`

| Method | Endpoint | Description |
|---|---|---|
| POST | `/entity/register` | Register a new entity (requires coldkey signature + 5,000 Theta) |
| POST | `/entity/create-subaccount` | Create a subaccount — standard or HL-linked (requires coldkey signature + collateral) |
| POST | `/entity/create-hl-subaccount` | Alias for `/entity/create-subaccount` with `hl_address` |
| GET | `/entity/<entity_hotkey>` | Get entity data and subaccount list |
| GET | `/entities` | List all registered entities |
| GET | `/entity/subaccount/<synthetic_hotkey>` | Get subaccount dashboard data |
| GET | `/v2/entity/subaccount/<synthetic_hotkey>` | Get v2 subaccount dashboard data |
| POST | `/entity/subaccount/payout` | Calculate payout for a subaccount by UUID and time range |
| POST | `/entity/subaccount/eliminate` | Manually eliminate a subaccount |

#### POST /entity/register

```json
{
  "entity_coldkey": "<coldkey_ss58>",
  "entity_hotkey": "<hotkey_ss58>",
  "signature": "<coldkey_signature>"
}
```

The signature is produced by signing `{"entity_coldkey": "...", "entity_hotkey": "..."}` (JSON, sorted keys) with the coldkey.

#### POST /entity/create-subaccount

Standard subaccount:

```json
{
  "entity_coldkey": "<coldkey_ss58>",
  "entity_hotkey": "<hotkey_ss58>",
  "account_size": 10000.0,
  "asset_class": "crypto",
  "signature": "<coldkey_signature>"
}
```

HL-linked subaccount (include `hl_address`; `asset_class` is always `"crypto"`):

```json
{
  "entity_coldkey": "<coldkey_ss58>",
  "entity_hotkey": "<hotkey_ss58>",
  "account_size": 10000.0,
  "hl_address": "0x1234...abcd",
  "payout_address": "0xAbCd...1234",
  "signature": "<coldkey_signature>"
}
```

The signature for a standard subaccount covers `{account_size, admin, asset_class, entity_coldkey, entity_hotkey}` (JSON, sorted keys). For an HL subaccount, `hl_address` (and `payout_address` if provided) are also included in the signed payload.

Response:

```json
{
  "status": "success",
  "message": "Subaccount created successfully",
  "subaccount": {
    "subaccount_id": 0,
    "subaccount_uuid": "uuid-string",
    "synthetic_hotkey": "5GhDr..._0",
    "account_size": 10000.0,
    "asset_class": "crypto"
  }
}
```

### Entity Miner Gateway (port 8088)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/create-subaccount` | Create a standard subaccount (proxies to validator) |
| POST | `/api/create-hl-subaccount` | Create an HL-linked subaccount (proxies to validator) |
| GET | `/api/hl/<hl_address>/dashboard` | Cached HL dashboard data |
| GET | `/api/hl/<hl_address>/events` | Order event ring buffer |
| GET | `/api/hl/<hl_address>/stream` | SSE real-time stream |
| GET | `/api/health` | Health check |

### Miner REST Server (port 8088)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/submit-order` | Submit a trading order for a subaccount (include `subaccount_id`) |
| GET | `/api/order-status/<order_uuid>` | Query order processing status |
| GET | `/api/health` | Health check |

## Key Concepts

### Synthetic Hotkeys

Each subaccount is identified by a synthetic hotkey with the format `{entity_hotkey}_{subaccount_id}`. For example, if your entity hotkey is `5abc...xyz` and you create subaccount 0, the synthetic hotkey is `5abc...xyz_0`.

- Entity hotkeys **cannot** place orders directly — only subaccounts can
- Eliminated subaccount IDs are never reused; new subaccounts always get the next incremental ID

### Limits

| Constraint | Value                                                                    |
|---|--------------------------------------------------------------------------|
| Max entities on the network | 5                                                                        |
| Max HL traders per entity miner | Configurable via `max_hl_traders` / `MAX_HL_TRADERS` (no limit if unset) |
| Max account size per subaccount | $100,000 USD                                                             |
| Challenge period return threshold | ≥ 8% for fx, 10% for crypto                                              |
| Challenge period drawdown threshold | 5% elimination                                                           |

### Elimination

Subaccounts can be eliminated for:
- **Challenge period failure** — drawdown exceeds 5% before achieving the 8% return threshold
- **Max drawdown** — exceeding 8% drawdown after the challenge period
- **Plagiarism** — detected order similarity with other miners

Eliminated subaccounts are permanently retired. Create a new subaccount to replace an eliminated one.

## Dashboard

Monitor subaccount performance at:

- Mainnet: https://dashboard.taoshi.io
- Testnet: https://testnet.dashboard.taoshi.io

Log in using a [polkadot.js](https://polkadot.js.org/extension/) browser wallet. API key tier 200 is required to access subaccount dashboard data.

## Troubleshooting

### Entity Miner Gateway fails to start
- Verify `mining/miner_secrets.json` exists and contains valid wallet credentials
- Check that `wallet_password` decrypts the coldkey successfully
- Ensure port 8088 is not already in use

### WebSocket not connecting
- Confirm `validator_ws_host` and `validator_ws_port` in secrets match the validator's WebSocket server
- Check network connectivity to the validator
- The gateway retries with exponential backoff (1s to 60s) on connection failures

### Subaccount creation fails
- Ensure your entity is registered with the validator
- Verify you have sufficient Theta collateral for the requested account size
- Verify the validator REST API is reachable at the configured `validator_url`
- Check that you haven't exceeded the maximum number of active subaccounts

### Orders rejected for subaccount
- Confirm the subaccount is active (not eliminated or still in `pending` status)
- Verify you're using `subaccount_id` (not the full synthetic hotkey) in the order request
- Check that the subaccount's asset class matches the trade pair

## Security Notes

- Do not expose your coldkey or private keys.
- Always test on testnet (netuid 116) before mainnet.
- Do not reuse the password of your mainnet wallet on testnet.
- The entity coldkey is used to sign subaccount creation requests — keep it secure.
- Do not commit `mining/miner_secrets.json` to version control.
