# Entity Miner

Entity miners are a type of participant in the Vanta Network distinct from regular miners. Rather than operating a single trading account, an entity creates and manages multiple **subaccounts** — each acting as an independent trader competing in the network.

The **entity hotkey** identifies the operator on the validator. Under it, the entity creates **subaccounts** that submit orders and earn incentives. Each subaccount is identified by a **synthetic hotkey** in the format `{entity_hotkey}_{subaccount_id}` (e.g., `5GhDr..._0`, `5GhDr..._1`). These synthetic hotkeys participate in trading exactly like regular miners.

**Key rule: Entity hotkeys cannot submit orders directly. Only subaccounts can place trades.**

## Basic Rules

1. Entity hotkeys must be registered on the Bittensor network and have sufficient Theta collateral.
2. An entity pays a one-time registration fee of **1,000 Theta**, which is permanently slashed on registration.
3. Each subaccount requires collateral proportional to its account size (see [Collateral Requirements](#collateral-requirements)).
4. Each subaccount selects an asset class (`crypto`, `forex`, or `equities`) at creation. This **cannot be changed**.
5. New subaccounts enter a **challenge period** with stricter thresholds and reduced leverage (see [Challenge Period](#challenge-period--subaccount-lifecycle)).
6. Entity hotkeys **cannot place orders**. Orders must be submitted using the subaccount's synthetic hotkey.
7. Subaccounts follow the same trading rules as regular miners: uni-directional positions, leverage limits, market hours, rate limits, etc.
8. A maximum of **5 entities** can be registered on the network at any time.
9. Each entity supports multiple subaccounts.
10. **CRITICAL**: Never reuse synthetic hotkeys from eliminated subaccounts. Eliminated synthetic hotkeys are permanently blacklisted.

## Collateral Requirements

Collateral is denominated in **Theta**, deposited via the Vanta CLI. Entity miners must hold theta for two distinct purposes: a one-time registration fee per subaccount, and an ongoing cross-margin requirement based on open positions.

### Registration Fee

When a subaccount is created, the required theta is burned from the entity's collateral balance. The amount depends on account size:

| Action | Theta Required |
|---|----------------|
| Entity registration (one-time) | 1,000 Theta    |
| Subaccount with $5,000 account size | 2 Theta        |
| Subaccount with $10,000 account size | 4 Theta        |
| Subaccount with $25,000 account size | 5 Theta        |
| Subaccount with $50,000 account size | 10 Theta       |
| Subaccount with $100,000 account size (max) | 20 Theta       |

### Registration Fee Formula

```
required_theta = account_size / CPT

CPT = 2,500  (if account_size ≤ $10,000)
CPT = 5,000  (if account_size > $10,000)
```
If the entity's balance is below the required theta, subaccount creation is rejected immediately. Otherwise the subaccount is created with `status = "pending"` and collateral is burned asynchronously — transitioning to `active` on success or `failed` if the balance is insufficient.

### Cross-Margin Requirement

After subaccounts are funded, the entity must maintain enough theta on-chain to cover the combined open-position exposure of all funded subaccounts. Each funded subaccount's margin is capped at 8% of its account size (the max drawdown threshold), and theta is consumed at a rate of **35 USD per theta**. Challenge period subaccounts are fully exempt, and do not require or consume any margin collateral.

Incoming orders from funded subaccounts are blocked if the projected required collateral would exceed the entity's deposited balance. The validator reads deposited balances from an on-chain cache refreshed every ~60 seconds — if the cache has no entry for the entity, orders are rejected.

Collateral is slashed proportionally to realized losses each time a position closes at a loss, up to a maximum of 8% of the subaccount's account size. If a subaccount is eliminated, all remaining collateral headroom is slashed in a single call. Withdrawals are rejected if they would leave the entity below its current cross-margin requirement.

### Cross-Margin Formulas

Per-subaccount margin (USD):

```
max_slash_usd      = account_size × 8%
remaining_headroom = max_slash_usd - cumulative_slashed_usd
margin_usd         = min(total_open_position_value, remaining_headroom)
```

Entity-level required collateral (theta):

```
required_theta = sum(margin_usd across all funded subaccounts) / 35
```

<details>
<summary><strong>Collateral System Details</strong></summary>

### Cross-Margin Example

Entity has three subaccounts. `CPT_RISK = 35`, `MDD = 8%`.

| Subaccount | Account Size | Max Slash (8%) | Cum. Slashed | Remaining Headroom | Open Position Value | Margin (USD) | Margin (theta) |
|-----------|-------------|---------------|-------------|-------------------|---------------------|-------------|---------------|
| A (funded)     | $100,000    | $8,000        | $2,000      | $6,000             | $50,000             | $6,000       | 171.4 theta   |
| B (funded)     | $25,000     | $2,000        | $0          | $2,000             | $800                | $800         | 22.9 theta    |
| C (challenge)  | $50,000     | $4,000        | $0          | $4,000             | $30,000             | **$0** (exempt) | 0 theta   |
| **Total**  |             |               |             |                    |                     |             | **194.3 theta** |

### Order Blocking Formula

```
margin_delta_usd   = min(current_position + new_position, headroom) - min(current_position, headroom)
margin_delta_theta = margin_delta_usd / 35
projected_required = current_required_theta + margin_delta_theta

if projected_required > deposited_theta → ORDER REJECTED
```

### Slashing on Position Close

```
cumulative_realized_loss += abs(realized_pnl)
target_slash  = min(cumulative_realized_loss, max_slash_usd)
slash_delta   = target_slash - cumulative_slashed
slash_theta   = slash_delta / 35

if slash_delta > 0:
    slash_miner_collateral(entity_hotkey, slash_theta)
    cumulative_slashed += slash_delta
```

**Example** (\$25,000 account, 8% MDD → \$2,000 max slash, CPT_RISK = 35):

| Trade | Loss   | Cum. Loss | Target Slash | Cum. Slashed | Slash Delta | Theta Slashed |
|-------|--------|-----------|-------------|-------------|-------------|---------------|
| 1     | $800   | $800      | $800        | $0          | $800        | 22.9 theta |
| 2     | $900   | $1,700    | $1,700      | $800        | $900        | 25.7 theta |
| 3     | $1,200 | $2,900    | $2,000 (cap)| $1,700      | $300        | 8.6 theta (eliminated) |

### Slashing on Elimination

```
remaining = max_slash_usd - cumulative_slashed
slash_on_realized_loss(entity_hotkey, hotkey, remaining)
```

### Withdrawal Check

```
balance_after = current_balance - withdrawal_amount

if balance_after < required_theta → WITHDRAWAL REJECTED
```

If the entity has no open positions, `required_theta = 0` and the full balance is withdrawable.

### Configuration Reference

| Config Key | Value | Description |
|-----------|-------|-------------|
| `ENTITY_COST_PER_THETA` | 5,000 | USD per theta for subaccount registration (accounts > $10k) |
| `ENTITY_COST_PER_THETA_LOW` | 2,500 | USD per theta for subaccount registration (accounts ≤ $10k) |
| `ENTITY_COST_PER_THETA_LOW_THRESHOLD` | $10,000 | Account size threshold for two-tier CPT |
| `MAX_SUBACCOUNT_ACCOUNT_SIZE` | $100,000 | Maximum USD account size per subaccount |
| `ENTITY_MAX_SUBACCOUNTS` | 10,000 | Maximum subaccounts per entity |
| `SUBACCOUNT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD` | 8% | MDD cap applied to funded subaccounts |
| `ENTITY_COLLATERAL_CPT_RISK` | 35 | USD of loss capacity per theta (used for margin and slash-to-theta conversion) |
| `ENTITY_COLLATERAL_CACHE_REFRESH_S` | 60 | Seconds between on-chain collateral cache refreshes |

</details>

## Challenge Period & Subaccount Lifecycle

Every new subaccount enters a challenge period. The lifecycle is:

```
pending → active → [SUBACCOUNT_CHALLENGE] → [SUBACCOUNT_FUNDED]
                                ↓
                           eliminated
```

| Stage | Bucket | Description |
|---|---|---|
| SUBACCOUNT_CHALLENGE | 1× dust | Challenge phase — reduced leverage, no payout |
| SUBACCOUNT_FUNDED | earning | Passed challenge — full leverage, earns payouts |
| eliminated | — | Permanently removed from competition |

### Challenge Period Requirements

**To pass the challenge period**, a subaccount must achieve:

| Asset Class     | Minimum Return Required |
|-----------------|-------------------------|
| Forex, Equities | ≥ 8%                    |
| Crypto          | ≥ 10%                   |

Passing is evaluated continuously — a subaccount is promoted immediately once `min(account balance, account equity)` meets the threshold. Assessment runs automatically via the validator's EntityServer daemon every 5 minutes.

**Elimination during challenge:** A subaccount is eliminated if its intraday drawdown or drawdown from the end-of-day high-water mark reaches **5%**.

**Leverage reduction:** During the challenge period, a subaccount's maximum portfolio leverage is as follows. These are 4x lower than the funded status limits to limit risk exposure.

| Asset Class | Leverage Limit |
|-------------|----------------|
| Crypto      | 1.25x          |
| Forex       | 5x             |
| Equities    | 1x             |

### After the Challenge Period

Once in SUBACCOUNT_FUNDED:
- Standard **8% max drawdown** elimination applies (same as regular miners).
- After **90 days** in SUBACCOUNT_FUNDED meeting the thresholds, the subaccount is eligible for additional funding.

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

Deposit Theta collateral via the Vanta CLI. You need at least **1,000 Theta** to register an entity, plus additional Theta for each subaccount you plan to create.

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

Register your entity hotkey on the validator. This costs **1,000 Theta** (permanently slashed):

```bash
# Mainnet
vanta entity register --wallet-name <wallet> --wallet-hotkey <entity>

# Testnet
vanta entity register --wallet-name <wallet> --wallet-hotkey <entity> --network test
```

On success, the entity hotkey is assigned to the `ENTITY` bucket and receives a baseline 4× dust weight in the incentive system.

### 7. Obtain Validator API Key

After registering your entity, request a validator API key. This key authenticates the entity miner gateway's WebSocket connection to the validator, which is required for real-time subaccount dashboard streaming.

```bash
# Mainnet
vanta entity apikey --wallet-name <wallet> --wallet-hotkey <entity>

# Testnet
vanta entity apikey --wallet-name <wallet> --wallet-hotkey <entity> --network test
```

The command prints your API key. Store it — you will need it in `miner_secrets.json` as `validator_api_key`. Running the command again returns the same key (idempotent).

### 8. Configure Miner Secrets

Create `mining/miner_secrets.json` with your wallet credentials:

```json
{
  "api_key": "your_api_key",
  "wallet_name": "your_wallet_name",
  "wallet_hotkey": "your_hotkey_name",
  "wallet_password": "your_wallet_password",
  "validator_api_key": "your_validator_api_key"
}
```

| Field | Description |
|---|---|
| `api_key` | API key for authenticating requests to your miner's REST server |
| `wallet_name` | Bittensor wallet name |
| `wallet_hotkey` | Bittensor hotkey name |
| `wallet_password` | Wallet coldkey password (used for signing subaccount creation requests) |
| `validator_api_key` | API key for the validator WebSocket connection (obtained via `vanta entity apikey`) |

To register your entity miner's public endpoint URL with the validator, add:

```json
{
  "entity_endpoint_url": "https://your-domain.com:8088"
}
```

Or set the `ENTITY_MINER_ENDPOINT_URL` environment variable instead.

### 9. Run the Miner

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

| Flag | Default  | Description |
|---|----------|---|
| `--netuid` | 8        | Subnet UID (8 for mainnet, 116 for testnet) |
| `--entity-miner` | disabled | Enable the Entity Miner Gateway |
| `--api-host` | 0.0.0.0  | Host address for the API server |
| `--api-rest-port` | 8088     | Port for the standard Miner REST API |
| `--run-position-inspector` | disabled | Enable the position inspector thread |

This starts the miner REST API on port 8088, which handles both order submission and subaccount management.

### 10. Create Subaccounts

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

| Field | Type | Required | Description                            |
|---|---|---|----------------------------------------|
| `asset_class` | string | Yes | `"crypto"`, `"forex"`, or `"equities"`   |
| `account_size` | float | Yes | Account size in USD                    |

### 11. Submit Orders

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
  "timestamp": 1700000000.0
}
```

## Payout Computation

Entity miner payouts use the same **debt-based scoring system** as regular miners, with a few differences:

- **SUBACCOUNT_CHALLENGE**: No payout — subaccounts in the challenge period do not earn incentives.
- **SUBACCOUNT_FUNDED**: Subaccounts earn payouts based on their PnL performance checkpoints, exactly like MAINCOMP miners.
- **Entity hotkey**: Receives a baseline **4× dust weight** (the minimum floor weight) regardless of subaccount performance.

The payout for a subaccount is calculated from its debt ledger checkpoints that fall within the SUBACCOUNT_FUNDED status window. Performance is weighted 100% on average daily PnL (same as regular miners). All subaccount debt ledgers are aggregated into a single entity-level ledger for weight calculation. Eliminated subaccounts are excluded from aggregation.

**Dust weight multipliers:**

| Bucket | Dust Multiplier |
|---|---|
| ENTITY (entity hotkey) | 4× dust |
| SUBACCOUNT_FUNDED | earning (proportional to debt) |
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

| Method | Endpoint | Description                                                       |
|---|---|-------------------------------------------------------------------|
| POST | `/entity/register` | Register a new entity (requires coldkey signature + 1,000 Theta)  |
| POST | `/entity/create-subaccount` | Create a subaccount (requires coldkey signature + collateral)     |
| GET | `/entity/<entity_hotkey>` | Get entity data and subaccount list                               |
| GET | `/entities` | List all registered entities                                      |
| GET | `/entity/subaccount/<synthetic_hotkey>` | Get subaccount dashboard data (deprecated, use v2 endpoint below) |
| GET | `/v2/entity/subaccount/<synthetic_hotkey>` | Get v2 subaccount dashboard data                                  |
| POST | `/entity/subaccount/payout` | Calculate payout for a subaccount by UUID and time range          |
| POST | `/entity/subaccount/eliminate` | Manually eliminate a subaccount                                   |

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

```json
{
  "entity_coldkey": "<coldkey_ss58>",
  "entity_hotkey": "<hotkey_ss58>",
  "account_size": 10000.0,
  "asset_class": "crypto",
  "signature": "<coldkey_signature>"
}
```

The signature covers `{account_size, admin, asset_class, entity_coldkey, entity_hotkey}` (JSON, sorted keys).

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
| GET | `/api/health` | Health check |

### Miner REST Server (port 8088)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/submit-order` | Submit a trading order for a subaccount (include `subaccount_id`) |
| GET | `/api/order-status/<order_uuid>` | Query order processing status |
| GET | `/api/health` | Health check |

## Notes

### Synthetic Hotkeys

Each subaccount is identified by a synthetic hotkey with the format `{entity_hotkey}_{subaccount_id}`. For example, if your entity hotkey is `5abc...xyz` and you create subaccount 0, the synthetic hotkey is `5abc...xyz_0`.

- Entity hotkeys **cannot** place orders directly — only subaccounts can
- Eliminated subaccount IDs are never reused; new subaccounts always get the next incremental ID

### Limits

| Constraint                          | Value                                  |
|-------------------------------------|----------------------------------------|
| Max entities on the network         | 10                                     |
| Max account size per subaccount     | $100,000 USD                           |
| Challenge period return threshold   | ≥ 8% for fx + equities, 10% for crypto |
| Challenge period drawdown threshold | 5%                                     |
| Funded period drawdown threshold    | 8%                                     |

### Elimination

Subaccounts can be eliminated for:
- **Challenge period failure** — drawdown exceeds 5% before achieving the 8% return threshold
- **Funded period failure** — drawdown exceeds 8%
- **Plagiarism** — detected order similarity with other miners

Eliminated subaccount ids are permanently retired. Create a new subaccount to replace an eliminated one.

## Dashboard

Monitor subaccount performance at:

- Mainnet: https://dashboard.taoshi.io
- Testnet: https://testnet.dashboard.taoshi.io

Log in using a [polkadot.js](https://polkadot.js.org/extension/) browser wallet. API key tier 200 is required to access subaccount dashboard data. Contact a team member for an API key if you are interested in registering an entity miner.

## Troubleshooting

### Entity Miner Gateway fails to start
- Verify `mining/miner_secrets.json` exists and contains valid wallet credentials
- Check that `wallet_password` decrypts the coldkey successfully
- Ensure port 8088 is not already in use

### Subaccount creation fails
- Ensure your entity is registered with the validator
- Verify you have sufficient Theta collateral for the requested account size
- Check that you haven't exceeded the maximum number of active subaccounts

### Orders rejected for subaccount
- Confirm the subaccount is active (not eliminated or still in `pending` status)
- Verify you're using `subaccount_id` (not the full synthetic hotkey) in the order request
- Check that the subaccount's asset class matches the trade pair

## Hyperliquid Subaccounts

Hyperliquid-linked subaccounts automatically forward trades from a Hyperliquid address as Vanta signals. They always use `crypto` as their asset class. This section applies only if you intend to link Hyperliquid traders — it is not required for standard entity miners.

The information below is optional for entity miners that do not choose to host Hyperliquid subaccount miners.

### Miner Secrets

Add the following fields to `mining/miner_secrets.json`:

```json
{
  "validator_url": "https://validator.mainnet.vantatrading.io",
  "validator_ws_url": "ws://34.65.245.134:8765",
  "validator_api_key": "your_validator_api_key"
}
```

| Field | Description |
|---|---|
| `validator_url` | Validator REST API URL  |
| `validator_ws_url` | Validator WebSocket URL |
| `validator_api_key` | API key for the validator WebSocket connection (obtained via `vanta entity apikey`) |
| `max_hl_traders` | Maximum number of Hyperliquid traders that can be registered (optional, no limit if unset). Can also be set via `MAX_HL_TRADERS` env var (env var takes precedence). |

### Creating HL-Linked Subaccounts

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

#### Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `hl_address` | string | Yes | Hyperliquid wallet address (0x + 40 hex chars) |
| `account_size` | float | Yes | Account size in USD |
| `payout_address` | string | No | Optional EVM payout address |

### Monitoring

#### Health Check (HL fields)

When HL is configured, the health check response includes additional fields:

| Field | Description |
|---|---|
| `ws_connected` | Whether the WebSocket connection to the validator is active |
| `hl_addresses_tracked` | Number of Hyperliquid addresses currently being tracked |
| `max_hl_traders` | Configured limit on HL traders |
| `dashboard_cache_size` | Number of HL dashboards cached |
| `sse_subscribers` | Number of active SSE stream subscribers |

#### Hyperliquid Dashboard

Get cached dashboard data for a Hyperliquid address:

```bash
curl http://localhost:8088/api/hl/<hl_address>/dashboard \
  -H "Authorization: your_api_key"
```

#### Order Events

Get the ring buffer of recent order events (accepted/rejected):

```bash
curl "http://localhost:8088/api/hl/<hl_address>/events?since=1700000000000" \
  -H "Authorization: your_api_key"
```

#### Real-Time SSE Stream

Subscribe to a server-sent events stream for real-time dashboard updates and rejection notifications:

```bash
curl -N http://localhost:8088/api/hl/<hl_address>/stream \
  -H "Authorization: your_api_key"
```

### REST API

#### Validator REST Server

| Method | Endpoint | Description |
|---|---|---|
| POST | `/entity/create-hl-subaccount` | Alias for `/entity/create-subaccount` with `hl_address` |

HL-linked subaccount creation (include `hl_address`; `asset_class` is always `"crypto"`):

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

The signature covers `{account_size, admin, asset_class, entity_coldkey, entity_hotkey, hl_address}` (JSON, sorted keys), plus `payout_address` if provided.

#### Entity Miner Gateway (port 8088)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/create-hl-subaccount` | Create an HL-linked subaccount (proxies to validator) |
| GET | `/api/hl/<hl_address>/dashboard` | Cached HL dashboard data |
| GET | `/api/hl/<hl_address>/events` | Order event ring buffer |
| GET | `/api/hl/<hl_address>/stream` | SSE real-time stream |

### Limits

| Constraint | Value |
|---|---|
| Max HL traders per entity miner | Configurable via `max_hl_traders` / `MAX_HL_TRADERS` (no limit if unset) |

### Troubleshooting

#### WebSocket not connecting
- Confirm `validator_ws_url` in secrets matches the validator's WebSocket server (`ws://34.65.245.134:8765`)
- Confirm `validator_api_key` is set in `miner_secrets.json` and was obtained via `vanta entity apikey`. An invalid or missing key causes an immediate disconnect after the WebSocket upgrade.
- Check network connectivity to the validator
- The gateway retries with exponential backoff (1s to 60s) on connection failures

#### Subaccount creation fails
- Verify the validator REST API is reachable at the configured `validator_url` (`https://validator.mainnet.vantatrading.io`)

## Security Notes

- Do not expose your coldkey or private keys.
- Always test on testnet (netuid 116) before mainnet.
- Do not reuse the password of your mainnet wallet on testnet.
- The entity coldkey is used to sign subaccount creation requests — keep it secure.
- Do not commit `mining/miner_secrets.json` to version control.
