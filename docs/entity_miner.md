# Entity Miner

Entity miners are a new type of participant in the Vanta Network distinct from regular miners. Rather than operating a single trading account, an entity creates and manages multiple **subaccounts** — each acting as an independent trader competing in the network.

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
9. Each entity supports up to **10,000 subaccounts**.
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
| Forex | ≥8%                     |
| Crypto | ≥8%                     |

Passing is evaluated continuously — a subaccount is promoted immediately upon hitting the threshold.

**Elimination during challenge:** A subaccount is eliminated if its drawdown from the high-water mark reaches **5%**. This is stricter than the standard 10% threshold that applies after the challenge period.

**Leverage reduction:** During the challenge period, a subaccount's maximum leverage and account multiplier are divided by **4** to limit risk exposure.

### After the Challenge Period

Once in SUBACCOUNT_FUNDED:
- Standard **10% max drawdown** elimination applies (same as regular miners).
- Subaccounts in the bottom 25 of their asset class enter a **60-day probation period**.
- After **90 days** in SUBACCOUNT_FUNDED, the subaccount is promoted to SUBACCOUNT_ALPHA.

## Getting Started

### Prerequisites

- Python 3.10+
- [Bittensor](https://github.com/opentensor/bittensor#install)
- Vanta CLI installed:
  ```bash
  pip install git+https://github.com/taoshidev/vanta-cli.git
  ```

### 1. Install Vanta

Clone repository:

```bash
git clone https://github.com/taoshidev/vanta-network.git
cd vanta-network
```

Create and activate virtual environment:

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

Follow the prompts to pay the registration fee and confirm. Check that registration succeeded:

```bash
btcli wallet overview --wallet.name <wallet>
```

| Environment | Netuid |
|---|---|
| Mainnet | 8 |
| Testnet | 116 |

### 4. Add Stake

Before depositing Theta collateral, add TAO stake for your hotkey on the correct netuid. This is required by the collateral deposit mechanism:

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

### 7. Run the Miner

Entity miners run `neurons/miner.py` just like regular miners. When the miner process starts, a REST server starts automatically on port 8088 that handles order submission and subaccount creation.

```bash
# Mainnet
python neurons/miner.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <entity>

# Testnet
python neurons/miner.py --netuid 116 --subtensor.network test --wallet.name <wallet> --wallet.hotkey <entity>
```

Create `vanta_api/api_keys.json` with your API key before running:

```json
{
  "my_api_key": {
    "key": "xxxx",
    "tier": 200
  }
}
```

### 8. Create Subaccounts

Create subaccounts under your entity. Each subaccount is an independent trading account with its own asset class and account size.

```bash
# Mainnet
vanta entity create-subaccount \
  --wallet-name <wallet> \
  --wallet-hotkey <entity> \
  --account-size <usd_amount> \
  --asset-class <crypto|forex>

# Testnet
vanta entity create-subaccount \
  --wallet-name <wallet> \
  --wallet-hotkey <entity> \
  --account-size <usd_amount> \
  --asset-class <crypto|forex> \
  --network test
```

On success, the command returns the **synthetic hotkey** (e.g., `5GhDr..._0`) along with the `subaccount_id`, `subaccount_uuid`, and collateral charged. Use the synthetic hotkey to submit orders for that subaccount.

Subaccounts can also be created via the miner REST server (if running `neurons/miner.py`):

```bash
curl -X POST http://127.0.0.1:8088/api/create-subaccount \
  -H "Content-Type: application/json" \
  -d '{"asset_class": "crypto", "account_size": 10000}'
```

### 9. Submit Orders

Orders are submitted to the entity miner's REST server at `POST http://127.0.0.1:8088/api/submit-order`. Include the `subaccount_id` field to route the order to the correct subaccount. The miner constructs the synthetic hotkey internally and forwards the order to validators.

**Do not use the entity hotkey directly** — it will be rejected. Only subaccount orders (identified by `subaccount_id`) are accepted.

```bash
curl -X POST http://127.0.0.1:8088/api/submit-order \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer xxxx" \
  -d '{
    "trade_pair": "BTCUSD",
    "order_type": "LONG",
    "leverage": 0.1,
    "execution_type": "MARKET",
    "subaccount_id": 0
  }'
```

The `subaccount_id` is the integer returned when the subaccount was created (e.g., `0`, `1`, `2`). Each subaccount is routed independently — orders for different subaccounts can be submitted concurrently.

Full API documentation for order fields, limit orders, and response formats is available in [miner_rest_server.md](miner_rest_server.md).

## Payout Computation

Entity miner payouts use the same **debt-based scoring system** as regular miners, with a few differences:

- **SUBACCOUNT_CHALLENGE**: No payout — subaccounts in the challenge period do not earn incentives.
- **SUBACCOUNT_FUNDED** and **SUBACCOUNT_ALPHA**: Subaccounts earn payouts based on their PnL performance checkpoints, exactly like MAINCOMP miners.
- **Entity hotkey**: Receives a baseline **4× dust weight** (the minimum floor weight) regardless of subaccount performance.

The payout for a subaccount is calculated from its debt ledger checkpoints that fall within the SUBACCOUNT_FUNDED or SUBACCOUNT_ALPHA status window. Performance is weighted 100% on average daily PnL (same as regular miners).

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

## REST API Endpoints

All validator endpoints require a valid API key (tier 200) in the `Authorization` header.

**Base URL:**
- Mainnet: `https://validator.mainnet.vantatrading.io`
- Testnet: `https://validator.testnet.vantatrading.io`
- Local validator: `http://<validator-ip>:48888`

### Validator REST Server

| Method | Endpoint | Description |
|---|---|---|
| POST | `/entity/register` | Register a new entity (requires coldkey signature + 5,000 Theta) |
| POST | `/entity/create-subaccount` | Create a subaccount (requires coldkey signature + collateral) |
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

```json
{
  "entity_coldkey": "<coldkey_ss58>",
  "entity_hotkey": "<hotkey_ss58>",
  "account_size": 10000.0,
  "asset_class": "crypto",
  "signature": "<coldkey_signature>"
}
```

The signature is produced by signing `{"account_size": ..., "admin": false, "asset_class": "...", "entity_coldkey": "...", "entity_hotkey": "..."}` (JSON, sorted keys) with the coldkey.

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

### Miner REST Server (port 8088)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/submit-order` | Submit a trading order for a subaccount |
| POST | `/api/create-subaccount` | Create a subaccount (wallet loaded from miner config, proxies to validator) |
| GET | `/api/order-status/<order_uuid>` | Query order processing status |
| GET | `/api/health` | Health check |

#### POST /api/submit-order (entity miner)

Include `subaccount_id` to route the order to the correct subaccount:

```json
{
  "trade_pair": "BTCUSD",
  "order_type": "LONG",
  "leverage": 0.1,
  "execution_type": "MARKET",
  "subaccount_id": 0
}
```

The `subaccount_id` is used to construct the synthetic hotkey (`{entity_hotkey}_{subaccount_id}`) for position tracking. Omitting it will route the order to the entity hotkey, which will be rejected.

#### POST /api/create-subaccount

```json
{
  "asset_class": "crypto",
  "account_size": 10000.0
}
```

## Dashboard

Monitor subaccount performance at:

- Mainnet: https://dashboard.taoshi.io
- Testnet: https://testnet.dashboard.taoshi.io

Log in using a [polkadot.js](https://polkadot.js.org/extension/) browser wallet. API key tier 200 is required to access subaccount dashboard data.

## Security Notes

- Do not expose your coldkey or private keys.
- Always test on testnet (netuid 116) before mainnet.
- Do not reuse the password of your mainnet wallet on testnet.
- The entity coldkey is used to sign subaccount creation requests — keep it secure.
