# Entity Miner Collateral System

## Overview

Entity miners must hold theta collateral for two distinct purposes:

1. **Registration fee** — A one-time theta burn when creating a subaccount, proportional to the subaccount's account size.
2. **Cross-margin requirement** — Ongoing theta held on-chain to cover the open-position exposure of all funded subaccounts.

Individual miners (non-entity) are unaffected by this system.

---

## 1. Subaccount Registration Fee

When `create_subaccount` is called, the validator checks the entity's live collateral balance before allowing creation. The required theta is:

```
required_theta = account_size / CPT

CPT = 2,500  (if account_size ≤ $10,000)
CPT = 5,000  (if account_size > $10,000)
```

If the entity's current balance is below `required_theta`, the request is rejected immediately.

If the balance is sufficient, the subaccount is created with `status = "pending"` and a background thread calls `slash_miner_collateral(entity_hotkey, required_theta)` on-chain. On success the subaccount becomes `status = "active"`; on failure it becomes `status = "failed"`.

Admin subaccounts skip this check and are created immediately as `status = "admin"`.

### Example

| Account Size | CPT   | Required Theta |
|-------------|-------|----------------|
| $5,000      | 2,500 | 2.0 theta      |
| $10,000     | 2,500 | 4.0 theta      |
| $25,000     | 5,000 | 5.0 theta      |
| $100,000    | 5,000 | 20.0 theta     |

---

## 2. Cross-Margin Requirement (Ongoing)

After subaccounts are active, the entity must maintain enough theta to cover the combined open-position exposure of all its funded subaccounts. This is computed dynamically on every incoming order.

### Key parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `SUBACCOUNT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD` | 8% | MDD cap — maximum loss that can ever be slashed per subaccount |
| `ENTITY_COLLATERAL_CPT_RISK` | 35 | USD of remaining loss capacity per theta |

### Per-subaccount margin (USD)

```
max_slash_usd      = account_size × 8%
remaining_headroom = max_slash_usd - cumulative_slashed_usd
margin_usd         = min(total_open_position_value, remaining_headroom)
```

- **Challenge period subaccounts are fully exempt** — their payout is zero, so no margin is required.
- Once `cumulative_slashed >= max_slash`, `remaining_headroom = 0` and the subaccount contributes nothing to the margin requirement (it will also be eliminated).

### Entity-level required collateral (theta)

```
required_theta = sum(margin_usd across all funded subaccounts) / CPT_RISK
               = sum(margin_usd) / 35
```

### Example

Entity has three subaccounts. `CPT_RISK = 35`, `MDD = 8%`.

| Subaccount | Account Size | Max Slash (8%) | Cum. Slashed | Remaining Headroom | Open Position Value | Margin (USD) | Margin (theta) |
|-----------|-------------|---------------|-------------|-------------------|---------------------|-------------|---------------|
| A (funded)     | $100,000    | $8,000        | $2,000      | $6,000             | $50,000             | $6,000       | 171.4 theta   |
| B (funded)     | $25,000     | $2,000        | $0          | $2,000             | $800                | $800         | 22.9 theta    |
| C (challenge)  | $50,000     | $4,000        | $0          | $4,000             | $30,000             | **$0** (exempt) | 0 theta   |
| **Total**  |             |               |             |                    |                     |             | **194.3 theta** |

The entity must hold at least **194.3 theta** on-chain for new orders from subaccounts A or B to be accepted.

---

## 3. Order Blocking

When a subaccount submits a new order (`MarketOrderManager` → `EntityCollateralClient.try_gate_position_open`):

1. Regular miner hotkeys pass through unconditionally.
2. Challenge period subaccounts pass through unconditionally.
3. For funded subaccounts, the validator computes the **projected** required collateral if this order goes through — adding the new position's value to the ordering subaccount's margin — and compares it to the entity's deposited balance.

```
margin_delta_usd   = min(current_position + new_position, headroom) - min(current_position, headroom)
margin_delta_theta = margin_delta_usd / 35
projected_required = current_required_theta + margin_delta_theta

if projected_required > deposited_theta → ORDER REJECTED
```

The deposited balance is read from a local cache refreshed every ~60 seconds from on-chain contracts (`validation/entity_collateral_cache.json`). If the cache has no entry for the entity, the order is rejected.

---

## 4. Collateral Slashing

### On position close with loss

Triggered in `MarketOrderManager` via `EntityCollateralClient.try_slash_on_position_close`. Only fires when `realized_pnl < 0`.

```
cumulative_realized_loss += abs(realized_pnl)
target_slash  = min(cumulative_realized_loss, max_slash_usd)
slash_delta   = target_slash - cumulative_slashed
slash_theta   = slash_delta / 35

if slash_delta > 0:
    slash_miner_collateral(entity_hotkey, slash_theta)
    cumulative_slashed += slash_delta
```

Losses are accumulated across the subaccount's lifetime. Once `cumulative_slashed` reaches `max_slash_usd`, no further slashing occurs for that subaccount.

**Example** (\$25,000 account, 8% MDD → \$2,000 max slash, CPT_RISK = 35):

| Trade | Loss   | Cum. Loss | Target Slash | Cum. Slashed | Slash Delta | Theta Slashed                                          |
|-------|--------|-----------|-------------|-------------|-------------|--------------------------------------------------------|
| 1     | $800   | $800      | $800        | $0          | $800        | 22.9 theta                                             |
| 2     | $900   | $1,700    | $1,700      | $800        | $900        | 25.7 theta                                             |
| 3     | $1,200 | $2,900    | $2,000 (cap)| $1,700      | $300        | 8.6 theta (Miner is eliminated for exceeding drawdown) |

After trade 3, the drawdown is exceeded and max slash is exhausted. The subaccount is eliminated.

### On elimination

Triggered in `EliminationManager` via `EntityCollateralClient.try_slash_on_elimination`. Slashes all remaining headroom in one call:

```
remaining = max_slash_usd - cumulative_slashed
slash_on_realized_loss(entity_hotkey, hotkey, remaining)
```

Challenge period subaccounts and regular miner hotkeys are exempt from slashing.

---

## 5. Withdrawal Check

An entity miner can withdraw theta as long as the post-withdrawal balance covers the current cross-margin requirement:

```
balance_after = current_balance - withdrawal_amount

if balance_after < required_theta → WITHDRAWAL REJECTED
```

`required_theta` is computed live via `compute_entity_required_collateral` at withdrawal time. If the entity has no open positions across any funded subaccounts, `required_theta = 0` and the full balance is withdrawable (subject to standard MDD slashing penalties applied to all miners).

---

## Configuration Reference

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
