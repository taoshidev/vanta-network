# Entity Miner Collateral System

## Problem Statement

Entity miners currently burn theta to register subaccounts, where each theta unit maps to controlled capital via the CPT (capacity per theta) parameter — currently 1 theta = $5,000 USD. As entities onboard anonymous accounts, adversarial strategies (e.g., martingale distributed across many subaccounts) become difficult to detect or police. A collateral system is needed to ensure entities share downside risk and prevent abuse.

## Scope

This system applies **only to entity miners and their subaccounts** — individual miners are unaffected. The two core validator changes are:

1. **Order blocking** — Reject new positions from subaccounts when the entity's cross-margin is fully utilized.
2. **Collateral slashing** — Slash entity collateral on realized losses from subaccount position closes.

Challenge period subaccounts are **exempt** from margin requirements since their payout is zero.

## Design

### Collateral Model: Aggregate Cross-Margin

A single collateral pool backs all of an entity's subaccounts. Collateral requirements are calculated dynamically based on actual position exposure.

#### Required Collateral Calculation

Per subaccount (non-challenge-period only):

```
Risk Exposure = min(sum(abs(Position Value)), Account Balance × MDD%)
```

Entity-level:

```
Required Collateral = sum(Risk Exposure) across all non-challenge-period subaccounts
```

**Example:**

| Subaccount | Balance | Position Value | MDD Cap (10%) | Risk Exposure |
|------------|---------|---------------|---------------|---------------|
| A          | $100K   | $700K         | $10K          | $10K          |
| B          | $50K    | $3K           | $5K           | $3K           |
| C (challenge) | $100K | $20K        | $10K          | $0 (exempt)   |
| **Total**  |         |               |               | **$13K**      |

### Collateral Cache

Entity collateral balances are read from on-chain contracts via a **background task** and cached to disk. This keeps order-path latency low — the validator references the local cache rather than querying the contract on every order.

- A periodic background task reads each entity's collateral contract and writes the result to an on-disk cache (e.g., `validation/entity_collateral_cache.json`).
- The cache maps `entity_hotkey → deposited_collateral_usd`.
- The order-blocking check reads from this cache.
- Refresh interval should be frequent enough that deposits/withdrawals are reflected promptly (e.g., every 60s).

### Order Blocking

When a subaccount submits a new order, the validator:

1. Skips margin check if the subaccount is in challenge period.
2. Computes the entity's current `Required Collateral` across all non-challenge-period subaccounts (including the impact of the new order).
3. Looks up the entity's `Deposited Collateral` from the local cache.
4. If `Required Collateral > Deposited Collateral`, the order is rejected.

### Slashing Mechanics

Triggered on subaccount position close with realized loss:

```
Max Slash         = Account Balance × MDD%
Remaining Limit   = Max Slash - Cumulative Slashed
Actual Slash      = min(Realized Loss, Remaining Limit)
Cumulative Slashed += Actual Slash
```

When `Cumulative Slashed >= Max Slash`, the subaccount is eliminated.

**Example** ($100K account, 10% MDD → $10K max slash):

| Trade | Loss  | Remaining Limit | Slash | Cumulative | Outcome     |
|-------|-------|-----------------|-------|------------|-------------|
| 1     | -$3K  | $10K            | $3K   | $3K        |             |
| 2     | -$5K  | $7K             | $5K   | $8K        |             |
| 3     | -$4K  | $2K             | $2K   | $10K       | Eliminated  |

## Engineering Components

### 1. Collateral Contracts

Reuse existing miner collateral deposit contracts. The validator reads contract contents to determine available collateral per entity.

### 2. Collateral Cache Service

- Background task that periodically reads entity collateral contracts and writes to `validation/entity_collateral_cache.json`.
- Provides a fast lookup interface: `get_entity_collateral(entity_hotkey) → float`.
- Refresh interval: ~60s.

### 3. Validator Changes

- **Cross-margin evaluation** — On each incoming order from a subaccount, aggregate all non-challenge-period subaccount positions to compute current cross-margin usage. Reject if entity collateral (from cache) is insufficient.
- **Slashing** — On subaccount position close with realized loss, slash collateral from the entity pool up to the per-subaccount MDD cap.
- **Withdrawal blocking** — Reject entity collateral withdrawal requests while subaccounts have open positions.
