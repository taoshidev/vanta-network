# Signal Reference

Complete reference for all order signal types supported by Vanta Network.

---

## Common Fields

| Field | Type | Description |
|-------|------|-------------|
| `execution_type` | string | One of: `MARKET`, `LIMIT`, `STOP_LIMIT`, `BRACKET`, `LIMIT_CANCEL`, `LIMIT_EDIT`, `FLAT_ALL` |
| `trade_pair_id` | string | Asset identifier (e.g. `BTCUSD`, `EURUSD`) |
| `order_type` | string | `LONG`, `SHORT`, or `FLAT` |
| `leverage` | float | Position size as a multiplier of account size |
| `value` | float | Position size in USD notional |
| `quantity` | float | Position size in base currency units |
| `limit_price` | float | Execution price for LIMIT and STOP_LIMIT orders |
| `stop_price` | float | Trigger price for STOP_LIMIT orders |
| `stop_condition` | string | Trigger direction for STOP_LIMIT: `GTE` or `LTE` |
| `stop_loss` | float | Static stop loss price |
| `take_profit` | float | Take profit price |
| `trailing_stop` | object | Dynamic trailing stop — `{"trailing_percent": 0.02}` or `{"trailing_value": 500}` |
| `bracket_orders` | array | List of bracket order entries (see below) |
| `order_uuid` | string | Client-provided UUID (auto-generated if omitted) |

**Size fields:** Exactly one of `leverage`, `value`, or `quantity` must be specified (except BRACKET and FLAT orders).

---

## Execution Types

### MARKET

Executes immediately at the current market price.

**Required:** `trade_pair_id`, `order_type`, one size field

```json
{
  "execution_type": "MARKET",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0
}
```

#### FLAT (close position)

```json
{
  "execution_type": "MARKET",
  "trade_pair_id": "BTCUSD",
  "order_type": "FLAT"
}
```

#### With static stop loss and take profit

Stop loss and take profit create a bracket order that fires when price crosses the threshold.

```json
{
  "execution_type": "MARKET",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0,
  "stop_loss": 90000.0,
  "take_profit": 110000.0
}
```

Rules:
- LONG: `stop_loss` < fill price < `take_profit`
- SHORT: `take_profit` < fill price < `stop_loss`

#### With trailing stop

A trailing stop dynamically adjusts the stop loss as price moves favorably. `stop_loss` and `trailing_stop` can coexist — the more protective value is used.

```json
{
  "execution_type": "MARKET",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0,
  "trailing_stop": {"trailing_percent": 0.02}
}
```

```json
{
  "execution_type": "MARKET",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "quantity": 0.01,
  "trailing_stop": {"trailing_value": 500},
  "take_profit": 110000.0
}
```

`trailing_stop` fields:
- `trailing_percent` — distance as a fraction of best price (e.g. `0.02` = 2%). Must be `0 < x < 1`.
- `trailing_value` — distance in quote currency units (e.g. `500` = $500). Must be `> 0`.

Trailing SL computation:
| Direction | trailing_percent | trailing_value |
|-----------|-----------------|----------------|
| LONG | `best_price × (1 - pct)` | `best_price - value` |
| SHORT | `best_price × (1 + pct)` | `best_price + value` |

When both `stop_loss` and `trailing_stop` are set:
- LONG: effective SL = `max(stop_loss, trailing_sl)`
- SHORT: effective SL = `min(stop_loss, trailing_sl)`

#### With explicit bracket orders

Use `bracket_orders` for multiple independent bracket entries or when specifying different sizes per bracket.

```json
{
  "execution_type": "MARKET",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0,
  "bracket_orders": [
    {"stop_loss": 90000.0, "value": 500.0},
    {"take_profit": 110000.0, "value": 500.0}
  ]
}
```

Each bracket entry requires at least one of `stop_loss`, `take_profit`, `trailing_percent`, or `trailing_value`. If no size is specified in a bracket entry, it will inherit the filled quantity of the parent order.

---

### LIMIT

Places a resting order that fills when price reaches `limit_price`.

**Required:** `trade_pair_id`, `order_type`, one size field, `limit_price`

```json
{
  "execution_type": "LIMIT",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0,
  "limit_price": 95000.0
}
```

Price validation:
- LONG: `stop_loss` < `limit_price` < `take_profit`
- SHORT: `take_profit` < `limit_price` < `stop_loss`
- SL validation is skipped when `trailing_stop` is set (SL computed at fill time)

#### With SL/TP (creates bracket on fill)

```json
{
  "execution_type": "LIMIT",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0,
  "limit_price": 95000.0,
  "stop_loss": 90000.0,
  "take_profit": 105000.0
}
```

#### With trailing stop

```json
{
  "execution_type": "LIMIT",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0,
  "limit_price": 95000.0,
  "trailing_stop": {"trailing_percent": 0.02}
}
```

#### With multiple bracket entries (creates brackets on fill)

```json
{
  "execution_type": "LIMIT",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0,
  "limit_price": 95000.0,
  "bracket_orders": [
    {"stop_loss": 90000.0, "value": 500.0},
    {"take_profit": 105000.0, "value": 500.0}
  ]
}
```

---

### STOP_LIMIT

A stop-limit order combines a stop trigger with a limit execution. When the market hits `stop_price` (per `stop_condition`), a LIMIT order at `limit_price` is created. This gives control over worst-case execution price after a breakout/breakdown trigger.

**Required:** `trade_pair_id`, `order_type`, one size field, `stop_price`, `stop_condition`, `limit_price`

```json
{
  "execution_type": "STOP_LIMIT",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0,
  "stop_price": 100000.0,
  "stop_condition": "GTE",
  "limit_price": 98000.0
}
```

`stop_condition` values:
- `GTE` — trigger when market price >= `stop_price`
- `LTE` — trigger when market price <= `stop_price`

Price validation:
- `stop_price` > 0, `limit_price` > 0
- `limit_price` can be above or below `stop_price` — it controls the worst-case fill price after the stop triggers
- `stop_loss` < `limit_price` < `take_profit` (LONG) or `take_profit` < `limit_price` < `stop_loss` (SHORT)

#### With SL/TP (creates bracket when the spawned limit order fills)

```json
{
  "execution_type": "STOP_LIMIT",
  "trade_pair_id": "BTCUSD",
  "order_type": "LONG",
  "value": 1000.0,
  "stop_price": 100000.0,
  "stop_condition": "GTE",
  "limit_price": 98000.0,
  "bracket_orders": [
    {"stop_loss": 93000.0, "take_profit": 115000.0}
  ]
}
```

#### Lifecycle

1. Order is stored as `STOP_LIMIT_UNFILLED`
2. Daemon checks if market price satisfies `stop_condition` against `stop_price`
3. When triggered, stop-limit is marked `STOP_LIMIT_FILLED` and a child LIMIT order is created with UUID `{parent_uuid}-limit`
4. The child limit order follows normal LIMIT lifecycle (may fill immediately if price is at or past `limit_price`)
5. `bracket_orders` are forwarded to the child and created when the child fills

#### Notes

- Stop-limit orders are **never filled immediately** on submission — they always wait for the daemon
- The trigger uses mid price (average of bid and ask)
- `FLAT` order type is not supported
- Cancellable via `LIMIT_CANCEL` while unfilled
- Editable via `LIMIT_EDIT` while unfilled
- Counts toward the max unfilled orders limit

---

### BRACKET

Sets a stop loss / take profit on an **existing open position**. Requires an open position for the trade pair.

**Required:** `trade_pair_id`, at least one of `stop_loss`, `take_profit`, `trailing_stop`

Size is optional — omit to inherit from the position.

```json
{
  "execution_type": "BRACKET",
  "trade_pair_id": "BTCUSD",
  "stop_loss": 90000.0,
  "take_profit": 110000.0
}
```

#### With explicit size

```json
{
  "execution_type": "BRACKET",
  "trade_pair_id": "BTCUSD",
  "value": 500.0,
  "stop_loss": 90000.0
}
```

#### With trailing stop

```json
{
  "execution_type": "BRACKET",
  "trade_pair_id": "BTCUSD",
  "trailing_stop": {"trailing_percent": 0.02},
  "take_profit": 110000.0
}
```

#### With multiple bracket entries

```json
{
  "execution_type": "BRACKET",
  "trade_pair_id": "BTCUSD",
  "bracket_orders": [
    {"stop_loss": 90000.0, "value": 500.0},
    {"take_profit": 110000.0, "value": 500.0}
  ]
}
```

---

### LIMIT_CANCEL

Cancels one or all pending limit/bracket orders.

#### Cancel all orders for a specific pair

```json
{
  "execution_type": "LIMIT_CANCEL",
  "trade_pair_id": "BTCUSD",
  "order_uuid": "ALL"
}
```

#### Cancel all limit orders across all pairs

```json
{
  "execution_type": "LIMIT_CANCEL",
  "order_uuid": "ALL"
}
```

#### Cancel a specific order by UUID

```json
{
  "execution_type": "LIMIT_CANCEL",
  "order_uuid": "your-order-uuid"
}
```

#### Cancel multiple orders by UUID

```json
{
  "execution_type": "LIMIT_CANCEL",
  "order_uuid": "uuid-one,uuid-two,uuid-three"
}
```

---

### LIMIT_EDIT

Modifies an existing pending limit or bracket order. The `order_uuid` must match an existing unfilled order.

**Required:** `trade_pair_id`, `order_uuid`

#### Edit a limit order price

```json
{
  "execution_type": "LIMIT_EDIT",
  "trade_pair_id": "BTCUSD",
  "order_uuid": "your-order-uuid",
  "order_type": "LONG",
  "value": 1000.0,
  "limit_price": 94000.0
}
```

#### Edit a bracket order

```json
{
  "execution_type": "LIMIT_EDIT",
  "trade_pair_id": "BTCUSD",
  "order_uuid": "your-order-uuid",
  "stop_loss": 89000.0,
  "take_profit": 111000.0
}
```

---

### FLAT_ALL

Closes open positions across all trade pairs.

#### Close all open positions

```json
{
  "execution_type": "FLAT_ALL",
  "order_uuid": "ALL"
}
```

#### Close specific positions by position UUID

```json
{
  "execution_type": "FLAT_ALL",
  "order_uuid": "position-uuid-one,position-uuid-two"
}
```

---

## `bracket_orders` Entry Format

Each entry in `bracket_orders` supports:

| Field | Description |
|-------|-------------|
| `stop_loss` | Static stop loss price |
| `take_profit` | Take profit price |
| `trailing_percent` | Trailing distance as fraction of best price |
| `trailing_value` | Trailing distance in quote currency |
| `leverage` | Close size as leverage |
| `value` | Close size in USD |
| `quantity` | Close size in base units |

At least one price field (`stop_loss`, `take_profit`, `trailing_percent`, or `trailing_value`) is required. If no size is specified, it will inherit the filled quantity of the parent order.

---

## Notes

- `order_uuid` is auto-generated if not provided. Providing your own UUID enables later cancellation by UUID.
- `FLAT` order type on MARKET closes the position without specifying size — it closes the full open position.
- Bracket orders created from a market/limit fill use the parent order's UUID as a prefix: `{parent_uuid}-bracket-{i}`.
- Trailing stop best price is tracked per tick and persisted to disk (rate-limited to once per minute).
