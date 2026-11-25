# Order Envelope for Permissionless Entity Miners

A trader who trades through a permissionless entity miner (PEM) is a registered miner with their own hotkey. The trader signs each order on their own device. The PEM forwards it to validators without being able to change it, and the validator records both the trader (whose position it is) and the PEM (which relayed it).

Reference implementation: [`shared_objects/pem_envelope.py`](../shared_objects/pem_envelope.py). Tests, including the vectors below: [`tests/vali_tests/test_pem_envelope.py`](../tests/vali_tests/test_pem_envelope.py).

---

## Layers

| Term | What it is | Made by |
|------|------------|---------|
| **Signal** | The trade: the same object a miner sends in `SendSignal.signal` (see [Signal Reference](signal_reference.md)) | Trader |
| **Payload** | The signal plus who placed it, through which PEM, on which netuid, when, and the order uuid, as a JSON string | Trader's device |
| **Signature** | The trader hotkey's sr25519 signature over the payload's exact UTF-8 bytes | Trader's device |
| **Envelope** | Payload and signature together | Trader's device → PEM |
| **Synapse** | `SendSignal`, carrying the envelope in `origin_payload` and `origin_signature`, signed by the PEM's dendrite | PEM |

The trader's signature covers the order's contents; the PEM cannot change them. The dendrite signature proves which PEM delivered it. The validator requires the two to agree (`relay_hotkey` must be the dendrite hotkey).

---

## Payload

A JSON object with exactly these fields. Unknown fields, missing fields and repeated keys are refused.

| Field | Type | Description |
|-------|------|-------------|
| `v` | integer | Envelope version. Must be `1` |
| `netuid` | integer | Subnet the order is for: `8` mainnet, `116` testnet. Stops an order signed for one network being replayed on the other |
| `origin_hotkey` | string | The trader's hotkey (SS58). The order is processed as this miner, and the signature is checked against it |
| `relay_hotkey` | string | The PEM's hotkey (SS58). Must differ from `origin_hotkey` and equal the hotkey that sends the synapse. Binds the order to one PEM: to switch PEMs, the trader signs again |
| `miner_order_uuid` | string | Order id, 1 to 256 characters. Used once: the validator refuses a uuid it has seen |
| `timestamp_ms` | integer | When the trader signed, in Unix milliseconds. The validator refuses a payload outside its freshness window |
| `signal` | object | The trade, as in the [Signal Reference](signal_reference.md). `trade_pair` takes the trade pair id, e.g. `"BTCUSDC"` |

Example (whitespace added for reading; see Signing for the bytes actually signed):

```json
{
  "v": 1,
  "netuid": 8,
  "origin_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
  "relay_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
  "miner_order_uuid": "0b3c1f4e-7a2d-4c1e-9f6a-2d8e5b7c9a01",
  "timestamp_ms": 1795000000000,
  "signal": {"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1}
}
```

---

## Signing

1. Serialize the payload to a JSON string.
2. Sign the string's UTF-8 bytes with the trader's hotkey (sr25519). Encode the 64-byte signature as hex; a `0x` prefix is optional.
3. Send the **exact string** that was signed. The validator checks the signature over the bytes it receives and only then parses them, so the string must not be re-serialized on the way.

The validator does not require a particular serialization: any JSON string that meets the Payload rules and carries a valid signature is accepted. The reference form, used by the test vectors, is sorted keys (including inside `signal`) with no whitespace, Python's `json.dumps(payload, sort_keys=True, separators=(",", ":"))`.

A signature over the string wrapped as `<Bytes>…</Bytes>`, which is what browser wallet extensions such as polkadot.js sign, is accepted too.

sr25519 signatures are randomized: signing the same payload twice gives two different signatures, both valid. A signer cannot be checked by comparing signatures byte for byte, only by verifying them (see Test vectors).

---

## Sending

The PEM sends a `SendSignal` synapse from its own registered hotkey, with:

| Field | Value |
|-------|-------|
| `origin_payload` | The payload string, unchanged (at most 8,192 characters) |
| `origin_signature` | The hex signature (at most 256 characters) |
| `miner_order_uuid` | The payload's `miner_order_uuid` |
| `signal` | Empty, or the payload's `signal`. The payload's copy is the one used |
| `subaccount_id` | Not set |

As with any order, the PEM sends it to the mothership and to the other validators chosen by stake weight. The same envelope legitimately reaches several validators; each processes it once.

`SendSignal.required_hash_fields` is still `["signal"]`, so the dendrite's body hash, and every existing miner, is unaffected by the new fields.

---

## Validator checks

The envelope module checks the payload's structure and the trader's signature (`verify_envelope`). The validator then also checks, against live state:

1. The sending hotkey is a registered entity with the `permissionless_entity_miner` role, and `relay_hotkey` equals it.
2. `netuid` is the validator's netuid.
3. `timestamp_ms` is within the freshness window.
4. `origin_hotkey` is registered on the subnet, and is neither a synthetic subaccount hotkey nor an entity.
5. `miner_order_uuid` has not been used by this trader.

Then the order is processed as `origin_hotkey`, under that miner's account rules, and the order records the PEM as `relay_hotkey`. The signature itself is not stored.

---

## Test vectors

Keys are the well-known development keys: the trader is `//Alice` (`5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY`), the PEM is `//Bob` (`5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty`). Never use them for real funds.

A signer implementation is correct when:

1. From each vector's inputs it builds the payload string exactly as shown (reference form).
2. Each sample signature below verifies against its payload and `//Alice`'s public key in your implementation.
3. Signatures your implementation makes over these payloads with `//Alice` verify with `verify_envelope` in Python.

### 1. Market order, mainnet

Inputs: `netuid` 8, `miner_order_uuid` `0b3c1f4e-7a2d-4c1e-9f6a-2d8e5b7c9a01`, `timestamp_ms` 1795000000000, signal `{"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1}`

```
{"miner_order_uuid":"0b3c1f4e-7a2d-4c1e-9f6a-2d8e5b7c9a01","netuid":8,"origin_hotkey":"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY","relay_hotkey":"5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty","signal":{"execution_type":"MARKET","leverage":0.1,"order_type":"LONG","trade_pair":"BTCUSDC"},"timestamp_ms":1795000000000,"v":1}
```

Sample signature:

```
3608e544c6cbbfb67d92c4f2739a2c7a9750d05b118108abe7ab94961aa77670deb617f4ba22b260596383da8d0bde17d89fef2f84eba8d0fdce199b87968185
```

### 2. Limit order, mainnet

Inputs: `netuid` 8, `miner_order_uuid` `5e2a9c7d-3b1f-4d8e-a6c2-7f0b9e1d4c32`, `timestamp_ms` 1795000060000, signal `{"execution_type": "LIMIT", "trade_pair": "ETHUSDC", "order_type": "SHORT", "value": 2500.5, "limit_price": 3125.75}`

```
{"miner_order_uuid":"5e2a9c7d-3b1f-4d8e-a6c2-7f0b9e1d4c32","netuid":8,"origin_hotkey":"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY","relay_hotkey":"5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty","signal":{"execution_type":"LIMIT","limit_price":3125.75,"order_type":"SHORT","trade_pair":"ETHUSDC","value":2500.5},"timestamp_ms":1795000060000,"v":1}
```

Sample signature:

```
b218bcf6e59bfaaff8e3bf93c92621b98e160dcb57738ffc540589c4078b8944c4863e45cc5c87e0090ef50654903b01f885a60436fc328f529dc36781a0cb82
```

### 3. Flat all, testnet

Inputs: `netuid` 116, `miner_order_uuid` `9d4b2e6f-1c8a-4f3d-b7e5-0a2c6d8f1b43`, `timestamp_ms` 1795000120000, signal `{"execution_type": "FLAT_ALL"}`

```
{"miner_order_uuid":"9d4b2e6f-1c8a-4f3d-b7e5-0a2c6d8f1b43","netuid":116,"origin_hotkey":"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY","relay_hotkey":"5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty","signal":{"execution_type":"FLAT_ALL"},"timestamp_ms":1795000120000,"v":1}
```

Sample signature:

```
f0f2aa6c593ece2d90a196d92bde9dfb36c2fb355417da142bc54d6f7dfdcd7486d3fd639cb6c6a72247c885cd63aa696b032ffa6e1516c46ee6634bdb1ee182
```

---

## Python reference

```python
from bittensor_wallet import Keypair
from shared_objects.pem_envelope import build_payload, sign_payload, verify_envelope

trader = Keypair.create_from_uri("//Alice")   # on the trader's device
pem_hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

payload = build_payload(trader.ss58_address, pem_hotkey,
                        {"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1},
                        netuid=8)               # fills a uuid and the current time
signature = sign_payload(payload, trader)

parsed = verify_envelope(payload, signature)   # raises EnvelopeError if malformed or not signed by origin_hotkey
```
