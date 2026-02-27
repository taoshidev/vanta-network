#!/usr/bin/env python3
"""
Create a Hyperliquid subaccount via the local miner REST API.

Usage:
    python create_hl_subaccount.py <hl_address> <account_size> [--payout-address <addr>]

Examples:
    python create_hl_subaccount.py 0xa1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2 25000
    python create_hl_subaccount.py 0xa1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2 25000 --payout-address 0xde1234567890abcdef1234567890abcdef123456
"""
import argparse
import json
import re
import sys

import requests

HL_ADDRESS_REGEX = r"^0x[a-fA-F0-9]{40}$"

MINER_API_HOST = "localhost"
MINER_API_PORT = 8088
API_KEY = "local_test_key"


def main():
    parser = argparse.ArgumentParser(description="Create a Hyperliquid subaccount")
    parser.add_argument("hl_address", help="Hyperliquid address (0x + 40 hex chars)")
    parser.add_argument("account_size", type=float, help="Account size in USD (> 0)")
    parser.add_argument("--payout-address", default=None, help="EVM payout address (0x + 40 hex chars)")
    parser.add_argument("--host", default=MINER_API_HOST, help=f"Miner API host (default: {MINER_API_HOST})")
    parser.add_argument("--port", type=int, default=MINER_API_PORT, help=f"Miner API port (default: {MINER_API_PORT})")
    parser.add_argument("--api-key", default=API_KEY, help="API key")
    args = parser.parse_args()

    if not re.match(HL_ADDRESS_REGEX, args.hl_address):
        print(f"Error: hl_address must be 0x followed by 40 hex characters, got: {args.hl_address}")
        sys.exit(1)

    if args.account_size <= 0:
        print(f"Error: account_size must be positive, got: {args.account_size}")
        sys.exit(1)

    if args.payout_address and not re.match(HL_ADDRESS_REGEX, args.payout_address):
        print(f"Error: payout_address must be 0x followed by 40 hex characters, got: {args.payout_address}")
        sys.exit(1)

    payload = {
        "hl_address": args.hl_address,
        "account_size": args.account_size,
    }
    if args.payout_address:
        payload["payout_address"] = args.payout_address

    url = f"http://{args.host}:{args.port}/api/create-hl-subaccount"

    print(f"Creating Hyperliquid subaccount...")
    print(f"  URL:            {url}")
    print(f"  HL Address:     {args.hl_address}")
    print(f"  Account Size:   ${args.account_size:,.2f}")
    if args.payout_address:
        print(f"  Payout Address: {args.payout_address}")
    print()

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {args.api_key}",
            },
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to miner API at {url}")
        print("Is the miner running? Check: pm2 status vanta_miner")
        sys.exit(1)

    print(f"HTTP {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(resp.text)

    sys.exit(0 if resp.status_code == 200 else 1)


if __name__ == "__main__":
    main()
