#!/usr/bin/env python3
# Copyright (c) 2025 Taoshi Inc
"""
Manual testing utility for the USDC payment service.

Usage:
    # Check wallet balances
    python runnable/test_usdc_payment.py --balance

    # Dry run (calculate payouts without sending)
    python runnable/test_usdc_payment.py --dry-run

    # Run payout for a specific period (dry run)
    python runnable/test_usdc_payment.py --dry-run --start-ms 1710000000000 --end-ms 1710604800000

    # Execute payouts for real
    python runnable/test_usdc_payment.py --execute

    # Show payment ledger history
    python runnable/test_usdc_payment.py --history
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miner_config import MinerConfig
from vali_objects.utils.vali_utils import ValiUtils


def load_config():
    """Load payment configuration from secrets."""
    secrets = ValiUtils.get_secrets(secrets_path=MinerConfig.get_secrets_file_path())

    usdc_private_key = os.environ.get("USDC_PRIVATE_KEY") or secrets.get("usdc_private_key")
    usdc_rpc_url = (
        os.environ.get("USDC_RPC_URL")
        or secrets.get("usdc_rpc_url")
        or MinerConfig.BASE_DEFAULT_RPC
    )
    validator_url = secrets.get("validator_url")
    validator_api_key = (
        os.environ.get("VALIDATOR_PAYOUT_API_KEY")
        or secrets.get("validator_payout_api_key")
    )
    entity_hotkey = secrets.get("entity_hotkey")

    # Try to get entity hotkey from wallet if not in secrets
    if not entity_hotkey:
        wallet_name = secrets.get("wallet_name")
        wallet_hotkey = secrets.get("wallet_hotkey")
        if wallet_name and wallet_hotkey:
            try:
                from bittensor_wallet import Wallet
                wallet = Wallet(name=wallet_name, hotkey=wallet_hotkey)
                entity_hotkey = wallet.hotkey.ss58_address
            except Exception as e:
                print(f"Warning: Could not load wallet for entity_hotkey: {e}")

    return {
        "usdc_private_key": usdc_private_key,
        "usdc_rpc_url": usdc_rpc_url,
        "validator_url": validator_url,
        "validator_api_key": validator_api_key,
        "entity_hotkey": entity_hotkey,
    }


def get_previous_week_period():
    """Calculate previous week's period (Sunday 00:00 UTC -> Sunday 00:00 UTC)."""
    now = datetime.now(timezone.utc)
    # Find the most recent Sunday 00:00 UTC
    days_since_sunday = now.weekday() + 1  # Monday=0, Sunday=6 -> +1 to get days since Sunday
    if days_since_sunday == 7:
        days_since_sunday = 0
    end = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_sunday)
    start = end - timedelta(days=7)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def cmd_balance(config):
    """Check wallet balances."""
    from entity_management.payment_ledger import PaymentLedger
    from entity_management.usdc_payment_service import USDCPaymentService

    ledger = PaymentLedger(MinerConfig.get_payment_ledger_file_path())
    service = USDCPaymentService(
        private_key=config["usdc_private_key"],
        rpc_url=config["usdc_rpc_url"],
        validator_url=config["validator_url"],
        validator_api_key=config["validator_api_key"],
        payment_ledger=ledger,
    )

    print(f"Wallet Address: {service.get_wallet_address()}")
    print(f"USDC Balance:   ${service.get_usdc_balance():.6f}")
    print(f"ETH Balance:    {service.get_eth_balance():.6f} ETH")
    print(f"Chain ID:       {MinerConfig.BASE_CHAIN_ID}")
    print(f"Total Paid Out: ${ledger.get_total_paid():.2f}")


def cmd_dry_run(config, start_ms, end_ms):
    """Dry run payout calculation."""
    from entity_management.payment_ledger import PaymentLedger
    from entity_management.usdc_payment_service import USDCPaymentService

    ledger = PaymentLedger(MinerConfig.get_payment_ledger_file_path())
    service = USDCPaymentService(
        private_key=config["usdc_private_key"],
        rpc_url=config["usdc_rpc_url"],
        validator_url=config["validator_url"],
        validator_api_key=config["validator_api_key"],
        payment_ledger=ledger,
    )

    entity_hotkey = config["entity_hotkey"]
    if not entity_hotkey:
        print("Error: entity_hotkey not configured")
        sys.exit(1)

    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    print(f"Entity:  {entity_hotkey}")
    print(f"Period:  {start_dt.isoformat()} to {end_dt.isoformat()}")
    print(f"Wallet:  {service.get_wallet_address()}")
    print(f"USDC:    ${service.get_usdc_balance():.6f}")
    print()

    result = service.process_payouts(
        entity_hotkey=entity_hotkey,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        dry_run=True
    )

    print(f"\nDry Run Results:")
    print(f"  Total subaccounts:  {result.total_subaccounts}")
    print(f"  Eligible:           {result.eligible_count}")
    print(f"  Skipped:            {result.skipped_count}")


def cmd_execute(config, start_ms, end_ms):
    """Execute payouts for real."""
    from entity_management.payment_ledger import PaymentLedger
    from entity_management.usdc_payment_service import USDCPaymentService

    ledger = PaymentLedger(MinerConfig.get_payment_ledger_file_path())
    service = USDCPaymentService(
        private_key=config["usdc_private_key"],
        rpc_url=config["usdc_rpc_url"],
        validator_url=config["validator_url"],
        validator_api_key=config["validator_api_key"],
        payment_ledger=ledger,
    )

    entity_hotkey = config["entity_hotkey"]
    if not entity_hotkey:
        print("Error: entity_hotkey not configured")
        sys.exit(1)

    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    print(f"Entity:  {entity_hotkey}")
    print(f"Period:  {start_dt.isoformat()} to {end_dt.isoformat()}")
    print(f"Wallet:  {service.get_wallet_address()}")
    print(f"USDC:    ${service.get_usdc_balance():.6f}")
    print(f"ETH:     {service.get_eth_balance():.6f} ETH")

    confirm = input("\nProceed with REAL payments? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        sys.exit(0)

    result = service.process_payouts(
        entity_hotkey=entity_hotkey,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        dry_run=False
    )

    print(f"\nExecution Results:")
    print(f"  Total subaccounts:  {result.total_subaccounts}")
    print(f"  Eligible:           {result.eligible_count}")
    print(f"  Successful:         {result.successful_count}")
    print(f"  Failed:             {result.failed_count}")
    print(f"  Skipped:            {result.skipped_count}")
    print(f"  Total Paid:         ${result.total_usd_paid:.2f} USDC")

    for payment in result.payments:
        status_icon = "OK" if payment.status == "confirmed" else "FAIL"
        print(
            f"  [{status_icon}] {payment.synthetic_hotkey}: "
            f"${payment.amount_usd:.2f} -> {payment.payout_address} "
            f"tx={payment.tx_hash or 'N/A'}"
        )


def cmd_history():
    """Show payment ledger history."""
    from entity_management.payment_ledger import PaymentLedger

    ledger = PaymentLedger(MinerConfig.get_payment_ledger_file_path())
    payments = ledger.get_all_payments()

    if not payments:
        print("No payment records found.")
        return

    print(f"Payment Ledger ({len(payments)} records):")
    print(f"{'Status':<12} {'Amount':>10} {'Subaccount':<20} {'Tx Hash':<20} {'Date'}")
    print("-" * 80)

    for p in sorted(payments, key=lambda x: x.created_at_ms, reverse=True):
        dt = datetime.fromtimestamp(p.created_at_ms / 1000, tz=timezone.utc)
        tx_short = (p.tx_hash[:16] + "...") if p.tx_hash else "N/A"
        hotkey_short = p.synthetic_hotkey[:18] + ".." if len(p.synthetic_hotkey) > 20 else p.synthetic_hotkey
        print(f"{p.status:<12} ${p.amount_usd:>9.2f} {hotkey_short:<20} {tx_short:<20} {dt.strftime('%Y-%m-%d %H:%M')}")

    print(f"\nTotal paid (confirmed): ${ledger.get_total_paid():.2f}")


def main():
    parser = argparse.ArgumentParser(description="USDC Payment Service Test Utility")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--balance", action="store_true", help="Check wallet balances")
    group.add_argument("--dry-run", action="store_true", help="Dry run payout calculation")
    group.add_argument("--execute", action="store_true", help="Execute payouts for real")
    group.add_argument("--history", action="store_true", help="Show payment ledger history")

    parser.add_argument("--start-ms", type=int, help="Period start timestamp (ms)")
    parser.add_argument("--end-ms", type=int, help="Period end timestamp (ms)")

    args = parser.parse_args()

    if args.history:
        cmd_history()
        return

    config = load_config()

    # Validate required config
    if not args.history:
        missing = []
        if not config["usdc_private_key"]:
            missing.append("usdc_private_key / USDC_PRIVATE_KEY")
        if not config["validator_url"]:
            missing.append("validator_url")
        if not config["validator_api_key"]:
            missing.append("validator_payout_api_key / VALIDATOR_PAYOUT_API_KEY")

        if missing:
            print(f"Error: Missing required configuration: {', '.join(missing)}")
            print("Configure in mining/miner_secrets.json or via environment variables.")
            sys.exit(1)

    if args.balance:
        cmd_balance(config)
    elif args.dry_run or args.execute:
        # Default to previous week if no timestamps provided
        if args.start_ms and args.end_ms:
            start_ms, end_ms = args.start_ms, args.end_ms
        else:
            start_ms, end_ms = get_previous_week_period()
            print(f"Using previous week period (default)")

        if args.dry_run:
            cmd_dry_run(config, start_ms, end_ms)
        else:
            cmd_execute(config, start_ms, end_ms)


if __name__ == "__main__":
    main()
