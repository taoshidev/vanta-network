#!/usr/bin/env python3
"""
Burn excess slashed theta to reconcile the slash/burn discrepancy.

Background:
    Prior to collateral_sdk commit eb6769be, slash operations only moved tokens
    into the EVM contract's slashedCollateral pool without submitting the
    corresponding burn_alpha extrinsic on Subtensor.  This script submits the
    burn extrinsic for the outstanding amount so that on-chain stake matches the
    EVM state.

Usage:
    python runnable/burn_excess_slashed_collateral.py \
        --wallet.name  <coldkey-name>   \
        --wallet.hotkey <hotkey-name>   \
        --network  mainnet              \
        --amount   <theta-to-burn>      \
        [--dry-run]

    --network   mainnet | testnet  (default: mainnet)
    --amount    amount in theta tokens to burn (e.g. 42.5)
                if omitted the script prints the current slashedCollateral and exits
    --dry-run   build and display the extrinsic without submitting it

The vault wallet (--wallet.name / --wallet.hotkey) must be the same wallet that
is used by the validator (the coldkey signs the burn_alpha extrinsic and the
hotkey is the stake address being burned from).
"""

import argparse
import sys

from bittensor.wallet import Wallet
from collateral_sdk import CollateralManager, Network

from vali_objects.utils.vali_utils import ValiUtils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Burn excess slashed collateral to reconcile slash/burn discrepancy."
    )
    parser.add_argument("--wallet.name",   dest="wallet_name",   required=True,
                        help="Coldkey wallet name")
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", required=True,
                        help="Hotkey name")
    parser.add_argument("--wallet.path",   dest="wallet_path",   default="~/.bittensor/wallets",
                        help="Wallet directory (default: ~/.bittensor/wallets)")
    parser.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet",
                        help="Network to use (default: mainnet)")
    parser.add_argument("--amount", type=float, default=None,
                        help="Amount in theta tokens to burn. Omit to inspect state only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the extrinsic but do not submit it.")
    return parser.parse_args()


def theta_to_rao(theta: float) -> int:
    return int(theta * 10 ** 9)


def rao_to_theta(rao: int) -> float:
    return rao / 10 ** 9


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    network = Network.MAINNET if args.network == "mainnet" else Network.TESTNET
    collateral_manager = CollateralManager(network)

    # -----------------------------------------------------------------------
    # 1. Inspect current on-chain state
    # -----------------------------------------------------------------------
    slashed_rao = collateral_manager.get_slashed_collateral()
    total_rao   = collateral_manager.get_total_collateral()
    slashed_theta = rao_to_theta(slashed_rao)
    total_theta   = rao_to_theta(total_rao)

    print(f"Network             : {args.network}")
    print(f"EVM contract        : {collateral_manager.program_address}")
    print(f"Total collateral    : {total_theta:.9f} theta  ({total_rao} rao)")
    print(f"Slashed collateral  : {slashed_theta:.9f} theta  ({slashed_rao} rao)")
    print()

    if args.amount is None:
        print("No --amount supplied. Exiting after state inspection.")
        sys.exit(0)

    burn_theta = args.amount
    burn_rao   = theta_to_rao(burn_theta)

    print(f"Requested burn      : {burn_theta:.9f} theta  ({burn_rao} rao)")

    # -----------------------------------------------------------------------
    # 2. Load vault wallet
    # -----------------------------------------------------------------------
    wallet = Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey, path=args.wallet_path)
    vault_hotkey_ss58 = wallet.hotkey.ss58_address
    print(f"Vault coldkey       : {wallet.coldkeypub.ss58_address}")
    print(f"Vault hotkey        : {vault_hotkey_ss58}")
    print()

    # -----------------------------------------------------------------------
    # 3. Check vault stake on Subtensor
    # -----------------------------------------------------------------------
    staked = collateral_manager.subtensor_api.staking.get_stake(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=vault_hotkey_ss58,
        netuid=network.netuid,
    )
    print(f"Vault stake on chain: {rao_to_theta(staked.rao):.9f} theta  ({staked.rao} rao)")

    if burn_rao > staked.rao:
        print(
            f"\nERROR: requested burn ({burn_theta:.9f} theta) exceeds "
            f"vault stake ({rao_to_theta(staked.rao):.9f} theta). Aborting."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 4. Load secrets for wallet password (optional)
    # -----------------------------------------------------------------------
    wallet_password = ValiUtils.get_secret("gcp_vali_pw_name")

    # -----------------------------------------------------------------------
    # 5. Create the burn_alpha extrinsic
    # -----------------------------------------------------------------------
    print("Creating burn_alpha extrinsic...")
    extrinsic = collateral_manager.create_burn_alpha_extrinsic(
        amount=burn_rao,
        hotkey_ss58=vault_hotkey_ss58,
        vault_wallet=wallet,
        wallet_password=wallet_password,
    )
    print(f"Extrinsic created: {extrinsic}")
    print()

    if args.dry_run:
        print("DRY RUN — extrinsic NOT submitted.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # 6. Submit the extrinsic
    # -----------------------------------------------------------------------
    print(f"Submitting burn_alpha for {burn_theta:.9f} theta ({burn_rao} rao)...")
    collateral_manager._submit_extrinsic_with_retry(
        create_extrinsic_fn=lambda: collateral_manager.create_burn_alpha_extrinsic(
            amount=burn_rao,
            hotkey_ss58=vault_hotkey_ss58,
            vault_wallet=wallet,
            wallet_password=wallet_password,
        ),
        error_message="Failed to burn alpha tokens on Subtensor",
    )

    print()
    print(f"SUCCESS: burned {burn_theta:.9f} theta ({burn_rao} rao) from vault stake.")

    # -----------------------------------------------------------------------
    # 7. Verify updated vault stake
    # -----------------------------------------------------------------------
    staked_after = collateral_manager.subtensor_api.staking.get_stake(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=vault_hotkey_ss58,
        netuid=network.netuid,
    )
    print(f"Vault stake after   : {rao_to_theta(staked_after.rao):.9f} theta  ({staked_after.rao} rao)")
    print(f"Difference          : {rao_to_theta(staked.rao - staked_after.rao):.9f} theta")


if __name__ == "__main__":
    main()
