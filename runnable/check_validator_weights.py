#!/usr/bin/env python3
"""
Script to check the last-set weights for a specific validator on subnet 8 (mainnet).
Displays miners sorted by weight assigned.

Usage:
    python check_validator_weights.py
"""

import bittensor as bt
import sys

def check_validator_weights(
    validator_hotkey: str,
    netuid: int = 8,
    network: str = "finney",
    tao_price_usd: float = 400.0,
    daily_subnet_emissions_tao: float = 225.0
):
    """
    Check the weights set by a specific validator.

    Args:
        validator_hotkey: The hotkey of the validator to check
        netuid: The subnet ID (default: 8)
        network: The network to connect to (default: "finney")
        tao_price_usd: Current TAO/USD price (default: 400)
        daily_subnet_emissions_tao: Estimated daily TAO emissions for the subnet (default: 225)
    """
    print(f"Connecting to {network} network, subnet {netuid}...")
    print(f"Validator hotkey: {validator_hotkey}")
    print(f"TAO Price: ${tao_price_usd:.2f}")
    print(f"Estimated Daily Subnet Emissions: {daily_subnet_emissions_tao:.2f} TAO")
    print("-" * 80)

    try:
        # Connect to subtensor
        subtensor = bt.Subtensor(network=network)

        # Query immunity period for the subnet
        print("Querying subnet parameters...")
        try:
            immunity_period_blocks = subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='ImmunityPeriod',
                params=[netuid]
            )

            if immunity_period_blocks is not None:
                immunity_blocks = int(immunity_period_blocks.value)
                # Bittensor block time is 12 seconds
                BLOCK_TIME_SECONDS = 12
                immunity_seconds = immunity_blocks * BLOCK_TIME_SECONDS
                immunity_hours = immunity_seconds / 3600
                immunity_days = immunity_hours / 24

                print(f"Immunity Period: {immunity_blocks} blocks ({immunity_hours:.2f} hours / {immunity_days:.2f} days)")
            else:
                print("Immunity Period: Could not query (using default: 7200 blocks / 1 day)")
        except Exception as e:
            print(f"Warning: Could not query immunity period: {e}")
            print("Using default estimate: 7200 blocks / 1 day")

        print("-" * 80)

        # Get metagraph
        print("Fetching metagraph...")
        metagraph = subtensor.metagraph(netuid)

        # Find the validator's UID
        validator_uid = None
        for neuron in metagraph.neurons:
            if neuron.hotkey == validator_hotkey:
                validator_uid = neuron.uid
                break

        if validator_uid is None:
            print(f"ERROR: Validator {validator_hotkey} not found in subnet {netuid}")
            sys.exit(1)

        print(f"Found validator at UID {validator_uid}")
        print("-" * 80)

        # Query weights directly from the chain for this validator
        print("Querying weights from chain...")
        try:
            # Method 1: Try using substrate query directly
            weights_data = subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='Weights',
                params=[netuid, validator_uid]
            )

            if weights_data is None or not weights_data.value:
                print(f"No weights found for validator UID {validator_uid}")
                print("This validator may not have set weights yet, or weights have expired.")
                sys.exit(0)

            # Parse the weights
            # weights_data.value is a list of tuples: [(uid, weight), (uid, weight), ...]
            # First pass: get raw weights and calculate sum
            raw_weights = {}
            raw_sum = 0
            for uid, weight in weights_data.value:
                raw_weights[uid] = float(weight)
                raw_sum += float(weight)

            # Second pass: normalize so weights sum to 1.0
            weights_dict = {}
            if raw_sum > 0:
                for uid, weight in raw_weights.items():
                    weights_dict[uid] = weight / raw_sum
            else:
                weights_dict = raw_weights

        except Exception as e:
            print(f"Error querying weights: {e}")
            print("Trying alternative method...")

            # Method 2: Try getting weights from metagraph.weights attribute
            try:
                # Some bittensor versions store weights differently
                all_weights = subtensor.weights(netuid)
                if validator_uid >= len(all_weights):
                    print(f"ERROR: Validator UID {validator_uid} out of range")
                    sys.exit(1)

                validator_weight_list = all_weights[validator_uid]
                weights_dict = {uid: float(w) for uid, w in enumerate(validator_weight_list) if w > 0}

            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
                sys.exit(1)

        # Build list of (uid, hotkey, weight) tuples
        weight_data = []
        total_weight = 0
        non_zero_count = 0

        for uid in range(len(metagraph.hotkeys)):
            weight = weights_dict.get(uid, 0.0)
            if weight > 0:
                non_zero_count += 1
                total_weight += weight

            miner_hotkey = metagraph.hotkeys[uid]
            weight_data.append((uid, miner_hotkey, weight))

        # Sort by weight (descending)
        weight_data.sort(key=lambda x: x[2], reverse=True)

        # Print summary
        print(f"Total miners in subnet: {len(metagraph.hotkeys)}")
        print(f"Miners with non-zero weight: {non_zero_count}")
        print(f"Total weight assigned: {total_weight:.6f}")
        print("-" * 80)

        # Print weights with daily USD estimates
        print(f"\n{'UID':<6} {'Weight':<12} {'Daily TAO':<12} {'Daily USD':<12} {'Hotkey'}")
        print("-" * 80)

        for uid, hotkey, weight in weight_data:
            if weight > 0:  # Only show miners with non-zero weights
                percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                daily_tao = weight * daily_subnet_emissions_tao
                daily_usd = daily_tao * tao_price_usd
                print(f"{uid:<6} {weight:<12.6f} ({percentage:>6.3f}%)  {daily_tao:<12.6f} ${daily_usd:<11.2f} {hotkey}")

        # Also show zero-weight miners count
        zero_weight_count = len(metagraph.hotkeys) - non_zero_count
        if zero_weight_count > 0:
            print("-" * 80)
            print(f"\n{zero_weight_count} miners have zero weight (not shown above)")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Check validator weights on Bittensor subnet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check single validator
  python check_validator_weights.py --hotkey 5HmkM6X1D3W3CuCSPuHhrbYyZNBy2aGAiZy9NczoJmtY25H7

  # Check multiple validators
  python check_validator_weights.py --hotkeys 5HmkM6X1D3W3CuCSPuHhrbYyZNBy2aGAiZy9NczoJmtY25H7 5DWjXZcmPcwg4SpBmG3UeTpZXVP3SgP7fP6MKeWCQZMLv56K

  # Custom TAO price and emissions
  python check_validator_weights.py --hotkeys validator1 validator2 --tao-price 500 --daily-emissions 300
        """
    )

    # Make hotkey and hotkeys mutually exclusive
    hotkey_group = parser.add_mutually_exclusive_group()
    hotkey_group.add_argument('--hotkey', type=str,
                             help='Single validator hotkey to check')
    hotkey_group.add_argument('--hotkeys', type=str, nargs='+',
                             help='Multiple validator hotkeys to check (space-separated)')

    parser.add_argument('--netuid', type=int, default=8, help='Subnet ID (default: 8)')
    parser.add_argument('--network', type=str, default='finney', help='Network name (default: finney)')
    parser.add_argument('--tao-price', type=float, default=400.0,
                       help='TAO/USD price for calculations (default: 400)')
    parser.add_argument('--daily-emissions', type=float, default=225.0,
                       help='Estimated daily subnet TAO emissions (default: 225)')

    args = parser.parse_args()

    bt.logging.enable_info()

    # Determine which hotkeys to check
    if args.hotkeys:
        hotkeys_to_check = args.hotkeys
    elif args.hotkey:
        hotkeys_to_check = [args.hotkey]
    else:
        # Default hotkey if none provided
        hotkeys_to_check = ['5DWjXZcmPcwg4SpBmG3UeTpZXVP3SgP7fP6MKeWCQZMLv56K', '5FeNwZ5oAqcJMitNqGx71vxGRWJhsdTqxFGVwPRfg8h2UZmo']

    # Check each validator
    for i, hotkey in enumerate(hotkeys_to_check):
        if i > 0:
            print("\n" + "=" * 80)
            print("=" * 80)
            print("\n")

        check_validator_weights(
            validator_hotkey=hotkey,
            netuid=args.netuid,
            network=args.network,
            tao_price_usd=args.tao_price,
            daily_subnet_emissions_tao=args.daily_emissions
        )

    # Print summary if multiple validators
    if len(hotkeys_to_check) > 1:
        print("\n" + "=" * 80)
        print(f"SUMMARY: Checked {len(hotkeys_to_check)} validators on subnet {args.netuid}")
        print("=" * 80)
