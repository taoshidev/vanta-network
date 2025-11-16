#!/usr/bin/env python3
"""
Query the weight setting interval (tempo) for a Bittensor subnet.

The tempo parameter controls how frequently validators can set weights.
It's measured in blocks (12 seconds per block on Bittensor).

Usage:
    python query_subnet_tempo.py [--netuid 8] [--network finney]
"""

import argparse
import bittensor as bt


def query_subnet_tempo(netuid: int = 8, network: str = "finney"):
    """
    Query the tempo (weight setting interval) for a subnet.

    Args:
        netuid: Subnet ID (default: 8)
        network: Network name (default: "finney")

    Returns:
        Tempo in blocks and derived time intervals
    """
    print(f"Querying subnet {netuid} on {network} network...")
    print("-" * 80)

    try:
        # Connect to subtensor
        subtensor = bt.subtensor(network=network)

        # Query tempo parameter
        print("Querying Tempo parameter...")
        tempo_blocks = subtensor.substrate.query(
            module='SubtensorModule',
            storage_function='Tempo',
            params=[netuid]
        )

        if tempo_blocks is None:
            print(f"ERROR: Could not query Tempo for subnet {netuid}")
            return None

        tempo = int(tempo_blocks.value)

        # Bittensor block time is 12 seconds
        BLOCK_TIME_SECONDS = 12

        # Calculate time intervals
        tempo_seconds = tempo * BLOCK_TIME_SECONDS
        tempo_minutes = tempo_seconds / 60
        tempo_hours = tempo_minutes / 60
        tempo_days = tempo_hours / 24

        # Calculate weight setting frequency per day
        weight_settings_per_day = (24 * 60 * 60) / tempo_seconds if tempo_seconds > 0 else 0

        print("\n" + "=" * 80)
        print("SUBNET TEMPO (Weight Setting Interval)")
        print("=" * 80)
        print(f"Subnet: {netuid}")
        print(f"Network: {network}")
        print(f"\nTempo: {tempo} blocks")
        print(f"\nTime Intervals:")
        print(f"  {tempo_seconds:.0f} seconds")
        print(f"  {tempo_minutes:.2f} minutes")
        print(f"  {tempo_hours:.4f} hours")
        print(f"  {tempo_days:.6f} days")
        print(f"\nWeight Setting Frequency:")
        print(f"  {weight_settings_per_day:.2f} times per day")
        print(f"  {weight_settings_per_day / 60:.2f} times per hour")
        print("=" * 80)

        # Additional context
        print("\nContext:")
        print(f"  Validators can commit new weights every {tempo} blocks")
        print(f"  This means weights can be updated approximately {weight_settings_per_day:.0f} times per day")
        print(f"  Each weight setting affects emission distribution until the next update")

        return {
            'tempo_blocks': tempo,
            'tempo_seconds': tempo_seconds,
            'tempo_minutes': tempo_minutes,
            'tempo_hours': tempo_hours,
            'weight_settings_per_day': weight_settings_per_day
        }

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Query subnet tempo (weight setting interval)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query subnet 8 on mainnet
  python query_subnet_tempo.py

  # Query different subnet
  python query_subnet_tempo.py --netuid 116 --network test

  # Query subnet 1
  python query_subnet_tempo.py --netuid 1
        """
    )

    parser.add_argument('--netuid', type=int, default=8,
                       help='Subnet ID (default: 8)')
    parser.add_argument('--network', type=str, default='finney',
                       help='Network name (default: finney)')

    args = parser.parse_args()

    bt.logging.enable_info()
    result = query_subnet_tempo(args.netuid, args.network)

    if result is None:
        exit(1)
