#!/usr/bin/env python3
"""Query coldkey for a given hotkey on Bittensor testnet"""

import bittensor as bt

# Configuration
HOTKEY = "5G1iCdwUBjnXxGfJYzho1dToWTCkYyBF6Vq5sAJP7ftHKE1b"
NETUID = 116  # testnet subnet

def main():
    print(f"Querying coldkey for hotkey: {HOTKEY}")
    print(f"Network: testnet")
    print(f"Netuid: {NETUID}")
    print("-" * 60)

    try:
        # Connect to testnet subtensor
        subtensor = bt.subtensor(network="test")
        print(f"Connected to subtensor: {subtensor.network}")

        # Get metagraph for the subnet
        print(f"Loading metagraph for netuid {NETUID}...")
        metagraph = subtensor.metagraph(netuid=NETUID)
        print(f"Metagraph loaded: {len(metagraph.hotkeys)} hotkeys registered")

        # Find the hotkey in the metagraph
        if HOTKEY in metagraph.hotkeys:
            idx = metagraph.hotkeys.index(HOTKEY)
            coldkey = metagraph.coldkeys[idx]

            print("\n" + "=" * 60)
            print("RESULT:")
            print("=" * 60)
            print(f"Hotkey:  {HOTKEY}")
            print(f"Coldkey: {coldkey}")
            print("=" * 60)

            # Additional info
            print(f"\nAdditional Info:")
            print(f"  UID: {metagraph.uids[idx]}")
            print(f"  Stake: {metagraph.S[idx]:.4f} TAO")
            print(f"  Trust: {metagraph.T[idx]:.4f}")
            print(f"  Incentive: {metagraph.I[idx]:.4f}")
            print(f"  Emission: {metagraph.E[idx]:.4f}")

        else:
            print(f"\n❌ Hotkey {HOTKEY} not found in subnet {NETUID}")
            print(f"   It may not be registered or may be registered to a different subnet")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
