#!/usr/bin/env python3
"""
Generate default_departed_hotkeys.json file for re-registration tracking.

This script:
1. Queries the current metagraph for subnet 8 to get all active hotkeys
2. Reads ALL historical eliminations from the taoshi.ts database
3. Identifies eliminated hotkeys that are NOT currently in the metagraph (departed)
4. Creates the default_departed_hotkeys.json file in data/ directory for commit to repo

This default file serves as a fallback when the runtime departed_hotkeys.json
doesn't exist (e.g., after a fresh validator deployment or data migration).
"""

import os
# Set taoshi-ts environment variables for database access
os.environ["TAOSHI_TS_DEPLOYMENT"] = "DEVELOPMENT"
os.environ["TAOSHI_TS_PLATFORM"] = "LOCAL"

import bittensor as bt
import argparse
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from time_util.time_util import TimeUtil

# Default to mainnet subnet 8
DEFAULT_NETUID = 8
DEFAULT_NETWORK = "finney"

def main():
    parser = argparse.ArgumentParser(
        description='Generate default_departed_hotkeys.json file from historical database',
        add_help=True
    )
    bt.logging.add_args(parser)
    parser.add_argument(
        '--netuid',
        type=int,
        default=DEFAULT_NETUID,
        help=f'Subnet netuid (default: {DEFAULT_NETUID})'
    )
    parser.add_argument(
        '--network',
        type=str,
        default=DEFAULT_NETWORK,
        help=f'Network to connect to: finney (mainnet) or test (testnet) (default: {DEFAULT_NETWORK})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: data/default_departed_hotkeys.json)'
    )

    config = bt.config(parser)
    args = config

    print("=" * 80)
    print("GENERATE DEFAULT DEPARTED HOTKEYS FILE FROM HISTORICAL DATABASE")
    print("=" * 80)
    print(f"Network: {args.network}")
    print(f"Netuid: {args.netuid}")
    print()

    # Step 2: Read ALL historical eliminations from database
    print("Step 2: Reading ALL historical eliminations from database...")
    print("  (querying directly to bypass DI container issues)")

    try:
        # Read database URL from config-development.json (like daily_portfolio_returns.py does)
        import json
        config_file = "config-development.json"
        if not os.path.exists(config_file):
            print(f"✗ Error: {config_file} not found in current directory")
            print(f"  Current directory: {os.getcwd()}")
            print(f"  Please run this script from the repo root directory")
            return 1

        with open(config_file, 'r') as f:
            config = json.load(f)

        db_url = config.get('secrets', {}).get('db_ptn_editor_url')
        if not db_url:
            print(f"✗ Error: db_ptn_editor_url not found in {config_file}")
            return 1

        print(f"  Database: {db_url.split('@')[1].split('/')[0] if '@' in db_url else 'configured'}")

        # Query database directly (bypassing DI container issues)
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from taoshi.ts.model import EliminationModel

        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Query all elimination records
        elimination_records = session.query(EliminationModel).all()
        print(f"✓ Loaded {len(elimination_records)} elimination records from database")

        # Convert to dict format
        all_eliminations = []
        for elim in elimination_records:
            all_eliminations.append({
                'hotkey': elim.miner_hotkey,
                'miner_hotkey': elim.miner_hotkey,
                'max_drawdown': elim.max_drawdown,
                'elimination_time_ms': elim.elimination_ms,
                'elimination_ms': elim.elimination_ms,
                'elimination_reason': elim.elimination_reason,
                'creation_ms': elim.creation_ms,
                'updated_ms': elim.updated_ms,
            })

        session.close()

        # Calculate summary
        if all_eliminations:
            timestamps = [e['elimination_ms'] for e in all_eliminations if e.get('elimination_ms')]
            if timestamps:
                print(f"  Time range: {TimeUtil.millis_to_formatted_date_str(min(timestamps))} to {TimeUtil.millis_to_formatted_date_str(max(timestamps))}")

            from collections import Counter
            reasons = Counter(e['elimination_reason'] for e in all_eliminations if e.get('elimination_reason'))
            print(f"  Reasons: {dict(reasons)}")

    except Exception as e:
        print(f"✗ Error loading eliminations from database: {e}")
        import traceback
        traceback.print_exc()
        return 1


    # Step 1: Query the metagraph for current hotkeys
    print("Step 1: Querying metagraph for current hotkeys...")
    try:
        subtensor = bt.subtensor(network=args.network)
        print(f"Connected to subtensor: {subtensor.network}")

        metagraph = subtensor.metagraph(netuid=args.netuid)
        current_hotkeys = set(metagraph.hotkeys) if metagraph.hotkeys else set()
        print(f"✓ Loaded metagraph: {len(current_hotkeys)} hotkeys currently registered")
    except Exception as e:
        print(f"✗ Error querying metagraph: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()


    print()

    # Step 3: Identify departed hotkeys
    print("Step 3: Identifying departed hotkeys...")
    print("  (eliminated hotkeys NOT currently in metagraph)")

    eliminated_hotkeys = set()
    hotkey_to_elimination_time = {}

    for elimination in all_eliminations:
        hotkey = elimination.get('hotkey') or elimination.get('miner_hotkey')
        elim_time_ms = elimination.get('elimination_time_ms') or elimination.get('elimination_ms', 0)
        reason = elimination.get('elimination_reason', 'UNKNOWN')

        if hotkey:
            eliminated_hotkeys.add(hotkey)
            # Keep the earliest elimination time for each hotkey
            if hotkey not in hotkey_to_elimination_time:
                hotkey_to_elimination_time[hotkey] = (elim_time_ms, reason)
            else:
                # Keep earliest time
                existing_time, existing_reason = hotkey_to_elimination_time[hotkey]
                if elim_time_ms < existing_time:
                    hotkey_to_elimination_time[hotkey] = (elim_time_ms, reason)

    print(f"  Found {len(eliminated_hotkeys)} unique eliminated hotkeys from database")

    # Departed = eliminated AND not in current metagraph
    departed_hotkeys = eliminated_hotkeys - current_hotkeys

    print(f"  Current metagraph has {len(current_hotkeys)} hotkeys")
    print(f"✓ Identified {len(departed_hotkeys)} departed hotkeys")

    print()

    # Step 4: Generate the default_departed_hotkeys file
    print("Step 4: Generating default_departed_hotkeys.json...")

    # Create the departed_hotkeys dict with metadata
    departed_dict = {}
    current_time_ms = TimeUtil.now_in_millis()

    for hotkey in sorted(departed_hotkeys):
        elim_time_ms, reason = hotkey_to_elimination_time.get(
            hotkey,
            (current_time_ms, 'UNKNOWN')
        )
        departed_dict[hotkey] = {
            "detected_ms": elim_time_ms
        }
        print(f"  • {hotkey[:16]}... (eliminated: {reason}, {TimeUtil.millis_to_formatted_date_str(elim_time_ms)})")

    # Prepare the file data
    from vali_objects.utils.elimination_manager import DEPARTED_HOTKEYS_KEY
    file_data = {
        DEPARTED_HOTKEYS_KEY: departed_dict
    }

    # Determine output path - default to data/ directory for commit to repo
    if args.output:
        output_path = args.output
    else:
        # Store in data/ directory with default_ prefix to distinguish from runtime file
        base_dir = ValiBkpUtils.get_vali_dir(running_unit_tests=False).replace('/validation/', '')
        output_path = os.path.join(base_dir, 'data', 'default_departed_hotkeys.json')

    print()
    print(f"Writing to: {output_path}")

    try:
        ValiBkpUtils.write_file(output_path, file_data)
        print(f"✓ Successfully wrote {len(departed_dict)} departed hotkeys to file")
    except Exception as e:
        print(f"✗ Error writing file: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total eliminations in database: {len(all_eliminations)}")
    print(f"Unique eliminated hotkeys:      {len(eliminated_hotkeys)}")
    print(f"Currently in metagraph:         {len(current_hotkeys)}")
    print(f"Departed (not in metagraph):    {len(departed_hotkeys)}")
    print()
    print(f"✓ Default departed_hotkeys.json created successfully!")
    print(f"  File: {output_path}")
    print()
    print("This file should be committed to the repository.")
    print("It will be used as a fallback when validation/departed_hotkeys.json doesn't exist.")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    exit(main())
