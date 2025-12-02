#!/usr/bin/env python3
"""
Test script to check if metagraph provides TAO and ALPHA reserve data.
"""
import bittensor as bt

# Initialize subtensor and metagraph
print("Initializing subtensor and metagraph...")
import argparse

parser = argparse.ArgumentParser()
bt.subtensor.add_args(parser)
config = bt.config(parser)
config.subtensor.network = "finney"

netuid = 8
subtensor = bt.subtensor(config=config)
metagraph = subtensor.metagraph(netuid)

print(f"\nMetagraph synced for netuid {netuid}")
print(f"Total hotkeys: {len(metagraph.hotkeys)}")

# Check all attributes on metagraph
print("\n" + "="*80)
print("METAGRAPH ATTRIBUTES:")
print("="*80)

# Standard attributes we know about
standard_attrs = [
    'hotkeys', 'neurons', 'uids', 'block_at_registration', 'emission',
    'incentive', 'consensus', 'trust', 'validator_trust', 'dividends',
    'stake', 'ranks', 'active', 'axons'
]

for attr in standard_attrs:
    if hasattr(metagraph, attr):
        value = getattr(metagraph, attr)
        if isinstance(value, list):
            print(f"✓ {attr}: list with {len(value)} items")
            if len(value) > 0 and attr == 'emission':
                print(f"  └─ Sample: {value[0]}")
        else:
            print(f"✓ {attr}: {type(value).__name__}")

# Check for reserve-related attributes
print("\n" + "="*80)
print("CHECKING FOR RESERVE-RELATED ATTRIBUTES:")
print("="*80)

reserve_keywords = [
    'reserve', 'tao', 'alpha', 'pool', 'liquidity',
    'subnet_tao', 'subnet_alpha', 'alpha_in', 'alpha_out'
]

found_reserve_attrs = []
for attr in dir(metagraph):
    if any(keyword in attr.lower() for keyword in reserve_keywords):
        found_reserve_attrs.append(attr)
        value = getattr(metagraph, attr, None)
        print(f"✓ {attr}: {value if not callable(value) else 'method'}")

if not found_reserve_attrs:
    print("✗ No reserve-related attributes found on metagraph")

# Check subtensor for reserve query methods
print("\n" + "="*80)
print("CHECKING SUBTENSOR FOR RESERVE DATA:")
print("="*80)

try:
    # Try to query reserves directly
    tao_reserve_query = subtensor.substrate.query(
        module='SubtensorModule',
        storage_function='SubnetTAO',
        params=[netuid]
    )

    alpha_reserve_query = subtensor.substrate.query(
        module='SubtensorModule',
        storage_function='SubnetAlphaIn',
        params=[netuid]
    )

    if tao_reserve_query and alpha_reserve_query:
        tao_rao = float(tao_reserve_query.value if hasattr(tao_reserve_query, 'value') else tao_reserve_query)
        alpha_rao = float(alpha_reserve_query.value if hasattr(alpha_reserve_query, 'value') else alpha_reserve_query)

        print(f"✓ TAO Reserve (substrate query): {tao_rao / 1e9:.2f} TAO ({tao_rao:.0f} RAO)")
        print(f"✓ ALPHA Reserve (substrate query): {alpha_rao / 1e9:.2f} ALPHA ({alpha_rao:.0f} RAO)")
        print(f"✓ Conversion Rate: {tao_rao / alpha_rao:.6f} TAO/ALPHA")

        # Compare with metagraph.pool values
        print("\n" + "="*80)
        print("VALUE COMPARISON: Substrate Query vs metagraph.pool")
        print("="*80)

        if hasattr(metagraph, 'pool') and metagraph.pool:
            # Metagraph pool values are in tokens, convert to RAO for comparison
            tao_reserve_rao_from_pool = metagraph.pool.tao_in * 1e9
            alpha_reserve_rao_from_pool = metagraph.pool.alpha_in * 1e9

            print(f"\nTAO Reserve:")
            print(f"  Substrate query: {tao_rao:.0f} RAO ({tao_rao / 1e9:.2f} TAO)")
            print(f"  metagraph.pool:  {tao_reserve_rao_from_pool:.0f} RAO ({metagraph.pool.tao_in:.2f} TAO)")
            tao_diff = abs(tao_rao - tao_reserve_rao_from_pool)
            tao_match = tao_diff < 1.0  # Within 1 RAO
            print(f"  Difference: {tao_diff:.0f} RAO {'✓ MATCH' if tao_match else '✗ MISMATCH'}")

            print(f"\nALPHA Reserve:")
            print(f"  Substrate query: {alpha_rao:.0f} RAO ({alpha_rao / 1e9:.2f} ALPHA)")
            print(f"  metagraph.pool:  {alpha_reserve_rao_from_pool:.0f} RAO ({metagraph.pool.alpha_in:.2f} ALPHA)")
            alpha_diff = abs(alpha_rao - alpha_reserve_rao_from_pool)
            alpha_match = alpha_diff < 1.0  # Within 1 RAO
            print(f"  Difference: {alpha_diff:.0f} RAO {'✓ MATCH' if alpha_match else '✗ MISMATCH'}")

            if tao_match and alpha_match:
                print("\n→ CONCLUSION: metagraph.pool provides IDENTICAL reserve data!")
                print("→ Separate substrate queries are NOT needed - use metagraph.pool instead")
            else:
                print("\n→ CONCLUSION: Values differ - separate substrate queries may be necessary")
        else:
            print("✗ metagraph.pool not available for comparison")
    else:
        print("✗ Reserve queries returned None")

except Exception as e:
    print(f"✗ Error querying reserves: {e}")

# Check all metagraph attributes for completeness
print("\n" + "="*80)
print("ALL METAGRAPH ATTRIBUTES (excluding private/methods):")
print("="*80)
all_attrs = [attr for attr in dir(metagraph) if not attr.startswith('_') and not callable(getattr(metagraph, attr))]
print(f"Total public attributes: {len(all_attrs)}")
for attr in sorted(all_attrs):
    print(f"  - {attr}")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print("Standard metagraph.sync() provides:")
print("  ✓ Hotkeys, neurons, UIDs")
print("  ✓ Emission rates (TAO per tempo per UID)")
print("  ✓ Incentive, consensus, trust, stake")
print("  ✓ metagraph.pool with reserve data (alpha_in, tao_in)")
print("\nReserve Data Sources:")
print("  1. metagraph.pool.tao_in / alpha_in (built-in)")
print("  2. Substrate queries: SubnetTAO / SubnetAlphaIn (our implementation)")
print("\n→ See VALUE COMPARISON section above to determine if separate queries are needed")
