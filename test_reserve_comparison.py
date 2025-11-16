#!/usr/bin/env python3
"""
Compare substrate query values with metagraph.pool values.
"""
import bittensor as bt
import argparse

# Initialize subtensor and metagraph
print("Initializing subtensor and metagraph...")
parser = argparse.ArgumentParser()
bt.subtensor.add_args(parser)
config = bt.config(parser)
config.subtensor.network = "finney"

netuid = 8
subtensor = bt.subtensor(config=config)
metagraph = subtensor.metagraph(netuid)

print(f"\nMetagraph synced for netuid {netuid}")

# Query substrate directly (our implementation)
print("\n" + "="*80)
print("SUBSTRATE QUERY (Our Implementation)")
print("="*80)

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

tao_rao_from_query = float(tao_reserve_query.value if hasattr(tao_reserve_query, 'value') else tao_reserve_query)
alpha_rao_from_query = float(alpha_reserve_query.value if hasattr(alpha_reserve_query, 'value') else alpha_reserve_query)

print(f"TAO Reserve:   {tao_rao_from_query:.0f} RAO ({tao_rao_from_query / 1e9:.9f} TAO)")
print(f"ALPHA Reserve: {alpha_rao_from_query:.0f} RAO ({alpha_rao_from_query / 1e9:.9f} ALPHA)")
print(f"Conversion:    {tao_rao_from_query / alpha_rao_from_query:.9f} TAO/ALPHA")

# Get metagraph.pool values
print("\n" + "="*80)
print("METAGRAPH.POOL (Built-in)")
print("="*80)

print(f"metagraph.pool: {metagraph.pool}")
print(f"\nTAO Reserve:   {metagraph.pool.tao_in * 1e9:.0f} RAO ({metagraph.pool.tao_in:.9f} TAO)")
print(f"ALPHA Reserve: {metagraph.pool.alpha_in * 1e9:.0f} RAO ({metagraph.pool.alpha_in:.9f} ALPHA)")
print(f"Conversion:    {metagraph.pool.tao_in / metagraph.pool.alpha_in:.9f} TAO/ALPHA")

# Compare values
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

tao_rao_from_pool = metagraph.pool.tao_in * 1e9
alpha_rao_from_pool = metagraph.pool.alpha_in * 1e9

tao_diff = abs(tao_rao_from_query - tao_rao_from_pool)
alpha_diff = abs(alpha_rao_from_query - alpha_rao_from_pool)

print(f"\nTAO Reserve Difference:   {tao_diff:.0f} RAO")
print(f"ALPHA Reserve Difference: {alpha_diff:.0f} RAO")

# Tolerance: within 1 RAO
tao_match = tao_diff < 1.0
alpha_match = alpha_diff < 1.0

print(f"\nTAO Match (< 1 RAO):   {'✓ YES' if tao_match else '✗ NO'}")
print(f"ALPHA Match (< 1 RAO): {'✓ YES' if alpha_match else '✗ NO'}")

if tao_match and alpha_match:
    print("\n" + "="*80)
    print("CONCLUSION: VALUES ARE IDENTICAL")
    print("="*80)
    print("✓ metagraph.pool provides the SAME reserve data as substrate queries")
    print("✓ We can simplify by using metagraph.pool instead of separate queries")
    print("✓ This eliminates the need for custom substrate queries")
else:
    print("\n" + "="*80)
    print("CONCLUSION: VALUES DIFFER")
    print("="*80)
    print("✗ Separate substrate queries are necessary")
    print(f"✗ TAO difference: {tao_diff:.0f} RAO")
    print(f"✗ ALPHA difference: {alpha_diff:.0f} RAO")
