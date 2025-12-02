# Default Departed Hotkeys File

## Overview

The `default_departed_hotkeys.json` file contains a historical record of all hotkeys that have been eliminated from the network and subsequently de-registered from the metagraph. This file serves as a fallback when the runtime `validation/departed_hotkeys.json` file doesn't exist (e.g., on fresh validator deployments).

## Purpose

The departed hotkeys tracking system prevents miners who have been eliminated and de-registered from re-registering to the subnet. When a hotkey appears in both:
1. The current metagraph (actively registered)
2. The departed_hotkeys data (previously departed)

The validator will reject all orders from that hotkey with a re-registration error message.

## File Structure

```json
{
  "departed_hotkeys": {
    "5FxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxG": {
      "detected_ms": 1758835913415
    },
    "5GxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxC5": {
      "detected_ms": 1759363497136
    }
  }
}
```

Each entry maps a hotkey to metadata containing:
- `detected_ms`: Timestamp (in milliseconds) when the hotkey was eliminated/departed

## How It Works

1. **Runtime Operation**: The validator tracks departures in `validation/departed_hotkeys.json` during normal operation
2. **Fallback**: If that file doesn't exist, the system loads `data/default_departed_hotkeys.json` (this file)
3. **Persistence**: New departures are added to the runtime file automatically

## Regenerating the Default File

To regenerate this file with the complete historical elimination data from the database:

### Prerequisites

1. Database access (config-development.json must be configured)
2. taoshi-ts packages available in PYTHONPATH
3. Current working directory must be the repo root

### Steps

```bash
# Ensure you're in the repo root
cd /path/to/proprietary-trading-network

# Set up the PYTHONPATH for taoshi-ts modules
export PYTHONPATH=/path/to/taoshi-ts/taoshi-ts-database-ptn:/path/to/taoshi-ts/taoshi-ts-ptnhs:$PYTHONPATH

# Run the generation script (mainnet)
python3 generate_default_departed_hotkeys.py --netuid 8 --network finney

# Run the generation script (testnet)
python3 generate_default_departed_hotkeys.py --netuid 116 --network test
```

The script will:
1. Query the current metagraph to get all active hotkeys
2. Load ALL historical eliminations from the taoshi.ts database
3. Identify hotkeys that are eliminated but NOT in the current metagraph (departed)
4. Generate `data/default_departed_hotkeys.json` with the results

### Script Output

The script provides detailed output:
- Number of current hotkeys in metagraph
- Total elimination records from database
- Unique eliminated hotkeys
- Departed hotkeys identified
- Time ranges and elimination reasons

### When to Regenerate

Regenerate this file:
- Before major releases
- After significant time periods (quarterly/yearly)
- When deploying to new environments
- When the elimination tracking system is updated

## Notes

- This file should be committed to the repository
- The runtime file (`validation/departed_hotkeys.json`) should NOT be committed
- The system automatically syncs new departures to the runtime file
- Old departures remain in the departed hotkeys list permanently (no automatic cleanup)

## See Also

- `generate_default_departed_hotkeys.py` - Script to regenerate this file
- `vali_objects/utils/elimination_manager.py` - Departed hotkeys tracking implementation
- `tests/vali_tests/test_reregistration.py` - Tests for re-registration prevention
