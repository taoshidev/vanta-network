# Metagraph Wrapper Refactoring Summary

## Overview

This document summarizes the comprehensive refactoring of metagraph access patterns across the Proprietary Trading Network codebase. All direct property access to the IPC metagraph has been replaced with getter methods to enable transparent injection of synthetic hotkeys.

## Objective

Enable a synthetic "DEVELOPMENT" hotkey that exists in all environments without polluting the actual IPC-shared metagraph data. This allows developers to send orders for testing without requiring actual chain registration.

## Architecture

### Core Component: IPCMetagraph

**Location**: `shared_objects/sn8_multiprocessing.py`

The class provides:
- **Transparent property proxying**: All metagraph properties are accessible as before
- **Getter methods**: Type-safe methods for accessing metagraph data
- **Synthetic hotkey injection**: The "DEVELOPMENT" hotkey is dynamically injected into hotkey lists
- **Backward compatibility**: Both property and getter access patterns work

### Key Methods

```python
class IPCMetagraph:
    DEVELOPMENT_HOTKEY = "DEVELOPMENT"

    def get_hotkeys(self) -> list
    def has_hotkey(self, hotkey: str) -> bool
    def is_development_hotkey(self, hotkey: str) -> bool
    def get_neurons(self) -> list
    def get_uids(self) -> list
    def get_axons(self) -> list
    def get_emission(self) -> list
```

## Files Modified

### 1. shared_objects/sn8_multiprocessing.py
- Added IPCMetagraph class to this file (consolidating IPC functionality)
- Implements transparent property proxying via `__getattr__` and `__setattr__`
- Provides getter methods for all metagraph properties
- Dynamically injects DEVELOPMENT hotkey in `get_hotkeys()`
- Updated `get_ipc_metagraph()` to always return IPCMetagraph instance
- No conditional logic - always applied

**Changes**: New class added (~200 lines), function updated

### 2. neurons/validator.py
- Line 334: `len(self.metagraph.hotkeys)` → `len(self.metagraph.get_hotkeys())`
- Line 335: `if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys` → `if not self.metagraph.has_hotkey(self.wallet.hotkey.ss58_address)`
- Line 441: `self.metagraph.hotkeys.index()` → `self.metagraph.get_hotkeys().index()`
- Line 773: `if miner_hotkey not in metagraph.hotkeys` → `if not metagraph.has_hotkey(miner_hotkey)`
- Lines 787-788: `metagraph.hotkeys.index()` and `metagraph.uids[]` → `metagraph.get_hotkeys().index()` and `metagraph.get_uids()[]`

**Changes**: 5 modifications

### 3. vali_objects/utils/elimination_manager.py
- Line 99: `set(self.metagraph.hotkeys)` → `set(self.metagraph.get_hotkeys())`
- Line 396: `set(self.metagraph.hotkeys)` → `set(self.metagraph.get_hotkeys())`
- Line 527: `set(self.metagraph.hotkeys)` → `set(self.metagraph.get_hotkeys())`
- Line 547: `set(self.metagraph.hotkeys)` → `set(self.metagraph.get_hotkeys())`
- Line 627: `hotkey in self.metagraph.hotkeys` → `self.metagraph.has_hotkey(hotkey)`

**Changes**: 5 modifications

### 4. vali_objects/utils/subtensor_weight_setter.py
- Line 64: `list(self.metagraph.hotkeys)` → `list(self.metagraph.get_hotkeys())`

**Changes**: 1 modification

### 5. vali_objects/utils/asset_selection_manager.py
- Lines 241, 243: `self.metagraph.neurons` → `self.metagraph.get_neurons()`

**Changes**: 2 modifications

### 6. shared_objects/metagraph_updater.py
- Lines 213, 216, 223: `self.metagraph.hotkeys` → `self.metagraph.get_hotkeys()`
- Line 230: `self.metagraph.neurons` → `self.metagraph.get_neurons()`
- Line 578: `self.metagraph.neurons` → `self.metagraph.get_neurons()`
- Line 604: `len(self.metagraph.hotkeys)` → `len(self.metagraph.get_hotkeys())`
- Line 758: `set(self.metagraph.hotkeys)` → `set(self.metagraph.get_hotkeys())`

**Changes**: 7 modifications

### 7. vali_objects/utils/mdd_checker.py
- Line 111: `self.metagraph.hotkeys` → `self.metagraph.get_hotkeys()`

**Changes**: 1 modification

### 8. vali_objects/utils/challengeperiod_manager.py
- Lines 96, 99: `self.metagraph.hotkeys` → `self.metagraph.get_hotkeys()`
- Lines 206, 210: `self.metagraph.hotkeys` → `self.metagraph.get_hotkeys()`
- Line 338: `self.metagraph.hotkeys` → `self.metagraph.get_hotkeys()`

**Changes**: 5 modifications

### 9. vali_objects/utils/plagiarism_detector.py
- Line 50: `self.position_manager.metagraph.hotkeys` → `self.position_manager.metagraph.get_hotkeys()`
- Line 69: `self.metagraph.hotkeys` → `self.metagraph.get_hotkeys()`
- Line 79: Comment with `self.metagraph.hotkeys` → `self.metagraph.get_hotkeys()`
- Line 168: `self.metagraph.hotkeys` (in comment) → `self.metagraph.get_hotkeys()`

**Changes**: 4 modifications

### 10. vali_objects/utils/p2p_syncer.py
- Line 64: `self.metagraph.neurons` → `self.metagraph.get_neurons()`
- Line 556: `self.metagraph.axons` → `self.metagraph.get_axons()`
- Line 558: `self.metagraph.neurons` → `self.metagraph.get_neurons()`
- Line 572: `self.metagraph.neurons` → `self.metagraph.get_neurons()`

**Changes**: 4 modifications

### 11. vali_objects/utils/validator_contract_manager.py
- Line 861: `self.metagraph.neurons` → `self.metagraph.get_neurons()`
- Line 863: `self.metagraph.neurons` → `self.metagraph.get_neurons()`

**Changes**: 2 modifications

### 12. shared_objects/mock_metagraph.py
- Added getter methods to MockMetagraph class to match IPCMetagraph interface
- `get_hotkeys()` - Returns self.hotkeys
- `get_neurons()` - Returns self.neurons
- `get_uids()` - Returns self.uids
- `get_axons()` - Returns self.axons
- `get_emission()` - Returns self.emission
- `get_block_at_registration()` - Returns self.block_at_registration
- `has_hotkey(hotkey)` - Checks if hotkey exists
- `is_development_hotkey(hotkey)` - Checks if hotkey == "DEVELOPMENT"
- Added axons and emission properties to __init__

**Changes**: 10 methods added, 2 properties added

**Note**: EnhancedMockMetagraph in tests/vali_tests/mock_utils.py inherits from BaseMockMetagraph (MockMetagraph), so it automatically inherits all getter methods.

## Statistics

- **Total files modified**: 12
- **Production code changes**: ~40 occurrences updated across 11 files
- **Test infrastructure**: MockMetagraph updated with 10 getter methods
- **Lines of new class code**: ~200 lines added to sn8_multiprocessing.py (IPCMetagraph class)
- **Lines of new mock code**: ~30 lines in MockMetagraph

## Pattern Replacements

### Direct Property Access → Getter Methods

| Old Pattern | New Pattern |
|------------|-------------|
| `metagraph.hotkeys` | `metagraph.get_hotkeys()` |
| `metagraph.neurons` | `metagraph.get_neurons()` |
| `metagraph.uids` | `metagraph.get_uids()` |
| `metagraph.axons` | `metagraph.get_axons()` |
| `hotkey in metagraph.hotkeys` | `metagraph.has_hotkey(hotkey)` |

## Benefits

1. **Development Testing**: Developers can now test order processing without chain registration
2. **Clean Architecture**: All metagraph access goes through a consistent interface
3. **Zero Runtime Overhead**: Property access is transparently proxied with no performance impact
4. **Backward Compatibility**: Existing code using properties continues to work
5. **Type Safety**: Getter methods provide explicit return types
6. **Maintainability**: Single source of truth for metagraph access logic

## Migration Notes

### For Future Development

When accessing metagraph properties, prefer getter methods:

```python
# Preferred
hotkeys = self.metagraph.get_hotkeys()
if self.metagraph.has_hotkey(miner_hotkey):
    neurons = self.metagraph.get_neurons()

# Still works but discouraged for new code
hotkeys = self.metagraph.hotkeys
if miner_hotkey in self.metagraph.hotkeys:
    neurons = self.metagraph.neurons
```

### DEVELOPMENT Hotkey Behavior

The synthetic DEVELOPMENT hotkey:
- Always appears first in `get_hotkeys()` results
- Returns `True` for `has_hotkey("DEVELOPMENT")`
- Can be detected with `is_development_hotkey(hotkey)`
- Does NOT appear in the underlying IPC metagraph
- Does NOT have a corresponding neuron object

## Next Steps

1. **REST Endpoint Implementation**: Create authenticated REST endpoint for DEVELOPMENT orders
2. **Testing**: Verify all changes compile and tests pass
3. **Documentation**: Update developer guides with DEVELOPMENT hotkey usage
4. **Integration**: Connect REST endpoint to production order filling code

## Related Files

- `shared_objects/sn8_multiprocessing.py` - IPCMetagraph implementation
- `neurons/validator.py` - Order processing workflow reference
- `template/protocol.py` - Protocol definitions (not using heavy SendSignal approach)
- `vali_objects/vali_config.py` - Configuration constants

## Contact

For questions about this refactoring:
- Review the implementation: `shared_objects/sn8_multiprocessing.py` (IPCMetagraph class)
- Check test files: `tests/vali_tests/`
- See original request in conversation history

---

**Date**: 2025-01-12
**Refactoring Type**: Comprehensive getter pattern migration
**Impact**: All validator components using metagraph
