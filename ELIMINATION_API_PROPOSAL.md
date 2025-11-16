# Elimination Manager API Refactoring Proposal

## Executive Summary

Refactor the `EliminationManager` to use getter/setter methods instead of direct dictionary access, similar to the recent metagraph wrapper refactoring. This will improve encapsulation, enable validation, simplify testing, and make future changes easier.

## Current State

### Direct Access Patterns Found

**Production Code**: 15 occurrences across 2 files
- `vali_objects/utils/elimination_manager.py`: 13 occurrences
- `runnable/daily_portfolio_returns.py`: 2 occurrences

**Test Code**: 18 occurrences across 5 files
- `test_elimination_core.py`: 7 occurrences
- `test_elimination_weight_calculation.py`: 4 occurrences
- `test_auto_sync.py`: 3 occurrences
- `test_elimination_persistence_recovery.py`: 3 occurrences
- `test_elimination_integration.py`: 1 occurrence

### Current Access Patterns

```python
# Direct dictionary access (current anti-pattern)
self.eliminations[hotkey] = elimination_data
elimination = self.eliminations.get(hotkey)
for elimination in self.eliminations.values():
    ...
if hotkey in self.eliminations:
    ...
self.eliminations.clear()
```

### Existing API (Partial)

The `EliminationManager` already has some getter methods:

```python
# Existing good patterns
is_hotkey_eliminated(hotkey)          # Fast O(1) check
hotkey_in_eliminations(hotkey)        # Returns full elimination dict
get_eliminated_hotkeys()              # Returns set of hotkeys
get_eliminations_from_memory()        # Returns list of all eliminations
append_elimination_row(...)           # Adds elimination with validation
delete_eliminations(hotkeys)          # Remove multiple eliminations
clear_eliminations()                  # Clear all
```

## Problems with Current Approach

1. **No Encapsulation**: Internal dict structure exposed throughout codebase
2. **No Validation**: Data can be added without validation
3. **Inconsistent Access**: Mix of direct access and method calls
4. **Hard to Change**: Can't easily modify underlying data structure
5. **Testing Complexity**: Tests need to know implementation details
6. **No Locking Control**: Direct access bypasses potential synchronization
7. **Type Safety**: No type hints on direct dictionary access

## Proposed API

### Core Getter Methods

```python
def get_elimination(self, hotkey: str) -> Optional[dict]:
    """
    Get elimination details for a hotkey.

    Args:
        hotkey: The hotkey to look up

    Returns:
        Elimination dict if found, None otherwise

    Example:
        elim = manager.get_elimination("miner_hotkey")
        if elim:
            print(f"Eliminated for: {elim['reason']}")
    """
    return deepcopy(self.eliminations.get(hotkey))

def is_eliminated(self, hotkey: str) -> bool:
    """
    Fast check if a hotkey is eliminated (O(1)).
    Alias for is_hotkey_eliminated() for consistency.

    Args:
        hotkey: The hotkey to check

    Returns:
        True if eliminated, False otherwise

    Example:
        if manager.is_eliminated("miner_hotkey"):
            print("Miner is eliminated")
    """
    return self.is_hotkey_eliminated(hotkey)

def get_all_eliminations(self) -> List[dict]:
    """
    Get all elimination records as a list.
    Alias for get_eliminations_from_memory() for consistency.

    Returns:
        List of elimination dicts

    Example:
        for elim in manager.get_all_eliminations():
            print(f"{elim['hotkey']}: {elim['reason']}")
    """
    return self.get_eliminations_from_memory()

def get_eliminated_hotkeys(self) -> set:
    """
    Get set of all eliminated hotkeys.
    Already exists - included for completeness.

    Returns:
        Set of eliminated hotkeys
    """
    return set(self.eliminations.keys()) if self.eliminations else set()

def iter_eliminations(self):
    """
    Iterate over elimination dicts.
    Provides safe iteration without exposing underlying dict.

    Yields:
        Elimination dicts

    Example:
        for elim in manager.iter_eliminations():
            process_elimination(elim)
    """
    for elimination in self.eliminations.values():
        yield deepcopy(elimination)

def get_elimination_count(self) -> int:
    """
    Get count of eliminations.

    Returns:
        Number of eliminated miners
    """
    return len(self.eliminations)
```

### Core Setter Methods

```python
def add_elimination(self, hotkey: str, elimination_data: dict) -> bool:
    """
    Add or update an elimination record.

    Args:
        hotkey: The hotkey to eliminate
        elimination_data: Elimination dict with required fields

    Returns:
        True if added, False if already exists

    Raises:
        ValueError: If elimination_data is invalid

    Example:
        manager.add_elimination("miner_hotkey", {
            'hotkey': "miner_hotkey",
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.12,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis()
        })
    """
    # Validate required fields
    required_fields = ['hotkey', 'reason', 'elimination_initiated_time_ms']
    for field in required_fields:
        if field not in elimination_data:
            raise ValueError(f"Missing required field: {field}")

    # Ensure hotkey matches
    if elimination_data['hotkey'] != hotkey:
        raise ValueError(f"Hotkey mismatch: {hotkey} != {elimination_data['hotkey']}")

    already_exists = hotkey in self.eliminations
    self.eliminations[hotkey] = elimination_data

    return not already_exists

def remove_elimination(self, hotkey: str) -> bool:
    """
    Remove a single elimination.

    Args:
        hotkey: The hotkey to remove

    Returns:
        True if removed, False if not found

    Example:
        if manager.remove_elimination("miner_hotkey"):
            print("Elimination removed")
    """
    if hotkey in self.eliminations:
        del self.eliminations[hotkey]
        return True
    return False

def clear_all_eliminations(self):
    """
    Clear all eliminations from memory and disk.
    Alias for clear_eliminations() for consistency.

    Example:
        manager.clear_all_eliminations()
    """
    self.clear_eliminations()

def update_elimination(self, hotkey: str, updates: dict) -> bool:
    """
    Update specific fields of an elimination.

    Args:
        hotkey: The hotkey to update
        updates: Dict of fields to update

    Returns:
        True if updated, False if not found

    Example:
        manager.update_elimination("miner_hotkey", {
            'dd': 0.15,  # Update drawdown
            'price_info': {...}  # Add price info
        })
    """
    if hotkey not in self.eliminations:
        return False

    self.eliminations[hotkey].update(updates)
    return True
```

### Enhanced Existing Methods

Keep these existing methods but enhance documentation:

```python
def append_elimination_row(self, hotkey, current_dd, reason, t_ms=None, price_info=None, return_info=None):
    """
    Add elimination with full validation and persistence.
    This is the recommended way to add eliminations.

    Generates elimination row with all required fields,
    adds to memory dict, and saves to disk.
    """
    # Existing implementation - already good!
    pass

def sync_eliminations(self, dat) -> list:
    """
    Sync eliminations from another validator (P2P sync).
    Replaces all eliminations with synced data.
    """
    # Existing implementation - already good!
    pass
```

## Migration Strategy

### Phase 1: Add New API Methods (Non-Breaking)

Add all new getter/setter methods to `EliminationManager` without removing any existing functionality.

**Files to Modify**:
- `vali_objects/utils/elimination_manager.py` - Add new methods

**Estimated Effort**: 2-3 hours

### Phase 2: Update Internal Implementation

Update internal code within `elimination_manager.py` to use new API.

**Patterns to Replace**:

```python
# OLD: Direct dict access
if hotkey in self.eliminations:
    elimination = self.eliminations[hotkey]

# NEW: Use getter
elimination = self.get_elimination(hotkey)
if elimination:
    ...

# OLD: Iterate over dict
for elimination in self.eliminations.values():
    process(elimination)

# NEW: Use iterator
for elimination in self.iter_eliminations():
    process(elimination)

# OLD: Manual dict assignment
self.eliminations[hotkey] = elimination_data

# NEW: Use setter
self.add_elimination(hotkey, elimination_data)
```

**Files to Modify**:
- `vali_objects/utils/elimination_manager.py` - 13 occurrences

**Estimated Effort**: 2-3 hours

### Phase 3: Update Production Code

Update other production code to use new API.

**Files to Modify**:
- `runnable/daily_portfolio_returns.py` - 2 occurrences

**Estimated Effort**: 30 minutes

### Phase 4: Update Test Code

Update all test files to use new API.

**Files to Modify**:
- `test_elimination_core.py` - 7 occurrences
- `test_elimination_weight_calculation.py` - 4 occurrences
- `test_elimination_persistence_recovery.py` - 3 occurrences
- `test_auto_sync.py` - 3 occurrences
- `test_elimination_integration.py` - 1 occurrence

**Estimated Effort**: 2-3 hours

### Phase 5: Add Property Deprecation (Optional)

Optionally make direct access raise warnings:

```python
@property
def eliminations(self):
    """
    DEPRECATED: Direct access to eliminations dict.
    Use get_elimination(), get_all_eliminations(), etc. instead.
    """
    import warnings
    warnings.warn(
        "Direct access to eliminations dict is deprecated. "
        "Use get_elimination() or get_all_eliminations() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self._eliminations

@eliminations.setter
def eliminations(self, value):
    import warnings
    warnings.warn(
        "Direct assignment to eliminations dict is deprecated. "
        "Use add_elimination() or sync_eliminations() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    self._eliminations = value
```

**Estimated Effort**: 1 hour

## Implementation Plan

### Step-by-Step Execution

**Week 1: Foundation**
- [ ] Day 1-2: Add new getter/setter methods to `EliminationManager`
- [ ] Day 2-3: Add unit tests for new methods
- [ ] Day 3-4: Update internal `elimination_manager.py` implementation
- [ ] Day 4-5: Update `runnable/daily_portfolio_returns.py`

**Week 2: Testing & Documentation**
- [ ] Day 1-3: Update all test files
- [ ] Day 3-4: Run full test suite and fix any issues
- [ ] Day 4-5: Update documentation and code review

**Optional Week 3: Deprecation**
- [ ] Add deprecation warnings
- [ ] Monitor for any remaining direct accesses

## Benefits

### 1. **Encapsulation**
```python
# Internal implementation can change without breaking callers
# Could switch from dict to OrderedDict, defaultdict, etc.
```

### 2. **Validation**
```python
# All additions validated for required fields
manager.add_elimination("hotkey", {
    'hotkey': "hotkey",
    'reason': "...",  # Required
    'elimination_initiated_time_ms': 123  # Required
})
# ValueError raised if fields missing
```

### 3. **Type Safety**
```python
# Clear return types in method signatures
def get_elimination(self, hotkey: str) -> Optional[dict]:
    """Returns elimination or None"""

def is_eliminated(self, hotkey: str) -> bool:
    """Returns True/False"""
```

### 4. **Thread Safety** (Future)
```python
# Can add locking in getters/setters
def get_elimination(self, hotkey: str) -> Optional[dict]:
    with self.eliminations_lock:
        return deepcopy(self.eliminations.get(hotkey))
```

### 5. **Testing**
```python
# Tests don't need to know about internal dict structure
# Can mock getters/setters easily
with patch.object(manager, 'get_elimination', return_value=mock_elim):
    # Test code that depends on eliminations
```

### 6. **Consistency**
```python
# All code uses same API
# Easier to maintain and understand
# Matches patterns from metagraph wrapper
```

### 7. **Future-Proof**
```python
# Can easily switch data structures
# Can add caching, metrics, logging
# Can migrate to database storage
```

## API Comparison

### Before (Current)

```python
# Check if eliminated
if hotkey in self.eliminations:
    # Get elimination
    elim = self.eliminations[hotkey]
    print(elim['reason'])

# Add elimination
self.eliminations[hotkey] = {
    'hotkey': hotkey,
    'reason': 'MAX_TOTAL_DRAWDOWN',
    'dd': 0.12,
    'elimination_initiated_time_ms': 123
}

# Iterate
for elim in self.eliminations.values():
    process(elim)

# Remove
if hotkey in self.eliminations:
    del self.eliminations[hotkey]

# Clear
self.eliminations.clear()
```

### After (Proposed)

```python
# Check if eliminated
if self.is_eliminated(hotkey):
    # Get elimination
    elim = self.get_elimination(hotkey)
    print(elim['reason'])

# Add elimination
self.add_elimination(hotkey, {
    'hotkey': hotkey,
    'reason': 'MAX_TOTAL_DRAWDOWN',
    'dd': 0.12,
    'elimination_initiated_time_ms': 123
})

# Iterate
for elim in self.iter_eliminations():
    process(elim)

# Remove
self.remove_elimination(hotkey)

# Clear
self.clear_all_eliminations()
```

## Testing Strategy

### New Unit Tests

```python
def test_get_elimination():
    """Test get_elimination returns correct data"""
    manager.add_elimination("test_hotkey", test_data)
    result = manager.get_elimination("test_hotkey")
    assert result == test_data
    assert result is not test_data  # Verify deepcopy

def test_get_elimination_not_found():
    """Test get_elimination returns None for missing hotkey"""
    assert manager.get_elimination("nonexistent") is None

def test_add_elimination_validation():
    """Test add_elimination validates required fields"""
    with pytest.raises(ValueError):
        manager.add_elimination("hotkey", {})  # Missing fields

def test_add_elimination_duplicate():
    """Test add_elimination handles duplicates"""
    manager.add_elimination("hotkey", valid_data)
    result = manager.add_elimination("hotkey", valid_data)
    assert result is False  # Already exists

def test_iter_eliminations():
    """Test iterator returns all eliminations"""
    manager.add_elimination("hk1", data1)
    manager.add_elimination("hk2", data2)
    eliminations = list(manager.iter_eliminations())
    assert len(eliminations) == 2

def test_remove_elimination():
    """Test remove_elimination removes correctly"""
    manager.add_elimination("hotkey", valid_data)
    assert manager.remove_elimination("hotkey") is True
    assert manager.get_elimination("hotkey") is None

def test_remove_elimination_not_found():
    """Test remove_elimination handles missing hotkey"""
    assert manager.remove_elimination("nonexistent") is False
```

### Integration Tests

```python
def test_elimination_flow_with_api():
    """Test complete elimination flow using only public API"""
    # Add elimination
    manager.append_elimination_row(hotkey, dd, reason)

    # Verify using getter
    assert manager.is_eliminated(hotkey)
    elim = manager.get_elimination(hotkey)
    assert elim['reason'] == reason

    # Process eliminations
    for elim in manager.iter_eliminations():
        assert 'hotkey' in elim
        assert 'reason' in elim

    # Remove elimination
    manager.remove_elimination(hotkey)
    assert not manager.is_eliminated(hotkey)
```

## Migration Checklist

- [ ] Create `ELIMINATION_API_PROPOSAL.md` (this document)
- [ ] Review proposal with team
- [ ] Add new getter/setter methods to `EliminationManager`
- [ ] Add unit tests for new methods
- [ ] Update `elimination_manager.py` internal code (13 occurrences)
- [ ] Update `runnable/daily_portfolio_returns.py` (2 occurrences)
- [ ] Update `test_elimination_core.py` (7 occurrences)
- [ ] Update `test_elimination_weight_calculation.py` (4 occurrences)
- [ ] Update `test_elimination_persistence_recovery.py` (3 occurrences)
- [ ] Update `test_auto_sync.py` (3 occurrences)
- [ ] Update `test_elimination_integration.py` (1 occurrence)
- [ ] Run full test suite
- [ ] Code review
- [ ] (Optional) Add deprecation warnings
- [ ] Update `CLAUDE.md` with new API patterns
- [ ] Close proposal and archive document

## Questions for Discussion

1. **Naming**: Should we use `add_elimination()` or `set_elimination()`?
   - `add_` implies it shouldn't exist yet
   - `set_` implies it can overwrite
   - Current: `append_elimination_row()` is the main way to add

2. **Validation Level**: How strict should validation be?
   - Strict: Raise errors for any invalid data
   - Lenient: Log warnings but allow
   - Current: `append_elimination_row()` does validation

3. **Backward Compatibility**: Should we keep direct access working?
   - Option A: Deprecation warnings only
   - Option B: Make it fail immediately
   - Option C: Keep it working indefinitely

4. **Scope**: Should we also refactor `departed_hotkeys` dict?
   - It has similar direct access patterns
   - Could use same getter/setter approach

## Conclusion

This refactoring will improve code quality, maintainability, and consistency across the codebase. The estimated total effort is 2-3 weeks of work, with the option to do it incrementally without breaking existing functionality.

The approach mirrors our successful metagraph wrapper refactoring, applying the same principles to the elimination manager for consistent API design across the codebase.

---

**Author**: Claude Code
**Date**: 2025-01-12
**Related**: METAGRAPH_WRAPPER_REFACTORING.md
