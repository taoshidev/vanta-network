# ChallengePeriodManager API Refactoring Proposal

## Executive Summary

Refactor the `ChallengePeriodManager` to use getter/setter methods instead of direct dictionary access, following the successful pattern from the recent `EliminationManager` refactoring. This will improve encapsulation, enable validation, simplify testing, and make future changes easier.

## Current State

### Data Structures

The `ChallengePeriodManager` maintains two core dictionaries that are directly accessed throughout the codebase:

**1. `active_miners` dict**
```python
# Structure: dict[str, tuple[MinerBucket, int, MinerBucket|None, int|None]]
# Example: {
#     "hotkey_abc": (MinerBucket.CHALLENGE, 1700000000000, None, None),
#     "hotkey_xyz": (MinerBucket.PLAGIARISM, 1700000000000, MinerBucket.MAINCOMP, 1699000000000)
# }

# Tuple fields:
# [0] - Current bucket (CHALLENGE, MAINCOMP, PROBATION, PLAGIARISM)
# [1] - Bucket start time in milliseconds
# [2] - Previous bucket (used for plagiarism demotions, None otherwise)
# [3] - Previous bucket start time (used for plagiarism demotions, None otherwise)
```

**2. `eliminations_with_reasons` dict**
```python
# Structure: dict[str, tuple[str, float]]
# Example: {
#     "hotkey_abc": ("FAILED_CHALLENGE_PERIOD_TIME", -1),
#     "hotkey_xyz": ("FAILED_CHALLENGE_PERIOD_DRAWDOWN", 0.12)
# }

# Tuple fields:
# [0] - Elimination reason (EliminationReason enum value)
# [1] - Drawdown percentage (or -1 if not applicable)
```

### Direct Access Patterns Found

**Production Code**: 46+ occurrences across 3 files
- `vali_objects/utils/challengeperiod_manager.py`: 35 occurrences (`active_miners`), 6 occurrences (`eliminations_with_reasons`)
- `vali_objects/utils/elimination_manager.py`: 2 occurrences (`active_miners`), 5 occurrences (`eliminations_with_reasons`)
- `vali_objects/utils/position_manager.py`: 4 occurrences (`active_miners`)

**Test Code**: 7+ files
- `test_elimination_manager.py`
- `test_elimination_core.py`
- `test_elimination_weight_calculation.py`
- `test_auto_sync.py`
- `test_probation_comprehensive.py`
- `test_plagiarism.py`
- `test_challengeperiod_integration.py`

### Current Access Patterns

```python
# Direct dictionary access (current anti-pattern)

# active_miners
self.active_miners[hotkey] = (MinerBucket.CHALLENGE, start_time, None, None)
bucket = self.active_miners[hotkey][0]
start_time = self.active_miners[hotkey][1]
if hotkey in self.active_miners:
    del self.active_miners[hotkey]
self.active_miners.clear()
self.active_miners.update(synced_data)
for hotkey, (bucket, start_time, prev_bucket, prev_time) in self.active_miners.items():
    ...

# eliminations_with_reasons
self.eliminations_with_reasons[hotkey] = (reason, drawdown)
self.eliminations_with_reasons.clear()
self.eliminations_with_reasons.update(new_eliminations)
```

### Existing API (Partial)

The `ChallengePeriodManager` already has some getter methods:

```python
# Existing good patterns
get_miner_bucket(hotkey)          # Returns bucket enum or None
get_testing_miners()              # Returns dict of CHALLENGE miners {hotkey: start_time}
get_success_miners()              # Returns dict of MAINCOMP miners {hotkey: start_time}
get_probation_miners()            # Returns dict of PROBATION miners {hotkey: start_time}
get_plagiarism_miners()           # Returns dict of PLAGIARISM miners {hotkey: start_time}
get_hotkeys_by_bucket(bucket)     # Returns list of hotkeys in specific bucket
```

## Problems with Current Approach

1. **No Encapsulation**: Internal dict structure with tuples exposed throughout codebase
2. **No Validation**: Data can be added without validating tuple structure
3. **Inconsistent Access**: Mix of direct access and method calls
4. **Hard to Change**: Can't easily modify tuple structure or add new fields
5. **Testing Complexity**: Tests need to know exact tuple field positions
6. **No Locking Control**: Direct access bypasses potential synchronization (especially critical for IPC dicts)
7. **Type Safety**: No type hints on direct dictionary access
8. **Fragile Tuple Access**: Index-based access (e.g., `[0]`, `[1]`) is error-prone and hard to maintain
9. **IPC Dict Sharing**: Recent bug where mock IPC manager returned same dict instance for multiple dicts

## Proposed API

### Core Getter Methods - `active_miners`

```python
def get_miner_info(self, hotkey: str) -> Optional[tuple[MinerBucket, int, Optional[MinerBucket], Optional[int]]]:
    """
    Get complete miner information tuple.

    Args:
        hotkey: The miner hotkey to look up

    Returns:
        Tuple of (bucket, start_time, prev_bucket, prev_time) if found, None otherwise

    Example:
        info = manager.get_miner_info("miner_hotkey")
        if info:
            bucket, start_time, prev_bucket, prev_time = info
            print(f"Miner in {bucket.value} since {start_time}")
    """
    info = self.active_miners.get(hotkey)
    return copy.deepcopy(info) if info else None

def get_miner_start_time(self, hotkey: str) -> Optional[int]:
    """
    Get the start time of a miner's current bucket.

    Args:
        hotkey: The miner hotkey to look up

    Returns:
        Start time in milliseconds, or None if not found

    Example:
        start_time = manager.get_miner_start_time("miner_hotkey")
        if start_time:
            print(f"Started at {start_time}")
    """
    info = self.active_miners.get(hotkey)
    return info[1] if info else None

def get_miner_previous_bucket(self, hotkey: str) -> Optional[MinerBucket]:
    """
    Get the previous bucket of a miner (used for plagiarism demotions).

    Args:
        hotkey: The miner hotkey to look up

    Returns:
        Previous bucket enum, or None if not found or not set

    Example:
        prev_bucket = manager.get_miner_previous_bucket("miner_hotkey")
        if prev_bucket:
            print(f"Was previously in {prev_bucket.value}")
    """
    info = self.active_miners.get(hotkey)
    return info[2] if info else None

def get_miner_previous_time(self, hotkey: str) -> Optional[int]:
    """
    Get the start time of a miner's previous bucket.

    Args:
        hotkey: The miner hotkey to look up

    Returns:
        Previous bucket start time in milliseconds, or None if not found or not set
    """
    info = self.active_miners.get(hotkey)
    return info[3] if info else None

def get_all_active_miners(self) -> dict[str, tuple]:
    """
    Get all active miners as a dictionary.

    Returns:
        Dict mapping hotkeys to (bucket, start_time, prev_bucket, prev_time) tuples

    Example:
        all_miners = manager.get_all_active_miners()
        for hotkey, (bucket, start_time, _, _) in all_miners.items():
            print(f"{hotkey}: {bucket.value}")
    """
    return copy.deepcopy(dict(self.active_miners))

def has_miner(self, hotkey: str) -> bool:
    """
    Fast check if a miner is in active_miners (O(1)).

    Args:
        hotkey: The miner hotkey to check

    Returns:
        True if miner is active, False otherwise

    Example:
        if manager.has_miner("miner_hotkey"):
            print("Miner is active")
    """
    return hotkey in self.active_miners

def iter_active_miners(self):
    """
    Iterate over active miners.
    Provides safe iteration without exposing underlying dict.

    Yields:
        Tuples of (hotkey, bucket, start_time, prev_bucket, prev_time)

    Example:
        for hotkey, bucket, start_time, prev_bucket, prev_time in manager.iter_active_miners():
            process_miner(hotkey, bucket, start_time)
    """
    for hotkey, (bucket, start_time, prev_bucket, prev_time) in self.active_miners.items():
        yield hotkey, bucket, start_time, prev_bucket, prev_time

def get_active_miner_count(self) -> int:
    """
    Get count of active miners.

    Returns:
        Number of active miners across all buckets
    """
    return len(self.active_miners)
```

### Core Setter Methods - `active_miners`

```python
def set_miner_bucket(
    self,
    hotkey: str,
    bucket: MinerBucket,
    start_time: int,
    prev_bucket: Optional[MinerBucket] = None,
    prev_time: Optional[int] = None
) -> bool:
    """
    Set or update a miner's bucket information.

    Args:
        hotkey: The miner hotkey
        bucket: The current bucket
        start_time: Bucket start time in milliseconds
        prev_bucket: Previous bucket (for plagiarism demotions)
        prev_time: Previous bucket start time (for plagiarism demotions)

    Returns:
        True if this is a new miner, False if updating existing

    Raises:
        ValueError: If bucket is not a MinerBucket enum
        ValueError: If start_time is negative

    Example:
        # Add new miner to challenge period
        manager.set_miner_bucket("miner_hotkey", MinerBucket.CHALLENGE, current_time)

        # Demote to plagiarism, preserving previous state
        manager.set_miner_bucket(
            "miner_hotkey",
            MinerBucket.PLAGIARISM,
            current_time,
            prev_bucket=MinerBucket.MAINCOMP,
            prev_time=previous_time
        )
    """
    # Validation
    if not isinstance(bucket, MinerBucket):
        raise ValueError(f"bucket must be a MinerBucket enum, got {type(bucket)}")
    if start_time < 0:
        raise ValueError(f"start_time cannot be negative: {start_time}")
    if prev_bucket is not None and not isinstance(prev_bucket, MinerBucket):
        raise ValueError(f"prev_bucket must be a MinerBucket enum or None, got {type(prev_bucket)}")
    if prev_time is not None and prev_time < 0:
        raise ValueError(f"prev_time cannot be negative: {prev_time}")

    is_new = hotkey not in self.active_miners
    self.active_miners[hotkey] = (bucket, start_time, prev_bucket, prev_time)
    return is_new

def remove_miner(self, hotkey: str) -> bool:
    """
    Remove a miner from active_miners.

    Args:
        hotkey: The miner hotkey to remove

    Returns:
        True if removed, False if not found

    Example:
        if manager.remove_miner("miner_hotkey"):
            print("Miner removed")
    """
    if hotkey in self.active_miners:
        del self.active_miners[hotkey]
        return True
    return False

def clear_all_miners(self):
    """
    Clear all miners from active_miners.

    Example:
        manager.clear_all_miners()
    """
    self.active_miners.clear()

def update_miners(self, miners_dict: dict) -> int:
    """
    Bulk update active_miners from a dict.
    Used for syncing from another validator.

    Args:
        miners_dict: Dict mapping hotkeys to (bucket, start_time, prev_bucket, prev_time) tuples

    Returns:
        Number of miners updated

    Example:
        synced_data = {...}
        count = manager.update_miners(synced_data)
        print(f"Updated {count} miners")
    """
    count = len(miners_dict)
    self.active_miners.update(miners_dict)
    return count
```

### Core Getter Methods - `eliminations_with_reasons`

```python
def get_elimination_reason(self, hotkey: str) -> Optional[tuple[str, float]]:
    """
    Get elimination reason and drawdown for a hotkey.

    Args:
        hotkey: The miner hotkey to look up

    Returns:
        Tuple of (reason, drawdown) if found, None otherwise

    Example:
        result = manager.get_elimination_reason("miner_hotkey")
        if result:
            reason, drawdown = result
            print(f"Eliminated for {reason} with drawdown {drawdown}")
    """
    with self.eliminations_lock if self.eliminations_lock else contextlib.nullcontext():
        result = self.eliminations_with_reasons.get(hotkey)
        return copy.deepcopy(result) if result else None

def get_all_elimination_reasons(self) -> dict[str, tuple[str, float]]:
    """
    Get all elimination reasons.

    Returns:
        Dict mapping hotkeys to (reason, drawdown) tuples

    Example:
        all_reasons = manager.get_all_elimination_reasons()
        for hotkey, (reason, dd) in all_reasons.items():
            print(f"{hotkey}: {reason} (dd={dd})")
    """
    with self.eliminations_lock if self.eliminations_lock else contextlib.nullcontext():
        return copy.deepcopy(dict(self.eliminations_with_reasons))

def has_elimination_reason(self, hotkey: str) -> bool:
    """
    Fast check if a hotkey has an elimination reason (O(1)).

    Args:
        hotkey: The miner hotkey to check

    Returns:
        True if elimination reason exists, False otherwise
    """
    with self.eliminations_lock if self.eliminations_lock else contextlib.nullcontext():
        return hotkey in self.eliminations_with_reasons

def get_elimination_reason_count(self) -> int:
    """
    Get count of elimination reasons.

    Returns:
        Number of miners with elimination reasons
    """
    with self.eliminations_lock if self.eliminations_lock else contextlib.nullcontext():
        return len(self.eliminations_with_reasons)
```

### Core Setter Methods - `eliminations_with_reasons`

```python
def add_elimination_reason(self, hotkey: str, reason: str, drawdown: float) -> bool:
    """
    Add an elimination reason for a miner.

    Args:
        hotkey: The miner hotkey
        reason: The elimination reason (EliminationReason enum value)
        drawdown: The drawdown percentage (or -1 if not applicable)

    Returns:
        True if added, False if already exists

    Raises:
        ValueError: If reason is empty
        ValueError: If drawdown is invalid (< -1 or > 1)

    Example:
        manager.add_elimination_reason(
            "miner_hotkey",
            EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value,
            0.12
        )
    """
    # Validation
    if not reason:
        raise ValueError("Elimination reason cannot be empty")
    if drawdown < -1 or drawdown > 1:
        raise ValueError(f"Drawdown must be between -1 and 1, got {drawdown}")

    with self.eliminations_lock if self.eliminations_lock else contextlib.nullcontext():
        already_exists = hotkey in self.eliminations_with_reasons
        self.eliminations_with_reasons[hotkey] = (reason, drawdown)
        return not already_exists

def clear_elimination_reasons(self):
    """
    Clear all elimination reasons.

    Example:
        manager.clear_elimination_reasons()
    """
    with self.eliminations_lock if self.eliminations_lock else contextlib.nullcontext():
        self.eliminations_with_reasons.clear()

def update_elimination_reasons(self, reasons_dict: dict[str, tuple[str, float]]) -> int:
    """
    Bulk update elimination reasons from a dict.
    Replaces all existing elimination reasons.

    Args:
        reasons_dict: Dict mapping hotkeys to (reason, drawdown) tuples

    Returns:
        Number of elimination reasons set

    Example:
        new_reasons = {
            "miner1": ("FAILED_CHALLENGE_PERIOD_TIME", -1),
            "miner2": ("FAILED_CHALLENGE_PERIOD_DRAWDOWN", 0.12)
        }
        count = manager.update_elimination_reasons(new_reasons)
    """
    with self.eliminations_lock if self.eliminations_lock else contextlib.nullcontext():
        self.eliminations_with_reasons.clear()
        self.eliminations_with_reasons.update(reasons_dict)
        return len(reasons_dict)
```

### Enhanced Existing Methods

Keep these existing methods but enhance documentation:

```python
def get_miner_bucket(self, hotkey: str) -> Optional[MinerBucket]:
    """
    Get the current bucket for a miner.
    EXISTING METHOD - Keep as-is for backward compatibility.

    Returns:
        MinerBucket enum or None if not found
    """
    return self.active_miners.get(hotkey, [None])[0]

def get_testing_miners(self) -> dict[str, int]:
    """
    Get all miners in CHALLENGE bucket.
    EXISTING METHOD - Already uses good pattern with deepcopy.

    Returns:
        Dict mapping hotkeys to start times
    """
    return copy.deepcopy(self._bucket_view(MinerBucket.CHALLENGE))

def get_success_miners(self) -> dict[str, int]:
    """Get all miners in MAINCOMP bucket"""
    return copy.deepcopy(self._bucket_view(MinerBucket.MAINCOMP))

def get_probation_miners(self) -> dict[str, int]:
    """Get all miners in PROBATION bucket"""
    return copy.deepcopy(self._bucket_view(MinerBucket.PROBATION))

def get_plagiarism_miners(self) -> dict[str, int]:
    """Get all miners in PLAGIARISM bucket"""
    return copy.deepcopy(self._bucket_view(MinerBucket.PLAGIARISM))

def get_hotkeys_by_bucket(self, bucket: MinerBucket) -> list[str]:
    """
    Get list of hotkeys in a specific bucket.
    EXISTING METHOD - Keep as-is.
    """
    return [hotkey for hotkey, (b, _, _, _) in self.active_miners.items() if b == bucket]
```

## Migration Strategy

### Phase 1: Add New API Methods (Non-Breaking)

Add all new getter/setter methods to `ChallengePeriodManager` without removing any existing functionality.

**Files to Modify**:
- `vali_objects/utils/challengeperiod_manager.py` - Add new methods

**Estimated Effort**: 3-4 hours

### Phase 2: Update Internal Implementation

Update internal code within `challengeperiod_manager.py` to use new API.

**Patterns to Replace**:

```python
# OLD: Direct tuple access
if hotkey in self.active_miners:
    bucket = self.active_miners[hotkey][0]
    start_time = self.active_miners[hotkey][1]

# NEW: Use getters
bucket = self.get_miner_bucket(hotkey)
start_time = self.get_miner_start_time(hotkey)
if bucket and start_time:
    ...

# OLD: Direct dict assignment
self.active_miners[hotkey] = (MinerBucket.CHALLENGE, start_time, None, None)

# NEW: Use setter
self.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, start_time)

# OLD: Direct iteration
for hotkey, (bucket, start_time, prev_bucket, prev_time) in self.active_miners.items():
    process(hotkey, bucket, start_time)

# NEW: Use iterator
for hotkey, bucket, start_time, prev_bucket, prev_time in self.iter_active_miners():
    process(hotkey, bucket, start_time)

# OLD: Direct elimination reasons update
self.eliminations_with_reasons.clear()
self.eliminations_with_reasons.update(new_reasons)

# NEW: Use setter
self.update_elimination_reasons(new_reasons)
```

**Files to Modify**:
- `vali_objects/utils/challengeperiod_manager.py` - 35+ occurrences (`active_miners`), 6 occurrences (`eliminations_with_reasons`)

**Estimated Effort**: 4-6 hours

### Phase 3: Update Production Code

Update other production code to use new API.

**Files to Modify**:
- `vali_objects/utils/elimination_manager.py` - 2 occurrences (`active_miners`), 5 occurrences (`eliminations_with_reasons`)
- `vali_objects/utils/position_manager.py` - 4 occurrences (`active_miners`)

**Estimated Effort**: 1-2 hours

### Phase 4: Update Test Code

Update all test files to use new API.

**Files to Modify**:
- `test_elimination_manager.py`
- `test_elimination_core.py`
- `test_elimination_weight_calculation.py`
- `test_auto_sync.py`
- `test_probation_comprehensive.py`
- `test_plagiarism.py`
- `test_challengeperiod_integration.py`

**Estimated Effort**: 3-4 hours

### Phase 5: Add Property Deprecation (Optional)

Optionally make direct access raise warnings:

```python
@property
def active_miners(self):
    """
    DEPRECATED: Direct access to active_miners dict.
    Use get_miner_info(), get_all_active_miners(), etc. instead.
    """
    import warnings
    warnings.warn(
        "Direct access to active_miners dict is deprecated. "
        "Use get_miner_info() or get_all_active_miners() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self._active_miners

@active_miners.setter
def active_miners(self, value):
    import warnings
    warnings.warn(
        "Direct assignment to active_miners dict is deprecated. "
        "Use set_miner_bucket() or update_miners() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    self._active_miners = value
```

**Estimated Effort**: 1-2 hours

## Implementation Plan

### Step-by-Step Execution

**Week 1: Foundation**
- [ ] Day 1-2: Add new getter/setter methods to `ChallengePeriodManager`
- [ ] Day 2-3: Add unit tests for new methods
- [ ] Day 3-4: Update internal `challengeperiod_manager.py` implementation
- [ ] Day 4-5: Update `elimination_manager.py` and `position_manager.py`

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
# Could switch tuple structure, add new fields, change ordering
```

### 2. **Validation**
```python
# All additions validated for correct types
manager.set_miner_bucket("hotkey", MinerBucket.CHALLENGE, start_time)
# ValueError raised if bucket is not MinerBucket enum
# ValueError raised if start_time is negative
```

### 3. **Type Safety**
```python
# Clear return types in method signatures
def get_miner_bucket(self, hotkey: str) -> Optional[MinerBucket]:
    """Returns bucket enum or None"""

def has_miner(self, hotkey: str) -> bool:
    """Returns True/False"""
```

### 4. **Thread Safety**
```python
# Already using locks for eliminations_with_reasons
# Can extend to active_miners if needed
def get_elimination_reason(self, hotkey: str) -> Optional[tuple[str, float]]:
    with self.eliminations_lock:
        return deepcopy(self.eliminations_with_reasons.get(hotkey))
```

### 5. **Testing**
```python
# Tests don't need to know about internal tuple structure
# Can mock getters/setters easily
with patch.object(manager, 'get_miner_bucket', return_value=MinerBucket.CHALLENGE):
    # Test code that depends on miner bucket
```

### 6. **Consistency**
```python
# All code uses same API
# Easier to maintain and understand
# Matches patterns from EliminationManager refactoring
```

### 7. **Future-Proof**
```python
# Can easily change tuple structure
# Can add new fields without breaking existing code
# Can add caching, metrics, logging
# Can migrate to database storage
```

### 8. **Safer Tuple Access**
```python
# OLD: Fragile index-based access
bucket = self.active_miners[hotkey][0]  # What is [0]? Easy to forget
start_time = self.active_miners[hotkey][1]  # What is [1]?

# NEW: Named methods make intent clear
bucket = self.get_miner_bucket(hotkey)
start_time = self.get_miner_start_time(hotkey)
```

## API Comparison

### Before (Current)

```python
# Check if miner exists
if hotkey in self.active_miners:
    # Get bucket and start time
    bucket = self.active_miners[hotkey][0]
    start_time = self.active_miners[hotkey][1]
    prev_bucket = self.active_miners[hotkey][2]
    prev_time = self.active_miners[hotkey][3]

# Add miner to challenge period
self.active_miners[hotkey] = (MinerBucket.CHALLENGE, current_time, None, None)

# Demote to plagiarism
self.active_miners[hotkey] = (MinerBucket.PLAGIARISM, current_time, prev_bucket, prev_time)

# Iterate
for hotkey, (bucket, start_time, prev_bucket, prev_time) in self.active_miners.items():
    process(hotkey, bucket)

# Remove
if hotkey in self.active_miners:
    del self.active_miners[hotkey]

# Clear
self.active_miners.clear()

# Update elimination reasons
self.eliminations_with_reasons.clear()
self.eliminations_with_reasons.update(new_reasons)
```

### After (Proposed)

```python
# Check if miner exists
if self.has_miner(hotkey):
    # Get bucket and start time
    bucket = self.get_miner_bucket(hotkey)
    start_time = self.get_miner_start_time(hotkey)
    prev_bucket = self.get_miner_previous_bucket(hotkey)
    prev_time = self.get_miner_previous_time(hotkey)

# Add miner to challenge period
self.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, current_time)

# Demote to plagiarism
self.set_miner_bucket(
    hotkey,
    MinerBucket.PLAGIARISM,
    current_time,
    prev_bucket=prev_bucket,
    prev_time=prev_time
)

# Iterate
for hotkey, bucket, start_time, prev_bucket, prev_time in self.iter_active_miners():
    process(hotkey, bucket)

# Remove
self.remove_miner(hotkey)

# Clear
self.clear_all_miners()

# Update elimination reasons
self.update_elimination_reasons(new_reasons)
```

## Testing Strategy

### New Unit Tests

```python
def test_get_miner_info():
    """Test get_miner_info returns correct data"""
    manager.set_miner_bucket("test_hotkey", MinerBucket.CHALLENGE, 1000)
    result = manager.get_miner_info("test_hotkey")
    assert result == (MinerBucket.CHALLENGE, 1000, None, None)
    assert result is not manager.active_miners["test_hotkey"]  # Verify deepcopy

def test_get_miner_info_not_found():
    """Test get_miner_info returns None for missing hotkey"""
    assert manager.get_miner_info("nonexistent") is None

def test_set_miner_bucket_validation():
    """Test set_miner_bucket validates inputs"""
    with pytest.raises(ValueError):
        manager.set_miner_bucket("hotkey", "INVALID", 1000)  # Not enum
    with pytest.raises(ValueError):
        manager.set_miner_bucket("hotkey", MinerBucket.CHALLENGE, -1)  # Negative time

def test_set_miner_bucket_new_vs_update():
    """Test set_miner_bucket returns correct boolean"""
    assert manager.set_miner_bucket("hotkey", MinerBucket.CHALLENGE, 1000) is True  # New
    assert manager.set_miner_bucket("hotkey", MinerBucket.MAINCOMP, 2000) is False  # Update

def test_iter_active_miners():
    """Test iterator returns all miners"""
    manager.set_miner_bucket("hk1", MinerBucket.CHALLENGE, 1000)
    manager.set_miner_bucket("hk2", MinerBucket.MAINCOMP, 2000)
    miners = list(manager.iter_active_miners())
    assert len(miners) == 2

def test_remove_miner():
    """Test remove_miner removes correctly"""
    manager.set_miner_bucket("hotkey", MinerBucket.CHALLENGE, 1000)
    assert manager.remove_miner("hotkey") is True
    assert manager.get_miner_info("hotkey") is None

def test_remove_miner_not_found():
    """Test remove_miner handles missing hotkey"""
    assert manager.remove_miner("nonexistent") is False

def test_add_elimination_reason():
    """Test add_elimination_reason adds correctly"""
    manager.add_elimination_reason("hotkey", "FAILED_CHALLENGE_PERIOD_TIME", -1)
    reason, dd = manager.get_elimination_reason("hotkey")
    assert reason == "FAILED_CHALLENGE_PERIOD_TIME"
    assert dd == -1

def test_add_elimination_reason_validation():
    """Test add_elimination_reason validates inputs"""
    with pytest.raises(ValueError):
        manager.add_elimination_reason("hotkey", "", -1)  # Empty reason
    with pytest.raises(ValueError):
        manager.add_elimination_reason("hotkey", "REASON", 2.0)  # Invalid drawdown

def test_elimination_reasons_thread_safety():
    """Test elimination reasons use locks correctly"""
    # Verify that lock is used when accessing elimination reasons
    # This ensures thread safety for IPC dict access
    pass
```

### Integration Tests

```python
def test_challengeperiod_flow_with_api():
    """Test complete challenge period flow using only public API"""
    # Add miner to challenge period
    manager.set_miner_bucket("miner1", MinerBucket.CHALLENGE, current_time)

    # Verify using getter
    assert manager.has_miner("miner1")
    assert manager.get_miner_bucket("miner1") == MinerBucket.CHALLENGE

    # Promote to maincomp
    manager.set_miner_bucket("miner1", MinerBucket.MAINCOMP, current_time)
    assert manager.get_miner_bucket("miner1") == MinerBucket.MAINCOMP

    # Demote to plagiarism, preserving state
    prev_bucket = manager.get_miner_bucket("miner1")
    prev_time = manager.get_miner_start_time("miner1")
    manager.set_miner_bucket(
        "miner1",
        MinerBucket.PLAGIARISM,
        current_time,
        prev_bucket=prev_bucket,
        prev_time=prev_time
    )

    # Verify plagiarism state
    assert manager.get_miner_bucket("miner1") == MinerBucket.PLAGIARISM
    assert manager.get_miner_previous_bucket("miner1") == prev_bucket
    assert manager.get_miner_previous_time("miner1") == prev_time

    # Add elimination reason
    manager.add_elimination_reason("miner1", "PLAGIARISM", -1)
    assert manager.has_elimination_reason("miner1")

    # Remove miner
    manager.remove_miner("miner1")
    assert not manager.has_miner("miner1")
```

## Migration Checklist

- [ ] Create `CHALLENGEPERIOD_API_PROPOSAL.md` (this document)
- [ ] Review proposal with team
- [ ] Add new getter/setter methods to `ChallengePeriodManager`
- [ ] Add `import contextlib` for lock context manager
- [ ] Add unit tests for new methods
- [ ] Update `challengeperiod_manager.py` internal code (35+ occurrences for `active_miners`, 6 for `eliminations_with_reasons`)
- [ ] Update `elimination_manager.py` (2 occurrences for `active_miners`, 5 for `eliminations_with_reasons`)
- [ ] Update `position_manager.py` (4 occurrences for `active_miners`)
- [ ] Update test files:
  - [ ] `test_elimination_manager.py`
  - [ ] `test_elimination_core.py`
  - [ ] `test_elimination_weight_calculation.py`
  - [ ] `test_auto_sync.py`
  - [ ] `test_probation_comprehensive.py`
  - [ ] `test_plagiarism.py`
  - [ ] `test_challengeperiod_integration.py`
- [ ] Run full test suite
- [ ] Code review
- [ ] (Optional) Add deprecation warnings
- [ ] Update `CLAUDE.md` with new API patterns
- [ ] Close proposal and archive document

## Questions for Discussion

1. **Naming**: Should we use `get_miner_info()` or `get_miner_data()`?
   - `info` suggests metadata about the miner
   - `data` is more generic
   - Current proposal: `get_miner_info()`

2. **Validation Level**: How strict should validation be?
   - Strict: Raise errors for any invalid data (proposed)
   - Lenient: Log warnings but allow
   - Recommendation: Strict validation to catch bugs early

3. **Backward Compatibility**: Should we keep direct access working?
   - Option A: Deprecation warnings only (recommended)
   - Option B: Make it fail immediately
   - Option C: Keep it working indefinitely

4. **Tuple Structure**: Should we eventually replace tuples with a dataclass?
   - Current: `tuple[MinerBucket, int, MinerBucket|None, int|None]`
   - Future: `@dataclass MinerInfo` with named fields
   - Benefits: Better type hints, more readable, easier to extend
   - Recommendation: Consider for Phase 6

5. **Lock Strategy**: Should we add locks to `active_miners` access?
   - `eliminations_with_reasons` already uses locks
   - `active_miners` is also an IPC dict in production
   - Recommendation: Add locks for consistency and safety

## Lessons Learned from EliminationManager Refactoring

### Applied to This Proposal

1. **IPC Dict Mock Bug**: Fixed in elimination tests by using `side_effect=lambda: {}` instead of `return_value={}`
   - Apply same pattern when updating challenge period tests
   - Ensure each IPC dict gets unique instance

2. **Test Data Cleanup**: Added `clear_perf_ledger_eliminations_from_disk()` to prevent state bleeding
   - Add similar cleanup for challenge period data in tests
   - Ensure `_clear_challengeperiod_in_memory_and_disk()` is called in tearDown

3. **Defensive Code**: User feedback emphasized fixing root cause over workarounds
   - Apply same principle: fix data structure issues, don't add defensive `.get()` calls
   - Validate data at write time, not read time

4. **Incremental Migration**: Phase 1 (add methods) was completed successfully without breaking changes
   - Follow same pattern: add all methods first, then migrate usage
   - Run tests after each phase

## Conclusion

This refactoring will improve code quality, maintainability, and consistency across the codebase. The estimated total effort is 2-3 weeks of work, with the option to do it incrementally without breaking existing functionality.

The approach mirrors our successful EliminationManager refactoring, applying the same principles to the ChallengePeriodManager for consistent API design across the codebase.

By replacing fragile tuple index access with named getter/setter methods, we make the code more readable, less error-prone, and easier to maintain and extend in the future.

---

**Author**: Claude Code
**Date**: 2025-01-12
**Related**: ELIMINATION_API_PROPOSAL.md, IPC_DICT_REFERENCE_BUG_FIX.md
