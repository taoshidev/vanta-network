# ChallengePeriodManager RPC Refactoring Plan

## Current Architecture Problems

### Issue #1: Shared IPC Dicts
```python
# Current (slow):
self.active_miners = ipc_manager.dict()  # Shared across processes
self.eliminations_with_reasons = ipc_manager.dict()  # Shared across processes
```

**Problem**: Every dict operation is an RPC call (50-200x slower than local dicts)

### Issue #2: Shared Lock Between Managers
```python
# Current (complex):
self.eliminations_lock = eliminations_lock  # Shared with EliminationManager
with self.eliminations_lock:
    self.eliminations_with_reasons.clear()
```

**Problem**: Cross-process lock coordination is complex and error-prone

### Issue #3: Tight Coupling
```python
# EliminationManager directly accesses ChallengePeriodManager's dict:
eliminations_with_reasons = self.challengeperiod_manager.get_all_elimination_reasons()
```

**Problem**: Managers are tightly coupled via shared memory

## New RPC Architecture

### Design Principles
1. **Local dicts on server** - No IPC overhead
2. **RPC methods for communication** - No shared locks needed
3. **Loose coupling** - Managers communicate via RPC API

### ChallengePeriodManagerServer (Process)
```python
class ChallengePeriodManagerServer(CacheController):
    def __init__(self, ...):
        # Local dicts (fast!)
        self.active_miners: Dict[str, tuple] = {}
        self.eliminations_with_reasons: Dict[str, tuple[str, float]] = {}

        # No locks needed - RPC methods are automatically serialized

    # RPC Methods exposed to client
    def health_check_rpc(self) -> dict
    def get_all_elimination_reasons_rpc(self) -> dict
    def has_elimination_reasons_rpc(self) -> bool
    def clear_elimination_reasons_rpc(self) -> None
    def update_elimination_reasons_rpc(self, reasons_dict) -> int

    def has_miner_rpc(self, hotkey) -> bool
    def get_miner_bucket_rpc(self, hotkey) -> Optional[str]
    def get_miner_start_time_rpc(self, hotkey) -> Optional[int]
    def get_hotkeys_by_bucket_rpc(self, bucket) -> list[str]
    def get_all_miner_hotkeys_rpc(self) -> list[str]

    def set_miner_bucket_rpc(self, hotkey, bucket, start_time, prev_bucket, prev_time) -> bool
    def remove_miner_rpc(self, hotkey) -> bool

    def get_testing_miners_rpc(self) -> dict
    def get_success_miners_rpc(self) -> dict
    def get_probation_miners_rpc(self) -> dict
    def get_plagiarism_miners_rpc(self) -> dict

    # Internal methods (not exposed)
    def refresh(self, current_time, iteration_epoch)
    def run_update_loop(self)
    # ... all other internal logic
```

### ChallengePeriodManager Client (RPC Client)
```python
class ChallengePeriodManager(RPCServiceBase, CacheController):
    def __init__(self, ...):
        RPCServiceBase.__init__(self, service_name="ChallengePeriodManagerServer", port=50005, ...)
        CacheController.__init__(self, ...)

        # Start RPC service
        self._initialize_service()

    def _create_direct_server(self):
        return ChallengePeriodManagerServer(...)

    def _start_server_process(self, address, authkey, server_ready):
        process = Process(target=start_challengeperiod_manager_server, ...)
        return process

    # Client methods proxy to RPC
    def get_all_elimination_reasons(self) -> dict:
        return self._server_proxy.get_all_elimination_reasons_rpc()

    def has_elimination_reasons(self) -> bool:
        return self._server_proxy.has_elimination_reasons_rpc()

    def clear_elimination_reasons(self) -> None:
        self._server_proxy.clear_elimination_reasons_rpc()

    def has_miner(self, hotkey) -> bool:
        return self._server_proxy.has_miner_rpc(hotkey)

    # ... all other client methods
```

## Communication Flow

### Before (IPC Shared Dicts):
```
ChallengePeriodManager (Process A)  →  IPC Dict  ←  EliminationManager (Process B)
├── eliminations_with_reasons.update({hotkey: (reason, dd)})  # RPC call
└── Uses shared lock for synchronization

EliminationManager reads from same IPC dict:
├── eliminations_with_reasons.get_all()  # RPC call
└── eliminations_with_reasons.clear()     # RPC call
```

**Problem**: 3+ RPC calls for shared IPC dict operations

### After (RPC Services):
```
ChallengePeriodManager (Client)  ──RPC──>  ChallengePeriodManagerServer (Process)
└── Calls: update_elimination_reasons_rpc({hotkey: (reason, dd)})  # 1 RPC call

EliminationManager (Client)  ──RPC──>  ChallengePeriodManagerServer (Process)
├── Calls: get_all_elimination_reasons_rpc()  # 1 RPC call
└── Calls: clear_elimination_reasons_rpc()    # 1 RPC call
```

**Benefit**: Same number of RPC calls, but simpler architecture (no shared locks)

## EliminationManager Integration

### Update EliminationManagerServer
```python
class EliminationManagerServer(CacheController):
    def __init__(self, ..., challengeperiod_manager):
        # challengeperiod_manager is now RPC client (not direct reference)
        self.challengeperiod_manager = challengeperiod_manager

    def handle_challenge_period_eliminations(self, ...):
        # OLD: Direct dict access
        # eliminations_with_reasons = self.challengeperiod_manager.eliminations_with_reasons

        # NEW: RPC call
        eliminations_with_reasons = self.challengeperiod_manager.get_all_elimination_reasons()

        # Process eliminations...

        # OLD: Direct dict clear
        # self.challengeperiod_manager.eliminations_with_reasons.clear()

        # NEW: RPC call
        self.challengeperiod_manager.clear_elimination_reasons()
```

## Port Allocation

- PositionManager: 50002
- LimitOrderManager: 50003
- LivePriceFetcher: 50001
- EliminationManager: 50004
- **ChallengePeriodManager: 50005** ← NEW

## Performance Impact

### Before (IPC Managerized Dicts):
```python
# Update 10 elimination reasons
for hotkey, reason in eliminations.items():
    manager.eliminations_with_reasons[hotkey] = reason  # 10 RPC calls
```

### After (RPC Service):
```python
# Update 10 elimination reasons
manager.update_elimination_reasons(eliminations)  # 1 RPC call
```

**Result**: 10x fewer RPC calls for batch operations

## Backward Compatibility

All public methods maintain same signatures:
- ✅ `get_all_elimination_reasons()` - Same interface
- ✅ `has_elimination_reasons()` - Same interface
- ✅ `clear_elimination_reasons()` - Same interface
- ✅ `has_miner(hotkey)` - Same interface
- ✅ `get_miner_bucket(hotkey)` - Same interface
- ✅ All other methods - Same interfaces

**No breaking changes!**

## Migration Steps

1. ✅ Create `ChallengePeriodManagerServer` with local dicts
2. ✅ Create `ChallengePeriodManager` client extending `RPCServiceBase`
3. ✅ Update `EliminationManagerServer` to use RPC calls (not direct dict access)
4. ✅ Remove `ipc_manager` and `eliminations_lock` parameters
5. ✅ Test coordination between managers

## Testing Strategy

1. **Unit tests**: Use direct mode (`running_unit_tests=True`)
2. **Integration tests**: Verify ChallengePeriodManager ↔ EliminationManager coordination
3. **Performance tests**: Measure RPC call reduction

## Expected Benefits

1. **Simpler architecture**: No shared IPC dicts, no shared locks
2. **Better performance**: Local dicts on server (10-100x faster)
3. **Clearer separation**: Managers communicate via well-defined RPC API
4. **Easier debugging**: RPC calls are explicit, not hidden in IPC dict operations
5. **Consistent pattern**: All managers use same RPC architecture

## Implementation Files

### New Files
- `vali_objects/utils/challengeperiod_manager_server.py` - Server implementation
- `vali_objects/utils/challengeperiod_manager.py` - Client (rewritten)

### Modified Files
- `vali_objects/utils/elimination_manager_server.py` - Use ChallengePeriodManager RPC client
- `neurons/validator.py` - Update initialization (no more `ipc_manager`, `eliminations_lock` params)

## Conclusion

This refactoring:
- Eliminates shared IPC dicts (50-200x faster)
- Removes shared lock complexity
- Maintains 100% backward compatibility
- Follows established RPC service pattern
- Enables proper process isolation

Both ChallengePeriodManager and EliminationManager will be clean RPC services communicating via well-defined APIs.
