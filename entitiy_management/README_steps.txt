# Implementation Prompt: Entity Miners Feature for Vanta Network

## Feature Overview
Implement an "Entity Miners" system that allows one entity hotkey (VANTA_ENTITY_HOTKEY) to manage multiple subaccounts, each with synthetic hotkeys. This enables entities to operate multiple trading strategies under a single parent entity with collateral verification and elimination tracking.

### Phase 1 Specifications
- Each entity miner can host up to **500 active sub-accounts** during Phase 1
- Order rate limits are enforced **per sub-account** (per synthetic hotkey)
- Since synthetic hotkeys are passed into the system, existing per-hotkey rate limiting automatically applies
- **No changes needed to rate limiting logic** - it already works per-hotkey
- This allows entities to submit orders much faster by distributing across multiple sub-accounts

### Sub-account Challenge Period
- Sub-accounts enter a **90-day challenge period** upon creation
- **Instantaneous pass criteria**: 3% returns against 6% drawdown within 90 days
- Challenge period assessment runs in EntityManager daemon (similar to MDD checker)
- Uses PerfLedgerClient to get returns and drawdown at different intervals
- Once passed, sub-account exits challenge period and operates normally
- Failed sub-accounts are eliminated after 90 days if criteria not met

## Context Files to Review
Before starting, review these files to understand existing patterns:
1. `vali_objects/challenge_period/` - Reference for Manager/Server/Client RPC pattern
2. `vali_objects/utils/elimination_manager.py` - Elimination logic patterns
3. `vali_objects/utils/mdd_checker/` - MDD checker pattern (returns and drawdown at intervals)
4. `vali_objects/vali_dataclasses/perf_ledger.py` - Performance ledger client usage
5. `shared_objects/rpc/rpc_server_base.py` - Base RPC server class
6. `shared_objects/rpc/rpc_client_base.py` - Base RPC client class
7. `neurons/validator.py` - Look for `broadcast_asset_selection_to_validators` method
8. `vali_objects/vali_config.py` - Port definitions and configuration
9. `vanta_api/rest_server.py` - REST API patterns
10. `shared_objects/metagraph_manager.py` - Metagraph logic and `has_hotkey` method

## Architecture Requirements

### 1. Data Model
Create entity data structures with:
- Entity hotkey (VANTA_ENTITY_HOTKEY)
- Subaccount list with monotonically increasing IDs
- Synthetic hotkey format: `{VANTA_ENTITY_HOTKEY}_{subaccount_id}`
- Subaccount status: active/eliminated/unknown
- Challenge period tracking per subaccount:
  - challenge_period_active: bool (default True for new subaccounts)
  - challenge_period_passed: bool (default False)
  - challenge_period_start_ms: timestamp
  - challenge_period_end_ms: timestamp (90 days from start)
- Collateral tracking per entity
- Slot allowance tracking
- Registration timestamps
- UUID generation for subaccounts

### 2. EntityManager (entitiy_management/entity_manager.py)
Implement core business logic following the challenge_period pattern:
- Persistent disk storage (similar to challenge_period state management)
- Entity registration and tracking
- Subaccount creation with monotonic ID generation
  - Track active subaccount IDs (max 500 active at once)
  - Track next_subaccount_id (monotonically increasing, never reused)
  - Eliminated IDs implicitly: IDs in [0, next_id) that aren't in active set
  - Initialize challenge period fields on creation
- Subaccount status management (active/eliminated)
- Challenge period assessment (daemon):
  - Uses PerfLedgerClient to get returns and drawdown
  - Checks 3% returns against 6% drawdown threshold
  - Operates similar to MDD checker with interval checks
  - Marks challenge_period_passed=True on success
  - Eliminates subaccount after 90 days if not passed
- Collateral verification methods (placeholder implementation)
- Slot allowance checking
- Elimination criteria assessment (placeholder for periodic daemon)
- Thread-safe operations with proper locking
- Getters: get_entity_data, get_subaccount_status, get_all_entities, is_synthetic_hotkey, is_registered_entity
- Setters: register_entity, create_subaccount, eliminate_subaccount, update_collateral, mark_challenge_period_passed
- Dependencies: PerfLedgerClient (for challenge period assessment)
- PLACEHOLDER: Collateral transfer during subaccount creation (blackbox function)
- PLACEHOLDER: Account size initialization via ContractClient during subaccount creation

### 3. EntityServer (entitiy_management/entity_server.py)
RPC server inheriting from RPCServerBase:
- Service name: "EntityServer"
- Port: Add RPC_ENTITY_SERVER_PORT to vali_config.py (e.g., 50023)
- Connection modes: LOCAL and RPC support
- Delegate all operations to EntityManager instance
- RPC-exposed methods matching manager's public API
- Background daemon thread for periodic assessment:
  - Challenge period evaluation (every 5 minutes or configurable interval)
  - Checks returns/drawdown via PerfLedgerClient
  - Marks passed subaccounts or eliminates expired ones
  - Elimination assessment (placeholder logic)
- Health check endpoint
- Graceful shutdown handling

### 4. EntityClient (entitiy_management/entity_client.py)
RPC client inheriting from RPCClientBase:
- Connect to EntityServer
- Proxy methods for all manager operations
- Support both LOCAL and RPC connection modes
- Error handling and retry logic
- Methods: register_entity, create_subaccount, get_subaccount_status, get_entity_data, is_synthetic_hotkey, eliminate_subaccount

### 5. REST API Integration (vanta_api/rest_server.py)
Add new endpoints to VantaRestServer:

**POST /register_subaccount**
- Input: entity_hotkey, signature/auth
- Collateral verification (placeholder call)
- Slot allowance check via EntityClient
- Create subaccount via EntityClient
- Broadcast new subaccount to all validators (synapse message)
- Return: {success: bool, subaccount_id: int, subaccount_uuid: str, synthetic_hotkey: str}

**GET /subaccount_status/{subaccount_id}**
- Query EntityClient for status
- Return: {status: "active"|"eliminated"|"unknown", synthetic_hotkey: str}

**GET /entity_data/{entity_hotkey}**
- Query EntityClient for full entity data
- Return: {entity_hotkey, subaccounts: [...], collateral, active_count}

### 6. Validator Order Placement Restrictions (neurons/validator.py)
Add validation to prevent entity hotkeys from placing orders:
- **RULE**: Only synthetic hotkeys (subaccounts) can place orders
- **RULE**: Entity hotkeys (VANTA_ENTITY_HOTKEY) cannot place orders directly
- Add validation in signal processing:
  - Check if hotkey is a registered entity (not synthetic)
  - If entity hotkey detected, reject order with error message
  - Only allow orders from synthetic hotkeys
- Implementation location: Signal validation in validator.py
- Use EntityClient.is_registered_entity() and is_synthetic_hotkey() methods

### 7. Validator Syncing (neurons/validator.py or similar)
Implement broadcast mechanism following broadcast_asset_selection_to_validators pattern:
- Create new synapse type for subaccount registration
- `broadcast_subaccount_registration(entity_hotkey, subaccount_id, subaccount_uuid)` method
- Send to all validators in metagraph
- Handle responses and log failures
- Ensure idempotent registration (handle duplicates gracefully)

### 8. Metagraph Integration (shared_objects/metagraph_manager.py)
Update metagraph logic to support synthetic hotkeys:

**Update `has_hotkey` method:**
- Detect synthetic hotkeys (contains underscore)
- Parse entity hotkey and subaccount_id from synthetic format
- Verify entity hotkey exists in raw metagraph
- Query EntityClient to check if subaccount is active
- Return True only if both entity exists and subaccount is active

**Consider updates to:**
- UID resolution for synthetic hotkeys (may need synthetic UID mapping)
- Hotkey validation methods
- Any caching mechanisms that assume hotkeys don't have underscores

### 9. Debt Ledger Aggregation for Entity Scoring
Implement aggregation layer for entity debt-based scoring:

**Requirement**: Aggregate all subaccount debt ledgers into a single entity debt ledger
- **Key**: Entity hotkey (VANTA_ENTITY_HOTKEY)
- **Value**: Sum of all active subaccount performance metrics
- **Update trigger**: Whenever any subaccount performance changes

**Implementation**:
- Add method: `aggregate_entity_debt_ledger(entity_hotkey) -> DebtLedger`
  - Query EntityClient for all active subaccounts
  - Read individual debt ledgers for each synthetic hotkey
  - Sum performance metrics (PnL, returns, etc.)
  - Store aggregated ledger under entity hotkey
- Location: Debt ledger scoring system (vali_objects/vali_dataclasses/debt_ledger.py or scoring/)
- Scoring reads entity-level aggregated ledger for weight calculation
- Individual subaccounts maintain separate ledgers for tracking

**Aggregation logic**:
```python
def aggregate_entity_debt_ledger(entity_hotkey: str) -> DebtLedger:
    entity_data = entity_client.get_entity_data(entity_hotkey)
    active_subaccounts = entity_data.get_active_subaccounts()

    aggregated_ledger = DebtLedger(hotkey=entity_hotkey)

    for subaccount in active_subaccounts:
        synthetic_hotkey = subaccount.synthetic_hotkey
        subaccount_ledger = debt_ledger_client.get_ledger(synthetic_hotkey)
        aggregated_ledger.add_ledger(subaccount_ledger)  # Sum metrics

    return aggregated_ledger
```

### 10. Subaccount Registration Enhancements
Add collateral and account size initialization to registration flow:

**During subaccount creation** (entity_manager.py or REST endpoint):
1. **PLACEHOLDER**: Transfer collateral from entity to subaccount
   - `blackbox_transfer_collateral(from_hotkey=entity_hotkey, to_hotkey=synthetic_hotkey, amount=AMOUNT)`
   - This is a placeholder for future collateral SDK integration

2. **PLACEHOLDER**: Initialize account size via ContractClient
   - `contract_client.set_account_size(hotkey=synthetic_hotkey, account_size=FIXED_SIZE)`
   - Fixed account size for all subaccounts (e.g., 10000 USD)
   - This happens after collateral transfer succeeds

**Updated create_subaccount flow**:
```python
# Create subaccount metadata
subaccount_info = SubaccountInfo(...)

# PLACEHOLDER: Transfer collateral
# success = blackbox_transfer_collateral(entity_hotkey, synthetic_hotkey, amount)
# if not success:
#     return False, None, "Collateral transfer failed"

# PLACEHOLDER: Set account size
# contract_client.set_account_size(synthetic_hotkey, FIXED_SUBACCOUNT_SIZE)

return True, subaccount_info, "Success"
```

### 11. Configuration (vali_objects/vali_config.py)
Add configuration parameters:
- RPC_ENTITY_SERVER_PORT = 50023
- ENTITY_ELIMINATION_CHECK_INTERVAL = 300  # 5 minutes (for challenge period + elimination checks)
- ENTITY_MAX_SUBACCOUNTS = 500  # Maximum active subaccounts per entity
- ENTITY_DATA_DIR = "validation/entities/"  # Persistence directory
- FIXED_SUBACCOUNT_SIZE = 10000.0  # Fixed account size for subaccounts (USD)
- SUBACCOUNT_COLLATERAL_AMOUNT = 1000.0  # Placeholder collateral amount
- SUBACCOUNT_CHALLENGE_PERIOD_DAYS = 90  # Challenge period duration
- SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD = 0.03  # 3% returns required
- SUBACCOUNT_CHALLENGE_DRAWDOWN_THRESHOLD = 0.06  # 6% max drawdown allowed

## Implementation Steps

### Phase 1: Core Infrastructure (entitiy_management/)
1. Implement EntityManager class with data structures and persistence
   - Add challenge period fields to SubaccountInfo
   - Initialize challenge period on subaccount creation
2. Implement EntityServer with RPC capabilities
   - Configure daemon interval for challenge period checks (5 minutes)
3. Implement EntityClient for RPC communication
4. Add PerfLedgerClient dependency to EntityManager
5. Implement challenge period assessment logic in daemon:
   - Check returns and drawdown via PerfLedgerClient
   - Mark challenge_period_passed=True on success (3% returns, 6% drawdown)
   - Eliminate subaccounts after 90 days if not passed
6. Add comprehensive unit tests for manager logic
7. Add integration tests for RPC communication (LOCAL and RPC modes)
8. Add challenge period tests (pass, fail, edge cases)

### Phase 2: Validator Order Placement Restrictions
1. Add is_registered_entity() method to EntityManager
2. Expose is_registered_entity_rpc() in EntityServer
3. Add is_registered_entity() to EntityClient
4. Update validator.py signal validation to check entity hotkeys
5. Reject orders from entity hotkeys (non-synthetic)
6. Test order rejection for entity hotkeys
7. Test order acceptance for synthetic hotkeys

Note: Rate limiting works automatically per synthetic hotkey since existing logic is per-hotkey. No changes needed.

### Phase 3: Metagraph Integration
1. Update has_hotkey method in metagraph_manager.py
2. Add synthetic hotkey detection utility methods
3. Test synthetic hotkey validation end-to-end
4. Ensure existing position management works with synthetic hotkeys

### Phase 4: REST API & Validator Syncing
1. Add REST endpoints to VantaRestServer
2. Add placeholder collateral transfer in subaccount creation
3. Add placeholder account size initialization via ContractClient
4. Implement synapse message type for subaccount registration
5. Implement broadcast mechanism in validator
6. Add API authentication/authorization for entity endpoints
7. Test REST API with synthetic hotkeys

### Phase 5: Debt Ledger Aggregation
1. Review debt ledger system architecture
2. Implement aggregate_entity_debt_ledger() method
3. Add DebtLedger.add_ledger() method for summing metrics
4. Update scoring logic to use aggregated entity ledgers
5. Test aggregation with multiple active subaccounts
6. Test that eliminated subaccounts are excluded from aggregation
7. Verify entity-level weights use aggregated performance

### Phase 6: System Integration
1. Integrate EntityServer into server orchestrator startup
2. Test full flow: register entity → create subaccount → submit orders → track positions
3. Verify order rejection for entity hotkeys
4. Verify order acceptance for synthetic hotkeys
5. Test debt ledger aggregation for scoring
6. Verify elimination logic (placeholder) runs periodically
7. Test validator syncing across multiple validator instances
8. Load testing with multiple entities and subaccounts

### Phase 7: Placeholder Implementation Notes
1. Collateral verification: Add TODO comment with interface specification
2. Collateral transfer: Add placeholder blackbox_transfer_collateral()
3. Account size: Add placeholder ContractClient.set_account_size()
4. Elimination criteria: Add placeholder that logs but doesn't eliminate
5. Real collateral integration: Document expected collateral SDK integration points

## Testing Strategy

### Unit Tests (tests/vali_tests/test_entity_*.py)
- test_entity_manager.py: Test all manager operations, persistence, thread safety
- test_entity_server_client.py: Test RPC communication in both modes
- test_synthetic_hotkeys.py: Test hotkey parsing and validation

### Integration Tests
- test_entity_rest_api.py: Test REST endpoints with EntityClient
- test_entity_metagraph.py: Test metagraph with synthetic hotkeys
- test_entity_positions.py: Test position tracking with synthetic hotkeys
- test_entity_elimination.py: Test elimination flow for subaccounts

### Key Test Cases
1. Entity registration with collateral check
2. Subaccount creation with monotonic IDs
3. Monotonic ID behavior: Verify next_subaccount_id never reuses eliminated IDs
4. Active subaccount limit: Verify max 500 active subaccounts per entity
5. Synthetic hotkey generation and parsing
6. Subaccount elimination and status updates
7. Challenge period initialization on subaccount creation
8. Challenge period pass: 3% returns against 6% drawdown
9. Challenge period failure: Elimination after 90 days
10. Challenge period assessment via daemon (PerfLedgerClient integration)
11. Entity hotkey order rejection (cannot place orders)
12. Synthetic hotkey order acceptance (can place orders)
13. Verify rate limiting works independently per synthetic hotkey (existing logic)
14. Debt ledger aggregation for entity scoring
15. Aggregation excludes eliminated subaccounts
16. Collateral transfer placeholder integration
17. Account size initialization placeholder integration
18. Metagraph has_hotkey with synthetic hotkeys
19. REST API authentication and authorization
20. Validator broadcast and sync
21. Position submission and tracking with synthetic hotkeys
22. Concurrent subaccount operations (thread safety)
23. Persistence and recovery from disk
24. Challenge period state persistence across restarts

## Code Style & Patterns
- Follow existing RPC architecture patterns (see challenge_period/)
- Use Pydantic for data validation and serialization
- Implement proper logging with context (entity_hotkey, subaccount_id)
- Use snake_case for file and method names
- Add docstrings for all public methods
- Handle errors gracefully with descriptive messages
- Use type hints throughout
- Follow validator state persistence patterns

## Success Criteria
✓ Entity can register and create multiple subaccounts
✓ Synthetic hotkeys work seamlessly with existing position management
✓ REST API endpoints functional and secured
✓ Validator syncing maintains consistency across network
✓ Metagraph correctly validates synthetic hotkeys
✓ Elimination logic framework in place (with placeholders)
✓ Comprehensive test coverage (>80%)
✓ Documentation complete with usage examples

## Key Edge Cases to Handle
- Duplicate subaccount registration attempts
- Entity hotkey that looks synthetic (contains underscore)
- Entity hotkey attempting to place orders (should be rejected)
- Subaccount operations after entity is eliminated
- Subaccount operations after parent entity is removed from metagraph
- Race conditions in subaccount ID generation
- Validator sync failures and retry logic
- Disk persistence failures and recovery
- Collateral verification failures
- Collateral transfer failures during subaccount creation
- Account size initialization failures
- Maximum subaccount limit enforcement (500 active)
- Challenge period edge cases:
  - Subaccount reaching exactly 3% returns at exactly 6% drawdown
  - PerfLedgerClient unavailable during daemon check
  - Challenge period expiry during validator restart
  - Multiple subaccounts passing challenge period simultaneously
  - Challenge period status persistence across crashes
  - Subaccount elimination while in active challenge period
- Debt ledger aggregation when some subaccounts have no ledger data
- Debt ledger aggregation performance with 500 active subaccounts
- next_subaccount_id overflow (unlikely but handle gracefully)

## Additional Considerations
- Migration path for existing miners to entities (if needed)
- Monitoring and alerting for entity/subaccount operations
- Rate limiting for subaccount creation
- Audit logging for all entity operations
- Performance impact of synthetic hotkey validation
- Backward compatibility with non-entity miners

---

## Quick Start Command for Implementation
Once you start implementing, use this approach:
1. Read all context files listed above to understand patterns
2. Start with entity_manager.py (core logic, no RPC dependencies)
3. Add entity_server.py and entity_client.py (RPC layer)
4. Update vali_config.py with new configuration
5. Update metagraph logic for synthetic hotkeys
6. Add REST API endpoints
7. Implement validator broadcast mechanism
8. Write comprehensive tests for each component
9. Integration test the full flow

Use LOCAL connection mode for fast unit testing, RPC mode for integration testing.
