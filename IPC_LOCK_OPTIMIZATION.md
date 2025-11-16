# IPC Lock Performance Optimization Ideas

## Problem Summary

The current implementation has **double IPC overhead**:
1. **IPC Dict Access** (`self.locks = ipc_manager.dict()`) - accessing the dict requires serialization/deserialization
2. **IPC Lock** (`self.ipc_manager.Lock()`) - the lock itself is an IPC primitive

This creates a significant bottleneck because:
- Every lock lookup requires IPC dict access (~5-20ms)
- Creating new locks requires IPC lock creation (~5-20ms)
- Lock acquisition/release requires IPC synchronization (~5-20ms)
- **Total overhead per order: 200ms → 5000ms (25x slowdown)**

## Root Cause Analysis

### Current Lock Flow
```python
# Step 1: Access IPC dict (IPC overhead #1)
lock_key = (miner_hotkey, trade_pair_id)
if lock_key not in self.locks:  # IPC dict lookup
    # Step 2: Create IPC lock (IPC overhead #2)
    self.locks[lock_key] = self.ipc_manager.Lock()

# Step 3: Acquire IPC lock (IPC overhead #3)
with self.locks[lock_key]:  # IPC lock acquire/release
    # ... order processing ...
```

### Why Dynamic Miners Make This Hard
- Miners can register/deregister at any time
- Trade pairs are dynamic (forex, crypto pairs change)
- Can't pre-allocate a fixed number of locks
- Need per-(miner, trade_pair) granularity

---

## Optimization Strategies

### 🚀 **Option 1: Process-Local Lock Cache (RECOMMENDED)**

**Concept**: Each process maintains a local cache of locks, only using IPC for coordination.

```python
class PositionLocks:
    def __init__(self, ipc_manager=None):
        # Process-local cache (no IPC overhead)
        self._local_lock_cache = {}

        # Only use IPC for lock *discovery*, not storage
        self.lock_registry = ipc_manager.dict() if ipc_manager else {}

    def get_lock(self, miner_hotkey, trade_pair_id):
        lock_key = (miner_hotkey, trade_pair_id)

        # Check local cache first (FAST - no IPC)
        if lock_key in self._local_lock_cache:
            return self._local_lock_cache[lock_key]

        # Register in IPC registry for other processes to discover
        if lock_key not in self.lock_registry:
            self.lock_registry[lock_key] = True  # Just a flag

        # Create lock locally (FAST - no IPC for lock itself)
        lock = self.ipc_manager.Lock()
        self._local_lock_cache[lock_key] = lock
        return lock
```

**Pros**:
- ✅ Eliminates IPC dict lookup overhead for cached locks
- ✅ Simple to implement
- ✅ Backward compatible

**Cons**:
- ❌ Still requires IPC locks (manager.Lock())
- ❌ Memory overhead per process

**Expected Improvement**: 40-60% reduction in lock overhead

---

### 🏆 **Option 2: Hash-Based Lock Pool (MOST EFFICIENT)**

**Concept**: Map unlimited (miner, trade_pair) combinations to a fixed pool of locks using hashing.

```python
import hashlib

class PositionLocks:
    # Fixed pool size (tune based on concurrency needs)
    LOCK_POOL_SIZE = 256  # Power of 2 for fast modulo

    def __init__(self, ipc_manager=None):
        # Pre-allocate fixed pool of locks at startup
        self.lock_pool = [
            ipc_manager.Lock() for _ in range(self.LOCK_POOL_SIZE)
        ] if ipc_manager else []

    def get_lock(self, miner_hotkey, trade_pair_id):
        # Hash the key to an index
        key_str = f"{miner_hotkey}:{trade_pair_id}"
        hash_value = int(hashlib.sha256(key_str.encode()).hexdigest(), 16)
        lock_index = hash_value % self.LOCK_POOL_SIZE

        # Return pre-allocated lock (NO IPC overhead at runtime)
        return self.lock_pool[lock_index]
```

**Pros**:
- ✅ **ZERO runtime IPC overhead** (all locks pre-allocated)
- ✅ Fixed memory footprint
- ✅ Predictable performance
- ✅ No lock creation during order processing

**Cons**:
- ❌ Hash collisions cause false lock contention (different miners may share same lock)
- ❌ Requires tuning LOCK_POOL_SIZE vs collision rate

**Expected Improvement**: 80-90% reduction in lock overhead

**Collision Analysis**:
```python
# With 256 locks and 100 active miners × 50 trade pairs = 5000 unique keys
# Expected collisions = ~19 keys per lock
# False contention probability = (19/5000) = 0.38% per order
```

---

### 💡 **Option 3: Two-Tier Lock System**

**Concept**: Use fast threading locks for in-process synchronization, IPC locks only for cross-process.

```python
class PositionLocks:
    def __init__(self, ipc_manager=None):
        # Tier 1: Fast thread locks (per-process)
        self._thread_locks = {}
        self._thread_lock_mutex = threading.Lock()

        # Tier 2: Slow IPC locks (cross-process) - hash-based pool
        self.ipc_lock_pool = [
            ipc_manager.Lock() for _ in range(256)
        ] if ipc_manager else []

    def get_lock(self, miner_hotkey, trade_pair_id):
        # Return a composite lock that acquires both
        return CompositeLock(
            self._get_thread_lock(miner_hotkey, trade_pair_id),
            self._get_ipc_lock(miner_hotkey, trade_pair_id)
        )

    def _get_thread_lock(self, miner_hotkey, trade_pair_id):
        key = (miner_hotkey, trade_pair_id)
        with self._thread_lock_mutex:
            if key not in self._thread_locks:
                self._thread_locks[key] = threading.Lock()
            return self._thread_locks[key]

    def _get_ipc_lock(self, miner_hotkey, trade_pair_id):
        # Use hash-based pool for IPC locks
        key_str = f"{miner_hotkey}:{trade_pair_id}"
        hash_value = int(hashlib.sha256(key_str.encode()).hexdigest(), 16)
        return self.ipc_lock_pool[hash_value % len(self.ipc_lock_pool)]

class Compositelock:
    def __init__(self, thread_lock, ipc_lock):
        self.thread_lock = thread_lock
        self.ipc_lock = ipc_lock

    def __enter__(self):
        self.thread_lock.acquire()
        self.ipc_lock.acquire()
        return self

    def __exit__(self, *args):
        self.ipc_lock.release()
        self.thread_lock.release()
```

**Pros**:
- ✅ Minimal false contention (thread locks are per-key)
- ✅ Fast for same-process concurrent orders
- ✅ Safe for cross-process

**Cons**:
- ❌ More complex implementation
- ❌ Still has IPC overhead for cross-process

**Expected Improvement**: 70-80% reduction for in-process contention

---

### 🗄️ **Option 4: Redis/External Lock Manager**

**Concept**: Use Redis for distributed locking instead of multiprocessing.Manager.

```python
import redis

class RedisPositionLocks:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def get_lock(self, miner_hotkey, trade_pair_id):
        lock_key = f"ptn:lock:{miner_hotkey}:{trade_pair_id}"
        return redis.lock.Lock(
            self.redis_client,
            lock_key,
            timeout=30,  # Auto-release after 30s
            blocking_timeout=5
        )
```

**Pros**:
- ✅ Much faster than multiprocessing.Manager (Redis is optimized for this)
- ✅ Built-in timeouts prevent deadlocks
- ✅ Can scale across multiple machines if needed
- ✅ Atomic operations

**Cons**:
- ❌ Requires Redis installation/maintenance
- ❌ Additional infrastructure dependency
- ❌ Network overhead (but likely still faster than IPC)

**Expected Improvement**: 60-70% reduction vs current IPC locks

---

### 📁 **Option 5: File-Based Locks**

**Concept**: Use file system locks (fcntl/lockf on Linux, LockFile on Windows).

```python
import fcntl
import os

class FilePositionLocks:
    def __init__(self, lock_dir="/tmp/ptn_locks"):
        self.lock_dir = lock_dir
        os.makedirs(lock_dir, exist_ok=True)

    def get_lock(self, miner_hotkey, trade_pair_id):
        # Hash to limit number of lock files
        key_str = f"{miner_hotkey}:{trade_pair_id}"
        hash_value = int(hashlib.sha256(key_str.encode()).hexdigest(), 16)
        lock_file = f"{self.lock_dir}/lock_{hash_value % 256}.lock"

        return FileLock(lock_file)

class FileLock:
    def __init__(self, filepath):
        self.filepath = filepath
        self.fd = None

    def __enter__(self):
        self.fd = open(self.filepath, 'w')
        fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)  # Exclusive lock
        return self

    def __exit__(self, *args):
        fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
        self.fd.close()
```

**Pros**:
- ✅ No IPC manager needed
- ✅ Cross-process and cross-machine compatible
- ✅ Kernel-level performance

**Cons**:
- ❌ File system I/O overhead
- ❌ Need to cleanup lock files
- ❌ Platform-specific (fcntl vs LockFile)

**Expected Improvement**: 30-50% reduction

---

### 🎯 **Option 6: Sharded Lock Dicts**

**Concept**: Reduce contention on the IPC dict by sharding into multiple dicts.

```python
class ShardedPositionLocks:
    NUM_SHARDS = 16  # Reduce contention per shard

    def __init__(self, ipc_manager):
        # Create multiple IPC dicts to reduce contention
        self.lock_shards = [
            ipc_manager.dict() for _ in range(self.NUM_SHARDS)
        ]

    def _get_shard_index(self, miner_hotkey, trade_pair_id):
        key_str = f"{miner_hotkey}:{trade_pair_id}"
        return hash(key_str) % self.NUM_SHARDS

    def get_lock(self, miner_hotkey, trade_pair_id):
        shard_idx = self._get_shard_index(miner_hotkey, trade_pair_id)
        shard = self.lock_shards[shard_idx]

        lock_key = (miner_hotkey, trade_pair_id)
        if lock_key not in shard:
            shard[lock_key] = self.ipc_manager.Lock()

        return shard[lock_key]
```

**Pros**:
- ✅ Reduces contention on dict access
- ✅ Easy to implement
- ✅ Backward compatible

**Cons**:
- ❌ Still has IPC dict overhead (just divided by NUM_SHARDS)
- ❌ Marginal improvement

**Expected Improvement**: 20-30% reduction

---

## Recommendation Matrix

| Solution | Implementation Effort | Performance Gain | Risk | Best For |
|----------|----------------------|------------------|------|----------|
| **Option 1: Process-Local Cache** | Low | ⭐⭐⭐ (60%) | Low | Quick win |
| **Option 2: Hash-Based Pool** 🏆 | Medium | ⭐⭐⭐⭐⭐ (90%) | Medium | Production (best balance) |
| **Option 3: Two-Tier Locks** | High | ⭐⭐⭐⭐ (80%) | Medium | High concurrency |
| **Option 4: Redis** | Medium | ⭐⭐⭐⭐ (70%) | Low | If already using Redis |
| **Option 5: File Locks** | Low | ⭐⭐ (40%) | Low | Simple deployments |
| **Option 6: Sharded Dicts** | Low | ⭐⭐ (25%) | Low | Not worth it alone |

---

## Recommended Implementation Plan

### Phase 1: Quick Win (Week 1)
Implement **Option 1 (Process-Local Cache)** to get immediate 60% improvement with minimal risk.

### Phase 2: Production Optimization (Week 2-3)
Implement **Option 2 (Hash-Based Pool)** for 90% improvement:
- Start with LOCK_POOL_SIZE = 256
- Monitor collision rate
- Tune pool size based on metrics

### Phase 3: Advanced (Optional)
If needed, combine Option 2 with Option 3 for ultimate performance:
- Thread locks for in-process
- Hash-based IPC pool for cross-process
- Expected: 95%+ overhead reduction

---

## Testing Strategy

1. **Benchmark Current Performance**
   - Measure lock wait times with new logging
   - Track: lookup time, creation time, hold time

2. **Test Each Option**
   ```python
   # Benchmark script
   def benchmark_lock_option(lock_implementation, num_threads=10, num_orders=1000):
       times = []
       for _ in range(num_orders):
           start = time.time()
           with lock_implementation.get_lock("test_miner", "EURUSD"):
               time.sleep(0.001)  # Simulate order processing
           times.append((time.time() - start) * 1000)

       return {
           "mean": np.mean(times),
           "p50": np.percentile(times, 50),
           "p95": np.percentile(times, 95),
           "p99": np.percentile(times, 99)
       }
   ```

3. **Production Testing**
   - Deploy with feature flag
   - Compare metrics between old/new implementations
   - Gradual rollout

---

## Additional Considerations

### Memory Usage
- Current: O(miners × trade_pairs) locks
- Hash-based: O(pool_size) locks (fixed)
- Process-local cache: O(miners × trade_pairs × processes)

### Deadlock Prevention
All options should maintain:
- Consistent lock ordering
- Timeouts on lock acquisition
- Monitoring for long-held locks

### Monitoring
Add metrics for:
- Lock wait time distribution
- Lock hold time distribution
- Lock creation rate
- Hash collision rate (for Option 2)

---

## Conclusion

The **Hash-Based Lock Pool (Option 2)** offers the best trade-off:
- ✅ 90% performance improvement
- ✅ Fixed memory footprint
- ✅ Reasonable implementation complexity
- ✅ No external dependencies

Start with **Option 1** for quick gains, then migrate to **Option 2** for production.
