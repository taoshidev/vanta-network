# RPC Services Deployment Guide

## TL;DR

**DO NOT fork the validator process after it initializes RPC services.** RPC client proxies cannot survive process forking.

## The Problem

If you see this error in production:
```
ERROR | LivePriceFetcher health check failed: _server_proxy is None
ERROR | Process was forked after RPC initialization (DEPLOYMENT ERROR)
```

This means your deployment is forking the validator **AFTER** it creates RPC services. This is a **misconfiguration**, not a bug.

## Why Forking Breaks RPC Services

Our architecture:
```
Validator Process (main)
├── Creates RPC client proxies (LivePriceFetcher, PositionManager, etc.)
│   └── Each proxy connects to a server process via socket
├── RPC Server Processes (separate PIDs)
    └── LivePriceFetcherServer (PID 1234, port 50001)
    └── PositionManagerServer (PID 1235, port 50002)
    └── etc.
```

When PM2/systemd forks the validator:
```
Parent Process (validator)
├── RPC client proxies (connected to servers)
└── FORK →
    Child Process (daemon)
    ├── Inherited RPC proxies (BROKEN - socket connections invalid)
    └── RPC servers still running (connected to parent, not child)
```

**Result**: Child process has broken proxy objects that can't communicate with servers.

## Correct Deployment Patterns

### Option 1: PM2 Without Fork Mode (Recommended)

```bash
# In your run.sh or PM2 config
pm2 start neurons/validator.py \
  --interpreter python3 \
  --name validator \
  --no-daemon \  # DON'T fork
  -- --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>
```

PM2 manages the process directly without forking.

### Option 2: Systemd Service (Production)

```ini
# /etc/systemd/system/validator.service
[Unit]
Description=Proprietary Trading Network Validator
After=network.target

[Service]
Type=simple  # NOT forking
User=validator
WorkingDirectory=/path/to/proprietary-trading-network
ExecStart=/usr/bin/python3 neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Systemd manages the process without requiring fork.

### Option 3: Direct Execution (Development)

```bash
# Just run it directly (no PM2, no systemd)
python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>
```

### Option 4: Screen/tmux (Simple Alternative)

```bash
# Start a screen session
screen -S validator

# Run validator
python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>

# Detach: Ctrl+A then D
# Reattach: screen -r validator
```

## Incorrect Deployment Patterns (DO NOT USE)

### ❌ PM2 Fork Mode

```bash
# WRONG - PM2 will fork after RPC initialization
pm2 start neurons/validator.py --interpreter python3 --name validator --daemon
```

### ❌ Systemd with Type=forking

```ini
# WRONG - systemd expects process to fork itself
[Service]
Type=forking  # DON'T USE
```

### ❌ Manual Fork After Startup

```python
# WRONG - never fork after creating RPC services
validator = Validator()  # Creates RPC services
os.fork()  # BREAKS RPC proxies
```

## How Our RPC Design Works

1. **Validator initialization** (main process)
   ```python
   # neurons/validator.py
   self.live_price_fetcher = LivePriceFetcherClient(...)
   self.position_manager = PositionManager(...)
   self.elimination_manager = EliminationManager(...)
   # etc.
   ```

2. **Each RPC client spawns a server** (separate process)
   ```python
   # In RPCServiceBase._initialize_service()
   self._server_process = self._start_server_process(...)  # spawn()
   self._connect_client()  # socket connection
   ```

3. **Server processes run independently**
   - LivePriceFetcherServer: PID 1234, port 50001
   - PositionManagerServer: PID 1235, port 50002
   - LimitOrderManagerServer: PID 1236, port 50003
   - EliminationManagerServer: PID 1237, port 50004
   - ChallengePeriodManagerServer: PID 1238, port 50005

4. **Main validator uses RPC proxies** to communicate
   ```python
   # These are proxy objects with socket connections
   price = self.live_price_fetcher.get_close_for_prediction(trade_pair)
   ```

## Why spawn() Works But fork() Doesn't

- **`multiprocessing.Process()`** uses `spawn()` by default on macOS
  - Creates fresh process
  - Doesn't inherit proxy connections
  - RPC client creates NEW connection to server

- **PM2/systemd fork** happens AFTER initialization
  - Child inherits broken proxy connections
  - Can't create new connections (servers belong to parent)
  - Result: broken RPC communication

## Debugging Fork Issues

If you suspect fork issues:

### 1. Check Process Tree

```bash
# See validator and its RPC server children
ps auxf | grep validator
ps auxf | grep LivePriceFetcher
ps auxf | grep PositionManager
```

You should see:
```
validator (PID 1000)
├── LivePriceFetcherServer (PID 1001)
├── PositionManagerServer (PID 1002)
├── LimitOrderManagerServer (PID 1003)
└── etc.
```

### 2. Check Logs for Fork Detection

```bash
tail -f logs/validator.log | grep "forked after RPC initialization"
```

If you see this, your deployment is wrong.

### 3. Verify Port Ownership

```bash
# Check what process owns RPC ports
lsof -i :50001  # LivePriceFetcher
lsof -i :50002  # PositionManager
lsof -i :50003  # LimitOrderManager
lsof -i :50004  # EliminationManager
lsof -i :50005  # ChallengePeriodManager
```

All should show active server processes.

## Migration from Fork-Based Deployment

If you're currently using PM2 fork mode:

```bash
# 1. Stop current deployment
pm2 stop validator
pm2 delete validator

# 2. Update run.sh to use --no-daemon
# Edit run.sh and add --no-daemon flag

# 3. Restart with correct config
./run.sh --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>

# 4. Verify no fork warnings
pm2 logs validator | grep -i fork
```

## Summary

| Method | Fork After Init? | Supported? | Recommended? |
|--------|-----------------|-----------|--------------|
| PM2 --no-daemon | ❌ No | ✅ Yes | ✅ Yes |
| PM2 --daemon | ✅ Yes | ❌ No | ❌ No |
| Systemd Type=simple | ❌ No | ✅ Yes | ✅ Yes (prod) |
| Systemd Type=forking | ✅ Yes | ❌ No | ❌ No |
| Screen/tmux | ❌ No | ✅ Yes | ✅ Yes (dev) |
| Direct execution | ❌ No | ✅ Yes | ✅ Yes (dev) |

## Questions?

**Q: Why not just handle forks gracefully?**
A: Forking breaks the fundamental client-server architecture. Server processes belong to parent, child can't manage them. Better to fail loudly than silently corrupt state.

**Q: Can I daemonize the validator?**
A: Yes! Use PM2 --no-daemon or systemd Type=simple. They manage the process without forking it.

**Q: What if I need true daemonization (detach from terminal)?**
A: Use screen/tmux or let PM2/systemd handle it. They detach without forking.

**Q: Will this affect performance?**
A: No. The difference is WHO manages the process (PM2 vs kernel), not how it runs.
