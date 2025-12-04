# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Vanta Network (formerly Proprietary Trading Network/PTN) is a Bittensor subnet (netuid 8 mainnet, 116 testnet) developed by Taoshi. It operates as a competitive trading signal network where miners submit trading strategies and validators evaluate their performance using sophisticated metrics and risk-adjusted scoring.

## Development Commands

### Python Environment Setup
```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python3 -m pip install -e .
```

### Running Components
```bash
# Validator (production with PM2) - uses "vanta" process name
./run.sh --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>

# Miner
python neurons/miner.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <miner>

# Validator (development mode)
python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <default>

# Signal reception server for miners
./run_receive_signals_server.sh

# Utility scripts in runnable/
python runnable/check_validator_weights.py
python runnable/daily_portfolio_returns.py
python runnable/local_debt_ledger.py
```

### Testing
```bash
# Run all validator tests
python tests/run_vali_testing_suite.py

# Run specific test file
python tests/run_vali_testing_suite.py test_positions.py
```

### Miner Dashboard (React/TypeScript)
```bash
cd miner_objects/miner_dashboard
npm install
npm run dev      # Development server
npm run build    # TypeScript compilation + Vite build
npm run lint     # ESLint
npm run preview  # Preview production build
```

## Architecture Overview

### Core Network Components
- **`neurons/`** - Main network participants
  - `validator.py` - Validator orchestration and network management
  - `miner.py` - Miner signal generation and submission
  - `validator_base.py` - Base validator functionality
  - `backtest_manager.py` - Backtesting utilities
- **`vali_objects/`** - Validator logic and services
  - `challenge_period/` - Challenge period management for new miners
  - `plagiarism/` - Plagiarism detection and scoring
  - `position_management/` - Position tracking and management
  - `price_fetcher/` - Real-time price data services
  - `scoring/` - Performance metrics calculation
  - `statistics/` - Miner performance statistics
  - `utils/` - Utility services (elimination, asset selection, MDD checking, limit orders)
  - `vali_dataclasses/` - Data structures for positions, orders, ledgers
- **`shared_objects/`** - Common infrastructure
  - `rpc/` - RPC architecture (server_orchestrator, rpc_server_base, rpc_client_base)
  - `locks/` - Position locking mechanisms
  - `metagraph/` - Metagraph management and caching
  - Utilities: cache_controller, slack_notifier, error_utils
- **`miner_objects/`** - Miner tooling
  - `miner_dashboard/` - React/TypeScript dashboard for monitoring
  - `prop_net_order_placer.py` - Order placement utilities
  - `position_inspector.py` - Position analysis tools
- **`template/`** - Bittensor protocol definitions and base classes

### Data Infrastructure
- **`data_generator/`** - Financial market data services (Polygon, Tiingo, Binance, Bybit, Kraken)
- **`vanta_api/`** - Vanta Network API layer
  - `rest_server.py` - REST API server for signal submission and queries
  - `websocket_server.py` / `websocket_client.py` - Real-time WebSocket communication
  - `api_manager.py` - API key and authentication management
  - `nonce_manager.py` - Request nonce handling
- **`mining/`** - Signal processing pipeline
  - `received_signals/` - Incoming miner signals
  - `processed_signals/` - Validated and processed signals
- **`validation/`** - Validator state persistence
  - `miners/` - Per-miner performance and position data
  - `plagiarism/` - Plagiarism detection cache
  - `tmp/` - Temporary processing files
- **`runnable/`** - Utility scripts and analysis tools
  - Portfolio analytics, debt ledger management, elimination analysis
  - Checkpoint validation and migration scripts
- **`tests/`** - Test suites
  - `vali_tests/` - Comprehensive validator unit and integration tests
  - `validation/` - Validation-specific test scenarios
  - `shared_objects/` - Shared infrastructure tests

### RPC Architecture
The system uses a distributed RPC architecture for inter-process communication:
- **Server Orchestrator**: Manages lifecycle of all RPC servers
- **18+ RPC Services**: Position management, elimination, plagiarism, price fetching, ledgers, etc.
- **Port Range**: 50000-50022 (centrally managed in vali_config.py)
- **Connection Modes**: LOCAL (direct/testing) and RPC (network/production)

### Key Configuration Files
- **`vali_objects/vali_config.py`** - Main validator configuration
  - RPC service definitions and ports
  - Trade pair definitions (crypto, forex, equities, indices)
  - Scoring weights and risk parameters
  - Challenge period and elimination thresholds
- **`miner_config.py`** - Miner configuration
- **`requirements.txt`** - Python dependencies (Bittensor 9.9.0, Pydantic 2.10.3, financial APIs)
- **`meta/meta.json`** - Version management (subnet_version: 8.8.8)
- **`setup.py`** - Package setup (taoshi-prop-net)

## Trading System Architecture

### Signal Flow
1. Miners submit LONG/SHORT/FLAT signals via Vanta API (REST/WebSocket)
2. Validators receive and validate signals through `vanta_api/rest_server.py`
3. Real-time price validation using multiple data sources (Polygon, Tiingo, Binance, Bybit, Kraken)
4. Position tracking via RPC services with leverage limits and slippage modeling
5. Performance calculation using debt-based scoring system

### Supported Assets
- **Crypto**: BTC/USD, ETH/USD, SOL/USD, XRP/USD, DOGE/USD, ADA/USD (6 pairs)
- **Forex**: 32 major currency pairs (EUR/USD, GBP/USD, USD/JPY, etc.)
  - Grouped into G1-G5 subcategories by liquidity/volume
- **Equities**: 7 major stocks (NVDA, AAPL, TSLA, AMZN, MSFT, GOOG, META) - currently blocked
- **Indices**: 6 global indices (SPX, DJI, NDX, VIX, FTSE, GDAXI) - currently blocked
- **Commodities**: XAU/USD, XAG/USD - currently blocked

### Performance Evaluation
- **Current Scoring**: Debt-based system tracking emissions, performance, and penalties
  - PnL weight: 100% (other metrics set to 0 in current config)
  - Weighted average with decay rate (0.075) for recent performance emphasis
  - 120-day target ledger window
- **Legacy Metrics** (configurable): Calmar, Sharpe, Omega, Sortino ratios + returns
- **Risk Management**:
  - 10% max drawdown elimination threshold (MAX_TOTAL_DRAWDOWN = 0.9)
  - 5% daily drawdown limit (MAX_DAILY_DRAWDOWN = 0.95)
  - Risk-adjusted performance penalties based on Sharpe, Sortino, Calmar, Omega ratios
- **Fees**:
  - Carry fees: 10.95% annually (crypto), 3% annually (forex)
  - Spread fees: 0.1% × leverage (crypto only)
  - Slippage costs: Higher for high leverage and low liquidity assets
- **Leverage Limits**:
  - Crypto: 0.01 to 0.5x
  - Forex: 0.1 to 5x
  - Equities: 0.1 to 3x
  - Indices: 0.1 to 5x
  - Portfolio cap: 10x across all positions

### Elimination Mechanisms
- **Plagiarism**: Cross-correlation analysis detecting order similarity
  - 75% similarity threshold, 10-day lookback window
  - Time-lag analysis for follower detection
  - 2-week review period before elimination
- **Max Drawdown**: Automatic elimination at 10% MDD
  - Continuous monitoring via `mdd_checker/` service
  - 60-second refresh interval
- **Challenge Period**: New miners enter 61-90 day challenge period
  - Must reach 75th percentile to enter main competition
  - Minimal weights during challenge period
- **Probation**: Miners below rank 25 in asset class
  - 60-day probation period
  - Must outscore 15th-ranked miner to avoid elimination

## Development Patterns

### File Naming Conventions
- Use snake_case for Python files
- RPC servers: `*_server.py` (e.g., `position_manager_server.py`)
- RPC clients: `*_client.py` (e.g., `elimination_client.py`)
- Test files: `test_*.py` prefix
- Configuration files: descriptive names (vali_config.py, miner_config.py)

### Code Organization
- **RPC Architecture**: Services communicate via RPC for modularity and fault isolation
  - `shared_objects/rpc/rpc_server_base.py` - Base RPC server class
  - `shared_objects/rpc/rpc_client_base.py` - Base RPC client class
  - `shared_objects/rpc/server_orchestrator.py` - Manages server lifecycle
  - Connection modes: LOCAL (testing) vs RPC (production)
- **Validators**: Orchestrate multiple RPC services for position tracking, scoring, elimination
- **Miners**: Signal generation and submission via Vanta API
- **Shared Objects**: Common utilities (locks, metagraph, cache, error handling)
- **Data Flow**: Real-time → Vanta API → RPC services → Performance ledgers

### RPC Service Pattern
```python
# Server implementation inherits from RPCServerBase
from shared_objects.rpc.rpc_server_base import RPCServerBase

class MyServer(RPCServerBase):
    def __init__(self, config, connection_mode=RPCConnectionMode.RPC):
        super().__init__(
            service_name="MyServer",
            port=config.RPC_MY_PORT,
            config=config,
            connection_mode=connection_mode
        )

    def my_rpc_method(self, arg1, arg2):
        # RPC-exposed method
        return result

# Client usage
from vali_objects.vali_config import RPCConnectionMode

# Production (RPC mode)
client = MyClient(connection_mode=RPCConnectionMode.RPC)

# Testing (LOCAL mode - bypass RPC)
client = MyClient(connection_mode=RPCConnectionMode.LOCAL)
client.set_direct_server(server_instance)
```

### External Dependencies
- **Bittensor 9.9.0** - Blockchain and subnet integration
- **Pydantic 2.10.3** - Data validation and serialization
- **Financial APIs**:
  - Polygon API Client 1.15.3 ($248/month)
  - Tiingo 0.15.6 ($50/month)
- **ML Stack**: scikit-learn 1.5.0, scipy 1.13.0, pandas 2.2.2
- **Web Services**:
  - Flask 3.0.3 + Waitress 2.1.2 for REST API
  - WebSockets for real-time communication
- **Data Visualization**: matplotlib 3.9.0
- **Cloud Services**: Google Cloud Storage 2.17.0, Secret Manager 2.21.1
- **Taoshi SDKs**:
  - collateral_sdk@1.0.6 - Collateral management
  - vanta-cli@2.0.0 - Vanta network CLI tools

## Production Deployment

### PM2 Process Management
The `run.sh` script provides production deployment with:
- Process name: `vanta` (migrated from legacy `ptn` name)
- Automatic version checking every 30 minutes (at :07 and :37)
- Git pull and automatic updates from GitHub (taoshidev/vanta-network)
- Exponential backoff retry logic (1s → 60s max) for version checks
- Process monitoring with auto-restart on failure
- Minimum uptime: 5 minutes, Max restarts: 5

### Version Management
- Current version: 8.8.8 (in `meta/meta.json`)
- Version checking against GitHub API
- Automatic pip install and package updates
- Safe rollback: git pull only if version is newer

### State Management
- **Backups**: Automatic timestamped validator state backups via `vali_bkp_utils.py`
  - Compressed checkpoint files (`validator_checkpoint.json.gz`)
  - Migration scripts in `runnable/` for state transformations
- **Persistence**:
  - Position data per miner in `validation/miners/`
  - Performance ledgers (RPC service)
  - Debt ledgers (RPC service)
  - Elimination tracking (RPC service)
  - Plagiarism scores in `validation/plagiarism/`
- **Recovery**: State regeneration via `restore_validator_from_backup.py`

### RPC Service Management
- Server orchestrator manages 18+ RPC services
- Health monitoring and automatic restarts
- Exponential backoff for failed connections
- Port conflict detection and resolution
- Graceful shutdown coordination

## Testing Strategy

### Test Organization
- **`tests/vali_tests/`** - Comprehensive validator test suite (60+ test files)
  - Position management and tracking (`test_positions*.py`)
  - Plagiarism detection (`test_plagiarism*.py`)
  - Elimination logic (`test_elimination*.py`)
  - Challenge period (`test_challengeperiod*.py`)
  - Debt and performance ledgers (`test_debt*.py`, `test_ledger*.py`)
  - Limit orders (`test_limit_order*.py`)
  - Auto-sync and checkpointing (`test_auto_sync*.py`)
  - Risk profiling and metrics (`test_risk*.py`, `test_metrics*.py`)
  - Asset selection and segmentation
- **`tests/validation/`** - Validation scenario tests
- **`tests/shared_objects/`** - Infrastructure tests

### Test Execution
```bash
# Run entire test suite
python tests/run_vali_testing_suite.py

# Run specific test file
python tests/run_vali_testing_suite.py test_positions.py

# Run with pytest directly
python -m pytest tests/vali_tests/test_elimination_manager.py -v
```

### Test Patterns
- Use `RPCConnectionMode.LOCAL` for fast unit tests (bypass RPC)
- Use `RPCConnectionMode.RPC` for integration tests (full RPC behavior)
- Mock utilities in `tests/vali_tests/mock_utils.py`
- Base objects in `tests/vali_tests/base_objects/`
- Fixtures defined in `conftest.py`

## Requirements
- **Python**: 3.10+ (required), supports 3.10, 3.11, 3.12
- **Hardware**:
  - CPU: 2-4 vCPU minimum
  - RAM: 8-16 GB recommended
  - Storage: Sufficient for checkpoints and position data
- **Network**:
  - Registration: 2.5 TAO on mainnet
  - Stable internet connection for API access
  - Open ports: 50000-50022 (RPC services), 48888 (REST API), 8765 (WebSocket)
- **Software**:
  - PM2 for process management
  - jq for JSON parsing (required by run.sh)
  - Git for version management