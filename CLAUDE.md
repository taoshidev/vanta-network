# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Proprietary Trading Network (PTN) is a Bittensor subnet (netuid 8 mainnet, 116 testnet) developed by Taoshi. It operates as a competitive trading signal network where miners submit trading strategies and validators evaluate their performance using sophisticated metrics.

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
# Validator (production with PM2)
./run.sh --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>

# Miner
python neurons/miner.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <miner>

# Validator (development)
python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <default>

# Signal reception server for miners
./run_receive_signals_server.sh
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
- **`neurons/`** - Main network participants (miner.py, validator.py)
- **`vali_objects/`** - Validator logic, configurations, performance tracking
- **`miner_objects/`** - Miner tools including React dashboard and order placement
- **`shared_objects/`** - Common utilities for time management, validation, utilities
- **`template/`** - Bittensor protocol definitions and base classes

### Data Infrastructure
- **`data_generator/`** - Financial data services (Polygon, Tiingo, Binance, Bybit, Kraken)
- **`vanta_api/`** - API management for real-time data and communication
- **`mining/`** - Signal processing pipeline (received/processed/failed signals)
- **`validation/`** - Validator state (eliminations, plagiarism, performance ledgers)

### Key Configuration Files
- **`vali_objects/vali_config.py`** - Main validator configuration including supported trade pairs
- **`requirements.txt`** - Python dependencies (Bittensor 9.7.0, financial APIs, ML libraries)
- **`meta/meta.json`** - Version management (subnet_version: 6.3.0)

## Trading System Architecture

### Signal Flow
1. Miners submit LONG/SHORT/FLAT signals for forex and crypto pairs
2. Validators receive signals via API endpoints
3. Real-time price validation using multiple data sources
4. Position tracking with leverage limits and slippage modeling
5. Performance calculation using 5 risk-adjusted metrics (20% each)

### Performance Evaluation
- **Metrics**: Calmar, Sharpe, Omega, Sortino ratios + total return
- **Risk Management**: 10% max drawdown elimination threshold
- **Fees**: Carry fees (10.95%/3% annually) and slippage costs
- **Scoring**: Weighted average with recent performance emphasis

### Elimination Mechanisms
- **Plagiarism**: Order similarity analysis for copy detection
- **Drawdown**: Automatic elimination at 10% max drawdown
- **Probation**: 30-day period for miners below 25th rank

## Development Patterns

### File Naming Conventions
- Use snake_case for Python files
- Prefix test files with `test_`
- Configuration files use descriptive names (vali_config.py, miner_config.py)

### Code Organization
- Validators handle all position tracking and performance calculation
- Miners focus on signal generation and submission
- Shared objects contain common utilities (time, validation, crypto)
- Real-time data flows through dedicated API layer

### External Dependencies
- **Bittensor 9.7.0** for blockchain integration
- **Financial APIs**: Polygon ($248/month), Tiingo ($50/month)
- **ML Stack**: scikit-learn, pandas, scipy for analysis
- **Web**: Flask for APIs, React/TypeScript/Vite for dashboard

## Production Deployment

### PM2 Process Management
The `run.sh` script provides production deployment with:
- Automatic version checking and updates from GitHub
- Process monitoring and restart capabilities
- Version comparison and rollback safety

### State Management
- **Backups**: Automatic timestamped validator state backups
- **Persistence**: Position data, performance ledgers, elimination tracking
- **Recovery**: Validator state regeneration capabilities

## Testing Strategy

Test files located in `tests/vali_tests/` cover:
- Position management and tracking
- Plagiarism detection algorithms
- Market hours and pricing validation
- Risk profiling and metrics calculation
- Challenge period integration
- Elimination manager functionality

## Requirements
- Python 3.10+ (required)
- Hardware: 2-4 vCPU, 8-16 GB RAM
- Network registration: 2.5 TAO on mainnet