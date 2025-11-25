# Vanta Network Validator

This guide details how to set up and run a Vanta Network validator on Bittensor's Subnet 8.

## Overview

Your validator receives trade signals from miners and maintains a portfolio per miner with all their positions on disk in the `validation/miners` directory.

Your validator:
- Tracks portfolio returns using live price information
- Eliminates miners whose portfolio value declines beyond the drawdown limits
- Detects & eliminates miners copying from the network by performing analysis on every order received
- Maintains information on plagiarizing miners in `validation/miner_copying.json`
- Records eliminated miners (due to drawdown limits or plagiarism) in `validation/eliminations.json`
- Maintains debt ledgers for each miner, combining emissions, penalties, and performance data
- Sets weights every 5 minutes using a debt-based scoring system that pays miners based on previous month's performance (PnL scaled by penalties), targeting payout completion by day 25 of each month

**Important Note**: Only registered non-eliminated miners can be given weights. Once eliminated, a miner can no longer send requests to validators until they are deregistered by the network and then re-register.

## Environment Options

| Environment | Netuid | Description |
| ----------- | -----: | ----------- |
| Mainnet     |      8 | Production environment with real TAO rewards |
| Testnet     |    116 | Testing environment recommended before mainnet |

> **IMPORTANT**: We strongly recommend running on testnet first before registering on mainnet.

## System Requirements

- **Hardware**: 4 vCPU + 16 GB memory with 1 TB balanced persistent disk
- **Software**: Python 3.10 (required)
- **Token**: 1000 SN8 Alpha (theta) Token staked
- **Data Provider Subscriptions**:
  - [Tiingo API](https://www.tiingo.com/) with "Commercial" ($50/month) subscription
  - [Polygon API](https://polygon.io/) with both "Currencies Starter" ($49/month) and "Stocks Advanced" (199/month) subscriptions

> **IMPORTANT**: After subscribing to Polygon, complete the KYC questionnaire to enable realtime US equities prices. Message a Taoshi team member ASAP if you need guidance with this step!

## Installation

### 1. Install Python 3.10

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y software-properties-common \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget \
    curl llvm libncurses5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

cd /usr/src
sudo wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
sudo tar xzf Python-3.10.12.tgz
cd Python-3.10.12

sudo ./configure --enable-optimizations
sudo make -j$(nproc)
sudo make altinstall

python3.10 --version
```

### 2. Clone Repository and Install Dependencies

```bash
git clone https://github.com/taoshidev/proprietary-trading-network.git
cd proprietary-trading-network
python3.10 -m venv venv
. venv/bin/activate
export PIP_NO_CACHE_DIR=1
pip install -r requirements.txt
python -m pip install -e .
```

> **Note**: Disregard any warnings about updating Bittensor. Use the version specified in `requirements.txt`.

## Wallet Setup

### 1. Create Wallet Keys

```bash
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

List local wallets:
```bash
btcli wallet list
```

### 2. Get Testnet TAO (For Testnet Only)

Join the [Bittensor Discord](https://discord.com/invite/bittensor) and request testnet TAO in the [help-forum channel](https://discord.com/channels/799672011265015819/1331693251589312553/1331694633822060544).

### 3. Register Your Validator

```bash
btcli subnet register --wallet.name validator --wallet.hotkey default
```

For testnet, add the flags: `--subtensor.network test --netuid 116`

Follow the prompts to complete registration:

```bash
>> Enter netuid (0): # Enter the appropriate netuid for your environment (8 for the mainnet)
Your balance is: # Your wallet balance will be shown
The cost to register by recycle is τ0.000000001 # Current registration costs
>> Do you want to continue? [y/n] (n): # Enter y to continue
>> Enter password to unlock key: # Enter your wallet password
>> Recycle τ0.000000001 to register on subnet:8? [y/n]: # Enter y to register
📡 Checking Balance...
Balance:
  τ5.000000000 ➡ τ4.999999999
✅ Registered
```

### 4. Verify Registration

```bash
btcli wallet overview --wallet.name validator
```

For testnet, add: `--subtensor.network test`

The above command will display:

```bash
Subnet: 8 # or 116 on testnet
COLDKEY    HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
validator  default  197    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                56  none  5GKkQKmDLfsKaumnkD479RBoD5CsbN2yRbMpY88J8YeC5DT4
1          1        1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                                Wallet balance: τ0.000999999
```

## Running Your Validator

### Prerequisites

1. Install PM2 and jq:
```bash
npm install -g pm2
brew install jq  # or apt install jq for Linux
```

2. Create a `secrets.json` file in the repository root:
```json
{
  "polygon_apikey": "YOUR_POLYGON_API_KEY",
  "tiingo_apikey": "YOUR_TIINGO_API_KEY"
}
```

### Launch Options

The `run.sh` script provides automated updating and simplified execution.

#### Optional Flags:
- `--start-generate`: Enables JSON file generation for trade data (can be sold via Request Network)
- `--autosync`: Synchronizes your data with a Taoshi-trusted validator (recommended)

#### For Mainnet:
```bash
pm2 start run.sh --name sn8 -- --wallet.name validator --wallet.hotkey default --netuid 8 [--start-generate] [--autosync]
```

#### For Testnet:
```bash
pm2 start run.sh --name sn8 -- --wallet.name validator --wallet.hotkey default --netuid 116 --subtensor.network test [--start-generate]
```

These commands initialize two PM2 processes:
- Validator process (named `vanta`)
- Auto-update process (named `sn8`) that checks for and applies updates every 30 minutes

### Manual Synchronization

If you prefer not to use `--autosync` but need to synchronize your validator:
1. Follow the [manual restore mechanism](https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/regenerating_validator_state.md)
2. This performs a "nuke and force rebuild" to synchronize with trusted validator data

### Alternative Testing Method

You can also test Vanta on the testnet with a direct run command:

```bash
python neurons/validator.py --netuid 116 --subtensor.network test --wallet.name validator --wallet.hotkey default
```

Note this won't launch the autoupdater. To launch with the autoupdater, use the run.sh command.

### Stopping Your Validator

Press CTRL+C in the terminal or use:
```bash
pm2 stop sn8 vanta
```

### Relaunching with Different Configuration
You will need to do this if you want to change any runtime configuration to run.sh such as adding or removing the `--start-generate`/ `--autosync` flags. Prepare your new `pm2 start run.sh ...` command before proceeding to minimize downtime.
```bash
cd proprietary-trading-network/
. venv/bin/activate
pm2 stop sn8 vanta
pm2 delete sn8 vanta
pm2 start run.sh --name sn8 -- [YOUR NEW OPTIONS]
pm2 save
```

Verify processes are running:
```bash
pm2 status
pm2 log
```

## Commit-Reveal and Emissions Timeline

Subnet 8 uses a commit-reveal mechanism in mainnet for weight setting:

- **What is Commit-Reveal?** Weights are set in two phases:
  1. **Commit Phase**: Publish a hashed version of your weights
  2. **Reveal Phase**: Reveal actual weights, verified against the committed hash

- **Timeline**:
  - Each epoch = 360 blocks (~12 seconds per block)
  - 20 epochs = ~24 hours wait time
  - Emissions begin only after weights are successfully revealed

- **Why no immediate emissions?**
  - You're protected by the 65,535-block immunity period
  - After ~24 hours from first weight commit, you should see:
    - Trust, rank, incentive values > 0
    - Emissions (τ) > 0

## Common Pitfalls and Prevention

1. **Rate Limiting**: Run a local subtensor to avoid rate limit issues on finney that prevent weights from being set
   - [Subtensor installation guide](https://github.com/opentensor/subtensor)

2. **Testnet Configuration**: Always include `--subtensor.network test` and `--netuid 116` for testnet

3. **JSON Format**: Ensure `secrets.json` is correctly formatted with proper syntax

4. **API Key Sharing**: Do not share API keys across multiple validators/scripts
   - Each key allows only one websocket connection

5. **Port Configuration**: In cloud environments (e.g., Runpod), you may not have your Bittensor default port open (8091)
   - This will cause your validator to be unable to communicate with miners and thus have a low VTRUST
   - Explicitly open a TCP port and pass it with: `--axon.port <YOUR_OPEN_PORT>`

6. **Generated Trade Data**: Reach out to the Taoshi team on how to sell the generated trade data via the Request Network

7. **Port Issues**: When running a validator in certain cloud environments such as Runpod, you may not have your Bittensor default port open (8091). This will cause your validator to be unable to communicate with miners and thus have a low VTRUST as your validator isn't receiving the latest orders. In order to correct this issue, explicitly open a tcp port, and pass this as an arugment with `--axon.port <YOUR_OPEN_PORT>`

## Security Warnings

- **DO NOT** expose your private keys
- **ONLY** use your testnet wallet for testing
- **DO NOT** reuse passwords between mainnet and testnet

## Additional Resources

- [Bittensor Documentation](https://github.com/opentensor/bittensor)
- [Manual Validator Synchronization Guide](https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/regenerating_validator_state.md)
